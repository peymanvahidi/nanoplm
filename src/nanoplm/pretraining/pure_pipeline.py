"""Pure PyTorch pretraining pipeline.

This is the same outer behavior as the HF Trainer pipeline (datasets, run naming,
checkpoint layout, resume mechanics), but the training loop is plain torch.
"""

import gc
import json
import math
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from nanoplm.data.manifest import read_manifest, validate_manifest_for_pipeline
from nanoplm.pretraining.collator import PackingCollator, ProtDataCollatorForLM
from nanoplm.pretraining.dataset import ShardedDataset
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.optim import MuonAdamW
from nanoplm.pretraining.pipeline import (
    PretrainingConfig,
    ResumeConfig,
    _get_num_workers,
    _prepare_run_and_steps,
)
from nanoplm.utils.common import create_dirs, get_device
from nanoplm.utils.logger import logger

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # FSDP2 modifies the model in-place; no wrapper to strip.
    return model


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _use_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    lname = name.lower()
    if not param.requires_grad:
        return False
    if param.ndim < 2:
        return False
    if "bias" in lname or "norm" in lname:
        return False
    return True


# H100 SXM peak BF16 tensor-core throughput (TFLOPS).
H100_PEAK_TFLOPS = 989.4


def _estimate_model_flops_per_token(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    seq_len: int,
    vocab_size: int,
) -> int:
    """Estimate FLOPs per token for a single forward pass.

    Uses the standard transformer FLOPs counting:
      Attention (per layer): 8*h^2 + 4*s*h  (QKV + output projections + attn logits/ctx)
      SwiGLU MLP (per layer): 6*h*I           (gate + up + down projections)
      Embedding/unembedding:  2*V*h

    Training FLOPs = 3 * forward FLOPs (forward + ~2x backward).
    """
    per_layer = 8 * hidden_size**2 + 4 * seq_len * hidden_size + 6 * hidden_size * intermediate_size
    forward_flops = num_layers * per_layer + 2 * vocab_size * hidden_size
    return 3 * forward_flops  # training = 3x forward


def _is_embedding_or_unembedding_param(name: str) -> bool:
    lname = name.lower()
    if "embeddings.tok_embeddings" in lname:
        return True
    if lname.endswith("decoder.weight") or lname.endswith("decoder.bias"):
        return True
    return (
        "embedding" in lname
        or "lm_head" in lname
        or "unembedding" in lname
    )


def _build_muon_optimizer(
    model: torch.nn.Module,
    cfg: PretrainingConfig,
    distributed_mesh=None,
) -> MuonAdamW:
    raw_model = _unwrap_model(model)

    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        # Muon is for hidden-layer matrices only.
        if param.ndim == 1:
            adamw_params.append(param)
            continue
        if _is_embedding_or_unembedding_param(name):
            adamw_params.append(param)
            continue
        if param.ndim == 2:
            muon_params.append(param)
            continue
        adamw_params.append(param)

    if not muon_params:
        raise ValueError(
            "No eligible matrix parameters found for Muon (expected 2D hidden-layer weights)."
        )

    logger.info(
        "Muon grouping: "
        f"muon_params={len(muon_params)} tensors, "
        f"adamw_params={len(adamw_params)} tensors"
    )

    return MuonAdamW(
        muon_params=muon_params,
        adamw_params=adamw_params,
        muon_learning_rate=cfg.muon_learning_rate,
        muon_weight_decay=cfg.muon_weight_decay,
        muon_cautious_weight_decay=cfg.muon_cautious_weight_decay,
        muon_use_polar_express=cfg.muon_use_polar_express,
        muon_momentum=cfg.muon_momentum,
        muon_nesterov=cfg.muon_nesterov,
        muon_eps=cfg.muon_eps,
        use_normuon=str(cfg.optimizer).lower() == "normuon",
        adamw_learning_rate=cfg.learning_rate,
        adamw_weight_decay=cfg.weight_decay,
        adamw_betas=(cfg.adam_beta1, cfg.adam_beta2),
        adamw_epsilon=cfg.adam_epsilon,
        distributed_mesh=distributed_mesh,
    )


def _create_optimizer(
    model: torch.nn.Module,
    cfg: PretrainingConfig,
    distributed_mesh=None,
) -> torch.optim.Optimizer:
    name = str(cfg.optimizer).lower()
    if name in {"muon", "normuon"}:
        return _build_muon_optimizer(model, cfg, distributed_mesh=distributed_mesh)

    raw_model = _unwrap_model(model)
    decay, no_decay = [], []

    for p_name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if _use_weight_decay(p_name, param):
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": float(cfg.weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {
        "params": groups,
        "lr": float(cfg.learning_rate),
        "betas": (float(cfg.adam_beta1), float(cfg.adam_beta2)),
        "eps": float(cfg.adam_epsilon),
    }

    if name == "adamw":
        return torch.optim.AdamW(**kwargs)
    if name == "stable_adamw":
        stable_adamw = getattr(torch.optim, "StableAdamW", None)
        if stable_adamw is None:
            logger.warning("StableAdamW unavailable; falling back to AdamW.")
            return torch.optim.AdamW(**kwargs)
        return stable_adamw(**kwargs)

    raise ValueError(f"Invalid optimizer: {cfg.optimizer}. Supported: [adamw, stable_adamw, muon, normuon]")


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)


def _num_update_steps_per_epoch(train_loader_len: int, grad_accum: int) -> int:
    if train_loader_len <= 0:
        return 1
    return max(1, math.ceil(train_loader_len / max(1, grad_accum)))


def _get_warmup_steps(total_steps: int, warmup_value: float) -> int:
    if warmup_value >= 1:
        return int(warmup_value)
    return math.ceil(total_steps * warmup_value)


def _resolve_world_size(cfg: PretrainingConfig) -> int:
    if not cfg.multi_gpu:
        return 1
    if cfg.world_size == "auto":
        env_world_size = os.environ.get("WORLD_SIZE")
        return int(env_world_size) if env_world_size else max(torch.cuda.device_count(), 1)
    return int(cfg.world_size) if cfg.world_size else 1


def _dist_barrier(local_rank: int) -> None:
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    distributed: bool,
    amp_dtype: Optional[torch.dtype],
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in eval_loader:
        batch = _move_batch_to_device(batch, device)
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with amp_ctx:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        loss = out["loss"] if isinstance(out, dict) else out.loss
        batch_size = batch["input_ids"].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if distributed and dist.is_initialized():
        stats = torch.tensor([total_loss, total_samples], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        total_samples = int(stats[1].item())

    model.train()
    return total_loss / max(1, total_samples)


def _to_full_tensors(obj):
    """Recursively convert DTensors to plain tensors (collective operation)."""
    if isinstance(obj, DTensor):
        return obj.full_tensor()
    if isinstance(obj, dict):
        return {k: _to_full_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_full_tensors(v) for v in obj]
    return obj


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    global_step: int,
    epoch: int,
    output_dir: str,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    distributed: bool = False,
    is_main: bool = True,
) -> None:
    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"

    if distributed:
        # Collective: gather full model & optimizer state dicts.
        model_sd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        opt_sd = _to_full_tensors(optimizer.state_dict())
    else:
        model_sd = _unwrap_model(model).state_dict()
        opt_sd = optimizer.state_dict()

    if not is_main:
        return

    create_dirs(str(checkpoint_dir))
    torch.save(model_sd, checkpoint_dir / "pytorch_model.bin")
    torch.save(opt_sd, checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    torch.save(
        {
            "torch_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        checkpoint_dir / "rng_state.pth",
    )

    (checkpoint_dir / "training_state.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "epoch": epoch,
                "logging_steps": logging_steps,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Checkpoint saved -> {checkpoint_dir}")


def _load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    checkpoint_dir: str,
    device: torch.device,
    distributed: bool = False,
) -> Tuple[int, int]:
    ckp = Path(checkpoint_dir)

    model_sd = torch.load(ckp / "pytorch_model.bin", map_location=device, weights_only=True)
    if distributed:
        set_model_state_dict(
            model, model_sd, options=StateDictOptions(full_state_dict=True)
        )
    else:
        _unwrap_model(model).load_state_dict(model_sd)

    optimizer.load_state_dict(
        torch.load(ckp / "optimizer.pt", map_location=device, weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(ckp / "scheduler.pt", map_location=device, weights_only=True)
    )

    rng_path = ckp / "rng_state.pth"
    if rng_path.exists():
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        torch.random.set_rng_state(rng["torch_rng"])
        if torch.cuda.is_available() and rng["cuda_rng"]:
            torch.cuda.set_rng_state_all(rng["cuda_rng"])
        np.random.set_state(rng["numpy_rng"])
        random.setstate(rng["python_rng"])

    state_path = ckp / "training_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state.get("global_step", 0)), int(state.get("epoch", 0))
    return 0, 0


def run_pure_pretraining(
    model: PureProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:
    _set_seed(pretrain_config.seed)
    tokenizer = model.tokenizer
    device = torch.device(get_device())

    dataset_dir = Path(pretrain_config.dataset_dir)

    manifest = read_manifest(dataset_dir)
    validate_manifest_for_pipeline(manifest=manifest, expected_mode="pretrain")

    train_shard_dir = dataset_dir / manifest.train_dir
    val_shard_dir = dataset_dir / manifest.val_dir
    train_sequences = manifest.train_sequences

    logger.info(f"Loaded config from manifest: {dataset_dir}")
    logger.info(f"  train_shards: {train_shard_dir}")
    logger.info(f"  val_shards: {val_shard_dir}")
    logger.info(f"  max_length: {manifest.max_seq_len}")
    logger.info(f"  train_sequences: {manifest.train_sequences}")
    logger.info(f"  val_sequences: {manifest.val_sequences}")

    if not train_shard_dir.exists():
        raise FileNotFoundError(f"Train shard directory not found: {train_shard_dir}")
    if not val_shard_dir.exists():
        raise FileNotFoundError(f"Validation shard directory not found: {val_shard_dir}")

    logger.info("Using ShardedDataset for pre-tokenized binary shards")
    try:
        train_ds = ShardedDataset(data_dir=str(train_shard_dir))
        val_ds = ShardedDataset(data_dir=str(val_shard_dir))
    except FileNotFoundError as e:
        logger.error(
            f"Binary shards not found! You need to create them first.\n"
            f"Run: nanoplm data from-yaml with pipeline_mode: 'pretrain'\n"
            f"Error: {e}"
        )
        raise

    use_packing = bool(getattr(pretrain_config, "use_packing", False))
    target_rows = getattr(pretrain_config, "target_packed_rows", None)
    if target_rows is not None:
        target_rows = int(target_rows)
    use_static_packing = use_packing and target_rows is not None

    if use_packing:
        collator = PackingCollator(
            tokenizer=tokenizer,
            max_seq_len=manifest.max_seq_len,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
            target_rows=target_rows,
            batch_size=pretrain_config.micro_batch_size if target_rows else None,
        )
        if use_static_packing:
            logger.info(
                f"Sequence packing ENABLED (static shapes, "
                f"target_rows={target_rows}, "
                f"flat_size={target_rows * manifest.max_seq_len:,})"
            )
        else:
            logger.info("Sequence packing ENABLED (PackingCollator, dynamic shapes)")
    else:
        collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )

    create_dirs(pretrain_config.ckp_dir)

    effective_world_size = _resolve_world_size(pretrain_config)

    inferred_grad_accum_steps = pretrain_config.inferred_grad_accum_steps
    global_batch_size_samples = pretrain_config.global_batch_size_samples
    achieved_global_batch_tokens = pretrain_config.achieved_global_batch_tokens

    if (
        inferred_grad_accum_steps is None
        or global_batch_size_samples is None
        or achieved_global_batch_tokens is None
    ):
        raise ValueError(
            "Batch setup is missing on PretrainingConfig. "
            "Run pretraining through nanoplm CLI so inferred batch fields are populated."
        )

    world_tokens_per_micro_step = achieved_global_batch_tokens // max(
        1, inferred_grad_accum_steps
    )

    logger.info(
        "Batch setup: "
        f"target_global_batch_size={pretrain_config.global_batch_size:,} tokens, "
        f"micro_step_tokens={world_tokens_per_micro_step:,}, "
        f"grad_accum_steps={inferred_grad_accum_steps}, "
        f"effective_global_batch_size={achieved_global_batch_tokens:,} tokens"
    )

    (
        _run_name,
        wandb_run_name,
        output_dir,
        num_epochs,
        logging_steps,
        eval_steps,
        save_steps,
        _resume_step,
    ) = _prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=train_sequences,
        global_batch_size_samples=global_batch_size_samples,
    )

    num_workers = _get_num_workers(pretrain_config.num_workers, effective_world_size)
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    prefetch_factor = pretrain_config.prefetch_factor if num_workers > 0 else None

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = bool(pretrain_config.multi_gpu)

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            if backend == "nccl":
                dist.init_process_group(backend=backend, device_id=local_rank)
            else:
                dist.init_process_group(backend=backend)

    model.to(device)

    # -- Precision detection (needed before FSDP for MixedPrecisionPolicy) --
    use_bf16 = (
        pretrain_config.bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = (
        pretrain_config.bf16
        and ((device.type == "cuda" and not torch.cuda.is_bf16_supported()) or device.type == "mps")
    )

    fsdp_mesh = None
    if distributed:
        # Apply FSDP2 per transformer layer, then at the root.
        fsdp_kwargs: dict = {}
        if use_bf16:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        fsdp_mesh = init_device_mesh("cuda", (effective_world_size,))
        for layer in model.model.layers:
            fully_shard(layer, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)
        fully_shard(model, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)

        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=pretrain_config.seed)
        eval_sampler = DistributedSampler(val_ds, shuffle=False)
        is_main = dist.get_rank() == 0
    else:
        train_sampler = RandomSampler(train_ds)
        eval_sampler = SequentialSampler(val_ds)
        is_main = True

    # Compile model for faster training. Keep orig_model for checkpointing/eval
    # (eval inputs may change shape, which would cause recompilation).
    orig_model = model
    if use_static_packing:
        # Static shapes: collator pre-flattens and pads to fixed sizes,
        # so no data-dependent ops inside the compiled graph.
        model = torch.compile(model, dynamic=False)
        logger.info("Model compiled with torch.compile(dynamic=False)")
    else:
        # dynamic=True because varlen flash-attention produces variable-length
        # unpadded tensors per batch.
        model = torch.compile(model, dynamic=True)
        logger.info("Model compiled with torch.compile(dynamic=True)")

    # Eval always uses a standard padding collator (no packing) so that
    # _evaluate() sees 2-D batches with attention_mask regardless of the
    # training collator configuration.
    eval_collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
    )

    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=pretrain_config.micro_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=use_static_packing,  # static shapes require fixed batch size
    )
    eval_loader = DataLoader(
        val_ds,
        sampler=eval_sampler,
        batch_size=pretrain_config.micro_batch_size,
        collate_fn=eval_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,  # eval always processes all samples
    )

    optimizer = _create_optimizer(model, pretrain_config, distributed_mesh=fsdp_mesh)

    steps_per_epoch = _num_update_steps_per_epoch(
        train_loader_len=len(train_loader),
        grad_accum=inferred_grad_accum_steps,
    )
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = _get_warmup_steps(total_steps, float(pretrain_config.warmup_ratio))
    scheduler = _create_scheduler(optimizer, warmup_steps, total_steps)

    start_step = 0
    start_epoch = 0
    if resume_config and resume_config.is_resume:
        logger.info(f"Resuming from checkpoint: {resume_config.checkpoint_dir}")
        start_step, start_epoch = _load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=resume_config.checkpoint_dir,
            device=device,
            distributed=distributed,
        )
        logger.info(f"Resumed at global_step={start_step}, epoch={start_epoch}")

    resume_micro_step = 0
    resume_epoch = start_epoch
    if resume_config and resume_config.is_resume:
        steps_done = max(0, start_step - start_epoch * steps_per_epoch)
        resume_micro_step = steps_done * inferred_grad_accum_steps
        if resume_micro_step >= len(train_loader):
            resume_micro_step = 0
            start_epoch = min(start_epoch + 1, num_epochs)
            resume_epoch = start_epoch
        if resume_micro_step > 0:
            logger.info(
                f"Skipping {resume_micro_step} micro-steps in resumed epoch {resume_epoch}"
            )

    wandb_enabled = False
    if is_main:
        try:
            wandb.init(
                project=pretrain_config.project_name,
                name=wandb_run_name,
                config={
                    "pretrain": pretrain_config.__dict__,
                    "total_steps": total_steps,
                    "warmup_steps": warmup_steps,
                },
            )
            wandb_enabled = wandb.run is not None
            if wandb_enabled:
                # Match HF Trainer's W&B convention for the x-axis.
                wandb.define_metric("train/global_step")
                wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
        except Exception as exc:
            logger.warning(f"W&B init failed, continuing without logging. Error: {exc}")

    # When FSDP handles bf16 via MixedPrecisionPolicy, autocast is not needed.
    # Keep autocast only for non-distributed bf16 or fp16 paths.
    amp_dtype: Optional[torch.dtype] = None
    if use_bf16 and not distributed:
        amp_dtype = torch.bfloat16
    elif use_fp16:
        amp_dtype = torch.float16

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(pretrain_config.tf32)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(pretrain_config.tf32)

    scaler = (
        torch.amp.GradScaler(enabled=(device.type == "cuda" and use_fp16))
        if torch.cuda.is_available()
        else None
    )

    # Pre-compute FLOPs-per-token for MFU estimation.
    _raw_model = _unwrap_model(model)
    _mb_cfg = _raw_model.config if hasattr(_raw_model, "config") else _raw_model._orig_mod.config
    _flops_per_token = _estimate_model_flops_per_token(
        num_layers=_mb_cfg.num_hidden_layers,
        hidden_size=_mb_cfg.hidden_size,
        intermediate_size=_mb_cfg.intermediate_size,
        seq_len=manifest.max_seq_len,
        vocab_size=_mb_cfg.vocab_size,
    )
    _peak_flops_per_gpu = H100_PEAK_TFLOPS * 1e12  # convert to FLOPS
    logger.info(
        f"MFU estimation: {_flops_per_token:,} training FLOPs/token, "
        f"H100 peak = {H100_PEAK_TFLOPS} TFLOPS"
    )

    logger.info(
        "Starting pure-torch training: "
        f"epochs={num_epochs}, total_steps={total_steps}, warmup_steps={warmup_steps}, "
        f"grad_accum={inferred_grad_accum_steps}"
    )
    logger.info(
        f"Precision config: bf16={use_bf16}, fp16={use_fp16}, "
        f"tf32={(pretrain_config.tf32 and device.type == 'cuda')}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    accum_loss = 0.0
    window_loss = 0.0
    window_steps = 0

    token_count = 0
    raw_token_count = 0
    token_t0: Optional[float] = None
    first_step_of_run = True

    for epoch in range(start_epoch, num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        for micro_step, batch in enumerate(train_loader):
            if resume_micro_step > 0 and epoch == resume_epoch and micro_step < resume_micro_step:
                continue

            batch = _move_batch_to_device(batch, device)

            # FSDP2: only reduce-scatter gradients at accumulation boundaries.
            if distributed:
                _is_sync_step = (
                    (micro_step + 1) % inferred_grad_accum_steps == 0
                    or micro_step + 1 == len(train_loader)
                )
                model.set_requires_gradient_sync(_is_sync_step)

            if raw_token_count == 0:
                token_t0 = time.perf_counter()

            # Token counting: for the static path there is no attention_mask,
            # so count real tokens from cu_seqlens[-1] and total from tensor size.
            if "attention_mask" in batch:
                token_count += int(batch["attention_mask"].sum().item())
                raw_token_count += int(batch["attention_mask"].numel())
            else:
                # Static packed path: cu_seqlens[-1] = total real tokens.
                token_count += int(batch["cu_seqlens"][-1].item())
                raw_token_count += int(batch["input_ids"].numel())

            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with amp_ctx:
                fwd_kwargs = dict(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                if "attention_mask" in batch:
                    fwd_kwargs["attention_mask"] = batch["attention_mask"]
                if "cu_seqlens" in batch:
                    fwd_kwargs["cu_seqlens"] = batch["cu_seqlens"]
                    fwd_kwargs["max_seqlen"] = batch["max_seqlen"]
                if "position_ids" in batch:
                    fwd_kwargs["position_ids"] = batch["position_ids"]
                out = model(**fwd_kwargs)

            loss = out["loss"] if isinstance(out, dict) else out.loss
            loss = loss / inferred_grad_accum_steps

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

            grad_accum = inferred_grad_accum_steps
            is_boundary = (micro_step + 1) % grad_accum == 0
            is_last_micro = micro_step + 1 == len(train_loader)

            if not (is_boundary or is_last_micro):
                continue

            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            optimizer_step_skipped = False
            if scaler is not None and scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_skipped = scaler.get_scale() < old_scale
            else:
                optimizer.step()

            if isinstance(optimizer, MuonAdamW):
                learning_rate = optimizer.adamw.param_groups[0]["lr"]
                muon_lr = optimizer.muon.param_groups[0]["lr"]
            else:
                learning_rate = optimizer.param_groups[0]["lr"]
                muon_lr = None
            if not optimizer_step_skipped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            window_loss += accum_loss
            window_steps += 1
            accum_loss = 0.0

            t1 = time.perf_counter()
            if token_t0 is None:
                token_t0 = t1
            elapsed = t1 - token_t0
            tok = float(token_count)
            raw_tok = float(raw_token_count)

            if distributed and dist.is_initialized():
                tok_tensor = torch.tensor(tok, device=device)
                raw_tok_tensor = torch.tensor(raw_tok, device=device)
                elapsed_tensor = torch.tensor(float(elapsed), device=device)
                dist.all_reduce(tok_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(raw_tok_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                tok = tok_tensor.item()
                raw_tok = raw_tok_tensor.item()
                elapsed = elapsed_tensor.item()

            tokens_per_sec = tok / max(elapsed, 1e-9)
            raw_tokens_per_sec = raw_tok / max(elapsed, 1e-9)

            # MFU: (achieved FLOPs/s per GPU) / (H100 peak FLOPs/s)
            achieved_flops_per_sec = raw_tokens_per_sec * _flops_per_token
            per_gpu_flops = achieved_flops_per_sec / max(effective_world_size, 1)
            mfu = per_gpu_flops / _peak_flops_per_gpu

            token_count = 0
            raw_token_count = 0
            token_t0 = None

            should_log = global_step % logging_steps == 0
            if should_log and is_main:
                loss_to_log = window_loss / max(1, window_steps)
                payload = {
                    "train/global_step": global_step,
                    "train/loss": loss_to_log,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": learning_rate,
                    "train/epoch": epoch + (micro_step + 1) / len(train_loader),
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/raw_tokens_per_sec": raw_tokens_per_sec,
                }
                if muon_lr is not None:
                    payload["train/muon_lr"] = muon_lr
                if wandb_enabled and wandb.run is not None:
                    try:
                        wandb.log(payload)
                    except Exception as exc:
                        wandb_enabled = False
                        logger.warning(f"W&B log failed; disabling logging. Error: {exc}")
                muon_lr_str = f"muon_lr={muon_lr:.2e} " if muon_lr is not None else ""
                logger.info(
                    f"[step {global_step}/{total_steps}] "
                    f"loss={loss_to_log:.4f} lr={learning_rate:.2e} {muon_lr_str}"
                    f"grad_norm={grad_norm:.4f} tok/s={tokens_per_sec:,.0f} "
                    f"raw_tok/s={raw_tokens_per_sec:,.0f} "
                    f"h100_mfu={mfu:.2%}"
                )

            # Tame the garbage collector: after the first optimizer step
            # (which triggers torch.compile warmup), collect garbage from
            # setup, freeze surviving objects, and disable automatic GC.
            # This avoids ~500ms GC pauses during training.
            if first_step_of_run:
                first_step_of_run = False
                gc.collect()
                gc.freeze()
                gc.disable()
            elif global_step % 5000 == 0:
                gc.collect()  # periodic manual collect for very long runs

            if should_log:
                window_loss = 0.0
                window_steps = 0

            if global_step % eval_steps == 0:
                eval_loss = _evaluate(
                    model=orig_model,
                    eval_loader=eval_loader,
                    device=device,
                    distributed=distributed,
                    amp_dtype=amp_dtype,
                )
                if is_main:
                    if wandb_enabled and wandb.run is not None:
                        try:
                            wandb.log(
                                {
                                    "train/global_step": global_step,
                                    "eval/loss": eval_loss,
                                }
                            )
                        except Exception as exc:
                            wandb_enabled = False
                            logger.warning(f"W&B log failed; disabling logging. Error: {exc}")
                    logger.info(f"[step {global_step}] eval_loss={eval_loss:.4f}")
                model.train()

            if global_step % save_steps == 0:
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    output_dir=output_dir,
                    logging_steps=logging_steps,
                    eval_steps=eval_steps,
                    save_steps=save_steps,
                    distributed=distributed,
                    is_main=is_main,
                )

        if resume_micro_step > 0 and epoch == resume_epoch:
            resume_micro_step = 0

    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    _save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch=num_epochs,
        output_dir=output_dir,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        distributed=distributed,
        is_main=is_main,
    )
    if is_main:
        if wandb_enabled and wandb.run is not None:
            try:
                (Path(output_dir) / "wandb_run_id.txt").write_text(wandb.run.id, encoding="utf-8")
                wandb.finish()
            except Exception as exc:
                logger.warning(f"W&B finalize failed. Error: {exc}")

    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    logger.info("Pure-torch pretraining complete.")
