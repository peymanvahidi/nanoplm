"""Pure PyTorch pretraining pipeline.

Same outer behavior as the HF Trainer pipeline (datasets, run naming,
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
from nanoplm.pretraining.collator import (
    DataCollatorWithFlattening,
    ProtDataCollatorForLM,
    build_power_of_two_buckets,
)
from nanoplm.pretraining.dataset import ShardedDataset, TokenPackingDataset
from nanoplm.pretraining.fp8 import (
    Float8Linear,
    Float8LinearConfig,
    convert_to_float8_training,
)
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from dion import Muon as DionMuon, NorMuon as DionNorMuon
from nanoplm.pretraining.optim import build_optimizer
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

# H100 SXM peak BF16 tensor-core throughput (TFLOPS).
H100_PEAK_TFLOPS = 989.4

# https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#forward-backward-with-prefetching
N_PREFETCH_LAYERS_FSDP2 = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def _format_vram_for_log(
    *,
    device: torch.device,
    distributed: bool,
    reset_peak: bool = False,
) -> str:
    """Return CUDA VRAM usage string; aggregate max across ranks when distributed."""
    if device.type != "cuda":
        return "vram=n/a"

    alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

    if distributed and dist.is_initialized():
        stats = torch.tensor(
            [alloc_mb, reserved_mb, peak_alloc_mb, peak_reserved_mb],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        alloc_mb, reserved_mb, peak_alloc_mb, peak_reserved_mb = (
            float(stats[0].item()),
            float(stats[1].item()),
            float(stats[2].item()),
            float(stats[3].item()),
        )

    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)

    return (
        f"vram={alloc_mb:,.0f}/{reserved_mb:,.0f}MB "
        f"peak={peak_alloc_mb:,.0f}/{peak_reserved_mb:,.0f}MB"
    )


def _use_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    if not param.requires_grad or param.ndim < 2:
        return False
    lname = name.lower()
    return "bias" not in lname and "norm" not in lname


def _is_embedding_or_unembedding_param(name: str) -> bool:
    lname = name.lower()
    return (
        "embeddings.tok_embeddings" in lname
        or lname.endswith("decoder.weight")
        or lname.endswith("decoder.bias")
        or "embedding" in lname
        or "lm_head" in lname
        or "unembedding" in lname
    )


def _dist_barrier(local_rank: int) -> None:
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def _estimate_model_flops_per_token(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    seq_len: int,
    vocab_size: int,
) -> int:
    """Training FLOPs per token (forward + ~2x backward)."""
    per_layer = (
        8 * hidden_size ** 2
        + 4 * seq_len * hidden_size
        + 6 * hidden_size * intermediate_size
    )
    forward_flops = num_layers * per_layer + 2 * vocab_size * hidden_size
    return 3 * forward_flops


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------

def _compile_inner_layers_for_fsdp(model: torch.nn.Module, *, dynamic: bool) -> None:
    """Compile transformer layers individually (before FSDP wrapping)."""
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = torch.compile(layer, dynamic=dynamic)


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------

def _build_muon_optimizer(model, cfg, distributed_mesh=None):
    muon_params, adamw_params = [], []
    resid_params, x0_params = [], []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        lname = name.lower()
        if lname.endswith("resid_lambdas") or ".resid_lambdas" in lname:
            resid_params.append(param)
            continue
        if lname.endswith("x0_lambdas") or ".x0_lambdas" in lname:
            x0_params.append(param)
            continue
        if param.ndim == 1 or _is_embedding_or_unembedding_param(name):
            adamw_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    if not muon_params:
        raise ValueError("No eligible 2-D matrix parameters found for Muon.")

    logger.info(
        f"Muon grouping: muon_params={len(muon_params)} tensors, "
        f"adamw_params={len(adamw_params)} tensors, "
        f"resid_scalar_params={len(resid_params)} tensors, "
        f"x0_scalar_params={len(x0_params)} tensors"
    )
    return build_optimizer(
        muon_params=muon_params,
        adamw_params=adamw_params,
        resid_params=resid_params,
        x0_params=x0_params,
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


def _create_optimizer(model, cfg, distributed_mesh=None):
    name = str(cfg.optimizer).lower()
    if name in {"muon", "normuon"}:
        return _build_muon_optimizer(model, cfg, distributed_mesh=distributed_mesh)

    decay, no_decay = [], []
    for p_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if _use_weight_decay(p_name, param) else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": float(cfg.weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(
        params=groups,
        lr=float(cfg.learning_rate),
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=float(cfg.adam_epsilon),
    )

    if name == "adamw":
        return torch.optim.AdamW(**kwargs)
    if name == "stable_adamw":
        cls = getattr(torch.optim, "StableAdamW", None)
        if cls is None:
            logger.warning("StableAdamW unavailable; falling back to AdamW.")
            return torch.optim.AdamW(**kwargs)
        return cls(**kwargs)
    raise ValueError(f"Invalid optimizer: {cfg.optimizer}. Supported: [adamw, stable_adamw, muon, normuon]")


def _create_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    learning_rate: float,
    lr_decay_to_fraction: float,
    lr_schedule: str = "Linear",
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        if lr_schedule.lower() == "cosin":
            return lr_decay_to_fraction + 0.5 * (1.0 - lr_decay_to_fraction) * (1.0 + math.cos(math.pi * progress))
        else:
            return max(lr_decay_to_fraction, 1.0 - (1.0 - lr_decay_to_fraction) * progress)

    return LambdaLR(optimizer, lr_lambda)


def _resolve_world_size(cfg: PretrainingConfig) -> int:
    if not cfg.multi_gpu:
        return 1
    if cfg.world_size == "auto":
        env = os.environ.get("WORLD_SIZE")
        return int(env) if env else max(torch.cuda.device_count(), 1)
    return int(cfg.world_size) if cfg.world_size else 1


def _num_update_steps_per_epoch(train_loader_len: int, grad_accum: int) -> int:
    if train_loader_len <= 0:
        return 1
    return max(1, math.ceil(train_loader_len / max(1, grad_accum)))


def _sync_train_loader_len(
    train_loader_len: int, distributed: bool, device: torch.device
) -> int:
    """Ensure all ranks see the same number of micro-batches per epoch."""
    if not (distributed and dist.is_initialized()):
        return train_loader_len
    t = torch.tensor(train_loader_len, device=device, dtype=torch.int64)
    mn, mx = t.clone(), t.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    if int(mn) != int(mx):
        raise RuntimeError(
            f"Mismatched train loader lengths across ranks (min={int(mn)}, max={int(mx)})."
        )
    return int(mx)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _evaluate(model, eval_loader, device, distributed, amp_dtype) -> float:
    model.eval()
    total_loss, total_samples = 0.0, 0

    for batch in eval_loader:
        batch = _move_batch_to_device(batch, device)
        ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with ctx:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        loss = out["loss"] if isinstance(out, dict) else out.loss
        bs = batch["input_ids"].size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    if distributed and dist.is_initialized():
        stats = torch.tensor([total_loss, total_samples], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_samples = stats[0].item(), int(stats[1].item())

    model.train()
    return total_loss / max(1, total_samples)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

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
    model, optimizer, scheduler, global_step, epoch,
    output_dir, logging_steps, eval_steps, save_steps,
    distributed=False, is_main=True,
) -> None:
    ckpt = Path(output_dir) / f"checkpoint-{global_step}"

    if distributed:
        model_sd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        opt_sd = _to_full_tensors(optimizer.state_dict())
    else:
        model_sd = model.state_dict()
        opt_sd = optimizer.state_dict()

    if not is_main:
        return

    create_dirs(str(ckpt))
    torch.save(model_sd, ckpt / "pytorch_model.bin")
    torch.save(opt_sd, ckpt / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
    torch.save(
        {
            "torch_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        ckpt / "rng_state.pth",
    )
    (ckpt / "training_state.json").write_text(
        json.dumps(dict(
            global_step=global_step, epoch=epoch,
            logging_steps=logging_steps, eval_steps=eval_steps, save_steps=save_steps,
        ), indent=2),
        encoding="utf-8",
    )
    logger.info(f"Checkpoint saved -> {ckpt}")


def _load_checkpoint(model, optimizer, scheduler, checkpoint_dir, device, distributed=False) -> Tuple[int, int]:
    ckp = Path(checkpoint_dir)

    model_sd = torch.load(ckp / "pytorch_model.bin", map_location=device, weights_only=True)
    if distributed:
        set_model_state_dict(model, model_sd, options=StateDictOptions(full_state_dict=True))
    else:
        model.load_state_dict(model_sd)

    opt_sd = torch.load(ckp / "optimizer.pt", map_location=device, weights_only=True)

    # When resuming under FSDP2, the checkpoint contains full (unsharded) optimizer
    # state tensors but the live parameters are sharded across ranks. We must slice
    # each optimizer buffer to match the corresponding parameter's local shard shape.
    if distributed and dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()

        # Build map: param index -> local parameter tensor
        param_by_idx: dict[int, torch.nn.Parameter] = {}
        all_params = list(optimizer.param_groups[0]["params"]) + list(optimizer.param_groups[1]["params"])
        for idx, p in enumerate(all_params):
            param_by_idx[idx] = p

        for param_idx, state_dict_entry in opt_sd.get("state", {}).items():
            local_param = param_by_idx.get(param_idx)
            if local_param is None:
                continue
            # Get local shape (after FSDP sharding)
            local_shape = local_param.shape if not isinstance(local_param, DTensor) else local_param._local_tensor.shape
            for key, val in state_dict_entry.items():
                if not isinstance(val, torch.Tensor):
                    continue
                # If shapes already match, skip
                if val.shape == local_shape:
                    continue
                # Shard along dim 0 (FSDP default sharding dimension)
                if val.ndim >= 1 and val.shape[0] == local_shape[0] * world:
                    chunk_size = val.shape[0] // world
                    state_dict_entry[key] = val.narrow(0, rank * chunk_size, chunk_size).contiguous()
                elif val.ndim >= 1 and val.shape[0] != local_shape[0]:
                    # Non-standard sharding; try simple chunk
                    chunks = val.chunk(world, dim=0)
                    if chunks[rank].shape[0] == local_shape[0]:
                        state_dict_entry[key] = chunks[rank].contiguous()
                    else:
                        logger.warning(
                            f"Optimizer state param {param_idx} key '{key}': "
                            f"shape {val.shape} cannot be sharded to match local {local_shape}"
                        )

    optimizer.load_state_dict(opt_sd)
    scheduler.load_state_dict(torch.load(ckp / "scheduler.pt", map_location=device, weights_only=True))

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pure_pretraining(
    model: PureProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:
    _set_seed(pretrain_config.seed)
    tokenizer = model.tokenizer
    device = torch.device(get_device())

    # ---- Dataset ----
    dataset_dir = Path(pretrain_config.dataset_dir)
    manifest = read_manifest(dataset_dir)
    validate_manifest_for_pipeline(manifest=manifest, expected_mode="pretrain")
    if manifest.max_seq_len <= 0:
        raise ValueError(f"Invalid manifest max_seq_len: {manifest.max_seq_len}")

    model_max_pos = int(getattr(model.config, "max_position_embeddings", 0))
    if model_max_pos > 0 and manifest.max_seq_len > model_max_pos:
        raise ValueError(
            f"Dataset max_seq_len ({manifest.max_seq_len}) exceeds "
            f"model max_position_embeddings ({model_max_pos})."
        )

    train_shard_dir = dataset_dir / manifest.train_dir
    val_shard_dir = dataset_dir / manifest.val_dir
    for d in (train_shard_dir, val_shard_dir):
        if not d.exists():
            raise FileNotFoundError(f"Shard directory not found: {d}")

    logger.info(f"Loaded config from manifest: {dataset_dir}")
    logger.info(f"  train_shards: {train_shard_dir}")
    logger.info(f"  val_shards: {val_shard_dir}")
    logger.info(f"  max_length: {manifest.max_seq_len}")
    logger.info(f"  train_sequences: {manifest.train_sequences}")
    logger.info(f"  val_sequences: {manifest.val_sequences}")

    logger.info("Using ShardedDataset for pre-tokenized binary shards")
    train_ds = ShardedDataset(data_dir=str(train_shard_dir))
    val_ds = ShardedDataset(data_dir=str(val_shard_dir))

    use_packing = bool(pretrain_config.use_packing)
    use_static_inp_size = bool(pretrain_config.use_static_inp_size and use_packing)

    # ---- Batch sizing ----
    create_dirs(pretrain_config.ckp_dir)
    effective_world_size = _resolve_world_size(pretrain_config)

    inferred_grad_accum_steps = pretrain_config.inferred_grad_accum_steps
    global_batch_size_samples = pretrain_config.global_batch_size_samples
    achieved_global_batch_tokens = pretrain_config.achieved_global_batch_tokens
    if any(v is None for v in (inferred_grad_accum_steps, global_batch_size_samples, achieved_global_batch_tokens)):
        raise ValueError("Batch setup missing on PretrainingConfig. Run through nanoplm CLI.")

    world_tokens_per_micro_step = achieved_global_batch_tokens // max(1, inferred_grad_accum_steps)
    logger.info(
        f"Batch setup: target_global_batch_size={pretrain_config.global_batch_size:,} tokens, "
        f"micro_step_tokens={world_tokens_per_micro_step:,}, "
        f"grad_accum_steps={inferred_grad_accum_steps}, "
        f"effective_global_batch_size={achieved_global_batch_tokens:,} tokens"
    )

    # ---- Run naming / output dir ----
    (
        _run_name, wandb_run_name, output_dir, num_epochs,
        logging_steps, eval_steps, save_steps, _resume_step,
    ) = _prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=manifest.train_sequences,
        global_batch_size_samples=global_batch_size_samples,
    )

    num_workers = _get_num_workers(pretrain_config.num_workers, effective_world_size)
    pin_memory = device.type == "cuda"
    # Disable persistent workers when packing: TokenPackingDataset + persistent_workers
    # can cause hangs near epoch boundaries (set_epoch doesn't reach workers, worker
    # queue non-determinism). Workers restart each epoch, fixing the stall.
    persistent_workers = num_workers > 0 and not use_packing
    prefetch_factor = pretrain_config.prefetch_factor if num_workers > 0 else None

    # ---- Distributed init ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = bool(pretrain_config.multi_gpu)

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, **({"device_id": local_rank} if backend == "nccl" else {}))

    is_main = (not distributed) or dist.get_rank() == 0
    eval_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else SequentialSampler(val_ds)

    # ---- Collator / Sampler / Packing ----
    train_sampler = None

    if use_packing:
        tokens_per_micro = pretrain_config.micro_batch_size * manifest.max_seq_len
        inner_sampler = DistributedSampler(train_ds, shuffle=True, seed=pretrain_config.seed) if distributed else RandomSampler(train_ds)

        train_ds = TokenPackingDataset(
            train_ds,
            max_tokens_per_batch=tokens_per_micro,
            drop_last=True,
            split_samples=False,
            sampler=inner_sampler,
        )
        inner_collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )
        if use_static_inp_size:
            min_tokens_per_seq = max(
                1,
                int(manifest.min_seq_len) + int(tokenizer.num_special_tokens_to_add(pair=False)),
            )
            # +1 leaves room for an optional trailing dummy sequence used for fixed-token padding.
            max_sequences_per_batch = max(1, tokens_per_micro // min_tokens_per_seq + 1)
            seq_count_buckets = build_power_of_two_buckets(max_sequences_per_batch, min_power_of_two=32)
            max_seqlen_buckets = build_power_of_two_buckets(int(manifest.max_seq_len), min_power_of_two=32)
            collator = DataCollatorWithFlattening(
                collator=inner_collator,
                return_position_ids=True,
                fixed_tokens_per_batch=tokens_per_micro,
                seq_count_buckets=seq_count_buckets,
                max_seqlen_buckets=max_seqlen_buckets,
            )
            logger.info(
                "Sequence packing ENABLED (static input size + bucketed metadata), "
                f"target={tokens_per_micro:,} tokens, "
                f"min_tokens_per_seq={min_tokens_per_seq}, "
                f"seq_count_buckets={seq_count_buckets}, "
                f"max_seqlen_buckets={max_seqlen_buckets}"
            )
        else:
            collator = DataCollatorWithFlattening(
                collator=inner_collator,
                pad_to_multiple_of=8,
                return_position_ids=True,
            )
            logger.info(
                "Sequence packing ENABLED (dynamic metadata), "
                f"target={tokens_per_micro:,} tokens"
            )
    else:
        collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=pretrain_config.seed) if distributed else RandomSampler(train_ds)

    # ---- FP8 (optional) ----
    if pretrain_config.fp8:
        if device.type != "cuda":
            raise ValueError("fp8=True requires CUDA.")
        fp8_filter = lambda mod, _: isinstance(mod, torch.nn.Linear) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
        convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name("tensorwise"), module_filter_fn=fp8_filter)
        n_fp8 = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
        n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        logger.info(f"FP8 enabled (tensorwise): converted={n_fp8} linear layers, skipped={max(0, n_lin - n_fp8)}")

    # ---- Model to device + compile + FSDP ----
    model.to(device)

    compile_dynamic_inner = bool(use_packing and not use_static_inp_size)
    compile_dynamic_root = not bool(use_packing and use_static_inp_size)
    if distributed:
        _compile_inner_layers_for_fsdp(model, dynamic=compile_dynamic_inner)
        logger.info(
            "Compiled inner transformer layers with "
            f"torch.compile(dynamic={compile_dynamic_inner}) before FSDP2 wrapping"
        )

    # Precision detection (needed before FSDP MixedPrecisionPolicy)
    use_bf16 = pretrain_config.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = pretrain_config.bf16 and (
        (device.type == "cuda" and not torch.cuda.is_bf16_supported()) or device.type == "mps"
    )

    fsdp_mesh = None
    if distributed:
        fsdp_kwargs: dict = {}
        if use_bf16:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fsdp_mesh = init_device_mesh("cuda", (effective_world_size,))
        for layer in model.model.layers:
            fully_shard(layer, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)
        fully_shard(model, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)

        # Explicit prefetching for FSDP2
        if N_PREFETCH_LAYERS_FSDP2 > 1:
            layers = model.model.layers
            for i, layer in enumerate(layers):
                if i + 1 < len(layers):
                    layer.set_modules_to_forward_prefetch(layers[i + 1 : i + 1 + N_PREFETCH_LAYERS_FSDP2])
                if i - 1 >= 0:
                    layer.set_modules_to_backward_prefetch(list(reversed(layers[max(0, i - N_PREFETCH_LAYERS_FSDP2) : i])))

    # Keep orig_model reference for checkpointing/eval (eval changes shapes → recompilation)
    orig_model = model
    if distributed:
        logger.info("Skipping root model torch.compile under FSDP2 (using pre-wrapped compiled inner layers)")
    else:
        model = torch.compile(model, dynamic=compile_dynamic_root)
        logger.info(f"Model compiled with torch.compile(dynamic={compile_dynamic_root})")

    # ---- DataLoaders ----
    eval_collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
    )

    dl_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    if use_packing:
        train_loader = DataLoader(train_ds, batch_size=None, collate_fn=collator, **dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=pretrain_config.micro_batch_size, collate_fn=collator, drop_last=False, **dl_kwargs)
    eval_loader = DataLoader(val_ds, sampler=eval_sampler, batch_size=pretrain_config.micro_batch_size, collate_fn=eval_collator, drop_last=False, **dl_kwargs)

    # ---- Optimizer / Scheduler ----
    optimizer = _create_optimizer(model, pretrain_config, distributed_mesh=fsdp_mesh)

    synced_len = _sync_train_loader_len(len(train_loader), distributed, device)
    # With TokenPackingDataset + DistributedSampler, greedy packing can yield different batch
    # counts per rank, causing desync and deadlock. Cap iteration at a safe minimum.
    if use_packing and distributed:
        _total_tokens = train_ds.dataset.total_tokens
        _tokens_per_rank = _total_tokens // effective_world_size
        _min_safe_batches = max(1, _tokens_per_rank // train_ds.max_tokens_per_batch)
        if _min_safe_batches < synced_len:
            logger.info(
                f"Capping micro-batches per epoch to {_min_safe_batches} (from {synced_len}) "
                "to prevent distributed deadlock with variable packing"
            )
            synced_len = _min_safe_batches
    steps_per_epoch = max(1, math.ceil(synced_len / max(1, inferred_grad_accum_steps)))
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(pretrain_config.warmup_steps, total_steps)
    scheduler = _create_scheduler(
        optimizer, warmup_steps, total_steps,
        pretrain_config.learning_rate, pretrain_config.lr_decay_to_fraction,
        pretrain_config.lr_schedule,
    )

    # ---- Resume ----
    start_step, start_epoch = 0, 0
    resume_micro_step, resume_epoch = 0, 0
    if resume_config and resume_config.is_resume:
        logger.info(f"Resuming from checkpoint: {resume_config.checkpoint_dir}")
        start_step, start_epoch = _load_checkpoint(model, optimizer, scheduler, resume_config.checkpoint_dir, device, distributed)
        logger.info(f"Resumed at global_step={start_step}, epoch={start_epoch}")
        resume_epoch = start_epoch
        steps_done = max(0, start_step - start_epoch * steps_per_epoch)
        resume_micro_step = steps_done * inferred_grad_accum_steps
        if resume_micro_step >= synced_len:
            resume_micro_step = 0
            start_epoch = min(start_epoch + 1, num_epochs)
            resume_epoch = start_epoch
        elif resume_micro_step > 0:
            logger.info(f"Skipping {resume_micro_step} micro-steps in resumed epoch {resume_epoch}")

    # ---- W&B ----
    wandb_enabled = False
    if is_main:
        try:
            wandb.init(
                project=pretrain_config.project_name,
                name=wandb_run_name,
                config={"pretrain": pretrain_config.__dict__, "total_steps": total_steps, "warmup_steps": warmup_steps},
            )
            wandb_enabled = wandb.run is not None
            if wandb_enabled:
                wandb.define_metric("train/global_step")
                wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
        except Exception as exc:
            logger.warning(f"W&B init failed, continuing without logging. Error: {exc}")

    # ---- AMP / TF32 ----
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
        if torch.cuda.is_available() else None
    )

    # ---- MFU estimation ----
    _raw = model if hasattr(model, "config") else getattr(model, "_orig_mod", model)
    _cfg = _raw.config
    _flops_per_token = _estimate_model_flops_per_token(
        _cfg.num_hidden_layers, _cfg.hidden_size, _cfg.intermediate_size,
        manifest.max_seq_len, _cfg.vocab_size,
    )
    _peak_flops = H100_PEAK_TFLOPS * 1e12
    logger.info(f"MFU estimation: {_flops_per_token:,} training FLOPs/token, H100 peak = {H100_PEAK_TFLOPS} TFLOPS")

    logger.info(
        f"Starting pure-torch training: epochs={num_epochs}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, grad_accum={inferred_grad_accum_steps},"
        f"achieved_global_batch_size={achieved_global_batch_tokens:,} tokens"
    )
    logger.info(
        f"Precision config: bf16={use_bf16}, fp16={use_fp16}, "
        f"tf32={(pretrain_config.tf32 and device.type == 'cuda')}, fp8={pretrain_config.fp8}"
    )

    # ---- Training loop ----
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

    epoch_setter = train_ds if use_packing else train_sampler

    for epoch in range(start_epoch, num_epochs):
        if epoch_setter is not None and hasattr(epoch_setter, "set_epoch"):
            epoch_setter.set_epoch(epoch)

        train_iter = iter(train_loader)
        for micro_step in range(synced_len):
            has_batch = True
            try:
                batch = next(train_iter)
            except StopIteration:
                has_batch = False
                batch = None

            # Token packing can still yield uneven batch counts per rank. If any rank
            # runs out early, stop this epoch on all ranks to avoid collective deadlock.
            if distributed and dist.is_initialized():
                has_batch_t = torch.tensor(
                    1 if has_batch else 0,
                    device=device,
                    dtype=torch.int32,
                )
                dist.all_reduce(has_batch_t, op=dist.ReduceOp.MIN)
                if int(has_batch_t.item()) == 0:
                    if is_main and micro_step + 1 < synced_len:
                        logger.warning(
                            "Ending epoch early at micro_step=%s due to uneven packed batches across ranks "
                            "(configured=%s).",
                            micro_step,
                            synced_len,
                        )
                    break
            elif not has_batch:
                break

            if not has_batch:
                break

            if resume_micro_step > 0 and epoch == resume_epoch and micro_step < resume_micro_step:
                continue

            if distributed and N_PREFETCH_LAYERS_FSDP2 > 1:
                model.unshard()

            batch = _move_batch_to_device(batch, device)

            # FSDP2: only reduce-scatter gradients at accumulation boundaries
            if distributed:
                sync = (micro_step + 1) % inferred_grad_accum_steps == 0 or micro_step + 1 == synced_len
                model.set_requires_gradient_sync(sync)

            if raw_token_count == 0:
                token_t0 = time.perf_counter()

            # Token counting
            if "num_valid_tokens" in batch:
                token_count += int(batch["num_valid_tokens"])
                raw_token_count += int(batch["input_ids"].numel())
            elif "attention_mask" in batch:
                token_count += int(batch["attention_mask"].sum().item())
                raw_token_count += int(batch["attention_mask"].numel())
            else:
                token_count += int(batch["input_ids"].numel())
                raw_token_count += int(batch["input_ids"].numel())

            # Forward
            amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
            with amp_ctx:
                fwd_kwargs = dict(input_ids=batch["input_ids"], labels=batch["labels"])
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

            # Backward
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

            # Skip to next micro-step if not at accumulation boundary
            is_boundary = (micro_step + 1) % inferred_grad_accum_steps == 0
            is_last = micro_step + 1 == synced_len
            if not (is_boundary or is_last):
                continue

            # accum_loss = sum(Li/grad_accum). For partial last step of epoch, true mean = sum(Li)/n_micro.
            # Scale so reported loss = true mean: accum_loss * grad_accum / n_micro
            n_micro = (micro_step + 1) % inferred_grad_accum_steps or inferred_grad_accum_steps
            accum_loss *= inferred_grad_accum_steps / n_micro

            # Optimizer step
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            step_skipped = False
            if scaler is not None and scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_skipped = scaler.get_scale() < old_scale
            else:
                optimizer.step()

            if isinstance(optimizer, (DionMuon, DionNorMuon)):
                muon_lr = optimizer.param_groups[0]["lr"]
                learning_rate = optimizer.param_groups[1]["lr"]
            else:
                learning_rate = optimizer.param_groups[0]["lr"]
                muon_lr = None

            if not step_skipped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            window_loss += accum_loss
            window_steps += 1
            accum_loss = 0.0

            # ---- Throughput / MFU ----
            t1 = time.perf_counter()
            elapsed = t1 - (token_t0 or t1)
            tok, raw_tok = float(token_count), float(raw_token_count)

            if distributed and dist.is_initialized():
                buf = torch.tensor([tok, raw_tok, elapsed], device=device)
                dist.all_reduce(buf[:2], op=dist.ReduceOp.SUM)
                dist.all_reduce(buf[2:], op=dist.ReduceOp.MAX)
                tok, raw_tok, elapsed = buf[0].item(), buf[1].item(), buf[2].item()

            tps = tok / max(elapsed, 1e-9)
            raw_tps = raw_tok / max(elapsed, 1e-9)
            mfu = (raw_tps * _flops_per_token / max(effective_world_size, 1)) / _peak_flops

            token_count = 0
            raw_token_count = 0
            token_t0 = None

            # ---- Logging ----
            should_log = global_step % logging_steps == 0
            vram_log = ""
            if should_log:
                vram_log = _format_vram_for_log(
                    device=device,
                    distributed=distributed,
                    reset_peak=True,
                )
            if should_log and is_main:
                avg_loss = window_loss / max(1, window_steps)
                waste = (1.0 - tok / max(raw_tok, 1)) * 100
                muon_str = f"muon_lr={muon_lr:.2e} " if muon_lr is not None else ""
                logger.info(
                    f"[step {global_step}/{total_steps}] "
                    f"loss={avg_loss:.4f} lr={learning_rate:.2e} {muon_str}"
                    f"grad_norm={grad_norm:.4f} tok/s={tps:,.0f} "
                    f"raw_tok/s={raw_tps:,.0f} "
                    f"step_tokens={int(tok):,} waste={waste:.1f}% "
                    f"h100_mfu={mfu:.2%} {vram_log}"
                )
                payload = {
                    "train/global_step": global_step,
                    "train/loss": avg_loss,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": learning_rate,
                    "train/epoch": epoch + (micro_step + 1) / synced_len,
                    "train/tokens_per_sec": tps,
                    "train/raw_tokens_per_sec": raw_tps,
                    "train/step_real_tokens": int(tok),
                    "train/step_raw_tokens": int(raw_tok),
                    "train/packing_waste_pct": waste,
                }
                if muon_lr is not None:
                    payload["train/muon_lr"] = muon_lr
                if wandb_enabled and wandb.run is not None:
                    try:
                        wandb.log(payload)
                    except Exception as exc:
                        wandb_enabled = False
                        logger.warning(f"W&B log failed; disabling. Error: {exc}")

            # GC: after first step, freeze; then periodic collect
            if first_step_of_run:
                first_step_of_run = False
                gc.collect(); gc.freeze(); gc.disable()
            elif global_step % 5000 == 0:
                gc.collect()

            if should_log:
                window_loss = 0.0
                window_steps = 0

            # ---- Eval ----
            if global_step % eval_steps == 0:
                eval_loss = _evaluate(orig_model, eval_loader, device, distributed, amp_dtype)
                if is_main:
                    logger.info(f"[step {global_step}] eval_loss={eval_loss:.4f}")
                    if wandb_enabled and wandb.run is not None:
                        try:
                            wandb.log({"train/global_step": global_step, "eval/loss": eval_loss})
                        except Exception as exc:
                            wandb_enabled = False
                            logger.warning(f"W&B log failed; disabling. Error: {exc}")
                model.train()

            # ---- Save ----
            if global_step % save_steps == 0:
                _save_checkpoint(
                    model, optimizer, scheduler, global_step, epoch,
                    output_dir, logging_steps, eval_steps, save_steps,
                    distributed, is_main,
                )

        if resume_micro_step > 0 and epoch == resume_epoch:
            resume_micro_step = 0

    # ---- Finalize ----
    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    _save_checkpoint(
        model, optimizer, scheduler, global_step, num_epochs,
        output_dir, logging_steps, eval_steps, save_steps,
        distributed, is_main,
    )

    if is_main and wandb_enabled and wandb.run is not None:
        try:
            (Path(output_dir) / "wandb_run_id.txt").write_text(wandb.run.id, encoding="utf-8")
            wandb.finish()
        except Exception as exc:
            logger.warning(f"W&B finalize failed. Error: {exc}")

    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    logger.info("Pure-torch pretraining complete.")
