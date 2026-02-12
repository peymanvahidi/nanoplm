"""
Pure-torch pretraining pipeline.

Drop-in alternative to ``pipeline.py`` that replaces the HF ``Trainer``
with a manual PyTorch training loop.  Everything else (datasets, collator,
run naming, checkpointing, W&B logging) is deliberately kept compatible.

Usage via CLI:  ``nanoplm pretrain from-yaml pretrain.yaml --pure-torch``
"""

import os
import json
import math
import re
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from typing import Optional, Tuple
from pathlib import Path

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.dataset import (
    LoadShardedFastaMLMDataset,
)
from nanoplm.pretraining.collator import ProtDataCollatorForLM

# Reuse helpers from the existing pipeline (configs, run naming, workers)
from nanoplm.pretraining.pipeline import (
    PretrainingConfig,
    ResumeConfig,
    _prepare_run_and_steps,
    _get_num_workers,
)
from nanoplm.data.manifest import read_manifest, validate_manifest_for_pipeline
from nanoplm.data.validation import validate_pretrain_dataset, ValidationError
from nanoplm.utils.logger import logger
from nanoplm.utils.common import get_device, create_dirs


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Optimizer (matches HF Trainer's AdamW parameter grouping)
# ---------------------------------------------------------------------------

def _create_optimizer(
    model: torch.nn.Module,
    cfg: PretrainingConfig,
) -> torch.optim.Optimizer:
    """Create optimizer with HF-compatible parameter grouping."""
    raw_model = model.module if isinstance(model, DDP) else model
    decay_params = []
    no_decay_params = []

    # HF Trainer equivalent:
    # get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
    forbidden_name_patterns = [
        r"bias",
        r"layernorm",
        r"rmsnorm",
        r"(?:^|\.)norm(?:$|\.)",
        r"_norm(?:$|\.)",
    ]
    decay_parameter_names = set(
        _get_parameter_names(
            raw_model,
            forbidden_layer_types=[nn.LayerNorm],
            forbidden_layer_names=forbidden_name_patterns,
        )
    )

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if name in decay_parameter_names:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer_name = str(cfg.optimizer).lower()
    common_kwargs = {
        "params": groups,
        "lr": float(cfg.learning_rate),
        "betas": (float(cfg.adam_beta1), float(cfg.adam_beta2)),
        "eps": float(cfg.adam_epsilon),
    }

    if optimizer_name == "adamw":
        return torch.optim.AdamW(**common_kwargs)

    if optimizer_name == "stable_adamw":
        stable_adamw = getattr(torch.optim, "StableAdamW", None)
        if stable_adamw is None:
            logger.warning(
                "torch.optim.StableAdamW is unavailable in this environment; "
                "falling back to AdamW in pure-torch pipeline."
            )
            return torch.optim.AdamW(**common_kwargs)
        return stable_adamw(**common_kwargs)

    raise ValueError(
        f"Invalid optimizer: {cfg.optimizer}. Currently supported: [adamw, stable_adamw]"
    )


# ---------------------------------------------------------------------------
# LR scheduler  (linear warmup  + linear decay — matches HF default)
# ---------------------------------------------------------------------------

def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    # Mirrors transformers.optimization._get_linear_schedule_with_warmup_lr_lambda
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def _get_parameter_names(
    model: torch.nn.Module,
    forbidden_layer_types: list[type[torch.nn.Module]],
    forbidden_layer_names: Optional[list[str]] = None,
) -> list[str]:
    """
    Local copy of HF trainer_pt_utils.get_parameter_names.

    Returns names of parameters not inside forbidden layer types and not matching
    forbidden name patterns.
    """
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names]
        if forbidden_layer_names is not None
        else []
    )

    result: list[str] = []
    for name, child in model.named_children():
        child_params = _get_parameter_names(
            child,
            forbidden_layer_types=forbidden_layer_types,
            forbidden_layer_names=forbidden_layer_names,
        )
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
        ]

    # Parameters defined directly on this module (not in any child module)
    result += [
        k
        for k in model._parameters
        if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]

    return result


def _hf_num_update_steps_per_epoch(
    train_loader_len: int,
    gradient_accumulation_steps: int,
) -> int:
    """Match HF Trainer update-step counting (ceil over micro-steps)."""
    if train_loader_len <= 0:
        return 1
    steps = train_loader_len // gradient_accumulation_steps
    if train_loader_len % gradient_accumulation_steps != 0:
        steps += 1
    return max(1, steps)


def _hf_get_warmup_steps(num_training_steps: int, warmup_value: float) -> int:
    """Mirror transformers.TrainingArguments.get_warmup_steps."""
    if warmup_value >= 1:
        return int(warmup_value)
    return math.ceil(num_training_steps * warmup_value)


def _dist_barrier(local_rank: int) -> None:
    """Distributed barrier that avoids NCCL device warning."""
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    distributed: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in eval_loader:
        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
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
        bs = batch["input_ids"].size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    # All-reduce across ranks so each process gets the global mean eval loss
    if distributed and dist.is_initialized():
        stats = torch.tensor([total_loss, total_samples], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_samples = stats[0].item(), stats[1].item()

    model.train()
    return total_loss / max(1, total_samples)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

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
) -> None:
    ckp_dir = Path(output_dir) / f"checkpoint-{global_step}"
    create_dirs(str(ckp_dir))

    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(raw_model.state_dict(), ckp_dir / "pytorch_model.bin")
    torch.save(optimizer.state_dict(), ckp_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckp_dir / "scheduler.pt")
    torch.save({
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, ckp_dir / "rng_state.pth")

    state = {
        "global_step": global_step,
        "epoch": epoch,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
    }
    (ckp_dir / "training_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    logger.info(f"Checkpoint saved → {ckp_dir}")


def _load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    checkpoint_dir: str,
    device: torch.device,
) -> Tuple[int, int]:
    """Load checkpoint and return ``(global_step, epoch)``."""
    ckp = Path(checkpoint_dir)

    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(
        torch.load(ckp / "pytorch_model.bin", map_location=device, weights_only=True)
    )

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
        state = json.loads(state_path.read_text())
        return state["global_step"], state["epoch"]
    return 0, 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pure_pretraining(
    model: PureProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:
    """Pure-torch training loop — drop-in replacement for ``run_pretraining``."""

    _set_seed(pretrain_config.seed)
    tokenizer = model.tokenizer
    device = torch.device(get_device())

    # ---- datasets (same manifest-driven logic as HF pipeline) -----------
    dataset_dir = Path(pretrain_config.dataset_dir)

    logger.info(f"Validating pretraining dataset at {dataset_dir}...")
    try:
        validation_result = validate_pretrain_dataset(dataset_dir)
        logger.info(
            f"Dataset validated: {validation_result['manifest']['train_sequences']:,} train, "
            f"{validation_result['manifest']['val_sequences']:,} val sequences"
        )
    except ValidationError as e:
        logger.error(f"Dataset validation failed: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        raise

    manifest = read_manifest(dataset_dir)
    validate_manifest_for_pipeline(manifest=manifest, expected_mode="pretrain")

    train_hdf5_dir = dataset_dir / manifest.train_dir
    val_hdf5_dir = dataset_dir / manifest.val_dir
    train_sequences = manifest.train_sequences
    val_sequences = manifest.val_sequences

    logger.info(f"Loaded config from manifest: {dataset_dir}")
    logger.info(f"  train_hdf5: {train_hdf5_dir}")
    logger.info(f"  val_hdf5: {val_hdf5_dir}")
    logger.info(f"  max_length: {manifest.max_seq_len}")
    logger.info(f"  train_sequences: {train_sequences}")
    logger.info(f"  val_sequences: {val_sequences}")

    if not train_hdf5_dir.exists():
        raise FileNotFoundError(f"Train HDF5 directory not found: {train_hdf5_dir}")
    if not val_hdf5_dir.exists():
        raise FileNotFoundError(f"Validation HDF5 directory not found: {val_hdf5_dir}")

    logger.info("Using LoadShardedFastaMLMDataset for pre-tokenized HDF5 shards")
    try:
        train_ds = LoadShardedFastaMLMDataset(
            hdf5_dir=str(train_hdf5_dir),
            load_all_in_memory=pretrain_config.load_all_in_memory,
        )
        val_ds = LoadShardedFastaMLMDataset(
            hdf5_dir=str(val_hdf5_dir),
            load_all_in_memory=pretrain_config.load_all_in_memory,
        )
    except FileNotFoundError as e:
        logger.error(
            f"HDF5 shards not found! You need to create them first.\n"
            f"Run: nanoplm data from-yaml with pipeline_mode: 'pretrain'\n"
            f"Error: {e}"
        )
        raise

    # ---- collator --------------------------------------------------------
    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
    )

    create_dirs(pretrain_config.ckp_dir)

    # ---- world size / DDP ------------------------------------------------
    if pretrain_config.multi_gpu:
        if pretrain_config.world_size == "auto":
            env_ws = os.environ.get("WORLD_SIZE")
            effective_world_size = (
                int(env_ws) if env_ws else max(torch.cuda.device_count(), 1)
            )
        else:
            effective_world_size = (
                int(pretrain_config.world_size) if pretrain_config.world_size else 1
            )
    else:
        effective_world_size = 1

    global_batch_size = (
        pretrain_config.gradient_accumulation_steps
        * pretrain_config.batch_size
        * effective_world_size
    )

    # ---- run naming & step intervals (reuse existing helper) -------------
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
        global_batch_size=global_batch_size,
    )

    num_workers = _get_num_workers(pretrain_config.num_workers, effective_world_size)
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    prefetch_factor = pretrain_config.prefetch_factor if num_workers > 0 else None

    # ---- DDP setup -------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if pretrain_config.multi_gpu:
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
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)
        train_sampler = DistributedSampler(
            train_ds, shuffle=True, seed=pretrain_config.seed
        )
    else:
        model.to(device)
        train_sampler = RandomSampler(train_ds)

    if pretrain_config.multi_gpu:
        eval_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        eval_sampler = SequentialSampler(val_ds)

    # ---- data loaders ----------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=pretrain_config.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )
    eval_loader = DataLoader(
        val_ds,
        sampler=eval_sampler,
        batch_size=pretrain_config.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )

    # ---- optimizer & scheduler -------------------------------------------
    optimizer = _create_optimizer(model, pretrain_config)

    # Match HF Trainer: ceil update steps across micro-batches.
    steps_per_epoch = _hf_num_update_steps_per_epoch(
        train_loader_len=len(train_loader),
        gradient_accumulation_steps=pretrain_config.gradient_accumulation_steps,
    )
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = _hf_get_warmup_steps(
        num_training_steps=total_steps,
        warmup_value=float(pretrain_config.warmup_ratio),
    )
    scheduler = _create_scheduler(optimizer, warmup_steps, total_steps)

    # ---- resume ----------------------------------------------------------
    start_step = 0
    start_epoch = 0
    if resume_config and resume_config.is_resume:
        logger.info(f"Resuming from checkpoint: {resume_config.checkpoint_dir}")
        start_step, start_epoch = _load_checkpoint(
            model, optimizer, scheduler, resume_config.checkpoint_dir, device
        )
        logger.info(f"Resumed at global_step={start_step}, epoch={start_epoch}")
    resume_micro_step = 0
    resume_epoch = start_epoch
    if resume_config and resume_config.is_resume:
        steps_completed_in_epoch = max(0, start_step - start_epoch * steps_per_epoch)
        resume_micro_step = steps_completed_in_epoch * pretrain_config.gradient_accumulation_steps
        if resume_micro_step >= len(train_loader):
            resume_micro_step = 0
            start_epoch = min(start_epoch + 1, num_epochs)
            resume_epoch = start_epoch
        if resume_micro_step > 0:
            logger.info(
                f"Skipping {resume_micro_step} micro-steps in resumed epoch {resume_epoch}"
            )

    # ---- W&B -------------------------------------------------------------
    is_main = local_rank == 0
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
        except Exception as exc:
            logger.warning(f"W&B init failed; continuing without logging. Error: {exc}")

    # ---- training loop ---------------------------------------------------
    logger.info(
        f"Starting pure-torch training: {num_epochs} epochs, "
        f"{total_steps} total optimiser steps, "
        f"warmup={warmup_steps}, "
        f"grad_accum={pretrain_config.gradient_accumulation_steps}"
    )

    global_step = start_step

    # ---- mixed precision + tf32 (match HF TrainingArguments behavior) ----
    use_bf16 = (
        pretrain_config.bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = (
        pretrain_config.bf16
        and (
            (device.type == "cuda" and not torch.cuda.is_bf16_supported())
            or device.type == "mps"
        )
    )
    amp_dtype: Optional[torch.dtype] = None
    if use_bf16:
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

    logger.info(
        f"Precision config: bf16={use_bf16}, fp16={use_fp16}, "
        f"tf32={(pretrain_config.tf32 and device.type == 'cuda')}"
    )

    model.train()

    # tokens/sec tracking
    _tok_count = 0              # non-padding tokens processed since last log
    _raw_tok_count = 0          # raw tokens processed since last log
    _tok_t0: Optional[float] = None  # wall-clock at first micro-step after last log
    window_loss = 0.0
    window_step_count = 0

    for epoch in range(start_epoch, num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        accum_loss = 0.0

        for micro_step, batch in enumerate(train_loader):
            if resume_micro_step > 0 and epoch == resume_epoch and micro_step < resume_micro_step:
                continue
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # count tokens in this micro-batch
            if _raw_tok_count == 0:
                _tok_t0 = time.perf_counter()
            _tok_count += int(batch["attention_mask"].sum().item())
            _raw_tok_count += int(batch["attention_mask"].numel())

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
            loss = loss / pretrain_config.gradient_accumulation_steps
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()

            is_accum_boundary = (micro_step + 1) % pretrain_config.gradient_accumulation_steps == 0
            is_last_step = micro_step + 1 == len(train_loader)

            if is_accum_boundary or is_last_step:
                # grad_norm before clipping (matches HF Trainer)
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                optimizer_step_was_skipped = False
                if scaler is not None and scaler.is_enabled():
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = scaler.get_scale()
                    optimizer_step_was_skipped = new_scale < old_scale
                else:
                    optimizer.step()
                # Match HF: log LR before scheduler update, and skip scheduler step on overflow.
                learning_rate = optimizer.param_groups[0]["lr"]
                if not optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                window_loss += accum_loss
                window_step_count += 1

                # --- tokens/sec (every step) ---
                _tok_t1 = time.perf_counter()
                if _tok_t0 is None:
                    _tok_t0 = _tok_t1
                _tok_elapsed = _tok_t1 - _tok_t0
                tok_count = float(_tok_count)
                raw_tok_count = float(_raw_tok_count)
                tok_elapsed = float(_tok_elapsed)
                if pretrain_config.multi_gpu and dist.is_initialized():
                    tok_tensor = torch.tensor(tok_count, device=device)
                    raw_tok_tensor = torch.tensor(raw_tok_count, device=device)
                    time_tensor = torch.tensor(tok_elapsed, device=device)
                    dist.all_reduce(tok_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(raw_tok_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
                    tok_count = tok_tensor.item()
                    raw_tok_count = raw_tok_tensor.item()
                    tok_elapsed = time_tensor.item()
                tokens_per_sec = tok_count / max(tok_elapsed, 1e-9)
                raw_tokens_per_sec = raw_tok_count / max(tok_elapsed, 1e-9)
                _tok_count = 0
                _raw_tok_count = 0
                _tok_t0 = None

                # Match HF Trainer default logging cadence (logging_first_step=False):
                # log strictly on logging_steps boundaries.
                should_log = (global_step % logging_steps == 0)
                if is_main and should_log:
                    loss_to_log = window_loss / max(1, window_step_count)
                    log_payload = {
                        "train/loss": loss_to_log,
                        "train/grad_norm": grad_norm,
                        "train/learning_rate": learning_rate,
                        "train/epoch": epoch + (micro_step + 1) / len(train_loader),
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/raw_tokens_per_sec": raw_tokens_per_sec,
                    }
                    if wandb_enabled and wandb.run is not None:
                        try:
                            wandb.log(log_payload, step=global_step)
                        except Exception as exc:
                            wandb_enabled = False
                            logger.warning(f"W&B log failed; disabling logging. Error: {exc}")
                    logger.info(
                        f"[step {global_step}/{total_steps}] "
                        f"loss={loss_to_log:.4f}  lr={learning_rate:.2e}  "
                        f"grad_norm={grad_norm:.4f}  "
                        f"tok/s={tokens_per_sec:,.0f}  "
                        f"raw_tok/s={raw_tokens_per_sec:,.0f}"
                    )
                if should_log:
                    window_loss = 0.0
                    window_step_count = 0

                accum_loss = 0.0

                # --- evaluation ---
                if global_step % eval_steps == 0:
                    eval_loss = _evaluate(
                        model,
                        eval_loader,
                        device,
                        distributed=pretrain_config.multi_gpu,
                        amp_dtype=amp_dtype,
                    )
                    if is_main:
                        if wandb_enabled and wandb.run is not None:
                            try:
                                wandb.log({"eval/loss": eval_loss}, step=global_step)
                            except Exception as exc:
                                wandb_enabled = False
                                logger.warning(f"W&B log failed; disabling logging. Error: {exc}")
                        logger.info(
                            f"[step {global_step}] eval_loss={eval_loss:.4f}"
                        )
                    model.train()

                # --- checkpoint ---
                if global_step % save_steps == 0:
                    if is_main:
                        _save_checkpoint(
                            model, optimizer, scheduler,
                            global_step, epoch, output_dir,
                            logging_steps, eval_steps, save_steps,
                        )

        if resume_micro_step > 0 and epoch == resume_epoch:
            resume_micro_step = 0

    # ---- final save & cleanup --------------------------------------------
    if pretrain_config.multi_gpu and dist.is_initialized():
        _dist_barrier(local_rank)
    if is_main:
        _save_checkpoint(
            model, optimizer, scheduler,
            global_step, num_epochs, output_dir,
            logging_steps, eval_steps, save_steps,
        )
        if wandb_enabled and wandb.run is not None:
            try:
                run_id_path = Path(output_dir) / "wandb_run_id.txt"
                run_id_path.write_text(wandb.run.id, encoding="utf-8")
                wandb.finish()
            except Exception as exc:
                logger.warning(f"W&B finalize failed. Error: {exc}")
    if pretrain_config.multi_gpu and dist.is_initialized():
        _dist_barrier(local_rank)

    logger.info("Pure-torch pretraining complete.")
