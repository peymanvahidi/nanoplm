import os
import math
import json
import time
import torch
import torch.distributed as dist
import wandb
from datetime import datetime
from typing import Optional
from pathlib import Path

from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
from nanoplm.pretraining.models.modern_bert import ProtModernBertMLM
from nanoplm.pretraining.dataset import ShardedDataset
from nanoplm.pretraining.collator import ProtDataCollatorForLM
from dion import Muon as DionMuon, NorMuon as DionNorMuon
from torch.optim.lr_scheduler import LambdaLR

from nanoplm.pretraining.optim import build_muon_optimizer, is_muon_optimizer, unwrap_model
from nanoplm.data.validation import validate_pretrain_dataset
from nanoplm.utils.logger import logger
from nanoplm.utils.common import get_device, create_dirs
from nanoplm.utils.wandb_artifacts import upload_run_source_snapshot

# TODO: these are from the master branch
# from nanoplm.pretraining.optim import build_muon_optimizer, is_muon_optimizer
# from nanoplm.pretraining.utils import (
#     compute_batch_setup,
#     get_num_workers,
#     prepare_run_and_steps,
# )
# from nanoplm.data.validation import validate_pretrain_dataset
# from nanoplm.utils.logger import logger
# from nanoplm.utils.common import get_device, create_dirs, resolve_world_size


def _create_scheduler(optimizer, warmup_steps: int, total_steps: int, learning_rate: float, lr_decay_to_fraction: float, lr_schedule: str = "Linear") -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        if lr_schedule.lower() == "cosine":
            return lr_decay_to_fraction + 0.5 * (1.0 - lr_decay_to_fraction) * (1.0 + math.cos(math.pi * progress))
        else:
            return max(lr_decay_to_fraction, 1.0 - (1.0 - lr_decay_to_fraction) * progress)

    return LambdaLR(optimizer, lr_lambda)


class TokenTrackingTrainer(Trainer):
    """Trainer subclass that injects tokens/sec into wandb logs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_tok_count = 0
        self._step_raw_tok_count = 0
        self._step_t0 = time.perf_counter()
        self._last_tokens_per_sec = 0.0
        self._last_raw_tokens_per_sec = 0.0

    def training_step(self, model, inputs, num_items_in_batch=None):
        if "attention_mask" in inputs:
            self._step_tok_count += int(inputs["attention_mask"].sum().item())
            self._step_raw_tok_count += int(inputs["attention_mask"].numel())
        elif "input_ids" in inputs:
            self._step_raw_tok_count += int(inputs["input_ids"].numel())

        loss = super().training_step(model, inputs, num_items_in_batch)

        t1 = time.perf_counter()
        elapsed = t1 - self._step_t0
        tok_count = float(self._step_tok_count)
        raw_tok_count = float(self._step_raw_tok_count)
        tok_elapsed = float(elapsed)
        if dist.is_available() and dist.is_initialized():
            if "attention_mask" in inputs:
                device = inputs["attention_mask"].device
            elif "input_ids" in inputs:
                device = inputs["input_ids"].device
            else:
                device = loss.device
            tok_tensor = torch.tensor(tok_count, device=device)
            raw_tok_tensor = torch.tensor(raw_tok_count, device=device)
            time_tensor = torch.tensor(tok_elapsed, device=device)
            dist.all_reduce(tok_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(raw_tok_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
            tok_count = tok_tensor.item()
            raw_tok_count = raw_tok_tensor.item()
            tok_elapsed = time_tensor.item()
        self._last_tokens_per_sec = tok_count / max(tok_elapsed, 1e-9)
        self._last_raw_tokens_per_sec = raw_tok_count / max(tok_elapsed, 1e-9)

        self._step_tok_count = 0
        self._step_raw_tok_count = 0
        self._step_t0 = t1
        return loss

    def log(self, logs, start_time=None, **kwargs):
        if logs is None:
            logs = {}

        optimizer = self.optimizer
        seen: set[int] = set()
        while optimizer is not None and not is_muon_optimizer(optimizer):
            opt_id = id(optimizer)
            if opt_id in seen:
                break
            seen.add(opt_id)
            inner = getattr(optimizer, "optimizer", None)
            if inner is None or inner is optimizer:
                break
            optimizer = inner

        if is_muon_optimizer(optimizer):
            # param_groups[0] = muon, param_groups[1] = adamw
            muon_lr = optimizer.param_groups[0]["lr"]
            adamw_lr = optimizer.param_groups[1]["lr"]
            logs["learning_rate"] = adamw_lr
            logs["adamw_lr"] = adamw_lr
            logs["muon_lr"] = muon_lr
        logs["tokens_per_sec"] = self._last_tokens_per_sec
        logs["raw_tokens_per_sec"] = self._last_raw_tokens_per_sec
        super().log(logs, start_time=start_time, **kwargs)


# TODO: in the master branch we have moved all the cofigs to the config.py file, we should move them from here to that file
class WandbSourceSnapshotCallback(TrainerCallback):
    """Upload run source snapshot once W&B is active."""

    def on_train_begin(self, args, state, control, **kwargs):
        upload_run_source_snapshot()
        return control


@dataclass
class PretrainingConfig:
    # Dataset directory (contains .data_manifest from nanoplm data from-yaml)
    dataset_dir: Union[str, Path]

    # Checkpoint and output
    ckp_dir: str = "output/pretraining"

    # Training hyperparameters
    micro_batch_size: int = 64
    num_epochs: int = 10
    warmup_steps: int = 302
    # RePO activation warmup (optimizer-step based) for pure-torch path.
    # If None, defaults to warmup_steps (backward-compatible behavior).
    repo_rope_warmup_steps: Optional[int] = None
    lr_decay_to_fraction: float = 0.1
    lr_schedule: str = "cosine"
    optimizer: str = "normuon"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    # Gradient clipping threshold (L2 norm). Set float("inf") to disable clipping.
    max_grad_norm: float = float("inf")
    # Muon-specific hyperparameters (used only when optimizer == "muon" or "normuon").
    # Plain learning_rate/weight_decay/adam_* are used for the AdamW sub-optimizer.
    muon_learning_rate: float = 1e-3
    muon_weight_decay: float = 0.01
    muon_cautious_weight_decay: bool = True
    muon_use_polar_express: bool = False
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_eps: float = 1e-7
    # Target effective batch size in tokens per optimizer step.
    # gradient_accumulation_steps is inferred from this value at runtime.
    global_batch_size: int = 256000
    inferred_grad_accum_steps: Optional[int] = None
    global_batch_size_samples: Optional[int] = None
    achieved_global_batch_tokens: Optional[int] = None

    # Mixed precision
    bf16: bool = True
    tf32: bool = True
    fp8: bool = False

    # MLM settings
    mlm_probability: float = 0.3
    mask_replace_prob: float = 0.8
    random_token_prob: float = 0.1
    keep_probability: float = 0.1

    # Logging/checkpointing
    logging_steps: int = 1
    eval_steps: int = 250
    save_steps: int = 5000
    seed: int = 42

    # Data loading
    num_workers: Union[int, str] = "auto"
    prefetch_factor: int = 2

    # Sequence packing (packs multiple sequences per row to eliminate padding waste).
    # Requires flash attention (varlen path).  Falls back to padding if disabled.
    use_packing: bool = True
    # When packing is enabled, force fixed flat token count and bucketed attention metadata
    # (cu_seqlens/max_seqlen). This enables static-shape execution for torch.compile
    # (dynamic=False) and improves CUDA graph capture reuse.
    use_static_inp_size: bool = True
    # If enabled, pure-torch compile uses mode='max-autotune-no-cudagraphs'.
    # This may improve steady-state performance, but increases compile/autotune time
    # noticeably at the start of a run.
    use_compile_max_autotune: bool = False
    # TorchInductor Triton setting. Persistent reductions can exceed per-block shared
    # memory limits for some fused kernels (e.g. LayerNorm backward at hidden >= 1536),
    # causing "No valid triton configs / out of resource". Set to False to force
    # non-persistent reduction kernels while keeping torch.compile enabled.
    compile_triton_persistent_reductions: bool = False
    # TorchInductor Triton setting. Mix-order reductions can generate persistent
    # reduction kernels that exceed shared memory limits on some shapes (notably
    # layernorm backward fusions at larger hidden sizes). Disable to prefer the
    # legacy reduction codegen while keeping torch.compile enabled.
    compile_triton_mix_order_reduction: bool = False

    # Profiling (pure_torch / pure_te pipelines). When enabled on rank 0:
    # - If running under nsys: uses CUDA Profiler API (start/stop at steps) for .nsys-rep traces.
    # - Otherwise: uses PyTorch profiler and exports a Chrome trace (chrome://tracing) to ckp_dir.
    profiler_enabled: bool = False
    profiler_start_step: int = 10
    profiler_end_step: int = 15

    # Distributed training
    multi_gpu: bool = True
    world_size: Union[int, str] = "auto"
    project_name: str = "nanoplm-pretraining"


@dataclass
class ResumeConfig:
    is_resume: bool
    checkpoint_dir: str
    extra_epochs: Optional[int] = None
    # Schedule override options for resumed training.
    # When set to None (the YAML default), the corresponding value is read
    # from the checkpoint's saved pretrain_config.yaml so the original
    # schedule is reconstructed exactly.  Explicit values override.
    warmup_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    muon_learning_rate: Optional[float] = None
    lr_schedule: Optional[str] = None
    lr_decay_to_fraction: Optional[float] = None
    # reset_scheduler: rebuild the LR schedule from step 0 covering only the
    # remaining training steps (new warmup + decay).  Useful when you want a
    # fresh learning-rate curve after a long pause.
    reset_scheduler: bool = False
    # skip_warmup: jump straight to peak LR on resume (sets warmup_steps=0).
    skip_warmup: bool = False


def _archive_future_checkpoints(run_dir: Path, resume_step: int) -> None:
    """Archive checkpoints with steps greater than resume_step.

    When resuming from a checkpoint, any checkpoints with higher step numbers
    are moved to an archived subdirectory to prevent conflicts while preserving
    the data for potential future analysis.

    Args:
        run_dir: The run directory containing checkpoints
        resume_step: The step number being resumed from
    """
    checkpoints_to_archive = []

    for ckpt_dir in run_dir.glob("checkpoint-*"):
        try:
            step = int(ckpt_dir.name.split("-")[1])
            if step > resume_step:
                checkpoints_to_archive.append((step, ckpt_dir))
        except (IndexError, ValueError):
            continue

    if checkpoints_to_archive:
        checkpoints_to_archive.sort()

        # Create archive directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = run_dir / f"archived_{timestamp}"
        archive_dir.mkdir(exist_ok=True)

        logger.warning(
            f"Found {len(checkpoints_to_archive)} checkpoint(s) with steps > {resume_step}. "
            f"Moving to archive: {[s for s, _ in checkpoints_to_archive]}"
        )

        for step, ckpt_path in checkpoints_to_archive:
            dest = archive_dir / ckpt_path.name
            logger.info(f"Archiving checkpoint-{step} to {archive_dir.name}/")
            shutil.move(str(ckpt_path), str(dest))

        logger.info(f"Archived checkpoints moved to: {archive_dir}")


def _prepare_run_and_steps(
    pretrain_config: "PretrainingConfig",
    resume_config: Optional["ResumeConfig"],
    train_samples: int,
    global_batch_size_samples: int,
) -> Tuple[str, str, str, int, int, int, int, Optional[int]]:
    """Prepare run naming/dirs and compute epochs & step intervals.

    Args:
        pretrain_config: Pretraining configuration
        resume_config: Resume configuration (if resuming)
        train_samples: Number of training samples (from manifest)
        global_batch_size_samples: Effective samples per optimizer step

    Returns a tuple: (run_name, wandb_run_name, output_dir, num_epochs,
                      logging_steps, eval_steps, save_steps, resume_step)
    """
    ckp_root = Path(pretrain_config.ckp_dir)

    # Determine run directory and name
    if resume_config and resume_config.is_resume:
        checkpoint_path = Path(resume_config.checkpoint_dir)
        original_run_name = checkpoint_path.parent.name
        run_name = original_run_name  # Continue in same directory
        run_root = ckp_root / run_name

        # Track resume counter for W&B run naming
        counter_file = run_root / ".resume_counter"
        if counter_file.exists():
            try:
                resume_counter = int(counter_file.read_text().strip()) + 1
            except (ValueError, FileNotFoundError):
                resume_counter = 1
        else:
            resume_counter = 1

        # Save updated counter
        counter_file.write_text(str(resume_counter), encoding="utf-8")

        # Create W&B run name with counter
        wandb_run_name = f"{run_name}-re{resume_counter}"
        logger.info(f"Resume session #{resume_counter}: W&B run name = {wandb_run_name}")

        # Archive any future checkpoints to prevent conflicts
        resume_step = None
        try:
            resume_step = int(checkpoint_path.name.split("-")[1])
            _archive_future_checkpoints(run_root, resume_step)
        except (IndexError, ValueError) as e:
            logger.warning(
                f"Could not extract step number from checkpoint path: {checkpoint_path.name}. "
                f"Skipping future checkpoint archival. Error: {e}"
            )
    else:
        base_stamp = datetime.now().strftime("%d%m%H%M")
        base_name = f"run-{base_stamp}"
        candidate = base_name
        if ckp_root.exists():
            suffix = 2
            while (ckp_root / candidate).exists():
                candidate = f"{base_name}-{suffix}"
                suffix += 1
        run_name = candidate
        run_root = ckp_root / run_name
        wandb_run_name = run_name  # Same as directory name for new runs
        resume_step = None

    create_dirs(str(run_root))
    output_dir = str(run_root)

    # Persist run metadata for future resumes
    try:
        (Path(output_dir) / "run_name.txt").write_text(run_name, encoding="utf-8")
    except Exception:
        pass

    # Compute epochs and step intervals
    if resume_config and resume_config.is_resume:
        training_args_path = Path(resume_config.checkpoint_dir) / "training_args.bin"

        if resume_config.extra_epochs is not None and resume_config.extra_epochs > 0:
            num_epochs = pretrain_config.num_epochs + int(resume_config.extra_epochs)
        else:
            num_epochs = pretrain_config.num_epochs

        # Preserve original logging/eval/save intervals when available
        if training_args_path.exists():
            try:
                original_args = torch.load(training_args_path, weights_only=False)
                logging_steps = original_args.logging_steps
                eval_steps = original_args.eval_steps
                save_steps = original_args.save_steps
                logger.info(
                    f"Resuming with preserved intervals: save_steps={save_steps}, eval_steps={eval_steps}"
                )
            except Exception:
                # Use ceiling division to ensure at least 1 step per epoch
                steps_per_epoch = (train_samples + global_batch_size_samples - 1) // global_batch_size_samples
                total_steps = num_epochs * steps_per_epoch
                # Use direct step counts from config (clamped to valid range)
                logging_steps = max(1, min(total_steps, pretrain_config.logging_steps))
                eval_steps = max(1, min(total_steps, pretrain_config.eval_steps))
                save_steps = max(1, min(total_steps, pretrain_config.save_steps))
        else:
            # Use ceiling division to ensure at least 1 step per epoch
            steps_per_epoch = (train_samples + global_batch_size_samples - 1) // global_batch_size_samples
            total_steps = num_epochs * steps_per_epoch
            # Use direct step counts from config (clamped to valid range)
            logging_steps = max(1, min(total_steps, pretrain_config.logging_steps))
            eval_steps = max(1, min(total_steps, pretrain_config.eval_steps))
            save_steps = max(1, min(total_steps, pretrain_config.save_steps))
    else:
        num_epochs = pretrain_config.num_epochs
        # Use ceiling division to ensure at least 1 step per epoch
        steps_per_epoch = (train_samples + global_batch_size_samples - 1) // global_batch_size_samples
        total_steps = num_epochs * steps_per_epoch
        # Use direct step counts from config (clamped to valid range)
        logging_steps = max(1, min(total_steps, pretrain_config.logging_steps))
        eval_steps = max(1, min(total_steps, pretrain_config.eval_steps))
        save_steps = max(1, min(total_steps, pretrain_config.save_steps))

    return run_name, wandb_run_name, output_dir, num_epochs, logging_steps, eval_steps, save_steps, resume_step


def run_pretraining(
    model: ProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:

    device = get_device()

    tokenizer = model.tokenizer
    model.to(device)

    # Validate dataset: manifest + shard files
    dataset_dir = Path(pretrain_config.dataset_dir)
    validation_result = validate_pretrain_dataset(dataset_dir)
    manifest = validation_result['manifest']

    # Get data from typed manifest
    train_shard_dir = dataset_dir / manifest.train_dir
    val_shard_dir = dataset_dir / manifest.val_dir
    train_sequences = manifest.train_sequences
    val_sequences = manifest.val_sequences

    # Load pre-tokenized binary shards
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

    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
    )

    create_dirs(pretrain_config.ckp_dir)

    effective_world_size = resolve_world_size(pretrain_config.multi_gpu, pretrain_config.world_size)

    batch = compute_batch_setup(pretrain_config, manifest.max_seq_len, effective_world_size)

    inferred_grad_accum_steps = batch.grad_accum_steps
    global_batch_size_samples = batch.global_batch_size_samples

    # Prepare run info and step intervals in a single place
    (
        run_name,
        wandb_run_name,
        output_dir,
        num_epochs,
        logging_steps,
        eval_steps,
        save_steps,
        resume_step,
    ) = prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=train_sequences,
        global_batch_size_samples=global_batch_size_samples,
    )

    steps_per_epoch = (train_sequences + global_batch_size_samples - 1) // global_batch_size_samples
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(pretrain_config.warmup_steps, total_steps)

    # Configure Weights & Biases via environment variables so HF Trainer attaches correctly
    os.environ["WANDB_PROJECT"] = pretrain_config.project_name
    os.environ["WANDB_NAME"] = wandb_run_name

    num_workers = get_num_workers(pretrain_config.num_workers, effective_world_size)

    training_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": pretrain_config.micro_batch_size,
        "per_device_eval_batch_size": pretrain_config.micro_batch_size,
        "gradient_accumulation_steps": inferred_grad_accum_steps,
        "num_train_epochs": num_epochs,
        "adam_learning_rate": pretrain_config.adam_learning_rate,
        "adam_weight_decay": pretrain_config.adam_weight_decay,
        "max_grad_norm": pretrain_config.max_grad_norm,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "logging_dir": Path(output_dir) / "logs",
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "seed": pretrain_config.seed,
        "bf16": pretrain_config.bf16 and device == "cuda" and torch.cuda.is_bf16_supported(),
        "fp16": pretrain_config.bf16 and ((device == "cuda" and not torch.cuda.is_bf16_supported()) or device == "mps"),
        "tf32": pretrain_config.tf32 and device == "cuda",
        "report_to": "wandb",
        "run_name": wandb_run_name,
        "dataloader_pin_memory": True if device == "cuda" else False,
        "dataloader_num_workers": num_workers,
        "dataloader_persistent_workers": False,
    }

    if num_workers > 0:
        training_dict["dataloader_prefetch_factor"] = pretrain_config.prefetch_factor
        training_dict["dataloader_persistent_workers"] = True

    # Build optimizer and scheduler (warmup_steps + min_lr) for all optimizer types
    optimizer_name = pretrain_config.optimizer.lower()
    raw_model = unwrap_model(model)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=pretrain_config.learning_rate,
            weight_decay=pretrain_config.weight_decay,
            betas=(pretrain_config.adam_beta1, pretrain_config.adam_beta2),
            eps=pretrain_config.adam_epsilon,
        )
    elif optimizer_name == "stable_adamw":
        stable_cls = getattr(torch.optim, "StableAdamW", None)
        if stable_cls is None:
            logger.warning("StableAdamW unavailable; falling back to AdamW.")
            stable_cls = torch.optim.AdamW
        optimizer = stable_cls(
            raw_model.parameters(),
            lr=pretrain_config.learning_rate,
            weight_decay=pretrain_config.weight_decay,
            betas=(pretrain_config.adam_beta1, pretrain_config.adam_beta2),
            eps=pretrain_config.adam_epsilon,
        )
    elif optimizer_name in {"muon", "normuon"}:
      # TODO: optimzer configs have been moved to the optim.py
        optimizer = build_muon_optimizer(model, pretrain_config)
    else:
        raise ValueError(
            f"Invalid optimizer: {pretrain_config.optimizer}. "
            f"Currently supported: [adamw, stable_adamw, muon, normuon]"
        )

    scheduler = _create_scheduler(
        optimizer, warmup_steps, total_steps,
        pretrain_config.learning_rate, pretrain_config.lr_decay_to_fraction,
        pretrain_config.lr_schedule,
    )

    if pretrain_config.multi_gpu:
        training_dict["ddp_backend"] = "nccl" if torch.cuda.is_available() else "gloo"
        training_dict["ddp_find_unused_parameters"] = True

    args = TrainingArguments(**training_dict)

    trainer = TokenTrackingTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
        callbacks=[WandbSourceSnapshotCallback()],
    )

    logger.info("Starting Trainer")

    # Start training and capture W&B run ID immediately after trainer initialization
    try:
        if resume_config:
            logger.info(
                f"Resuming training from checkpoint: {resume_config.checkpoint_dir}"
            )
            trainer.train(resume_from_checkpoint=resume_config.checkpoint_dir)
        else:
            trainer.train()

        # Add W&B metadata for resume tracking
        if resume_config and resume_config.is_resume and wandb.run is not None:
            try:
                if resume_step is not None:
                    wandb.config.update(
                        {
                            "resumed_from_step": resume_step,
                            "resume_timestamp": datetime.now().isoformat(),
                        },
                        allow_val_change=True,
                    )
                    # Add tag to mark this as a resumed run
                    current_tags = list(wandb.run.tags) if wandb.run.tags else []
                    if f"resumed-from-{resume_step}" not in current_tags:
                        wandb.run.tags = current_tags + [f"resumed-from-{resume_step}"]
                    logger.info(
                        f"Added W&B metadata: resumed from step {resume_step}"
                    )
            except Exception as e:
                logger.warning(f"Failed to add W&B resume metadata: {e}")

        # Capture and save W&B run ID for future resumes (if W&B is active)
        if wandb.run is not None:
            actual_run_id = wandb.run.id
            run_id_path = Path(output_dir) / "wandb_run_id.txt"
            if (
                not run_id_path.exists()
                or run_id_path.read_text().strip() != actual_run_id
            ):
                run_id_path.write_text(actual_run_id, encoding="utf-8")
                logger.info(f"Saved W&B run ID: {actual_run_id}")
    except Exception as e:
        logger.warning(f"Error during training or saving W&B run ID: {e}")
        raise

    logger.info("Saving final model and tokenizer")
    trainer.save_model(output_dir)
    trainer.save_state()
