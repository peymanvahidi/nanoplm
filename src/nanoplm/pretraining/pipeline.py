import os
import json
import time
import shutil
import torch
import torch.distributed as dist
import wandb
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from pathlib import Path

from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
)

from nanoplm.pretraining.models.modern_bert import (
    ProtModernBertMLM,
    ProtModernBertTokenizer,
)
from nanoplm.pretraining.dataset import LoadShardedFastaMLMDataset, get_pretraining_worker_init_fn
from nanoplm.pretraining.collator import ProtDataCollatorForLM
from nanoplm.data.manifest import read_manifest, validate_manifest_for_pipeline
from nanoplm.data.validation import validate_pretrain_dataset, ValidationError
from nanoplm.utils.logger import logger
from nanoplm.utils.common import get_device, create_dirs


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
        logs["tokens_per_sec"] = self._last_tokens_per_sec
        logs["raw_tokens_per_sec"] = self._last_raw_tokens_per_sec
        super().log(logs, start_time=start_time, **kwargs)


@dataclass
class PretrainingConfig:
    # Dataset directory (contains .data_manifest from nanoplm data from-yaml)
    dataset_dir: Union[str, Path]

    # Checkpoint and output
    ckp_dir: str = "output/pretraining"

    # Dataset config (can be overridden by manifest)
    load_all_in_memory: bool = True

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1

    # Mixed precision
    bf16: bool = True
    tf32: bool = True

    # MLM settings
    mlm_probability: float = 0.3
    mask_replace_prob: float = 0.8
    random_token_prob: float = 0.1
    keep_probability: float = 0.1

    # Logging/checkpointing
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    seed: int = 42

    # Data loading
    num_workers: Union[int, str] = "auto"
    prefetch_factor: int = 2

    # Distributed training
    multi_gpu: bool = False
    world_size: Union[int, str] = 1
    project_name: str = "nanoplm-pretraining"


@dataclass
class ResumeConfig:
    is_resume: bool
    checkpoint_dir: str
    extra_epochs: Optional[int] = None


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
    global_batch_size: int,
) -> Tuple[str, str, str, int, int, int, int, Optional[int]]:
    """Prepare run naming/dirs and compute epochs & step intervals.

    Args:
        pretrain_config: Pretraining configuration
        resume_config: Resume configuration (if resuming)
        train_samples: Number of training samples (from manifest)
        global_batch_size: Global batch size for training

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

        if resume_config.extra_epochs > 0:
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
                steps_per_epoch = (train_samples + global_batch_size - 1) // global_batch_size
                total_steps = num_epochs * steps_per_epoch
                # Use direct step counts from config (clamped to valid range)
                logging_steps = max(1, min(total_steps, pretrain_config.logging_steps))
                eval_steps = max(1, min(total_steps, pretrain_config.eval_steps))
                save_steps = max(1, min(total_steps, pretrain_config.save_steps))
        else:
            # Use ceiling division to ensure at least 1 step per epoch
            steps_per_epoch = (train_samples + global_batch_size - 1) // global_batch_size
            total_steps = num_epochs * steps_per_epoch
            # Use direct step counts from config (clamped to valid range)
            logging_steps = max(1, min(total_steps, pretrain_config.logging_steps))
            eval_steps = max(1, min(total_steps, pretrain_config.eval_steps))
            save_steps = max(1, min(total_steps, pretrain_config.save_steps))
    else:
        num_epochs = pretrain_config.num_epochs
        # Use ceiling division to ensure at least 1 step per epoch
        steps_per_epoch = (train_samples + global_batch_size - 1) // global_batch_size
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

    # Read manifest and resolve paths
    dataset_dir = Path(pretrain_config.dataset_dir)

    # Validate dataset before starting training
    logger.info(f"Validating pretraining dataset at {dataset_dir}...")
    try:
        validation_result = validate_pretrain_dataset(dataset_dir)
        logger.info(f"Dataset validated: {validation_result['manifest']['train_sequences']:,} train, "
                    f"{validation_result['manifest']['val_sequences']:,} val sequences")
    except ValidationError as e:
        logger.error(f"Dataset validation failed: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        raise

    manifest = read_manifest(dataset_dir)
    validate_manifest_for_pipeline(
        manifest=manifest,
        expected_mode="pretrain"
    )

    # Get data from manifest
    max_length = manifest.max_seq_len
    train_hdf5_dir = dataset_dir / manifest.train_dir
    val_hdf5_dir = dataset_dir / manifest.val_dir
    train_sequences = manifest.train_sequences
    val_sequences = manifest.val_sequences

    logger.info(f"Loaded config from manifest: {dataset_dir}")
    logger.info(f"  train_hdf5: {train_hdf5_dir}")
    logger.info(f"  val_hdf5: {val_hdf5_dir}")
    logger.info(f"  max_length: {max_length}")
    logger.info(f"  train_sequences: {train_sequences}")
    logger.info(f"  val_sequences: {val_sequences}")

    # Validate paths exist
    if not train_hdf5_dir.exists():
        raise FileNotFoundError(f"Train HDF5 directory not found: {train_hdf5_dir}")
    if not val_hdf5_dir.exists():
        raise FileNotFoundError(f"Validation HDF5 directory not found: {val_hdf5_dir}")

    # Load pre-tokenized HDF5 shards
    logger.info("Using LoadShardedFastaMLMDataset for pre-tokenized HDF5 shards")

    try:
        train_ds = LoadShardedFastaMLMDataset(
            hdf5_dir=str(train_hdf5_dir),
            load_all_in_memory=pretrain_config.load_all_in_memory
        )
        val_ds = LoadShardedFastaMLMDataset(
            hdf5_dir=str(val_hdf5_dir),
            load_all_in_memory=pretrain_config.load_all_in_memory
        )
    except FileNotFoundError as e:
        logger.error(
            f"HDF5 shards not found! You need to create them first.\n"
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

    # Determine effective world size
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
    ) = _prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=train_sequences,
        global_batch_size=global_batch_size,
    )

    # Configure Weights & Biases via environment variables so HF Trainer attaches correctly
    os.environ["WANDB_PROJECT"] = pretrain_config.project_name
    os.environ["WANDB_NAME"] = wandb_run_name

    num_workers = _get_num_workers(pretrain_config.num_workers, effective_world_size)

    training_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": pretrain_config.batch_size,
        "per_device_eval_batch_size": pretrain_config.batch_size,
        "gradient_accumulation_steps": pretrain_config.gradient_accumulation_steps,
        "num_train_epochs": num_epochs,
        "learning_rate": pretrain_config.learning_rate,
        "weight_decay": pretrain_config.weight_decay,
        "warmup_ratio": pretrain_config.warmup_ratio,
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

    # Configure optimizer through TrainingArguments
    optimizer_name = pretrain_config.optimizer.lower()
    if optimizer_name == "adamw":
        training_dict["optim"] = "adamw_torch"
    elif optimizer_name == "stable_adamw":
        training_dict["optim"] = "stable_adamw"
    else:
        raise ValueError(
            f"Invalid optimizer: {pretrain_config.optimizer}. Currently supported: [adamw, stable_adamw]"
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

def _get_num_workers(user_value: Union[int, str], world_size: int) -> int:

    if isinstance(user_value, str) and user_value == "auto":
        cpu_cores = os.cpu_count() or 1

        # Leave some room for OS / other processes
        max_reasonable = max(1, cpu_cores - 2)

        # Heuristic: 4 workers per GPU is a good starting point
        workers_per_gpu = 4
        target = workers_per_gpu * max(1, world_size)

        workers = max(1, min(target, max_reasonable))   

        logger.info(f"Auto-setting num_workers to {workers} for {world_size} GPU(s).")

        return workers

    # Normalize string values to int if possible
    if isinstance(user_value, str):
        try:
            user_value = int(user_value)
        except ValueError:
            raise ValueError(
                f"Invalid num_workers value: {user_value}. Must be a non-negative integer or 'auto'"
            )

    # At this point we expect an int
    if isinstance(user_value, int) and user_value >= 0:
        return user_value
    else:
        raise ValueError(
            f"Invalid num_workers value: {user_value}. Must be a non-negative integer"
        )
