"""Shared utilities for pretraining pipelines (HF Trainer and pure-torch)."""

from __future__ import annotations

import math
import os
import shutil
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

from nanoplm.utils.common import create_dirs
from nanoplm.utils.logger import logger

if TYPE_CHECKING:
    from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig


@dataclass
class BatchSetup:
    """Derived batch parameters inferred from PretrainingConfig at runtime."""
    grad_accum_steps: int
    global_batch_size_samples: int
    achieved_global_batch_tokens: int


def compute_batch_setup(
    cfg: PretrainingConfig,
    max_seq_len: int,
    effective_world_size: int,
) -> BatchSetup:
    """Infer grad_accum and effective batch sizes from global_batch_size."""
    if cfg.global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be > 0, got {cfg.global_batch_size}")

    world_tokens_per_micro_step = (
        cfg.micro_batch_size * max_seq_len * effective_world_size
    )
    if world_tokens_per_micro_step <= 0:
        raise ValueError(
            f"Invalid token throughput per micro-step: {world_tokens_per_micro_step}. "
            "Check micro_batch_size, max_seq_len, and world_size."
        )

    grad_accum = max(1, math.ceil(cfg.global_batch_size / world_tokens_per_micro_step))
    achieved_tokens = grad_accum * world_tokens_per_micro_step
    batch_samples = grad_accum * cfg.micro_batch_size * effective_world_size

    logger.info(
        f"Batch setup: "
        f"target={cfg.global_batch_size:,} tok, "
        f"micro_step={world_tokens_per_micro_step:,} tok, "
        f"grad_accum={grad_accum}, "
        f"achieved={achieved_tokens:,} tok"
    )

    return BatchSetup(
        grad_accum_steps=grad_accum,
        global_batch_size_samples=batch_samples,
        achieved_global_batch_tokens=achieved_tokens,
    )


def get_num_workers(user_value: Union[int, str], world_size: int) -> int:

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


def prepare_run_and_steps(
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig],
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

        if resume_config.extra_epochs:
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
