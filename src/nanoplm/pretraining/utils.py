"""Shared utilities for pretraining pipelines (HF Trainer and pure-torch)."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.utils.common import git_sha, is_global_main, job_stamp
from nanoplm.utils.logger import logger

if TYPE_CHECKING:
    from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig


@dataclass
class BatchSetup:
    """Derived batch parameters inferred from PretrainingConfig at runtime."""
    grad_accum_steps: int
    global_batch_seqs: int
    achieved_global_batch_tokens: int


def compute_batch_setup(
    cfg: PretrainingConfig,
    max_seq_len: int,
    effective_world_size: int,
) -> BatchSetup:
    """Infer grad_accum and effective batch sizes from global_batch_tokens."""
    if cfg.global_batch_tokens <= 0:
        raise ValueError(f"global_batch_tokens must be > 0, got {cfg.global_batch_tokens}")

    world_tokens_per_micro_step = (
        cfg.micro_batch_seqs * max_seq_len * effective_world_size
    )
    if world_tokens_per_micro_step <= 0:
        raise ValueError(
            f"Invalid token throughput per micro-step: {world_tokens_per_micro_step}. "
            "Check micro_batch_seqs, max_seq_len, and world_size."
        )

    grad_accum = max(1, math.ceil(cfg.global_batch_tokens / world_tokens_per_micro_step))
    achieved_tokens = grad_accum * world_tokens_per_micro_step
    batch_samples = grad_accum * cfg.micro_batch_seqs * effective_world_size

    logger.info(
        f"Batch setup: "
        f"target={cfg.global_batch_tokens:,} tok, "
        f"micro_step={world_tokens_per_micro_step:,} tok, "
        f"grad_accum={grad_accum}, "
        f"achieved={achieved_tokens:,} tok"
    )

    return BatchSetup(
        grad_accum_steps=grad_accum,
        global_batch_seqs=batch_samples,
        achieved_global_batch_tokens=achieved_tokens,
    )


def tokens_to_steps(target_tokens: int, achieved_global_batch_tokens: int) -> int:
    """Optimizer steps needed to process `target_tokens`.

    Ceil so the budget is met (overshoot < one optimizer step of tokens). The
    same value drives both the loop stop condition and the LR-schedule horizon.
    """
    if target_tokens <= 0:
        raise ValueError(f"target token budget must be > 0, got {target_tokens}")
    if achieved_global_batch_tokens <= 0:
        raise ValueError(
            f"achieved_global_batch_tokens must be > 0, got {achieved_global_batch_tokens}"
        )
    return max(1, math.ceil(target_tokens / achieved_global_batch_tokens))


@dataclass
class StepHorizon:
    """Resolved optimizer-step horizon: total_steps, plus stable_steps/decay_steps for wsd."""
    total_steps: int
    stable_steps: Optional[int] = None   # wsd only
    decay_steps: Optional[int] = None    # wsd only


def resolve_total_steps(cfg, achieved_global_batch_tokens: int) -> StepHorizon:
    """Map the configured token/step budget to optimizer steps (spec sections 4-5)."""
    sched = (cfg.lr_schedule or "").lower()

    def _steps(token_field, step_field):
        stp = getattr(cfg, step_field)
        if stp is not None:
            return max(1, int(stp))
        return tokens_to_steps(getattr(cfg, token_field), achieved_global_batch_tokens)

    if sched == "wsd":
        S = _steps("stable_tokens", "stable_steps")
        Dd = _steps("decay_tokens", "decay_steps")
        return StepHorizon(total_steps=S + Dd, stable_steps=S, decay_steps=Dd)
    total = _steps("train_tokens", "max_steps")
    return StepHorizon(total_steps=total)


def format_passes_hint(budget_tokens: int, total_train_tokens: int,
                       approx_total: Optional[int] = None) -> str:
    """One-line 'passes over the train set' hint for a token budget."""
    denom = total_train_tokens if total_train_tokens and total_train_tokens > 0 else approx_total
    if not denom:
        return f"Budget: {budget_tokens:,} tokens (dataset token count unknown)"
    passes = budget_tokens / denom
    tag = "" if (total_train_tokens and total_train_tokens > 0) else " (approx)"
    return (f"Budget: {budget_tokens:,} tokens → ≈{passes:.1f} passes over the "
            f"train set (each sample seen ~{passes:.1f}×){tag}")


def get_num_workers(user_value: Union[int, str]) -> int:
    """Resolve ``num_workers``, expanding ``"auto"`` to 4 workers for this rank.

    DataLoader workers are spawned per process, so the count must never be
    derived from a rank count. Every rank sharing a node pays it separately, and
    scaling by the global world size put 256 worker processes on a 4-node host
    and OOM-killed job 1702372. The only ceiling that belongs here is this
    task's own CPU budget, which ``sched_getaffinity`` reports and ``cpu_count``
    does not, since ``cpu_count`` sees the whole node through a SLURM binding.
    """
    if isinstance(user_value, str) and user_value == "auto":
        get_affinity = getattr(os, "sched_getaffinity", None)
        task_cpus = len(get_affinity(0)) if get_affinity else (os.cpu_count() or 1)

        workers = max(1, min(4, task_cpus - 1))

        logger.info(
            f"Auto-setting num_workers to {workers} per rank "
            f"(task CPU budget {task_cpus}, node CPUs {os.cpu_count()})."
        )

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


def checkpoint_step(checkpoint_dir: Union[str, Path]) -> int:
    """Resolve the optimizer step for a ``checkpoint-*`` directory.

    Numbered dirs (``checkpoint-<N>``) parse the step from the name. Role-named
    dirs (``checkpoint-final``, ``checkpoint-stable-end``) have no numeric
    suffix, so the step is read from ``training_state.json["global_step"]``
    (``checkpoint-final`` carries ``total_steps``, ``checkpoint-stable-end``
    carries ``stable_steps``). Returns ``-1`` for non-checkpoint names or when
    the role dir's step cannot be determined.
    """
    path = Path(checkpoint_dir)
    name = path.name
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.split("-", 1)[1])
    except ValueError:
        pass

    state_path = path / "training_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state["global_step"])
    except (OSError, ValueError, KeyError, TypeError):
        return -1


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
        if not ckpt_dir.is_dir():
            continue
        step = checkpoint_step(ckpt_dir)
        # step == -1 (unresolvable) must NOT be treated as future work.
        if step > resume_step:
            checkpoints_to_archive.append((step, ckpt_dir))

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


def _read_checkpoint_epoch(checkpoint_dir: Union[str, Path]) -> int:
    """Read the current epoch from a checkpoint directory.

    Tries the pure-torch format (`training_state.json`) first, then falls
    back to the HuggingFace Trainer format (`trainer_state.json`).  HF stores
    `epoch` as a float (e.g. 4.5 mid-epoch); we truncate to the number of
    completed full epochs.

    Raises FileNotFoundError if neither file exists.
    """
    ckp = Path(checkpoint_dir)

    pure_state = ckp / "training_state.json"
    if pure_state.exists():
        with open(pure_state, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        return int(state.get("epoch", 0))

    hf_state = ckp / "trainer_state.json"
    if hf_state.exists():
        with open(hf_state, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        return int(state.get("epoch", 0))

    raise FileNotFoundError(
        f"Could not determine checkpoint epoch: neither "
        f"'{pure_state}' nor '{hf_state}' exists. Cannot honor resume_epochs > 0."
    )


@dataclass(frozen=True)
class RunIdentity:
    """Which run this process belongs to, and the cadence it runs at.

    ``wandb_run_name`` names a W&B session, which only the global main opens, so
    on every other rank it is just ``run_name``. Likewise ``resume_step`` is the
    step the global main archived forward from, and is None elsewhere.
    """
    run_name: str
    run_root: Path
    wandb_run_name: str
    is_resume: bool
    resume_step: Optional[int]
    logging_steps: int
    eval_steps: int
    save_steps: int


def _write_run_manifest(
    run: RunIdentity,
    pretrain_config: PretrainingConfig,
    effective_world_size: int,
) -> None:
    """Record in the run root what a checkpoint cannot tell you afterwards.

    A checkpoint's ``pretrain_config.yaml`` stores the config as written, so
    ``world_size: auto`` stays "auto" and the directories stay CWD-relative, and
    nothing anywhere records which commit ran or what was typed to start it.
    """
    manifest = {
        "run_name": run.run_name,
        "wandb_run_name": run.wandb_run_name,
        "is_resume": run.is_resume,
        "resume_step": run.resume_step,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "world_size": effective_world_size,
        "git_sha": git_sha(Path(__file__).resolve().parent),
        "argv": sys.argv,
        "dataset_dir": str(Path(pretrain_config.dataset_dir).resolve()),
        "ckp_dir": str(Path(pretrain_config.ckp_dir).resolve()),
    }
    try:
        with atomic_write_path(run.run_root / "run.json") as tmp:
            tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not write run.json into {run.run_root}: {exc}")


def prepare_run_and_steps(
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig],
    effective_world_size: int,
) -> RunIdentity:
    """Prepare run naming/dirs and resolve the raw logging/eval/save intervals.

    The training horizon (total_steps) is not derived here: it comes from the
    token budget and is computed by the caller via ``resolve_total_steps``. The
    returned intervals are therefore the *raw* configured ones (un-clamped); the
    caller clamps them to the token-budget total_steps once the horizon is known.

    Args:
        pretrain_config: Pretraining configuration
        resume_config: Resume configuration (if resuming)
        effective_world_size: Ranks this job actually runs on, already resolved
            from the config's ``world_size`` (which may be the string "auto")
    """
    ckp_root = Path(pretrain_config.ckp_dir)

    is_main = is_global_main()
    is_resume = bool(resume_config and resume_config.is_resume)

    if is_resume:
        checkpoint_path = Path(resume_config.checkpoint_dir)
        run_name = checkpoint_path.parent.name
        run_root = ckp_root / run_name
        wandb_run_name = run_name
        resume_step = None

        # run_root is derived from the checkpoint's name, not from where the
        # checkpoint lives, so it does not exist yet when resuming into a fresh
        # ckp_dir -- which is what you do to keep the source run read-only.
        if is_main:
            run_root.mkdir(parents=True, exist_ok=True)
            counter_file = run_root / ".resume_counter"
            try:
                resume_counter = int(counter_file.read_text().strip()) + 1
            except (OSError, ValueError):
                resume_counter = 1
            counter_file.write_text(str(resume_counter), encoding="utf-8")
            wandb_run_name = f"{run_name}-re{resume_counter}"
            logger.info(f"Resume session #{resume_counter}: W&B run name = {wandb_run_name}")

            # checkpoint_step resolves a role-named dir (e.g.
            # checkpoint-stable-end) from training_state.json rather than
            # failing to parse its name.
            step = checkpoint_step(checkpoint_path)
            if step >= 0:
                resume_step = step
                _archive_future_checkpoints(run_root, step)
            else:
                logger.warning(
                    f"Could not determine step for checkpoint path: {checkpoint_path.name}. "
                    f"Skipping future checkpoint archival."
                )
    else:
        run_name = f"{pretrain_config.run_name}-{job_stamp()}"
        run_root = ckp_root / run_name
        wandb_run_name = run_name
        resume_step = None

    if is_main:
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            (run_root / "run_name.txt").write_text(run_name, encoding="utf-8")
        except OSError as exc:
            logger.warning(f"Could not write run_name.txt into {run_root}: {exc}")

    logging_steps = max(1, pretrain_config.logging_steps)
    eval_steps = max(1, pretrain_config.eval_steps)
    save_steps = max(1, pretrain_config.save_steps)

    # A resumed run keeps the cadence recorded in the checkpoint, so logging
    # and saving stay aligned across resume sessions.
    if is_resume:
        training_args_path = Path(resume_config.checkpoint_dir) / "training_args.bin"
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
                pass

    run = RunIdentity(
        run_name=run_name,
        run_root=run_root,
        wandb_run_name=wandb_run_name,
        is_resume=is_resume,
        resume_step=resume_step,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
    )
    if is_main:
        _write_run_manifest(run, pretrain_config, effective_world_size)
    return run
