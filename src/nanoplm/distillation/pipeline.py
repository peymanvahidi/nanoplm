import os
import json
import shutil
import time
import torch
import wandb
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from safetensors.torch import save_file, load_file
from transformers import (
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from nanoplm.distillation.collator import DistillDataCollator
from nanoplm.distillation.trainer import DistillationTrainer
from nanoplm.distillation.models.student import ProtX, ProtXConfig, ProtXTokenizer
from nanoplm.distillation.models.teacher import ProtT5
from nanoplm.distillation.dataset import (
    KDDatasetOnTheFly,
    LoadKDDataset,
    LoadKDDatasetOptimized,
)
from nanoplm.data.manifest import read_manifest, validate_manifest_for_pipeline, get_dataset_paths
from nanoplm.utils import get_device, logger, create_dirs


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""

    # Dataset directory (contains .data_manifest)
    # When provided, reads dataset paths and params from manifest
    dataset_dir: Optional[Union[str, Path]] = None

    # Dataset paths (used when on_the_fly=True or when not using manifest)
    train_fasta: Optional[Union[str, Path]] = None
    val_fasta: Optional[Union[str, Path]] = None
    train_h5_prefix: Optional[Union[str, Path]] = None
    val_h5_prefix: Optional[Union[str, Path]] = None

    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05

    # LR scheduler
    lr_scheduler: str = "cosine"
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None

    # Dataset config (can be overridden by manifest)
    max_seq_len: Optional[int] = None
    max_seqs_num: Optional[int] = None
    val_ratio: Optional[float] = None
    on_the_fly: bool = False
    sharded: Optional[bool] = None
    train_sequences: Optional[int] = None  # Set from manifest
    val_sequences: Optional[int] = None  # Set from manifest

    # Data loader optimization
    use_optimized_loader: bool = True
    max_open_files: int = 5
    chunk_size: int = 32
    prefetch_batches: int = 2
    use_threading: bool = True
    num_workers: int = 8  # Increased from 4 for better throughput

    # Checkpointing
    ckp_dir: str = "output/distillation"
    project_name: str = "nanoplm-distillation"
    logging_steps_percentage: float = 0.01
    eval_steps_percentage: float = 0.01
    save_steps_percentage: float = 0.05

    # Distributed
    multi_gpu: bool = False
    world_size: Union[int, str] = 1
    seed: int = 42

    use_native: bool = False

    def __post_init__(self):
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}


@dataclass
class ResumeConfig:
    """Configuration for resuming training from a checkpoint."""

    is_resume: bool = False
    checkpoint_dir: str = ""
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
    distill_config: DistillationConfig,
    resume_config: Optional[ResumeConfig],
    train_samples: int,
    global_batch_size: int,
) -> Tuple[str, str, str, int, int, int, int, Optional[int]]:
    """Prepare run naming/dirs and compute epochs & step intervals.

    Args:
        distill_config: Distillation configuration
        resume_config: Resume configuration (if resuming)
        train_samples: Number of training samples (from manifest)
        global_batch_size: Global batch size for training

    Returns a tuple: (run_name, wandb_run_name, output_dir, num_epochs,
                      logging_steps, eval_steps, save_steps, resume_step)
    """
    ckp_root = Path(distill_config.ckp_dir)

    # Determine run directory and name
    if resume_config and resume_config.is_resume:
        # Resume in SAME directory
        checkpoint_path = Path(resume_config.checkpoint_dir)
        run_root = checkpoint_path.parent  # Use original directory

        # Read original run name from metadata
        run_name_file = run_root / "run_name.txt"
        if run_name_file.exists():
            run_name = run_name_file.read_text().strip()
        else:
            # Fallback to directory name if file missing
            run_name = run_root.name
            logger.warning(f"run_name.txt not found, using directory name: {run_name}")

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
        # New run: create unique directory with timestamp
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

    # Persist run metadata for future resumes (only for new runs)
    if not (resume_config and resume_config.is_resume):
        try:
            (Path(output_dir) / "run_name.txt").write_text(run_name, encoding="utf-8")
        except Exception:
            pass

    # Compute epochs and step intervals
    num_epochs = distill_config.num_epochs
    if resume_config and resume_config.is_resume and resume_config.extra_epochs:
        num_epochs = distill_config.num_epochs + int(resume_config.extra_epochs)

    # Calculate total steps using train_samples from manifest
    # Use ceiling division to ensure at least 1 step per epoch
    steps_per_epoch = (train_samples + global_batch_size - 1) // global_batch_size
    total_steps = steps_per_epoch * num_epochs

    logging_steps = max(1, int(total_steps * distill_config.logging_steps_percentage))
    eval_steps = max(1, int(total_steps * distill_config.eval_steps_percentage))
    save_steps = max(1, int(total_steps * distill_config.save_steps_percentage))

    return run_name, wandb_run_name, output_dir, num_epochs, logging_steps, eval_steps, save_steps, resume_step


def _resolve_config_from_manifest(distill_config: DistillationConfig) -> DistillationConfig:
    """Resolve config values from manifest if dataset_dir is provided.

    Args:
        distill_config: The distillation configuration

    Returns:
        Updated configuration with manifest values filled in
    """
    if distill_config.dataset_dir is None:
        return distill_config

    dataset_dir = Path(distill_config.dataset_dir)
    manifest = read_manifest(dataset_dir)
    validate_manifest_for_pipeline(manifest, "distillation")

    # Check if manifest specifies on_the_fly mode
    manifest_on_the_fly = manifest.on_the_fly if manifest.on_the_fly is not None else False

    # Update config with manifest's on_the_fly flag
    distill_config.on_the_fly = manifest_on_the_fly

    # Update common config values from manifest
    if not distill_config.max_seq_len:
        distill_config.max_seq_len = manifest.max_seq_len
    if not distill_config.max_seqs_num:
        distill_config.max_seqs_num = manifest.train_sequences + manifest.val_sequences
    if distill_config.val_ratio is None:
        distill_config.val_ratio = manifest.val_ratio

    # Always update sequence counts from manifest (authoritative source)
    distill_config.train_sequences = manifest.train_sequences
    distill_config.val_sequences = manifest.val_sequences

    if manifest_on_the_fly:
        # On-the-fly mode: Use FASTA paths from manifest
        if manifest.train_fasta and distill_config.train_fasta is None:
            distill_config.train_fasta = manifest.train_fasta
        if manifest.val_fasta and distill_config.val_fasta is None:
            distill_config.val_fasta = manifest.val_fasta

        logger.info(f"Loaded config from manifest (on-the-fly mode): {dataset_dir}")
        logger.info(f"  train_fasta: {distill_config.train_fasta}")
        logger.info(f"  val_fasta: {distill_config.val_fasta}")
        logger.info(f"  max_seq_len: {distill_config.max_seq_len}")
        logger.info(f"  max_seqs_num: {distill_config.max_seqs_num}")
    else:
        # Pre-computed mode: Use H5 paths from manifest
        paths = get_dataset_paths(dataset_dir, manifest)

        if distill_config.train_h5_prefix is None:
            distill_config.train_h5_prefix = str(paths.get("train_h5_prefix", paths["train_dir"]))
        if distill_config.val_h5_prefix is None:
            distill_config.val_h5_prefix = str(paths.get("val_h5_prefix", paths["val_dir"]))
        if distill_config.sharded is None:
            distill_config.sharded = manifest.sharded

        logger.info(f"Loaded config from manifest (pre-computed mode): {dataset_dir}")
        logger.info(f"  train_h5_prefix: {distill_config.train_h5_prefix}")
        logger.info(f"  val_h5_prefix: {distill_config.val_h5_prefix}")
        logger.info(f"  max_seq_len: {distill_config.max_seq_len}")
        logger.info(f"  max_seqs_num: {distill_config.max_seqs_num}")
        logger.info(f"  sharded: {distill_config.sharded}")

    return distill_config


def _load_datasets(
    distill_config: DistillationConfig,
    teacher=None,
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets."""

    if distill_config.on_the_fly:
        if not distill_config.train_fasta or not distill_config.val_fasta:
            raise ValueError(
                "train_fasta and val_fasta are required when on_the_fly=True"
            )
        train_dataset = KDDatasetOnTheFly(
            input_fasta=distill_config.train_fasta,
            teacher=teacher,
            max_seq_len=distill_config.max_seq_len,
            device=get_device(),
            read_batch_size=32,  # Batched tokenization for better performance
        )
        val_dataset = KDDatasetOnTheFly(
            input_fasta=distill_config.val_fasta,
            teacher=teacher,
            max_seq_len=distill_config.max_seq_len,
            device=get_device(),
            read_batch_size=32,
        )
    else:
        if not distill_config.train_h5_prefix or not distill_config.val_h5_prefix:
            raise ValueError(
                "train_h5_prefix and val_h5_prefix are required when on_the_fly=False. "
                "Either provide dataset_dir (to read from manifest) or specify paths directly."
            )

        effective_seed = seed if seed is not None else int(time.time())

        if distill_config.use_optimized_loader:
            logger.info("Using LoadKDDatasetOptimized for better performance")
            train_dataset = LoadKDDatasetOptimized(
                h5_path=distill_config.train_h5_prefix,
                device=get_device(),
                seed=effective_seed,
                sharded=distill_config.sharded,
                max_open_files=distill_config.max_open_files,
                chunk_size=distill_config.chunk_size,
                prefetch_batches=distill_config.prefetch_batches,
                use_threading=distill_config.use_threading,
            )
            val_dataset = LoadKDDatasetOptimized(
                h5_path=distill_config.val_h5_prefix,
                device=get_device(),
                seed=effective_seed + 1,
                sharded=distill_config.sharded,
                max_open_files=distill_config.max_open_files,
                chunk_size=distill_config.chunk_size,
                prefetch_batches=distill_config.prefetch_batches,
                use_threading=distill_config.use_threading,
            )
        else:
            logger.info("Using standard LoadKDDataset")
            train_dataset = LoadKDDataset(
                h5_path=distill_config.train_h5_prefix,
                device=get_device(),
                seed=effective_seed,
                sharded=distill_config.sharded,
            )
            val_dataset = LoadKDDataset(
                h5_path=distill_config.val_h5_prefix,
                device=get_device(),
                seed=effective_seed + 1,
                sharded=distill_config.sharded,
            )

    return train_dataset, val_dataset


def _get_scheduler(
    optimizer,
    lr_scheduler: str,
    lr_scheduler_kwargs: Dict[str, Any],
    num_training_steps: int,
):
    """Create the learning rate scheduler."""

    if num_training_steps <= 0:
        logger.warning(
            "Number of training steps is 0 or less. No scheduler will be created."
        )
        return None

    logger.info(
        f"Creating {lr_scheduler} scheduler with {num_training_steps} training steps"
    )

    # Set warmup steps
    warmup_steps = lr_scheduler_kwargs.get(
        "num_warmup_steps", int(num_training_steps * 0.05)
    )
    if lr_scheduler == "constant":
        warmup_steps = lr_scheduler_kwargs.get("num_warmup_steps", 0)

    if lr_scheduler == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )
    elif lr_scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif lr_scheduler == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif lr_scheduler == "constant":
        logger.info("Using constant learning rate scheduler.")
        return get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps
        )
    else:
        raise ValueError(f"Unknown learning rate scheduler: {lr_scheduler}")


def _save_training_config(output_dir: Path, config: Dict[str, Any]):
    """Save the training configuration to a file for future resuming."""
    config_file = output_dir / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Saved training configuration to {config_file}")


def run_distillation(
    model_config: ProtXConfig,
    distill_config: DistillationConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:
    """
    Run knowledge distillation training.

    Args:
        model_config: The student ProtX model to train
        distill_config: Configuration for distillation training
        resume_config: Optional configuration for resuming from checkpoint
    """

    # Resolve config from manifest if dataset_dir is provided
    distill_config = _resolve_config_from_manifest(distill_config)

    device = get_device()
    model = ProtX(model_config)
    model.to(device)

    # Get distributed training info
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    # Calculate effective world size
    if distill_config.multi_gpu:
        if distill_config.world_size == "auto":
            env_ws = os.environ.get("WORLD_SIZE")
            effective_world_size = (
                int(env_ws) if env_ws else max(torch.cuda.device_count(), 1)
            )
        else:
            effective_world_size = (
                int(distill_config.world_size) if distill_config.world_size else 1
            )
    else:
        effective_world_size = 1

    global_batch_size = (
        distill_config.gradient_accumulation_steps
        * distill_config.batch_size
        * effective_world_size
    )

    # Setup teacher model and tokenizers if on-the-fly mode
    teacher = None
    teacher_model_for_collator = None
    teacher_tokenizer = None
    student_tokenizer = None
    if distill_config.on_the_fly:
        teacher = ProtT5()
        teacher_model_for_collator = teacher.encoder_model
        teacher_tokenizer = teacher.tokenizer
        student_tokenizer = ProtXTokenizer()

    # Load datasets
    train_dataset, val_dataset = _load_datasets(
        distill_config=distill_config,
        teacher=teacher,
        seed=distill_config.seed,
    )

    # Prepare run info and step intervals
    is_resuming = resume_config and resume_config.is_resume

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
        distill_config=distill_config,
        resume_config=resume_config,
        train_samples=distill_config.train_sequences,
        global_batch_size=global_batch_size,
    )

    # Load model weights if resuming
    if is_resuming:
        checkpoint_path = Path(resume_config.checkpoint_dir)
        safetensors_path = checkpoint_path / "model.safetensors"

        model_loaded = False
        if safetensors_path.exists():
            logger.info(f"Loading model weights from {safetensors_path}")
            state_dict = load_file(safetensors_path, device=device)
            model.load_state_dict(state_dict)
            model_loaded = True
        if not model_loaded:
            logger.warning(
                f"Could not find model weights in {resume_config.checkpoint_dir}. Training from scratch."
            )

    # Save training configuration for future resumes
    if not is_resuming:
        training_config = {
            "train_fasta": str(distill_config.train_fasta),
            "val_fasta": str(distill_config.val_fasta),
            "train_h5_prefix": str(distill_config.train_h5_prefix),
            "val_h5_prefix": str(distill_config.val_h5_prefix),
            "num_epochs": distill_config.num_epochs,
            "batch_size": distill_config.batch_size,
            "learning_rate": distill_config.learning_rate,
            "gradient_accumulation_steps": distill_config.gradient_accumulation_steps,
            "lr_scheduler": distill_config.lr_scheduler,
            "lr_scheduler_kwargs": distill_config.lr_scheduler_kwargs,
            "max_seq_len": distill_config.max_seq_len,
            "max_seqs_num": distill_config.max_seqs_num,
            "val_ratio": distill_config.val_ratio,
            "on_the_fly": distill_config.on_the_fly,
            "sharded": distill_config.sharded,
            "project_name": distill_config.project_name,
        }
        _save_training_config(Path(output_dir), training_config)

    # Create data collator
    data_collator = DistillDataCollator(
        teacher_model=teacher_model_for_collator,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        on_the_fly=distill_config.on_the_fly,
        max_seq_len=distill_config.max_seq_len,
    )

    # Calculate num_training_steps for scheduler using train_sequences from manifest
    train_samples = distill_config.train_sequences
    # Use ceiling division to ensure at least 1 step per epoch
    steps_per_epoch = (train_samples + global_batch_size - 1) // global_batch_size
    num_training_steps = steps_per_epoch * num_epochs

    logger.info(f"Training configuration:")
    logger.info(f"  Multi-GPU: {distill_config.multi_gpu}")
    logger.info(f"  World size: {effective_world_size}")
    logger.info(f"  Per-device batch size: {distill_config.batch_size}")
    logger.info(
        f"  Gradient accumulation steps: {distill_config.gradient_accumulation_steps}"
    )
    logger.info(f"  Effective batch size: {global_batch_size}")
    logger.info(f"  Total training steps: {num_training_steps}")
    logger.info(f"  Training samples: {train_samples}")

    # Configure Weights & Biases via environment variables
    os.environ["WANDB_PROJECT"] = distill_config.project_name
    os.environ["WANDB_NAME"] = wandb_run_name
    os.environ["WANDB_LOG_MODEL"] = "end"

    # Determine if we're using CUDA (for DataLoader optimizations)
    is_cuda = torch.cuda.is_available() and device.startswith('cuda')

    # For on-the-fly mode, use 0 workers to avoid pickling issues with teacher model
    # For pre-computed embeddings, use configured workers
    effective_num_workers = 0 if distill_config.on_the_fly else max(4, distill_config.num_workers)

    # Build training arguments
    training_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "max_steps": int(num_training_steps),
        "per_device_train_batch_size": distill_config.batch_size,
        "per_device_eval_batch_size": distill_config.batch_size,
        "warmup_steps": int(num_training_steps * distill_config.warmup_ratio),
        "learning_rate": distill_config.learning_rate,
        "logging_dir": str(Path(output_dir) / "logs"),
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "report_to": "wandb",
        "run_name": wandb_run_name,
        # DataLoader optimizations for better throughput
        "dataloader_num_workers": effective_num_workers,
        "dataloader_pin_memory": is_cuda,  # Only on CUDA (MPS doesn't support)
        "dataloader_persistent_workers": True if effective_num_workers > 0 else False,
        "dataloader_prefetch_factor": 4 if effective_num_workers > 0 else None,
        "remove_unused_columns": False,
        "label_names": ["teacher_embeddings"],
        "gradient_accumulation_steps": distill_config.gradient_accumulation_steps,
        "seed": distill_config.seed,
    }

    if distill_config.multi_gpu:
        training_dict["ddp_backend"] = "nccl" if torch.cuda.is_available() else "gloo"
        training_dict["ddp_find_unused_parameters"] = True

    training_args = TrainingArguments(**training_dict)

    # Initialize W&B
    if is_main_process:
        wandb_config = {
            "project": distill_config.project_name,
            "name": wandb_run_name,
            "config": training_args.to_dict(),
            "settings": wandb.Settings(start_method="fork"),
        }

        # Always create a new W&B run (with counter-based name for resumes)
        wandb_config["id"] = wandb_run_name

        wandb.init(**wandb_config)

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=distill_config.learning_rate)

    # Create scheduler
    scheduler = _get_scheduler(
        optimizer=optimizer,
        lr_scheduler=distill_config.lr_scheduler,
        lr_scheduler_kwargs=distill_config.lr_scheduler_kwargs or {},
        num_training_steps=num_training_steps,
    )

    # Create trainer
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )

    logger.info(f"Starting training with Hugging Face Trainer. Output dir: {output_dir}")
    if is_main_process:
        wandb.config.update(training_args.to_dict())

    # Train
    try:
        if is_resuming:
            logger.info(
                f"Resuming training from checkpoint: {resume_config.checkpoint_dir}"
            )
            train_result = trainer.train(
                resume_from_checkpoint=resume_config.checkpoint_dir
            )
        else:
            train_result = trainer.train()

        # Add W&B metadata for resume tracking
        if is_resuming and resume_step is not None and wandb.run is not None:
            try:
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
                logger.info(f"Added W&B metadata: resumed from step {resume_step}")
            except Exception as e:
                logger.warning(f"Failed to add W&B resume metadata: {e}")

        # Save W&B run ID for future resumes
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

    # Save final model
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Evaluate
    if val_dataset:
        logger.info(f"Evaluating on {len(val_dataset)} samples")
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if is_main_process:
        wandb.finish()
        logger.info("Training complete!")


# ============================================================================
# Native PyTorch Training Pipeline (without HuggingFace Trainer)
# ============================================================================


def _initialize_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    # Initialize process group if not already initialized
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
        logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}, backend={backend}")

    return rank, world_size, local_rank


def _setup_device(local_rank: int):
    """Setup device for training."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device


def _wrap_model_ddp(model: torch.nn.Module, device: torch.device, local_rank: int, world_size: int):
    """Wrap model in DistributedDataParallel if needed."""
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
        logger.info(f"Wrapped model in DDP with device_ids=[{local_rank}]")

    return model


def _compute_distillation_loss(
    student_repr: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss between student and teacher embeddings."""
    mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
    diff = ((student_repr - teacher_embeddings) ** 2) * mask
    loss = diff.sum() / mask.sum().clamp(min=1)
    return loss


def _save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    rank: int,
):
    """Save checkpoint (only rank 0)."""
    if rank != 0:
        return

    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get actual model (unwrap DDP if needed)
    model_to_save = model.module if isinstance(model, DDP) else model

    # Save model weights using safetensors
    model_path = checkpoint_dir / "model.safetensors"
    save_file(model_to_save.state_dict(), model_path)

    # Save optimizer and scheduler states
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
    }, checkpoint_dir / "training_state.pt")

    logger.info(f"Saved checkpoint to {checkpoint_dir}")


def _load_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[int, int]:
    """Load checkpoint and return (epoch, step)."""
    # Load model weights
    model_path = checkpoint_dir / "model.safetensors"
    if model_path.exists():
        state_dict = load_file(model_path, device=str(device))
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {model_path}")

    # Load optimizer and scheduler states
    training_state_path = checkpoint_dir / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location=device)
        optimizer.load_state_dict(training_state["optimizer"])
        if scheduler and training_state.get("scheduler"):
            scheduler.load_state_dict(training_state["scheduler"])
        epoch = training_state.get("epoch", 0)
        step = training_state.get("step", 0)
        logger.info(f"Loaded training state: epoch={epoch}, step={step}")
        return epoch, step

    return 0, 0


def _train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    epoch: int,
    start_step: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    output_dir: Path,
    rank: int,
    world_size: int,
) -> int:
    """Train for one epoch. Returns the last global step."""
    model.train()
    total_loss = 0.0
    log_loss = 0.0
    global_step = start_step

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            training_mode=True,
        )
        student_repr = outputs.last_hidden_state

        # Compute loss
        teacher_embeddings = batch["teacher_embeddings"]
        loss = _compute_distillation_loss(
            student_repr=student_repr,
            teacher_embeddings=teacher_embeddings,
            attention_mask=batch["attention_mask"],
        )

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Accumulate losses for logging
        total_loss += loss.item()
        log_loss += loss.item()

        # Update weights after accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Logging (only rank 0)
            if rank == 0 and global_step % logging_steps == 0:
                # avg_loss = sum of scaled losses / logging_steps
                # Since scaled_loss = actual_loss / G, and we accumulate over L*G micro-batches:
                # log_loss = L*G * (avg_actual / G) = L * avg_actual
                # avg_loss = log_loss / L = avg_actual (correct!)
                avg_loss = log_loss / logging_steps
                lr = optimizer.param_groups[0]["lr"]

                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": lr,
                    "train/epoch": epoch,
                    "train/step": global_step,
                })

                logger.info(
                    f"Epoch {epoch} | Step {global_step} | Loss: {avg_loss * gradient_accumulation_steps:.4f} | LR: {lr:.2e}"
                )
                log_loss = 0.0

            # Evaluation (at eval_steps intervals)
            if global_step % eval_steps == 0 and val_dataloader is not None:
                logger.info(f"Running evaluation at step {global_step}")
                eval_metrics = _evaluate(
                    model=model,
                    dataloader=val_dataloader,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                )

                if rank == 0:
                    wandb.log({
                        "eval/loss": eval_metrics["eval_loss"],
                        "eval/epoch": epoch,
                        "eval/step": global_step,
                    })
                    logger.info(f"Eval loss: {eval_metrics['eval_loss']:.4f}")

                # Return to training mode
                model.train()

            # Save checkpoint (only rank 0)
            if global_step % save_steps == 0:
                _save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    rank=rank,
                )

    return global_step


def _evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    """Evaluate model and aggregate metrics across ranks."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                training_mode=True,
            )
            student_repr = outputs.last_hidden_state

            # Compute loss
            teacher_embeddings = batch["teacher_embeddings"]
            loss = _compute_distillation_loss(
                student_repr=student_repr,
                teacher_embeddings=teacher_embeddings,
                attention_mask=batch["attention_mask"],
            )

            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    # Aggregate across all ranks if distributed
    if world_size > 1:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)

        torch.distributed.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)

        total_loss = total_loss_tensor.item()
        total_samples = total_samples_tensor.item()

    avg_loss = total_loss / max(total_samples, 1)

    return {"eval_loss": avg_loss}


def run_distillation_native(
    model_config: ProtXConfig,
    distill_config: DistillationConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:
    """
    Run knowledge distillation training with native PyTorch (no HuggingFace Trainer).

    This implementation provides granular control over:
    - Distributed Data Parallel (DDP) setup
    - Gradient accumulation and clipping
    - Custom loss computation
    - Checkpointing and resuming
    - Logging (W&B)

    Args:
        model_config: Configuration for the student ProtX model
        distill_config: Configuration for distillation training
        resume_config: Optional configuration for resuming from checkpoint
    """

    # Resolve config from manifest if dataset_dir is provided
    distill_config = _resolve_config_from_manifest(distill_config)

    # ========================================================================
    # 1. Initialize distributed training
    # ========================================================================
    rank, world_size, local_rank = _initialize_distributed()
    is_main_process = rank == 0

    # ========================================================================
    # 2. Setup device
    # ========================================================================
    device = _setup_device(local_rank)

    # ========================================================================
    # 3. Load datasets
    # ========================================================================
    teacher = None
    teacher_model_for_collator = None
    teacher_tokenizer = None
    student_tokenizer = None
    if distill_config.on_the_fly:
        teacher = ProtT5()
        teacher_model_for_collator = teacher.encoder_model
        teacher_tokenizer = teacher.tokenizer
        student_tokenizer = ProtXTokenizer()

    train_dataset, val_dataset = _load_datasets(
        distill_config=distill_config,
        teacher=teacher,
        seed=distill_config.seed,
    )

    # Calculate effective batch size and steps
    global_batch_size = (
        distill_config.gradient_accumulation_steps
        * distill_config.batch_size
        * world_size
    )

    # Prepare run directories and step intervals
    is_resuming = resume_config and resume_config.is_resume
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
        distill_config=distill_config,
        resume_config=resume_config,
        train_samples=distill_config.train_sequences,
        global_batch_size=global_batch_size,
    )
    output_dir_path = Path(output_dir)

    # ========================================================================
    # 4. Setup model and wrap with DDP
    # ========================================================================
    model = ProtX(model_config)

    # Load checkpoint if resuming
    start_epoch = 0
    start_step = 0
    if is_resuming:
        checkpoint_path = Path(resume_config.checkpoint_dir)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            start_epoch, start_step = _load_checkpoint(
                checkpoint_dir=checkpoint_path,
                model=model,
                optimizer=None,  # Will load optimizer state later
                scheduler=None,
                device=device,
            )

    model = _wrap_model_ddp(model, device, local_rank, world_size)

    # ========================================================================
    # 5. Setup data loaders with DistributedSampler
    # ========================================================================
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=distill_config.seed,
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if world_size > 1 else None

    # Determine num_workers based on dataset type
    is_cuda = torch.cuda.is_available() and str(device).startswith('cuda')
    effective_num_workers = 0 if distill_config.on_the_fly else max(4, distill_config.num_workers)

    data_collator = DistillDataCollator(
        teacher_model=teacher_model_for_collator,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        on_the_fly=distill_config.on_the_fly,
        max_seq_len=distill_config.max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=distill_config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if no sampler
        num_workers=effective_num_workers,
        pin_memory=is_cuda,
        collate_fn=data_collator,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=4 if effective_num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=distill_config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=is_cuda,
        collate_fn=data_collator,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=4 if effective_num_workers > 0 else None,
    )

    # ========================================================================
    # 6. Setup optimizer and scheduler
    # ========================================================================
    optimizer = AdamW(model.parameters(), lr=distill_config.learning_rate)

    # Calculate total training steps using train_sequences from manifest
    train_samples = distill_config.train_sequences
    # Use ceiling division to ensure at least 1 step per epoch
    steps_per_epoch = (train_samples + global_batch_size - 1) // global_batch_size
    num_training_steps = steps_per_epoch * num_epochs

    scheduler = _get_scheduler(
        optimizer=optimizer,
        lr_scheduler=distill_config.lr_scheduler,
        lr_scheduler_kwargs=distill_config.lr_scheduler_kwargs or {},
        num_training_steps=num_training_steps,
    )

    # Load optimizer/scheduler state if resuming
    if is_resuming:
        checkpoint_path = Path(resume_config.checkpoint_dir)
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state["optimizer"])
            if scheduler and training_state.get("scheduler"):
                scheduler.load_state_dict(training_state["scheduler"])
            logger.info("Loaded optimizer and scheduler states")

    # ========================================================================
    # 7. Initialize W&B (only main process)
    # ========================================================================
    if is_main_process:
        wandb_config = {
            "project": distill_config.project_name,
            "name": wandb_run_name,
            "config": {
                "model_config": model_config.__dict__,
                "distill_config": distill_config.__dict__,
                "world_size": world_size,
                "global_batch_size": global_batch_size,
            },
            "settings": wandb.Settings(start_method="fork"),
        }

        # Always create a new W&B run (with counter-based name for resumes)
        wandb_config["id"] = wandb_run_name

        wandb.init(**wandb_config)

        # Add W&B metadata for resume tracking
        if is_resuming and resume_step is not None and wandb.run is not None:
            try:
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
                logger.info(f"Added W&B metadata: resumed from step {resume_step}")
            except Exception as e:
                logger.warning(f"Failed to add W&B resume metadata: {e}")

        # Save W&B run ID
        if wandb.run is not None:
            actual_run_id = wandb.run.id
            run_id_path = output_dir_path / "wandb_run_id.txt"
            run_id_path.write_text(actual_run_id, encoding="utf-8")
            logger.info(f"Saved W&B run ID: {actual_run_id}")

    # Save training configuration
    if not is_resuming and is_main_process:
        training_config = {
            "train_fasta": str(distill_config.train_fasta),
            "val_fasta": str(distill_config.val_fasta),
            "train_h5_prefix": str(distill_config.train_h5_prefix),
            "val_h5_prefix": str(distill_config.val_h5_prefix),
            "num_epochs": distill_config.num_epochs,
            "batch_size": distill_config.batch_size,
            "learning_rate": distill_config.learning_rate,
            "gradient_accumulation_steps": distill_config.gradient_accumulation_steps,
            "max_seq_len": distill_config.max_seq_len,
        }
        _save_training_config(output_dir_path, training_config)

    # ========================================================================
    # 8. Training loop
    # ========================================================================
    logger.info(f"Starting native PyTorch training. Output dir: {output_dir}")
    logger.info(f"Training configuration:")
    logger.info(f"  World size: {world_size}")
    logger.info(f"  Rank: {rank}")
    logger.info(f"  Per-device batch size: {distill_config.batch_size}")
    logger.info(f"  Gradient accumulation steps: {distill_config.gradient_accumulation_steps}")
    logger.info(f"  Global batch size: {global_batch_size}")
    logger.info(f"  Total training steps: {num_training_steps}")
    logger.info(f"  Number of epochs: {num_epochs}")

    global_step = start_step

    for epoch in range(start_epoch, num_epochs):
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        global_step = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=distill_config.gradient_accumulation_steps,
            max_grad_norm=1.0,
            epoch=epoch,
            start_step=global_step,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=output_dir_path,
            rank=rank,
            world_size=world_size,
        )

        # Evaluate at the end of each epoch
        if val_dataset:
            logger.info(f"Evaluating at end of epoch {epoch + 1}")
            eval_metrics = _evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                rank=rank,
                world_size=world_size,
            )

            if is_main_process:
                wandb.log({
                    "eval/loss": eval_metrics["eval_loss"],
                    "eval/epoch": epoch,
                    "eval/step": global_step,
                })
                logger.info(f"Eval loss: {eval_metrics['eval_loss']:.4f}")

        # Save checkpoint at end of epoch
        _save_checkpoint(
            output_dir=output_dir_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            step=global_step,
            rank=rank,
        )

    # ========================================================================
    # 9. Final evaluation and cleanup
    # ========================================================================
    if val_dataset:
        logger.info("Final evaluation")
        eval_metrics = _evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            rank=rank,
            world_size=world_size,
        )

        if is_main_process:
            wandb.log({
                "final_eval/loss": eval_metrics["eval_loss"],
            })
            logger.info(f"Final eval loss: {eval_metrics['eval_loss']:.4f}")

    # Save final model
    _save_checkpoint(
        output_dir=output_dir_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=num_epochs,
        step=global_step,
        rank=rank,
    )

    if is_main_process:
        wandb.finish()
        logger.info("Native training complete!")

    # Cleanup distributed
    if world_size > 1:
        torch.distributed.destroy_process_group()
