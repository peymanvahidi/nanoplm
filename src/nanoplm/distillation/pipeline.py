import os
import json
import time
import torch
import wandb
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path

from torch.utils.data import Dataset
from torch.optim import AdamW
from safetensors.torch import load_file
from transformers import (
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from nanoplm.distillation.collator import DistillDataCollator
from nanoplm.distillation.trainer import DistillationTrainer
from nanoplm.distillation.models.student import ProtX
from nanoplm.distillation.models.teacher import ProtT5
from nanoplm.distillation.dataset import (
    KDDatasetOnTheFly,
    LoadKDDataset,
    LoadKDDatasetOptimized,
)
from nanoplm.utils import get_device, logger, create_dirs


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""

    # Dataset paths
    train_fasta: Union[str, Path]
    val_fasta: Union[str, Path]
    train_h5_prefix: Union[str, Path]
    val_h5_prefix: Union[str, Path]

    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05

    # LR scheduler
    lr_scheduler: str = "cosine"
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None

    # Dataset config
    max_seq_len: int = 1024
    max_seqs_num: int = 100000
    val_ratio: float = 0.1
    on_the_fly: bool = False
    sharded: bool = False

    # Data loader optimization
    use_optimized_loader: bool = True
    max_open_files: int = 5
    chunk_size: int = 32
    prefetch_batches: int = 2
    use_threading: bool = True
    num_workers: int = 4

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

    def __post_init__(self):
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}


@dataclass
class StudentModelConfig:
    """Configuration for the student model architecture."""

    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_activation: str = "swiglu"
    use_feature_embedding: bool = False
    feature_window_size: int = 15
    projection_layer: bool = True


@dataclass
class ResumeConfig:
    """Configuration for resuming training from a checkpoint."""

    is_resume: bool = False
    checkpoint_dir: str = ""
    extra_epochs: Optional[int] = None


def _prepare_run_and_steps(
    distill_config: DistillationConfig,
    resume_config: Optional[ResumeConfig],
    train_ds: Dataset,
    global_batch_size: int,
) -> Tuple[str, str, int, int, int, int]:
    """Prepare run naming/dirs and compute epochs & step intervals.

    Returns a tuple: (run_name, output_dir, num_epochs, logging_steps, eval_steps, save_steps)
    """
    ckp_root = Path(distill_config.ckp_dir)

    # Determine run directory and name
    if resume_config and resume_config.is_resume:
        checkpoint_path = Path(resume_config.checkpoint_dir)
        original_run_name = checkpoint_path.parent.name
        run_name = f"{original_run_name}-resume"
        run_root = ckp_root / run_name
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

    create_dirs(str(run_root))
    output_dir = str(run_root)

    # Persist run metadata for future resumes
    try:
        (Path(output_dir) / "run_name.txt").write_text(run_name, encoding="utf-8")
    except Exception:
        pass

    # Compute epochs and step intervals
    num_epochs = distill_config.num_epochs
    if resume_config and resume_config.is_resume and resume_config.extra_epochs:
        num_epochs = distill_config.num_epochs + int(resume_config.extra_epochs)

    # Calculate total steps and interval steps
    train_samples = int(distill_config.max_seqs_num * (1 - distill_config.val_ratio))
    total_steps = (train_samples // global_batch_size) * num_epochs

    logging_steps = max(1, int(total_steps * distill_config.logging_steps_percentage))
    eval_steps = max(1, int(total_steps * distill_config.eval_steps_percentage))
    save_steps = max(1, int(total_steps * distill_config.save_steps_percentage))

    return run_name, output_dir, num_epochs, logging_steps, eval_steps, save_steps


def _load_datasets(
    distill_config: DistillationConfig,
    teacher=None,
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets."""

    if distill_config.on_the_fly:
        train_dataset = KDDatasetOnTheFly(
            input_fasta=distill_config.train_fasta,
            teacher=teacher,
            max_seq_len=distill_config.max_seq_len,
            device=get_device(),
        )
        val_dataset = KDDatasetOnTheFly(
            input_fasta=distill_config.val_fasta,
            teacher=teacher,
            max_seq_len=distill_config.max_seq_len,
            device=get_device(),
        )
    else:
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
    model: ProtX,
    distill_config: DistillationConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:
    """
    Run knowledge distillation training.

    Args:
        model: The student ProtX model to train
        distill_config: Configuration for distillation training
        resume_config: Optional configuration for resuming from checkpoint
    """

    device = get_device()
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

    # Setup teacher model if on-the-fly mode
    teacher = None
    teacher_model_for_collator = None
    if distill_config.on_the_fly:
        teacher = ProtT5()
        teacher_model_for_collator = teacher.encoder_model

    # Load datasets
    train_dataset, val_dataset = _load_datasets(
        distill_config=distill_config,
        teacher=teacher,
        seed=distill_config.seed,
    )

    # Prepare run info and step intervals
    is_resuming = resume_config and resume_config.is_resume

    run_name, output_dir, num_epochs, logging_steps, eval_steps, save_steps = (
        _prepare_run_and_steps(
            distill_config=distill_config,
            resume_config=resume_config,
            train_ds=train_dataset,
            global_batch_size=global_batch_size,
        )
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
        teacher_model=teacher_model_for_collator, on_the_fly=distill_config.on_the_fly
    )

    # Calculate num_training_steps for scheduler
    train_samples = int(distill_config.max_seqs_num * (1 - distill_config.val_ratio))
    num_training_steps = (train_samples // global_batch_size) * num_epochs

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
    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_LOG_MODEL"] = "end"

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
        "run_name": run_name,
        "dataloader_num_workers": distill_config.num_workers,
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
            "name": run_name,
            "config": training_args.to_dict(),
            "settings": wandb.Settings(start_method="fork"),
            "id": run_name,
        }
        if is_resuming:
            wandb_config["resume"] = "allow"
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
