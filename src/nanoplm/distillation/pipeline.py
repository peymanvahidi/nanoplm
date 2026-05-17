import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_PERSISTENT_REDUCTIONS", "0")
os.environ.setdefault("TORCHINDUCTOR_MIX_ORDER_REDUCTION", "0")

import json
import shutil
import time
import torch
import wandb

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path

from torch.utils.data import Dataset
from torch.optim import AdamW
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
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.distillation.models.teacher import ProtT5
from nanoplm.distillation.dataset import (
    KDDatasetOnTheFly,
    LoadKDDataset,
)
from nanoplm.data.infra.manifest import get_dataset_paths
from nanoplm.data.infra.validation import validate_distillation_dataset, ValidationError
from nanoplm.pretraining.utils import _read_checkpoint_epoch
from nanoplm.utils import get_device, logger, create_dirs
from nanoplm.utils.wandb_artifacts import upload_run_source_snapshot


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
    max_grad_norm: float = float("inf")

    # Optimizer — "adamw" | "muon" | "normuon"
    # Muon/NorMuon routes 2-D weight matrices through dion.Muon/NorMuon and
    # biases / embeddings / special zero-init params through fused AdamW,
    # matching the pretraining optimizer setup.
    optimizer: str = "adamw"
    # Muon-specific (ignored when optimizer == "adamw")
    muon_learning_rate: float = 1e-3
    muon_weight_decay: float = 0.01
    muon_cautious_weight_decay: bool = True
    muon_use_polar_express: bool = True
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_eps: float = 1e-7
    # AdamW sub-optimizer used for 1-D / embedding params when optimizer is
    # muon/normuon, and as the sole optimizer when optimizer == "adamw".
    adam_learning_rate: float = 1e-4
    adam_weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Mixed precision
    bf16: bool = True
    tf32: bool = True

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
    max_open_files: int = 5
    chunk_size: int = 32
    prefetch_batches: int = 2
    use_threading: bool = True
    num_workers: int = 8  # Increased from 4 for better throughput

    # Checkpointing
    ckp_dir: str = "output/distillation"
    project_name: str = "nanoplm-distillation"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100

    # torch.compile
    use_compile: bool = False
    use_compile_max_autotune: bool = False
    # Disable TorchInductor persistent/mix-order reductions — can exceed
    # shared-memory limits for large hidden sizes (same flags as pretraining).
    compile_triton_persistent_reductions: bool = False
    compile_triton_mix_order_reduction: bool = False

    # Distributed
    multi_gpu: bool = False
    world_size: Union[int, str] = 1
    # "ddp" (default) or "fsdp" (FSDP2 with MixedPrecisionPolicy bf16/fp32)
    distributed_mode: str = "ddp"
    fsdp_shard_granularity: str = "layer"  # "layer" | "sublayer"
    seed: int = 42

    # Profiling (rank-0 only)
    profiler_enabled: bool = False
    profiler_start_step: int = 10
    profiler_end_step: int = 15

    def __post_init__(self):
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}


@dataclass
class ResumeConfig:
    """Configuration for resuming training from a checkpoint."""

    is_resume: bool = False
    checkpoint_dir: str = ""
    # resume_epochs: number of additional epochs to train from the checkpoint.
    # 0 (or None) means "continue to the original num_epochs from distill config".
    # N > 0 means "train for exactly N more epochs from the checkpoint" (i.e.,
    # num_epochs = checkpoint_epoch + N).
    resume_epochs: Optional[int] = None


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
    if (
        resume_config
        and resume_config.is_resume
        and resume_config.resume_epochs
        and int(resume_config.resume_epochs) > 0
    ):
        checkpoint_epoch = _read_checkpoint_epoch(resume_config.checkpoint_dir)
        num_epochs = checkpoint_epoch + int(resume_config.resume_epochs)
        if num_epochs <= checkpoint_epoch:
            raise ValueError(
                f"resume_epochs would result in zero or negative training: "
                f"checkpoint is at epoch {checkpoint_epoch}, requested target "
                f"is {num_epochs}."
            )

    # Calculate total steps using train_samples from manifest
    # Use ceiling division to ensure at least 1 step per epoch
    steps_per_epoch = (train_samples + global_batch_size - 1) // global_batch_size
    total_steps = steps_per_epoch * num_epochs

    # Use direct step counts from config (clamped to valid range)
    logging_steps = max(1, min(total_steps, distill_config.logging_steps))
    eval_steps = max(1, min(total_steps, distill_config.eval_steps))
    save_steps = max(1, min(total_steps, distill_config.save_steps))

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

    # Validate dataset before starting training (reads + validates manifest and data files)
    logger.info(f"Validating distillation dataset at {dataset_dir}...")
    try:
        validation_result = validate_distillation_dataset(dataset_dir)
        if validation_result['mode'] == 'on_the_fly':
            logger.info("On-the-fly mode: FASTA files validated")
        else:
            logger.info(f"Pre-computed mode: {len(validation_result.get('train_shards', []))} train shards, "
                        f"{len(validation_result.get('val_shards', []))} val shards validated")
    except ValidationError as e:
        logger.error(f"Dataset validation failed: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        raise

    # Use the typed manifest from validation (already read + validated)
    manifest = validation_result['manifest']

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

        train_dataset = LoadKDDataset(
            h5_path=distill_config.train_h5_prefix,
            device=get_device(),
            seed=effective_seed,
            sharded=distill_config.sharded,
            max_open_files=distill_config.max_open_files,
            chunk_size=distill_config.chunk_size,
            prefetch_batches=distill_config.prefetch_batches,
            use_threading=distill_config.use_threading,
        )
        val_dataset = LoadKDDataset(
            h5_path=distill_config.val_h5_prefix,
            device=get_device(),
            seed=effective_seed + 1,
            sharded=distill_config.sharded,
            max_open_files=distill_config.max_open_files,
            chunk_size=distill_config.chunk_size,
            prefetch_batches=distill_config.prefetch_batches,
            use_threading=distill_config.use_threading,
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
    model_config: ProtModernBertMLMConfig,
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
    is_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

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

    device_mesh = None

    # FSDP2 wrapping (must happen before torch.compile)
    if distill_config.distributed_mode == "fsdp" and effective_world_size > 1:
        import torch.distributed as dist
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
        from torch.distributed.device_mesh import init_device_mesh

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        device_mesh = init_device_mesh("cuda", (effective_world_size,))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        # Wrap at layer or sublayer granularity, matching pretraining
        if distill_config.fsdp_shard_granularity == "sublayer":
            for layer in model.model.layers:
                # MHCLiteSublayersLayer: .mhc_attn / .mhc_mlp at top level
                if hasattr(layer, "mhc_attn"):
                    fully_shard(layer.mhc_attn, mesh=device_mesh, mp_policy=mp_policy)
                    if hasattr(layer, "mhc_mlp") and layer.mhc_mlp is not None:
                        fully_shard(layer.mhc_mlp, mesh=device_mesh, mp_policy=mp_policy)
                # MHCLiteBlock: nested .layer.attn / .layer.mlp
                elif hasattr(layer, "layer") and hasattr(layer.layer, "attn"):
                    fully_shard(layer.layer.attn, mesh=device_mesh, mp_policy=mp_policy)
                    if hasattr(layer.layer, "mlp") and layer.layer.mlp is not None:
                        fully_shard(layer.layer.mlp, mesh=device_mesh, mp_policy=mp_policy)
                # Vanilla ModernBertEncoderLayer: .attn / .mlp
                elif hasattr(layer, "attn"):
                    fully_shard(layer.attn, mesh=device_mesh, mp_policy=mp_policy)
                    if hasattr(layer, "mlp") and layer.mlp is not None:
                        fully_shard(layer.mlp, mesh=device_mesh, mp_policy=mp_policy)
                else:
                    logger.warning(
                        f"FSDP2 sublayer wrap: skipping layer {type(layer).__name__} "
                        "(no recognized sub-modules)"
                    )
        else:  # "layer" — wrap whole transformer blocks
            for layer in model.model.layers:
                fully_shard(layer, mesh=device_mesh, mp_policy=mp_policy)

        fully_shard(model, mesh=device_mesh, mp_policy=mp_policy)
        logger.info(
            f"FSDP2 enabled: granularity={distill_config.fsdp_shard_granularity}, "
            f"world_size={effective_world_size}"
        )

    # torch.compile (applied after FSDP2 wrapping, before optimizer creation)
    if distill_config.use_compile and is_cuda:
        if not distill_config.compile_triton_persistent_reductions:
            torch._inductor.config.triton.persistent_reductions = False
        os.environ["TORCHINDUCTOR_MIX_ORDER_REDUCTION"] = (
            "1" if distill_config.compile_triton_mix_order_reduction else "0"
        )
        torch._inductor.config.triton.mix_order_reduction = (
            distill_config.compile_triton_mix_order_reduction
        )
        compile_mode = (
            "max-autotune-no-cudagraphs"
            if distill_config.use_compile_max_autotune
            else "default"
        )
        model = torch.compile(model, dynamic=True, mode=compile_mode)
        logger.info(f"torch.compile enabled (mode={compile_mode})")

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
            "train_sequences": distill_config.train_sequences,
            "val_sequences": distill_config.val_sequences,
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
        teacher_preprocess=None,
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
        "bf16": distill_config.bf16 and is_cuda and torch.cuda.is_bf16_supported(),
        "fp16": distill_config.bf16 and ((is_cuda and not torch.cuda.is_bf16_supported()) or device == "mps"),
        "tf32": distill_config.tf32 and is_cuda,
        "max_grad_norm": distill_config.max_grad_norm,
    }

    if distill_config.multi_gpu and distill_config.distributed_mode == "ddp":
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
        upload_run_source_snapshot()

    # Create optimizer
    opt_name = distill_config.optimizer.lower()
    if opt_name in ("muon", "normuon"):
        from nanoplm.pretraining.optim import build_muon_optimizer
        optimizer = build_muon_optimizer(model, distill_config, distributed_mesh=device_mesh)
        logger.info(f"Using {opt_name} optimizer (muon_lr={distill_config.muon_learning_rate}, "
                    f"adam_lr={distill_config.adam_learning_rate})")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=distill_config.learning_rate,
            fused=is_cuda and distill_config.distributed_mode != "fsdp",
        )
        logger.info(f"Using AdamW optimizer (lr={distill_config.learning_rate}, fused={is_cuda})")

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
        n_val = distill_config.val_sequences or "?"
        logger.info(f"Evaluating on {n_val} samples") 
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if is_main_process:
        wandb.finish()
        logger.info("Training complete!")
