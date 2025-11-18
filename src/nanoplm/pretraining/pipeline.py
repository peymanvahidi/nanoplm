import os
import json
import torch
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
from nanoplm.pretraining.dataset import (
    LazyFastaMLMDataset,
    LoadShardedFastaMLMDataset,
)
from nanoplm.pretraining.collator import ProtDataCollatorForLM
from nanoplm.utils.logger import logger
from nanoplm.utils.common import get_device, create_dirs


@dataclass
class PretrainingConfig:
    train_fasta: Union[str, Path]
    val_fasta: Union[str, Path]
    ckp_dir: str = "output/pretraining"
    max_length: int = 1024
    batch_size: int = 32
    num_epochs: int = 10
    lazy_dataset: bool = False
    train_hdf5: str = "output/data/split/train_hdf5"
    val_hdf5: str = "output/data/split/val_hdf5"
    load_all_in_memory: bool = False
    warmup_ratio: float = 0.05
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    mlm_probability: float = 0.3
    mask_replace_prob: float = 0.8
    random_token_prob: float = 0.1
    keep_probability: float = 0.1
    logging_steps_percentage: float = 0.01
    eval_steps_percentage: float = 0.025
    save_steps_percentage: float = 0.1
    seed: int = 42
    num_workers: Union[int, str] = "auto"
    prefetch_factor: int = 2
    multi_gpu: bool = False
    world_size: Union[int, str] = 1
    project_name: str = "nanoplm-pretraining"


@dataclass
class ResumeConfig:
    is_resume: bool
    checkpoint_dir: str
    extra_epochs: Optional[int] = None


def _prepare_run_and_steps(
    pretrain_config: "PretrainingConfig",
    resume_config: Optional["ResumeConfig"],
    train_ds: Dataset,
    global_batch_size: int,
) -> Tuple[str, str, int, int, int, int]:
    """Prepare run naming/dirs and compute epochs & step intervals.

    Returns a tuple: (run_name, output_dir, num_epochs, logging_steps, eval_steps, save_steps)
    """
    ckp_root = Path(pretrain_config.ckp_dir)

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
                total_steps = num_epochs * len(train_ds) // global_batch_size
                logging_steps = max(
                    1, int(total_steps * pretrain_config.logging_steps_percentage)
                )
                eval_steps = max(
                    1, int(total_steps * pretrain_config.eval_steps_percentage)
                )
                save_steps = max(
                    1, int(total_steps * pretrain_config.save_steps_percentage)
                )
        else:
            total_steps = num_epochs * len(train_ds) // global_batch_size
            logging_steps = max(
                1, int(total_steps * pretrain_config.logging_steps_percentage)
            )
            eval_steps = max(
                1, int(total_steps * pretrain_config.eval_steps_percentage)
            )
            save_steps = max(
                1, int(total_steps * pretrain_config.save_steps_percentage)
            )
    else:
        num_epochs = pretrain_config.num_epochs
        total_steps = num_epochs * len(train_ds) // global_batch_size
        logging_steps = max(
            1, int(total_steps * pretrain_config.logging_steps_percentage)
        )
        eval_steps = max(1, int(total_steps * pretrain_config.eval_steps_percentage))
        save_steps = max(1, int(total_steps * pretrain_config.save_steps_percentage))

    return run_name, output_dir, num_epochs, logging_steps, eval_steps, save_steps


def run_pretraining(
    model: ProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:

    device = get_device()

    tokenizer = model.tokenizer
    model.to(device)

    if pretrain_config.lazy_dataset:
        if pretrain_config.train_fasta is None or pretrain_config.val_fasta is None:
            raise ValueError("Train and validation FASTA files are required when lazy-dataset mode is enabled")
        if not Path(pretrain_config.train_fasta).exists():
            raise FileNotFoundError(f"Train FASTA file not found: {pretrain_config.train_fasta}")
        if not Path(pretrain_config.val_fasta).exists():
            raise FileNotFoundError(f"Validation FASTA file not found: {pretrain_config.val_fasta}")

        # Use lazy loading: tokenize on-the-fly from FASTA
        logger.info("Using LazyFastaMLMDataset for on-the-fly tokenization")
        train_ds, val_ds = _create_lazy_datasets(
            train_fasta=pretrain_config.train_fasta,
            val_fasta=pretrain_config.val_fasta,
            max_length=pretrain_config.max_length,
            tokenizer=tokenizer,
        )
    else:
        if pretrain_config.train_hdf5 is None or pretrain_config.val_hdf5 is None:
            raise ValueError("Train and validation HDF5 directories are required when lazy-dataset mode is disabled")
        if not Path(pretrain_config.train_hdf5).exists():
            raise FileNotFoundError(f"Train HDF5 directory not found: {pretrain_config.train_hdf5}")
        if not Path(pretrain_config.val_hdf5).exists():
            raise FileNotFoundError(f"Validation HDF5 directory not found: {pretrain_config.val_hdf5}")

        # Load pre-tokenized HDF5 shards
        logger.info("Using LoadShardedFastaMLMDataset for pre-tokenized HDF5 shards")
        logger.info(f"Expected train shards: {pretrain_config.train_hdf5}")
        logger.info(f"Expected val shards: {pretrain_config.val_hdf5}")

        try:
            train_ds = LoadShardedFastaMLMDataset(hdf5_dir=pretrain_config.train_hdf5, load_all_in_memory=pretrain_config.load_all_in_memory)
            val_ds = LoadShardedFastaMLMDataset(hdf5_dir=pretrain_config.val_hdf5, load_all_in_memory=pretrain_config.load_all_in_memory)
        except FileNotFoundError as e:
            logger.error(
                f"HDF5 shards not found! You need to create them first.\n"
                f"Run: nanoplm data from-yaml --pretrain <your_data_config.yaml>\n"
                f"Or set lazy_dataset=True in your pretrain.yaml to use on-the-fly tokenization.\n"
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
    run_name, output_dir, num_epochs, logging_steps, eval_steps, save_steps = (
        _prepare_run_and_steps(
            pretrain_config=pretrain_config,
            resume_config=resume_config,
            train_ds=train_ds,
            global_batch_size=global_batch_size,
        )
    )

    # Configure Weights & Biases via environment variables so HF Trainer attaches correctly
    os.environ["WANDB_PROJECT"] = pretrain_config.project_name
    os.environ["WANDB_NAME"] = run_name

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
        "report_to": "wandb",
        "run_name": run_name,
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

    trainer = Trainer(
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

def _create_lazy_datasets(
    train_fasta: Union[str, Path],
    val_fasta: Union[str, Path],
    max_length: int,
    tokenizer: ProtModernBertTokenizer,
) -> Tuple[Dataset, Optional[Dataset]]:

    train_ds = LazyFastaMLMDataset(
        fasta_path=train_fasta,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_ds = LazyFastaMLMDataset(
        fasta_path=val_fasta,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return train_ds, val_ds
