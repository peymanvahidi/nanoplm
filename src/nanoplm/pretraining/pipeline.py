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
from nanoplm.pretraining.dataset import FastaMLMDataset
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
    samples_per_shard: int = 1000000
    max_workers: int = 1
    load_shards: bool = False
    warmup_ratio: float = 0.05
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    learning_rate: float = 3e-6
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
    num_workers: int = 0
    multi_gpu: bool = False
    world_size: Union[int, str] = 1
    project_name: str = "nanoplm-pretraining"


@dataclass
class ResumeConfig:
    is_resume: bool
    checkpoint_dir: str
    num_epochs: int


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
        trainer_state_path = Path(resume_config.checkpoint_dir) / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            current_epoch = trainer_state.get("epoch", 0)
            num_epochs = current_epoch + resume_config.num_epochs

            training_args_path = Path(resume_config.checkpoint_dir) / "training_args.bin"
            if training_args_path.exists():
                original_args = torch.load(training_args_path, weights_only=False)
                logging_steps = original_args.logging_steps
                eval_steps = original_args.eval_steps
                save_steps = original_args.save_steps
                logger.info(
                    f"Resuming with preserved intervals: save_steps={save_steps}, eval_steps={eval_steps}"
                )
            else:
                total_steps = num_epochs * len(train_ds) // global_batch_size
                logging_steps = max(1, int(total_steps * pretrain_config.logging_steps_percentage))
                eval_steps = max(1, int(total_steps * pretrain_config.eval_steps_percentage))
                save_steps = max(1, int(total_steps * pretrain_config.save_steps_percentage))
        else:
            num_epochs = resume_config.num_epochs
            total_steps = num_epochs * len(train_ds) // global_batch_size
            logging_steps = max(1, int(total_steps * pretrain_config.logging_steps_percentage))
            eval_steps = max(1, int(total_steps * pretrain_config.eval_steps_percentage))
            save_steps = max(1, int(total_steps * pretrain_config.save_steps_percentage))
    else:
        num_epochs = pretrain_config.num_epochs
        total_steps = num_epochs * len(train_ds) // global_batch_size
        logging_steps = max(1, int(total_steps * pretrain_config.logging_steps_percentage))
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

    train_ds, val_ds = _create_datasets(
        train_fasta=pretrain_config.train_fasta,
        val_fasta=pretrain_config.val_fasta,
        max_length=pretrain_config.max_length,
        lazy=pretrain_config.lazy_dataset,
        tokenizer=tokenizer,
        train_hdf5=pretrain_config.train_hdf5,
        val_hdf5=pretrain_config.val_hdf5,
        samples_per_shard=pretrain_config.samples_per_shard,
        max_workers=pretrain_config.max_workers,
        load_shards=pretrain_config.load_shards
    )
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
            effective_world_size = int(env_ws) if env_ws else max(torch.cuda.device_count(), 1)
        else:
            effective_world_size = int(pretrain_config.world_size) if  pretrain_config.world_size else 1
    else:
        effective_world_size = 1

    global_batch_size = pretrain_config.gradient_accumulation_steps * pretrain_config.batch_size * effective_world_size

    # Prepare run info and step intervals in a single place
    run_name, output_dir, num_epochs, logging_steps, eval_steps, save_steps = _prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_ds=train_ds,
        global_batch_size=global_batch_size,
    )

    # Configure Weights & Biases via environment variables so HF Trainer attaches correctly
    os.environ["WANDB_PROJECT"] = pretrain_config.project_name
    os.environ["WANDB_NAME"] = run_name

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
        "dataloader_num_workers": pretrain_config.num_workers,
    }

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
            logger.info(f"Resuming training from checkpoint: {resume_config.checkpoint_dir}")
            trainer.train(resume_from_checkpoint=resume_config.checkpoint_dir)
        else:
            trainer.train()
            
        # Capture and save W&B run ID for future resumes (if W&B is active)
        if wandb.run is not None:
            actual_run_id = wandb.run.id
            run_id_path = Path(output_dir) / "wandb_run_id.txt"
            if not run_id_path.exists() or run_id_path.read_text().strip() != actual_run_id:
                run_id_path.write_text(actual_run_id, encoding="utf-8")
                logger.info(f"Saved W&B run ID: {actual_run_id}")
    except Exception as e:
        logger.warning(f"Error during training or saving W&B run ID: {e}")
        raise

    logger.info("Saving final model and tokenizer")
    trainer.save_model(output_dir)
    trainer.save_state()


def _create_datasets(
    train_fasta: Union[str, Path],
    val_fasta: Union[str, Path],
    max_length: int,
    lazy: bool,
    tokenizer: ProtModernBertTokenizer,
    train_hdf5: Union[str, Path],
    val_hdf5: Union[str, Path],
    samples_per_shard: int,
    max_workers: int,
    load_shards: bool
) -> Tuple[Dataset, Optional[Dataset]]:

    train_ds = FastaMLMDataset(
        fasta_path=train_fasta,
        tokenizer=tokenizer,
        max_length=max_length,
        lazy=lazy,
        hdf5_dir=train_hdf5,
        samples_per_shard=samples_per_shard,
        max_workers=max_workers,
        load_shards=load_shards
    )

    val_ds = FastaMLMDataset(
        fasta_path=val_fasta,
        tokenizer=tokenizer,
        max_length=max_length,
        lazy=lazy,
        hdf5_dir=val_hdf5,
        samples_per_shard=samples_per_shard,
        max_workers=max_workers,
        load_shards=load_shards # for 
    )

    return train_ds, val_ds
