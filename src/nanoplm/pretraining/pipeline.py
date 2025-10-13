import os
import torch
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
    run_name: str = "nanoplm-pretraining"


@dataclass
class ResumeConfig:
    checkpoint_dir: str
    num_epochs: int


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
    )
    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
    )

    create_dirs(pretrain_config.ckp_dir)

    if pretrain_config.world_size == "auto":
        env_ws = os.environ.get("WORLD_SIZE")
        pretrain_config.world_size = int(env_ws) if env_ws else max(torch.cuda.device_count(), 1)

    global_batch_size = pretrain_config.gradient_accumulation_steps * pretrain_config.batch_size * pretrain_config.world_size

    total_steps = pretrain_config.num_epochs * len(train_ds) // global_batch_size

    training_dict = {
        "output_dir": pretrain_config.ckp_dir,
        "per_device_train_batch_size": pretrain_config.batch_size,
        "per_device_eval_batch_size": pretrain_config.batch_size,
        "gradient_accumulation_steps": pretrain_config.gradient_accumulation_steps,
        "num_train_epochs": pretrain_config.num_epochs,
        "learning_rate": pretrain_config.learning_rate,
        "weight_decay": pretrain_config.weight_decay,
        "warmup_ratio": pretrain_config.warmup_ratio,
        "logging_strategy": "steps",
        "logging_steps": max(1, int(total_steps * pretrain_config.logging_steps_percentage)),
        "logging_dir": Path(pretrain_config.ckp_dir) / "logs",
        "eval_strategy": "steps",
        "eval_steps": max(1, int(total_steps * pretrain_config.eval_steps_percentage)),
        "save_strategy": "steps",
        "save_steps": max(1, int(total_steps * pretrain_config.save_steps_percentage)),
        "seed": pretrain_config.seed,
        "report_to": "wandb",
        "run_name": pretrain_config.run_name,
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
    trainer.train()

    logger.info("Saving final model and tokenizer")
    trainer.save_model(pretrain_config.ckp_dir)


def _create_datasets(
    train_fasta: Union[str, Path],
    val_fasta: Union[str, Path],
    max_length: int,
    lazy: bool,
    tokenizer: ProtModernBertTokenizer,
) -> Tuple[Dataset, Optional[Dataset]]:

    train_ds = FastaMLMDataset(
        fasta_path=train_fasta,
        tokenizer=tokenizer,
        max_length=max_length,
        lazy=lazy,
    )

    val_ds = FastaMLMDataset(
        fasta_path=val_fasta,
        tokenizer=tokenizer,
        max_length=max_length,
        lazy=lazy,
    )

    return train_ds, val_ds
