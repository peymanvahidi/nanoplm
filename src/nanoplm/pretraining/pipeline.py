import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, Union
from pathlib import Path

from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import (
    Trainer,
    TrainingArguments,
)

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.dataset import FastaMLMDataset
from nanoplm.pretraining.collator import MLMDataCollator
from nanoplm.utils.logger import logger
from nanoplm.utils.common import get_device, create_dirs

@dataclass
class PretrainingConfig:
    train_fasta: Union[str, Path]
    val_fasta: Union[str, Path]
    output_dir: str = "output/pretraining"
    max_length: int = 512
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    mlm_probability: float = 0.15
    gradient_accumulation_steps: int = 1
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    logging_steps: int = 50
    seed: int = 42
    mask_replace_prob: float = 0.8
    random_token_prob: float = 0.1
    leave_unchanged_prob: float = 0.1


def run_pretraining(
    model: ProtModernBertMLM,
    config: PretrainingConfig
) -> None:

    device = get_device()

    tokenizer = model.tokenizer
    model.to(device)

    train_ds, val_ds = _create_datasets(
        train_fasta=config.train_fasta,
        val_fasta=config.val_fasta,
        max_length=config.max_length,
        tokenizer=tokenizer
    )
    collator = MLMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=config.mlm_probability,
        mask_token_probability=config.mask_replace_prob,
        random_token_probability=config.random_token_prob,
        leave_unchanged_probability=config.leave_unchanged_prob,
    )

    create_dirs(config.output_dir)

    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if config.eval_steps else "no",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        seed=config.seed,
        report_to=["wandb"],
        # dataloader_pin_memory=True,
        # dataloader_num_workers=max(os.cpu_count() - 1, 1),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        optimizers=[optimizer, None],
    )

    logger.info("Starting Trainer.fit()")
    trainer.train()

    logger.info("Saving final model and tokenizer")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

def _create_datasets(
    train_fasta: Union[str, Path],
    val_fasta: Union[str, Path],
    max_length: int,
    tokenizer: ProtModernBertTokenizer,
) -> Tuple[Dataset, Optional[Dataset]]:

    train_ds = FastaMLMDataset(
        fasta_path=train_fasta,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_ds = FastaMLMDataset(
        fasta_path=val_fasta,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return train_ds, val_ds
