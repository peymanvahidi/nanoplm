import os
import time
import torch
import torch.distributed as dist
import wandb
from datetime import datetime
from typing import Optional
from pathlib import Path

from transformers import (
    Trainer,
    TrainingArguments,
)

from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
from nanoplm.pretraining.models.modern_bert import ProtModernBertMLM
from nanoplm.pretraining.dataset import ShardedDataset
from nanoplm.pretraining.collator import ProtDataCollatorForLM
from nanoplm.pretraining.optim import build_muon_optimizer, is_muon_optimizer
from nanoplm.pretraining.utils import (
    compute_batch_setup,
    get_num_workers,
    prepare_run_and_steps,
)
from nanoplm.data.validation import validate_pretrain_dataset
from nanoplm.utils.logger import logger
from nanoplm.utils.common import get_device, create_dirs, resolve_world_size


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
        if logs is None:
            logs = {}

        optimizer = self.optimizer
        seen: set[int] = set()
        while optimizer is not None and not is_muon_optimizer(optimizer):
            opt_id = id(optimizer)
            if opt_id in seen:
                break
            seen.add(opt_id)
            inner = getattr(optimizer, "optimizer", None)
            if inner is None or inner is optimizer:
                break
            optimizer = inner

        if is_muon_optimizer(optimizer):
            # param_groups[0] = muon, param_groups[1] = adamw
            muon_lr = optimizer.param_groups[0]["lr"]
            adamw_lr = optimizer.param_groups[1]["lr"]
            logs["learning_rate"] = adamw_lr
            logs["adamw_lr"] = adamw_lr
            logs["muon_lr"] = muon_lr
        logs["tokens_per_sec"] = self._last_tokens_per_sec
        logs["raw_tokens_per_sec"] = self._last_raw_tokens_per_sec
        super().log(logs, start_time=start_time, **kwargs)


def run_pretraining(
    model: ProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:

    device = get_device()

    tokenizer = model.tokenizer
    model.to(device)

    # Validate dataset: manifest + shard files
    dataset_dir = Path(pretrain_config.dataset_dir)
    validation_result = validate_pretrain_dataset(dataset_dir)
    manifest = validation_result['manifest']

    # Get data from typed manifest
    train_shard_dir = dataset_dir / manifest.train_dir
    val_shard_dir = dataset_dir / manifest.val_dir
    train_sequences = manifest.train_sequences
    val_sequences = manifest.val_sequences

    # Load pre-tokenized binary shards
    logger.info("Using ShardedDataset for pre-tokenized binary shards")

    try:
        train_ds = ShardedDataset(data_dir=str(train_shard_dir))
        val_ds = ShardedDataset(data_dir=str(val_shard_dir))
    except FileNotFoundError as e:
        logger.error(
            f"Binary shards not found! You need to create them first.\n"
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

    effective_world_size = resolve_world_size(pretrain_config.multi_gpu, pretrain_config.world_size)

    batch = compute_batch_setup(pretrain_config, manifest.max_seq_len, effective_world_size)

    inferred_grad_accum_steps = batch.grad_accum_steps
    global_batch_size_samples = batch.global_batch_size_samples

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
    ) = prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=train_sequences,
        global_batch_size_samples=global_batch_size_samples,
    )

    # Configure Weights & Biases via environment variables so HF Trainer attaches correctly
    os.environ["WANDB_PROJECT"] = pretrain_config.project_name
    os.environ["WANDB_NAME"] = wandb_run_name

    num_workers = get_num_workers(pretrain_config.num_workers, effective_world_size)

    training_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": pretrain_config.micro_batch_size,
        "per_device_eval_batch_size": pretrain_config.micro_batch_size,
        "gradient_accumulation_steps": inferred_grad_accum_steps,
        "num_train_epochs": num_epochs,
        "learning_rate": pretrain_config.adam_learning_rate,
        "weight_decay": pretrain_config.adam_weight_decay,
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
    custom_optimizer = None
    if optimizer_name == "adamw":
        training_dict["optim"] = "adamw_torch"
    elif optimizer_name == "stable_adamw":
        training_dict["optim"] = "stable_adamw"
    elif optimizer_name in {"muon", "normuon"}:
        custom_optimizer = build_muon_optimizer(model, pretrain_config)
    else:
        raise ValueError(
            f"Invalid optimizer: {pretrain_config.optimizer}. "
            f"Currently supported: [adamw, stable_adamw, muon, normuon]"
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
        # When provided, custom optimizer/scheduler override TrainingArguments.optim.
        optimizers=(custom_optimizer, None),
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
