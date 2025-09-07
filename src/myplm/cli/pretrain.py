#!/usr/bin/env python3
"""
MyPLM CLI - Pretraining subcommands for MyPLM package
"""

import click
from pathlib import Path
from typing import Optional

from myplm.models.student.pretraining import (
    ProtXMLMConfig,
    ProtXMLM,
    ProtXMLMTokenizer,
    ProteinMLMDataset,
    MLMDataCollator,
    MLMTrainer,
    create_training_args,
    create_trainer
)
from myplm.utils import logger


@click.group(name="pretrain")
@click.help_option('--help', '-h')
def pretrain():
    """Pretraining-related commands: mlm."""
    pass


@pretrain.command("mlm")
@click.help_option('--help', '-h')
@click.option(
    '--train-file',
    type=str,
    required=True,
    help='Path to the training FASTA file'
)
@click.option(
    '--val-file',
    type=str,
    required=False,
    help='Path to the validation FASTA file (optional)'
)
@click.option(
    '--output-dir',
    type=str,
    default='./mlm_checkpoints',
    help='Directory to save model checkpoints'
)
@click.option(
    '--embed-dim',
    type=int,
    default=512,
    help='Embedding dimension of the model'
)
@click.option(
    '--num-layers',
    type=int,
    default=12,
    help='Number of transformer layers'
)
@click.option(
    '--num-heads',
    type=int,
    default=8,
    help='Number of attention heads'
)
@click.option(
    '--mlp-activation',
    type=str,
    default='swiglu',
    help='MLP activation function'
)
@click.option(
    '--max-seq-len',
    type=int,
    default=512,
    help='Maximum sequence length'
)
@click.option(
    '--min-seq-len',
    type=int,
    default=20,
    help='Minimum sequence length'
)
@click.option(
    '--subsample-ratio',
    type=float,
    default=1.0,
    help='Ratio of data to use for training (0.0-1.0)'
)
@click.option(
    '--mlm-probability',
    type=float,
    default=0.15,
    help='Probability of masking tokens for MLM'
)
@click.option(
    '--mask-token-prob',
    type=float,
    default=0.8,
    help='Probability of replacing masked tokens with [MASK]'
)
@click.option(
    '--random-token-prob',
    type=float,
    default=0.1,
    help='Probability of replacing masked tokens with random tokens'
)
@click.option(
    '--num-epochs',
    type=int,
    default=3,
    help='Number of training epochs'
)
@click.option(
    '--batch-size',
    type=int,
    default=16,
    help='Batch size for training'
)
@click.option(
    '--learning-rate',
    type=float,
    default=5e-4,
    help='Learning rate'
)
@click.option(
    '--weight-decay',
    type=float,
    default=0.01,
    help='Weight decay'
)
@click.option(
    '--warmup-steps',
    type=int,
    default=500,
    help='Number of warmup steps'
)
@click.option(
    '--max-steps',
    type=int,
    default=-1,
    help='Maximum number of training steps (-1 for epoch-based)'
)
@click.option(
    '--save-steps',
    type=int,
    default=1000,
    help='Save checkpoint every N steps'
)
@click.option(
    '--eval-steps',
    type=int,
    default=500,
    help='Evaluate every N steps'
)
@click.option(
    '--logging-steps',
    type=int,
    default=100,
    help='Log every N steps'
)
@click.option(
    '--save-total-limit',
    type=int,
    default=3,
    help='Maximum number of checkpoints to keep'
)
@click.option(
    '--fp16',
    is_flag=True,
    help='Use 16-bit floating point precision'
)
@click.option(
    '--gradient-accumulation-steps',
    type=int,
    default=1,
    help='Number of gradient accumulation steps'
)
@click.option(
    '--dataloader-num-workers',
    type=int,
    default=0,
    help='Number of workers for data loading'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility'
)
@click.option(
    '--device',
    type=str,
    default='auto',
    help='Device to use for training (auto, cpu, cuda, mps)'
)
@click.option(
    '--wandb-project',
    type=str,
    default=None,
    help='Weights & Biases project name'
)
@click.option(
    '--wandb-run-name',
    type=str,
    default=None,
    help='Weights & Biases run name'
)
@click.option(
    '--resume-from-checkpoint',
    type=str,
    default=None,
    help='Path to checkpoint to resume from'
)
def mlm(
    train_file: str,
    val_file: Optional[str],
    output_dir: str,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    mlp_activation: str,
    max_seq_len: int,
    min_seq_len: int,
    subsample_ratio: float,
    mlm_probability: float,
    mask_token_prob: float,
    random_token_prob: float,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: int,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    save_total_limit: int,
    fp16: bool,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    seed: int,
    device: str,
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
    resume_from_checkpoint: Optional[str]
):
    """
    Run masked language model (MLM) pretraining on protein sequences.
    
    This command performs self-supervised pretraining using masked language modeling
    on protein sequences. The model learns to predict masked amino acids in protein
    sequences, which can then be fine-tuned for downstream tasks or used for 
    knowledge distillation.
    
    Examples:
    
    \b
      # Basic MLM pretraining
      myplm pretrain mlm --train-file data/train.fasta --output-dir ./mlm_out
      
    \b
      # MLM with validation and custom parameters
      myplm pretrain mlm \\
        --train-file data/train.fasta \\
        --val-file data/val.fasta \\
        --embed-dim 256 \\
        --num-layers 6 \\
        --batch-size 32 \\
        --num-epochs 5
    """
    
    logger.info("Starting MLM pretraining...")
    logger.info(f"Train file: {train_file}")
    logger.info(f"Val file: {val_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Validate input files
    train_path = Path(train_file)
    if not train_path.exists():
        raise click.ClickException(f"Training file not found: {train_file}")
    
    val_path = None
    if val_file:
        val_path = Path(val_file)
        if not val_path.exists():
            raise click.ClickException(f"Validation file not found: {val_file}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up device
    import torch
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Set random seed
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        # Create tokenizer
        logger.info("Creating tokenizer...")
        tokenizer = ProtXMLMTokenizer()
        
        # Create datasets
        logger.info("Loading training dataset...")
        train_dataset = ProteinMLMDataset(
            fasta_path=train_path,
            tokenizer=tokenizer,
            max_length=max_seq_len,
            min_length=min_seq_len,
            subsample_ratio=subsample_ratio,
            seed=seed
        )
        
        val_dataset = None
        if val_path:
            logger.info("Loading validation dataset...")
            val_dataset = ProteinMLMDataset(
                fasta_path=val_path,
                tokenizer=tokenizer,
                max_length=max_seq_len,
                min_length=min_seq_len,
                subsample_ratio=subsample_ratio,
                seed=seed + 1  # Different seed for validation
            )
        
        # Create data collator
        logger.info("Creating data collator...")
        leave_unchanged_prob = 1.0 - mask_token_prob - random_token_prob
        data_collator = MLMDataCollator(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            mask_token_probability=mask_token_prob,
            random_token_probability=random_token_prob,
            leave_unchanged_probability=leave_unchanged_prob
        )
        
        # Create model config
        logger.info("Creating model configuration...")
        config = ProtXMLMConfig(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_activation=mlp_activation,
            vocab_size=len(tokenizer.vocab),
            max_position_embeddings=max_seq_len
        )
        
        # Create model
        logger.info("Creating model...")
        model = ProtXMLM(config)
        model = model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {num_params:,} parameters")
        
        # Setup wandb if requested
        report_to = []
        if wandb_project:
            report_to = ["wandb"]
            import os
            os.environ["WANDB_PROJECT"] = wandb_project
            if wandb_run_name:
                os.environ["WANDB_RUN_NAME"] = wandb_run_name
        
        # Create training arguments
        logger.info("Creating training arguments...")
        training_args = create_training_args(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps if val_dataset else -1,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            fp16=fp16,
            dataloader_num_workers=dataloader_num_workers,
            remove_unused_columns=False,
            report_to=report_to,
            seed=seed
        )
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = MLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        
        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        logger.info("MLM pretraining completed successfully!")
        
    except Exception as e:
        logger.error(f"MLM pretraining failed: {e}")
        raise click.ClickException(f"Training failed: {e}")


@pretrain.command("convert-for-distillation")
@click.help_option('--help', '-h')
@click.option(
    '--checkpoint-path',
    type=str,
    required=True,
    help='Path to MLM checkpoint directory'
)
@click.option(
    '--output-path',
    type=str,
    required=True,
    help='Path to save converted model (should end with .safetensors)'
)
def convert_for_distillation(checkpoint_path: str, output_path: str):
    """
    Convert a pretrained MLM model for use in knowledge distillation.
    
    This removes the MLM head and prepares the model weights for loading
    into a ProtX model for distillation training.
    
    Examples:
    
    \b
      myplm pretrain convert-for-distillation \\
        --checkpoint-path ./mlm_checkpoints \\
        --output-path ./pretrained_protx.safetensors
    """
    
    logger.info("Converting MLM checkpoint for distillation...")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Output path: {output_path}")
    
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise click.ClickException(f"Checkpoint directory not found: {checkpoint_path}")
    
    try:
        # Load the MLM model
        from transformers import AutoConfig, AutoModel
        
        # Try to load config
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            config = AutoConfig.from_pretrained(checkpoint_dir)
            logger.info(f"Loaded config: {config}")
        else:
            logger.warning("No config.json found, will try to infer from model")
        
        # Load the MLM model
        logger.info("Loading MLM model...")
        model = ProtXMLM.from_pretrained(checkpoint_dir)
        
        # Save for distillation
        from myplm.models.student.pretraining import save_mlm_model_for_downstream
        save_mlm_model_for_downstream(
            model=model,
            output_path=output_path,
            save_config=True
        )
        
        logger.info("Conversion completed successfully!")
        logger.info(f"Converted model saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise click.ClickException(f"Conversion failed: {e}")
