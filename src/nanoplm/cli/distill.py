#!/usr/bin/env python3
"""
nanoPLM CLI - Distillation subcommands for nanoPLM package
"""

import click
import json
from typing import Optional

from nanoplm.distillation.pipeline_builder import DistillationPipelineBuilder
from nanoplm.utils import logger


@click.group(name="distill")
@click.help_option('--help', '-h')
def distill():
    """Group of commands for distillation models."""
    pass

@distill.command("run")
@click.help_option('--help', '-h')
@click.option(
    '--train-file',
    type=str,
    required=True,
    help='Path to the training file'
)
@click.option(
    '--val-file',
    type=str,
    required=True,
    help='Path to the validation file'
)
@click.option(
    '--protx-train-prefix',
    type=str,
    required=True,
    help='Prefix of the training nanoplm dataset'
)
@click.option(
    '--protx-val-prefix',
    type=str,
    required=True,
    help='Prefix of the validation nanoplm dataset'
)
@click.option(
    '--student-embed-dim',
    type=int,
    default=512,
    help='Embedding dimension of the student model'
)
@click.option(
    '--student-num-layers',
    type=int,
    default=6,
    help='Number of layers of the student model'
)
@click.option(
    '--student-num-heads',
    type=int,
    default=8,
    help='Number of heads of the student model'
)
@click.option(
    '--on-the-fly',
    is_flag=True,
    help='Whether to use on-the-fly teacher embeddings'
)
@click.option(
    '--multi-gpu',
    is_flag=True,
    help='Whether to use multiple GPUs for training'
)
@click.option(
    '--num-epochs',
    type=int,
    default=10,
    help='Number of epochs to train the student model'
)
@click.option(
    '--batch-size',
    type=int,
    default=64,
    help='Batch size for training'
)
@click.option(
    '--max-lr',
    type=float,
    default=1e-3,
    help='Maximum learning rate'
)
@click.option(
    '--max-grad-norm',
    type=float,
    default=None,
    help='Maximum gradient norm for gradient clipping'
)
@click.option(
    '--max-seqs-num',
    type=int,
    required=True,
    help='Maximum number of sequences to use for training'
)
@click.option(
    '--max-seq-len',
    type=int,
    default=1024,
    help='Maximum sequence length'
)
@click.option(
    '--val-ratio',
    type=float,
    default=0.1,
    help='Ratio of validation set'
)
@click.option(
    '--num-workers',
    type=int,
    default=4,
    help='Number of workers to use for training'
)
@click.option(
    '--project-name',
    type=str,
    default="protx_distillation",
    help='Name of the project'
)
# @click.option(
#     '--checkpoint-dir',
#     type=str,
#     default=None,
#     help='Path to the checkpoint directory to resume training from'
# )
@click.option(
    '--wandb-dir',
    type=str,
    default=None,
    help='Path to the directory to save the wandb logs'
)
@click.option(
    '--device',
    default='cuda',
    type=click.Choice(['cuda', 'mps', 'cpu']),
    help='Device to use'
)
@click.option(
    '--lr-scheduler',
    type=click.Choice(['cosine', 'linear', 'polynomial', 'constant']),
    default='cosine',
    help='Learning rate scheduler type'
)
@click.option(
    '--lr-scheduler-kwargs',
    type=str,
    default=None,
    help='JSON string of additional kwargs for the learning rate scheduler (optional). ' +
         'Example: \'{"num_cycles": 1.0, "power": 1.0}\' for cosine/polynomial schedulers'
)
@click.option(
    '--sharded',
    is_flag=True,
    help='Whether to use sharded H5 files for data loading (improves performance with large files)'
)
@click.option(
    '--use-optimized-loader',
    is_flag=True,
    default=True,
    help='Use optimized data loader for better performance with large datasets (default: enabled)'
)
@click.option(
    '--max-open-files',
    type=int,
    default=5,
    help='Maximum number of H5 files to keep open simultaneously (memory vs performance trade-off)'
)
@click.option(
    '--chunk-size',
    type=int,
    default=32,
    help='Number of samples to read per chunk for better I/O efficiency'
)
@click.option(
    '--prefetch-batches',
    type=int,
    default=2,
    help='Number of batches to prefetch in background for smoother data loading'
)
@click.option(
    '--no-threading',
    is_flag=True,
    help='Disable threading for I/O operations (use if experiencing threading issues)'
)
@click.option(
    '--no-projection-layer',
    is_flag=True,
    help='Disable projection layer (student and teacher embeddings must have same dimension 1024)'
)
def run(
    train_file: str,
    val_file: str,
    protx_train_prefix: str,
    protx_val_prefix: str,
    student_embed_dim: int,
    student_num_layers: int,
    student_num_heads: int,
    on_the_fly: bool,
    multi_gpu: bool,
    num_epochs: int,
    batch_size: int,
    max_lr: float,
    max_grad_norm: Optional[float],
    max_seqs_num: int,
    max_seq_len: int,
    val_ratio: float,
    num_workers: int,
    project_name: str,
    # checkpoint_dir: str,
    wandb_dir: str,
    device: str,
    lr_scheduler: str,
    lr_scheduler_kwargs: str,
    sharded: bool,
    use_optimized_loader: bool,
    max_open_files: int,
    chunk_size: int,
    prefetch_batches: int,
    no_threading: bool,
    no_projection_layer: bool,
):
    """Distill the teacher model into a student model"""
    # Parse lr_scheduler_kwargs if provided
    parsed_lr_kwargs = {}
    if lr_scheduler_kwargs:
        try:
            parsed_lr_kwargs = json.loads(lr_scheduler_kwargs)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for lr-scheduler-kwargs: {e}")
            click.echo(f"Invalid JSON for lr-scheduler-kwargs: {e}", err=True)
            raise click.Abort()
    
    builder = DistillationPipelineBuilder()

    pipeline = (
        builder
            .with_training_files(
                train_file=train_file,
                val_file=val_file,
                protx_train_prefix=protx_train_prefix,
                protx_val_prefix=protx_val_prefix
            )
            .with_model_config(
                student_embed_dim=student_embed_dim,
                student_num_layers=student_num_layers,
                student_num_heads=student_num_heads,
                projection_layer=not no_projection_layer,
            )
            .with_training_config(
                num_epochs=num_epochs,
                batch_size=batch_size,
                max_lr=max_lr,
                max_seqs_num=max_seqs_num,
                max_seq_len=max_seq_len,
                val_ratio=val_ratio,
                num_workers=num_workers,
                lr_scheduler=lr_scheduler,
                lr_scheduler_kwargs=parsed_lr_kwargs,
                max_grad_norm=max_grad_norm,
            )
            .with_experiment_config(
                project_name=project_name,
                wandb_dir=wandb_dir,
                device=device,
                on_the_fly=on_the_fly,
                multi_gpu=multi_gpu,
                sharded=sharded,
                use_optimized_loader=use_optimized_loader,
                max_open_files=max_open_files,
                chunk_size=chunk_size,
                prefetch_batches=prefetch_batches,
                use_threading=not no_threading,
            )
        .build()
    )

    pipeline.train()

@distill.command("run-resume")
@click.help_option('--help', '-h')
@click.option(
    '--checkpoint-dir',
    type=str,
    required=True,
    help='Path to the checkpoint directory to resume training from'
)
@click.option(
    '--num-epochs',
    type=int,
    required=True,
    help='Number of epochs to train the student model'
)
@click.option(
    '--lr',
    type=float,
    default=None,
    help='Override learning rate for resumed training (optional)'
)
@click.option(
    '--lr-scheduler',
    type=click.Choice(['cosine', 'linear', 'polynomial', 'constant']),
    default=None,
    help='Learning rate scheduler type (optional, defaults to cosine)'
)
@click.option(
    '--lr-scheduler-kwargs',
    type=str,
    default=None,
    help='JSON string of additional kwargs for the learning rate scheduler (optional). ' +
         'Example: \'{"num_cycles": 1.0, "power": 1.0}\' for cosine/polynomial schedulers'
)
@click.option(
    '--max-grad-norm',
    type=float,
    default=None,
    help='Override gradient clipping norm for resumed training (optional)'
)
def resume_distillation(
    checkpoint_dir: str,
    num_epochs: int,
    lr: float,
    lr_scheduler: str,
    lr_scheduler_kwargs: str,
    max_grad_norm: float
):
    """Resume distillation from a checkpoint with optional learning rate and scheduler overrides.
    
    Examples:
        # Resume with new learning rate:
        protx distill resume --checkpoint-dir ./run-123/checkpoint-1500 --num-epochs 20 --lr 5e-4

        # Resume with linear scheduler:
        protx distill resume --checkpoint-dir ./run-123/checkpoint-1500 --num-epochs 20 --lr-scheduler linear

        # Resume with constant learning rate:
        protx distill resume --checkpoint-dir ./run-123/checkpoint-1500 --num-epochs 20 --lr-scheduler constant
    """
    # Parse lr_scheduler_kwargs if provided
    parsed_lr_kwargs = {}
    if lr_scheduler_kwargs:
        try:
            parsed_lr_kwargs = json.loads(lr_scheduler_kwargs)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for lr-scheduler-kwargs: {e}")
            click.echo(f"Invalid JSON for lr-scheduler-kwargs: {e}", err=True)
            raise click.Abort()
    
    builder = DistillationPipelineBuilder()
    
    # Build overrides dict
    overrides = {"num_epochs": num_epochs}
    if lr is not None:
        overrides["max_lr"] = lr
    if max_grad_norm is not None:
        overrides["max_grad_norm"] = max_grad_norm
    if lr_scheduler is not None:
        overrides["lr_scheduler"] = lr_scheduler
    if parsed_lr_kwargs:
        overrides["lr_scheduler_kwargs"] = parsed_lr_kwargs
    
    pipeline = builder.resume_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        **overrides
    )
    pipeline.train()

