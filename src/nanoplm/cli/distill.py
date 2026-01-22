#!/usr/bin/env python3
"""
nanoPLM CLI - Distillation subcommands for nanoPLM package
"""

import click
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path

from nanoplm.distillation.pipeline import (
    DistillationConfig,
    StudentModelConfig,
    ResumeConfig,
    run_distillation,
)
from nanoplm.distillation.models.student import ProtX
from nanoplm.utils.common import read_yaml, create_dirs
from nanoplm.utils.logger import logger


@click.group(name="distill")
@click.help_option('--help', '-h')
def distill():
    """Group of commands for distillation models."""
    pass


@distill.command("run")
@click.help_option('--help', '-h')
@click.option(
    '--train-fasta',
    type=str,
    required=True,
    help='Path to the training FASTA file'
)
@click.option(
    '--val-fasta',
    type=str,
    required=True,
    help='Path to the validation FASTA file'
)
@click.option(
    '--train-h5-prefix',
    type=str,
    required=True,
    help='Prefix path for training H5 dataset'
)
@click.option(
    '--val-h5-prefix',
    type=str,
    required=True,
    help='Prefix path for validation H5 dataset'
)
@click.option(
    '--embed-dim',
    type=int,
    default=512,
    help='Embedding dimension of the student model'
)
@click.option(
    '--num-layers',
    type=int,
    default=6,
    help='Number of layers of the student model'
)
@click.option(
    '--num-heads',
    type=int,
    default=8,
    help='Number of attention heads of the student model'
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
    '--learning-rate',
    type=float,
    default=1e-3,
    help='Maximum learning rate'
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
    help='Number of workers to use for data loading'
)
@click.option(
    '--project-name',
    type=str,
    default="nanoplm-distillation",
    help='Name of the W&B project'
)
@click.option(
    '--ckp-dir',
    type=str,
    default="output/distillation",
    help='Directory to save checkpoints'
)
@click.option(
    '--lr-scheduler',
    type=click.Choice(['cosine', 'linear', 'polynomial', 'constant']),
    default='cosine',
    help='Learning rate scheduler type'
)
@click.option(
    '--sharded',
    is_flag=True,
    help='Whether to use sharded H5 files for data loading'
)
@click.option(
    '--no-projection-layer',
    is_flag=True,
    help='Disable projection layer (student and teacher embeddings must have same dimension 1024)'
)
@click.option(
    '--gradient-accumulation-steps',
    type=int,
    default=1,
    help='Gradient accumulation steps'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed'
)
def run(
    train_fasta: str,
    val_fasta: str,
    train_h5_prefix: str,
    val_h5_prefix: str,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    on_the_fly: bool,
    multi_gpu: bool,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seqs_num: int,
    max_seq_len: int,
    val_ratio: float,
    num_workers: int,
    project_name: str,
    ckp_dir: str,
    lr_scheduler: str,
    sharded: bool,
    no_projection_layer: bool,
    gradient_accumulation_steps: int,
    seed: int,
):
    """Distill the teacher model into a student model"""

    # Create student model
    model = ProtX(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        projection_layer=not no_projection_layer,
    )

    # Create distillation config
    distill_config = DistillationConfig(
        train_fasta=train_fasta,
        val_fasta=val_fasta,
        train_h5_prefix=train_h5_prefix,
        val_h5_prefix=val_h5_prefix,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler=lr_scheduler,
        max_seq_len=max_seq_len,
        max_seqs_num=max_seqs_num,
        val_ratio=val_ratio,
        on_the_fly=on_the_fly,
        sharded=sharded,
        num_workers=num_workers,
        ckp_dir=ckp_dir,
        project_name=project_name,
        multi_gpu=multi_gpu,
        seed=seed,
    )

    # Log model info
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")

    # Run distillation
    run_distillation(model=model, distill_config=distill_config)


@distill.command("from-yaml")
@click.help_option('--help', '-h')
@click.argument(
    "config",
    default="distill.yaml",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
def from_yaml(config: str):
    """Run distillation from a YAML configuration file.

    Expected YAML structure:
    model: {...}
    distillation: {...}
    resume: {...}

    If resume.is_resume is True, training will resume from the given
    checkpoint using the hyperparameters in the 'distillation' block.
    """
    config = Path(config)

    if config.is_absolute():
        cwd = config.parent
        distill_yaml = config
    else:
        cwd = Path.cwd()
        distill_yaml = cwd / config

    raw = read_yaml(distill_yaml)

    # Allow both nested and flat formats
    model_dict = raw.get("model")
    distill_dict = raw.get("distillation")
    resume_dict = raw.get("resume")

    # Validate and load configs
    model_config = _load_model_config(model_dict)
    distill_config = _load_distill_config(distill_dict)
    resume_config = _load_resume_config(resume_dict)

    # Create student model
    model = ProtX(
        embed_dim=model_config.embed_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        mlp_activation=model_config.mlp_activation,
        use_feature_embedding=model_config.use_feature_embedding,
        feature_window_size=model_config.feature_window_size,
        projection_layer=model_config.projection_layer,
    )

    # Log model info
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")

    # Run distillation
    run_distillation(
        model=model,
        distill_config=distill_config,
        resume_config=resume_config if resume_config.is_resume else None,
    )


@distill.command("get-yaml")
@click.help_option('--help', '-h')
@click.argument(
    "output",
    required=False,
    type=click.Path(dir_okay=True, writable=True, resolve_path=True)
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite if file exists"
)
def get_yaml(output: Optional[str], force: bool):
    """Generate a distillation YAML template.

    If OUTPUT is omitted, the file is saved as ./distill.yaml in the current directory.
    If OUTPUT is a directory, the file will be saved as distill.yaml inside it.
    """

    if output is None:
        output_path = Path.cwd() / "distill.yaml"
    else:
        output_path = Path(output)
        # If directory provided, use default filename
        if output_path.is_dir():
            output_path = output_path / "distill.yaml"

    # Ensure target directory exists
    create_dirs(output_path)

    # Prevent accidental overwrite unless forced
    if output_path.exists() and not force:
        raise click.ClickException(
            f"File already exists: {output_path}. Use --force to overwrite."
        )

    template = (
        "# Distillation configuration for nanoPLM\n"
        "\n"
        "model:\n"
        "  embed_dim: 512\n"
        "  num_layers: 6\n"
        "  num_heads: 8\n"
        "  mlp_activation: \"swiglu\"\n"
        "  use_feature_embedding: False\n"
        "  feature_window_size: 15\n"
        "  projection_layer: True\n"
        "\n"
        "distillation:\n"
        "  # Dataset paths\n"
        "  # Note: these paths are RELATIVE to where you RUN the command NOT the YAML file.\n"
        "  train_fasta: \"output/data/split/train.fasta\"\n"
        "  val_fasta: \"output/data/split/val.fasta\"\n"
        "  train_h5_prefix: \"output/data/kd_dataset/train/train_kd_dataset.h5\"\n"
        "  val_h5_prefix: \"output/data/kd_dataset/val/val_kd_dataset.h5\"\n"
        "\n"
        "  # Output checkpoint path\n"
        "  ckp_dir: \"output/distillation\"\n"
        "\n"
        "  # Training hyperparameters\n"
        "  num_epochs: 10\n"
        "  batch_size: 64\n"
        "  learning_rate: 1e-3\n"
        "  gradient_accumulation_steps: 1\n"
        "  warmup_ratio: 0.05\n"
        "\n"
        "  # LR scheduler\n"
        "  lr_scheduler: \"cosine\"  # cosine, linear, polynomial, constant\n"
        "  lr_scheduler_kwargs: {}\n"
        "\n"
        "  # Dataset config\n"
        "  max_seq_len: 1024\n"
        "  max_seqs_num: 100000\n"
        "  val_ratio: 0.1\n"
        "  on_the_fly: False\n"
        "  sharded: False\n"
        "\n"
        "  # Data loader optimization\n"
        "  use_optimized_loader: True\n"
        "  max_open_files: 5\n"
        "  chunk_size: 32\n"
        "  prefetch_batches: 2\n"
        "  use_threading: True\n"
        "  num_workers: 4\n"
        "\n"
        "  # Checkpointing\n"
        "  project_name: \"nanoplm-distillation\"\n"
        "  logging_steps_percentage: 0.01\n"
        "  eval_steps_percentage: 0.01\n"
        "  save_steps_percentage: 0.05\n"
        "\n"
        "  # Distributed training\n"
        "  multi_gpu: False\n"
        "  world_size: 1\n"
        "  seed: 42\n"
        "\n"
        "resume:\n"
        "  # Set is_resume: true to resume training from a checkpoint\n"
        "  # When resuming, the model, tokenizer, and training state will be loaded from checkpoint_dir\n"
        "  # extra_epochs: adds to 'distillation.num_epochs' to define total epochs.\n"
        "  is_resume: False\n"
        "  checkpoint_dir: \"output/distillation/run-1/checkpoint-1\"\n"
        "  extra_epochs: 0\n"
    )

    # If forcing, remove existing file first
    if output_path.exists() and force:
        output_path.unlink()

    output_path.write_text(template, encoding="utf-8")
    click.echo(f"Template written to: {output_path}")


def _load_model_config(config: Dict[str, Any]) -> StudentModelConfig:
    """Load and validate model configuration from YAML."""
    if config is None:
        raise ValueError("Model configuration is required but not found in YAML")

    expected_keys = set(StudentModelConfig.__annotations__.keys())
    present_keys = set(config.keys())

    extra = []
    kwargs: Dict[str, Any] = {}

    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = config.get(key)
        if value is not None:
            kwargs[key] = value

    if extra:
        raise ValueError(
            f"Unexpected model configuration keys: {', '.join(sorted(extra))}"
        )

    return StudentModelConfig(**kwargs)


def _load_distill_config(config: Dict[str, Any]) -> DistillationConfig:
    """Load and validate distillation configuration from YAML."""
    if config is None:
        raise ValueError("Distillation configuration is required but not found in YAML")

    expected_keys = set(DistillationConfig.__annotations__.keys())
    present_keys = set(config.keys())

    missing = []
    extra = []
    kwargs: Dict[str, Any] = {}

    # Required keys (those without defaults)
    required_keys = {"train_fasta", "val_fasta", "train_h5_prefix", "val_h5_prefix"}

    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = config.get(key)
        if value is not None:
            kwargs[key] = value

    # Check for required keys
    for key in required_keys:
        if key not in kwargs or kwargs[key] is None:
            missing.append(key)

    if missing:
        raise ValueError(
            f"Missing required distillation configuration keys: {', '.join(sorted(missing))}"
        )
    if extra:
        raise ValueError(
            f"Unexpected distillation configuration keys: {', '.join(sorted(extra))}"
        )

    # Handle learning_rate conversion
    if isinstance(kwargs.get('learning_rate'), str):
        try:
            kwargs['learning_rate'] = float(kwargs['learning_rate'])
        except ValueError:
            raise ValueError(f"Invalid learning_rate value: {kwargs['learning_rate']}. Must be a number.")

    # Handle boolean values
    for bool_key in ['multi_gpu', 'on_the_fly', 'sharded', 'use_optimized_loader', 'use_threading']:
        if bool_key in kwargs:
            value = kwargs[bool_key]
            if isinstance(value, bool):
                continue
            elif isinstance(value, str):
                kwargs[bool_key] = value.lower() == 'true'

    return DistillationConfig(**kwargs)


def _load_resume_config(config: Dict[str, Any]) -> ResumeConfig:
    """Load and validate resume configuration from YAML."""
    if config is None:
        return ResumeConfig(is_resume=False, checkpoint_dir="", extra_epochs=None)

    expected_keys = set(ResumeConfig.__annotations__.keys())
    present_keys = set(config.keys())

    extra = []
    kwargs: Dict[str, Any] = {}

    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = config.get(key)
        if value is not None:
            kwargs[key] = value

    checkpoint_dir = kwargs.get("checkpoint_dir", "")
    is_resume = kwargs.get("is_resume", False)

    if is_resume:
        if not checkpoint_dir:
            raise click.ClickException(
                "Resume requested but 'checkpoint_dir' is missing under 'resume'"
            )

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise click.ClickException(
                f"Checkpoint directory does not exist: {checkpoint_dir}"
            )

    return ResumeConfig(**kwargs)
