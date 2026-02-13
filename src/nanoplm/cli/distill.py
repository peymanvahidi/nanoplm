#!/usr/bin/env python3
"""
nanoPLM CLI - Distillation subcommands for nanoPLM package
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import click

from nanoplm.distillation.models.student import ProtXConfig
from nanoplm.distillation.pipeline import (
    DistillationConfig,
    ResumeConfig,
    run_distillation
)
from nanoplm.utils.common import create_dirs, read_yaml


@click.group(name="distill")
@click.help_option('--help', '-h')
def distill():
    """Group of commands for distillation models."""
    pass


@distill.command("run")
@click.help_option('--help', '-h')
# Dataset options
@click.option(
    '--dataset-dir',
    type=str,
    required=False,
    help='Path to dataset directory containing .data_manifest (required unless --resume-from is used)'
)
# Model architecture options
@click.option(
    '--hidden-size',
    type=int,
    default=512,
    help='Hidden dimension of the student model'
)
@click.option(
    '--intermediate-size',
    type=int,
    default=1024,
    help='Intermediate (MLP) dimension of the student model'
)
@click.option(
    '--num-hidden-layers',
    type=int,
    default=6,
    help='Number of layers of the student model'
)
@click.option(
    '--num-attention-heads',
    type=int,
    default=8,
    help='Number of attention heads of the student model'
)
@click.option(
    '--mlp-activation',
    type=str,
    default='swiglu',
    help='MLP activation function (swiglu, gelu, relu, etc.)'
)
@click.option(
    '--mlp-dropout',
    type=float,
    default=0.0,
    help='Dropout probability for MLP layers'
)
@click.option(
    '--mlp-bias/--no-mlp-bias',
    default=False,
    help='Whether to use bias in MLP layers'
)
@click.option(
    '--attention-bias/--no-attention-bias',
    default=False,
    help='Whether to use bias in attention layers'
)
@click.option(
    '--attention-dropout',
    type=float,
    default=0.0,
    help='Dropout probability for attention layers'
)
@click.option(
    '--classifier-activation',
    type=str,
    default='gelu',
    help='Activation function for classifier head'
)
@click.option(
    '--projection-layer/--no-projection-layer',
    default=True,
    help='Enable/disable projection layer (disable if student hidden_size matches teacher 1024)'
)
# Training hyperparameters
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
    '--gradient-accumulation-steps',
    type=int,
    default=1,
    help='Gradient accumulation steps'
)
@click.option(
    '--warmup-ratio',
    type=float,
    default=0.05,
    help='Ratio of total training steps used for warmup'
)
# LR scheduler options
@click.option(
    '--lr-scheduler',
    type=click.Choice(['cosine', 'linear', 'polynomial', 'constant']),
    default='cosine',
    help='Learning rate scheduler type'
)
@click.option(
    '--lr-scheduler-kwargs',
    type=str,
    default='{}',
    help='JSON string with additional LR scheduler arguments (e.g., \'{"power": 2}\')'
)
# Data loader optimization options
@click.option(
    '--max-open-files',
    type=int,
    default=5,
    help='Maximum number of open HDF5 files (for optimized loader)'
)
@click.option(
    '--chunk-size',
    type=int,
    default=32,
    help='Chunk size for optimized loader'
)
@click.option(
    '--prefetch-batches',
    type=int,
    default=2,
    help='Number of batches to prefetch'
)
@click.option(
    '--use-threading/--no-threading',
    default=True,
    help='Use threading for data loading'
)
@click.option(
    '--num-workers',
    type=int,
    default=8,
    help='Number of workers to use for data loading'
)
# Checkpointing options
@click.option(
    '--ckp-dir',
    type=str,
    default="output/distillation",
    help='Directory to save checkpoints'
)
@click.option(
    '--project-name',
    type=str,
    default="nanoplm-distillation",
    help='Name of the W&B project'
)
@click.option(
    '--logging-steps',
    type=int,
    default=10,
    help='Number of steps between logging'
)
@click.option(
    '--eval-steps',
    type=int,
    default=50,
    help='Number of steps between evaluations'
)
@click.option(
    '--save-steps',
    type=int,
    default=100,
    help='Number of steps between checkpoint saves'
)
# Mixed precision options
@click.option(
    '--bf16/--no-bf16',
    default=True,
    help='Enable mixed precision training (bf16 if supported, fp16 fallback)'
)
@click.option(
    '--tf32/--no-tf32',
    default=True,
    help='Enable TF32 mode on Ampere+ GPUs for faster fp32 matmuls'
)
# Distributed training options
@click.option(
    '--multi-gpu',
    is_flag=True,
    help='Whether to use multiple GPUs for training'
)
@click.option(
    '--world-size',
    type=str,
    default='1',
    help='Number of GPUs to use (or "auto" for automatic detection)'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed'
)
# Resume options
@click.option(
    '--resume-from',
    type=str,
    default=None,
    help='Path to checkpoint directory to resume training from'
)
@click.option(
    '--extra-epochs',
    type=int,
    default=0,
    help='Additional epochs to train when resuming (added to num-epochs)'
)
def run(
    dataset_dir: Optional[str],
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    mlp_activation: str,
    mlp_dropout: float,
    mlp_bias: bool,
    attention_bias: bool,
    attention_dropout: float,
    classifier_activation: str,
    projection_layer: bool,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    lr_scheduler: str,
    lr_scheduler_kwargs: str,
    max_open_files: int,
    chunk_size: int,
    prefetch_batches: int,
    use_threading: bool,
    num_workers: int,
    bf16: bool,
    tf32: bool,
    ckp_dir: str,
    project_name: str,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    multi_gpu: bool,
    world_size: str,
    seed: int,
    resume_from: Optional[str],
    extra_epochs: int,
):
    """Distill the teacher model into a student model.

    Requires --dataset-dir pointing to a directory with .data_manifest file
    (generated by 'nanoplm data from-yaml' with pipeline_mode: 'distillation').

    When using --resume-from, --dataset-dir is optional as dataset configuration
    is loaded from the checkpoint's training_config.json.
    """

    # Validate: either dataset_dir or resume_from must be provided
    if not dataset_dir and not resume_from:
        raise click.ClickException(
            "--dataset-dir is required unless --resume-from is specified.\n"
            "Run 'nanoplm data from-yaml' with pipeline_mode: 'distillation' first."
        )

    # If resuming without dataset_dir, load it from checkpoint's training_config.json
    if resume_from and not dataset_dir:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise click.ClickException(
                f"Resume checkpoint directory does not exist: {resume_from}"
            )

        # Find the run directory (parent of checkpoint directory)
        # Checkpoint is typically at: ckp_dir/run-name/checkpoint-N
        run_dir = resume_path.parent
        training_config_path = run_dir / "training_config.json"

        if not training_config_path.exists():
            raise click.ClickException(
                f"Cannot resume without --dataset-dir: training_config.json not found in {run_dir}\n"
                "Please provide --dataset-dir explicitly."
            )

        try:
            with open(training_config_path, "r") as f:
                saved_config = json.load(f)

            # Extract dataset_dir from saved config if available
            # The training_config.json stores individual paths, not dataset_dir
            # We need to reconstruct or require dataset_dir
            # For now, check if there's a dataset_dir stored, otherwise use the paths directly
            click.echo(f"Loading dataset configuration from: {training_config_path}")

            # The saved config has: train_fasta, val_fasta, train_h5_prefix, val_h5_prefix
            # We'll pass these to DistillationConfig directly when dataset_dir is not provided
        except (json.JSONDecodeError, IOError) as e:
            raise click.ClickException(
                f"Failed to read training_config.json: {e}\n"
                "Please provide --dataset-dir explicitly."
            )

    # Parse lr_scheduler_kwargs JSON string
    try:
        lr_scheduler_kwargs_dict = json.loads(lr_scheduler_kwargs)
    except json.JSONDecodeError as e:
        raise click.ClickException(
            f"Invalid JSON for --lr-scheduler-kwargs: {e}\n"
            f"Example: --lr-scheduler-kwargs '{{\"power\": 2}}'"
        )

    # Parse world_size (can be int or "auto")
    world_size_value: Union[int, str]
    if world_size.lower() == "auto":
        world_size_value = "auto"
    else:
        try:
            world_size_value = int(world_size)
        except ValueError:
            raise click.ClickException(
                f"Invalid --world-size value: {world_size}. Must be an integer or 'auto'."
            )

    # Create student model config
    model_config = ProtXConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        mlp_activation=mlp_activation,
        mlp_dropout=mlp_dropout,
        mlp_bias=mlp_bias,
        attention_bias=attention_bias,
        attention_dropout=attention_dropout,
        classifier_activation=classifier_activation,
        projection_layer=projection_layer,
    )

    # Create distillation config
    # Base config from CLI arguments
    distill_config_kwargs = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_kwargs": lr_scheduler_kwargs_dict,
        "max_open_files": max_open_files,
        "chunk_size": chunk_size,
        "prefetch_batches": prefetch_batches,
        "use_threading": use_threading,
        "num_workers": num_workers,
        "bf16": bf16,
        "tf32": tf32,
        "ckp_dir": ckp_dir,
        "project_name": project_name,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "multi_gpu": multi_gpu,
        "world_size": world_size_value,
        "seed": seed,
    }

    # Add dataset configuration
    if dataset_dir:
        # Use manifest-based configuration
        distill_config_kwargs["dataset_dir"] = dataset_dir
    elif resume_from:
        # Load dataset paths from checkpoint's training_config.json
        # saved_config was loaded earlier in the validation block
        distill_config_kwargs["train_fasta"] = saved_config.get("train_fasta")
        distill_config_kwargs["val_fasta"] = saved_config.get("val_fasta")
        distill_config_kwargs["train_h5_prefix"] = saved_config.get("train_h5_prefix")
        distill_config_kwargs["val_h5_prefix"] = saved_config.get("val_h5_prefix")
        distill_config_kwargs["max_seq_len"] = saved_config.get("max_seq_len")
        distill_config_kwargs["max_seqs_num"] = saved_config.get("max_seqs_num")
        distill_config_kwargs["val_ratio"] = saved_config.get("val_ratio")
        distill_config_kwargs["on_the_fly"] = saved_config.get("on_the_fly", False)
        distill_config_kwargs["sharded"] = saved_config.get("sharded")
        # Also load train/val sequence counts if available
        distill_config_kwargs["train_sequences"] = saved_config.get("train_sequences")
        distill_config_kwargs["val_sequences"] = saved_config.get("val_sequences")

        # Filter out None values for paths (they might be stored as "None" strings)
        for key in ["train_fasta", "val_fasta", "train_h5_prefix", "val_h5_prefix"]:
            if distill_config_kwargs.get(key) in (None, "None", "null"):
                distill_config_kwargs[key] = None

    distill_config = DistillationConfig(**distill_config_kwargs)

    # Create resume config if resuming
    resume_config = None
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise click.ClickException(
                f"Resume checkpoint directory does not exist: {resume_from}"
            )
        resume_config = ResumeConfig(
            is_resume=True,
            checkpoint_dir=resume_from,
            extra_epochs=extra_epochs if extra_epochs > 0 else None,
        )

    # Run distillation
    run_distillation(
        model_config=model_config,
        distill_config=distill_config,
        resume_config=resume_config,
    )


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
    config_path = Path(config)
    distill_yaml = config_path if config_path.is_absolute() else Path.cwd() / config_path

    raw = read_yaml(distill_yaml)

    # Validate YAML structure
    if not isinstance(raw, dict):
        raise click.ClickException("YAML config must be a dictionary")

    expected_keys = {"model", "distillation"}
    missing_keys = expected_keys - set(raw.keys())
    if missing_keys:
        raise click.ClickException(
            f"Missing required sections in YAML: {', '.join(sorted(missing_keys))}"
        )

    # Extract config sections
    model_dict = raw.get("model")
    distill_dict = raw.get("distillation")
    resume_dict = raw.get("resume")

    # Validate and load configs
    model_config = _load_model_config(model_dict)
    distill_config = _load_distill_config(distill_dict)
    resume_config = _load_resume_config(resume_dict)

    # Run distillation (choose implementation based on YAML setting)
    run_distillation(
        model_config=model_config,
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
        "#\n"
        "# IMPORTANT: Before running distillation, ensure you have prepared your data with:\n"
        "#   1. Set pipeline_mode: 'distillation' in params.yaml\n"
        "#   2. Set distillation_config.on_the_fly in params.yaml:\n"
        "#      - false (default): Pre-compute teacher embeddings during data preparation\n"
        "#      - true: Generate teacher embeddings on-the-fly during training\n"
        "#   3. Run: nanoplm data from-yaml\n"
        "# This will generate a .data_manifest file with the appropriate configuration.\n"
        "\n"
        "model:\n"
        "  hidden_size: 512\n"
        "  intermediate_size: 1024\n"
        "  num_hidden_layers: 6\n"
        "  num_attention_heads: 8\n"
        "  mlp_activation: \"swiglu\"\n"
        "  mlp_dropout: 0.0\n"
        "  mlp_bias: false\n"
        "  attention_bias: false\n"
        "  attention_dropout: 0.0\n"
        "  classifier_activation: \"gelu\"\n"
        "  projection_layer: true  # Set to false if student hidden_size matches teacher (1024)\n"
        "\n"
        "distillation:\n"
        "\n"
        "  # Dataset directory (contains .data_manifest from nanoplm data from-yaml)\n"
        "  # Note: paths are RELATIVE to where you RUN the command, NOT the YAML file.\n"
        "  dataset_dir: \"output/data/distillation_data\"\n"
        "\n"
        "  # Output checkpoint path\n"
        "  ckp_dir: \"output/distillation_checkpoints\"\n"
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
        "  # Data loader optimization\n"
        "  max_open_files: 5\n"
        "  chunk_size: 32\n"
        "  prefetch_batches: 2\n"
        "  use_threading: true\n"
        "  num_workers: 4\n"
        "\n"
        "  # Checkpointing\n"
        "  project_name: \"nanoplm-distillation\"\n"
        "  logging_steps: 10\n"
        "  eval_steps: 50\n"
        "  save_steps: 100\n"
        "\n"
        "  # Mixed precision training (recommended: keep enabled for 1.5-3x speedup)\n"
        "  # When bf16 is true, automatically selects the best precision for your hardware:\n"
        "  #   - CUDA Ampere+ (A100, RTX 3090+): bf16 + TF32\n"
        "  #   - CUDA Volta/Turing (V100, RTX 2080): fp16 fallback\n"
        "  #   - Apple Silicon (M1/M2/M3): fp16 (hardware accelerated)\n"
        "  #   - CPU: fp32 (no mixed precision)\n"
        "  bf16: true\n"
        "  tf32: true  # TF32 mode on Ampere+ CUDA GPUs only (automatically not used on MPS/CPU)\n"
        "             # Provides 3x faster fp32 matmuls with negligible precision loss\n"
        "\n"
        "  # Distributed training\n"
        "  multi_gpu: false\n"
        "  world_size: 1\n"
        "  seed: 42\n"
        "\n"
        "resume:\n"
        "  # Set is_resume: true to resume training from a checkpoint\n"
        "  # When resuming, the model, tokenizer, and training state will be loaded from checkpoint_dir\n"
        "  # extra_epochs: adds to 'distillation.num_epochs' to define total epochs.\n"
        "  is_resume: false\n"
        "  checkpoint_dir: \"output/distillation/run-1/checkpoint-1\"\n"
        "  extra_epochs: 0\n"
    )

    # If forcing, remove existing file first
    if output_path.exists() and force:
        output_path.unlink()

    output_path.write_text(template, encoding="utf-8")
    click.echo(f"Template written to: {output_path}")


def _load_model_config(config: Dict[str, Any]) -> ProtXConfig:
    """Load and validate model configuration from YAML."""
    if config is None:
        raise ValueError("Model configuration is required but not found in YAML")

    expected_keys = set(ProtXConfig.__annotations__.keys())
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

    return ProtXConfig(**kwargs)


def _load_distill_config(config: Dict[str, Any]) -> DistillationConfig:
    """Load and validate distillation configuration from YAML."""
    if config is None:
        raise ValueError("Distillation configuration is required but not found in YAML")

    expected_keys = set(DistillationConfig.__annotations__.keys())
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
            f"Unexpected distillation configuration keys: {', '.join(sorted(extra))}"
        )

    # Validate: dataset_dir is required (manifest provides all dataset configuration)
    if not kwargs.get('dataset_dir'):
        raise ValueError(
            "dataset_dir is required. It should point to a directory containing .data_manifest file.\n"
            "Run 'nanoplm data from-yaml' with pipeline_mode: 'distillation' to generate the manifest."
        )

    # Handle learning_rate conversion
    if isinstance(kwargs.get('learning_rate'), str):
        try:
            kwargs['learning_rate'] = float(kwargs['learning_rate'])
        except ValueError:
            raise ValueError(f"Invalid learning_rate value: {kwargs['learning_rate']}. Must be a number.")

    # Handle boolean values
    bool_keys = [
        'multi_gpu', 'on_the_fly', 'sharded', 'use_threading', 'bf16', 'tf32',
    ]
    for bool_key in bool_keys:
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
