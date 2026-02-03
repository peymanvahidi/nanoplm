#!/usr/bin/env python3
"""
nanoPLM CLI - Distillation subcommands for nanoPLM package
"""

from pathlib import Path
from typing import Any, Dict, Optional

import click

from nanoplm.distillation.models.student import ProtXConfig
from nanoplm.distillation.pipeline import (
    DistillationConfig,
    ResumeConfig,
    run_distillation,
    run_distillation_native,
)
from nanoplm.utils.common import create_dirs, read_yaml


@click.group(name="distill")
@click.help_option('--help', '-h')
def distill():
    """Group of commands for distillation models."""
    pass


@distill.command("run")
@click.help_option('--help', '-h')
@click.option(
    '--dataset-dir',
    type=str,
    required=False,
    help='Path to dataset directory containing .data_manifest (from nanoplm data from-yaml)'
)
@click.option(
    '--train-fasta',
    type=str,
    required=False,
    help='Path to the training FASTA file (required for --on-the-fly mode)'
)
@click.option(
    '--val-fasta',
    type=str,
    required=False,
    help='Path to the validation FASTA file (required for --on-the-fly mode)'
)
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
    '--on-the-fly',
    is_flag=True,
    help='Use on-the-fly teacher embeddings (requires --train-fasta and --val-fasta)'
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
@click.option(
    '--use-native',
    is_flag=True,
    help='Use native PyTorch training loop (no HuggingFace Trainer)'
)
def run(
    dataset_dir: Optional[str],
    train_fasta: Optional[str],
    val_fasta: Optional[str],
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    on_the_fly: bool,
    multi_gpu: bool,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    project_name: str,
    ckp_dir: str,
    lr_scheduler: str,
    no_projection_layer: bool,
    gradient_accumulation_steps: int,
    seed: int,
    use_native: bool,
):
    """Distill the teacher model into a student model.

    Either provide --dataset-dir (reads from manifest) or --on-the-fly with FASTA files.
    """

    # Validate input options
    if on_the_fly:
        if not train_fasta or not val_fasta:
            raise click.ClickException(
                "--train-fasta and --val-fasta are required when using --on-the-fly mode"
            )
    else:
        if not dataset_dir:
            raise click.ClickException(
                "--dataset-dir is required when not using --on-the-fly mode.\n"
                "Run 'nanoplm data from-yaml' with pipeline_mode: 'distillation' first."
            )

    # Create student model config
    model_config = ProtXConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        projection_layer=not no_projection_layer,
    )

    # Create distillation config
    distill_config = DistillationConfig(
        dataset_dir=dataset_dir,
        train_fasta=train_fasta,
        val_fasta=val_fasta,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler=lr_scheduler,
        on_the_fly=on_the_fly,
        num_workers=num_workers,
        ckp_dir=ckp_dir,
        project_name=project_name,
        multi_gpu=multi_gpu,
        seed=seed,
    )

    # Run distillation (choose implementation based on flag)
    train_func = run_distillation_native if use_native else run_distillation
    train_func(
        model_config=model_config,
        distill_config=distill_config,
    )


@distill.command("from-yaml")
@click.help_option('--help', '-h')
@click.argument(
    "config",
    default="distill.yaml",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    '--use-native',
    is_flag=True,
    help='Use native PyTorch training loop (no HuggingFace Trainer). Can also be set in YAML.'
)
def from_yaml(config: str, use_native: bool):
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

    # Check if use_native is set in YAML (CLI flag overrides)
    yaml_use_native = distill_dict.get("use_native", False) if distill_dict else False
    final_use_native = use_native or yaml_use_native

    # Create student model config (field names now match between StudentModelConfig and ProtXConfig)
    model_config = ProtXConfig(
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        mlp_activation=model_config.mlp_activation,
        mlp_dropout=model_config.mlp_dropout,
        mlp_bias=model_config.mlp_bias,
        attention_bias=model_config.attention_bias,
        attention_dropout=model_config.attention_dropout,
        classifier_activation=model_config.classifier_activation,
        projection_layer=model_config.projection_layer,
    )

    # Run distillation (choose implementation based on flag)
    train_func = run_distillation_native if final_use_native else run_distillation
    train_func(
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
        "  # Training implementation\n"
        "  use_native: false  # Set to true for native PyTorch training loop (no HuggingFace Trainer)\n"
        "\n"
        "  # Dataset directory (contains .data_manifest from nanoplm data from-yaml)\n"
        "  # The manifest automatically provides:\n"
        "  #   - max_seq_len, max_seqs_num, val_ratio\n"
        "  #   - on_the_fly mode and dataset paths (FASTA or H5)\n"
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
        "  use_optimized_loader: true\n"
        "  max_open_files: 5\n"
        "  chunk_size: 32\n"
        "  prefetch_batches: 2\n"
        "  use_threading: true\n"
        "  num_workers: 4\n"
        "\n"
        "  # Checkpointing\n"
        "  project_name: \"nanoplm-distillation\"\n"
        "  logging_steps_percentage: 0.01\n"
        "  eval_steps_percentage: 0.01\n"
        "  save_steps_percentage: 0.05\n"
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
