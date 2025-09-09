#!/usr/bin/env python3
"""
nanoPLM CLI - Pretraining subcommands for MLM pretraining
"""

import click
from typing import Optional, Dict, Any
from pathlib import Path

from nanoplm.pretraining.pipeline import (
    PretrainingConfig,
    run_pretraining,
)
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM
from nanoplm.utils.common import read_yaml, create_dirs


@click.group(name="pretrain")
@click.help_option("--help", "-h")
def pretrain():
    """Group of commands for model pretraining."""
    pass


@pretrain.command("run")
@click.help_option("--help", "-h")
# Dataset and output
@click.option("--train-fasta", type=str, required=True, help="Training FASTA path")
@click.option("--val-fasta", type=str, required=True, help="Validation FASTA path")
@click.option("--output-dir", type=str, default=None, help="Output directory")
# Training hyperparameters
@click.option("--max-length", type=int, default=None, help="Max sequence length")
@click.option("--batch-size", type=int, default=None, help="Per-device batch size")
@click.option("--num-epochs", type=int, default=None, help="Number of epochs")
@click.option("--learning-rate", type=float, default=None, help="Learning rate")
@click.option("--weight-decay", type=float, default=None, help="Weight decay")
@click.option("--warmup-ratio", type=float, default=None, help="Warmup ratio")
@click.option("--mlm-probability", type=float, default=None, help="MLM probability")
@click.option(
    "--gradient-accumulation-steps",
    type=int,
    default=None,
    help="Gradient accumulation steps",
)
@click.option("--eval-steps", type=int, default=None, help="Evaluation steps interval")
@click.option("--save-steps", type=int, default=None, help="Checkpoint save steps")
@click.option("--logging-steps", type=int, default=None, help="Logging steps")
@click.option("--seed", type=int, default=None, help="Random seed")
@click.option(
    "--mask-replace-prob",
    type=float,
    default=None,
    help="Probability of replacing masked tokens with [MASK]",
)
@click.option("--random-token-prob", type=float, default=None, help="Probability of replacing masked tokens with random tokens")
@click.option("--leave-unchanged-prob", type=float, default=None, help="Probability of leaving masked tokens unchanged")
@click.option("--mask-token-prob", type=float, default=None, help="Probability of replacing masked tokens with [MASK]")
# Model hyperparameters (ModernBERT)
@click.option("--hidden-size", type=int, default=512, help="Model hidden size")
@click.option(
    "--intermediate-size",
    type=int,
    default=2048,
    help="Intermediate (FFN) size",
)
@click.option(
    "--num-hidden-layers",
    type=int,
    default=6,
    help="Number of transformer layers",
)
@click.option(
    "--num-attention-heads",
    type=int,
    default=8,
    help="Number of attention heads",
)
@click.option(
    "--mlp-activation",
    type=click.Choice(["relu", "gelu", "swiglu"], case_sensitive=False),
    default="swiglu",
    help="MLP activation",
)
@click.option("--mlp-dropout", type=float, default=0.0, help="MLP dropout")
@click.option("--mlp-bias", is_flag=True, default=False, help="Use MLP bias")
@click.option("--attention-bias", is_flag=True, default=False, help="Use attn bias")
@click.option("--attention-dropout", type=float, default=0.0, help="Attn dropout")
@click.option(
    "--classifier-activation",
    type=click.Choice(["relu", "gelu"], case_sensitive=False),
    default="gelu",
    help="Classifier activation",
)
def run(
    # dataset/output
    train_fasta: Optional[str],
    val_fasta: Optional[str],
    output_dir: Optional[str],
    # training hp
    max_length: Optional[int],
    batch_size: Optional[int],
    num_epochs: Optional[int],
    learning_rate: Optional[float],
    weight_decay: Optional[float],
    warmup_ratio: Optional[float],
    mlm_probability: Optional[float],
    gradient_accumulation_steps: Optional[int],
    eval_steps: Optional[int],
    save_steps: Optional[int],
    logging_steps: Optional[int],
    seed: Optional[int],
    mask_replace_prob: Optional[float],
    random_token_prob: Optional[float],
    leave_unchanged_prob: Optional[float],
    # model hp
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
):
    """Run MLM pretraining with ModernBERT backbone."""

    # Build config from CLI arguments
    cfg = PretrainingConfig(train_fasta=train_fasta, val_fasta=val_fasta)

    # Apply CLI overrides if provided
    override_values = {
        "train_fasta": train_fasta,
        "val_fasta": val_fasta,
        "output_dir": output_dir,
        "max_length": max_length,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "mlm_probability": mlm_probability,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "seed": seed,
        "mask_replace_prob": mask_replace_prob,
        "random_token_prob": random_token_prob,
        "leave_unchanged_prob": leave_unchanged_prob,
    }
    for field_name, value in override_values.items():
        if value is not None:
            setattr(cfg, field_name, value)

    # Construct model
    model = ProtModernBertMLM(
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
    )

    # Execute training
    run_pretraining(model=model, config=cfg)


@pretrain.command("from-yaml")
@click.help_option("--help", "-h")
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
def from_yaml(config: str):
    """Run pretraining from a YAML file with training and model parameters.

    Expected YAML structure:
    training: {...}
    model: {...}
    """

    raw = read_yaml(config)

    # Allow both nested and flat formats; prefer nested under key 'training'
    training_block = raw.get("training", raw)

    def _load_config_from_dict(d: Dict[str, Any]) -> PretrainingConfig:
        kwargs = {k: v for k, v in d.items() if k in PretrainingConfig.__annotations__}
        return PretrainingConfig(**kwargs)

    cfg = _load_config_from_dict(training_block)

    # Validate required fields
    if not getattr(cfg, "train_fasta", None) or not getattr(cfg, "val_fasta", None):
        raise click.ClickException(
            "YAML must provide training.train_fasta and training.val_fasta (or top-level train_fasta/val_fasta)."
        )

    # Model parameters with sensible defaults
    model_block = raw.get("model", {})
    allowed_model_keys = {
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "vocab_size": 29,
        "mlp_activation": "swiglu",
        "mlp_dropout": 0.0,
        "mlp_bias": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "classifier_activation": "gelu",
    }

    # Start with defaults then override with YAML-provided values
    model_kwargs: Dict[str, Any] = dict(allowed_model_keys)
    for key in allowed_model_keys.keys():
        if key in model_block and model_block[key] is not None:
            model_kwargs[key] = model_block[key]

    model = ProtModernBertMLM(**model_kwargs)

    run_pretraining(model=model, config=cfg)


@pretrain.command("get-yaml")
@click.help_option("--help", "-h")
@click.argument("output", required=False, type=click.Path(dir_okay=True, writable=True, resolve_path=True))
@click.option("--force", is_flag=True, default=False, help="Overwrite if file exists")
def get_yaml(output: Optional[str], force: bool):
    """Generate a pretraining YAML template.

    If OUTPUT is omitted, the file is saved as ./pretrain.yaml in the current directory.
    If OUTPUT is a directory, the file will be saved as pretrain.yaml inside it.
    """

    if output is None:
        output_path = Path.cwd() / "pretrain.yaml"
    else:
        output_path = Path(output)
        # If directory provided, use default filename
        if output_path.is_dir():
            output_path = output_path / "pretrain.yaml"

    # Ensure target directory exists
    create_dirs(output_path)

    # Prevent accidental overwrite unless forced
    if output_path.exists() and not force:
        raise click.ClickException(
            f"File already exists: {output_path}. Use --force to overwrite."
        )

    template = (
        "# Example pretraining configuration for nanoPLM\n"
        "\n"
        "training:\n"
        "  # Dataset\n"
        "  train_fasta: data/train.fasta\n"
        "  val_fasta: data/val.fasta\n"
        "\n"
        "  # Output\n"
        "  output_dir: output/pretraining\n"
        "\n"
        "  # Training hyperparameters\n"
        "  max_length: 512\n"
        "  batch_size: 32\n"
        "  num_epochs: 3\n"
        "  learning_rate: 0.0005\n"
        "  weight_decay: 0.0\n"
        "  warmup_ratio: 0.0\n"
        "  mlm_probability: 0.15\n"
        "  gradient_accumulation_steps: 1\n"
        "  eval_steps: 500\n"
        "  save_steps: 500\n"
        "  logging_steps: 50\n"
        "  seed: 42\n"
        "  mask_replace_prob: 0.8\n"
        "  random_token_prob: 0.1\n"
        "  leave_unchanged_prob: 0.1\n"
        "\n"
        "model:\n"
        "  hidden_size: 512\n"
        "  intermediate_size: 2048\n"
        "  num_hidden_layers: 6\n"
        "  num_attention_heads: 8\n"
        "  vocab_size: 29\n"
        "  mlp_activation: swiglu\n"
        "  mlp_dropout: 0.0\n"
        "  mlp_bias: false\n"
        "  attention_bias: false\n"
        "  attention_dropout: 0.0\n"
        "  classifier_activation: gelu\n"
    )

    # If forcing, remove existing file first
    if output_path.exists() and force:
        output_path.unlink()

    output_path.write_text(template, encoding="utf-8")
    click.echo(f"Template written to: {output_path}")
