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
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM, ProtModernBertMLMConfig
from nanoplm.utils.common import read_yaml, create_dirs


@click.group(name="pretrain")
@click.help_option(
    "--help",
    "-h"
)
def pretrain():
    """Group of commands for model pretraining."""
    pass


@pretrain.command("run")
@click.help_option(
    "--help",
    "-h"
)
# Dataset and output
@click.option(
    "--train-fasta",
    type=str,
    required=True,
    help="Training FASTA path"
)
@click.option(
    "--val-fasta",
    type=str,
    required=True,
    help="Validation FASTA path"
)
@click.option(
    "--ckp-dir",
    type=str,
    default="output/pretraining",
    help="Checkpoint directory"
)
# Training hyperparameters
@click.option(
    "--max-length",
    type=int,
    default=1024,
    help="Max sequence length"
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Per-device batch size"
)
@click.option(
    "--num-epochs",
    type=int,
    default=10,
    help="Number of epochs"
)
@click.option(
    "--learning-rate",
    type=float,
    default=3e-6,
    help="Learning rate"
)
@click.option(
    "--weight-decay",
    type=float,
    default=0.0,
    help="Weight decay"
)
@click.option(
    "--warmup-ratio",
    type=float,
    default=0.1,
    help="Warmup ratio"
)
@click.option(
    "--mlm-probability",
    type=float,
    default=0.3,
    help="MLM probability"
)
@click.option(
    "--gradient-accumulation-steps",
    type=int,
    default=1,
    help="Gradient accumulation steps",
)
@click.option(
    "--eval-steps",
    type=int,
    default=100,
    help="Evaluation steps interval"
)
@click.option(
    "--save-steps",
    type=int,
    default=1000,
    help="Checkpoint save steps"
)
@click.option(
    "--seed",
    type=int,
    default=25,
    help="Random seed"
)
@click.option(
    "--mask-replace-prob",
    type=float,
    default=0.8,
    help="Probability of replacing masked tokens with [MASK]",
)
@click.option(
    "--random-token-prob",
    type=float,
    default=0.1,
    help="Probability of replacing masked tokens with random tokens"
)
@click.option(
    "--leave-unchanged-prob",
    type=float,
    default=0.1,
    help="Probability of leaving masked tokens unchanged"
)
# Model hyperparameters (ModernBERT)
@click.option(
    "--hidden-size",
    type=int,
    default=1024,
    help="Model hidden size"
)
@click.option(
    "--intermediate-size",
    type=int,
    default=2048,
    help="Intermediate (FFN) size",
)
@click.option(
    "--num-hidden-layers",
    type=int,
    default=16,
    help="Number of transformer layers",
)
@click.option(
    "--num-attention-heads",
    type=int,
    default=16,
    help="Number of attention heads",
)
@click.option(
    "--mlp-activation",
    type=click.Choice(["swiglu"], case_sensitive=False),
    default="swiglu",
    help="MLP activation",
)
@click.option(
    "--mlp-dropout",
    type=float,
    default=0.0,
    help="MLP dropout"
)
@click.option(
    "--mlp-bias",
    is_flag=True,
    default=False,
    help="Use MLP bias"
)
@click.option(
    "--attention-bias",
    is_flag=True,
    default=False,
    help="Use attn bias"
)
@click.option(
    "--attention-dropout",
    type=float,
    default=0.0,
    help="Attn dropout"
)
@click.option(
    "--classifier-activation",
    type=click.Choice(["relu", "gelu"], case_sensitive=False),
    default="gelu",
    help="Classifier activation",
)
def run(
    # dataset/output
    train_fasta: str,
    val_fasta: str,
    ckp_dir: str,
    # training hp
    max_length: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    mlm_probability: float,
    gradient_accumulation_steps: int,
    eval_steps: int,
    save_steps: int,
    seed: int,
    mask_replace_prob: float,
    random_token_prob: float,
    leave_unchanged_prob: float,
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
    cfg = PretrainingConfig(
        train_fasta=train_fasta,
        val_fasta=val_fasta,
        ckp_dir=ckp_dir,
        max_length=max_length,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        mlm_probability=mlm_probability,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        seed=seed,
        mask_replace_prob=mask_replace_prob,
        random_token_prob=random_token_prob,
        leave_unchanged_prob=leave_unchanged_prob)
    
    model_cfg = ProtModernBertMLMConfig(
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

    model = ProtModernBertMLM(model_cfg)

    run_pretraining(model=model, config=cfg)


@pretrain.command("from-yaml")
@click.help_option(
    "--help",
    "-h"
)
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
def from_yaml(config: str):
    """Run pretraining from a YAML file with training and model parameters.

    Expected YAML structure:
    pretraining: {...}
    model: {...}
    """

    raw = read_yaml(config)

    # Allow both nested and flat formats; prefer nested under key 'training'
    pretrain_dict = raw.get("pretraining")
    model_dict = raw.get("model")

    # validate and load config
    pretrain_config = _load_pretrain_config(pretrain_dict)
    model_config = _load_model_config(model_dict)

    model = ProtModernBertMLM(config=model_config)

    run_pretraining(model=model, config=pretrain_config)


@pretrain.command("get-yaml")
@click.help_option(
    "--help",
    "-h"
)
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
        "# Pretraining configuration for nanoPLM\n"
        "\n"
        "model:\n"
        "  hidden_size: 1024\n"
        "  intermediate_size: 2048\n"
        "  num_hidden_layers: 16\n"
        "  num_attention_heads: 16\n"
        "  vocab_size: 29\n"
        "  mlp_activation: swiglu\n"
        "  mlp_dropout: 0.0\n"
        "  mlp_bias: false\n"
        "  attention_bias: false\n"
        "  attention_dropout: 0.0\n"
        "  classifier_activation: gelu\n"
        "\n"
        "pretraining:\n"
        "  # Dataset\n"
        "  # Note: these paths are RELATIVE to where you RUN the command NOT the YAML file.\n"
        "  train_fasta: data/train.fasta\n"
        "  val_fasta: data/val.fasta\n"
        "\n"
        "  # Output model path\n"
        "  ckp_dir: output/pretraining\n"
        "\n"
        "  # Hyperparameters\n"
        "  max_length: 1024\n"
        "  batch_size: 32\n"
        "  num_epochs: 10\n"
        "  warmup_ratio: 0.05\n"
        "  optimizer: adamw\n" # adamw, stable_adamw
        "  adam_beta1: 0.9\n"
        "  adam_beta2: 0.999\n"
        "  adam_epsilon: 1e-8\n"
        "  learning_rate: 3e-6\n"
        "  weight_decay: 0.0\n"
        "  gradient_accumulation_steps: 1\n"
        "  mlm_probability: 0.3\n"
        "  mask_replace_prob: 0.8\n"
        "  random_token_prob: 0.1\n"
        "  leave_unchanged_prob: 0.1\n"
        "  eval_steps: 100\n"
        "  save_steps: 1000\n"
        "  seed: 42\n"
        "  num_workers: 0\n"
        "  multi_gpu: False\n"
        "  run_name: nanoplm-pretraining\n"
    )

    # If forcing, remove existing file first
    if output_path.exists() and force:
        output_path.unlink()

    output_path.write_text(template, encoding="utf-8")
    click.echo(f"Template written to: {output_path}")

def _load_pretrain_config(d: Dict[str, Any]) -> PretrainingConfig:
    expected_keys = set(PretrainingConfig.__annotations__.keys())
    present_keys = set(d.keys())

    missing = []
    extra = []
    kwargs: Dict[str, Any] = {}

    # Classify provided keys in one pass
    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = d.get(key)
        if value is None:
            missing.append(key)
            continue
        kwargs[key] = value

    # Any expected-but-absent keys are also missing
    for key in expected_keys:
        if key not in present_keys:
            missing.append(key)

    if missing:
        raise ValueError(
            f"Missing required training configuration keys: {', '.join(sorted(missing))}"
        )
    if extra:
        raise ValueError(
            f"Unexpected training configuration keys: {', '.join(sorted(extra))}"
        )

    # Explicitly convert learning_rate to float if it's a string (handles scientific notation)
    if isinstance(kwargs.get('learning_rate'), str):
        try:
            kwargs['learning_rate'] = float(kwargs['learning_rate'])
        except ValueError:
            raise ValueError(f"Invalid learning_rate value: {kwargs['learning_rate']}. Must be a number.")
    
    if isinstance(kwargs.get('multi_gpu'), bool):
        pass
    elif isinstance(kwargs.get('multi_gpu'), str):
        value = kwargs['multi_gpu'].lower()
        if value == 'true':
            kwargs['multi_gpu'] = True
        elif value == 'false':
            kwargs['multi_gpu'] = False
        else:
            raise ValueError(f"Invalid multi_gpu value: {kwargs['multi_gpu']}. [True/False/true/false]")
    else:
        raise ValueError(f"Invalid multi_gpu value: {kwargs['multi_gpu']}. Must be a boolean or string [True/False/true/false]")

    return PretrainingConfig(**kwargs)

def _load_model_config(d: Dict[str, Any]) -> ProtModernBertMLMConfig:
    expected_keys = set(ProtModernBertMLMConfig.__annotations__.keys())
    present_keys = set(d.keys())

    missing = []
    extra = []
    kwargs: Dict[str, Any] = {}

    # Classify provided keys in one pass
    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = d.get(key)
        if value is None:
            missing.append(key)
            continue
        kwargs[key] = value

    # Any expected-but-absent keys are also missing
    for key in expected_keys:
        if key not in present_keys:
            missing.append(key)

    if missing:
        raise ValueError(
            f"Missing required model configuration keys: {', '.join(sorted(missing))}"
        )
    if extra:
        raise ValueError(
            f"Unexpected model configuration keys: {', '.join(sorted(extra))}"
        )

    return ProtModernBertMLMConfig(**kwargs)