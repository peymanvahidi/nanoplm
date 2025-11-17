#!/usr/bin/env python3
"""
nanoPLM CLI - Pretraining subcommands for MLM pretraining
"""

import click
from typing import Optional, Dict, Any, Union
from pathlib import Path

from nanoplm.pretraining.pipeline import (
    PretrainingConfig,
    ResumeConfig,
    run_pretraining,
)
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM, ProtModernBertMLMConfig
from nanoplm.utils.common import read_yaml, create_dirs, is_flash_attention_available
from nanoplm.utils.logger import logger


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
    help="Training FASTA path"
)
@click.option(
    "--val-fasta",
    type=str,
    help="Validation FASTA path"
)
@click.option(
    "--train-hdf5",
    type=str,
    help="Directory of pre-tokenized training HDF5 shards (used when --lazy-dataset is False)"
)
@click.option(
    "--val-hdf5",
    type=str,
    help="Directory of pre-tokenized validation HDF5 shards (used when --lazy-dataset is False)"
)
@click.option(
    "--load-all-in-memory",
    is_flag=True,
    default=False,
    help="Load all dataset in memory"
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
    "--lazy-dataset",
    is_flag=True,
    default=False,
    help="Use lazy dataset loading"
)
@click.option(
    "--learning-rate",
    type=float,
    default=1e-3,
    help="Maximum Learning rate in the warmup"
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
    default=0.05,
    help="Warmup ratio"
)
@click.option(
    "--optimizer",
    type=click.Choice(["adamw", "stable_adamw"], case_sensitive=False),
    default="adamw",
    help="Optimizer to use"
)
@click.option(
    "--adam-beta1",
    type=float,
    default=0.9,
    help="Adam beta1"
)
@click.option(
    "--adam-beta2",
    type=float,
    default=0.999,
    help="Adam beta2"
)
@click.option(
    "--adam-epsilon",
    type=float,
    default=1e-8,
    help="Adam epsilon"
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
    "--logging-steps-percentage",
    type=float,
    default=0.01,
    help="Fraction of total steps between log events"
)
@click.option(
    "--eval-steps-percentage",
    type=float,
    default=0.025,
    help="Fraction of total steps between evaluations"
)
@click.option(
    "--save-steps-percentage",
    type=float,
    default=0.1,
    help="Fraction of total steps between checkpoint saves"
)
@click.option(
    "--seed",
    type=int,
    default=42,
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
    "--keep-probability",
    type=float,
    default=0.1,
    help="Probability of leaving masked tokens unchanged"
)
@click.option(
    "--num-workers",
    type=Union[int, str],
    default=None,
    help="Number of DataLoader workers. Use 'auto' to use all available CPUs"
)
@click.option(
    "--prefetch-factor",
    type=int,
    default=2,
    help="DataLoader prefetch factor"
)
@click.option(
    "--multi-gpu",
    is_flag=True,
    default=False,
    help="Enable multi-GPU training"
)
@click.option(
    "--world-size",
    type=str,
    default="1",
    help="Total number of processes for distributed training; use 'auto' to use all available GPUs"
)
@click.option(
    "--project-name",
    type=str,
    default="nanoplm-pretraining",
    help="Weights & Biases project name (new runs named run-DDMMHHMM, unique)"
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
    "--vocab-size",
    type=int,
    default=32,
    help="Number of the vocabs being used in the model (should be equal to the vocab size in the tokenizer)"
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
    train_hdf5: str,
    val_hdf5: str,
    load_all_in_memory: bool,
    ckp_dir: str,
    # training hp
    max_length: int,
    batch_size: int,
    num_epochs: int,
    lazy_dataset: bool,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    optimizer: str,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    mlm_probability: float,
    gradient_accumulation_steps: int,
    logging_steps_percentage: float,
    eval_steps_percentage: float,
    save_steps_percentage: float,
    seed: int,
    mask_replace_prob: float,
    random_token_prob: float,
    keep_probability: float,
    num_workers: Union[int, str],
    prefetch_factor: int,
    multi_gpu: bool,
    world_size: str,
    project_name: str,
    # model hp
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    vocab_size: int,
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
        train_hdf5=train_hdf5,
        val_hdf5=val_hdf5,
        load_all_in_memory=load_all_in_memory,
        ckp_dir=ckp_dir,
        max_length=max_length,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lazy_dataset=lazy_dataset,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        optimizer=optimizer,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        mlm_probability=mlm_probability,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mask_replace_prob=mask_replace_prob,
        random_token_prob=random_token_prob,
        keep_probability=keep_probability,
        logging_steps_percentage=logging_steps_percentage,
        eval_steps_percentage=eval_steps_percentage,
        save_steps_percentage=save_steps_percentage,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        multi_gpu=multi_gpu,
        world_size=world_size,
        project_name=project_name,
    )
    
    model_cfg = ProtModernBertMLMConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        vocab_size=vocab_size,
        mlp_activation=mlp_activation,
        mlp_dropout=mlp_dropout,
        mlp_bias=mlp_bias,
        attention_bias=attention_bias,
        attention_dropout=attention_dropout,
        classifier_activation=classifier_activation,
    )

    model = ProtModernBertMLM(model_cfg)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")
    logger.info(f"Flash attention available: {is_flash_attention_available()}")

    run_pretraining(model=model, pretrain_config=cfg)


@pretrain.command("from-yaml")
@click.help_option(
    "--help",
    "-h"
)
@click.argument(
    "config",
    default="pretrain.yaml",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
def from_yaml(config: str):
    """Run pretraining from a YAML file with training and model parameters.

    Expected YAML structure:
    pretraining: {...}
    model: {...}
    resume: {...}

    If resume.is_resume is True, training will resume from the given
    checkpoint using the hyperparameters in the 'pretraining' block.
    """
    config = Path(config)

    if config.is_absolute():
        cwd = config.parent
        pretrain_yaml = config
    else:
        cwd = Path.cwd()
        pretrain_yaml = cwd / config

    raw = read_yaml(pretrain_yaml)

    # Allow both nested and flat formats; prefer nested under key 'training'
    pretrain_dict = raw.get("pretraining")
    model_dict = raw.get("model")
    resume_dict = raw.get("resume")

    # validate and load config
    pretrain_config = _load_pretrain_config(pretrain_dict)
    model_config = _load_model_config(model_dict)
    resume_config = _load_resume_config(resume_dict)

    model = ProtModernBertMLM(config=model_config)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")
    logger.info(f"Flash attention available: {is_flash_attention_available()}")
        
    run_pretraining(
        model=model,
        pretrain_config=pretrain_config,
        resume_config=resume_config if resume_config.is_resume else None,
    )


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
        "  vocab_size: 32\n"
        "  mlp_activation: \"swiglu\"\n"
        "  mlp_dropout: 0.0\n"
        "  mlp_bias: False\n"
        "  attention_bias: False\n"
        "  attention_dropout: 0.0\n"
        "  classifier_activation: \"gelu\"\n"
        "\n"
        "pretraining:\n"
        "  # Dataset\n"
        "  # Note: these paths are RELATIVE to where you RUN the command NOT the YAML file.\n"
        "  train_fasta: \"output/data/split/train.fasta\"\n"
        "  val_fasta: \"output/data/split/val.fasta\"\n"
        "\n"
        "  # Output model path\n"
        "  ckp_dir: \"output/pretraining_checkpoints\"\n"
        "\n"
        "  # Hyperparameters\n"
        "  max_length: 512\n"
        "  batch_size: 32\n"
        "  num_epochs: 10\n"
        "\n"
        "  # Dataset loading strategy:\n"
        "  # - lazy_dataset: True  => tokenize on-the-fly from FASTA\n"
        "  #                          Slower iteration, no preprocessing needed\n"
        "  # - lazy_dataset: False => load pre-tokenized HDF5 shards\n"
        "  #                          Faster iteration, requires preprocessing\n"
        "  # Important: To have the shards, you need to set pretrain_config.enable to True in the params.yaml file\n"
        "  # and run 'nanoplm data from-yaml' to create shards\n"
        "  # or your need to run 'nanoplm data save-pretrain-dataset' using your desired FASTA file as input to create shards\n"
        "  lazy_dataset: False\n"
        "  train_hdf5: \"output/data/pretrain_shards/train_hdf5\"\n"
        "  val_hdf5: \"output/data/pretrain_shards/val_hdf5\"\n"
        "  load_all_in_memory: False\n"
        "\n"
        "  optimizer: \"adamw\" # adamw, stable_adamw\n"
        "  adam_beta1: 0.9\n"
        "  adam_beta2: 0.999\n"
        "  adam_epsilon: 1e-8\n"
        "  learning_rate: 1e-3\n # This is the maximum learning in the warmup phase \n"
        "  warmup_ratio: 0.05\n"
        "  weight_decay: 0.0\n"
        "  gradient_accumulation_steps: 1\n"
        "  mlm_probability: 0.3\n"
        "  mask_replace_prob: 0.8\n"
        "  random_token_prob: 0.1\n"
        "  keep_probability: 0.1\n"
        "  logging_steps_percentage: 0.01 # 100 logging in total \n"
        "  eval_steps_percentage: 0.025 # 40 evaluations in total \n"
        "  save_steps_percentage: 0.1 # 10 saves in total \n"
        "  seed: 42\n"
        "  num_workers: \"auto\"\n"
        "  prefetch_factor: 2\n"
        "  multi_gpu: False\n"
        "  world_size: 1 # Use \"auto\" if you want to use all available GPUs\n"
        "  project_name: \"nanoplm-pretraining\"\n"
        "\n"
        "resume:\n"
        "  # Set is_resume: true to resume training from a checkpoint\n"
        "  # When resuming, the model, tokenizer, and training state will be loaded from checkpoint_dir\n"
        "  # extra_epochs: adds to 'pretraining.num_epochs' to define total epochs.\n"
        "  is_resume: False\n"
        "  checkpoint_dir: \"output/pretraining_checkpoints/run-1/checkpoint-1\"\n"
        "  extra_epochs: 0\n"
    )

    # If forcing, remove existing file first
    if output_path.exists() and force:
        output_path.unlink()

    output_path.write_text(template, encoding="utf-8")
    click.echo(f"Template written to: {output_path}")

def _load_pretrain_config(config: Dict[str, Any]) -> PretrainingConfig:
    expected_keys = set(PretrainingConfig.__annotations__.keys())
    present_keys = set(config.keys())

    missing = []
    extra = []
    kwargs: Dict[str, Any] = {}

    # Classify provided keys in one pass
    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = config.get(key)
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

def _load_model_config(config: Dict[str, Any]) -> ProtModernBertMLMConfig:
    if config is None:
        raise ValueError("Model configuration is required but not found in YAML")

    expected_keys = set(ProtModernBertMLMConfig.__annotations__.keys())
    present_keys = set(config.keys())

    missing = []
    extra = []
    kwargs: Dict[str, Any] = {}

    # Classify provided keys in one pass
    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = config.get(key)
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

def _load_resume_config(config: Dict[str, Any]) -> ResumeConfig:
    if config is None:
        return ResumeConfig(is_resume=False, checkpoint_dir="", extra_epochs=None)

    expected_keys = set(ResumeConfig.__annotations__.keys())
    present_keys = set(config.keys())

    missing = []
    extra = []
    kwargs: Dict[str, Any] = {}

    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = config.get(key)
        if value is None:
            missing.append(key)
            continue
        kwargs[key] = value

    checkpoint_dir = kwargs.get("checkpoint_dir")

    if "extra_epochs" in config:
        kwargs["extra_epochs"] = config.get("extra_epochs")
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