#!/usr/bin/env python3
"""
nanoPLM CLI - Pretraining subcommands for MLM pretraining
"""

import click
import random
import math
import os
import numpy as np
import torch
from typing import Optional, Dict, Any, Union
from pathlib import Path

from nanoplm.pretraining.pipeline import (
    PretrainingConfig,
    ResumeConfig,
    run_pretraining,
)
from nanoplm.pretraining.pure_pipeline import run_pure_pretraining
from nanoplm.pretraining.te_pipeline import run_te_pretraining
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM, ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM, TEProtModernBertMLM
from nanoplm.data.validation import validate_pretrain_dataset
from nanoplm.utils.common import read_yaml, create_dirs, is_flash_attention_available
from nanoplm.utils.logger import logger


def _set_seed_for_init(seed: int) -> None:
    """Set seed before model creation so both pipelines start with identical weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_effective_world_size(cfg: PretrainingConfig) -> int:
    if not cfg.multi_gpu:
        return 1
    if cfg.world_size == "auto":
        env_ws = os.environ.get("WORLD_SIZE")
        return int(env_ws) if env_ws else max(torch.cuda.device_count(), 1)
    return int(cfg.world_size) if cfg.world_size else 1


def _populate_batch_setup(cfg: PretrainingConfig) -> None:
    dataset_dir = Path(cfg.dataset_dir)
    validation_result = validate_pretrain_dataset(dataset_dir)
    manifest = validation_result["manifest"]

    effective_world_size = _resolve_effective_world_size(cfg)
    if cfg.global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be > 0, got {cfg.global_batch_size}")

    world_tokens_per_micro_step = (
        cfg.micro_batch_size * manifest.max_seq_len * effective_world_size
    )
    if world_tokens_per_micro_step <= 0:
        raise ValueError(
            f"Invalid token throughput per micro-step: {world_tokens_per_micro_step}. "
            "Check micro_batch_size, max_seq_len, and world_size."
        )

    inferred_grad_accum_steps = max(
        1,
        math.ceil(cfg.global_batch_size / world_tokens_per_micro_step),
    )
    achieved_global_batch_tokens = inferred_grad_accum_steps * world_tokens_per_micro_step
    global_batch_size_samples = (
        inferred_grad_accum_steps * cfg.micro_batch_size * effective_world_size
    )

    cfg.inferred_grad_accum_steps = inferred_grad_accum_steps
    cfg.global_batch_size_samples = global_batch_size_samples
    cfg.achieved_global_batch_tokens = achieved_global_batch_tokens


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
    "--dataset-dir",
    type=str,
    required=True,
    help="Path to dataset directory containing .data_manifest (from nanoplm data from-yaml)"
)
@click.option(
    "--ckp-dir",
    type=str,
    default="output/pretraining",
    help="Checkpoint directory"
)
# Training hyperparameters
@click.option(
    "--micro-batch-size",
    type=int,
    default=32,
    help="Per-device micro-batch size (samples per GPU per forward pass)",
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
    "--global-batch-size",
    type=int,
    default=2 ** 20,
    help="Target tokens per optimizer step (grad_accum inferred automatically)",
)
@click.option(
    "--optimizer",
    type=click.Choice(["adamw", "stable_adamw", "muon", "normuon"], case_sensitive=False),
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
    "--muon-learning-rate",
    type=float,
    default=2e-2,
    help="Muon LR (used only when optimizer=muon or normuon; learning-rate remains AdamW LR)",
)
@click.option(
    "--muon-weight-decay",
    type=float,
    default=0.1,
    help="Muon weight decay (used only when optimizer=muon or normuon)",
)
@click.option(
    "--muon-cautious-weight-decay/--no-muon-cautious-weight-decay",
    default=True,
    help="Enable cautious weight decay in Muon/NorMuon (used only when optimizer=muon or normuon)",
)
@click.option(
    "--muon-use-polar-express/--no-muon-use-polar-express",
    default=False,
    help="Use Polar Express orthogonalization for Muon/NorMuon (used only when optimizer=muon or normuon)",
)
@click.option(
    "--muon-momentum",
    type=float,
    default=0.95,
    help="Muon momentum (used only when optimizer=muon or normuon)",
)
@click.option(
    "--muon-nesterov/--no-muon-nesterov",
    default=True,
    help="Enable Nesterov in Muon (used only when optimizer=muon or normuon)",
)
@click.option(
    "--muon-eps",
    type=float,
    default=1e-7,
    help="Muon epsilon (used only when optimizer=muon or normuon)",
)
@click.option(
    "--mlm-probability",
    type=float,
    default=0.3,
    help="MLM probability"
)
@click.option(
    "--logging-steps",
    type=int,
    default=10,
    help="Number of steps between log events"
)
@click.option(
    "--eval-steps",
    type=int,
    default=50,
    help="Number of steps between evaluations"
)
@click.option(
    "--save-steps",
    type=int,
    default=100,
    help="Number of steps between checkpoint saves"
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
    "--use-packing/--no-packing",
    default=False,
    help="Enable sequence packing to eliminate padding waste (requires flash attention)"
)
@click.option(
    "--bf16/--no-bf16",
    default=True,
    help="Enable mixed precision training (bf16 if supported, fp16 fallback)"
)
@click.option(
    "--tf32/--no-tf32",
    default=True,
    help="Enable TF32 mode on Ampere+ GPUs for faster fp32 matmuls"
)
@click.option(
    "--fp8/--no-fp8",
    default=False,
    help="Enable FP8 Linear matmuls in pure-torch/TE paths (CUDA only, best on H100+)",
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
@click.option(
    "--pure-torch",
    is_flag=True,
    default=False,
    help="Use custom pure-torch model and training loop instead of HF Trainer",
)
@click.option(
    "--pure-te",
    is_flag=True,
    default=False,
    help="Use Transformer Engine model and training loop instead of HF Trainer",
)
def run(
    # dataset/output
    dataset_dir: str,
    ckp_dir: str,
    # training hp
    micro_batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    global_batch_size: int,
    optimizer: str,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    muon_learning_rate: float,
    muon_weight_decay: float,
    muon_cautious_weight_decay: bool,
    muon_use_polar_express: bool,
    muon_momentum: float,
    muon_nesterov: bool,
    muon_eps: float,
    mlm_probability: float,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    seed: int,
    mask_replace_prob: float,
    random_token_prob: float,
    keep_probability: float,
    num_workers: Union[int, str],
    prefetch_factor: int,
    use_packing: bool,
    bf16: bool,
    tf32: bool,
    fp8: bool,
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
    pure_torch: bool,
    pure_te: bool,
):
    """Run MLM pretraining with ModernBERT backbone."""

    # Build config from CLI arguments
    cfg = PretrainingConfig(
        dataset_dir=dataset_dir,
        ckp_dir=ckp_dir,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        global_batch_size=global_batch_size,
        optimizer=optimizer,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        muon_learning_rate=muon_learning_rate,
        muon_weight_decay=muon_weight_decay,
        muon_cautious_weight_decay=muon_cautious_weight_decay,
        muon_use_polar_express=muon_use_polar_express,
        muon_momentum=muon_momentum,
        muon_nesterov=muon_nesterov,
        muon_eps=muon_eps,
        mlm_probability=mlm_probability,
        mask_replace_prob=mask_replace_prob,
        random_token_prob=random_token_prob,
        keep_probability=keep_probability,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        use_packing=use_packing,
        bf16=bf16,
        tf32=tf32,
        fp8=fp8,
        multi_gpu=multi_gpu,
        world_size=world_size,
        project_name=project_name,
    )
    
    _populate_batch_setup(cfg)
    
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

    if pure_torch and pure_te:
        raise click.ClickException("--pure-torch and --pure-te are mutually exclusive.")

    _set_seed_for_init(seed)
    if pure_te:
        logger.info("Using Transformer Engine model and training loop")
        model = TEProtModernBertMLM(model_cfg)
    elif pure_torch:
        logger.info("Using pure-torch model and training loop")
        model = PureProtModernBertMLM(model_cfg)
    else:
        model = ProtModernBertMLM(model_cfg)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")
    logger.info(f"Flash attention available: {is_flash_attention_available()}")

    if pure_te:
        run_te_pretraining(model=model, pretrain_config=cfg)
    elif pure_torch:
        run_pure_pretraining(model=model, pretrain_config=cfg)
    else:
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
@click.option(
    "--pure-torch",
    is_flag=True,
    default=False,
    help="Use custom pure-torch model and training loop instead of HF Trainer",
)
@click.option(
    "--pure-te",
    is_flag=True,
    default=False,
    help="Use Transformer Engine model and training loop instead of HF Trainer",
)
def from_yaml(config: str, pure_torch: bool, pure_te: bool):
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

    # Support pure_torch from YAML or CLI flag (CLI flag takes precedence)
    if not pure_torch:
        pure_torch = bool(raw.get("pure_torch", False))
    if not pure_te:
        pure_te = bool(raw.get("pure_te", False))

    if pure_torch and pure_te:
        raise click.ClickException("pure_torch and pure_te cannot both be true.")

    # validate and load config
    pretrain_config = _load_pretrain_config(pretrain_dict)
    _populate_batch_setup(pretrain_config)
    model_config = _load_model_config(model_dict)
    resume_config = _load_resume_config(resume_dict)

    _set_seed_for_init(pretrain_config.seed)
    if pure_te:
        logger.info("Using Transformer Engine model and training loop")
        model = TEProtModernBertMLM(config=model_config)
    elif pure_torch:
        logger.info("Using pure-torch model and training loop")
        model = PureProtModernBertMLM(config=model_config)
    else:
        model = ProtModernBertMLM(config=model_config)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")
    logger.info(f"Flash attention available: {is_flash_attention_available()}")

    if pure_te:
        run_te_pretraining(
            model=model,
            pretrain_config=pretrain_config,
            resume_config=resume_config if resume_config.is_resume else None,
        )
    elif pure_torch:
        run_pure_pretraining(
            model=model,
            pretrain_config=pretrain_config,
            resume_config=resume_config if resume_config.is_resume else None,
        )
    else:
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
        "#\n"
        "# IMPORTANT: Before running pretraining, ensure you have prepared your data with:\n"
        "#   1. Set pipeline_mode: 'pretrain' in params.yaml\n"
        "#   2. Run: nanoplm data from-yaml\n"
        "# This will generate binary shards and a .data_manifest file.\n"
        "\n"
        "model:\n"
        "  hidden_size: 1024\n"
        "  intermediate_size: 2048\n"
        "  num_hidden_layers: 16\n"
        "  num_attention_heads: 16\n"
        "  vocab_size: 32\n"
        "  mlp_activation: \"swiglu\"\n"
        "  mlp_dropout: 0.0\n"
        "  mlp_bias: false\n"
        "  attention_bias: false\n"
        "  attention_dropout: 0.0\n"
        "  classifier_activation: \"gelu\"\n"
        "\n"
        "pretraining:\n"
        "  # Dataset directory (contains .data_manifest from nanoplm data from-yaml)\n"
        "  # Note: paths are RELATIVE to where you RUN the command, NOT the YAML file.\n"
        "  dataset_dir: \"output/data/pretrain_data\"\n"
        "\n"
        "  # Output model path\n"
        "  ckp_dir: \"output/pretraining_checkpoints\"\n"
        "\n"
        "  # Hyperparameters\n"
        "  #   micro_batch_size: samples per GPU per forward pass (limited by GPU memory)\n"
        "  #   global_batch_size: total tokens per optimizer step across all GPUs\n"
        "  #   gradient_accumulation_steps is inferred automatically:\n"
        "  #     grad_accum = ceil(global_batch_size / (micro_batch_size * max_seq_len * num_gpus))\n"
        "  micro_batch_size: 32\n"
        "  global_batch_size: 1048576  # 2^20 ≈ 1M tokens/step (based on PLM best practices)\n"
        "  num_epochs: 10\n"
        "\n"
        "  optimizer: \"normuon\"  # adamw, stable_adamw, muon, normuon\n"
        "  # AdamW hyperparameters (also used for AdamW side [1D and embedding/unembed params] when optimizer=muon or normuon)\n"
        "  adam_beta1: 0.9\n"
        "  adam_beta2: 0.999\n"
        "  adam_epsilon: 1e-8\n"
        "  learning_rate: 1e-4  # AdamW LR (Muon uses muon_learning_rate)\n"
        "  warmup_ratio: 0.05\n"
        "  weight_decay: 0.0\n"
        "  # Muon/NorMuon hyperparameters (used only when optimizer: muon or normuon)\n"
        "  muon_learning_rate: 1e-3\n"
        "  muon_weight_decay: 0.01\n"
        "  muon_cautious_weight_decay: true\n"
        "  muon_use_polar_express: false\n"
        "  muon_momentum: 0.95\n"
        "  muon_nesterov: true\n"
        "  muon_eps: 1e-7\n"
        "  mlm_probability: 0.3\n"
        "  mask_replace_prob: 0.8\n"
        "  random_token_prob: 0.1\n"
        "  keep_probability: 0.1\n"
        "  logging_steps: 1\n"
        "  eval_steps: 250\n"
        "  save_steps: 5000\n"
        "  seed: 42\n"
        "  num_workers: \"auto\"\n"
        "  prefetch_factor: 2\n"
        "  # Sequence packing: concatenates shorter sequences into fewer rows to eliminate\n"
        "  # padding waste and increase GPU utilization. Requires flash attention and --pure-torch/--pure-te\n"
        "  use_packing: false\n"
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
        "  fp8: true  # Enable FP8 Linear matmuls in pure_torch/pure_te paths (CUDA, best on H100+)\n"
        "\n"
        "  multi_gpu: false\n"
        "  world_size: 1  # Use \"auto\" if you want to use all available GPUs\n"
        "  project_name: \"nanoplm-pretraining\"\n"
        "\n"
        "resume:\n"
        "  # Set is_resume: true to resume training from a checkpoint\n"
        "  # When resuming, the model, tokenizer, and training state will be loaded from checkpoint_dir\n"
        "  # extra_epochs: adds to 'pretraining.num_epochs' to define total epochs.\n"
        "  is_resume: false\n"
        "  checkpoint_dir: \"output/pretraining_checkpoints/run-1/checkpoint-1\"\n"
        "  extra_epochs: 0\n"
        "\n"
        "# Set pure_torch: true to use the custom pure-torch model and training loop\n"
        "# instead of HF Trainer. CLI equivalent: --pure-torch\n"
        "# pure_torch: false\n"
        "# Set pure_te: true to use Transformer Engine model and training loop.\n"
        "# CLI equivalent: --pure-te (mutually exclusive with pure_torch)\n"
        "# pure_te: false\n"
    )

    # If forcing, remove existing file first
    if output_path.exists() and force:
        output_path.unlink()

    output_path.write_text(template, encoding="utf-8")
    click.echo(f"Template written to: {output_path}")

def _load_pretrain_config(config: Dict[str, Any]) -> PretrainingConfig:
    if config is None:
        raise ValueError("Pretraining configuration is required but not found in YAML")

    expected_keys = set(PretrainingConfig.__annotations__.keys())
    present_keys = set(config.keys())

    extra = []
    kwargs: Dict[str, Any] = {}

    if "gradient_accumulation_steps" in present_keys:
        raise ValueError(
            "gradient_accumulation_steps is inferred automatically from "
            "global_batch_size, micro_batch_size, max_seq_len, and world_size. "
            "Remove gradient_accumulation_steps from your pretraining config."
        )

    # Required key
    if 'dataset_dir' not in config or not config['dataset_dir']:
        raise ValueError("dataset_dir is required in pretraining configuration")

    # Classify provided keys in one pass
    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = config.get(key)
        if value is not None:
            kwargs[key] = value

    if extra:
        raise ValueError(
            f"Unexpected training configuration keys: {', '.join(sorted(extra))}"
        )

    # Explicitly convert float-like fields if they are strings (handles scientific notation).
    float_fields = [
        "learning_rate",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "muon_learning_rate",
        "muon_weight_decay",
        "muon_momentum",
        "muon_eps",
        "warmup_ratio",
    ]
    for field in float_fields:
        if isinstance(kwargs.get(field), str):
            try:
                kwargs[field] = float(kwargs[field])
            except ValueError as exc:
                raise ValueError(f"Invalid {field} value: {kwargs[field]}. Must be a number.") from exc

    # Handle boolean values
    for bool_key in [
        'multi_gpu',
        'bf16',
        'tf32',
        'fp8',
        'muon_nesterov',
        'muon_cautious_weight_decay',
        'muon_use_polar_express',
        'use_packing',
    ]:
        if bool_key in kwargs:
            value = kwargs[bool_key]
            if isinstance(value, bool):
                continue
            elif isinstance(value, str):
                kwargs[bool_key] = value.lower() == 'true'

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
