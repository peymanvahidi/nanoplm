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

from nanoplm.pretraining.config import PretrainingConfig, PureTorchConfig, ResumeConfig
from nanoplm.pretraining.pipeline import run_pretraining
from nanoplm.pretraining.pure_pipeline import run_pure_pretraining
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM, ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.utils.common import read_yaml, create_dirs, is_flash_attention_available
from nanoplm.utils.logger import logger

def _check_muon_available(optimizer: str) -> None:
    """Abort early when a Muon variant is requested but ``dion`` is not installed."""
    if optimizer.lower() in {"muon", "normuon"}:
        try:
            import dion  # noqa: F401
        except ImportError:
            raise click.ClickException(
                f"Optimizer '{optimizer}' requires the 'dion' package which is not installed.\n"
                "Install it with:  pip install nanoplm[cuda]"
            )


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
    "--adam-learning-rate",
    type=float,
    default=1e-3,
    help="AdamW learning rate (Muon uses --muon-learning-rate)"
)
@click.option(
    "--adam-weight-decay",
    type=float,
    default=0.0,
    help="AdamW weight decay (Muon uses --muon-weight-decay)"
)
@click.option(
    "--adam-warmup-ratio",
    type=float,
    default=0.05,
    help="AdamW warmup ratio"
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
    "--compile/--no-compile",
    default=True,
    help="Enable torch.compile for faster training (disable for debugging or unsupported hardware)"
)
@click.option(
    "--use-packing/--no-packing",
    default=False,
    help="Enable sequence packing to eliminate padding waste (requires flash attention)"
)
@click.option(
    "--target-packed-rows",
    type=int,
    default=None,
    help="Fixed row count for static-shape compilation (enables dynamic=False). "
         "Set to ceil(micro_batch_size * avg_len / max_seq_len) + margin. "
         "Omit to use dynamic=True."
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
def run(
    # dataset/output
    dataset_dir: str,
    ckp_dir: str,
    # training hp
    micro_batch_size: int,
    num_epochs: int,
    adam_learning_rate: float,
    adam_weight_decay: float,
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
    compile: bool,
    use_packing: bool,
    target_packed_rows: Optional[int],
    bf16: bool,
    tf32: bool,
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
    max_position_embeddings: int,
    pure_torch: bool,
):
    """Run MLM pretraining with ModernBERT backbone."""

    _check_muon_available(optimizer)

    # Build config from CLI arguments
    cfg = PretrainingConfig(
        dataset_dir=dataset_dir,
        ckp_dir=ckp_dir,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        adam_learning_rate=adam_learning_rate,
        adam_weight_decay=adam_weight_decay,
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
        bf16=bf16,
        tf32=tf32,
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
        max_position_embeddings=max_position_embeddings,
    )

    if pure_torch:
        logger.info("Using pure-torch model and training loop")
        _set_seed_for_init(seed)
        model = PureProtModernBertMLM(model_cfg)
    else:
        _set_seed_for_init(seed)
        model = ProtModernBertMLM(model_cfg)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")
    logger.info(f"Flash attention available: {is_flash_attention_available()}")

    if pure_torch:
        pt_cfg = PureTorchConfig(
            use_compile=compile,
            use_packing=use_packing,
            target_packed_rows=target_packed_rows,
        )
        run_pure_pretraining(model=model, pretrain_config=cfg, pure_torch_config=pt_cfg)
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
def from_yaml(config: str):
    f"""Run pretraining from a YAML file with training and model parameters.

    Expected YAML structure::

        model: {...}
        pretraining: {...}
        pure_torch: {...}
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

    # Resolve pure_torch section
    pure_torch, pure_torch_config = _resolve_pure_torch_config(raw)

    # validate and load config
    pretrain_config = _load_pretrain_config(pretrain_dict)
    _check_muon_available(pretrain_config.optimizer)
    model_config = _load_model_config(model_dict)
    resume_config = _load_resume_config(resume_dict)

    if pure_torch:
        logger.info("Using pure-torch model and training loop")
        _set_seed_for_init(pretrain_config.seed)
        model = PureProtModernBertMLM(config=model_config)
    else:
        _set_seed_for_init(pretrain_config.seed)
        model = ProtModernBertMLM(config=model_config)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")
    logger.info(f"Flash attention available: {is_flash_attention_available()}")

    if pure_torch:
        run_pure_pretraining(
            model=model,
            pretrain_config=pretrain_config,
            pure_torch_config=pure_torch_config,
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
        "  max_position_embeddings: 1024 # needs to be at least as long as max seq length\n"
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
        "  warmup_ratio: 0.05\n"
        "\n"
        "  optimizer: \"adamw\"  # adamw, stable_adamw, muon, normuon (muon and normouon only supported with CUDA)\n"
        "  # AdamW hyperparameters (also used for AdamW side [1D and embedding/unembed params] when optimizer=muon or normuon)\n"
        "  adam_learning_rate: 1e-3\n"
        "  adam_weight_decay: 0.0\n"
        "  adam_beta1: 0.9\n"
        "  adam_beta2: 0.999\n"
        "  adam_epsilon: 1e-8\n"
        "  # Muon/NorMuon hyperparameters (used only when optimizer: muon or normuon)\n"
        "  muon_learning_rate: 1e-3\n"
        "  muon_weight_decay: 0.01\n"
        "  muon_cautious_weight_decay: true\n"
        "  muon_use_polar_express: false\n"
        "  muon_momentum: 0.95\n"
        "  muon_nesterov: true\n"
        "  muon_eps: 1e-7\n"
        "\n"
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
        "  multi_gpu: false\n"
        "  world_size: 1  # Use \"auto\" if you want to use all available GPUs\n"
        "  project_name: \"nanoplm-pretraining\"\n"
        "\n"
        "# Pure-torch training loop settings (alternative to HF Trainer).\n"
        "pure_torch:\n"
        "  enabled: false\n"
        "  # torch.compile: compile the model for faster training. Disable for debugging,\n"
        "  # unsupported hardware (e.g. Apple Silicon), or to avoid warmup overhead.\n"
        "  use_compile: true\n"
        "  # Sequence packing: concatenates shorter sequences into fewer rows to eliminate\n"
        "  # padding waste and increase GPU utilization. Requires flash attention.\n"
        "  use_packing: false\n"
        "  # Fixed row count for static-shape compilation when use_packing is true (enables torch.compile dynamic=False).\n"
        "  # Set to ceil(micro_batch_size * avg_len / max_seq_len) + margin. Leave null for dynamic=True.\n"
        "  target_packed_rows: null\n"
        "\n"
        "resume:\n"
        "  # Set is_resume: true to resume training from a checkpoint\n"
        "  # When resuming, the model, tokenizer, and training state will be loaded from checkpoint_dir\n"
        "  # extra_epochs: adds to 'pretraining.num_epochs' to define total epochs.\n"
        "  is_resume: false\n"
        "  checkpoint_dir: \"output/pretraining_checkpoints/run-1/checkpoint-1\"\n"
        "  extra_epochs: 0\n"
    )

    # If forcing, remove existing file first
    if output_path.exists() and force:
        output_path.unlink()

    output_path.write_text(template, encoding="utf-8")
    click.echo(f"Template written to: {output_path}")

def _resolve_pure_torch_config(raw: Dict[str, Any]) -> tuple:
    """Resolve pure_torch configuration from YAML.

    Expected YAML format::

        pure_torch:
          enabled: true
          use_compile: true
          use_packing: false
          target_packed_rows: null

    Returns ``(enabled, PureTorchConfig)``.
    """
    _PURE_TORCH_SUB_KEYS = ("use_compile", "use_packing", "target_packed_rows")

    pure_torch_raw = raw.get("pure_torch")
    pt_kwargs: Dict[str, Any] = {}

    if isinstance(pure_torch_raw, dict):
        enabled = bool(pure_torch_raw.get("enabled", False))
        for key in _PURE_TORCH_SUB_KEYS:
            if key in pure_torch_raw:
                pt_kwargs[key] = pure_torch_raw[key]
    else:
        enabled = False

    # Coerce types
    for bool_key in ("use_compile", "use_packing"):
        if bool_key in pt_kwargs and isinstance(pt_kwargs[bool_key], str):
            pt_kwargs[bool_key] = pt_kwargs[bool_key].lower() == "true"
    if "target_packed_rows" in pt_kwargs:
        val = pt_kwargs["target_packed_rows"]
        if isinstance(val, str):
            pt_kwargs["target_packed_rows"] = int(val)

    return enabled, PureTorchConfig(**pt_kwargs)


def _load_pretrain_config(config: Dict[str, Any]) -> PretrainingConfig:
    if config is None:
        raise ValueError("Pretraining configuration is required but not found in YAML")

    normalized_config = dict(config)
    legacy_aliases = {
        "learning_rate": "adam_learning_rate",
        "weight_decay": "adam_weight_decay",
    }
    for legacy_key, canonical_key in legacy_aliases.items():
        if legacy_key not in normalized_config:
            continue

        legacy_value = normalized_config.get(legacy_key)
        canonical_value = normalized_config.get(canonical_key)

        if canonical_value is not None and legacy_value is not None:
            same_value = legacy_value == canonical_value
            if not same_value:
                try:
                    same_value = float(legacy_value) == float(canonical_value)
                except (TypeError, ValueError):
                    same_value = False
            if not same_value:
                logger.warning(
                    f"Both '{legacy_key}' and '{canonical_key}' are set with different values. "
                    f"Using '{canonical_key}' and ignoring '{legacy_key}'."
                )
        elif canonical_value is None and legacy_value is not None:
            normalized_config[canonical_key] = legacy_value
            logger.warning(
                f"Deprecated key '{legacy_key}' detected; please use '{canonical_key}' instead."
            )

        normalized_config.pop(legacy_key, None)

    expected_keys = set(PretrainingConfig.__annotations__.keys())
    present_keys = set(normalized_config.keys())

    extra = []
    kwargs: Dict[str, Any] = {}

    if "gradient_accumulation_steps" in present_keys:
        raise ValueError(
            "gradient_accumulation_steps is inferred automatically from "
            "global_batch_size, micro_batch_size, max_seq_len, and world_size. "
            "Remove gradient_accumulation_steps from your pretraining config."
        )

    # Required key
    if 'dataset_dir' not in normalized_config or not normalized_config['dataset_dir']:
        raise ValueError("dataset_dir is required in pretraining configuration")

    # Classify provided keys in one pass
    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = normalized_config.get(key)
        if value is not None:
            kwargs[key] = value

    if extra:
        raise ValueError(
            f"Unexpected training configuration keys: {', '.join(sorted(extra))}"
        )

    # Explicitly convert float-like fields if they are strings (handles scientific notation).
    float_fields = [
        "adam_learning_rate",
        "adam_weight_decay",
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
        'muon_nesterov',
        'muon_cautious_weight_decay',
        'muon_use_polar_express',
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

def _set_seed_for_init(seed: int) -> None:
    """Set seed before model creation so both pipelines start with identical weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
