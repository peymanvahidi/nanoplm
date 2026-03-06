#!/usr/bin/env python3
"""
nanoPLM CLI - Pretraining subcommands for MLM pretraining
"""

import click
import random
import math
import os
from dataclasses import MISSING, fields
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path

# Inductor Triton compile workers (used by torch.compile) inherit settings from
# environment variables at process start. Default to safer settings to avoid
# "No valid triton configs / out of resource" from mix-order persistent
# reduction kernels at larger hidden sizes. Users can override by exporting
# these env vars before launching.
os.environ.setdefault("TORCHINDUCTOR_PERSISTENT_REDUCTIONS", "0")
os.environ.setdefault("TORCHINDUCTOR_MIX_ORDER_REDUCTION", "0")

import torch

from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
from nanoplm.pretraining.pipeline import run_pretraining
from nanoplm.pretraining.pure_pipeline import run_pure_pretraining
_TE_IMPORT_ERROR = None
try:
    from nanoplm.pretraining.te_pipeline import run_te_pretraining
except Exception as exc:  # pragma: no cover - depends on TE/FA availability
    _TE_IMPORT_ERROR = exc

    def run_te_pretraining(*_args, **_kwargs):
        raise ImportError(
            "Transformer Engine pretraining requested but unavailable in this environment."
        ) from _TE_IMPORT_ERROR

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM, ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM, TEProtModernBertMLM
from nanoplm.data.validation import validate_pretrain_dataset
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
    default="output/pretraining_checkpoints",
    help="Checkpoint directory"
)
# Training hyperparameters
@click.option(
    "--micro-batch-size",
    type=int,
    default=64,
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
    default=1e-4,
    help="AdamW learning rate (Muon uses --muon-learning-rate)"
)
@click.option(
    "--adam-weight-decay",
    type=float,
    default=0.0,
    help="AdamW weight decay (Muon uses --muon-weight-decay)"
)
@click.option(
    "--gradient-clipping/--no-gradient-clipping",
    default=False,
    help="Enable gradient clipping with max norm 1.0",
)
@click.option(
    "--warmup-steps",
    type=int,
    default=302,
    help="Number of optimizer steps for LR warmup",
)
@click.option(
    "--repo-rope-warmup-steps",
    type=int,
    default=None,
    help="Number of optimizer steps before enabling RePO RoPE offsets (pure-torch only). "
         "If unset, defaults to --warmup-steps.",
)
@click.option(
    "--lr-decay-to-fraction",
    type=float,
    default=0.1,
    help="Fraction of peak learning rate to decay to"
)
@click.option(
    "--lr-schedule",
    type=click.Choice(["linear", "cosine"], case_sensitive=False),
    default="cosine",
    help="Learning rate schedule to use after warmup"
)
@click.option(
    "--global-batch-size",
    type=int,
    default=256000,
    help="Target tokens per optimizer step (grad_accum inferred automatically)",
)
@click.option(
    "--optimizer",
    type=click.Choice(["adamw", "stable_adamw", "muon", "normuon"], case_sensitive=False),
    default="normuon",
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
    default=1e-3,
    help="Muon LR (used only when optimizer=muon or normuon; learning-rate remains AdamW LR)",
)
@click.option(
    "--muon-weight-decay",
    type=float,
    default=0.01,
    help="Muon weight decay (used only when optimizer=muon or normuon)",
)
@click.option(
    "--muon-cautious-weight-decay/--no-muon-cautious-weight-decay",
    default=True,
    help="Enable cautious weight decay in Muon/NorMuon (used only when optimizer=muon or normuon)",
)
@click.option(
    "--muon-use-polar-express/--no-muon-use-polar-express",
    default=True,
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
    default=1,
    help="Number of steps between log events"
)
@click.option(
    "--eval-steps",
    type=int,
    default=250,
    help="Number of steps between evaluations"
)
@click.option(
    "--save-steps",
    type=int,
    default=5000,
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
    default="auto",
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
    default=True,
    help="Enable sequence packing to eliminate padding waste (requires flash attention)"
)
@click.option(
    "--use-static-inp-size/--no-use-static-inp-size",
    default=True,
    help="When packing is enabled, use fixed flat token size + bucketed cu_seqlens/max_seqlen for static-shape execution",
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
    default=True,
    help="Enable multi-GPU training"
)
@click.option(
    "--world-size",
    type=str,
    default="auto",
    help="Total number of processes for distributed training; use 'auto' to use all available GPUs"
)
@click.option(
    "--project-name",
    type=str,
    default="nanoplm-pretraining",
    help="Weights & Biases project name (new runs named run-DDMMHHMM, unique)"
)
# Resume options
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume training from a checkpoint",
)
@click.option(
    "--resume-checkpoint-dir",
    type=str,
    default="",
    help="Checkpoint directory to resume from",
)
@click.option(
    "--resume-extra-epochs",
    type=int,
    default=None,
    help="Extra epochs to add on top of the original/current run length when resuming",
)
@click.option(
    "--resume-warmup-steps",
    type=int,
    default=None,
    help="Override warmup steps for resumed training",
)
@click.option(
    "--resume-learning-rate",
    type=float,
    default=None,
    help="Override AdamW learning rate when resuming",
)
@click.option(
    "--resume-muon-learning-rate",
    type=float,
    default=None,
    help="Override Muon learning rate when resuming",
)
@click.option(
    "--resume-lr-schedule",
    type=click.Choice(["linear", "cosine"], case_sensitive=False),
    default=None,
    help="Override LR schedule when resuming",
)
@click.option(
    "--resume-lr-decay-to-fraction",
    type=float,
    default=None,
    help="Override LR decay floor fraction when resuming",
)
@click.option(
    "--resume-reset-scheduler/--no-resume-reset-scheduler",
    default=False,
    help="Rebuild the LR scheduler from step 0 over the remaining steps when resuming",
)
@click.option(
    "--resume-skip-warmup/--no-resume-skip-warmup",
    default=False,
    help="Skip warmup on resume and jump directly to peak LR",
)
# Model hyperparameters (ModernBERT)
@click.option(
    "--hidden-size",
    type=int,
    default=768,
    help="Model hidden size"
)
@click.option(
    "--intermediate-size",
    type=int,
    default=1536,
    help="Intermediate (FFN) size",
)
@click.option(
    "--num-hidden-layers",
    type=int,
    default=12,
    help="Number of transformer layers",
)
@click.option(
    "--num-attention-heads",
    type=int,
    default=8,
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
    "--no-mlp-on-first-layer/--mlp-on-first-layer",
    default=True,
    help="Disable the MLP branch in encoder layer 0",
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
    "--use-resid-lambdas/--no-use-resid-lambdas",
    default=True,
    help="Enable per-layer residual scaling (resid_lambdas). Not compatible with --use-mhc-lite.",
)
@click.option(
    "--use-x0-lambdas/--no-use-x0-lambdas",
    default=True,
    help="Enable per-layer x0 shortcut scaling (x0_lambdas)",
)
@click.option(
    "--use-qk-norm/--no-use-qk-norm",
    default=False,
    help="Enable NanoChat-style RMS QK normalization in attention",
)
@click.option(
    "--use-canon-layers/--no-use-canon-layers",
    default=True,
    help="Enable bidirectional Canon-ABCD local mixing layers (pure-torch path only)",
)
@click.option(
    "--canon-layers-mode",
    type=str,
    default="ac",
    help="Subset of Canon insertion points to enable (A/B/C/D), e.g. 'abcd' or 'ac'",
)
@click.option(
    "--canon-layers-kernel-size",
    type=int,
    default=None,
    help="Canon kernel size. If omitted, defaults to 5. Allowed values: {3,5,7}.",
)
@click.option(
    "--use-repo/--no-use-repo",
    default=False,
    help="Enable RePO: learned per-head positions replacing fixed RoPE indices (pure-torch only)",
)
@click.option(
    "--repo-after-n-layers",
    type=int,
    default=3,
    help="First N layers keep standard RoPE; layers after use RePO (only when --use-repo)",
)
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    default=False,
    help="Enable activation checkpointing (recompute transformer layers in backward to save VRAM).",
)
@click.option(
    "--gradient-checkpointing-mode",
    type=click.Choice(["layer", "attn", "attn+mlp"], case_sensitive=False),
    default="layer",
    show_default=True,
    help="Checkpoint scope. 'layer' checkpoints the whole transformer layer; "
         "'attn' checkpoints only the attention residual branch (usually the best tradeoff); "
         "'attn+mlp' checkpoints both attention and MLP branches.",
)
@click.option(
    "--use-mhc-lite/--no-use-mhc-lite",
    default=False,
    help="Enable mHC-lite: multi-stream residual with doubly stochastic mixing (pure-torch only). "
         "Not compatible with --use-resid-lambdas.",
)
@click.option(
    "--mhc-n-streams",
    type=int,
    default=4,
    help="Number of residual streams for mHC-lite (uses n! permutation matrices)",
)
@click.option(
    "--mhc-lite-wrapping-level",
    type=click.Choice(["layer", "sublayers"], case_sensitive=False),
    default="layer",
    show_default=True,
    help="mHC-lite wrapping level (pure-torch only): 'layer' wraps the full transformer layer; "
         "'sublayers' wraps attention and MLP residual branches separately.",
)
@click.option(
    "--use-compile-max-autotune/--no-use-compile-max-autotune",
    default=False,
    help="Pure-torch only: compile with torch.compile(mode='max-autotune-no-cudagraphs'). "
         "May improve throughput, but increases compile time significantly at run start.",
)
@click.option(
    "--compile-triton-persistent-reductions/--no-compile-triton-persistent-reductions",
    default=False,
    help="Pure-torch only: TorchInductor Triton setting. Disable to avoid shared-memory "
         "resource errors in some fused reduction kernels (e.g. LayerNorm backward at large hidden dims) "
         "while keeping torch.compile enabled.",
)
@click.option(
    "--compile-triton-mix-order-reduction/--no-compile-triton-mix-order-reduction",
    default=False,
    help="Pure-torch only: TorchInductor Triton setting. Mix-order reductions can generate "
         "persistent reduction kernels that exceed shared memory limits on some shapes. "
         "Disable to prefer legacy reductions while keeping torch.compile enabled.",
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
    adam_learning_rate: float,
    adam_weight_decay: float,
    gradient_clipping: bool,
    warmup_steps: int,
    repo_rope_warmup_steps: Optional[int],
    lr_decay_to_fraction: float,
    lr_schedule: str,
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
    use_static_inp_size: bool,
    use_compile_max_autotune: bool,
    compile_triton_persistent_reductions: bool,
    compile_triton_mix_order_reduction: bool,
    bf16: bool,
    tf32: bool,
    fp8: bool,
    multi_gpu: bool,
    world_size: str,
    project_name: str,
    resume: bool,
    resume_checkpoint_dir: str,
    resume_extra_epochs: Optional[int],
    resume_warmup_steps: Optional[int],
    resume_learning_rate: Optional[float],
    resume_muon_learning_rate: Optional[float],
    resume_lr_schedule: Optional[str],
    resume_lr_decay_to_fraction: Optional[float],
    resume_reset_scheduler: bool,
    resume_skip_warmup: bool,
    # model hp
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    vocab_size: int,
    mlp_activation: str,
    mlp_dropout: float,
    mlp_bias: bool,
    no_mlp_on_first_layer: bool,
    attention_bias: bool,
    attention_dropout: float,
    classifier_activation: str,
    use_resid_lambdas: bool,
    use_x0_lambdas: bool,
    use_qk_norm: bool,
    use_canon_layers: bool,
    canon_layers_mode: str,
    canon_layers_kernel_size: Optional[int],
    use_repo: bool,
    repo_after_n_layers: int,
    gradient_checkpointing: bool,
    gradient_checkpointing_mode: str,
    use_mhc_lite: bool,
    mhc_n_streams: int,
    mhc_lite_wrapping_level: str,
    pure_torch: bool,
    pure_te: bool,
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
        max_grad_norm=1.0 if gradient_clipping else float("inf"),
        warmup_steps=warmup_steps,
        repo_rope_warmup_steps=repo_rope_warmup_steps,
        lr_decay_to_fraction=lr_decay_to_fraction,
        lr_schedule=lr_schedule,
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
        use_static_inp_size=use_static_inp_size,
        use_compile_max_autotune=use_compile_max_autotune,
        compile_triton_persistent_reductions=compile_triton_persistent_reductions,
        compile_triton_mix_order_reduction=compile_triton_mix_order_reduction,
        bf16=bf16,
        tf32=tf32,
        fp8=fp8,
        multi_gpu=multi_gpu,
        world_size=world_size,
        project_name=project_name,
    )
    
    if pure_torch and pure_te:
        raise click.ClickException("--pure-torch and --pure-te are mutually exclusive.")
    if use_canon_layers and not pure_torch:
        raise click.ClickException(
            "use_canon_layers requires --pure-torch. "
            "Canon layers are not implemented in HF/TE paths."
        )
    if (use_mhc_lite or mhc_lite_wrapping_level.lower() != "layer") and not pure_torch:
        raise click.ClickException(
            "mHC-lite is only supported in the pure-torch pipeline. "
            "Set --pure-torch (or disable mHC-lite settings)."
        )
    if use_mhc_lite and use_resid_lambdas:
        raise click.ClickException(
            "use_mhc_lite and use_resid_lambdas are mutually exclusive. "
            "resid_lambdas scales the hidden state before each layer, which breaks "
            "mHC-lite's doubly-stochastic stability guarantees."
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
        no_mlp_on_first_layer=no_mlp_on_first_layer,
        attention_bias=attention_bias,
        attention_dropout=attention_dropout,
        classifier_activation=classifier_activation,
        use_resid_lambdas=use_resid_lambdas,
        use_x0_lambdas=use_x0_lambdas,
        use_qk_norm=use_qk_norm,
        use_canon_layers=use_canon_layers,
        canon_layers_mode=canon_layers_mode,
        canon_layers_kernel_size=canon_layers_kernel_size,
        use_repo=use_repo,
        repo_after_n_layers=repo_after_n_layers,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_mode=gradient_checkpointing_mode.lower(),
        use_mhc_lite=use_mhc_lite,
        mhc_n_streams=mhc_n_streams,
        mhc_lite_wrapping_level=mhc_lite_wrapping_level.lower(),
    )

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

    resume_cfg = ResumeConfig(
        is_resume=resume,
        checkpoint_dir=resume_checkpoint_dir,
        extra_epochs=resume_extra_epochs,
        warmup_steps=resume_warmup_steps,
        learning_rate=resume_learning_rate,
        muon_learning_rate=resume_muon_learning_rate,
        lr_schedule=resume_lr_schedule,
        lr_decay_to_fraction=resume_lr_decay_to_fraction,
        reset_scheduler=resume_reset_scheduler,
        skip_warmup=resume_skip_warmup,
    )
    if resume_cfg.is_resume and not resume_cfg.checkpoint_dir:
        raise click.ClickException("--resume requires --resume-checkpoint-dir")

    if pure_te:
        run_te_pretraining(
            model=model,
            pretrain_config=cfg,
            resume_config=resume_cfg if resume_cfg.is_resume else None,
        )
    elif pure_torch:
        run_pure_pretraining(
            model=model,
            pretrain_config=cfg,
            resume_config=resume_cfg if resume_cfg.is_resume else None,
        )
    else:
        run_pretraining(
            model=model,
            pretrain_config=cfg,
            resume_config=resume_cfg if resume_cfg.is_resume else None,
        )


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

        model: {...}
        pretraining: {...}
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
    _check_muon_available(pretrain_config.optimizer)
    
    model_config = _load_model_config(model_dict)
    resume_config = _load_resume_config(resume_dict)
    if model_config.use_canon_layers and not pure_torch:
        raise click.ClickException(
            "model.use_canon_layers=true requires pure_torch: true (or --pure-torch). "
            "Canon layers are not implemented in HF/TE paths."
        )
    if (
        model_config.use_mhc_lite
        or str(getattr(model_config, "mhc_lite_wrapping_level", "layer")).lower() != "layer"
    ) and not pure_torch:
        raise click.ClickException(
            "model.use_mhc_lite=true (or model.mhc_lite_wrapping_level != 'layer') "
            "requires pure_torch: true (or --pure-torch). "
            "mHC-lite is not implemented in HF/TE paths."
        )
    if model_config.use_mhc_lite and model_config.use_resid_lambdas:
        raise click.ClickException(
            "model.use_mhc_lite=true is not compatible with model.use_resid_lambdas=true. "
            "resid_lambdas scales the hidden state before each layer, which breaks "
            "mHC-lite's doubly-stochastic stability guarantees."
        )
    _populate_batch_setup(pretrain_config)

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

    # Define the YAML template
    # TODO: there are some params that are currently in the pure_torch, let's move all of them to the pretrain params
    template = (
        "# Pretraining configuration for nanoPLM\n"
        "#\n"
        "# IMPORTANT: Before running pretraining, ensure you have prepared your data with:\n"
        "#   1. Set pipeline_mode: 'pretrain' in params.yaml\n"
        "#   2. Run: nanoplm data from-yaml\n"
        "# This will generate binary shards and a .data_manifest file.\n"
        "\n"
        "model:\n"
        "  hidden_size: 768\n"
        "  intermediate_size: 1536\n"
        "  num_hidden_layers: 12\n"
        "  num_attention_heads: 8\n"
        "  vocab_size: 32\n"
        "  mlp_activation: \"swiglu\"\n"
        "  mlp_dropout: 0.0\n"
        "  mlp_bias: false\n"
        "  no_mlp_on_first_layer: true\n"
        "  attention_bias: false\n"
        "  attention_dropout: 0.0\n"
        "  classifier_activation: \"gelu\"\n"
        "  # The options below only work on pure-torch and TE pipelines unless noted\n"
        "  use_resid_lambdas: false  # scales residual stream per layer (not compatible with use_mhc_lite)\n"
        "  use_x0_lambdas: false  # blends initial embedding x0 per layer\n"
        "  use_qk_norm: false  # applies RMS norm to Q/K in attention\n"
        "  use_canon_layers: false # enables Canon-ABCD local mixing layers (pure_torch only)\n"
        "  canon_layers_mode: \"ac\"  # subset of Canon sites: A/B/C/D (e.g. \"ac\" for lighter mode)\n"
        "  canon_layers_kernel_size: 5  # symmetric Canon kernel size (allowed: 3/5/7, default: 5)\n"
        "  use_repo: false  # RePO: learned per-head positions replacing fixed RoPE (pure_torch only)\n"
        "  repo_after_n_layers: 3  # first N layers keep standard RoPE, layers after use RePO\n"
        "  use_mhc_lite: false  # mHC-lite: multi-stream residual with doubly stochastic mixing (pure_torch only)\n"
        "  mhc_n_streams: 4  # number of residual streams for mHC-lite (n! permutation matrices)\n"
        "  mhc_lite_wrapping_level: \"layer\"  # mHC-lite wrapping: 'layer' or 'sublayers' (pure_torch only)\n"
        "  mhc_triton_fused: true  # use fused Triton kernels for mHC-lite stream ops; first run will start slow due to Triton autotune\n"
        "\n"
        "pretraining:\n"
        "  dataset_dir: \"output/data/pretrain_data\"\n"
        "  ckp_dir: \"output/pretraining_checkpoints\"\n"
        "  micro_batch_size: 64\n"
        "  global_batch_size: 256000\n"
        "  num_epochs: 10\n"
        "  optimizer: \"adamw\"\n"
        "  # AdamW hyperparameters (also used for AdamW side [1D and embedding/unembed params] when optimizer=muon or normuon)\n"
        "  adam_beta1: 0.9\n"
        "  adam_beta2: 0.999\n"
        "  adam_epsilon: 1e-8\n"
        "  adam_learning_rate: 1e-4\n"
        "  max_grad_norm: .inf\n"
        "  warmup_steps: 302\n"
        "  repo_rope_warmup_steps: 302\n"
        "  lr_decay_to_fraction: 0.1\n"
        "  lr_schedule: \"cosine\"\n"
        "  adam_weight_decay: 0.0\n"
        "  muon_learning_rate: 1e-3\n"
        "  muon_weight_decay: 0.01\n"
        "  muon_cautious_weight_decay: true\n"
        "  muon_use_polar_express: true\n"
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
        "  use_packing: true\n"
        "  use_static_inp_size: true\n"
        "  use_compile_max_autotune: false  # pure_torch only: may improve throughput, but causes long compile/autotune time at run start\n"
        "  compile_triton_persistent_reductions: false  # pure_torch only: set true if it works on your shapes/GPU and you want the faster persistent reductions\n"
        "  compile_triton_mix_order_reduction: false  # pure_torch only: set true to enable mix-order reductions (may be faster, but can hit shared-mem limits)\n"
        "\n"
        "  # Mixed precision training (recommended: keep enabled for 1.5-3x speedup)\n"
        "  # When bf16 is true, automatically selects the best precision for your hardware:\n"
        "  #   - CUDA Ampere+ (A100, RTX 3090+): bf16 + TF32\n"
        "  #   - CUDA Volta/Turing (V100, RTX 2080): fp16 fallback\n"
        "  #   - Apple Silicon (M1/M2/M3): fp16 (hardware accelerated)\n"
        "  #   - CPU: fp32 (no mixed precision)\n"
        "  bf16: true\n"
        "  tf32: true\n"
        "  fp8: false\n"
        "  multi_gpu: true\n"
        "  world_size: 'auto'\n"
        "  project_name: \"nanoplm-pretraining\"\n"
        "  profiler_enabled: false\n"
        "  profiler_start_step: 10\n"
        "  profiler_end_step: 15\n"
        "\n"
        "\n"
        "resume:\n"
        "  is_resume: false\n"
        "  checkpoint_dir: \"output/pretraining_checkpoints/run-1/checkpoint-1\"\n"
        "  extra_epochs: 0\n"
        "  warmup_steps: null\n"
        "  learning_rate: null\n"
        "  muon_learning_rate: null\n"
        "  lr_schedule: null\n"
        "  lr_decay_to_fraction: null\n"
        "  reset_scheduler: false\n"
        "  skip_warmup: false\n"
        "\n"
        "# pure_torch: false\n"
        "# pure_te: false\n"
    )

    # Prevent accidental overwrite unless forced
    if output_path.exists() and not force:
        raise click.ClickException(
            f"File already exists: {output_path}. Use --force to overwrite."
        )

    # Write the template to the file
    output_path.write_text(template, encoding="utf-8")
    click.echo(f"Template written to: {output_path}")

def _load_pretrain_config(config: Dict[str, Any]) -> PretrainingConfig:
    if config is None:
        raise ValueError("Pretraining configuration is required but not found in YAML")

    normalized_config = dict(config)

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
        "max_grad_norm",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "muon_learning_rate",
        "muon_weight_decay",
        "muon_momentum",
        "muon_eps",
        "min_lr",
    ]

    def _parse_float_like(raw_value: Any, field_name: str) -> float:
        if not isinstance(raw_value, str):
            return float(raw_value)

        stripped = raw_value.strip().lower()
        if stripped in {
            'float("inf")',
            "float('inf')",
            "inf",
            "+inf",
            "infinity",
            "+infinity",
            ".inf",
        }:
            return float("inf")
        if stripped in {
            'float("-inf")',
            "float('-inf')",
            "-inf",
            "-infinity",
            "-.inf",
        }:
            return float("-inf")
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name} value: {raw_value}. Must be a number.") from exc

    for field in float_fields:
        if field in kwargs and kwargs[field] is not None:
            kwargs[field] = _parse_float_like(kwargs[field], field)

    # Ensure warmup-related fields are ints (YAML may load as int or float).
    for warmup_key in ("warmup_steps", "repo_rope_warmup_steps"):
        if warmup_key in kwargs and kwargs[warmup_key] is not None:
            try:
                kwargs[warmup_key] = int(kwargs[warmup_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {warmup_key} value: {kwargs[warmup_key]}. Must be an integer."
                ) from exc

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
        'use_static_inp_size',
        'use_compile_max_autotune',
        'compile_triton_persistent_reductions',
        'compile_triton_mix_order_reduction',
        'profiler_enabled',
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

    model_fields = {f.name: f for f in fields(ProtModernBertMLMConfig)}
    expected_keys = set(model_fields.keys())
    present_keys = set(config.keys())

    if "canon_layer_type" in present_keys:
        legacy_canon_layer_type = config.get("canon_layer_type")
        if legacy_canon_layer_type is not None:
            legacy_value = str(legacy_canon_layer_type).strip().lower()
            if legacy_value == "causal":
                raise ValueError(
                    "model.canon_layer_type='causal' is no longer supported. "
                    "Causal Canon layers were removed. Delete model.canon_layer_type "
                    "and use the symmetric Canon layer (kernel sizes: 3/5/7)."
                )
            if legacy_value != "symmetric":
                raise ValueError(
                    f"model.canon_layer_type={legacy_canon_layer_type!r} is no longer supported. "
                    "Delete model.canon_layer_type; Canon layers are now symmetric-only."
                )
        logger.warning(
            "Ignoring deprecated model.canon_layer_type; Canon layers are now symmetric-only."
        )

    extra = []
    kwargs: Dict[str, Any] = {}

    # Classify provided keys in one pass
    for key in present_keys:
        if key == "canon_layer_type":
            continue
        if key not in expected_keys:
            extra.append(key)
            continue
        kwargs[key] = config.get(key)
    if extra:
        raise ValueError(
            f"Unexpected model configuration keys: {', '.join(sorted(extra))}"
        )

    missing_required = []
    normalized_kwargs: Dict[str, Any] = {}
    for key, f in model_fields.items():
        if key in kwargs and kwargs[key] is not None:
            value = kwargs[key]
        elif f.default is not MISSING:
            value = f.default
        elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
            value = f.default_factory()  # type: ignore[misc]
        else:
            missing_required.append(key)
            continue

        if f.type is bool and isinstance(value, str):
            value = value.lower() == "true"
        normalized_kwargs[key] = value

    if missing_required:
        raise ValueError(
            f"Missing required model configuration keys: {', '.join(sorted(missing_required))}"
        )

    try:
        return ProtModernBertMLMConfig(**normalized_kwargs)
    except TypeError:
        raise
    except ValueError as exc:
        # Provide a clearer error for common config incompatibilities.
        raise ValueError(f"Invalid model configuration: {exc}") from exc

def _load_resume_config(config: Dict[str, Any]) -> ResumeConfig:
    if config is None:
        return ResumeConfig(is_resume=False, checkpoint_dir="")

    expected_keys = set(ResumeConfig.__annotations__.keys())
    present_keys = set(config.keys())

    extra = []
    kwargs: Dict[str, Any] = {
        "is_resume": False,
        "checkpoint_dir": "",
    }

    def _parse_float_like(raw_value: Any, field_name: str) -> float:
        if not isinstance(raw_value, str):
            return float(raw_value)

        stripped = raw_value.strip().lower()
        if stripped in {
            'float("inf")',
            "float('inf')",
            "inf",
            "+inf",
            "infinity",
            "+infinity",
            ".inf",
        }:
            return float("inf")
        if stripped in {
            'float("-inf")',
            "float('-inf')",
            "-inf",
            "-infinity",
            "-.inf",
        }:
            return float("-inf")
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name} value: {raw_value}. Must be a number.") from exc

    for key in present_keys:
        if key not in expected_keys:
            extra.append(key)
            continue
        value = config.get(key)
        # Keep None values for Optional fields — they mean "use default /
        # inherit from checkpoint".  Only skip None for required fields
        # (is_resume, checkpoint_dir).
        if value is None and key in ("is_resume", "checkpoint_dir"):
            continue
        kwargs[key] = value

    for int_key in ("extra_epochs", "warmup_steps"):
        if int_key in kwargs and kwargs[int_key] is not None:
            try:
                kwargs[int_key] = int(kwargs[int_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {int_key} value: {kwargs[int_key]}. Must be an integer."
                ) from exc

    for float_key in ("learning_rate", "muon_learning_rate", "lr_decay_to_fraction"):
        if float_key in kwargs and kwargs[float_key] is not None:
            kwargs[float_key] = _parse_float_like(kwargs[float_key], float_key)

    if "lr_schedule" in kwargs and kwargs["lr_schedule"] is not None:
        lr_schedule = str(kwargs["lr_schedule"]).strip().lower()
        if lr_schedule not in {"linear", "cosine"}:
            raise ValueError(
                f"Invalid lr_schedule value: {kwargs['lr_schedule']}. Must be one of: linear, cosine."
            )
        kwargs["lr_schedule"] = lr_schedule

    for bool_key in ("is_resume", "reset_scheduler", "skip_warmup"):
        if bool_key in kwargs:
            value = kwargs[bool_key]
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                kwargs[bool_key] = value.lower() == "true"

    checkpoint_dir = kwargs.get("checkpoint_dir")
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

    if extra:
        logger.warning(
            f"Unknown keys in resume config (ignored): {extra}"
        )

    return ResumeConfig(**kwargs)

def _set_seed_for_init(seed: int) -> None:
    """Set seed before model creation so both pipelines start with identical weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
