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


from nanoplm.pretraining.models.modern_bert.model import (
    ProtModernBertMLM,
    ProtModernBertMLMConfig,
)
from nanoplm.pretraining.models.modern_bert.pure_model import (
    PureProtModernBertMLM,
    TEProtModernBertMLM,
)
from nanoplm.data.infra.validation import validate_pretrain_dataset
from nanoplm.data.infra.manifest import read_manifest, validate_manifest_for_pipeline, DataManifest
from nanoplm.utils.common import read_yaml, create_dirs, is_flash_attention_available
from nanoplm.utils.logger import logger


def _check_muon_available(optimizer: str) -> None:
    """Abort early when a Muon variant is requested but ``dion`` is not installed."""
    if optimizer.lower() in {"muon", "normuon", "muonh"}:
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
    if cfg.global_batch_tokens <= 0:
        raise ValueError(f"global_batch_tokens must be > 0, got {cfg.global_batch_tokens}")

    world_tokens_per_micro_step = (
        cfg.micro_batch_seqs * manifest.max_seq_len * effective_world_size
    )
    if world_tokens_per_micro_step <= 0:
        raise ValueError(
            f"Invalid token throughput per micro-step: {world_tokens_per_micro_step}. "
            "Check micro_batch_seqs, max_seq_len, and world_size."
        )

    inferred_grad_accum_steps = max(
        1,
        math.ceil(cfg.global_batch_tokens / world_tokens_per_micro_step),
    )
    achieved_global_batch_tokens = (
        inferred_grad_accum_steps * world_tokens_per_micro_step
    )
    global_batch_seqs = (
        inferred_grad_accum_steps * cfg.micro_batch_seqs * effective_world_size
    )

    cfg.inferred_grad_accum_steps = inferred_grad_accum_steps
    cfg.global_batch_seqs = global_batch_seqs
    cfg.achieved_global_batch_tokens = achieved_global_batch_tokens


def _normalize_distributed_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"fsdp", "ddp"}:
        raise click.ClickException(
            f"Unsupported distributed_mode={value!r}. Expected one of: fsdp, ddp."
        )
    return mode


def _validate_distributed_mode(
    cfg: PretrainingConfig,
    *,
    pure_torch: bool,
    pure_te: bool,
) -> None:
    cfg.distributed_mode = _normalize_distributed_mode(cfg.distributed_mode)
    cfg.ddp_bucket_cap_mb = int(cfg.ddp_bucket_cap_mb)
    if cfg.ddp_bucket_cap_mb <= 0:
        raise click.ClickException("ddp_bucket_cap_mb must be > 0.")
    if not (pure_torch or pure_te):
        return
    if cfg.distributed_mode == "ddp" and not cfg.multi_gpu:
        raise click.ClickException("distributed_mode=ddp requires multi_gpu=true.")
    if cfg.distributed_mode != "fsdp" and cfg.fsdp_reshard_after_forward:
        raise click.ClickException(
            "fsdp_reshard_after_forward applies only when distributed_mode=fsdp."
        )
    if cfg.distributed_mode != "ddp" and cfg.ddp_bucket_cap_mb != 25:
        raise click.ClickException(
            "ddp_bucket_cap_mb applies only when distributed_mode=ddp."
        )


def _validate_muonh_wo_init_compat(
    pretrain_cfg: PretrainingConfig,
    model_cfg: ProtModernBertMLMConfig,
) -> None:
    optimizer = str(pretrain_cfg.optimizer).strip().lower()
    wo_init = str(getattr(model_cfg, "wo_init", "zero")).strip().lower()
    if optimizer == "muonh" and wo_init == "zero":
        raise click.ClickException(
            "optimizer=muonh is incompatible with model.wo_init='zero'. "
            "MuonH initializes each matrix hyperball radius from ||W_init||_F, "
            "so zero-initialized Wo matrices are locked at zero. "
            "Set model.wo_init='kaiming' (or --wo-init kaiming)."
        )


def _validate_debug_layerwise_metrics_compat(
    cfg: PretrainingConfig,
    *,
    pure_te: bool,
) -> None:
    if pure_te and cfg.debug_layerwise_metrics:
        raise click.ClickException(
            "debug_layerwise_metrics is not supported with --pure-te / pure_te: true. "
            "Use the pure-torch path for layerwise debug hooks, or disable debug_layerwise_metrics."
        )


def _check_pipeline_supported(pure_torch: bool, pure_te: bool,
                              lr_schedule: str, resume_mode) -> None:
    """Pretraining requires pure_torch or pure_te; wsd/decay require pure_torch (spec section 14)."""
    if not (pure_torch or pure_te):
        raise ValueError(
            "Pretraining requires pure_torch: true or pure_te: true. "
            "The HF Trainer pretraining path is no longer supported."
        )
    sched = (lr_schedule or "").lower()
    mode = (resume_mode or "").lower() if resume_mode else None
    if not pure_torch and (sched == "wsd" or mode == "decay"):
        raise ValueError(
            "lr_schedule: wsd and resume mode: decay require pure_torch: true "
            "(TE supports warmup_stable/cosine/linear + continue only)."
        )


@click.group(name="pretrain")
@click.help_option("--help", "-h")
def pretrain():
    """Group of commands for model pretraining."""
    pass


@pretrain.command("run")
@click.help_option("--help", "-h")
# Dataset and output
@click.option(
    "--dataset-dir",
    type=str,
    required=True,
    help="Path to dataset directory containing .data_manifest (from nanoplm data from-yaml)",
)
@click.option(
    "--ckp-dir",
    type=str,
    default="output/pretraining_checkpoints",
    help="Checkpoint directory",
)
# Training hyperparameters
@click.option(
    "--micro-batch-seqs",
    type=int,
    default=64,
    help="Sequences per GPU per micro-step (forward pass)",
)
@click.option(
    "--adam-learning-rate",
    type=float,
    default=1e-4,
    help="AdamW learning rate (Muon uses --muon-learning-rate)",
)
@click.option(
    "--adam-weight-decay",
    type=float,
    default=0.0,
    help="AdamW weight decay (Muon uses --muon-weight-decay)",
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
    "--global-batch-warmup-steps",
    type=int,
    default=0,
    help="Optimizer steps to ramp the global batch from 1/8 to full (1/8 -> 1/4 -> 1/2 -> full). "
    "Default 0: no batch-size warmup.",
)
@click.option(
    "--lr-decay-to-fraction",
    type=float,
    default=0.1,
    help="Fraction of peak learning rate to decay to",
)
@click.option(
    "--lr-schedule",
    type=click.Choice(["linear", "cosine"], case_sensitive=False),
    default="cosine",
    help="Learning rate schedule to use after warmup",
)
@click.option(
    "--global-batch-tokens",
    type=int,
    default=256000,
    help="Target tokens per optimizer step (grad_accum inferred automatically)",
)
@click.option(
    "--optimizer",
    type=click.Choice(
        ["adamw", "stable_adamw", "muon", "normuon", "muonh"], case_sensitive=False
    ),
    default="normuon",
    help="Optimizer to use",
)
@click.option("--adam-beta1", type=float, default=0.9, help="Adam beta1")
@click.option("--adam-beta2", type=float, default=0.999, help="Adam beta2")
@click.option("--adam-epsilon", type=float, default=1e-8, help="Adam epsilon")
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
    "--muonh-normalization",
    type=click.Choice(["neuron", "short_axis", "none"], case_sensitive=False),
    default="neuron",
    help="MuonH update normalization (used only when optimizer=muonh): "
    "'neuron' = per-row, 'short_axis' = along the shorter matrix axis, 'none' = disabled.",
)
@click.option("--mlm-probability", type=float, default=0.3, help="MLM probability")
@click.option(
    "--logging-steps", type=int, default=1, help="Number of steps between log events"
)
@click.option(
    "--eval-steps", type=int, default=250, help="Number of steps between evaluations"
)
@click.option(
    "--save-steps",
    type=int,
    default=5000,
    help="Number of steps between checkpoint saves",
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option(
    "--debug-non-finite-params/--no-debug-non-finite-params",
    default=True,
    help="Check parameters for non-finite values after every optimizer step.",
)
@click.option(
    "--debug-layerwise-metrics/--no-debug-layerwise-metrics",
    default=False,
    help="Log per-layer grad/weight norms, residual-stream RMS, and logit stats. "
    "Has non-trivial overhead; installs forward hooks that may break torch.compile graphs. "
    "Not supported with --pure-te.",
)
@click.option(
    "--debug-layerwise-log-every",
    type=int,
    default=20,
    help="Log layerwise debug metrics every N optimizer steps.",
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
    help="Probability of replacing masked tokens with random tokens",
)
@click.option(
    "--keep-probability",
    type=float,
    default=0.1,
    help="Probability of leaving masked tokens unchanged",
)
@click.option(
    "--num-workers",
    type=str,
    default="auto",
    help="Number of DataLoader workers ('auto' or integer)",
)
@click.option(
    "--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor"
)
@click.option(
    "--compile/--no-compile",
    default=True,
    help="Enable torch.compile for faster training (disable for debugging or unsupported hardware)",
)
@click.option(
    "--use-packing/--no-packing",
    default=True,
    help="Enable sequence packing to eliminate padding waste (requires flash attention)",
)
@click.option(
    "--use-static-inp-size/--no-use-static-inp-size",
    default=True,
    help="When packing is enabled, use fixed flat token size + bucketed cu_seqlens/max_seqlen for static-shape execution",
)
@click.option(
    "--bf16/--no-bf16",
    default=True,
    help="Enable mixed precision training (bf16 if supported, fp16 fallback)",
)
@click.option(
    "--tf32/--no-tf32",
    default=True,
    help="Enable TF32 mode on Ampere+ GPUs for faster fp32 matmuls",
)
@click.option(
    "--fp8/--no-fp8",
    default=False,
    help="Enable FP8 Linear matmuls in pure-torch/TE paths (CUDA only, best on H100+)",
)
@click.option(
    "--multi-gpu", is_flag=True, default=True, help="Enable multi-GPU training"
)
@click.option(
    "--distributed-mode",
    type=click.Choice(["fsdp", "ddp"], case_sensitive=False),
    default="fsdp",
    show_default=True,
    help="Distributed wrapper for multi-GPU runs. DDP is supported only in pure-torch / pure-te.",
)
@click.option(
    "--fsdp-shard-granularity",
    type=click.Choice(["layer", "sublayer"], case_sensitive=False),
    default="layer",
    show_default=True,
    help="FSDP2 wrapping granularity. 'layer' = whole transformer blocks (fewer boundaries, "
    "best for NVLink/SXM). 'sublayer' = attn & mlp separately (finer overlap, best for PCIe).",
)
@click.option(
    "--fsdp-reshard-after-forward/--no-fsdp-reshard-after-forward",
    default=False,
    show_default=True,
    help="Universal FSDP2 reshard_after_forward setting for all fully_shard groups in pure-torch / pure-te.",
)
@click.option(
    "--world-size",
    type=str,
    default="auto",
    help="Total number of processes for distributed training; use 'auto' to use all available GPUs",
)
@click.option(
    "--ddp-bucket-cap-mb",
    type=int,
    default=25,
    show_default=True,
    help="DDP gradient bucket size in MB (only used when --distributed-mode=ddp).",
)
@click.option(
    "--project-name",
    type=str,
    default="nanoplm-pretraining",
    help="Weights & Biases project name",
)
@click.option(
    "--run-name",
    type=str,
    default="run",
    help=f"Weights & Biases run name under the project name as DDMMHHMM-<run_name>",
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
    "--resume-mode",
    type=click.Choice(["continue", "decay"], case_sensitive=False),
    default="continue",
    help=(
        "continue: finish the original token/step budget from the checkpoint. "
        "decay: anneal a trunk checkpoint over an additional decay budget."
    ),
)
@click.option(
    "--resume-decay-tokens",
    type=int,
    default=None,
    help="(mode=decay) Additional anneal token budget from the checkpoint.",
)
@click.option(
    "--resume-decay-steps",
    type=int,
    default=None,
    help="(mode=decay) Additional anneal step budget (mutually exclusive with --resume-decay-tokens).",
)
@click.option(
    "--resume-decay-shape",
    type=click.Choice(["1-sqrt", "linear"], case_sensitive=False),
    default=None,
    help="(mode=decay) Anneal shape: 1-sqrt | linear.",
)
@click.option(
    "--resume-lr-decay-to-fraction",
    type=float,
    default=None,
    help="(mode=decay) LR decay floor fraction for the anneal.",
)
# Model hyperparameters (ModernBERT)
@click.option("--hidden-size", type=int, default=768, help="Model hidden size")
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
    "--num-kv-heads",
    type=int,
    default=None,
    help="Number of KV heads for GQA (default: same as num_attention_heads = MHA)",
)
@click.option(
    "--vocab-size",
    type=int,
    default=32,
    help="Number of the vocabs being used in the model (should be equal to the vocab size in the tokenizer)",
)
@click.option(
    "--norm-type",
    type=click.Choice(["layernorm", "rmsnorm"], case_sensitive=False),
    default="layernorm",
    help="Normalization type: layernorm (default) or rmsnorm (slightly faster, no bias)",
)
@click.option(
    "--mlp-activation",
    type=click.Choice(["swiglu"], case_sensitive=False),
    default="swiglu",
    help="MLP activation",
)
@click.option("--mlp-dropout", type=float, default=0.0, help="MLP dropout")
@click.option("--mlp-bias", is_flag=True, default=False, help="Use MLP bias")
@click.option(
    "--no-mlp-on-first-layer/--mlp-on-first-layer",
    default=True,
    help="Disable the MLP branch in encoder layer 0",
)
@click.option("--attention-bias", is_flag=True, default=False, help="Use attn bias")
@click.option("--attention-dropout", type=float, default=0.0, help="Attn dropout")
@click.option(
    "--classifier-activation",
    type=click.Choice(["relu", "gelu", "srelu"], case_sensitive=False),
    default="gelu",
    help="Classifier activation (srelu is supported only in pure-torch and pure-te paths)",
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
    "--use-prores/--no-use-prores",
    default=False,
    help="Enable ProRes: progressive residual warmup (scales each layer's residual from 0 to 1, deeper layers warm up slower).",
)
@click.option(
    "--prores-t",
    type=int,
    default=1000,
    help="ProRes warmup length T for the first layer. Layer l finishes warmup at step T*l.",
)
@click.option(
    "--activation-checkpointing/--no-activation-checkpointing",
    default=False,
    help="Enable activation checkpointing (recompute transformer layers in backward to save VRAM).",
)
@click.option(
    "--activation-checkpointing-mode",
    type=click.Choice(["layer", "attn", "attn+mlp"], case_sensitive=False),
    default="layer",
    show_default=True,
    help="Checkpoint scope. 'layer' checkpoints the whole transformer layer; "
    "'attn' checkpoints only the attention residual branch (usually the best tradeoff); "
    "'attn+mlp' checkpoints both attention and MLP branches.",
)
@click.option(
    "--tie-word-embeddings/--no-tie-word-embeddings",
    default=True,
    help="Tie input embedding and output decoder weights (default: enabled).",
)
@click.option(
    "--use-diff-attn-v2/--no-use-diff-attn-v2",
    default=False,
    help="Enable Differential Attention V2: doubles query heads, applies differential "
    "subtraction for noise cancellation in attention (pure-torch only).",
)
@click.option(
    "--attn-layer-pattern",
    type=str,
    default=None,
    help="Attention layer pattern string using F (full) and S (sliding), e.g. 'FSS'. "
    "Tiles to cover all layers. Overrides --global-attn-every-n-layers when set.",
)
@click.option(
    "--fused-qkv/--no-fused-qkv",
    default=True,
    help="Use a single fused Wqkv projection (default). When disabled, Q/K/V use separate "
    "linear layers so Muon orthogonalizes each subspace independently.",
)
@click.option(
    "--fused-up-gate/--no-fused-up-gate",
    default=True,
    help="Use a single fused Wi projection for SwiGLU/GLU up+gate (default). When disabled, "
    "up and gate use separate linear layers so Muon orthogonalizes each independently.",
)
@click.option(
    "--wo-init",
    type=click.Choice(["kaiming", "zero"], case_sensitive=False),
    default="zero",
    show_default=True,
    help="Initialization for residual out-projections (attn.Wo and mlp.Wo).",
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
    micro_batch_seqs: int,
    adam_learning_rate: float,
    adam_weight_decay: float,
    gradient_clipping: bool,
    warmup_steps: int,
    repo_rope_warmup_steps: Optional[int],
    global_batch_warmup_steps: int,
    lr_decay_to_fraction: float,
    lr_schedule: str,
    global_batch_tokens: int,
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
    muonh_normalization: str,
    mlm_probability: float,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    seed: int,
    debug_non_finite_params: bool,
    debug_layerwise_metrics: bool,
    debug_layerwise_log_every: int,
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
    distributed_mode: str,
    fsdp_shard_granularity: str,
    fsdp_reshard_after_forward: bool,
    world_size: str,
    ddp_bucket_cap_mb: int,
    project_name: str,
    run_name: str,
    resume: bool,
    resume_checkpoint_dir: str,
    resume_mode: str,
    resume_decay_tokens: Optional[int],
    resume_decay_steps: Optional[int],
    resume_decay_shape: Optional[str],
    resume_lr_decay_to_fraction: Optional[float],
    # model hp
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_kv_heads: Optional[int],
    vocab_size: int,
    norm_type: str,
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
    use_prores: bool,
    prores_t: int,
    activation_checkpointing: bool,
    activation_checkpointing_mode: str,
    tie_word_embeddings: bool,
    use_diff_attn_v2: bool,
    use_paired_head_attention: bool,
    attn_layer_pattern: Optional[str],
    fused_qkv: bool,
    fused_up_gate: bool,
    wo_init: str,
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
        micro_batch_seqs=micro_batch_seqs,
        adam_learning_rate=adam_learning_rate,
        adam_weight_decay=adam_weight_decay,
        max_grad_norm=1.0 if gradient_clipping else float("inf"),
        warmup_steps=warmup_steps,
        repo_rope_warmup_steps=repo_rope_warmup_steps,
        global_batch_warmup_steps=global_batch_warmup_steps,
        lr_decay_to_fraction=lr_decay_to_fraction,
        lr_schedule=lr_schedule,
        global_batch_tokens=global_batch_tokens,
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
        muonh_normalization=muonh_normalization,
        mlm_probability=mlm_probability,
        mask_replace_prob=mask_replace_prob,
        random_token_prob=random_token_prob,
        keep_probability=keep_probability,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        seed=seed,
        debug_non_finite_params=debug_non_finite_params,
        debug_layerwise_metrics=debug_layerwise_metrics,
        debug_layerwise_log_every=debug_layerwise_log_every,
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
        distributed_mode=distributed_mode,
        fsdp_shard_granularity=fsdp_shard_granularity,
        fsdp_reshard_after_forward=fsdp_reshard_after_forward,
        world_size=world_size,
        ddp_bucket_cap_mb=ddp_bucket_cap_mb,
        project_name=project_name,
        run_name=run_name
    )

    if pure_torch and pure_te:
        raise click.ClickException("--pure-torch and --pure-te are mutually exclusive.")
    if use_canon_layers and not pure_torch:
        raise click.ClickException(
            "use_canon_layers requires --pure-torch. "
            "Canon layers are not implemented in HF/TE paths."
        )
    if classifier_activation.lower() == "srelu" and not (pure_torch or pure_te):
        raise click.ClickException(
            "classifier_activation=srelu requires --pure-torch or --pure-te. "
            "The default HF path supports only relu/gelu."
        )
    if (
        num_kv_heads is not None
        and num_kv_heads != num_attention_heads
        and not (pure_torch or pure_te)
    ):
        raise click.ClickException(
            "GQA requires --pure-torch or --pure-te. "
            "The default HF path does not implement num_kv_heads != num_attention_heads."
        )
    if (use_mhc_lite or mhc_lite_wrapping_level.lower() != "layer") and not (
        pure_torch or pure_te
    ):
        raise click.ClickException(
            "mHC-lite requires --pure-torch or --pure-te. "
            "Set one of those modes (or disable mHC-lite settings)."
        )
    if pure_te and mhc_lite_wrapping_level.lower() != "layer":
        raise click.ClickException(
            "Transformer Engine currently supports only layer-level mHC-lite. "
            "Set --mhc-lite-wrapping-level layer or use --pure-torch."
        )
    if use_mhc_lite and use_resid_lambdas:
        raise click.ClickException(
            "use_mhc_lite and use_resid_lambdas are mutually exclusive. "
            "resid_lambdas scales the hidden state before each layer, which breaks "
            "mHC-lite's doubly-stochastic stability guarantees."
        )
    _validate_debug_layerwise_metrics_compat(cfg, pure_te=pure_te)
    _validate_distributed_mode(cfg, pure_torch=pure_torch, pure_te=pure_te)

    _populate_batch_setup(cfg)

    model_cfg = ProtModernBertMLMConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        norm_type=norm_type,
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
        use_prores=use_prores,
        prores_T=prores_t,
        activation_checkpointing=activation_checkpointing,
        activation_checkpointing_mode=activation_checkpointing_mode.lower(),
        use_mhc_lite=use_mhc_lite,
        mhc_n_streams=mhc_n_streams,
        mhc_lite_wrapping_level=mhc_lite_wrapping_level.lower(),
        tie_word_embeddings=tie_word_embeddings,
        use_diff_attn_v2=use_diff_attn_v2,
        use_paired_head_attention=use_paired_head_attention,
        attn_layer_pattern=attn_layer_pattern,
        fused_qkv=fused_qkv,
        fused_up_gate=fused_up_gate,
        wo_init=wo_init,
    )

    _validate_muonh_wo_init_compat(cfg, model_cfg)

    _set_seed_for_init(seed)
    if use_diff_attn_v2 and not pure_torch:
        raise click.ClickException(
            "use_diff_attn_v2 requires --pure-torch. "
            "Differential Attention V2 is not implemented in HF/TE paths."
        )
    if use_paired_head_attention and not pure_torch:
        raise click.ClickException(
            "use_paired_head_attention requires --pure-torch. "
            "Paired head attention is not implemented in HF/TE paths."
        )
    if use_paired_head_attention and use_diff_attn_v2:
        raise click.ClickException(
            "use_paired_head_attention and use_diff_attn_v2 are mutually exclusive."
        )
    # Fail fast before any heavy manifest/model instantiation: only
    # pure_torch/pure_te pipelines are supported, and wsd/decay require
    # pure_torch. The HF Trainer pretraining path is no longer supported.
    try:
        _check_pipeline_supported(
            pure_torch, pure_te, cfg.lr_schedule,
            resume_mode if resume else None,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    manifest = _load_manifest_and_inject(cfg, model_cfg)
    if pure_te:
        logger.info("Using Transformer Engine model and training loop")
        model = TEProtModernBertMLM(model_cfg)
    elif pure_torch:
        logger.info("Using pure-torch model and training loop")
        model = PureProtModernBertMLM(model_cfg)
    else:
        model = ProtModernBertMLM(model_cfg)

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")
    if getattr(model_cfg, "use_moe", False):
        from nanoplm.pretraining.models.modern_bert.moe import MoELayer as _MoELayer
        inactive = 0
        for m in model.modules():
            if isinstance(m, _MoELayer):
                inactive += (m.num_experts - m.top_k) * m.Wi.shape[1] * m.Wi.shape[2]
                inactive += (m.num_experts - m.top_k) * m.Wo.shape[1] * m.Wo.shape[2]
        logger.info(f"Active Parameters (per token): {total_params - inactive}")
    logger.info(f"Flash attention available: {is_flash_attention_available()}")

    resume_cfg = ResumeConfig(
        is_resume=resume,
        checkpoint_dir=resume_checkpoint_dir,
        mode=resume_mode,
        decay_tokens=resume_decay_tokens,
        decay_steps=resume_decay_steps,
        decay_shape=resume_decay_shape,
        lr_decay_to_fraction=resume_lr_decay_to_fraction,
    )
    if resume_cfg.is_resume and not resume_cfg.checkpoint_dir:
        raise click.ClickException("--resume requires --resume-checkpoint-dir")
    _validate_resume_config_modes(resume_cfg)

    if pure_te:
        run_te_pretraining(
            model=model,
            pretrain_config=cfg,
            manifest=manifest,
            resume_config=resume_cfg if resume_cfg.is_resume else None,
        )
    elif pure_torch:
        run_pure_pretraining(
            model=model,
            pretrain_config=cfg,
            manifest=manifest,
            resume_config=resume_cfg if resume_cfg.is_resume else None,
        )


@pretrain.command("from-yaml")
@click.help_option("--help", "-h")
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
    _validate_resume_config_modes(resume_config)
    _validate_muonh_wo_init_compat(pretrain_config, model_config)
    if getattr(model_config, "use_noble", False) and not pure_torch:
        raise click.ClickException(
            "model.use_noble=true requires pure_torch: true (or --pure-torch). "
            "NOBLE is not implemented in HF/TE paths."
        )
    if model_config.use_canon_layers and not pure_torch:
        raise click.ClickException(
            "model.use_canon_layers=true requires pure_torch: true (or --pure-torch). "
            "Canon layers are not implemented in HF/TE paths."
        )
    if (
        model_config.use_mhc_lite
        or str(getattr(model_config, "mhc_lite_wrapping_level", "layer")).lower()
        != "layer"
    ) and not (pure_torch or pure_te):
        raise click.ClickException(
            "model.use_mhc_lite=true (or model.mhc_lite_wrapping_level != 'layer') "
            "requires pure_torch: true / --pure-torch or pure_te: true / --pure-te. "
            "mHC-lite is not implemented in the HF path."
        )
    if (
        pure_te
        and str(getattr(model_config, "mhc_lite_wrapping_level", "layer")).lower()
        != "layer"
    ):
        raise click.ClickException(
            "Transformer Engine currently supports only layer-level mHC-lite. "
            "Set model.mhc_lite_wrapping_level: layer or use pure_torch: true."
        )
    if model_config.use_mhc_lite and model_config.use_resid_lambdas:
        raise click.ClickException(
            "model.use_mhc_lite=true is not compatible with model.use_resid_lambdas=true. "
            "resid_lambdas scales the hidden state before each layer, which breaks "
            "mHC-lite's doubly-stochastic stability guarantees."
        )
    if getattr(model_config, "use_diff_attn_v2", False) and not pure_torch:
        raise click.ClickException(
            "model.use_diff_attn_v2=true requires pure_torch: true (or --pure-torch). "
            "Differential Attention V2 is not implemented in HF/TE paths."
        )
    if getattr(model_config, "use_paired_head_attention", False) and not pure_torch:
        raise click.ClickException(
            "model.use_paired_head_attention=true requires pure_torch: true (or --pure-torch). "
            "Paired head attention is not implemented in HF/TE paths."
        )
    if getattr(model_config, "use_paired_head_attention", False) and getattr(
        model_config, "use_diff_attn_v2", False
    ):
        raise click.ClickException(
            "model.use_paired_head_attention=true is not compatible with model.use_diff_attn_v2=true."
        )
    if str(
        getattr(model_config, "classifier_activation", "gelu")
    ).lower() == "srelu" and not (pure_torch or pure_te):
        raise click.ClickException(
            "model.classifier_activation=srelu requires pure_torch: true / --pure-torch or "
            "pure_te: true / --pure-te. The HF path supports only relu/gelu."
        )
    if (
        getattr(model_config, "num_kv_heads", None) is not None
        and model_config.num_kv_heads != model_config.num_attention_heads
        and not (pure_torch or pure_te)
    ):
        raise click.ClickException(
            "model.num_kv_heads != model.num_attention_heads requires pure_torch: true / --pure-torch "
            "or pure_te: true / --pure-te. The HF path does not implement GQA."
        )
    _validate_debug_layerwise_metrics_compat(pretrain_config, pure_te=pure_te)
    _validate_distributed_mode(
        pretrain_config,
        pure_torch=pure_torch,
        pure_te=pure_te,
    )
    _populate_batch_setup(pretrain_config)

    # Fail fast before any heavy model instantiation: only pure_torch/pure_te
    # pipelines are supported, and wsd/decay require pure_torch.
    _check_pipeline_supported(
        pure_torch, pure_te, pretrain_config.lr_schedule,
        resume_config.mode if resume_config.is_resume else None,
    )

    manifest = _load_manifest_and_inject(pretrain_config, model_config)
    _set_seed_for_init(pretrain_config.seed)
    if pure_te:
        logger.info("Using Transformer Engine model and training loop")
        model = TEProtModernBertMLM(config=model_config)
    elif pure_torch:
        logger.info("Using pure-torch model and training loop")
        model = PureProtModernBertMLM(config=model_config)

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Total Trainable Parameters: {total_params}")
    if getattr(model_config, "use_moe", False):
        from nanoplm.pretraining.models.modern_bert.moe import MoELayer as _MoELayer
        inactive = 0
        for m in model.modules():
            if isinstance(m, _MoELayer):
                inactive += (m.num_experts - m.top_k) * m.Wi.shape[1] * m.Wi.shape[2]
                inactive += (m.num_experts - m.top_k) * m.Wo.shape[1] * m.Wo.shape[2]
        logger.info(f"Active Parameters (per token): {total_params - inactive}")
    logger.info(f"Flash attention available: {is_flash_attention_available()}")

    if pure_te:
        run_te_pretraining(
            model=model,
            pretrain_config=pretrain_config,
            manifest=manifest,
            resume_config=resume_config if resume_config.is_resume else None,
        )
    elif pure_torch:
        run_pure_pretraining(
            model=model,
            pretrain_config=pretrain_config,
            manifest=manifest,
            resume_config=resume_config if resume_config.is_resume else None,
        )


@pretrain.command("get-yaml")
@click.help_option("--help", "-h")
@click.argument(
    "output",
    required=False,
    type=click.Path(dir_okay=True, writable=True, resolve_path=True),
)
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
        "  # num_kv_heads: 8  # GQA: number of KV heads (default: same as num_attention_heads = MHA)\n"
        "  vocab_size: 32\n"
        '  norm_type: "layernorm"  # layernorm | rmsnorm (rmsnorm is slightly faster, no bias)\n'
        '  mlp_activation: "swiglu"\n'
        "  mlp_dropout: 0.0\n"
        "  mlp_bias: false\n"
        "  no_mlp_on_first_layer: true\n"
        "  attention_bias: false\n"
        "  attention_dropout: 0.0\n"
        '  classifier_activation: "gelu"  # relu | gelu | srelu (srelu requires pure_torch or pure_te)\n'
        "  tie_word_embeddings: true  # tie input embedding and output decoder weights\n"
        "  # The options below only work on pure-torch and TE pipelines unless noted\n"
        "  use_resid_lambdas: false  # scales residual stream per layer (not compatible with use_mhc_lite)\n"
        "  use_x0_lambdas: false  # blends initial embedding x0 per layer\n"
        "  use_qk_norm: false  # applies RMS norm to Q/K in attention\n"
        "  use_canon_layers: false # enables Canon-ABCD local mixing layers (pure_torch only)\n"
        '  canon_layers_mode: "ac"  # subset of Canon sites: A/B/C/D (e.g. "ac" for lighter mode)\n'
        "  canon_layers_kernel_size: 5  # symmetric Canon kernel size (allowed: 3/5/7, default: 5)\n"
        "  use_repo: false  # RePO: learned per-head positions replacing fixed RoPE (pure_torch only)\n"
        "  repo_after_n_layers: 3  # first N layers keep standard RoPE, layers after use RePO\n"
        "  use_prores: false  # ProRes: progressive residual warmup (deeper layers warm up slower)\n"
        "  prores_T: 1000  # ProRes warmup length for first layer; layer l finishes at step T*l\n"
        "  use_mhc_lite: false  # mHC-lite: multi-stream residual with doubly stochastic mixing (pure_torch only)\n"
        "  mhc_n_streams: 4  # number of residual streams for mHC-lite (n! permutation matrices)\n"
        "  mhc_lite_wrapping_level: \"layer\"  # mHC-lite wrapping: 'layer' or 'sublayers' (pure_torch only)\n"
        "  mhc_triton_fused: true  # use fused Triton kernels for mHC-lite stream ops; first run will start slow due to Triton autotune\n"
        "  use_diff_attn_v2: false  # Differential Attention V2: doubles Q heads with differential subtraction (pure_torch only)\n"
        "  use_paired_head_attention: false  # pairs adjacent attention heads into a shared softmax (pure_torch only; MHA only; incompatible with diff_attn_v2, RePO, and Canon-B)\n"
        "  use_noble: false  # NOBLE: nonlinear low-rank branches for linear layers (pure_torch only)\n"
        "  noble_rank: 64  # NOBLE bottleneck rank (typical: 64-256)\n"
        "  noble_alpha: 0.01  # NOBLE W_up initialization scale\n"
        "  noble_half_kaiming: true  # halve main weight init scale for NOBLE layers (paper §3.3)\n"
        "  noble_targets: all  # which projections get NOBLE: all|attn|ffn|qkv|out\n"
        "  attn_layer_pattern: null  # Attention pattern string e.g. 'FSS' (F=full, S=sliding). Tiles to num_layers. Overrides global_attn_every_n_layers.\n"
        "  fused_qkv: true  # false = separate Wq/Wk/Wv instead of a single Wqkv\n"
        "  fused_up_gate: true  # false = separate W_up/W_gate instead of a single Wi for SwiGLU/GLU\n"
        '  wo_init: "zero"  # residual out-projection init for attn.Wo/mlp.Wo: kaiming | zero\n'
        "  activation_checkpointing: false  # Enable to reduce VRAM by recomputing activations during backward\n"
        '  activation_checkpointing_mode: "attn"  # (layer | attn | attn+mlp), attn mode has highest ROI \n'
        "  # MoE (Mixture-of-Experts) — pure_torch only, requires use_packing + use_static_inp_size\n"
        "  use_moe: false\n"
        "  moe_num_experts: 8              # number of routed experts\n"
        "  moe_top_k: 2                    # experts activated per token\n"
        "  # one shared expert is always enabled; routing is always sigmoid\n"
        "  moe_use_bias_correction: true   # expert-bias load balancing (selection-only)\n"
        "  moe_aux_loss_coef: 0.01         # differentiable balance regularizer over router mass\n"
        "  moe_z_loss_coef: 5e-5           # router z-loss for small-model MoE stability\n"
        "  moe_routed_scaling_factor: 1.0  # routed branch output scale; start conservative\n"
        "  moe_n_group: 1                  # group-limited routing: 1 disables, >1 partitions experts into groups\n"
        "  moe_topk_group: 1               # with moe_n_group > 1, keep top K groups per token\n"
        "  moe_bias_update_rate: 1e-3      # expert-bias correction step size\n"
        "  moe_leading_dense_layers: 1     # first N layers are always dense (DeepSeek uses 1)\n"
        "  # Huginn-style recycling — pure_torch only. recycling_loop_count: 1 disables recycling.\n"
        "  # Splits the L-layer stack into prelude (P, runs once) / recurrent (L-P-C, weight-tied,\n"
        "  # looped) / coda (C, runs once); a learned 2h->h adapter consumes [s, e] at each iter.\n"
        "  recycling_num_prelude_layers: 0           # P; first P layers run once\n"
        "  recycling_num_coda_layers: 0              # C; last C layers run once\n"
        "  recycling_loop_count: 1                   # target mean loop count (1 = off)\n"
        "  recycling_adapter_bias: false             # bias on the 2h->h adapter (Huginn: no bias)\n"
        "  recycling_s0_noise_sigma: 0.6324555320336759  # sqrt(2/5), Huginn s_0 noise scale\n"
        "  recycling_use_takase_init: true           # Takase spike init on recurrent block + adapter\n"
        "\n"
        "pretraining:\n"
        '  dataset_dir: "output/data/pretrain_data"\n'
        '  ckp_dir: "output/pretraining_checkpoints"\n'
        "  # Batch sizing (token-budget surface)\n"
        "  micro_batch_seqs: 64\n"
        "  global_batch_tokens: 256000\n"
        "  global_batch_warmup_steps: 0  # number of steps to warm up to the target global batch size, starts at 1/8 -> 1/4 -> 1/2 -> full (at the provided step). default: 0, no batch size warmup. \n"
        "  # Duration + LR schedule (token-budget surface)\n"
        '  lr_schedule: "wsd"  # wsd | warmup_stable | cosine | linear\n'
        "  warmup_steps: 1000\n"
        "  stable_tokens: 9000000   # wsd: tokens up to the pre-decay checkpoint (warmup included)\n"
        "  decay_tokens: 1000000    # wsd: additional anneal tokens\n"
        '  decay_shape: "1-sqrt"  # 1-sqrt | linear\n'
        "  lr_decay_to_fraction: 0.1\n"
        "  # train_tokens: 10000000  # single-phase budget for warmup_stable | cosine | linear (NOT used with wsd)\n"
        "  # Advanced step overrides (each replaces its *_tokens counterpart; never set both):\n"
        "  # stable_steps:  # ↔ stable_tokens (wsd)\n"
        "  # decay_steps:   # ↔ decay_tokens (wsd / resume-decay)\n"
        "  # max_steps:     # ↔ train_tokens (warmup_stable | cosine | linear)\n"
        '  optimizer: "adamw"  # adamw | stable_adamw | muon | normuon | muonh\n'
        "  # AdamW hyperparameters (also used for AdamW side [1D and embedding/unembed params] when optimizer=muon, normuon, or muonh)\n"
        "  adam_beta1: 0.9\n"
        "  adam_beta2: 0.999\n"
        "  adam_epsilon: 1e-8\n"
        "  adam_learning_rate: 1e-4\n"
        "  max_grad_norm: .inf\n"
        "  repo_rope_warmup_steps: 1000\n"
        "  adam_weight_decay: 0.0\n"
        "  muon_learning_rate: 1e-3\n"
        "  muon_weight_decay: 0.01\n"
        "  muon_cautious_weight_decay: true\n"
        "  muon_use_polar_express: true\n"
        "  muon_momentum: 0.95\n"
        "  muon_nesterov: true\n"
        "  muon_eps: 1e-7\n"
        '  muonh_normalization: "neuron"  # used only when optimizer=muonh: neuron | short_axis | none\n'
        "  # NOBLE optimizer LR scaling (only used when model.use_noble=true)\n"
        "  noble_base_lr: 3e-4  # independent base LR for NOBLE branch params\n"
        "  noble_lr_gamma: 0.3  # W_up LR power (actual exponent = 2γ = 0.6)\n"
        "  noble_lr_gamma_M: 0.45  # M mixing matrix LR power\n"
        "  noble_freq_lr_mult: 3.0  # ω frequency LR multiplier\n"
        "  noble_phase_lr_mult: 5.0  # φ phase LR multiplier\n"
        "  # Huginn-style recycling training-time options (active only when model.recycling_loop_count > 1)\n"
        "  recycling_bptt: 8                                # truncated BPTT depth K (last K iters get gradients)\n"
        '  recycling_loop_count_sampling: "lognormal_poisson"  # fixed | lognormal_poisson\n'
        "  recycling_loop_count_sigma: 0.5                  # log-normal std (Huginn value)\n"
        '  recycling_loop_count_curriculum: "1_sqrt"        # none | linear | 1_sqrt\n'
        "  recycling_loop_count_curriculum_steps: -1        # -1 = auto (25% of total training steps)\n"
        "  recycling_pretraining_eval_loop_counts: null     # null = single r at target mean; e.g. [1, 4] for in-loop eval sweep\n"
        "\n"
        "  mlm_probability: 0.3\n"
        "  mask_replace_prob: 0.8\n"
        "  random_token_prob: 0.1\n"
        "  keep_probability: 0.1\n"
        "  logging_steps: 1\n"
        "  eval_steps: 250\n"
        "  save_steps: 5000\n"
        "  seed: 42\n"
        "  debug_non_finite_params: true  # checks params for NaN/Inf after every optimizer step\n"
        "  debug_layerwise_metrics: false  # per-layer grad/weight norms, residual RMS, logit stats (overhead; breaks torch.compile graphs; not supported with pure_te)\n"
        "  debug_layerwise_log_every: 20  # log layerwise metrics every N optimizer steps\n"
        '  num_workers: "auto"\n'
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
        '  distributed_mode: "fsdp"  # fsdp | ddp (pure_torch/pure_te honor this; HF path ignores it)\n'
        '  fsdp_shard_granularity: "layer"  # layer (fewer boundaries, NVLink/SXM) | sublayer (finer overlap, PCIe)\n'
        "  fsdp_reshard_after_forward: false  # universal FSDP2 reshard_after_forward setting for pure_torch/pure_te\n"
        "  world_size: 'auto'\n"
        "  ddp_bucket_cap_mb: 25  # only used when distributed_mode: ddp in pure_torch/pure_te; HF path ignores it\n"
        '  project_name: "nanoplm-pretraining"\n'
        '  run_name: "run" # will be DDMMHHMM-<run_name>\n'
        "  profiler_enabled: false\n"
        "  profiler_start_step: 10\n"
        "  profiler_end_step: 15\n"
        "\n"
        "\n"
        "resume:\n"
        "  is_resume: false\n"
        '  checkpoint_dir: "output/pretraining_checkpoints/run-1/checkpoint-1"\n'
        "  # Two resume modes:\n"
        "  # continue - finish the original token/step budget from the checkpoint\n"
        "  #   with the original schedule. Takes NO decay_* fields.\n"
        "  # decay - anneal a stable-end trunk checkpoint over an additional budget.\n"
        "  #   Set exactly one of decay_tokens / decay_steps. decay_shape and\n"
        "  #   lr_decay_to_fraction are OPTIONAL overrides; leave null to fall\n"
        "  #   back to the schedule defaults.\n"
        '  mode: "decay"  # continue | decay\n'
        "  # decay-mode fields (leave null / omit for mode: continue):\n"
        "  decay_tokens: 1000000  # additional anneal tokens (or use decay_steps)\n"
        '  decay_shape: "1-sqrt"  # 1-sqrt | linear\n'
        "  lr_decay_to_fraction: 0.1\n"
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


def _validate_duration_config(cfg: PretrainingConfig) -> None:
    """Validate schedule <-> duration-budget consistency (spec section 12)."""
    sched = (cfg.lr_schedule or "").lower()
    valid = {"wsd", "warmup_stable", "cosine", "linear"}
    if sched not in valid:
        raise ValueError(f"lr_schedule must be one of {sorted(valid)}, got {cfg.lr_schedule!r}")

    def _one(token_field, step_field, label):
        tok = getattr(cfg, token_field)
        stp = getattr(cfg, step_field)
        for field, value in ((token_field, tok), (step_field, stp)):
            if value is not None and value <= 0:
                raise ValueError(f"{field} must be a positive integer, got {value}")
        if tok is not None and stp is not None:
            raise ValueError(
                f"Set exactly one of {token_field} or {step_field} for the {label} budget, not both."
            )
        if tok is None and stp is None:
            raise ValueError(
                f"{sched} requires a {label} budget: set {token_field} (tokens) or {step_field} (steps)."
            )

    if sched == "wsd":
        _one("stable_tokens", "stable_steps", "stable")
        _one("decay_tokens", "decay_steps", "decay")
        if (cfg.train_tokens is not None) or (cfg.max_steps is not None):
            raise ValueError("wsd uses stable_*/decay_* budgets; remove train_tokens/max_steps.")
    else:
        _one("train_tokens", "max_steps", "training")
        if any(getattr(cfg, f) is not None for f in
               ("stable_tokens", "decay_tokens", "stable_steps", "decay_steps")):
            raise ValueError(f"{sched} uses train_tokens/max_steps; remove stable_*/decay_* budgets.")


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
            "global_batch_tokens, micro_batch_seqs, max_seq_len, and world_size. "
            "Remove gradient_accumulation_steps from your pretraining config."
        )

    # Required key
    if "dataset_dir" not in normalized_config or not normalized_config["dataset_dir"]:
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
        "noble_base_lr",
        "noble_lr_gamma",
        "noble_lr_gamma_M",
        "noble_freq_lr_mult",
        "noble_phase_lr_mult",
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
            raise ValueError(
                f"Invalid {field_name} value: {raw_value}. Must be a number."
            ) from exc

    for field in float_fields:
        if field in kwargs and kwargs[field] is not None:
            kwargs[field] = _parse_float_like(kwargs[field], field)

    # Ensure warmup-related fields are ints (YAML may load as int or float).
    for warmup_key in (
        "warmup_steps",
        "repo_rope_warmup_steps",
        "global_batch_warmup_steps",
    ):
        if warmup_key in kwargs and kwargs[warmup_key] is not None:
            try:
                kwargs[warmup_key] = int(kwargs[warmup_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {warmup_key} value: {kwargs[warmup_key]}. Must be an integer."
                ) from exc

    # Handle boolean values
    for bool_key in [
        "multi_gpu",
        "bf16",
        "tf32",
        "fp8",
        "muon_nesterov",
        "muon_cautious_weight_decay",
        "muon_use_polar_express",
        "use_packing",
        "use_static_inp_size",
        "use_compile_max_autotune",
        "compile_triton_persistent_reductions",
        "compile_triton_mix_order_reduction",
        "profiler_enabled",
        "debug_non_finite_params",
        "debug_layerwise_metrics",
    ]:
        if bool_key in kwargs:
            value = kwargs[bool_key]
            if isinstance(value, bool):
                continue
            elif isinstance(value, str):
                kwargs[bool_key] = value.lower() == "true"

    cfg = PretrainingConfig(**kwargs)
    _validate_duration_config(cfg)
    return cfg


def _load_manifest_and_inject(
    pretrain_config: PretrainingConfig,
    model_config: ProtModernBertMLMConfig,
) -> DataManifest:
    """Read the dataset manifest, validate pipeline mode, and inject
    num_register_tokens into the model config so the tokenizer is built
    with the same value used at data-prep time.

    Returns the manifest so it can be passed to the pipeline entry point.
    """
    manifest = read_manifest(pretrain_config.dataset_dir)
    validate_manifest_for_pipeline(manifest, expected_mode="pretrain")
    model_config.num_register_tokens = manifest.num_register_tokens
    logger.info(
        f"Injected manifest.num_register_tokens = {manifest.num_register_tokens} "
        "into model_config"
    )
    return manifest


def _load_model_config(config: Dict[str, Any]) -> ProtModernBertMLMConfig:
    if config is None:
        raise ValueError("Model configuration is required but not found in YAML")

    config = dict(config)

    model_fields = {f.name: f for f in fields(ProtModernBertMLMConfig)}
    expected_keys = set(model_fields.keys())
    present_keys = set(config.keys())

    extra = []
    kwargs: Dict[str, Any] = {}

    # Classify provided keys in one pass
    for key in present_keys:
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


_RESUME_ALLOWED_KEYS = {
    "is_resume",
    "checkpoint_dir",
    "mode",
    "decay_tokens",
    "decay_steps",
    "decay_shape",
    "lr_decay_to_fraction",
}


def _load_resume_config(config: Dict[str, Any]) -> ResumeConfig:
    if config is None:
        return ResumeConfig(is_resume=False, checkpoint_dir="")

    present_keys = set(config.keys())
    extra = sorted(present_keys - _RESUME_ALLOWED_KEYS)
    if extra:
        raise ValueError(
            f"Unexpected resume configuration keys: {', '.join(extra)}"
        )

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
            raise ValueError(
                f"Invalid {field_name} value: {raw_value}. Must be a number."
            ) from exc

    kwargs: Dict[str, Any] = {
        "is_resume": False,
        "checkpoint_dir": "",
    }
    for key in present_keys:
        value = config.get(key)
        # Keep None values for Optional fields — they mean "use default /
        # inherit from checkpoint".  Only skip None for required fields
        # (is_resume, checkpoint_dir).
        if value is None and key in ("is_resume", "checkpoint_dir"):
            continue
        kwargs[key] = value

    for int_key in ("decay_tokens", "decay_steps"):
        if int_key in kwargs and kwargs[int_key] is not None:
            try:
                kwargs[int_key] = int(kwargs[int_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {int_key} value: {kwargs[int_key]}. Must be an integer."
                ) from exc

    if "lr_decay_to_fraction" in kwargs and kwargs["lr_decay_to_fraction"] is not None:
        kwargs["lr_decay_to_fraction"] = _parse_float_like(
            kwargs["lr_decay_to_fraction"], "lr_decay_to_fraction"
        )

    if "decay_shape" in kwargs and kwargs["decay_shape"] is not None:
        decay_shape = str(kwargs["decay_shape"]).strip().lower()
        if decay_shape not in {"1-sqrt", "linear"}:
            raise ValueError(
                f"Invalid decay_shape value: {kwargs['decay_shape']}. Must be one of: 1-sqrt, linear."
            )
        kwargs["decay_shape"] = decay_shape

    if "mode" in kwargs and kwargs["mode"] is not None:
        kwargs["mode"] = str(kwargs["mode"]).strip().lower()

    if "is_resume" in kwargs:
        value = kwargs["is_resume"]
        if isinstance(value, str):
            kwargs["is_resume"] = value.lower() == "true"

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

    return ResumeConfig(**kwargs)


def _validate_resume_config_modes(resume_config: ResumeConfig) -> None:
    """Validate the continue/decay resume contract (spec section 7)."""
    if not resume_config.is_resume:
        return
    mode = (resume_config.mode or "").lower()
    if mode not in {"continue", "decay"}:
        raise ValueError(f"resume.mode must be 'continue' or 'decay', got {resume_config.mode!r}")
    has_tok = resume_config.decay_tokens is not None
    has_stp = resume_config.decay_steps is not None
    if mode == "continue":
        set_fields = [
            name
            for name in ("decay_tokens", "decay_steps", "decay_shape", "lr_decay_to_fraction")
            if getattr(resume_config, name) is not None
        ]
        if set_fields:
            raise ValueError(
                f"resume.mode=continue takes no decay_* fields, but got: "
                f"{', '.join(set_fields)} (they belong to mode=decay)."
            )
        return
    # mode == "decay"
    if has_tok and has_stp:
        raise ValueError("resume decay: set exactly one of decay_tokens or decay_steps, not both.")
    if not has_tok and not has_stp:
        raise ValueError("resume.mode=decay requires a budget: set decay_tokens or decay_steps.")


def _set_seed_for_init(seed: int) -> None:
    """Set seed before model creation so both pipelines start with identical weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
