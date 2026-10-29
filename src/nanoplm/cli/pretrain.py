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

from nanoplm.pretraining.models.modern_bert.model import (
    ProtModernBertMLM,
    ProtModernBertMLMConfig,
)
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.data.infra.validation import validate_pretrain_dataset
from nanoplm.data.infra.manifest import read_manifest, validate_manifest_for_pipeline, DataManifest
from nanoplm.utils.common import read_yaml, create_dirs, is_flash_attention_available
from nanoplm.cli.templates import load_template
from nanoplm.cli._common import (
    YAML_FIRST_SECTIONS,
    ConfigErrorGroup,
    log_level_option,
    unknown_keys_message,
)
from nanoplm.utils.logger import logger


def _check_muon_available(optimizer: str) -> None:
    """Abort early when a Muon variant is requested but ``dion`` is not installed."""
    optimizer_name = optimizer.lower()
    if optimizer_name in {"muon", "normuon", "nordion2"}:
        try:
            import dion
        except ImportError:
            raise click.ClickException(
                f"Optimizer '{optimizer}' requires the 'dion' package which is not installed.\n"
                "Install it with:  pip install nanoplm[cuda]"
            )
        required_class = {"muon": "Muon", "normuon": "NorMuon", "nordion2": "NorDion2"}[optimizer_name]
        if not hasattr(dion, required_class):
            raise click.ClickException(
                f"Optimizer '{optimizer}' requires dion.{required_class}, but this Dion build does not expose it."
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
    if mode not in {"fsdp", "hsdp", "ddp"}:
        raise click.ClickException(
            f"Unsupported distributed_mode={value!r}. Expected one of: fsdp, hsdp, ddp."
        )
    return mode


def _validate_distributed_mode(
    cfg: PretrainingConfig,
    *,
    pure_torch: bool,
) -> None:
    cfg.distributed_mode = _normalize_distributed_mode(cfg.distributed_mode)
    cfg.ddp_bucket_cap_mb = int(cfg.ddp_bucket_cap_mb)
    if cfg.ddp_bucket_cap_mb <= 0:
        raise click.ClickException("ddp_bucket_cap_mb must be > 0.")
    if not pure_torch:
        return
    if cfg.distributed_mode in {"ddp", "hsdp"} and not cfg.multi_gpu:
        raise click.ClickException(
            f"distributed_mode={cfg.distributed_mode} requires multi_gpu=true."
        )
    if cfg.distributed_mode not in {"fsdp", "hsdp"} and cfg.fsdp_reshard_after_forward:
        raise click.ClickException(
            "fsdp_reshard_after_forward applies only when distributed_mode=fsdp or hsdp."
        )
    if cfg.distributed_mode != "ddp" and cfg.ddp_bucket_cap_mb != 25:
        raise click.ClickException(
            "ddp_bucket_cap_mb applies only when distributed_mode=ddp."
        )
    cfg.fsdp_reduce_dtype = str(cfg.fsdp_reduce_dtype).lower()
    if cfg.fsdp_reduce_dtype not in {"fp32", "bf16"}:
        raise click.ClickException("fsdp_reduce_dtype must be fp32 or bf16.")


def _check_pipeline_supported(pure_torch: bool) -> None:
    """Pretraining requires pure_torch (spec section 14)."""
    if not pure_torch:
        raise ValueError(
            "Pretraining requires pure_torch: true. "
            "The HF Trainer pretraining path is no longer supported."
        )


@click.group(name="pretrain", cls=ConfigErrorGroup, sections=YAML_FIRST_SECTIONS)
@click.help_option("--help", "-h")
def pretrain():
    """Group of commands for model pretraining."""
    pass


@pretrain.command("run")
@log_level_option
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
    default=128,
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
    default=True,
    help="Enable gradient clipping with max norm 1.0",
)
@click.option(
    "--warmup-steps",
    type=int,
    default=1000,
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
    type=click.Choice(
        ["wsd", "warmup_stable", "cosine", "linear"], case_sensitive=False
    ),
    default="wsd",
    help="LR schedule after warmup. wsd: warmup->stable->decay (use --stable-tokens/"
    "--decay-tokens); warmup_stable/cosine/linear: single phase (use --train-tokens).",
)
@click.option(
    "--train-tokens",
    type=int,
    default=None,
    help="Single-phase token budget (warmup_stable/cosine/linear). "
    "Set exactly one of --train-tokens or --max-steps for the training budget.",
)
@click.option(
    "--stable-tokens",
    type=int,
    default=None,
    help="wsd only: tokens up to the pre-decay checkpoint (warmup included). "
    "Set exactly one of --stable-tokens or --stable-steps for the stable budget.",
)
@click.option(
    "--decay-tokens",
    type=int,
    default=None,
    help="wsd only: additional anneal token budget. "
    "Set exactly one of --decay-tokens or --decay-steps for the decay budget.",
)
@click.option(
    "--max-steps",
    type=int,
    default=None,
    help="Single-phase step override for --train-tokens (warmup_stable/cosine/linear). "
    "Set exactly one of --train-tokens or --max-steps for the training budget.",
)
@click.option(
    "--stable-steps",
    type=int,
    default=None,
    help="wsd only: step override for --stable-tokens. "
    "Set exactly one of --stable-tokens or --stable-steps for the stable budget.",
)
@click.option(
    "--decay-steps",
    type=int,
    default=None,
    help="wsd only: step override for --decay-tokens. "
    "Set exactly one of --decay-tokens or --decay-steps for the decay budget.",
)
@click.option(
    "--decay-shape",
    type=click.Choice(["1-sqrt", "linear"], case_sensitive=False),
    default="1-sqrt",
    show_default=True,
    help="wsd/decay anneal shape: 1-sqrt | linear.",
)
@click.option(
    "--global-batch-tokens",
    type=int,
    default=1024000,
    help="Target tokens per optimizer step (grad_accum inferred automatically)",
)
@click.option(
    "--optimizer",
    type=click.Choice(
        ["adamw", "stable_adamw", "muon", "normuon", "nordion2"], case_sensitive=False
    ),
    default="normuon",
    help="Optimizer to use",
)
@click.option("--adam-beta1", type=float, default=0.9, help="Adam beta1")
@click.option("--adam-beta2", type=float, default=0.95, help="Adam beta2")
@click.option("--adam-epsilon", type=float, default=1e-8, help="Adam epsilon")
@click.option(
    "--muon-learning-rate",
    type=float,
    default=1e-3,
    help="Matrix-optimizer LR (used when optimizer=muon, normuon, or nordion2; learning-rate remains AdamW LR)",
)
@click.option(
    "--muon-weight-decay",
    type=float,
    default=0.01,
    help="Matrix-optimizer weight decay (used when optimizer=muon, normuon, or nordion2)",
)
@click.option(
    "--muon-cautious-weight-decay/--no-muon-cautious-weight-decay",
    default=True,
    help="Enable cautious weight decay where supported by the selected Muon-family optimizer",
)
@click.option(
    "--muon-use-polar-express/--no-muon-use-polar-express",
    default=True,
    help="Use Polar Express orthogonalization for Muon-family optimizers where supported",
)
@click.option(
    "--muon-momentum",
    type=float,
    default=0.95,
    help="Muon-family momentum",
)
@click.option(
    "--muon-nesterov/--no-muon-nesterov",
    default=True,
    help="Enable Nesterov in Muon-family optimizers",
)
@click.option(
    "--muon-eps",
    type=float,
    default=1e-7,
    help="Muon-family epsilon",
)
@click.option(
    "--muon-adjust-lr",
    type=click.Choice(["rms_norm", "spectral_norm", "none"], case_sensitive=False),
    default="rms_norm",
    help="Update LR adjustment for the matrix optimizer (muon/normuon/nordion2).",
)
@click.option(
    "--muon-beta2",
    type=float,
    default=0.95,
    help="Second-moment beta for NorMuon/NorDion2 (ignored by plain Muon)",
)
@click.option(
    "--muon-use-gram-newton-schulz/--no-muon-use-gram-newton-schulz",
    default=True,
    help="Use Gram Newton-Schulz orthogonalization (default; overrides --muon-use-polar-express)",
)
@click.option(
    "--muon-triton-post-ortho/--no-muon-triton-post-ortho",
    default=False,
    help="NorDion2: run the post-orthogonalization step in Triton",
)
@click.option(
    "--muon-flatten/--no-muon-flatten",
    default=False,
    help="Flatten >2D matrix params before orthogonalization where supported",
)
@click.option(
    "--nordion2-fraction",
    type=float,
    default=0.25,
    help="NorDion2 low-rank fraction (used only when optimizer=nordion2)",
)
@click.option(
    "--nordion2-selection-scope",
    type=click.Choice(["local", "global"], case_sensitive=False),
    default="local",
    help="NorDion2 FSDP2 row-shard selection scope (used only when optimizer=nordion2)",
)
@click.option("--mlm-probability", type=float, default=0.3, help="MLM probability")
@click.option(
    "--mlm-masking-strategy",
    type=click.Choice(["token", "span"], case_sensitive=False),
    default="token",
    show_default=True,
    help="MLM position sampling strategy: independent token masking or contiguous span masking.",
)
@click.option(
    "--mlm-span-mean-length",
    type=click.FloatRange(min=0.0, min_open=True),
    default=3.0,
    show_default=True,
    help="Mean span length for --mlm-masking-strategy span. Values <=1 produce length-1 spans.",
)
@click.option(
    "--mlm-span-max-length",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Maximum span length for --mlm-masking-strategy span.",
)
@click.option(
    "--mlm-standard-amino-acids-only/--mlm-all-non-special-tokens",
    default=True,
    help=(
        "Current behavior: supervise MLM only on the 20 standard amino acids "
        "and draw random replacements from those same tokens. Use "
        "--mlm-all-non-special-tokens to reproduce the legacy all-non-special "
        "token behavior."
    ),
)
@click.option(
    "--logging-steps", type=int, default=1, help="Number of steps between log events"
)
@click.option(
    "--eval-steps", type=int, default=1000, help="Number of steps between evaluations"
)
@click.option(
    "--save-steps",
    type=int,
    default=2500,
    help="Number of steps between checkpoint saves",
)
@click.option(
    "--max-wallclock-hours",
    type=float,
    default=None,
    help="Stop training after this many hours, checkpointing at the first "
    "optimizer step past the deadline, even if the token budget is unspent. "
    "For wall-clock-matched ablations. The clock starts a couple of optimizer "
    "steps in, so compile/warmup is excluded and total job time is longer; it "
    "also restarts on resume rather than accumulating.",
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option(
    "--debug-non-finite-params/--no-debug-non-finite-params",
    default=False,
    help="Check parameters for non-finite values after every optimizer step.",
)
@click.option(
    "--debug-layerwise-metrics/--no-debug-layerwise-metrics",
    default=False,
    help="Log per-layer grad/weight norms, residual-stream RMS, and logit stats. "
    "Activation stats are computed inside the compiled graph; small overhead. "
    "Residual RMS keys are block_XXX_rep{r} on every run (only rep0 without Loopie).",
)
@click.option(
    "--debug-layerwise-log-every",
    type=int,
    default=100,
    help="Log layerwise debug metrics every N optimizer steps.",
)
@click.option(
    "--debug-memory-snapshot/--no-debug-memory-snapshot",
    default=False,
    show_default=True,
    help="Pure-torch only: record CUDA allocator history and dump a memory_viz pickle to "
    "<run root>/memory_snapshots/ on OOM (and at --debug-memory-snapshot-step). "
    "All ranks record; LOCAL_RANK 0 takes the scheduled dump, an OOM dumps on the rank "
    "that hit it.",
)
@click.option(
    "--debug-memory-snapshot-step",
    type=int,
    default=-1,
    show_default=True,
    help="Also dump a memory snapshot right after this optimizer step; -1 dumps only on OOM "
    "(only when --debug-memory-snapshot).",
)
@click.option(
    "--debug-memory-snapshot-max-entries",
    type=int,
    default=100000,
    show_default=True,
    help="Allocator events kept in the memory-history ring buffer "
    "(only when --debug-memory-snapshot).",
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
    help="Enable FP8 Linear matmuls in the pure-torch path (CUDA only, best on H100+)",
)
@click.option(
    "--fp8-target",
    type=str,
    default="mlp",
    show_default=True,
    help=(
        "Pure-torch FP8 target(s): mlp, attn_wqkv, attn_wo. "
        "Combine with commas, e.g. attn_wqkv,attn_wo."
    ),
)
@click.option(
    "--fp8-skip-last-n-layers",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Pure-torch FP8: leave the final N transformer layers in bf16.",
)
@click.option(
    "--fp8-recipe",
    type=click.Choice(["delayed", "current", "block", "mxfp8", "nvfp4"], case_sensitive=False),
    default="delayed",
    show_default=True,
    help="Transformer Engine FP8 recipe. mxfp8 and nvfp4 require Blackwell+ GPUs.",
)
@click.option(
    "--fp8-format",
    type=click.Choice(["hybrid", "e4m3"], case_sensitive=False),
    default="hybrid",
    show_default=True,
    help="Transformer Engine FP8 format for recipes that accept one.",
)
@click.option(
    "--fp8-amax-history-len",
    type=int,
    default=64,
    show_default=True,
    help="Transformer Engine DelayedScaling amax history length.",
)
@click.option(
    "--fp8-reduce-amax/--no-fp8-reduce-amax",
    default=True,
    show_default=True,
    help="Transformer Engine DelayedScaling: synchronize amax across the FP8 reduction group.",
)
@click.option(
    "--multi-gpu", is_flag=True, default=True, help="Enable multi-GPU training"
)
@click.option(
    "--distributed-mode",
    type=click.Choice(["fsdp", "ddp"], case_sensitive=False),
    default="fsdp",
    show_default=True,
    help="Distributed wrapper for multi-GPU runs.",
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
    help="Universal FSDP2 reshard_after_forward setting for all fully_shard groups.",
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
@click.option(
    "--profiler-enabled/--no-profiler-enabled",
    default=False,
    show_default=True,
    help="Profile a step window and export a Chrome trace to --ckp-dir.",
)
@click.option(
    "--profiler-start-step",
    type=int,
    default=18,
    show_default=True,
    help="First optimizer step captured by the profiler (only when --profiler-enabled).",
)
@click.option(
    "--profiler-end-step",
    type=int,
    default=22,
    show_default=True,
    help="Last optimizer step captured by the profiler (only when --profiler-enabled). "
    "At most 2 steps are actually captured.",
)
@click.option(
    "--profiler-record-shapes/--no-profiler-record-shapes",
    default=False,
    show_default=True,
    help="Record per-op input shapes in the PyTorch profiler trace. Adds dispatch-path "
    "bookkeeping, so such a trace is not a steady-state wall-clock reference.",
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
@click.option("--hidden-size", type=int, default=1024, help="Model hidden size")
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
    "--num-kv-heads",
    type=int,
    default=8,
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
    default="rmsnorm",
    help="Normalization type: layernorm (default) or rmsnorm (slightly faster, no bias)",
)
@click.option(
    "--mlp-activation",
    type=click.Choice(["swiglu", "geglu", "srelu"], case_sensitive=False),
    default="swiglu",
    help="MLP activation",
)
@click.option("--mlp-dropout", type=float, default=0.0, help="MLP dropout")
@click.option("--mlp-bias", is_flag=True, default=False, help="Use MLP bias")
@click.option(
    "--no-mlp-on-first-layer/--mlp-on-first-layer",
    default=False,
    help="Disable the MLP branch in encoder layer 0",
)
@click.option("--attention-bias", is_flag=True, default=False, help="Use attn bias")
@click.option("--attention-dropout", type=float, default=0.0, help="Attn dropout")
@click.option(
    "--classifier-activation",
    type=click.Choice(["relu", "gelu", "srelu"], case_sensitive=False),
    default="gelu",
    help="Classifier activation (srelu is supported only in the pure-torch path)",
)
@click.option(
    "--use-resid-lambdas/--no-use-resid-lambdas",
    default=False,
    help="Enable per-layer residual scaling (resid_lambdas). Not compatible with --use-mhc-lite.",
)
@click.option(
    "--use-x0-lambdas/--no-use-x0-lambdas",
    default=False,
    help="Enable per-layer x0 shortcut scaling (x0_lambdas)",
)
@click.option(
    "--use-qk-norm/--no-use-qk-norm",
    default=False,
    help="Enable NanoChat-style RMS QK normalization in attention",
)
@click.option(
    "--global-rope-theta",
    type=float,
    default=40000.0,
    show_default=True,
    help="RoPE base for full-attention layers; lower (e.g. 10000-40000) suits short max_seq_len",
)
@click.option(
    "--reorder-rope-qknorm/--no-reorder-rope-qknorm",
    default=False,
    help="Apply QK norm BEFORE RoPE (LLaMA-style, better kernel fusion) instead of after. "
    "Only has effect when --use-qk-norm is set.",
)
@click.option(
    "--use-canon-layers/--no-use-canon-layers",
    default=False,
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
    default=5,
    show_default=True,
    help="Canon kernel size. Allowed values: {3,5,7}.",
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
    default="attn",
    show_default=True,
    help="Checkpoint scope. 'layer' checkpoints the whole transformer layer; "
    "'attn' checkpoints only the attention residual branch (usually the best tradeoff); "
    "'attn+mlp' checkpoints both attention and MLP branches.",
)
@click.option(
    "--tie-word-embeddings/--no-tie-word-embeddings",
    default=False,
    help="Tie input embedding and output decoder weights (default: disabled).",
)
@click.option(
    "--use-diff-attn-v2/--no-use-diff-attn-v2",
    default=False,
    help="Enable Differential Attention V2: doubles query heads, applies differential "
    "subtraction for noise cancellation in attention (pure-torch only).",
)
@click.option(
    "--use-paired-head-attention/--no-use-paired-head-attention",
    default=False,
    help="Enable paired-head attention (pure-torch only). "
    "Mutually exclusive with --use-diff-attn-v2.",
)
@click.option(
    "--attn-layer-pattern",
    type=str,
    default="F",
    help="Attention layer pattern string using F (full) and S (sliding), e.g. 'FSS'. "
    "Tiles to cover all layers. Overrides --global-attn-every-n-layers when set.",
)
@click.option(
    "--fused-qkv/--no-fused-qkv",
    default=True,
    help="Use a single fused Wqkv projection (default). When disabled, Q/K/V run as separate matmuls.",
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
    default="kaiming",
    show_default=True,
    help="Initialization for residual out-projections (attn.Wo and mlp.Wo).",
)
@click.option(
    "--moe-kernel-backend",
    type=click.Choice(["cutlass", "sonicmoe"], case_sensitive=False),
    default="sonicmoe",
    show_default=True,
    help="Routed-expert compute backend for MoE layers (sonicmoe = SonicMoE/quack fused path).",
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
    "--compile-mode",
    type=click.Choice(
        ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        case_sensitive=False,
    ),
    default=None,
    help="Pure-torch only: explicit torch.compile(mode=...). Unset uses Inductor's default "
    "mode (or max-autotune-no-cudagraphs when --use-compile-max-autotune); set it and it "
    "overrides that flag. 'reduce-overhead' enables CUDA graphs.",
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
    train_tokens: Optional[int],
    stable_tokens: Optional[int],
    decay_tokens: Optional[int],
    max_steps: Optional[int],
    stable_steps: Optional[int],
    decay_steps: Optional[int],
    decay_shape: str,
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
    muon_adjust_lr: str,
    muon_beta2: float,
    muon_use_gram_newton_schulz: bool,
    muon_triton_post_ortho: bool,
    muon_flatten: bool,
    nordion2_fraction: float,
    nordion2_selection_scope: str,
    mlm_probability: float,
    mlm_masking_strategy: str,
    mlm_span_mean_length: float,
    mlm_span_max_length: int,
    mlm_standard_amino_acids_only: bool,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    max_wallclock_hours: Optional[float],
    seed: int,
    debug_non_finite_params: bool,
    debug_layerwise_metrics: bool,
    debug_layerwise_log_every: int,
    debug_memory_snapshot: bool,
    debug_memory_snapshot_step: int,
    debug_memory_snapshot_max_entries: int,
    mask_replace_prob: float,
    random_token_prob: float,
    keep_probability: float,
    num_workers: Union[int, str],
    prefetch_factor: int,
    compile: bool,
    use_packing: bool,
    use_static_inp_size: bool,
    use_compile_max_autotune: bool,
    compile_mode: Optional[str],
    compile_triton_persistent_reductions: bool,
    compile_triton_mix_order_reduction: bool,
    bf16: bool,
    tf32: bool,
    fp8: bool,
    fp8_target: str,
    fp8_skip_last_n_layers: int,
    fp8_recipe: str,
    fp8_format: str,
    fp8_amax_history_len: int,
    fp8_reduce_amax: bool,
    multi_gpu: bool,
    distributed_mode: str,
    fsdp_shard_granularity: str,
    fsdp_reshard_after_forward: bool,
    world_size: str,
    ddp_bucket_cap_mb: int,
    project_name: str,
    run_name: str,
    profiler_enabled: bool,
    profiler_start_step: int,
    profiler_end_step: int,
    profiler_record_shapes: bool,
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
    global_rope_theta: float,
    reorder_rope_qknorm: bool,
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
    moe_kernel_backend: str,
    use_mhc_lite: bool,
    mhc_n_streams: int,
    mhc_lite_wrapping_level: str,
    pure_torch: bool,
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
        train_tokens=train_tokens,
        stable_tokens=stable_tokens,
        decay_tokens=decay_tokens,
        max_steps=max_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        decay_shape=decay_shape,
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
        muon_adjust_lr=muon_adjust_lr,
        muon_beta2=muon_beta2,
        muon_use_gram_newton_schulz=muon_use_gram_newton_schulz,
        muon_triton_post_ortho=muon_triton_post_ortho,
        muon_flatten=muon_flatten,
        nordion2_fraction=nordion2_fraction,
        nordion2_selection_scope=nordion2_selection_scope,
        mlm_probability=mlm_probability,
        mlm_masking_strategy=mlm_masking_strategy,
        mlm_span_mean_length=mlm_span_mean_length,
        mlm_span_max_length=mlm_span_max_length,
        mlm_standard_amino_acids_only=mlm_standard_amino_acids_only,
        mask_replace_prob=mask_replace_prob,
        random_token_prob=random_token_prob,
        keep_probability=keep_probability,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        max_wallclock_hours=max_wallclock_hours,
        seed=seed,
        debug_non_finite_params=debug_non_finite_params,
        debug_layerwise_metrics=debug_layerwise_metrics,
        debug_layerwise_log_every=debug_layerwise_log_every,
        debug_memory_snapshot=debug_memory_snapshot,
        debug_memory_snapshot_step=debug_memory_snapshot_step,
        debug_memory_snapshot_max_entries=debug_memory_snapshot_max_entries,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        use_packing=use_packing,
        use_static_inp_size=use_static_inp_size,
        use_compile_max_autotune=use_compile_max_autotune,
        compile_mode=compile_mode,
        compile_triton_persistent_reductions=compile_triton_persistent_reductions,
        compile_triton_mix_order_reduction=compile_triton_mix_order_reduction,
        bf16=bf16,
        tf32=tf32,
        fp8=fp8,
        fp8_target=fp8_target,
        fp8_skip_last_n_layers=fp8_skip_last_n_layers,
        fp8_recipe=fp8_recipe,
        fp8_format=fp8_format,
        fp8_amax_history_len=fp8_amax_history_len,
        fp8_reduce_amax=fp8_reduce_amax,
        multi_gpu=multi_gpu,
        distributed_mode=distributed_mode,
        fsdp_shard_granularity=fsdp_shard_granularity,
        fsdp_reshard_after_forward=fsdp_reshard_after_forward,
        world_size=world_size,
        ddp_bucket_cap_mb=ddp_bucket_cap_mb,
        project_name=project_name,
        run_name=run_name,
        profiler_enabled=profiler_enabled,
        profiler_start_step=profiler_start_step,
        profiler_end_step=profiler_end_step,
        profiler_record_shapes=profiler_record_shapes,
    )

    if use_canon_layers and not pure_torch:
        raise click.ClickException(
            "use_canon_layers requires --pure-torch. "
            "Canon layers are not implemented in the HF path."
        )
    if classifier_activation.lower() == "srelu" and not pure_torch:
        raise click.ClickException(
            "classifier_activation=srelu requires --pure-torch. "
            "The default HF path supports only relu/gelu."
        )
    if (
        num_kv_heads is not None
        and num_kv_heads != num_attention_heads
        and not pure_torch
    ):
        raise click.ClickException(
            "GQA requires --pure-torch. "
            "The default HF path does not implement num_kv_heads != num_attention_heads."
        )
    if (use_mhc_lite or mhc_lite_wrapping_level.lower() != "layer") and not pure_torch:
        raise click.ClickException(
            "mHC-lite requires --pure-torch. "
            "Set that mode (or disable mHC-lite settings)."
        )
    if use_mhc_lite and use_resid_lambdas:
        raise click.ClickException(
            "use_mhc_lite and use_resid_lambdas are mutually exclusive. "
            "resid_lambdas scales the hidden state before each layer, which breaks "
            "mHC-lite's doubly-stochastic stability guarantees."
        )
    _validate_distributed_mode(cfg, pure_torch=pure_torch)

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
        global_rope_theta=global_rope_theta,
        reorder_RoPE_QKNorm=reorder_rope_qknorm,
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
        moe_kernel_backend=moe_kernel_backend.lower(),
    )

    _set_seed_for_init(seed)
    if use_diff_attn_v2 and not pure_torch:
        raise click.ClickException(
            "use_diff_attn_v2 requires --pure-torch. "
            "Differential Attention V2 is not implemented in the HF path."
        )
    if use_paired_head_attention and not pure_torch:
        raise click.ClickException(
            "use_paired_head_attention requires --pure-torch. "
            "Paired head attention is not implemented in the HF path."
        )
    if use_paired_head_attention and use_diff_attn_v2:
        raise click.ClickException(
            "use_paired_head_attention and use_diff_attn_v2 are mutually exclusive."
        )
    # Fail fast before any heavy manifest/model instantiation: only the
    # pure_torch pipeline is supported. The HF Trainer pretraining path is
    # no longer supported.
    try:
        _check_pipeline_supported(pure_torch)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        _validate_duration_config(cfg)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    manifest = _load_manifest_and_inject(cfg, model_cfg)
    if pure_torch:
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

    if pure_torch:
        run_pure_pretraining(
            model=model,
            pretrain_config=cfg,
            manifest=manifest,
            resume_config=resume_cfg if resume_cfg.is_resume else None,
        )


@pretrain.command("from-yaml")
@log_level_option
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
def from_yaml(config: str, pure_torch: bool):
    """Run pretraining from a YAML file with training and model parameters.

        model: {...}
        pretraining: {...}
        resume: {...}

    If resume.is_resume is True, training will resume from the given
    checkpoint using the hyperparameters in the 'pretraining' block.
    """
    pretrain_yaml = Path(config).resolve()

    raw = read_yaml(pretrain_yaml)

    # Allow both nested and flat formats; prefer nested under key 'training'
    pretrain_dict = raw.get("pretraining")
    model_dict = raw.get("model")
    resume_dict = raw.get("resume")

    # Support pure_torch from YAML or CLI flag (CLI flag takes precedence)
    if not pure_torch:
        pure_torch = bool(raw.get("pure_torch", False))

    # validate and load config
    pretrain_config = _load_pretrain_config(pretrain_dict)
    # A relative dataset_dir or ckp_dir resolves against the process working
    # directory, not against the config file. Resolving here is what makes the
    # downstream "Validating pretraining dataset: ..." log name a path a reader
    # can still find weeks later.
    pretrain_config.dataset_dir = str(Path(pretrain_config.dataset_dir).resolve())
    pretrain_config.ckp_dir = str(Path(pretrain_config.ckp_dir).resolve())
    _check_muon_available(pretrain_config.optimizer)

    model_config = _load_model_config(model_dict)
    resume_config = _load_resume_config(resume_dict)
    _validate_resume_config_modes(resume_config)
    if getattr(model_config, "use_noble", False) and not pure_torch:
        raise click.ClickException(
            "model.use_noble=true requires pure_torch: true (or --pure-torch). "
            "NOBLE is not implemented in the HF path."
        )
    if model_config.use_canon_layers and not pure_torch:
        raise click.ClickException(
            "model.use_canon_layers=true requires pure_torch: true (or --pure-torch). "
            "Canon layers are not implemented in the HF path."
        )
    if (
        model_config.use_mhc_lite
        or str(getattr(model_config, "mhc_lite_wrapping_level", "layer")).lower()
        != "layer"
    ) and not pure_torch:
        raise click.ClickException(
            "model.use_mhc_lite=true (or model.mhc_lite_wrapping_level != 'layer') "
            "requires pure_torch: true / --pure-torch. "
            "mHC-lite is not implemented in the HF path."
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
            "Differential Attention V2 is not implemented in the HF path."
        )
    if getattr(model_config, "use_paired_head_attention", False) and not pure_torch:
        raise click.ClickException(
            "model.use_paired_head_attention=true requires pure_torch: true (or --pure-torch). "
            "Paired head attention is not implemented in the HF path."
        )
    if getattr(model_config, "use_paired_head_attention", False) and getattr(
        model_config, "use_diff_attn_v2", False
    ):
        raise click.ClickException(
            "model.use_paired_head_attention=true is not compatible with model.use_diff_attn_v2=true."
        )
    if str(
        getattr(model_config, "classifier_activation", "gelu")
    ).lower() == "srelu" and not pure_torch:
        raise click.ClickException(
            "model.classifier_activation=srelu requires pure_torch: true / --pure-torch. "
            "The HF path supports only relu/gelu."
        )
    if (
        getattr(model_config, "num_kv_heads", None) is not None
        and model_config.num_kv_heads != model_config.num_attention_heads
        and not pure_torch
    ):
        raise click.ClickException(
            "model.num_kv_heads != model.num_attention_heads requires "
            "pure_torch: true / --pure-torch. The HF path does not implement GQA."
        )
    _validate_distributed_mode(pretrain_config, pure_torch=pure_torch)
    _populate_batch_setup(pretrain_config)

    # Fail fast before any heavy model instantiation: only the pure_torch
    # pipeline is supported.
    _check_pipeline_supported(pure_torch)

    manifest = _load_manifest_and_inject(pretrain_config, model_config)
    _set_seed_for_init(pretrain_config.seed)
    if pure_torch:
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

    if pure_torch:
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

    output_path.write_text(load_template("pretrain"), encoding="utf-8")
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
        raise ValueError(unknown_keys_message("training", extra, expected_keys))

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
        "muon_beta2",
        "nordion2_fraction",
        "max_wallclock_hours",
        "min_lr",
        "noble_base_lr",
        "noble_lr_gamma",
        "noble_lr_gamma_M",
        "noble_freq_lr_mult",
        "noble_phase_lr_mult",
        "mlm_probability",
        "mask_replace_prob",
        "random_token_prob",
        "keep_probability",
        "mlm_span_mean_length",
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

    # Ensure integer-like fields are ints (YAML may load as int, float, or str).
    for int_key in (
        "warmup_steps",
        "repo_rope_warmup_steps",
        "global_batch_warmup_steps",
        "fp8_amax_history_len",
        "fp8_skip_last_n_layers",
        "mlm_span_max_length",
        "debug_memory_snapshot_step",
        "debug_memory_snapshot_max_entries",
    ):
        if int_key in kwargs and kwargs[int_key] is not None:
            try:
                kwargs[int_key] = int(kwargs[int_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {int_key} value: {kwargs[int_key]}. Must be an integer."
                ) from exc

    # Handle boolean values
    for bool_key in [
        "multi_gpu",
        "bf16",
        "tf32",
        "fp8",
        "fp8_reduce_amax",
        "fp8_cpu_activation_offload",
        "fp8_cpu_activation_offload_retain_buffers",
        "muon_nesterov",
        "muon_cautious_weight_decay",
        "muon_use_polar_express",
        "muon_use_gram_newton_schulz",
        "muon_triton_post_ortho",
        "muon_flatten",
        "use_packing",
        "use_static_inp_size",
        "use_compile_max_autotune",
        "compile_triton_persistent_reductions",
        "compile_triton_mix_order_reduction",
        "profiler_enabled",
        "profiler_record_shapes",
        "debug_non_finite_params",
        "debug_layerwise_metrics",
        "debug_memory_snapshot",
        "mlm_standard_amino_acids_only",
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
