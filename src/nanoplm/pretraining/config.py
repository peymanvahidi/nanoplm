"""Configuration dataclasses for pretraining pipelines."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class PretrainingConfig:
    # Dataset directory (contains .data_manifest from nanoplm data from-yaml)
    dataset_dir: Union[str, Path]

    # Checkpoint and output
    ckp_dir: str = "output/pretraining"

    # Training hyperparameters
    micro_batch_seqs: int = 64
    warmup_steps: int = 1000
    # RePO activation warmup (optimizer-step based) for pure-torch path.
    # If None, defaults to warmup_steps (backward-compatible behavior).
    repo_rope_warmup_steps: Optional[int] = None
    lr_decay_to_fraction: float = 0.1
    lr_schedule: str = "wsd"            # wsd | warmup_stable | cosine | linear
    decay_shape: str = "1-sqrt"         # 1-sqrt | linear (wsd/decay only)
    # Duration as a TOKEN budget (primary). Exactly one unit per phase.
    # (Mutual-exclusivity/precedence is ENFORCED later by duration-precedence validation at config load, not by this dataclass.)
    stable_tokens: Optional[int] = None   # wsd: tokens up to the pre-decay checkpoint (warmup incl.)
    decay_tokens: Optional[int] = None    # wsd: additional anneal tokens
    train_tokens: Optional[int] = None    # warmup_stable | cosine | linear: single budget
    # Advanced step overrides (mutually exclusive with their *_tokens counterpart):
    stable_steps: Optional[int] = None
    decay_steps: Optional[int] = None
    max_steps: Optional[int] = None    # ↔ train_tokens (single-phase step override)
    optimizer: str = "normuon"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_learning_rate: float = 1e-4
    adam_weight_decay: float = 0.0
    # Gradient clipping threshold (L2 norm). Set float("inf") to disable clipping.
    max_grad_norm: float = float("inf")
    # Muon-specific hyperparameters (used only when optimizer == "muon" or "normuon").
    # adam_* fields are used for the AdamW sub-optimizer.
    muon_learning_rate: float = 1e-3
    muon_weight_decay: float = 0.01
    muon_cautious_weight_decay: bool = True
    muon_use_polar_express: bool = True
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_eps: float = 1e-7
    # MuonH-specific. Only used when optimizer == "muonh".
    # 'neuron' normalizes along the last axis; 'short_axis' along the shorter
    # matrix axis. Set to "none" to disable update normalization in MuonH.
    muonh_normalization: str = "neuron"
    # Target effective batch size in tokens per optimizer step.
    # gradient_accumulation_steps is inferred from this value at runtime.
    global_batch_tokens: int = 256000
    inferred_grad_accum_steps: Optional[int] = None
    global_batch_seqs: Optional[int] = None
    achieved_global_batch_tokens: Optional[int] = None
    global_batch_warmup_steps: int = 0

    # Mixed precision
    bf16: bool = True
    tf32: bool = True
    fp8: bool = False

    # MLM settings
    mlm_probability: float = 0.3
    mask_replace_prob: float = 0.8
    random_token_prob: float = 0.1
    keep_probability: float = 0.1

    # Logging/checkpointing
    logging_steps: int = 1
    eval_steps: int = 250
    save_steps: int = 5000
    seed: int = 42
    debug_non_finite_params: bool = True
    debug_layerwise_metrics: bool = False
    debug_layerwise_log_every: int = 20

    # Data loading
    num_workers: Union[int, str] = "auto"
    prefetch_factor: int = 2

    # Sequence packing (packs multiple sequences per row to eliminate padding waste).
    # Requires flash attention (varlen path).  Falls back to padding if disabled.
    use_packing: bool = True
    # When packing is enabled, force fixed flat token count and bucketed attention metadata
    # (cu_seqlens/max_seqlen). This enables static-shape execution for torch.compile
    # (dynamic=False) and improves CUDA graph capture reuse.
    use_static_inp_size: bool = True
    # If enabled, pure-torch compile uses mode='max-autotune-no-cudagraphs'.
    # This may improve steady-state performance, but increases compile/autotune time
    # noticeably at the start of a run.
    use_compile_max_autotune: bool = False
    # TorchInductor Triton setting. Persistent reductions can exceed per-block shared
    # memory limits for some fused kernels (e.g. LayerNorm backward at hidden >= 1536),
    # causing "No valid triton configs / out of resource". Set to False to force
    # non-persistent reduction kernels while keeping torch.compile enabled.
    compile_triton_persistent_reductions: bool = False
    # TorchInductor Triton setting. Mix-order reductions can generate persistent
    # reduction kernels that exceed shared memory limits on some shapes (notably
    # layernorm backward fusions at larger hidden sizes). Disable to prefer the
    # legacy reduction codegen while keeping torch.compile enabled.
    compile_triton_mix_order_reduction: bool = False

    # Profiling (pure_torch / pure_te pipelines). When enabled on rank 0:
    # - If running under nsys: uses CUDA Profiler API (start/stop at steps) for .nsys-rep traces.
    # - Otherwise: uses PyTorch profiler and exports a Chrome trace (chrome://tracing) to ckp_dir.
    profiler_enabled: bool = False
    profiler_start_step: int = 10
    profiler_end_step: int = 15

    # Distributed training
    multi_gpu: bool = True
    world_size: Union[int, str] = "auto"
    distributed_mode: str = "fsdp"
    # FSDP2 wrapping granularity: "layer" wraps whole transformer blocks (fewer
    # boundaries, less CPU dispatch overhead — better for fast interconnects like
    # NVLink/SXM); "sublayer" wraps attn & mlp separately (more boundaries, but
    # finer-grained comm/compute overlap — better for slow interconnects like PCIe).
    fsdp_shard_granularity: str = "layer"
    fsdp_reshard_after_forward: bool = False
    ddp_bucket_cap_mb: int = 25
    project_name: str = "nanoplm-pretraining"
    run_name: str = "run"

    # NOBLE optimizer LR scaling (paper §3.3, Appendix A)
    noble_base_lr: float = 3e-4          # independent base LR for NOBLE branch params
    noble_lr_gamma: float = 0.3          # W_up LR power: actual exponent = 2γ = 0.6
    noble_lr_gamma_M: float = 0.45       # M mixing matrix LR power
    noble_freq_lr_mult: float = 3.0      # ω frequency LR multiplier
    noble_phase_lr_mult: float = 5.0     # φ phase LR multiplier

    # Huginn-style recycling — see docs/superpowers/specs/2026-05-24-huggin-style-recycling-design.md §2.
    # All inactive when model.recycling_loop_count == 1.
    recycling_bptt: int = 8                                   # truncated BPTT depth K
    recycling_loop_count_sampling: str = "lognormal_poisson"  # "fixed" | "lognormal_poisson"
    recycling_loop_count_sigma: float = 0.5                   # log-normal std (Huginn value)
    recycling_loop_count_curriculum: str = "1_sqrt"           # "none" | "linear" | "1_sqrt"
    recycling_loop_count_curriculum_steps: int = -1           # -1 = auto (25% of total training steps)
    recycling_pretraining_eval_loop_counts: Optional[list[int]] = None  # None = single r at target mean


@dataclass
class ResumeConfig:
    is_resume: bool
    checkpoint_dir: str
    mode: str = "continue"                 # continue | decay
    # decay-only fields (mode == "decay"):
    decay_tokens: Optional[int] = None
    decay_steps: Optional[int] = None
    decay_shape: Optional[str] = None      # 1-sqrt | linear
    lr_decay_to_fraction: Optional[float] = None
