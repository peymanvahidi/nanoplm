"""Configuration dataclasses for pretraining pipelines."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


def _normalize_fp8_targets(raw_targets) -> str | list[str]:
    aliases = {
        "mlp": "mlp",
        "attn_wqkv": "attn_wqkv",
        "attn-wqkv": "attn_wqkv",
        "attention_wqkv": "attn_wqkv",
        "attention-wqkv": "attn_wqkv",
        "qkv": "attn_wqkv",
        "attn_wo": "attn_wo",
        "attn-wo": "attn_wo",
        "attention_wo": "attn_wo",
        "attention-wo": "attn_wo",
        "wo": "attn_wo",
    }
    if isinstance(raw_targets, str):
        pieces = raw_targets.replace("+", ",").split(",")
    elif isinstance(raw_targets, (list, tuple)):
        pieces = []
        for item in raw_targets:
            pieces.extend(str(item).replace("+", ",").split(","))
    else:
        pieces = [str(raw_targets)]

    targets: list[str] = []
    for piece in pieces:
        key = str(piece).strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(
                "fp8_target must contain only: mlp, attn_wqkv, attn_wo "
                f"(got {raw_targets!r})"
            )
        target = aliases[key]
        if target not in targets:
            targets.append(target)

    if not targets:
        raise ValueError("fp8_target must not be empty.")
    return targets[0] if len(targets) == 1 else targets


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
    # Matrix-optimizer hyperparameters (used when optimizer is muon/normuon/nordion2).
    # adam_* fields are used for the AdamW sub-optimizer.
    muon_learning_rate: float = 1e-3
    muon_weight_decay: float = 0.01
    muon_cautious_weight_decay: bool = True  # Muon/NorMuon only (NorDion2 has no cautious_wd)
    muon_use_polar_express: bool = True
    muon_momentum: float = 0.95
    muon_nesterov: bool = True  # Muon/NorMuon only (NorDion2 has no nesterov)
    muon_eps: float = 1e-7
    # Shared matrix-optimizer knobs (passed to each Dion optimizer only where its
    # signature accepts it). muon_adjust_lr default 'rms_norm' preserves the prior
    # hardcoded muon/normuon behavior; dion's native default is 'spectral_norm'.
    muon_adjust_lr: str = "rms_norm"  # rms_norm | spectral_norm | none
    muon_beta2: float = 0.95  # NorMuon/NorDion2 second-moment beta (ignored by plain Muon)
    muon_use_gram_newton_schulz: bool = False
    muon_triton_post_ortho: bool = False  # NorDion2 only in the current dion build
    muon_flatten: bool = False
    # NorDion2-specific. Only used when optimizer == "nordion2".
    nordion2_fraction: float = 0.25  # low-rank fraction of the Dion2 update
    nordion2_selection_scope: str = "local"  # local | global (FSDP2 row-shard selection)
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
    # Pure-torch FP8 conversion target(s): mlp | attn_wqkv | attn_wo.
    # Any combination, as a comma-separated string or YAML list.
    fp8_target: Union[str, list[str]] = "mlp"
    # Pure-torch FP8 only: keep the final N transformer layers in bf16.
    fp8_skip_last_n_layers: int = 0
    # Transformer Engine FP8 recipe. Supported names:
    # delayed | current | block | mxfp8 | nvfp4
    fp8_recipe: str = "delayed"
    # TE FP8 format for recipes that accept it: hybrid | e4m3.
    # NVFP4 ignores this and uses its fixed FP4/FP8 formats.
    fp8_format: str = "hybrid"
    # DelayedScaling only: amax history window length.
    fp8_amax_history_len: int = 64
    # DelayedScaling only: synchronize amax across the FP8 reduction group.
    fp8_reduce_amax: bool = True
    # Pure-torch FP8 only: use Transformer Engine saved-tensor hooks to
    # offload layer activations to pinned CPU memory between forward/backward.
    fp8_cpu_activation_offload: bool = False
    # Reserved for future placement policies. Current pure-torch FP8 offload
    # targets all eligible TE LayerNormLinear and LayerNormMLP wrappers.
    fp8_cpu_activation_offload_layers: Union[int, str] = "auto"
    # Retain pinned CPU buffers across iterations. This can reduce allocator
    # churn for fixed shapes, but increases steady CPU memory footprint.
    fp8_cpu_activation_offload_retain_buffers: bool = False
    # MLM settings
    mlm_probability: float = 0.3
    mask_replace_prob: float = 0.8
    random_token_prob: float = 0.1
    keep_probability: float = 0.1
    mlm_masking_strategy: str = "token"  # token | span
    mlm_span_mean_length: float = 3.0
    mlm_span_max_length: int = 10
    # Current behavior: only the 20 standard amino acids are MLM targets and
    # random replacements. Set false to reproduce the legacy all-non-special
    # token behavior.
    mlm_standard_amino_acids_only: bool = True

    # Logging/checkpointing
    logging_steps: int = 1
    eval_steps: int = 250
    save_steps: int = 5000
    # Wall-clock budget for training, in hours. When set, the run checkpoints
    # and exits at the first optimizer step past the deadline, before the
    # token-budget horizon is reached. Intended for wall-clock-matched
    # ablations. The clock starts a couple of optimizer steps in, so compile,
    # autotune and warmup are excluded and the budget measures steady-state
    # training: expect total job time to exceed it. It also restarts from zero
    # on resume, measuring this run rather than cumulative time across resumes.
    max_wallclock_hours: Optional[float] = None
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
    distributed_mode: str = "fsdp"  # fsdp | hsdp | ddp
    # HSDP (Hybrid Sharded Data Parallel): shard params within a node (intra-node
    # FSDP over fast NVLink) and replicate + all-reduce gradients across nodes
    # (inter-node DDP). Only used when distributed_mode="hsdp". The shard group
    # size = GPUs per replica; "auto" reads LOCAL_WORLD_SIZE (GPUs per node), so
    # each node holds one full replica sharded across its GPUs. world_size must be
    # an integer multiple of the shard size. Makes sense for multi-node runs where
    # the inter-node fabric is the bottleneck and the model fits sharded within one
    # node; on a single node it degenerates to plain FSDP.
    hsdp_shard_size: Union[int, str] = "auto"
    # FSDP2 wrapping granularity: "layer" wraps whole transformer blocks (fewer
    # boundaries, less CPU dispatch overhead — better for fast interconnects like
    # NVLink/SXM); "sublayer" wraps attn & mlp separately (more boundaries, but
    # finer-grained comm/compute overlap — better for slow interconnects like PCIe).
    fsdp_shard_granularity: str = "layer"
    fsdp_reshard_after_forward: bool = False
    # FSDP2 MixedPrecisionPolicy.reduce_dtype for the gradient reduce-scatter:
    # "fp32" (exact) or "bf16" (half the bytes of the largest collective on the
    # wire; the sum across ranks then rounds to bf16).
    fsdp_reduce_dtype: str = "fp32"
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
    def __post_init__(self) -> None:
        for name in (
            "mlm_probability",
            "mask_replace_prob",
            "random_token_prob",
            "keep_probability",
            "mlm_span_mean_length",
        ):
            setattr(self, name, float(getattr(self, name)))

        strategy = str(self.mlm_masking_strategy).strip().lower()
        if strategy not in {"token", "span"}:
            raise ValueError(
                "mlm_masking_strategy must be one of: token, span "
                f"(got {self.mlm_masking_strategy!r})"
            )
        self.mlm_masking_strategy = strategy

        if self.mlm_span_mean_length <= 0:
            raise ValueError(
                f"mlm_span_mean_length must be > 0, got {self.mlm_span_mean_length}"
            )

        try:
            self.mlm_span_max_length = int(self.mlm_span_max_length)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"mlm_span_max_length must be an integer, got {self.mlm_span_max_length!r}"
            ) from exc
        if self.mlm_span_max_length < 1:
            raise ValueError(
                f"mlm_span_max_length must be >= 1, got {self.mlm_span_max_length}"
            )

        fp8_recipe_aliases = {
            "delayed": "delayed",
            "delayed-scaling": "delayed",
            "delayed_scaling": "delayed",
            "current": "current",
            "current-scaling": "current",
            "current_scaling": "current",
            "float8-current": "current",
            "float8_current": "current",
            "block": "block",
            "block-scaling": "block",
            "block_scaling": "block",
            "float8-block": "block",
            "float8_block": "block",
            "mxfp8": "mxfp8",
            "mx-fp8": "mxfp8",
            "mxfp8-block": "mxfp8",
            "mxfp8_block": "mxfp8",
            "nvfp4": "nvfp4",
            "nv-fp4": "nvfp4",
            "nvfp4-block": "nvfp4",
            "nvfp4_block": "nvfp4",
        }
        fp8_recipe = str(self.fp8_recipe).strip().lower()
        if fp8_recipe not in fp8_recipe_aliases:
            raise ValueError(
                "fp8_recipe must be one of: delayed, current, block, mxfp8, nvfp4 "
                f"(got {self.fp8_recipe!r})"
            )
        self.fp8_recipe = fp8_recipe_aliases[fp8_recipe]

        fp8_format_aliases = {
            "hybrid": "hybrid",
            "e4m3": "e4m3",
        }
        fp8_format = str(self.fp8_format).strip().lower()
        if fp8_format not in fp8_format_aliases:
            raise ValueError(
                "fp8_format must be one of: hybrid, e4m3 "
                f"(got {self.fp8_format!r})"
            )
        self.fp8_format = fp8_format_aliases[fp8_format]

        try:
            self.fp8_amax_history_len = int(self.fp8_amax_history_len)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fp8_amax_history_len must be an integer, "
                f"got {self.fp8_amax_history_len!r}"
            ) from exc
        if self.fp8_amax_history_len < 1:
            raise ValueError(
                "fp8_amax_history_len must be >= 1, "
                f"got {self.fp8_amax_history_len}"
            )

        self.fp8_cpu_activation_offload = bool(self.fp8_cpu_activation_offload)
        self.fp8_cpu_activation_offload_retain_buffers = bool(
            self.fp8_cpu_activation_offload_retain_buffers
        )
        if isinstance(self.fp8_cpu_activation_offload_layers, str):
            offload_layers = self.fp8_cpu_activation_offload_layers.strip().lower()
            if offload_layers != "auto":
                try:
                    offload_layers = int(offload_layers)
                except ValueError as exc:
                    raise ValueError(
                        "fp8_cpu_activation_offload_layers must be 'auto' or a "
                        f"positive integer, got {self.fp8_cpu_activation_offload_layers!r}"
                    ) from exc
            self.fp8_cpu_activation_offload_layers = offload_layers
        if isinstance(self.fp8_cpu_activation_offload_layers, int):
            if self.fp8_cpu_activation_offload_layers < 1:
                raise ValueError(
                    "fp8_cpu_activation_offload_layers must be >= 1 when set to "
                    f"an integer, got {self.fp8_cpu_activation_offload_layers}"
                )
        elif self.fp8_cpu_activation_offload_layers != "auto":
            raise ValueError(
                "fp8_cpu_activation_offload_layers must be 'auto' or a positive "
                f"integer, got {self.fp8_cpu_activation_offload_layers!r}"
            )

        try:
            self.fp8_skip_last_n_layers = int(self.fp8_skip_last_n_layers)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fp8_skip_last_n_layers must be an integer, "
                f"got {self.fp8_skip_last_n_layers!r}"
            ) from exc
        if self.fp8_skip_last_n_layers < 0:
            raise ValueError(
                "fp8_skip_last_n_layers must be >= 0, "
                f"got {self.fp8_skip_last_n_layers}"
            )

        self.fp8_target = _normalize_fp8_targets(self.fp8_target)

        self.muon_adjust_lr = str(self.muon_adjust_lr).strip().lower()
        if self.muon_adjust_lr not in {"rms_norm", "spectral_norm", "none"}:
            raise ValueError(
                "muon_adjust_lr must be one of: rms_norm, spectral_norm, none "
                f"(got {self.muon_adjust_lr!r})"
            )

        self.nordion2_selection_scope = str(self.nordion2_selection_scope).strip().lower()
        if self.nordion2_selection_scope not in {"local", "global"}:
            raise ValueError(
                "nordion2_selection_scope must be one of: local, global "
                f"(got {self.nordion2_selection_scope!r})"
            )


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
