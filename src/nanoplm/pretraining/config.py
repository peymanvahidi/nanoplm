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
    micro_batch_size: int = 64
    num_epochs: int = 10
    warmup_steps: int = 302
    # RePO activation warmup (optimizer-step based) for pure-torch path.
    # If None, defaults to warmup_steps (backward-compatible behavior).
    repo_rope_warmup_steps: Optional[int] = None
    lr_decay_to_fraction: float = 0.1
    lr_schedule: str = "cosine"
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
    muon_use_polar_express: bool = False
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_eps: float = 1e-7
    # Target effective batch size in tokens per optimizer step.
    # gradient_accumulation_steps is inferred from this value at runtime.
    global_batch_size: int = 256000
    inferred_grad_accum_steps: Optional[int] = None
    global_batch_size_samples: Optional[int] = None
    achieved_global_batch_tokens: Optional[int] = None

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
    ddp_bucket_cap_mb: int = 25
    project_name: str = "nanoplm-pretraining"


@dataclass
class ResumeConfig:
    is_resume: bool
    checkpoint_dir: str
    extra_epochs: Optional[int] = None
    # Schedule override options for resumed training.
    # When set to None (the YAML default), the corresponding value is read
    # from the checkpoint's saved pretrain_config.yaml so the original
    # schedule is reconstructed exactly.  Explicit values override.
    warmup_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    muon_learning_rate: Optional[float] = None
    lr_schedule: Optional[str] = None
    lr_decay_to_fraction: Optional[float] = None
    # reset_scheduler: rebuild the LR schedule from step 0 covering only the
    # remaining training steps (new warmup + decay).  Useful when you want a
    # fresh learning-rate curve after a long pause.
    reset_scheduler: bool = False
    # skip_warmup: jump straight to peak LR on resume (sets warmup_steps=0).
    skip_warmup: bool = False
