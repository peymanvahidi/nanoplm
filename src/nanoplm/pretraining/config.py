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
    micro_batch_size: int = 32
    num_epochs: int = 10
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_learning_rate: float = 1e-3
    adam_weight_decay: float = 0.0
    max_grad_norm: float = float("inf")
    warmup_steps: int = 302
    repo_rope_warmup_steps: Optional[int] = None
    lr_decay_to_fraction: float = 0.1
    lr_schedule: str = "cosine"
    # Muon-specific hyperparameters (used only when optimizer == "muon" or "normuon").
    # adam_* fields are used for the AdamW sub-optimizer.
    muon_learning_rate: float = 2e-2
    muon_weight_decay: float = 0.1
    muon_cautious_weight_decay: bool = True
    muon_use_polar_express: bool = False
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_eps: float = 1e-7
    # Target effective batch size in tokens per optimizer step.
    # gradient_accumulation_steps is inferred from this value at runtime.
    global_batch_size: int = 2 ** 20

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
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    seed: int = 42

    # Data loading
    num_workers: Union[int, str] = "auto"
    prefetch_factor: int = 2

    # Packing and compilation
    use_packing: bool = True
    use_static_inp_size: bool = True
    use_compile_max_autotune: bool = False
    compile_triton_persistent_reductions: bool = False
    compile_triton_mix_order_reduction: bool = False

    # Profiler
    profiler_enabled: bool = False
    profiler_start_step: int = 10
    profiler_end_step: int = 15

    # Distributed training
    multi_gpu: bool = False
    world_size: Union[int, str] = 1
    project_name: str = "nanoplm-pretraining"


@dataclass
class ResumeConfig:
    is_resume: bool
    checkpoint_dir: str
    extra_epochs: Optional[int] = None
