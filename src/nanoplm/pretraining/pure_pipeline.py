"""Pure PyTorch pretraining pipeline.

Same outer behavior as the HF Trainer pipeline (datasets, run naming,
checkpoint layout, resume mechanics), but the training loop is plain torch.
"""

import gc
import json
import math
import os
import yaml
from dataclasses import asdict
from dion import Muon,NorMuon
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Default to safer Inductor reduction codegen. Inductor's async compile uses
# subprocess workers that read these env vars at process start, so setting them
# here helps ensure stable defaults even when invoked via torchrun.
os.environ.setdefault("TORCHINDUCTOR_PERSISTENT_REDUCTIONS", "0")
os.environ.setdefault("TORCHINDUCTOR_MIX_ORDER_REDUCTION", "0")


import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from nanoplm.data.manifest import read_manifest, validate_manifest_for_pipeline
from nanoplm.pretraining.collator import (
    DataCollatorWithFlattening,
    ProtDataCollatorForLM,
    build_power_of_two_buckets,
)
from nanoplm.pretraining.dataset import ShardedDataset, TokenPackingDataset
from nanoplm.pretraining.fp8 import (
    Float8Linear,
    Float8LinearConfig,
    convert_to_float8_training,
)
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.optim import build_muon_optimizer, build_optimizer, is_muon_optimizer
from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
from nanoplm.pretraining.utils import (
    compute_batch_setup,
    get_num_workers,
    prepare_run_and_steps,
)
from nanoplm.utils.common import create_dirs, get_device, resolve_world_size
from nanoplm.utils.logger import logger
from nanoplm.utils.wandb_artifacts import upload_run_source_snapshot

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# H100 SXM peak BF16 tensor-core throughput (TFLOPS).
H100_PEAK_TFLOPS = 989.4

# https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#forward-backward-with-prefetching
N_PREFETCH_LAYERS_FSDP2 = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def _format_vram_for_log(
    *,
    device: torch.device,
    distributed: bool,
    reset_peak: bool = False,
) -> str:
    """Return CUDA VRAM usage string; aggregate max across ranks when distributed."""
    if device.type != "cuda":
        return "vram=n/a"

    alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
    peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

    if distributed and dist.is_initialized():
        stats = torch.tensor(
            [alloc_mb, reserved_mb, peak_alloc_mb, peak_reserved_mb],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        alloc_mb, reserved_mb, peak_alloc_mb, peak_reserved_mb = (
            float(stats[0].item()),
            float(stats[1].item()),
            float(stats[2].item()),
            float(stats[3].item()),
        )

    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)

    return (
        f"vram={alloc_mb:,.0f}/{reserved_mb:,.0f}MB "
        f"peak={peak_alloc_mb:,.0f}/{peak_reserved_mb:,.0f}MB"
    )

def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def _has_nonfinite_tensors(tensors: list[torch.Tensor]) -> bool:
    local_tensors = []
    for tensor in tensors:
        local = _local_tensor(tensor.detach())
        if local.numel() > 0:
            local_tensors.append(local)
    if not local_tensors:
        return False
    try:
        norms = torch._foreach_norm(local_tensors, float("inf"))
        return not torch.isfinite(torch.stack(list(norms))).all().item()
    except (RuntimeError, TypeError):
        for tensor in local_tensors:
            if not torch.isfinite(tensor).all():
                return True
        return False


def _first_nonfinite_grad(model) -> tuple[str, torch.Tensor] | None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = _local_tensor(param.grad.detach())
        if not torch.isfinite(grad).all():
            return name, grad
    return None


def _first_nonfinite_param(model) -> tuple[str, torch.Tensor] | None:
    for name, param in model.named_parameters():
        data = _local_tensor(param.detach())
        if not torch.isfinite(data).all():
            return name, data
    return None


def _has_nonfinite_params(model) -> bool:
    return _has_nonfinite_tensors([param for param in model.parameters()])


def _use_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    if not param.requires_grad or param.ndim < 2:
        return False
    lname = name.lower()
    return "bias" not in lname and "norm" not in lname


def _is_embedding_or_unembedding_param(name: str) -> bool:
    lname = name.lower()
    return (
        "embeddings.tok_embeddings" in lname
        or lname.endswith("decoder.weight")
        or lname.endswith("decoder.bias")
        or "embedding" in lname
        or "lm_head" in lname
        or "unembedding" in lname
    )


def _dist_barrier(local_rank: int) -> None:
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _make_pure_profiler(
    pretrain_config: PretrainingConfig,
    output_dir: str,
    is_main: bool,
):
    """Build profiler context and step callback for the pure-torch training loop.

    When profiler_enabled and is_main:
    - If running under nsys (NSYS_PROFILING_SESSION_ID): uses CUDA Profiler API so
      nsys can capture; start/stop at profiler_start_step / profiler_end_step.
    - Otherwise: uses PyTorch profiler with a schedule and exports a Chrome trace
      to output_dir/profiler_traces/chrome_trace.json (view in chrome://tracing).

    Returns:
        (context_manager, step_callback): use as ``with context: ...; step_callback(global_step)``
        after each optimizer step.
    """
    if not getattr(pretrain_config, "profiler_enabled", False) or not is_main:
        return nullcontext(), lambda _: None

    start_step = getattr(pretrain_config, "profiler_start_step", 10)
    end_step = getattr(pretrain_config, "profiler_end_step", 15)
    running_under_nsys = "NSYS_PROFILING_SESSION_ID" in os.environ

    if running_under_nsys:
        logger.info(
            "Profiling enabled (Nsight): CUDA Profiler API will start at step %s, stop at %s. "
            "Run with: nsys profile -o <trace> --trace=cuda,nvtx,osrt,cudnn,cublas "
            "--capture-range=cudaProfilerApi --capture-range-end=stop ...",
            start_step,
            end_step,
        )

        class _NsightController:
            def __init__(self):
                self.started = False
                self.finished = False

            def step(self, gs: int) -> None:
                if self.finished:
                    return
                if gs == start_step and not self.started:
                    try:
                        torch.cuda.cudart().cudaProfilerStart()  # type: ignore[attr-defined]
                        self.started = True
                        logger.info("Nsight profiling started at step %s", gs)
                    except Exception as e:
                        logger.error("Failed to start CUDA profiler: %s", e)
                elif gs == end_step and self.started:
                    try:
                        torch.cuda.cudart().cudaProfilerStop()  # type: ignore[attr-defined]
                        self.started = False
                        self.finished = True
                        logger.info("Nsight profiling stopped at step %s", gs)
                    except Exception as e:
                        logger.error("Failed to stop CUDA profiler: %s", e)

        ctrl = _NsightController()
        return nullcontext(), ctrl.step

    trace_dir = Path(output_dir) / "profiler_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "chrome_trace.json"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    wait = max(0, start_step)
    warmup = 1
    active = max(1, end_step - start_step - 1)

    def on_trace_ready(prof: torch.profiler.profile) -> None:
        prof.export_chrome_trace(str(trace_path))
        logger.info("Exported PyTorch profiler trace to %s", trace_path)

    prof = torch.profiler.profile(
        activities=activities,
        # Single profiling window; avoid repeating the cycle throughout training.
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=on_trace_ready,
        record_shapes=True,
    )
    logger.info(
        "Profiling enabled (PyTorch): trace will be written to %s (steps %s..%s). "
        "Open in chrome://tracing",
        trace_path,
        start_step,
        end_step,
    )
    return prof, lambda _: prof.step()


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def _estimate_model_flops_per_token(config, seq_len: int) -> int:
    """Training FLOPs per token (forward + backward ≈ 3× forward).

    Each matmul contributes 2 FLOPs per element (multiply + accumulate) in forward.
    Backward is ~2× forward for matmuls => total 6 FLOPs per matmul element.
    Attention QK and AV matmuls scale with effective sequence length per layer
    (capped by sliding window where applicable).

    Ref: https://arxiv.org/abs/2204.02311 (PaLM)
    Ref: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
    """
    h = config.hidden_size
    ff = config.intermediate_size
    n_layers = config.num_hidden_layers
    V = config.vocab_size
    n_heads = config.num_attention_heads
    head_dim = h // n_heads
    sliding_window = config.local_attention // 2

    # -- Attention matmul FLOPs per layer (varies with window size) --
    # QK matmul: (S, h) × (h, S) => 2*h*S FLOPs per token
    # AV matmul: (S, S) × (S, h) => 2*h*S FLOPs per token
    # Total: 4*h*effective_S per token per layer (forward only)
    attn_flops = 0
    for layer_type in config.layer_types:
        if layer_type == "full_attention":
            effective_seq = seq_len
        else:
            effective_seq = min(2 * sliding_window, seq_len)
        attn_flops += 4 * h * effective_seq

    # -- Attention projection FLOPs (same for all layers) --
    # Wqkv: h -> 3h  => 2 * h * 3h = 6h²
    # Wo:   h -> h    => 2 * h * h  = 2h²
    # Total: 8h² per layer
    attn_proj_flops = n_layers * 8 * h * h

    # -- MLP FLOPs --
    # swiglu/glu: Wi is h -> 2*ff, Wo is ff -> h => 2*h*2*ff + 2*ff*h = 6*h*ff
    # srelu:      Wi is h -> ff,   Wo is ff -> h => 2*h*ff   + 2*ff*h = 4*h*ff
    if config.mlp_activation == "srelu":
        mlp_flops_per_layer = 4 * h * ff
    else:
        mlp_flops_per_layer = 6 * h * ff
    mlp_flops = n_layers * mlp_flops_per_layer

    # -- LM head (decoder): h -> V --
    # Not counted when tied (it's the same weight as embedding), but the matmul
    # still happens in forward, so we count it.
    head_flops = 2 * V * h

    # -- Prediction head dense: h -> h --
    pred_head_flops = 2 * h * h

    # -- Canon layer FLOPs (depthwise conv1d) --
    # Each canon layer is a depthwise conv with kernel_size K over C channels.
    # Forward FLOPs per token = 2 * K * C  (K multiply-accumulates per channel).
    canon_flops = 0
    if getattr(config, "use_canon_layers", False) and config.canon_layer_set:
        K = config.canon_layers_kernel_size
        canon_set = config.canon_layer_set
        if "a" in canon_set:                        # before attention, C = h
            canon_flops += n_layers * 2 * K * h
        if "b" in canon_set:                        # on QKV, C = 3h
            canon_flops += n_layers * 2 * K * 3 * h
        if "c" in canon_set:                        # before MLP, C = h
            canon_flops += n_layers * 2 * K * h
        if "d" in canon_set and config.mlp_activation != "srelu":
            # after first MLP projection, C = 2*ff (gated MLPs only)
            canon_flops += n_layers * 2 * K * 2 * ff

    # -- mHC-lite FLOPs (multi-stream residual wrapper) --
    # MHCLiteBlock does *not* run attention/MLP per stream; it merges streams into a
    # single layer_input, runs the wrapped submodule once, then mixes streams.
    #
    # We count the dominant matmuls per MHCLiteBlock:
    #  - fused coefficient projection: (n*h) -> (2n + n!)  => 2 * (n*h) * (2n + n!)
    #  - permutation mix: (n!) @ (n*n)                    => 2 * (n!) * n * n
    #  - pre-map: (1,n) @ (n,h)                           => 2 * n * h
    #  - post-res: (n,n) @ (n,h)                          => 2 * n * n * h
    # Elementwise ops (sigmoid/softmax, H_merged arithmetic, h_post scaling) are
    # ignored elsewhere in this estimator, so we ignore them here too.
    mhc_flops = 0
    if getattr(config, "use_mhc_lite", False):
        n = int(getattr(config, "mhc_n_streams", 4))
        level = str(getattr(config, "mhc_lite_wrapping_level", "layer")).strip().lower()
        blocks_per_layer = 2 if level == "sublayers" else 1
        n_fact = math.factorial(n)
        total_out = 2 * n + n_fact
        proj_flops = 2 * (n * h) * total_out
        perm_mix_flops = 2 * n_fact * n * n
        pre_map_flops = 2 * n * h
        post_res_flops = 2 * n * n * h
        mhc_flops = n_layers * blocks_per_layer * (
            proj_flops + perm_mix_flops + pre_map_flops + post_res_flops
        )

    forward_flops = (
        attn_proj_flops
        + attn_flops
        + mlp_flops
        + head_flops
        + pred_head_flops
        + canon_flops
        + mhc_flops
    )
    return 3 * forward_flops


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------

def _is_zero_init_fragile_param(name: str) -> bool:
    """Detect 2-D parameters that are zero-initialized and have structurally
    dormant rows, making them unsafe for NorMuon's per-row variance tracking.

    Currently matches:
      - mHC-lite W_all.weight  (32×3072, 75% of rows intentionally suppressed)
      - RePO W_z.weight        (8×96, zero-init with per-head rows)
    """
    parts = name.split(".")
    # W_all.weight (mHC-lite fused projection)
    if len(parts) >= 2 and parts[-2] == "W_all" and parts[-1] == "weight":
        return True
    # W_z.weight (RePO position head)
    if len(parts) >= 2 and parts[-2] == "W_z" and parts[-1] == "weight":
        return True
    return False


def _build_muon_optimizer(model, cfg, distributed_mesh=None):
    muon_params, adamw_params = [], []
    resid_params, x0_params = [], []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        lname = name.lower()
        if lname.endswith("resid_lambdas") or ".resid_lambdas" in lname:
            resid_params.append(param)
            continue
        if lname.endswith("x0_lambdas") or ".x0_lambdas" in lname:
            x0_params.append(param)
            continue
        if param.ndim == 1 or _is_embedding_or_unembedding_param(name):
            adamw_params.append(param)
        elif param.ndim == 2:
            if _is_zero_init_fragile_param(name):
                adamw_params.append(param)
            else:
                muon_params.append(param)
        else:
            adamw_params.append(param)

    if not muon_params:
        raise ValueError("No eligible 2-D matrix parameters found for Muon.")

    logger.info(
        f"Muon grouping: muon_params={len(muon_params)} tensors, "
        f"adamw_params={len(adamw_params)} tensors, "
        f"resid_scalar_params={len(resid_params)} tensors, "
        f"x0_scalar_params={len(x0_params)} tensors"
    )
    return build_optimizer(
        muon_params=muon_params,
        adamw_params=adamw_params,
        resid_params=resid_params,
        x0_params=x0_params,
        muon_learning_rate=cfg.muon_learning_rate,
        muon_weight_decay=cfg.muon_weight_decay,
        muon_cautious_weight_decay=cfg.muon_cautious_weight_decay,
        muon_use_polar_express=cfg.muon_use_polar_express,
        muon_momentum=cfg.muon_momentum,
        muon_nesterov=cfg.muon_nesterov,
        muon_eps=cfg.muon_eps,
        use_normuon=str(cfg.optimizer).lower() == "normuon",
        adamw_learning_rate=cfg.adam_learning_rate,
        adamw_weight_decay=cfg.adam_weight_decay,
        adamw_betas=(cfg.adam_beta1, cfg.adam_beta2),
        adamw_epsilon=cfg.adam_epsilon,
        distributed_mesh=distributed_mesh,
    )


def _create_optimizer(model, cfg, distributed_mesh=None):
    name = str(cfg.optimizer).lower()
    if name in {"muon", "normuon"}:
        return build_muon_optimizer(model, cfg, distributed_mesh=distributed_mesh)

    decay, no_decay = [], []
    for p_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if _use_weight_decay(p_name, param) else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": float(cfg.adam_weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(
        params=groups,
        lr=float(cfg.adam_learning_rate),
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=float(cfg.adam_epsilon),
    )

    if name == "adamw":
        return torch.optim.AdamW(**kwargs)
    if name == "stable_adamw":
        cls = getattr(torch.optim, "StableAdamW", None)
        if cls is None:
            logger.warning("StableAdamW unavailable; falling back to AdamW.")
            return torch.optim.AdamW(**kwargs)
        return cls(**kwargs)
    raise ValueError(f"Invalid optimizer: {cfg.optimizer}. Supported: [adamw, stable_adamw, muon, normuon]")


def _create_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    learning_rate: float,
    lr_decay_to_fraction: float,
    lr_schedule: str = "Linear",
    last_epoch: int = -1,
) -> LambdaLR:
    """Build a LambdaLR scheduler with warmup + cosine/linear decay.

    Args:
        last_epoch: If ≥ 0 the scheduler is positioned at that step
            (useful when reconstructing the schedule on resume).
            ``initial_lr`` is injected into param groups automatically.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        if lr_schedule.lower() == "cosine":
            return lr_decay_to_fraction + 0.5 * (1.0 - lr_decay_to_fraction) * (1.0 + math.cos(math.pi * progress))
        else:
            return max(lr_decay_to_fraction, 1.0 - (1.0 - lr_decay_to_fraction) * progress)

    # When positioning at a non-zero step, LambdaLR requires `initial_lr`
    # in every param group.  Set it from the current LR (which the caller
    # has already configured to the desired base value).
    if last_epoch >= 0:
        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _resolve_world_size(cfg: PretrainingConfig) -> int:
    if not cfg.multi_gpu:
        return 1
    if cfg.world_size == "auto":
        env = os.environ.get("WORLD_SIZE")
        return int(env) if env else max(torch.cuda.device_count(), 1)
    return int(cfg.world_size) if cfg.world_size else 1


def _num_update_steps_per_epoch(train_loader_len: int, grad_accum: int) -> int:
    if train_loader_len <= 0:
        return 1
    return max(1, math.ceil(train_loader_len / max(1, grad_accum)))


def _sync_train_loader_len(
    train_loader_len: int, distributed: bool, device: torch.device
) -> int:
    """Return the shared number of micro-batches all ranks can safely execute."""
    if not (distributed and dist.is_initialized()):
        return train_loader_len
    t = torch.tensor(train_loader_len, device=device, dtype=torch.int64)
    mn, mx = t.clone(), t.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    min_len = int(mn)
    max_len = int(mx)
    if min_len != max_len and dist.get_rank() == 0:
        logger.warning(
            "Mismatched train loader lengths across ranks (min=%d, max=%d); "
            "using min length for synchronized micro-step scheduling.",
            min_len,
            max_len,
        )
    return min_len
  # TODO: this is comming from master branch
# def _dist_barrier(local_rank: int) -> None:
#     if not dist.is_initialized():
#         return
#     if dist.get_backend() == "nccl":
#         dist.barrier(device_ids=[local_rank])
#     else:
#         dist.barrier()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _evaluate(model, eval_loader, device, distributed, amp_dtype) -> float:
    model.eval()
    total_loss, total_samples = 0.0, 0

    for batch in eval_loader:
        batch = _move_batch_to_device(batch, device)
        ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with ctx:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        loss = out["loss"] if isinstance(out, dict) else out.loss
        bs = batch["input_ids"].size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    if distributed and dist.is_initialized():
        stats = torch.tensor([total_loss, total_samples], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_samples = stats[0].item(), int(stats[1].item())

    model.train()
    return total_loss / max(1, total_samples)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _to_full_tensors(obj):
    """Recursively convert DTensors to plain tensors (collective operation)."""
    if isinstance(obj, DTensor):
        return obj.full_tensor()
    if isinstance(obj, dict):
        return {k: _to_full_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_full_tensors(v) for v in obj]
    return obj


def _save_checkpoint(
    model, optimizer, scheduler, global_step, epoch,
    output_dir, logging_steps, eval_steps, save_steps,
    distributed=False, is_main=True,
    model_config=None, manifest=None,
    pretrain_config=None, total_steps=None, warmup_steps=None,
) -> None:
    ckpt = Path(output_dir) / f"checkpoint-{global_step}"

    if distributed:
        model_sd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        opt_sd = _to_full_tensors(optimizer.state_dict())
    else:
        model_sd = model.state_dict()
        opt_sd = optimizer.state_dict()

    if not is_main:
        return

    create_dirs(str(ckpt))
    torch.save(model_sd, ckpt / "pytorch_model.bin")
    torch.save(opt_sd, ckpt / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt / "scheduler.pt")
    torch.save(
        {
            "torch_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        ckpt / "rng_state.pth",
    )
    # Include schedule metadata so resumed runs can reconstruct the exact
    # LR curve without depending on the current pretrain.yaml.
    training_state = dict(
        global_step=global_step, epoch=epoch,
        logging_steps=logging_steps, eval_steps=eval_steps, save_steps=save_steps,
    )
    if total_steps is not None:
        training_state["total_steps"] = total_steps
    if warmup_steps is not None:
        training_state["warmup_steps"] = warmup_steps
    (ckpt / "training_state.json").write_text(
        json.dumps(training_state, indent=2),
        encoding="utf-8",
    )

    # Save model architecture config and data manifest so checkpoints are
    # self-contained (inference can reconstruct the model without the
    # original pretrain.yaml or dataset directory).
    if model_config is not None:
        cfg_dict = asdict(model_config) if hasattr(model_config, '__dataclass_fields__') else dict(model_config)
        with open(ckpt / "model_config.yaml", "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    if manifest is not None:
        manifest_dict = manifest.to_dict() if hasattr(manifest, 'to_dict') else dict(manifest)
        with open(ckpt / "data_manifest.yaml", "w") as f:
            yaml.dump(manifest_dict, f, default_flow_style=False, sort_keys=False)

    # Save full pretraining config so resumed runs can detect config drift
    # and reconstruct the original LR schedule exactly.
    if pretrain_config is not None:
        try:
            pc_dict = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in pretrain_config.__dict__.items()
            }
            with open(ckpt / "pretrain_config.yaml", "w") as f:
                yaml.dump(pc_dict, f, default_flow_style=False, sort_keys=False)
        except Exception as exc:
            logger.warning(f"Failed to save pretrain_config.yaml: {exc}")

    logger.info(f"Checkpoint saved -> {ckpt}")


def _load_checkpoint(
    model, optimizer, scheduler, checkpoint_dir, device,
    distributed=False, load_scheduler=True,
) -> Tuple[int, int]:
    """Restore model, optimizer, (optionally) scheduler, and RNG states.

    Args:
        load_scheduler: When *False* the scheduler state dict is **not**
            loaded.  This is used by the resume-override path which
            rebuilds the scheduler from saved/overridden schedule params
            instead of relying on the pickled ``lr_lambda``.
    """
    ckp = Path(checkpoint_dir)

    model_sd = torch.load(ckp / "pytorch_model.bin", map_location=device, weights_only=True)
    if distributed:
        set_model_state_dict(model, model_sd, options=StateDictOptions(full_state_dict=True))
    else:
        model.load_state_dict(model_sd)

    opt_sd = torch.load(ckp / "optimizer.pt", map_location=device, weights_only=True)

    # When resuming under FSDP2, the checkpoint contains full (unsharded) optimizer
    # state tensors but the live parameters are sharded across ranks. We must slice
    # each optimizer buffer to match the corresponding parameter's local shard shape.
    if distributed and dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()

        # Build map: param index -> local parameter tensor.
        # Iterate *all* param groups (muon, adamw, resid, x0, …).
        param_by_idx: dict[int, torch.nn.Parameter] = {}
        all_params: list[torch.nn.Parameter] = []
        for pg in optimizer.param_groups:
            all_params.extend(pg["params"])
        for idx, p in enumerate(all_params):
            param_by_idx[idx] = p

        for param_idx, state_dict_entry in opt_sd.get("state", {}).items():
            local_param = param_by_idx.get(param_idx)
            if local_param is None:
                continue
            # Get local shape (after FSDP sharding)
            local_shape = local_param.shape if not isinstance(local_param, DTensor) else local_param._local_tensor.shape
            for key, val in state_dict_entry.items():
                if not isinstance(val, torch.Tensor):
                    continue
                # If shapes already match, skip
                if val.shape == local_shape:
                    continue
                # Shard along dim 0 (FSDP default sharding dimension)
                if val.ndim >= 1 and val.shape[0] == local_shape[0] * world:
                    chunk_size = val.shape[0] // world
                    state_dict_entry[key] = val.narrow(0, rank * chunk_size, chunk_size).contiguous()
                elif val.ndim >= 1 and val.shape[0] != local_shape[0]:
                    # Non-standard / uneven sharding. tensor_split always returns
                    # exactly one slice per rank, including empty trailing shards
                    # when val.shape[0] < world.
                    chunks = torch.tensor_split(val, world, dim=0)
                    if chunks[rank].shape[0] == local_shape[0]:
                        state_dict_entry[key] = chunks[rank].contiguous()
                    else:
                        logger.warning(
                            f"Optimizer state param {param_idx} key '{key}': "
                            f"shape {val.shape} cannot be sharded to match local {local_shape}"
                        )

    optimizer.load_state_dict(opt_sd)

    if load_scheduler:
        sched_path = ckp / "scheduler.pt"
        if sched_path.exists():
            scheduler.load_state_dict(
                torch.load(sched_path, map_location=device, weights_only=True)
            )

    rng_path = ckp / "rng_state.pth"
    if rng_path.exists():
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        torch.random.set_rng_state(rng["torch_rng"])
        if torch.cuda.is_available() and rng.get("cuda_rng"):
            saved_cuda_states = rng["cuda_rng"]
            num_devices = torch.cuda.device_count()
            if len(saved_cuda_states) == num_devices:
                torch.cuda.set_rng_state_all(saved_cuda_states)
            else:
                logger.warning(
                    f"CUDA RNG state mismatch: checkpoint has {len(saved_cuda_states)} "
                    f"device(s), current environment has {num_devices}. "
                    "Skipping CUDA RNG restore (training is still deterministic "
                    "within the new world size)."
                )
        np.random.set_state(rng["numpy_rng"])
        random.setstate(rng["python_rng"])

    state_path = ckp / "training_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state.get("global_step", 0)), int(state.get("epoch", 0))
    return 0, 0


def _rebuild_scheduler_for_resume(
    optimizer,
    resume_config: "ResumeConfig",
    pretrain_config: "PretrainingConfig",
    checkpoint_dir: str,
    start_step: int,
    total_steps: int,
    warmup_steps: int,
    is_main: bool = True,
) -> LambdaLR:
    """Rebuild the LR scheduler for a resumed run.

    The default behaviour reconstructs the *original* LR curve from the
    checkpoint's saved ``pretrain_config.yaml`` (if present) so that the
    schedule is bit-exact even if the user has since changed their
    ``pretrain.yaml``.  Explicit overrides in ``resume_config`` take
    precedence over the saved values.

    Falls back to the current ``pretrain_config`` if the checkpoint
    pre-dates this feature (no ``pretrain_config.yaml`` file).

    Returns a :class:`LambdaLR` positioned at *start_step*.
    """
    ckp = Path(checkpoint_dir)

    # -- Load saved schedule metadata from checkpoint ----------------------
    saved_warmup = warmup_steps
    saved_total = total_steps
    saved_lr = pretrain_config.adam_learning_rate
    saved_muon_lr = pretrain_config.muon_learning_rate
    saved_decay = pretrain_config.lr_decay_to_fraction
    saved_schedule = pretrain_config.lr_schedule

    state_path = ckp / "training_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        saved_warmup = int(state.get("warmup_steps", saved_warmup))
        saved_total = int(state.get("total_steps", saved_total))

    saved_config_path = ckp / "pretrain_config.yaml"
    if saved_config_path.exists():
        with open(saved_config_path) as f:
            saved_cfg = yaml.safe_load(f) or {}
        saved_lr = float(saved_cfg.get("learning_rate", saved_lr))
        saved_muon_lr = float(saved_cfg.get("muon_learning_rate", saved_muon_lr))
        saved_decay = float(saved_cfg.get("lr_decay_to_fraction", saved_decay))
        saved_schedule = str(saved_cfg.get("lr_schedule", saved_schedule))

        # Detect config drift and warn (informational only).
        if is_main:
            diffs: list[str] = []
            for key in (
                "learning_rate", "muon_learning_rate", "warmup_steps",
                "lr_schedule", "lr_decay_to_fraction",
            ):
                ckpt_val = saved_cfg.get(key)
                yaml_val = getattr(pretrain_config, key, None)
                if ckpt_val is not None and yaml_val is not None and ckpt_val != yaml_val:
                    diffs.append(f"  {key}: {ckpt_val} (checkpoint) \u2192 {yaml_val} (current YAML)")
            if diffs:
                logger.warning(
                    "Schedule config drift detected between checkpoint and current YAML:\n"
                    + "\n".join(diffs)
                    + "\nUsing checkpoint values as base. "
                    "Override explicitly via resume.* fields if intended."
                )
    else:
        if is_main:
            logger.info(
                "No pretrain_config.yaml in checkpoint (pre-upgrade checkpoint). "
                "Using current YAML schedule params as base."
            )

    # -- Apply explicit overrides from resume config -----------------------
    eff_warmup = saved_warmup
    eff_total = saved_total
    eff_lr = saved_lr
    eff_muon_lr = saved_muon_lr
    eff_decay = saved_decay
    eff_schedule = saved_schedule

    if resume_config.extra_epochs is not None and resume_config.extra_epochs > 0:
        # total_steps was already extended by _prepare_run_and_steps;
        # use the caller-provided value which accounts for extra epochs.
        eff_total = total_steps

    if resume_config.warmup_steps is not None:
        eff_warmup = resume_config.warmup_steps
    if resume_config.learning_rate is not None:
        eff_lr = resume_config.learning_rate
    if resume_config.muon_learning_rate is not None:
        eff_muon_lr = resume_config.muon_learning_rate
    if resume_config.lr_schedule is not None:
        eff_schedule = resume_config.lr_schedule
    if resume_config.lr_decay_to_fraction is not None:
        eff_decay = resume_config.lr_decay_to_fraction
    if resume_config.skip_warmup:
        eff_warmup = 0

    # -- Override optimizer base LRs before creating scheduler -------------
    if isinstance(optimizer, (Muon, NorMuon)):
        # group[0] = muon, group[1+] = adamw / scalar
        optimizer.param_groups[0]["lr"] = eff_muon_lr
        for pg in optimizer.param_groups[1:]:
            if pg.get("algorithm") == "adamw":
                pg["lr"] = eff_lr
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = eff_lr

    # -- Build scheduler ---------------------------------------------------
    if resume_config.reset_scheduler:
        remaining = max(1, eff_total - start_step)
        if is_main:
            logger.info(
                f"Resetting LR schedule from step 0: remaining_steps={remaining}, "
                f"warmup={eff_warmup}, schedule={eff_schedule}, "
                f"base_lr={eff_lr:.2e}, decay_to={eff_decay}"
            )
        scheduler = _create_scheduler(
            optimizer, eff_warmup, remaining,
            eff_lr, eff_decay, eff_schedule,
        )
    else:
        # Reconstruct the original schedule and position at start_step.
        # last_epoch=N-1 causes __init__ to call step() once → last_epoch=N.
        scheduler = _create_scheduler(
            optimizer, eff_warmup, eff_total,
            eff_lr, eff_decay, eff_schedule,
            last_epoch=max(-1, start_step - 1),
        )

    current_lrs = [pg["lr"] for pg in optimizer.param_groups]
    if is_main:
        logger.info(
            f"Resume schedule built: warmup={eff_warmup}, total={eff_total}, "
            f"schedule={eff_schedule}, decay_to={eff_decay}, "
            f"current_lrs={[f'{lr:.2e}' for lr in current_lrs]}"
        )

    return scheduler


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pure_pretraining(
    model: PureProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:

    # Set allocator config (respect existing user settings)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Enable reduced-precision for training performance
    if pretrain_config.bf16 or pretrain_config.tf32:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _set_seed(pretrain_config.seed)
    tokenizer = model.tokenizer
    device = torch.device(get_device())

    # ---- Dataset ----
    dataset_dir = Path(pretrain_config.dataset_dir)
    manifest = read_manifest(dataset_dir)
    validate_manifest_for_pipeline(manifest=manifest, expected_mode="pretrain")
    if manifest.max_seq_len <= 0:
        raise ValueError(f"Invalid manifest max_seq_len: {manifest.max_seq_len}")

    # Capture config/manifest for checkpoint serialization so every saved
    # checkpoint is self-contained (no external pretrain.yaml needed).
    _model_config = getattr(model, "model_config", None)
    _manifest = manifest

    model_max_pos = int(getattr(model.config, "max_position_embeddings", 0))
    if model_max_pos > 0 and manifest.max_seq_len > model_max_pos:
        raise ValueError(
            f"Dataset max_seq_len ({manifest.max_seq_len}) exceeds "
            f"model max_position_embeddings ({model_max_pos})."
        )

    train_shard_dir = dataset_dir / manifest.train_dir
    val_shard_dir = dataset_dir / manifest.val_dir
    for d in (train_shard_dir, val_shard_dir):
        if not d.exists():
            raise FileNotFoundError(f"Shard directory not found: {d}")

    logger.info(f"Loaded config from manifest: {dataset_dir}")
    logger.info(f"  train_shards: {train_shard_dir}")
    logger.info(f"  val_shards: {val_shard_dir}")
    logger.info(f"  max_length: {manifest.max_seq_len}")
    logger.info(f"  train_sequences: {manifest.train_sequences}")
    logger.info(f"  val_sequences: {manifest.val_sequences}")

    logger.info("Using ShardedDataset for pre-tokenized binary shards")
    train_ds = ShardedDataset(data_dir=str(train_shard_dir))
    val_ds = ShardedDataset(data_dir=str(val_shard_dir))

    use_packing = bool(pretrain_config.use_packing)
    use_static_inp_size = bool(pretrain_config.use_static_inp_size and use_packing)

    # ---- Batch sizing ----
    create_dirs(pretrain_config.ckp_dir)
    effective_world_size = _resolve_world_size(pretrain_config)

    inferred_grad_accum_steps = pretrain_config.inferred_grad_accum_steps
    global_batch_size_samples = pretrain_config.global_batch_size_samples
    achieved_global_batch_tokens = pretrain_config.achieved_global_batch_tokens
    if any(v is None for v in (inferred_grad_accum_steps, global_batch_size_samples, achieved_global_batch_tokens)):
        raise ValueError("Batch setup missing on PretrainingConfig. Run through nanoplm CLI.")

    world_tokens_per_micro_step = achieved_global_batch_tokens // max(1, inferred_grad_accum_steps)
    logger.info(
        f"Batch setup: target_global_batch_size={pretrain_config.global_batch_size:,} tokens, "
        f"micro_step_tokens={world_tokens_per_micro_step:,}, "
        f"grad_accum_steps={inferred_grad_accum_steps}, "
        f"effective_global_batch_size={achieved_global_batch_tokens:,} tokens"
    )

    # ---- Run naming / output dir ----
    (
        _run_name,
        wandb_run_name,
        output_dir,
        num_epochs,
        logging_steps,
        eval_steps,
        save_steps,
        _resume_step,
    ) = prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=manifest.train_sequences,
        global_batch_size_samples=global_batch_size_samples,
    )

    num_workers = get_num_workers(pretrain_config.num_workers, effective_world_size)
    pin_memory = device.type == "cuda"
    # Disable persistent workers when packing: TokenPackingDataset + persistent_workers
    # can cause hangs near epoch boundaries (set_epoch doesn't reach workers, worker
    # queue non-determinism). Workers restart each epoch, fixing the stall.
    persistent_workers = num_workers > 0 and not use_packing
    prefetch_factor = pretrain_config.prefetch_factor if num_workers > 0 else None

    # ---- Distributed init ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = bool(pretrain_config.multi_gpu)


    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            if backend == "nccl":
                dist.init_process_group(backend=backend, device_id=device)
            else:
                dist.init_process_group(backend=backend)

    is_main = (not distributed) or dist.get_rank() == 0
    eval_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else SequentialSampler(val_ds)

    # ---- Collator / Sampler / Packing ----
    train_sampler = None

    if use_packing:
        tokens_per_micro = pretrain_config.micro_batch_size * manifest.max_seq_len
        inner_sampler = DistributedSampler(train_ds, shuffle=True, seed=pretrain_config.seed) if distributed else RandomSampler(train_ds)

        train_ds = TokenPackingDataset(
            train_ds,
            max_tokens_per_batch=tokens_per_micro,
            drop_last=True,
            split_samples=False,
            sampler=inner_sampler,
        )
        inner_collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )
        if use_static_inp_size:
            min_tokens_per_seq = max(
                1,
                int(manifest.min_seq_len) + int(tokenizer.num_special_tokens_to_add(pair=False)),
            )
            # +1 leaves room for an optional trailing dummy sequence used for fixed-token padding.
            max_sequences_per_batch = max(1, tokens_per_micro // min_tokens_per_seq + 1)
            seq_count_buckets = build_power_of_two_buckets(max_sequences_per_batch, min_power_of_two=32)
            max_seqlen_buckets = build_power_of_two_buckets(int(manifest.max_seq_len), min_power_of_two=32)
            collator = DataCollatorWithFlattening(
                collator=inner_collator,
                return_position_ids=True,
                fixed_tokens_per_batch=tokens_per_micro,
                seq_count_buckets=seq_count_buckets,
                max_seqlen_buckets=max_seqlen_buckets,
            )
            logger.info(
                "Sequence packing ENABLED (static input size + bucketed metadata), "
                f"target={tokens_per_micro:,} tokens, "
                f"min_tokens_per_seq={min_tokens_per_seq}, "
                f"seq_count_buckets={seq_count_buckets}, "
                f"max_seqlen_buckets={max_seqlen_buckets}"
            )
        else:
            collator = DataCollatorWithFlattening(
                collator=inner_collator,
                pad_to_multiple_of=8,
                return_position_ids=True,
            )
            logger.info(
                "Sequence packing ENABLED (dynamic metadata), "
                f"target={tokens_per_micro:,} tokens"
            )
    else:
        collator = ProtDataCollatorForLM(
            tokenizer=tokenizer,
            mlm_probability=pretrain_config.mlm_probability,
            mask_token_probability=pretrain_config.mask_replace_prob,
            random_token_probability=pretrain_config.random_token_prob,
            keep_probability=pretrain_config.keep_probability,
        )
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=pretrain_config.seed) if distributed else RandomSampler(train_ds)

    # ---- FP8 (optional) ----
    if pretrain_config.fp8:
        if device.type != "cuda":
            raise ValueError("fp8=True requires CUDA.")
        fp8_filter = lambda mod, _: isinstance(mod, torch.nn.Linear) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
        convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name("tensorwise"), module_filter_fn=fp8_filter)
        n_fp8 = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
        n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        logger.info(f"FP8 enabled (tensorwise): converted={n_fp8} linear layers, skipped={max(0, n_lin - n_fp8)}")

    # ---- Model to device + compile + FSDP ----
    model.to(device)

    compile_dynamic = not bool(use_packing and use_static_inp_size)

    # Precision detection (needed before FSDP MixedPrecisionPolicy)
    use_bf16 = pretrain_config.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = pretrain_config.bf16 and (
        (device.type == "cuda" and not torch.cuda.is_bf16_supported()) or device.type == "mps"
    )

    fsdp_mesh = None
    if distributed:
        fsdp_kwargs: dict = {}
        if use_bf16:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fsdp_mesh = init_device_mesh("cuda", (effective_world_size,))
        for layer in model.model.layers:
            fully_shard(layer, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)
        fully_shard(model, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_kwargs)

        # Explicit prefetching for FSDP2
        if N_PREFETCH_LAYERS_FSDP2 > 1:
            layers = model.model.layers
            for i, layer in enumerate(layers):
                if i + 1 < len(layers):
                    layer.set_modules_to_forward_prefetch(layers[i + 1 : i + 1 + N_PREFETCH_LAYERS_FSDP2])
                if i - 1 >= 0:
                    layer.set_modules_to_backward_prefetch(list(reversed(layers[max(0, i - N_PREFETCH_LAYERS_FSDP2) : i])))

    # Keep orig_model reference for checkpointing/eval (eval changes shapes → recompilation)
    orig_model = model
    compile_mode = (
        "max-autotune-no-cudagraphs"
        if getattr(pretrain_config, "use_compile_max_autotune", False)
        else None
    )

    # ---- TorchInductor knobs (must be set before torch.compile) ----
    if device.type == "cuda":
        try:
            import torch._inductor.config as inductor_config

            # NOTE: Inductor codecache compilation runs in subprocess workers by
            # default. Those workers read config from environment variables at
            # import time, so set the env var as well as the in-process config.
            persistent_reductions = bool(
                getattr(pretrain_config, "compile_triton_persistent_reductions", False)
            )
            mix_order_reduction = bool(
                getattr(pretrain_config, "compile_triton_mix_order_reduction", False)
            )
            os.environ["TORCHINDUCTOR_PERSISTENT_REDUCTIONS"] = (
                "1" if persistent_reductions else "0"
            )
            os.environ["TORCHINDUCTOR_MIX_ORDER_REDUCTION"] = (
                "1" if mix_order_reduction else "0"
            )
            inductor_config.triton.persistent_reductions = persistent_reductions
            inductor_config.triton.mix_order_reduction = mix_order_reduction
            logger.info(
                "TorchInductor: triton.persistent_reductions=%s (env TORCHINDUCTOR_PERSISTENT_REDUCTIONS=%s); "
                "triton.mix_order_reduction=%s (env TORCHINDUCTOR_MIX_ORDER_REDUCTION=%s)",
                persistent_reductions,
                os.environ.get("TORCHINDUCTOR_PERSISTENT_REDUCTIONS"),
                mix_order_reduction,
                os.environ.get("TORCHINDUCTOR_MIX_ORDER_REDUCTION"),
            )
        except Exception:
            logger.exception("Failed to apply TorchInductor config overrides; continuing.")

    if compile_mode is None:
        model = torch.compile(model, dynamic=compile_dynamic)
        logger.info(f"Model compiled with torch.compile(dynamic={compile_dynamic})")
    else:
        model = torch.compile(model, dynamic=compile_dynamic, mode=compile_mode)
        logger.info(
            "Model compiled with torch.compile(dynamic=%s, mode=%s)",
            compile_dynamic,
            compile_mode,
        )

    # ---- DataLoaders ----
    eval_collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
    )

    dl_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    if use_packing:
        train_loader = DataLoader(train_ds, batch_size=None, collate_fn=collator, **dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=pretrain_config.micro_batch_size, collate_fn=collator, drop_last=False, **dl_kwargs)
    eval_loader = DataLoader(val_ds, sampler=eval_sampler, batch_size=pretrain_config.micro_batch_size, collate_fn=eval_collator, drop_last=False, **dl_kwargs)

    # ---- Optimizer / Scheduler ----
    optimizer = _create_optimizer(model, pretrain_config, distributed_mesh=fsdp_mesh)

    synced_len = _sync_train_loader_len(len(train_loader), distributed, device)
    steps_per_epoch = max(1, math.ceil(synced_len / max(1, inferred_grad_accum_steps)))
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(pretrain_config.warmup_steps, total_steps)
    scheduler = _create_scheduler(
        optimizer, warmup_steps, total_steps,
        pretrain_config.adam_learning_rate, pretrain_config.lr_decay_to_fraction,
        pretrain_config.lr_schedule,
    )

    # ---- Resume ----
    start_step, start_epoch = 0, 0
    resume_micro_step, resume_epoch = 0, 0
    if resume_config and resume_config.is_resume:
        logger.info(f"Resuming from checkpoint: {resume_config.checkpoint_dir}")
        start_step, start_epoch = _load_checkpoint(
            model, optimizer, scheduler, resume_config.checkpoint_dir,
            device, distributed, load_scheduler=False,
        )
        logger.info(f"Resumed at global_step={start_step}, epoch={start_epoch}")

        # Rebuild the LR scheduler from the checkpoint's saved schedule
        # params (with any explicit overrides from resume config).
        scheduler = _rebuild_scheduler_for_resume(
            optimizer=optimizer,
            resume_config=resume_config,
            pretrain_config=pretrain_config,
            checkpoint_dir=resume_config.checkpoint_dir,
            start_step=start_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            is_main=is_main,
        )

        resume_epoch = start_epoch
        steps_done = max(0, start_step - start_epoch * steps_per_epoch)
        resume_micro_step = steps_done * inferred_grad_accum_steps
        if resume_micro_step >= synced_len:
            resume_micro_step = 0
            start_epoch = min(start_epoch + 1, num_epochs)
            resume_epoch = start_epoch
        elif resume_micro_step > 0:
            logger.info(f"Skipping {resume_micro_step} micro-steps in resumed epoch {resume_epoch}")

    # ---- W&B ----
    wandb_enabled = False
    if is_main:
        try:
            wandb.init(
                project=pretrain_config.project_name,
                name=wandb_run_name,
                config={"pretrain": pretrain_config.__dict__, "total_steps": total_steps, "warmup_steps": warmup_steps},
            )
            wandb_enabled = wandb.run is not None
            if wandb_enabled:
                wandb.define_metric("train/global_step")
                wandb.define_metric("time_elapsed_sec", step_metric="train/global_step", step_sync=True)
                wandb.define_metric("train/time_elapsed_sec", step_metric="train/global_step", step_sync=True)
                wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
                upload_run_source_snapshot()
        except Exception as exc:
            logger.warning(f"W&B init failed, continuing without logging. Error: {exc}")

    # ---- AMP / TF32 ----
    amp_dtype: Optional[torch.dtype] = None
    if use_bf16 and not distributed:
        amp_dtype = torch.bfloat16
    elif use_fp16:
        amp_dtype = torch.float16

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(pretrain_config.tf32)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(pretrain_config.tf32)

    scaler = (
        torch.amp.GradScaler(enabled=(device.type == "cuda" and use_fp16))
        if torch.cuda.is_available() else None
    )

    # ---- MFU estimation ----
    _raw = model if hasattr(model, "config") else getattr(model, "_orig_mod", model)
    _cfg = _raw.config
    _flops_per_token = _estimate_model_flops_per_token(_cfg, manifest.max_seq_len)
    _peak_flops = H100_PEAK_TFLOPS * 1e12
    logger.info(f"MFU estimation: {_flops_per_token:,} training FLOPs/token, H100 peak = {H100_PEAK_TFLOPS} TFLOPS")

    _run_t0 = time.perf_counter()
    logger.info(
        f"Starting pure-torch training: epochs={num_epochs}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, grad_accum={inferred_grad_accum_steps},"
        f"achieved_global_batch_size={achieved_global_batch_tokens:,} tokens"
    )
    logger.info(
        f"Precision config: bf16={use_bf16}, fp16={use_fp16}, "
        f"tf32={(pretrain_config.tf32 and device.type == 'cuda')}, fp8={pretrain_config.fp8}"
    )
    max_grad_norm = float(pretrain_config.max_grad_norm)
    logger.info(
        "Gradient clipping: "
        f"{'disabled (max_grad_norm=inf)' if math.isinf(max_grad_norm) else f'max_grad_norm={max_grad_norm}'}"
    )

    # ---- RePO activation schedule ----
    # Decoupled from LR warmup: when repo_rope_warmup_steps is not set,
    # fall back to warmup_steps for backward-compatible behavior.
    _repo_rope_warmup_cfg = getattr(pretrain_config, "repo_rope_warmup_steps", None)
    repo_rope_warmup_steps = (
        warmup_steps
        if _repo_rope_warmup_cfg is None
        else max(0, int(_repo_rope_warmup_cfg))
    )
    _use_repo = getattr(_cfg, "use_repo", False)
    if _use_repo:
        # Access the inner ModernBertModel to toggle repo_active
        _repo_model = _raw.model if hasattr(_raw, "model") else None
        if _repo_model is not None and hasattr(_repo_model, "repo_active"):
            # If resuming past RePO warmup, enable immediately.
            _repo_model.repo_active = start_step >= repo_rope_warmup_steps
            logger.info(
                f"RePO: repo_after_n_layers={_cfg.repo_after_n_layers}, "
                f"repo_rope_warmup_steps={repo_rope_warmup_steps}, "
                f"active={'yes' if _repo_model.repo_active else f'no (activates after step {repo_rope_warmup_steps})'}"
            )
        else:
            logger.warning("RePO: use_repo=True but model has no repo_active attribute")
            _use_repo = False

    # ---- Training loop ----
    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    accum_loss = torch.zeros((), device=device)
    window_loss = torch.zeros((), device=device)
    window_steps = 0
    discard_accumulation = False
    token_count = torch.zeros((), dtype=torch.long, device=device)
    raw_token_count = torch.zeros((), dtype=torch.long, device=device)
    log_window_t0 = time.perf_counter()
    first_step_of_run = True

    epoch_setter = train_ds if use_packing else train_sampler
    # [0]=loss, [1]=grad_norm (local)
    log_buf = torch.empty(2, device=device, dtype=torch.float32)

    profiler_ctx, profiler_step_cb = _make_pure_profiler(pretrain_config, output_dir, is_main)

    with profiler_ctx:
        for epoch in range(start_epoch, num_epochs):
            if epoch_setter is not None and hasattr(epoch_setter, "set_epoch"):
                epoch_setter.set_epoch(epoch)

            train_iter = iter(train_loader)
            # Reset timing window AFTER dataloader workers are ready so the
            # first logging window doesn't include iter(train_loader) overhead.
            if device.type == "cuda":
                torch.cuda.synchronize()
            log_window_t0 = time.perf_counter()

            epoch_ended_early = False
            for micro_step in range(synced_len):
                has_batch = True
                try:
                    batch = next(train_iter)
                except StopIteration:
                    has_batch = False
                    batch = None

                # When packing + num_workers > 0, greedy bin-packing can produce
                # different batch counts per rank.  Coordinate so all ranks break
                # together to avoid FSDP / NCCL deadlock.
                if distributed and dist.is_initialized():
                    has_batch_t = torch.tensor(1 if has_batch else 0, device=device, dtype=torch.int32)
                    dist.all_reduce(has_batch_t, op=dist.ReduceOp.MIN)
                    if int(has_batch_t.item()) == 0:
                        epoch_ended_early = True
                        break
                elif not has_batch:
                    epoch_ended_early = True
                    break

                if resume_micro_step > 0 and epoch == resume_epoch and micro_step < resume_micro_step:
                    continue

                at_accum_boundary = (micro_step + 1) % inferred_grad_accum_steps == 0
                if discard_accumulation:
                    if at_accum_boundary:
                        discard_accumulation = False
                    continue

                batch = _move_batch_to_device(batch, device)
                upcoming_step = global_step + 1

                if distributed and N_PREFETCH_LAYERS_FSDP2 > 1:
                    model.unshard()

                # FSDP2: only reduce-scatter gradients at regular accumulation
                # boundaries.  (Do NOT use synced_len as a fallback — the loop may
                # break early via all_reduce, making synced_len unreachable.
                # Partial accumulation at epoch end is discarded, not stepped.)
                if distributed:
                    sync = (micro_step + 1) % inferred_grad_accum_steps == 0
                    model.set_requires_gradient_sync(sync)

                # Token counting
                if "num_valid_tokens" in batch:
                    token_count += batch["num_valid_tokens"]
                    raw_token_count += batch["input_ids"].numel()
                elif "attention_mask" in batch:
                    token_count += batch["attention_mask"].sum()
                    raw_token_count += batch["attention_mask"].numel()
                else:
                    token_count += batch["input_ids"].numel()
                    raw_token_count += batch["input_ids"].numel()

                # Forward
                amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
                with amp_ctx:
                    fwd_kwargs = dict(input_ids=batch["input_ids"], labels=batch["labels"])
                    if "attention_mask" in batch:
                        fwd_kwargs["attention_mask"] = batch["attention_mask"]
                    if "cu_seqlens" in batch:
                        cu_seqlens = batch["cu_seqlens"]
                        # cu_seqlens has shape (num_seqs+1,) which varies per packing
                        # bucket.  Mark dim-0 dynamic so torch.compile(dynamic=False)
                        # doesn't create a separate compiled graph per bucket count.
                        torch._dynamo.mark_dynamic(cu_seqlens, 0)
                        fwd_kwargs["cu_seqlens"] = cu_seqlens
                        fwd_kwargs["max_seqlen"] = batch["max_seqlen"]
                    if "position_ids" in batch:
                        fwd_kwargs["position_ids"] = batch["position_ids"]
                    out = model(**fwd_kwargs)

                loss = out["loss"] if isinstance(out, dict) else out.loss
                if not torch.isfinite(loss.detach()).all():
                    rank = dist.get_rank() if distributed and dist.is_initialized() else 0
                    logger.error(
                        "Skipping optimizer step %d due to non-finite loss on rank %d (epoch=%d micro_step=%d).",
                        upcoming_step,
                        rank,
                        epoch,
                        micro_step,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss.zero_()
                    discard_accumulation = not at_accum_boundary
                    continue
                loss = loss / inferred_grad_accum_steps

                # Backward
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss = accum_loss + loss.detach()

                # Skip to next micro-step if not at a regular accumulation
                # boundary.  We intentionally do NOT treat the last micro-step of
                # the epoch as a boundary: the loop can break early (via the
                # all_reduce exhaustion check above), making synced_len
                # unreachable.  Any partial accumulation at epoch end is
                # discarded in the epoch-boundary cleanup below.
                if not at_accum_boundary:
                    continue

                # Optimizer step
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    if "non-finite" not in str(exc).lower():
                        raise
                    bad_grad = _first_nonfinite_grad(model)
                    bad_name = bad_grad[0] if bad_grad is not None else "<unknown>"
                    rank = dist.get_rank() if distributed and dist.is_initialized() else 0
                    logger.error(
                        "Skipping optimizer step %d due to non-finite gradient in %s on rank %d (epoch=%d micro_step=%d).",
                        upcoming_step,
                        bad_name,
                        rank,
                        epoch,
                        micro_step,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss.zero_()
                    continue

                step_skipped = False
                if scaler is not None and scaler.is_enabled():
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_skipped = scaler.get_scale() < old_scale
                else:
                    optimizer.step()
                if _has_nonfinite_params(model):
                    bad_param = _first_nonfinite_param(model)
                    bad_name = bad_param[0] if bad_param is not None else "<unknown>"
                    rank = dist.get_rank() if distributed and dist.is_initialized() else 0
                    raise RuntimeError(
                        f"Non-finite parameter detected after optimizer step {upcoming_step} "
                        f"in {bad_name} on rank {rank} (epoch={epoch} micro_step={micro_step})."
                    )

                if isinstance(optimizer, (Muon, NorMuon)):
                    muon_lr = optimizer.param_groups[0]["lr"]
                    learning_rate = optimizer.param_groups[1]["lr"]
                else:
                    learning_rate = optimizer.param_groups[0]["lr"]
                    muon_lr = None

                if not step_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                profiler_step_cb(global_step)

                # RePO: activate once RePO warmup completes.
                if _use_repo and global_step == repo_rope_warmup_steps:
                    _repo_model.repo_active = True
                    if is_main:
                        logger.info(f"[step {global_step}] RePO activated (repo warmup complete)")

                window_loss += accum_loss.detach()
                window_steps += 1
                accum_loss.zero_()

                # ---- Logging ----
                should_log = global_step % logging_steps == 0
                vram_log = ""
                tps = mfu = tok = raw_tok = 0.0
                step_tok = step_raw_tok = 0.0
                real_tps = real_tps_log = 0.0
                avg_step_ms = 0.0
                if should_log:
                    # Synchronize and measure only at logging boundaries to amortize sync cost.
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    window_dt = max(1e-9, t1 - log_window_t0)
                    log_window_t0 = t1
                    avg_step_ms = (window_dt * 1000.0) / max(1, window_steps)
                    tps = int((achieved_global_batch_tokens * window_steps) / window_dt)
                    # For real vs raw token tracking, still read the per-rank counters
                    log_buf[0] = window_loss / max(1, window_steps)
                    log_buf[1] = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else (grad_norm if isinstance(grad_norm, torch.Tensor) else float(grad_norm))
                    log_vals = log_buf.cpu()
                    avg_loss = float(log_vals[0])
                    grad_norm_val = float(log_vals[1])
                    tok = float(token_count.item())
                    raw_tok = float(raw_token_count.item())
                    if distributed and dist.is_initialized():
                        tok_buf = torch.tensor([tok, raw_tok], device=device)
                        dist.all_reduce(tok_buf, op=dist.ReduceOp.SUM)
                        tok, raw_tok = float(tok_buf[0].item()), float(tok_buf[1].item())
                    step_tok = tok / max(1, window_steps)
                    step_raw_tok = raw_tok / max(1, window_steps)
                    real_tps = tok / window_dt
                    real_tps_log = int(real_tps)
                    mfu = (
                        _flops_per_token * real_tps
                    ) / (_peak_flops * max(effective_world_size, 1))

                    token_count.zero_()
                    raw_token_count.zero_()

                    # VRAM logging only at eval steps (expensive: all_reduce + multiple .item())
                    if global_step % eval_steps == 0:
                        vram_log = _format_vram_for_log(
                            device=device,
                            distributed=distributed,
                            reset_peak=True,
                        )
                if should_log and is_main:
                    waste = (1.0 - step_tok / max(step_raw_tok, 1)) * 100
                    muon_str = f"muon_lr={muon_lr:.2e} " if muon_lr is not None else ""
                    wall_elapsed = time.perf_counter() - _run_t0
                    logger.info(
                        f"[step {global_step}/{total_steps}] "
                        f"loss={avg_loss:.4f} lr={learning_rate:.2e} {muon_str}"
                        f"grad_norm={grad_norm_val:.4f} tok/s={tps:,} real_tok/s={real_tps_log:,} "
                        f"real_tok/step={step_tok:,.0f} "
                        f"dt={avg_step_ms:.2f}ms wall={wall_elapsed:.1f}s waste={waste:.1f}% "
                        f"h100_mfu={mfu:.2%} {vram_log}"
                    )
                    payload = {
                        "train/global_step": global_step,
                        "train/loss": avg_loss,
                        "train/grad_norm": grad_norm_val,
                        "train/learning_rate": learning_rate,
                        "train/epoch": epoch + (micro_step + 1) / synced_len,
                        "train/tokens_per_sec": tps,
                        "train/real_tokens_per_sec": real_tps,
                        "train/step_real_tokens": step_tok,
                        "train/step_raw_tokens": step_raw_tok,
                        "train/packing_waste_pct": waste,
                        "time_elapsed_sec": wall_elapsed,
                        "train/time_elapsed_sec": wall_elapsed,
                    }
                    if muon_lr is not None:
                        payload["train/muon_lr"] = muon_lr
                    if wandb_enabled and wandb.run is not None:
                        try:
                            wandb.log(payload)
                        except Exception as exc:
                            wandb_enabled = False
                            logger.warning(f"W&B log failed; disabling. Error: {exc}")

                # GC: after first step, freeze; then periodic collect
                if first_step_of_run:
                    first_step_of_run = False
                    gc.collect(); gc.freeze(); gc.disable()
                elif global_step % 5000 == 0:
                    gc.collect()

                if should_log:
                    window_loss.zero_()
                    window_steps = 0

                # ---- Eval ----
                if global_step % eval_steps == 0:
                    eval_loss = _evaluate(orig_model, eval_loader, device, distributed, amp_dtype)
                    if is_main:
                        logger.info(f"[step {global_step}] eval_loss={eval_loss:.4f}")
                        if wandb_enabled and wandb.run is not None:
                            try:
                                wandb.log({"train/global_step": global_step, "eval/loss": eval_loss})
                            except Exception as exc:
                                wandb_enabled = False
                                logger.warning(f"W&B log failed; disabling. Error: {exc}")
                    model.train()

                # ---- Save ----
                if global_step % save_steps == 0:
                    _save_checkpoint(
                        model, optimizer, scheduler, global_step, epoch,
                        output_dir, logging_steps, eval_steps, save_steps,
                        distributed, is_main,
                        model_config=_model_config, manifest=_manifest,
                        pretrain_config=pretrain_config,
                        total_steps=total_steps, warmup_steps=warmup_steps,
                    )

            # ---- Epoch boundary cleanup ----
            # When the inner loop ends (either by exhausting synced_len or
            # breaking early via the all_reduce exhaustion check), there may
            # be partial gradient-accumulation state left over from micro-steps
            # that ran forward+backward but never reached an optimizer step.
            # Flush everything to prevent gradient leak (stale .grad tensors
            # carrying into next epoch), loss contamination, FSDP gradient
            # desync (partial micro-steps had sync=False), and token/loss
            # counter bleed across the boundary.  Timing window is reset at
            # the top of the next epoch iteration, after iter(train_loader).
            epoch_fwd_count = micro_step if epoch_ended_early else synced_len
            partial_discarded = epoch_fwd_count % inferred_grad_accum_steps
            if is_main:
                logger.info(
                    "Epoch %d/%d complete: %d micro-steps, %d optimizer steps%s",
                    epoch + 1, num_epochs, epoch_fwd_count,
                    epoch_fwd_count // inferred_grad_accum_steps,
                    f" ({partial_discarded} trailing micro-step(s) discarded)"
                    if partial_discarded else "",
                )
            optimizer.zero_grad(set_to_none=True)
            accum_loss.zero_()
            discard_accumulation = False
            token_count.zero_()
            raw_token_count.zero_()
            window_loss.zero_()
            window_steps = 0

            if resume_micro_step > 0 and epoch == resume_epoch:
                resume_micro_step = 0

    # ---- Finalize ----
    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    _save_checkpoint(
        model, optimizer, scheduler, global_step, num_epochs,
        output_dir, logging_steps, eval_steps, save_steps,
        distributed, is_main,
        model_config=_model_config, manifest=_manifest,
        pretrain_config=pretrain_config,
        total_steps=total_steps, warmup_steps=warmup_steps,
    )

    if is_main and wandb_enabled and wandb.run is not None:
        try:
            (Path(output_dir) / "wandb_run_id.txt").write_text(wandb.run.id, encoding="utf-8")
            wandb.finish()
        except Exception as exc:
            logger.warning(f"W&B finalize failed. Error: {exc}")

    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    logger.info("Pure-torch pretraining complete.")
