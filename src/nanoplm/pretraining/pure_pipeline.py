"""Pure PyTorch pretraining pipeline.

Same outer behavior as the HF Trainer pipeline (datasets, run naming,
checkpoint layout, resume mechanics), but the training loop is plain torch.
"""

import gc
import json
import math
import os
import yaml
from copy import deepcopy
from dataclasses import asdict
try:
    from dion import Muon
except ImportError:
    Muon = None  # type: ignore[assignment,misc]
try:
    from dion import NorMuon
except ImportError:
    NorMuon = None  # type: ignore[assignment,misc]
try:
    from dion import MuonH
except ImportError:
    MuonH = None  # type: ignore[assignment,misc]
try:
    from dion import Aurora
except ImportError:
    Aurora = None  # type: ignore[assignment,misc]
_DION_OPTIMIZER_TYPES = tuple(t for t in (Muon, NorMuon, MuonH, Aurora) if t is not None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
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
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from nanoplm.data.infra.manifest import validate_manifest_for_pipeline, DataManifest
from nanoplm.pretraining.collator import (
    DataCollatorWithFlattening,
    ProtDataCollatorForLM,
    build_power_of_two_buckets,
    mlm_masking_kwargs_from_config,
)
from nanoplm.data.datasets.pretrain import ShardedDataset, TokenPackingDataset
from nanoplm.pretraining import cpu_offload
from nanoplm.pretraining.fp8 import (
    convert_fp8_targets,
    iter_te_cpu_offload_target_wrappers,
    te_fp8_autocast_context,
    te_fp8_recipe_from_config,
)
from nanoplm.pretraining.models.modern_bert.modeling import (
    MHCLiteBlock,
    MHCLiteSublayersLayer,
    ModernBertEncoderLayer,
)
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.optim import (
    OptimizerGroup,
    build_muon_optimizer,
    is_muon_optimizer,
    MuonAdamWGroup,
)
from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
from nanoplm.pretraining.utils import (
    compute_batch_setup,
    format_passes_hint,
    get_num_workers,
    prepare_run_and_steps,
    resolve_total_steps,
    tokens_to_steps,
    StepHorizon,
)
from nanoplm.utils.common import create_dirs, get_device, resolve_world_size
from nanoplm.utils.logger import logger
from nanoplm.utils.wandb_artifacts import upload_run_source_snapshot
from nanoplm.pretraining.debug_metrics import LayerwiseDebugTracker

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# H100 SXM peak BF16 tensor-core throughput (TFLOPS).
H100_PEAK_TFLOPS = 989.4

# https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#forward-backward-with-prefetching
N_PREFETCH_LAYERS_FSDP2 = 1

# How many micro-steps before the synced epoch end the per-step batch
# exhaustion allreduce kicks in.  synced_len is the all-rank MIN of exact
# per-rank loader lengths, so ranks can only run dry inside this window;
# running the (blocking) coordination allreduce every step would serialize
# host and device once per micro-step.
BATCH_COORD_TAIL_STEPS = 64


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
    moved = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            moved[k] = v
            continue
        if v.device == device:
            moved[k] = v
            continue
        # DataLoader pinning should cover the common path already, but custom
        # collators and batch_size=None loaders can still surface pageable
        # tensors. Pin them here so the transfer can stay non-blocking.
        if device.type == "cuda" and v.device.type == "cpu" and not v.is_pinned():
            v = v.pin_memory()
        moved[k] = v.to(device, non_blocking=True)
    return moved


def _get_batch_warmup_params(
    global_step: int,
    full_grad_accum: int,
    full_micro_batch: int,
    warmup_steps: int,
) -> tuple[int, int, int]:
    """Return (effective_grad_accum, effective_micro_batch, mbs_divisor)."""
    full_grad_accum = max(1, int(full_grad_accum))
    full_micro_batch = max(1, int(full_micro_batch))
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps <= 0 or global_step >= warmup_steps:
        return full_grad_accum, full_micro_batch, 1

    phase_len = max(1, warmup_steps // 3)
    if global_step < phase_len:
        divisor = 8
    elif global_step < 2 * phase_len:
        divisor = 4
    elif global_step < warmup_steps:
        divisor = 2
    else:
        return full_grad_accum, full_micro_batch, 1

    eff_ga = max(1, full_grad_accum // divisor)
    achieved = max(1, full_grad_accum // eff_ga)
    remaining = max(1, divisor // achieved)
    eff_mbs = max(1, full_micro_batch // remaining)
    mbs_divisor = max(1, full_micro_batch // eff_mbs)
    return eff_ga, eff_mbs, mbs_divisor


def _split_standard_batch(batch: dict[str, Any], n_splits: int) -> list[dict[str, Any]]:
    """Split a regular batched dict along the batch dimension."""
    if n_splits <= 1:
        return [batch]

    tensor_chunks: dict[str, list[torch.Tensor]] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() >= 1:
            tensor_chunks[key] = list(torch.tensor_split(value, n_splits, dim=0))

    if not tensor_chunks:
        return [batch]

    ref_key = next(iter(tensor_chunks))
    sub_batches: list[dict[str, Any]] = []
    for split_idx, ref_chunk in enumerate(tensor_chunks[ref_key]):
        if ref_chunk.size(0) == 0:
            continue
        sub = {}
        for key, value in batch.items():
            if key in tensor_chunks:
                sub[key] = tensor_chunks[key][split_idx]
            else:
                sub[key] = value
        sub_batches.append(sub)
    return sub_batches or [batch]


def _split_packed_batch(
    batch: dict[str, Any],
    n_splits: int,
    effective_tokens_per_split: int,
) -> list[dict[str, Any]]:
    """Split a packed flat-token batch at sequence boundaries and re-pad each split."""
    if n_splits <= 1:
        return [batch]

    effective_tokens_per_split = max(1, int(effective_tokens_per_split))
    valid_tokens = int(batch.get("num_valid_tokens", batch["input_ids"].numel()))
    cu = batch["cu_seqlens"]
    if valid_tokens <= 0:
        return [batch]

    actual_seq_count = int(
        torch.searchsorted(cu[1:], cu.new_tensor(valid_tokens), right=False).item()
    ) + 1
    cu = cu[: actual_seq_count + 1]
    n_seqs = max(0, int(cu.numel() - 1))
    if n_seqs <= 0:
        return [batch]

    sub_batches: list[dict[str, Any]] = []
    start_seq = 0
    for split_idx in range(n_splits):
        if start_seq >= n_seqs:
            break

        remaining_splits = n_splits - split_idx
        end_seq = start_seq + 1
        while end_seq < n_seqs:
            actual = int(cu[end_seq + 1].item() - cu[start_seq].item())
            seqs_left = n_seqs - (end_seq + 1)
            if actual > effective_tokens_per_split or seqs_left < (remaining_splits - 1):
                break
            end_seq += 1

        s = int(cu[start_seq].item())
        e = int(cu[end_seq].item())
        actual_tokens = e - s
        pad_tokens = effective_tokens_per_split - actual_tokens
        sub_cu = cu[start_seq : end_seq + 1] - cu[start_seq]
        actual_max_seqlen = int((sub_cu[1:] - sub_cu[:-1]).max().item())

        sub = {
            "input_ids": batch["input_ids"].new_zeros(effective_tokens_per_split),
            "labels": batch["labels"].new_full((effective_tokens_per_split,), -100),
            "cu_seqlens": sub_cu,
            "max_seqlen": max(actual_max_seqlen, pad_tokens),
            "num_valid_tokens": actual_tokens,
        }
        if actual_tokens > effective_tokens_per_split:
            raise ValueError(
                f"Packed split produced {actual_tokens} tokens but target split size is "
                f"{effective_tokens_per_split}. Increase micro_batch_seqs or reduce warmup divisor."
            )
        sub["input_ids"][:actual_tokens] = batch["input_ids"][s:e]
        sub["labels"][:actual_tokens] = batch["labels"][s:e]
        if pad_tokens > 0:
            sub["cu_seqlens"] = torch.cat(
                [
                    sub["cu_seqlens"],
                    sub["cu_seqlens"][-1:] + pad_tokens,
                ]
            )

        if "position_ids" in batch:
            sub_pos = batch["position_ids"].new_zeros(effective_tokens_per_split)
            sub_pos[:actual_tokens] = batch["position_ids"][s:e]
            if pad_tokens > 0:
                sub_pos[actual_tokens:] = torch.arange(
                    pad_tokens,
                    dtype=batch["position_ids"].dtype,
                    device=batch["position_ids"].device,
                )
            sub["position_ids"] = sub_pos

        for key, value in batch.items():
            if key in sub or key in {"input_ids", "labels", "cu_seqlens", "max_seqlen", "position_ids", "num_valid_tokens"}:
                continue
            sub[key] = deepcopy(value) if not torch.is_tensor(value) else value

        sub_batches.append(sub)
        start_seq = end_seq

    return sub_batches or [batch]


def _split_batch(
    batch: dict[str, Any],
    *,
    n_splits: int,
    effective_tokens_per_split: Optional[int] = None,
) -> list[dict[str, Any]]:
    if n_splits <= 1:
        return [batch]
    if "cu_seqlens" in batch:
        if effective_tokens_per_split is None:
            raise ValueError("effective_tokens_per_split is required for packed batch splitting.")
        return _split_packed_batch(
            batch,
            n_splits=n_splits,
            effective_tokens_per_split=effective_tokens_per_split,
        )
    return _split_standard_batch(batch, n_splits=n_splits)


def _maybe_expand_dynamo_cache_for_batch_warmup(global_batch_warmup_steps: int) -> None:
    """Increase Dynamo cache room when warmup introduces a few extra static shapes."""
    if int(global_batch_warmup_steps) <= 0:
        return
    try:
        import torch._dynamo.config as dynamo_config

        current_limit = int(getattr(dynamo_config, "cache_size_limit", 8))
        if current_limit < 16:
            dynamo_config.cache_size_limit = 16
            logger.info(
                "torch._dynamo.config.cache_size_limit raised from %d to %d for batch-size warmup.",
                current_limit,
                dynamo_config.cache_size_limit,
            )
    except Exception:
        logger.exception("Failed to raise Dynamo cache_size_limit for batch-size warmup; continuing.")


def _build_compiled_training_model(
    base_model: torch.nn.Module,
    *,
    compile_dynamic: bool,
    compile_mode: Optional[str],
    distributed: bool,
    distributed_mode: str,
    ddp_bucket_cap_mb: int,
    local_rank: int,
    device: torch.device,
) -> torch.nn.Module:
    """Compile the training model and apply DDP wrapping when requested."""
    if compile_mode is None:
        compiled_model = torch.compile(base_model, dynamic=compile_dynamic)
        logger.info("Model compiled with torch.compile(dynamic=%s)", compile_dynamic)
    else:
        compiled_model = torch.compile(base_model, dynamic=compile_dynamic, mode=compile_mode)
        logger.info(
            "Model compiled with torch.compile(dynamic=%s, mode=%s)",
            compile_dynamic,
            compile_mode,
        )

    if distributed and distributed_mode == "ddp":
        ddp_kwargs = dict(
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=ddp_bucket_cap_mb,
        )
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        compiled_model = DistributedDataParallel(compiled_model, **ddp_kwargs)
    return compiled_model


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


def _fully_shard_encoder_sublayers(
    enc: ModernBertEncoderLayer,
    *,
    mesh,
    fsdp_kwargs: dict,
    reshard_after_forward: bool,
    shard_parent: bool = True,
    sublayer: bool = False,
) -> None:
    """Shard encoder weights with configurable granularity.

    When *sublayer* is True, attn and mlp are wrapped as separate FSDP units
    for finer comm/compute overlap (better on slow interconnects like PCIe).
    When False, only the parent layer is wrapped (fewer FSDP boundaries,
    less CPU dispatch overhead — better on fast interconnects like NVLink).
    """
    if sublayer:
        fully_shard(enc.attn, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)
        if enc.mlp is not None:
            fully_shard(enc.mlp, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)
    if shard_parent:
        fully_shard(enc, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)


def _fully_shard_transformer_layer(
    layer: torch.nn.Module,
    *,
    mesh,
    fsdp_kwargs: dict,
    reshard_after_forward: bool,
    sublayer: bool = False,
) -> None:
    """Shard one logical transformer layer bottom-up for better overlap."""
    if isinstance(layer, ModernBertEncoderLayer):
        _fully_shard_encoder_sublayers(
            layer,
            mesh=mesh,
            fsdp_kwargs=fsdp_kwargs,
            reshard_after_forward=reshard_after_forward,
            shard_parent=False,
            sublayer=sublayer,
        )
    elif isinstance(layer, MHCLiteBlock):
        _fully_shard_encoder_sublayers(
            layer.layer,
            mesh=mesh,
            fsdp_kwargs=fsdp_kwargs,
            reshard_after_forward=reshard_after_forward,
            sublayer=sublayer,
        )
    elif isinstance(layer, MHCLiteSublayersLayer):
        _fully_shard_encoder_sublayers(
            layer.enc,
            mesh=mesh,
            fsdp_kwargs=fsdp_kwargs,
            reshard_after_forward=reshard_after_forward,
            shard_parent=False,
            sublayer=sublayer,
        )
        if sublayer:
            fully_shard(layer.mhc_attn, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)
            if layer.mhc_mlp is not None:
                fully_shard(layer.mhc_mlp, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)
    fully_shard(layer, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)


def _fully_shard_root_groups(
    model: PureProtModernBertMLM,
    *,
    mesh,
    fsdp_kwargs: dict,
    reshard_after_forward: bool,
) -> None:
    """Split root-only params into separate groups to avoid a serialized root tail."""
    tied_embeddings = model.decoder.weight is model.model.embeddings.tok_embeddings.weight
    if tied_embeddings:
        # Keep the tied embedding/unembedding weight in one FSDP group.
        fully_shard(
            [model.model.embeddings, model.decoder],
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            **fsdp_kwargs,
        )
    else:
        fully_shard(model.model.embeddings, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)
        fully_shard(model.decoder, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)
    fully_shard(model.model.final_norm, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)
    fully_shard(model.head, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)


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


def _nonfinite_check_tensor(tensors: list[torch.Tensor]) -> torch.Tensor | None:
    """Device-side analog of _has_nonfinite_tensors: a scalar bool tensor
    (True = some tensor is non-finite) with no host synchronization."""
    local_tensors = []
    for tensor in tensors:
        local = _local_tensor(tensor.detach())
        if local.numel() > 0:
            local_tensors.append(local)
    if not local_tensors:
        return None
    try:
        norms = torch._foreach_norm(local_tensors, float("inf"))
        return ~torch.isfinite(torch.stack(list(norms))).all()
    except (RuntimeError, TypeError):
        return ~torch.stack([torch.isfinite(t).all() for t in local_tensors]).all()


class _DeferredFiniteCheck:
    """Non-finite tripwire that avoids blocking the host on the full stream.

    `.item()` on a fresh GPU scalar issues a stream-wide cudaStreamSynchronize,
    draining every queued kernel and resetting CPU run-ahead (measured 31 ms
    per step on GH200).  Instead, submit() enqueues a device->pinned-host copy
    of the verdict and records a CUDA event; poll() waits only on that event,
    which has long completed by the time the host loops back around.  On
    non-CUDA devices it degrades to a synchronous check.
    """

    def __init__(self, device: torch.device):
        self._is_cuda = device.type == "cuda"
        self._verdict = False
        self._pending = False
        if self._is_cuda:
            self._dev_buf = torch.zeros(1, dtype=torch.bool, device=device)
            self._host_buf = torch.zeros(1, dtype=torch.bool, pin_memory=True)
            self._event = torch.cuda.Event()

    def submit(self, nonfinite: torch.Tensor) -> None:
        if self._is_cuda:
            self._dev_buf.copy_(nonfinite.reshape(1))
            self._host_buf.copy_(self._dev_buf, non_blocking=True)
            self._event.record()
        else:
            self._verdict = bool(nonfinite.item())
        self._pending = True

    def poll(self) -> bool:
        """Verdict of the most recent submit(); False if nothing pending."""
        if not self._pending:
            return False
        self._pending = False
        if self._is_cuda:
            self._event.synchronize()
            return bool(self._host_buf.item())
        return self._verdict


def _use_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    if not param.requires_grad or param.ndim < 2:
        return False
    lname = name.lower()
    return "bias" not in lname and "norm" not in lname


def _dist_barrier(local_rank: int) -> None:
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


# Sharded modes that go through the FSDP2 fully_shard() code path. "hsdp" is
# plain FSDP2 with a 2D (replicate x shard) device mesh, so it shares every
# sharded branch (checkpoint gather/reshard, gradient-sync toggling, etc.).
_FSDP_MODES = ("fsdp", "hsdp")


def _is_fsdp(distributed_mode: str) -> bool:
    """True for any FSDP2-backed mode (plain FSDP or HSDP)."""
    return distributed_mode in _FSDP_MODES


def _get_distributed_mode(cfg: PretrainingConfig, *, distributed: bool) -> str:
    mode = str(getattr(cfg, "distributed_mode", "fsdp")).strip().lower()
    if mode not in {"fsdp", "hsdp", "ddp"}:
        raise ValueError(
            f"Unsupported distributed_mode={mode!r}. Expected 'fsdp', 'hsdp', or 'ddp'."
        )
    if not distributed and mode in {"ddp", "hsdp"}:
        raise ValueError(f"distributed_mode={mode!r} requires multi_gpu=True.")
    return mode


def _resolve_hsdp_shard_size(cfg: PretrainingConfig, world_size: int) -> int:
    """Resolve the intra-replica shard group size for HSDP.

    "auto" reads LOCAL_WORLD_SIZE (GPUs per node) so each node holds one full
    replica sharded across its GPUs. Validates that it evenly divides
    world_size and leaves a replicate dim > 1 (otherwise HSDP is pointless and
    plain "fsdp" should be used instead).
    """
    raw = getattr(cfg, "hsdp_shard_size", "auto")
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        env = os.environ.get("LOCAL_WORLD_SIZE")
        shard = int(env) if env else max(torch.cuda.device_count(), 1)
    else:
        shard = int(raw)
    if shard < 1 or shard > world_size:
        raise ValueError(
            f"hsdp_shard_size={shard} out of range for world_size={world_size}."
        )
    if world_size % shard != 0:
        raise ValueError(
            f"hsdp_shard_size={shard} must evenly divide world_size={world_size}."
        )
    if shard == world_size:
        raise ValueError(
            f"hsdp_shard_size={shard} equals world_size={world_size}; the replicate "
            "dim would be 1. Use distributed_mode='fsdp' instead of 'hsdp'."
        )
    return shard


def _checkpoint_distributed_mode(*, distributed: bool, distributed_mode: str) -> str:
    if not distributed:
        return "single"
    # HSDP and FSDP both save/restore full (gathered) state dicts, so their
    # checkpoints are interchangeable — tag both as "fsdp" for resume compat.
    return "fsdp" if _is_fsdp(distributed_mode) else distributed_mode


def _bootstrap_single_process_env() -> None:
    """Default the torchrun rendezvous env vars for a self-contained 1-process
    group, so a plain `nanoplm pretrain` invocation (no torchrun) can
    initialize NCCL with the env:// method."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])


def _ddp_sync_context(model: torch.nn.Module, *, distributed_mode: str, sync: bool):
    if distributed_mode == "ddp" and hasattr(model, "no_sync") and not sync:
        return model.no_sync()
    return nullcontext()


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

def _recycling_flops_components(config, seq_len: int):
    """Decompose training FLOPs/token into recycling-scalable + fixed parts.

    Returns ``(stack_flops, tail_flops, P, R, C, n_layers)``. ``stack_flops`` is
    the per-forward-pass cost of the full transformer stack; under recycling the
    recurrent block (R layers) repeats that work ``loop_count`` times while
    prelude (P) and coda (C) run once. ``tail_flops`` (lm-head + prediction head)
    fires once and does NOT scale with the loop count. Feed these to
    ``_recycling_flops_per_token`` to get FLOPs/token at a given loop count.

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
    # With DiffV2: 2h query heads × h KV heads (GQA ratio 2), so QK and AV
    # matmuls use 2h query dimension but shared KV.  Effective FLOPs double
    # for the QK and AV matmuls since there are 2× more query heads.
    use_diff_v2 = getattr(config, "use_diff_attn_v2", False)
    q_heads_mult = 2 if use_diff_v2 else 1
    attn_flops = 0
    for layer_type in config.layer_types:
        if layer_type == "full_attention":
            effective_seq = seq_len
        else:
            effective_seq = min(2 * sliding_window, seq_len)
        attn_flops += 4 * h * effective_seq * q_heads_mult

    # -- Attention projection FLOPs (same for all layers) --
    # Wqkv: h -> (num_q_heads + 2*num_kv_heads)*head_dim
    # Wo:   h -> h (2h²)
    # With DiffV2: num_q_heads = 2*n_heads, else num_q_heads = n_heads
    # With GQA: num_kv_heads < n_heads
    num_kv_heads = getattr(config, "num_kv_heads", n_heads) or n_heads
    num_q_heads = 2 * n_heads if use_diff_v2 else n_heads
    qkv_out_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
    attn_proj_flops = n_layers * (2 * h * qkv_out_dim + 2 * h * h)  # Wqkv + Wo

    # -- MLP FLOPs --
    # swiglu/glu: Wi is h -> 2*ff, Wo is ff -> h => 2*h*2*ff + 2*ff*h = 6*h*ff
    # srelu:      Wi is h -> ff,   Wo is ff -> h => 2*h*ff   + 2*ff*h = 4*h*ff
    if config.mlp_activation == "srelu":
        dense_mlp_flops = 4 * h * ff
    else:
        dense_mlp_flops = 6 * h * ff

    no_mlp_first = getattr(config, "no_mlp_on_first_layer", False)
    use_moe = getattr(config, "use_moe", False)
    if use_moe:
        # MoE active FLOPs per token per MoE layer:
        #  - Router gate: h -> num_experts (2 * h * E)
        #  - Routed experts: top_k experts, each SwiGLU with ff intermediate
        #      Wi: h -> 2*ff, Wo: ff -> h => top_k * 6 * h * ff
        #  - Shared expert: same as one dense MLP => 6 * h * ff (or 4 for srelu)
        moe_top_k = config.moe_top_k
        moe_E = config.moe_num_experts
        router_flops = 2 * h * moe_E
        routed_flops = moe_top_k * dense_mlp_flops
        shared_flops = dense_mlp_flops
        moe_mlp_flops = router_flops + routed_flops + shared_flops

        # Dense layers in MoE mode use the full active intermediate size.
        dense_ff = config.moe_dense_intermediate_size
        if config.mlp_activation == "srelu":
            moe_dense_mlp_flops = 4 * h * dense_ff
        else:
            moe_dense_mlp_flops = 6 * h * dense_ff

        moe_flags = config.moe_layer_flags
        mlp_flops = sum(
            0 if (no_mlp_first and i == 0) else
            (moe_mlp_flops if moe_flags[i] else moe_dense_mlp_flops)
            for i in range(n_layers)
        )
    else:
        n_mlp_layers = (n_layers - 1) if no_mlp_first else n_layers
        mlp_flops = n_mlp_layers * dense_mlp_flops

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
        if "b" in canon_set:                        # on QKV output
            canon_flops += n_layers * 2 * K * qkv_out_dim
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

    # Huginn-style recycling: prelude (P, runs once) + recurrent (R, looped) + coda
    # (C, runs once). head_flops + pred_head_flops fire once after the stack and do
    # NOT scale with the loop count -> they go into tail_flops. The loop count is
    # supplied by the caller: the target recycling_loop_count for the startup
    # estimate, or the *actual* sampled r per step for live MFU (the two differ
    # while the recycling curriculum ramps the mean loop count up from 1).
    stack_flops = attn_proj_flops + attn_flops + mlp_flops + canon_flops + mhc_flops
    tail_flops = head_flops + pred_head_flops
    P = max(0, int(getattr(config, "recycling_num_prelude_layers", 0)))
    C = max(0, int(getattr(config, "recycling_num_coda_layers", 0)))
    R = n_layers - P - C
    return stack_flops, tail_flops, P, R, C, n_layers


def _recycling_flops_per_token(components, loop_count: int, bptt: int | None = None) -> float:
    """Training FLOPs/token (forward + truncated-BPTT backward) at a loop count.

    ``components`` is the tuple from ``_recycling_flops_components``. The recurrent
    block (R layers) runs ``loop_count`` times in the forward pass; prelude/coda run
    once; the tail (lm-head + prediction head) fires once. Collapses to the dense
    n_layers cost when ``loop_count == 1`` (i.e. recycling disabled).

    Backward cost reflects truncated BPTT: ``_drive_recurrence`` runs the first
    ``loop_count - K`` recurrent iters under ``torch.no_grad()`` and retains a graph
    only for the last ``K = bptt`` iters (prelude/coda always differentiate, since
    ``e`` is not detached). So the hardware backward is ~2× the forward of the
    *differentiated* subgraph (P + K·R + C + tail), NOT 2× the full forward. Charging
    2× the full forward (the old ``3× forward``) overcounts backward FLOPs by
    ``2·(loop_count − K)·R`` layer-equivalents and inflates MFU for r > K.

    ``bptt=None`` (or ``bptt >= loop_count``) means full BPTT and recovers ``3× forward``.
    """
    stack_flops, tail_flops, P, R, C, n_layers = components
    loop_count = max(1, int(loop_count))
    K = loop_count if bptt is None else max(1, min(int(bptt), loop_count))

    def _forward(loops: int) -> float:
        effective_layer_count = P + loops * R + C  # == n_layers when loops == 1
        return (effective_layer_count / max(1, n_layers)) * stack_flops + tail_flops

    forward_flops = _forward(loop_count)   # all loop_count iters run forward
    backward_flops = 2.0 * _forward(K)     # backward retained only for last K iters
    return forward_flops + backward_flops


def _estimate_model_flops_per_token(config, seq_len: int, bptt: int | None = None) -> int:
    """Training FLOPs/token at the TARGET recycling loop count (startup estimate)."""
    components = _recycling_flops_components(config, seq_len)
    loop_count = max(1, int(getattr(config, "recycling_loop_count", 1)))
    return _recycling_flops_per_token(components, loop_count, bptt=bptt)


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------

def _create_optimizer(model, cfg, distributed_mesh=None):
    name = str(cfg.optimizer).lower()
    if name in {"muon", "normuon", "muonh", "aurora"}:
        return build_muon_optimizer(model, cfg, distributed_mesh=distributed_mesh)

    from nanoplm.pretraining.optim import _classify_noble_param

    decay, no_decay = [], []
    noble_wup, noble_M, noble_freq, noble_phase = [], [], [], []
    for p_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        noble_kind = _classify_noble_param(p_name)
        if noble_kind is not None:
            {"wup": noble_wup, "M": noble_M,
             "freq": noble_freq, "phase": noble_phase}[noble_kind].append(param)
        else:
            (decay if _use_weight_decay(p_name, param) else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": float(cfg.adam_weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # NOBLE param groups with elevated LRs (paper §3.3)
    noble_total = len(noble_wup) + len(noble_M) + len(noble_freq) + len(noble_phase)
    if noble_total > 0:
        raw = model.module if hasattr(model, "module") else model
        hidden_size = getattr(getattr(raw, "config", None), "hidden_size", None)
        noble_rank = getattr(getattr(raw, "config", None), "noble_rank", None)
        if hidden_size is None or noble_rank is None:
            raise ValueError("Cannot determine hidden_size/noble_rank for NOBLE LR scaling.")
        base_lr = cfg.noble_base_lr
        ratio = hidden_size / noble_rank
        wd = float(cfg.adam_weight_decay)
        betas = (float(cfg.adam_beta1), float(cfg.adam_beta2))
        eps = float(cfg.adam_epsilon)
        for params, lr in [
            (noble_wup, base_lr * (ratio ** (2 * cfg.noble_lr_gamma))),
            (noble_M, base_lr * (ratio ** cfg.noble_lr_gamma_M)),
            (noble_freq, base_lr * cfg.noble_freq_lr_mult),
            (noble_phase, base_lr * cfg.noble_phase_lr_mult),
        ]:
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": wd,
                               "betas": betas, "eps": eps})

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
    raise ValueError(f"Invalid optimizer: {cfg.optimizer}. Supported: [adamw, stable_adamw, muon, normuon, muonh, aurora]")


def _mlm_extra_excluded_token_ids(tokenizer, cfg):
    if bool(getattr(cfg, "mlm_standard_amino_acids_only", True)):
        return getattr(tokenizer, "NON_STANDARD_AA_TOKEN_IDS", None)
    return None


class _SchedulerGroup:
    """Wraps two LambdaLR schedulers (muon + adamw) for MuonAdamWGroup.

    Exposes the same interface as a single LambdaLR so the training loop
    can call .step() and .state_dict() / .load_state_dict() transparently.
    """

    def __init__(self, muon_scheduler, adamw_scheduler):
        self.muon_sched = muon_scheduler
        self.adamw_sched = adamw_scheduler

    @staticmethod
    def _iter_scheds(sched):
        return sched if isinstance(sched, list) else [sched]

    @staticmethod
    def _state_dict(sched):
        if isinstance(sched, list):
            return [s.state_dict() for s in sched]
        return sched.state_dict()

    @staticmethod
    def _load_state_dict(sched, state_dict):
        if isinstance(sched, list):
            if not isinstance(state_dict, list) or len(state_dict) != len(sched):
                raise ValueError("Scheduler state list does not match current scheduler layout.")
            for s, sd in zip(sched, state_dict):
                s.load_state_dict(sd)
        else:
            sched.load_state_dict(state_dict)

    def step(self):
        for sched in self._iter_scheds(self.muon_sched):
            sched.step()
        for sched in self._iter_scheds(self.adamw_sched):
            sched.step()

    def state_dict(self):
        return {
            "muon": self._state_dict(self.muon_sched),
            "adamw": self._state_dict(self.adamw_sched),
        }

    def load_state_dict(self, state_dict):
        if "muon" in state_dict and "adamw" in state_dict:
            self._load_state_dict(self.muon_sched, state_dict["muon"])
            self._load_state_dict(self.adamw_sched, state_dict["adamw"])
        else:
            # Legacy single-scheduler checkpoint — skip gracefully.
            logger.warning(
                "Scheduler checkpoint uses legacy single-scheduler format. "
                "Skipping scheduler state restore."
            )


def _create_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    learning_rate: float,
    lr_decay_to_fraction: float,
    lr_schedule: str = "Linear",
    last_epoch: int = -1,
    *,
    stable_steps: Optional[int] = None,
    decay_steps: Optional[int] = None,
    decay_shape: str = "1-sqrt",
):
    """Warmup + (wsd | warmup_stable | cosine | linear | decay) scheduler.

    wsd:           warmup -> constant over [warmup, stable_steps) -> decay over
                   [stable_steps, stable_steps+decay_steps) to lr_decay_to_fraction.
    warmup_stable: warmup -> constant for the rest (no decay).
    decay:         no warmup; decay 1.0 -> lr_decay_to_fraction over total_steps
                   (resume anneal from the checkpoint's current LR).
    cosine/linear: unchanged single-phase warmup -> decay over total_steps.

    For MuonAdamWGroup, returns a _SchedulerGroup wrapping two LambdaLR
    schedulers (one per inner optimizer). For plain optimizers, returns
    a single LambdaLR.

    Args:
        stable_steps: (wsd only) absolute step at which the constant phase ends
            and decay begins. Required for ``wsd``.
        decay_steps: (wsd only) number of steps over which the decay phase runs,
            starting at ``stable_steps``. Required for ``wsd``.
        decay_shape: Shape of the decay phase, one of {"1-sqrt", "linear"}
            (case-insensitive). Used by ``wsd`` and ``decay``.
        last_epoch: If >= 0 the scheduler is positioned at that step
            (useful when reconstructing the schedule on resume).
            ``initial_lr`` is injected into param groups automatically.

    Note:
        For ``wsd`` the floor (``lr_decay_to_fraction``) is the value AT step
        ``stable_steps + decay_steps`` (== total_steps), so the last executed
        step (total_steps - 1) sits just above the floor. This is by design and
        matches the schedule spec.
    """
    sched = lr_schedule.lower()

    if sched == "wsd" and (stable_steps is None or decay_steps is None):
        raise ValueError(
            "lr_schedule='wsd' requires stable_steps and decay_steps to be set"
        )

    shape = (decay_shape or "1-sqrt").lower()
    if shape not in {"1-sqrt", "linear"}:
        raise ValueError(
            f"decay_shape must be '1-sqrt' or 'linear', got {decay_shape!r}"
        )

    def _decay_factor(progress: float) -> float:
        progress = min(1.0, max(0.0, progress))
        shaped = math.sqrt(progress) if shape == "1-sqrt" else progress
        return lr_decay_to_fraction + (1.0 - lr_decay_to_fraction) * (1.0 - shaped)

    def lr_lambda(step: int) -> float:
        if sched == "wsd":
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step < stable_steps:
                return 1.0
            return _decay_factor((step - stable_steps) / max(1, decay_steps))
        if sched == "warmup_stable":
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0
        if sched == "decay":
            return _decay_factor(step / max(1, total_steps))
        # cosine / linear (unchanged single-phase behaviour)
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        decay_window = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_window)
        if sched == "cosine":
            return lr_decay_to_fraction + 0.5 * (1.0 - lr_decay_to_fraction) * (1.0 + math.cos(math.pi * progress))
        return max(lr_decay_to_fraction, 1.0 - (1.0 - lr_decay_to_fraction) * progress)

    def _make_scheduler(opt, last_ep):
        if last_ep >= 0:
            for pg in opt.param_groups:
                pg.setdefault("initial_lr", pg["lr"])
        return LambdaLR(opt, lr_lambda, last_epoch=last_ep)

    def _make_scheduler_any(opt, last_ep):
        if isinstance(opt, OptimizerGroup):
            return [_make_scheduler(inner, last_ep) for inner in opt.optimizers]
        return _make_scheduler(opt, last_ep)

    if isinstance(optimizer, MuonAdamWGroup):
        return _SchedulerGroup(
            _make_scheduler_any(optimizer.muon, last_epoch),
            _make_scheduler_any(optimizer.adamw, last_epoch),
        )

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

@torch.compiler.disable
@torch.inference_mode()
def _evaluate(
    model, eval_loader, device, distributed, amp_dtype,
    recycling_loop_count: int | None = None,
    recycling_bptt: int | None = None,
) -> float:
    # Eval intentionally stays eager. The eval loader uses the padded path
    # (no cu_seqlens/max_seqlen), which differs from packed training and can
    # otherwise trigger Dynamo recompiles of helper submodules.
    model.eval()
    total_loss, total_samples = 0.0, 0

    for batch in eval_loader:
        batch = _move_batch_to_device(batch, device)
        ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with ctx:
            fwd_kwargs = dict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            if recycling_loop_count is not None:
                fwd_kwargs["recycling_loop_count"] = recycling_loop_count
                fwd_kwargs["recycling_bptt"] = recycling_bptt
            out = model(**fwd_kwargs)
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

def _legacy_shard_opt_state_for_fsdp(opt_sd, optimizer):
    """Backward-compat fallback: manually shard a full optimizer state_dict for FSDP2.

    Used only when loading checkpoints saved before the switch to
    ``get_optimizer_state_dict`` / ``set_optimizer_state_dict``.
    """

    def _shard_sub(sub_sd, sub_optimizer):
        rank = dist.get_rank()
        world = dist.get_world_size()
        all_params: list[torch.nn.Parameter] = []
        for pg in sub_optimizer.param_groups:
            all_params.extend(pg["params"])
        param_by_idx = {idx: p for idx, p in enumerate(all_params)}

        for param_idx, entry in sub_sd.get("state", {}).items():
            local_param = param_by_idx.get(param_idx)
            if local_param is None:
                continue
            local_shape = (
                local_param._local_tensor.shape
                if isinstance(local_param, DTensor)
                else local_param.shape
            )
            for key, val in entry.items():
                if not isinstance(val, torch.Tensor):
                    continue
                if val.shape == local_shape:
                    continue
                if val.ndim >= 1 and val.shape[0] == local_shape[0] * world:
                    chunk_size = val.shape[0] // world
                    entry[key] = val.narrow(0, rank * chunk_size, chunk_size).contiguous()
                elif val.ndim >= 1 and val.shape[0] != local_shape[0]:
                    chunks = torch.tensor_split(val, world, dim=0)
                    if chunks[rank].shape[0] == local_shape[0]:
                        entry[key] = chunks[rank].contiguous()
                    else:
                        logger.warning(
                            "Optimizer state param %s key '%s': shape %s cannot "
                            "be sharded to match local %s",
                            param_idx, key, val.shape, local_shape,
                        )

    if isinstance(optimizer, MuonAdamWGroup) and "muon" in opt_sd:
        _shard_sub(opt_sd["muon"], optimizer.muon)
        _shard_sub(opt_sd["adamw"], optimizer.adamw)
    else:
        _shard_sub(opt_sd, optimizer)


def _checkpoint_dir_name(global_step: int, phase: str = "periodic") -> str:
    if phase == "stable-end":
        return "checkpoint-stable-end"
    if phase == "final":
        return "checkpoint-final"
    return f"checkpoint-{global_step}"


def _save_checkpoint(
    model, optimizer, scheduler, global_step, epoch,
    output_dir, logging_steps, eval_steps, save_steps,
    distributed=False, is_main=True, distributed_mode="fsdp",
    model_config=None, manifest=None,
    pretrain_config=None, total_steps=None, warmup_steps=None,
    dataset_fingerprint=None,
    epoch_data_step: Optional[int] = None,
    epoch_sub_batch_step: Optional[int] = None,
    batch_split_divisor: Optional[int] = None,
    phase: str = "periodic",
    tokens_seen: Optional[int] = None,
    achieved_global_batch_tokens: Optional[int] = None,
    stable_steps: Optional[int] = None,
    decay_steps: Optional[int] = None,
) -> None:
    ckpt = Path(output_dir) / _checkpoint_dir_name(global_step, phase)

    if distributed and _is_fsdp(distributed_mode):
        _fsdp_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_sd = get_model_state_dict(model, options=_fsdp_opts)
        if isinstance(optimizer, MuonAdamWGroup):
            opt_sd = {
                "muon": get_optimizer_state_dict(model, optimizer.muon, options=_fsdp_opts),
                "adamw": get_optimizer_state_dict(model, optimizer.adamw, options=_fsdp_opts),
            }
        else:
            opt_sd = get_optimizer_state_dict(model, optimizer, options=_fsdp_opts)
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
        distributed_mode=_checkpoint_distributed_mode(
            distributed=distributed,
            distributed_mode=distributed_mode,
        ),
    )
    if total_steps is not None:
        training_state["total_steps"] = total_steps
    if warmup_steps is not None:
        training_state["warmup_steps"] = warmup_steps
    if dataset_fingerprint is not None:
        training_state["dataset_fingerprint"] = dataset_fingerprint
    if epoch_data_step is not None:
        training_state["epoch_data_step"] = int(epoch_data_step)
    if epoch_sub_batch_step is not None:
        training_state["epoch_sub_batch_step"] = int(epoch_sub_batch_step)
    if batch_split_divisor is not None:
        training_state["batch_split_divisor"] = int(batch_split_divisor)
    # Schedule descriptor: lets resume reconstruct/validate the LR curve and
    # lets eval/resume target a role (stable-end/final) instead of a step number.
    training_state["phase"] = phase
    if pretrain_config is not None:
        training_state["lr_schedule"] = pretrain_config.lr_schedule
        training_state["decay_shape"] = pretrain_config.decay_shape
        training_state["lr_decay_to_fraction"] = pretrain_config.lr_decay_to_fraction
    if tokens_seen is not None:
        training_state["tokens_seen"] = int(tokens_seen)
    if achieved_global_batch_tokens is not None:
        training_state["achieved_global_batch_tokens"] = int(achieved_global_batch_tokens)
    if stable_steps is not None:
        training_state["stable_steps"] = int(stable_steps)
    if decay_steps is not None:
        training_state["decay_steps"] = int(decay_steps)
    (ckpt / "training_state.json").write_text(
        json.dumps(training_state, indent=2),
        encoding="utf-8",
    )

    # Save model architecture config and data manifest so checkpoints are
    # self-contained (inference can reconstruct the model without the
    # original pretrain.yaml or dataset directory).
    if model_config is not None:
        cfg_dict = asdict(model_config) if hasattr(model_config, '__dataclass_fields__') else dict(model_config)
        cfg_dict = json.loads(json.dumps(cfg_dict))  # tuples → lists; avoids !!python/tuple tags
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
    distributed=False, load_scheduler=True, distributed_mode="fsdp",
) -> Tuple[int, int]:
    """Restore model, optimizer, (optionally) scheduler, and RNG states.

    Args:
        load_scheduler: When *False* the scheduler state dict is **not**
            loaded.  This is used by the continue/decay resume path which
            rebuilds the scheduler from the saved schedule descriptor
            instead of relying on the pickled ``lr_lambda``.
    """
    ckp = Path(checkpoint_dir)
    state_path = ckp / "training_state.json"
    training_state = None
    if state_path.exists():
        training_state = json.loads(state_path.read_text(encoding="utf-8"))

    current_mode = _checkpoint_distributed_mode(
        distributed=distributed,
        distributed_mode=distributed_mode,
    )
    checkpoint_mode = (
        None if training_state is None else training_state.get("distributed_mode")
    )
    mode_mismatch = checkpoint_mode is not None and checkpoint_mode != current_mode

    model_sd = torch.load(ckp / "pytorch_model.bin", map_location=device, weights_only=True)
    if distributed and _is_fsdp(distributed_mode):
        set_model_state_dict(model, model_sd, options=StateDictOptions(full_state_dict=True))
    else:
        model.load_state_dict(model_sd)

    opt_path = ckp / "optimizer.pt"
    if mode_mismatch and opt_path.exists():
        raise RuntimeError(
            "Checkpoint distributed_mode mismatch: "
            f"checkpoint={checkpoint_mode}, current={current_mode}. "
            "Model weights were loaded, but optimizer-state resume across FSDP/DDP/single "
            "modes is not supported. Restart with a fresh optimizer state or resume "
            "with the matching distributed_mode."
        )

    opt_sd = torch.load(opt_path, map_location=device, weights_only=True)

    # Use FSDP2-aware optimizer state restore so that both DTensor and plain-
    # tensor optimizer buffers are correctly gathered/sharded.  Falls back to
    # legacy manual sharding for checkpoints saved before this change.
    if distributed and _is_fsdp(distributed_mode) and dist.is_initialized():
        _fsdp_opts = StateDictOptions(full_state_dict=True)
        try:
            if isinstance(optimizer, MuonAdamWGroup) and "muon" in opt_sd:
                set_optimizer_state_dict(
                    model, optimizer.muon,
                    optim_state_dict=opt_sd["muon"], options=_fsdp_opts,
                )
                set_optimizer_state_dict(
                    model, optimizer.adamw,
                    optim_state_dict=opt_sd["adamw"], options=_fsdp_opts,
                )
            else:
                set_optimizer_state_dict(
                    model, optimizer,
                    optim_state_dict=opt_sd, options=_fsdp_opts,
                )
        except Exception as exc:
            logger.warning(
                "FSDP2-aware optimizer state restore failed (%s). "
                "Falling back to legacy manual sharding.", exc,
            )
            _legacy_shard_opt_state_for_fsdp(opt_sd, optimizer)
            try:
                optimizer.load_state_dict(opt_sd)
            except ValueError as exc2:
                if "parameter group" in str(exc2):
                    logger.warning(
                        "Optimizer param-group layout changed since checkpoint was saved "
                        "(%s). Skipping optimizer state restore — optimizer will "
                        "restart with fresh momentum/variance buffers.", exc2,
                    )
                else:
                    raise
    else:
        try:
            optimizer.load_state_dict(opt_sd)
        except ValueError as exc:
            if "parameter group" in str(exc):
                logger.warning(
                    "Optimizer param-group layout changed since checkpoint was saved "
                    "(%s). Skipping optimizer state restore — optimizer will "
                    "restart with fresh momentum/variance buffers.", exc,
                )
            else:
                raise

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

    if training_state is not None:
        state = training_state
        return int(state.get("global_step", 0)), int(state.get("epoch", 0))
    return 0, 0


def _iter_param_groups(optimizer):
    """Yield param groups across plain optimizers and MuonAdamWGroup."""
    if isinstance(optimizer, MuonAdamWGroup):
        yield from optimizer.muon.param_groups
        yield from optimizer.adamw.param_groups
    else:
        yield from optimizer.param_groups


def _read_training_state(checkpoint_dir: str) -> dict:
    """Load ``training_state.json`` from a checkpoint directory.

    Returns an empty dict if the file is missing (pre-upgrade checkpoint).
    Raises ValueError (with the offending path) if the file exists but is not
    valid JSON, e.g. a checkpoint truncated by a crash mid-write.
    """
    state_path = Path(checkpoint_dir) / "training_state.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse training state at {state_path}: {exc}. "
            "The checkpoint may be corrupt (e.g. truncated by a crash mid-write)."
        ) from exc


def _rebuild_scheduler_for_resume(
    *,
    optimizer,
    resume_config: "ResumeConfig",
    pretrain_config: "PretrainingConfig",
    checkpoint_dir: str,
    decay_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    start_step: Optional[int] = None,
    stable_steps: Optional[int] = None,
    decay_shape: Optional[str] = None,
    is_main: bool = True,
):
    """Build a fresh LR scheduler for a resumed run (spec §7).

    Two intent-named modes (validated at YAML load by
    ``_validate_resume_config_modes``):

    ``continue``
        Finish an interrupted run to its original budget with the exact
        original schedule. The original schedule descriptor is read back from
        the checkpoint's ``training_state.json`` and the schedule is
        fast-forwarded via ``last_epoch = global_step`` so the LR curve picks
        up exactly where it left off.

    ``decay``
        Anneal a saved ``checkpoint-stable-end`` trunk. The peak LR is the
        checkpoint's current (stable) LR already loaded into the optimizer; a
        ``lr_schedule="decay"`` scheduler with no warmup runs ``decay_steps``
        from that peak down to ``lr_decay_to_fraction``. With ``last_epoch=-1``
        ``_create_scheduler`` captures each group's loaded ``lr`` as its
        ``initial_lr``, so decay anneals from the checkpoint LR.
    """
    mode = (resume_config.mode or "continue").lower()

    if mode == "decay":
        floor = resume_config.lr_decay_to_fraction
        if floor is None:
            raise ValueError("resume decay requires lr_decay_to_fraction")
        shape = (decay_shape or resume_config.decay_shape or "1-sqrt")
        if decay_steps is None or decay_steps <= 0:
            raise ValueError(
                f"resume decay needs a positive decay_steps, got {decay_steps}"
            )
        if is_main:
            peaks = [f"{pg['lr']:.2e}" for pg in _iter_param_groups(optimizer)]
            logger.info(
                f"Resume schedule built [decay]: decay_steps={decay_steps}, "
                f"floor={floor}, shape={shape}, peaks={peaks}"
            )
        # Peak = checkpoint's current (stable) LR, already loaded into the
        # optimizer; last_epoch=-1 captures it as initial_lr.
        return _create_scheduler(
            optimizer,
            warmup_steps=0,
            total_steps=int(decay_steps),
            learning_rate=0.0,
            lr_decay_to_fraction=float(floor),
            lr_schedule="decay",
            last_epoch=-1,
            decay_shape=shape,
        )

    # mode == "continue": reconstruct the ORIGINAL schedule and fast-forward.
    sd = _read_training_state(checkpoint_dir)
    sched_name = sd.get("lr_schedule", pretrain_config.lr_schedule)
    eff_total = total_steps if total_steps is not None else sd.get("total_steps")
    if eff_total is None:
        raise ValueError(
            "resume continue requires total_steps (none provided and the "
            "checkpoint's training_state.json has no 'total_steps')"
        )
    eff_last_epoch = (
        start_step if start_step is not None else int(sd.get("global_step", 0))
    )
    eff_stable = stable_steps if stable_steps is not None else sd.get("stable_steps")
    eff_decay = decay_steps if decay_steps is not None else sd.get("decay_steps")
    if is_main:
        peaks = [f"{pg['lr']:.2e}" for pg in _iter_param_groups(optimizer)]
        logger.info(
            f"Resume schedule built [continue]: start_step={eff_last_epoch}, "
            f"total_steps={eff_total}, schedule={sched_name}, peaks={peaks}"
        )
    return _create_scheduler(
        optimizer,
        warmup_steps=int(sd.get("warmup_steps", 0)),
        total_steps=int(eff_total),
        learning_rate=0.0,
        lr_decay_to_fraction=float(sd.get("lr_decay_to_fraction", 0.1)),
        lr_schedule=sched_name,
        last_epoch=int(eff_last_epoch),
        stable_steps=int(eff_stable) if eff_stable is not None else None,
        decay_steps=int(eff_decay) if eff_decay is not None else None,
        decay_shape=sd.get("decay_shape", "1-sqrt"),
    )


def _configure_fp8_cpu_activation_offload(
    *,
    model: PureProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    device: torch.device,
) -> None:
    if not bool(getattr(pretrain_config, "fp8_cpu_activation_offload", False)):
        cpu_offload.disable(model)
        return

    if not bool(getattr(pretrain_config, "fp8", False)):
        raise ValueError("fp8_cpu_activation_offload=true requires fp8=true.")
    if device.type != "cuda":
        raise ValueError("fp8_cpu_activation_offload=true requires CUDA.")

    model_config = getattr(model, "model_config", None)
    if bool(getattr(model_config, "activation_checkpointing", False)):
        raise ValueError(
            "fp8_cpu_activation_offload is not supported with activation_checkpointing=true."
        )
    if int(getattr(model_config, "recycling_loop_count", 1)) != 1:
        raise ValueError(
            "fp8_cpu_activation_offload is not supported with recycling_loop_count > 1."
        )

    targets = list(iter_te_cpu_offload_target_wrappers(model))
    if not targets:
        raise ValueError(
            "fp8_cpu_activation_offload=true found no eligible TE wrappers; "
            "set fp8_target to include 'attn_wqkv' and/or 'mlp'."
        )

    retain = bool(getattr(pretrain_config, "fp8_cpu_activation_offload_retain_buffers", False))
    n = cpu_offload.install(model, targets, retain_pinned_buffers=retain)
    logger.info(
        "FP8 CPU activation offload enabled: targets=%d (inputmat+ln_out, 1-step reload "
        "prefetch), retain_pinned_cpu_buffers=%s (torch.compile remains enabled)",
        n,
        retain,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pure_pretraining(
    model: PureProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    manifest: DataManifest,
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
    validate_manifest_for_pipeline(manifest=manifest, expected_mode="pretrain")
    if manifest.max_seq_len <= 0:
        raise ValueError(f"Invalid manifest max_seq_len: {manifest.max_seq_len}")

    # Capture config/manifest for checkpoint serialization so every saved
    # checkpoint is self-contained (no external pretrain.yaml needed).
    _model_config = getattr(model, "model_config", None)
    _manifest = manifest

    from nanoplm.config.recycling_compatibility import validate_recycling_compatibility
    validate_recycling_compatibility(model.model_config, pretrain_config)

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
    _dataset_fingerprint = train_ds.fingerprint()
    logger.info(f"Train dataset fingerprint: {_dataset_fingerprint}")

    use_packing = bool(pretrain_config.use_packing)
    use_static_inp_size = bool(pretrain_config.use_static_inp_size and use_packing)
    mlm_standard_amino_acids_only = bool(
        getattr(pretrain_config, "mlm_standard_amino_acids_only", True)
    )
    mlm_extra_excluded_token_ids = _mlm_extra_excluded_token_ids(
        tokenizer, pretrain_config
    )
    logger.info(
        "MLM token policy: %s",
        "standard amino acids only"
        if mlm_standard_amino_acids_only
        else "legacy all non-special tokens",
    )
    mlm_token_policy = (
        "standard amino acids only"
        if mlm_standard_amino_acids_only
        else "legacy all non-special tokens"
    )
    if getattr(pretrain_config, "mlm_masking_strategy", "token") == "span":
        logger.info(
            "MLM masking strategy: span (mean_length=%s, max_length=%s, token_policy=%s)",
            pretrain_config.mlm_span_mean_length,
            pretrain_config.mlm_span_max_length,
            mlm_token_policy,
        )
    else:
        logger.info("MLM masking strategy: token")

    # MoE guard rails.
    if getattr(model.config, "use_moe", False):
        if not use_packing:
            raise ValueError("MoE requires use_packing=True")
        if not use_static_inp_size:
            raise ValueError("MoE requires use_static_inp_size=True")

    # ---- Batch sizing ----
    create_dirs(pretrain_config.ckp_dir)
    effective_world_size = _resolve_world_size(pretrain_config)

    inferred_grad_accum_steps = pretrain_config.inferred_grad_accum_steps
    global_batch_seqs = pretrain_config.global_batch_seqs
    achieved_global_batch_tokens = pretrain_config.achieved_global_batch_tokens
    global_batch_warmup_steps = max(
        0,
        int(getattr(pretrain_config, "global_batch_warmup_steps", 0)),
    )
    if any(v is None for v in (inferred_grad_accum_steps, global_batch_seqs, achieved_global_batch_tokens)):
        raise ValueError("Batch setup missing on PretrainingConfig. Run through nanoplm CLI.")

    world_tokens_per_micro_step = achieved_global_batch_tokens // max(1, inferred_grad_accum_steps)
    tokens_per_micro = pretrain_config.micro_batch_seqs * manifest.max_seq_len
    logger.info(
        f"Batch setup: target_global_batch_tokens={pretrain_config.global_batch_tokens:,} tokens, "
        f"micro_step_tokens={world_tokens_per_micro_step:,}, "
        f"grad_accum_steps={inferred_grad_accum_steps}, "
        f"effective_global_batch_tokens={achieved_global_batch_tokens:,} tokens"
    )

    # ---- Run naming / output dir ----
    (
        _run_name,
        wandb_run_name,
        output_dir,
        _,  # 4th slot is always None: horizon is token-budget driven
        logging_steps,
        eval_steps,
        save_steps,
        _resume_step,
    ) = prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=manifest.train_sequences,
        global_batch_seqs=global_batch_seqs,
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
    distributed_mode = _get_distributed_mode(pretrain_config, distributed=distributed)
    ddp_bucket_cap_mb = int(getattr(pretrain_config, "ddp_bucket_cap_mb", 25))

    # Single-GPU bf16 runs are promoted to a 1-rank FSDP2 run so precision
    # semantics match the multi-GPU path: FSDP's MixedPrecisionPolicy casts
    # params to bf16 for forward (embeddings/norms/residual stream all bf16,
    # no autocast) while the optimizer keeps fp32 masters. The legacy
    # fp32-params + torch.autocast path leaves the residual stream in fp32,
    # which diverges numerically from production and blocks bf16-gated
    # kernels (e.g. the CuTeDSL canon backend). Set NANOPLM_SINGLE_GPU_AMP=1
    # to keep the legacy autocast behavior.
    if (
        not distributed
        and distributed_mode == "fsdp"
        and pretrain_config.bf16
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and os.environ.get("NANOPLM_SINGLE_GPU_AMP", "0") != "1"
    ):
        _bootstrap_single_process_env()
        distributed = True
        logger.info(
            "Single-GPU bf16: promoting to a 1-rank FSDP run so precision "
            "matches the multi-GPU path (bf16 forward params, fp32 masters, "
            "no autocast). Set NANOPLM_SINGLE_GPU_AMP=1 for the legacy "
            "fp32+autocast behavior."
        )

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
            standard_amino_acids_only=mlm_standard_amino_acids_only,
            extra_excluded_token_ids=mlm_extra_excluded_token_ids,
            **mlm_masking_kwargs_from_config(pretrain_config),
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
            standard_amino_acids_only=mlm_standard_amino_acids_only,
            extra_excluded_token_ids=mlm_extra_excluded_token_ids,
            **mlm_masking_kwargs_from_config(pretrain_config),
        )
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=pretrain_config.seed) if distributed else RandomSampler(train_ds)

    # Move parameters to CUDA before optional TE MLP conversion so replacement
    # modules are created on the target device/rank.
    model.to(device)

    # ---- FP8 (optional) ----
    if pretrain_config.fp8:
        if device.type != "cuda":
            raise ValueError("fp8=True requires CUDA.")

        fp8_recipe = te_fp8_recipe_from_config(pretrain_config)
        fp8_target = getattr(pretrain_config, "fp8_target", "mlp")
        fp8_skip_last = int(getattr(pretrain_config, "fp8_skip_last_n_layers", 0))
        n_fp8 = convert_fp8_targets(
            model,
            target=fp8_target,
            skip_last_n_layers=fp8_skip_last,
        )
        if n_fp8 == 0:
            raise ValueError(
                "fp8=True requires at least one eligible block for Transformer Engine "
                f"FP8 conversion target={fp8_target!r}, "
                f"skip_last_n_layers={fp8_skip_last}."
            )
        logger.info(
            "FP8 enabled (Transformer Engine conversion): "
            f"target={fp8_target}, recipe={fp8_recipe}, "
            f"skip_last_n_layers={fp8_skip_last}, converted_blocks={n_fp8}"
        )

    _configure_fp8_cpu_activation_offload(
        model=model,
        pretrain_config=pretrain_config,
        device=device,
    )

    # ---- Model to device + compile + FSDP ----
    base_model = model

    compile_dynamic = not bool(use_packing and use_static_inp_size)

    # Precision detection (needed before FSDP MixedPrecisionPolicy)
    use_bf16 = pretrain_config.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = pretrain_config.bf16 and (
        (device.type == "cuda" and not torch.cuda.is_bf16_supported()) or device.type == "mps"
    )

    fsdp_mesh = None
    optimizer_dist = None
    if distributed and _is_fsdp(distributed_mode):
        fsdp_kwargs: dict = {}
        if use_bf16:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if distributed_mode == "hsdp":
            # 2D mesh: FSDP2 auto-enables HSDP — all-gather/reduce-scatter within
            # the "shard" dim (intra-node), all-reduce grads across "replicate"
            # (inter-node). fully_shard() takes the full 2D mesh; the Dion-family
            # optimizer takes only the 1D "shard" sub-mesh (its distributed
            # Newton-Schulz runs within the shard group, and FSDP2 already syncs
            # gradients across the replicate dim before optimizer.step()).
            shard_size = _resolve_hsdp_shard_size(pretrain_config, effective_world_size)
            replicate_size = effective_world_size // shard_size
            fsdp_mesh = init_device_mesh(
                "cuda",
                (replicate_size, shard_size),
                mesh_dim_names=("replicate", "shard"),
            )
            optimizer_dist = fsdp_mesh["shard"]
            logger.info(
                "HSDP enabled: mesh=(replicate=%d, shard=%d), world_size=%d",
                replicate_size,
                shard_size,
                effective_world_size,
            )
        else:
            fsdp_mesh = init_device_mesh("cuda", (effective_world_size,))
            optimizer_dist = fsdp_mesh
        fsdp_sublayer = getattr(pretrain_config, "fsdp_shard_granularity", "layer") == "sublayer"
        fsdp_reshard_after_forward = bool(getattr(pretrain_config, "fsdp_reshard_after_forward", False))
                
        _encoder = base_model.model
        _recycling_active = int(getattr(_encoder.config, "recycling_loop_count", 1) or 1) > 1
        _recurrent_ids = (
            {id(l) for l in getattr(_encoder, "recurrent_layers", [])}
            if _recycling_active
            else set()
        )
        for layer in _encoder.layers:
            if id(layer) in _recurrent_ids:
                continue
            _fully_shard_transformer_layer(
                layer,
                mesh=fsdp_mesh,
                fsdp_kwargs=fsdp_kwargs,
                reshard_after_forward=fsdp_reshard_after_forward,
                sublayer=fsdp_sublayer,
            )
        _fully_shard_root_groups(
            base_model,
            mesh=fsdp_mesh,
            fsdp_kwargs=fsdp_kwargs,
            reshard_after_forward=fsdp_reshard_after_forward,
        )
        fully_shard(base_model, mesh=fsdp_mesh, reshard_after_forward=fsdp_reshard_after_forward, **fsdp_kwargs)
        # optimizer_dist was set above (full mesh for fsdp, "shard" sub-mesh for hsdp)

        # Explicit prefetching for FSDP2
        if N_PREFETCH_LAYERS_FSDP2 > 1:
            layers = base_model.model.layers
            for i, layer in enumerate(layers):
                if i + 1 < len(layers):
                    layer.set_modules_to_forward_prefetch(layers[i + 1 : i + 1 + N_PREFETCH_LAYERS_FSDP2])
                if i - 1 >= 0:
                    layer.set_modules_to_backward_prefetch(list(reversed(layers[max(0, i - N_PREFETCH_LAYERS_FSDP2) : i])))

    # Keep base_model for checkpointing/eval (eval changes shapes → recompilation).
    orig_model = base_model
    compile_mode = (
        "max-autotune-no-cudagraphs"
        if getattr(pretrain_config, "use_compile_max_autotune", False)
        else None
    )
    _maybe_expand_dynamo_cache_for_batch_warmup(global_batch_warmup_steps)

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

    model = _build_compiled_training_model(
        base_model,
        compile_dynamic=compile_dynamic,
        compile_mode=compile_mode,
        distributed=distributed,
        distributed_mode=distributed_mode,
        ddp_bucket_cap_mb=ddp_bucket_cap_mb,
        local_rank=local_rank,
        device=device,
    )
    if distributed and distributed_mode == "ddp":
        optimizer_dist = dist.group.WORLD

    # ---- DataLoaders ----
    eval_collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
        standard_amino_acids_only=mlm_standard_amino_acids_only,
        extra_excluded_token_ids=mlm_extra_excluded_token_ids,
        **mlm_masking_kwargs_from_config(pretrain_config),
    )

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    if use_packing:
        train_loader = DataLoader(train_ds, batch_size=None, collate_fn=collator, **dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=pretrain_config.micro_batch_seqs, collate_fn=collator, drop_last=False, **dl_kwargs)
    eval_loader = DataLoader(val_ds, sampler=eval_sampler, batch_size=pretrain_config.micro_batch_seqs, collate_fn=eval_collator, drop_last=False, **dl_kwargs)

    # ---- Optimizer / Scheduler ----
    optimizer = _create_optimizer(orig_model, pretrain_config, distributed_mesh=optimizer_dist)

    synced_len = _sync_train_loader_len(len(train_loader), distributed, device)
    if synced_len == 0:
        raise RuntimeError(
            "train_loader is empty (synced_len=0); cannot satisfy the token budget. "
            "Check the dataset/manifest and per-rank sharding."
        )

    # ---- Training horizon (token budget -> optimizer steps) ----
    # total_steps comes from the configured token/step budget, NOT from epochs.
    # The DataLoader is repeated as many times as needed to meet the budget.
    horizon = resolve_total_steps(pretrain_config, achieved_global_batch_tokens)
    total_steps = horizon.total_steps
    warmup_steps = min(pretrain_config.warmup_steps, total_steps)
    if is_main and warmup_steps < pretrain_config.warmup_steps:
        logger.warning(
            "warmup_steps clamped from %d to %d (the token budget resolves to only "
            "%d optimizer steps).",
            pretrain_config.warmup_steps, warmup_steps, total_steps,
        )

    # ---- Epochs-equivalent startup hint (passes over the train set) ----
    if is_main:
        _approx = manifest.train_sequences * (manifest.min_seq_len + manifest.max_seq_len) // 2
        if pretrain_config.lr_schedule.lower() == "wsd":
            for label, steps in (("stable", horizon.stable_steps), ("decay", horizon.decay_steps),
                                 ("total", horizon.total_steps)):
                toks = steps * achieved_global_batch_tokens
                logger.info(f"[{label}] " + format_passes_hint(toks, manifest.total_train_tokens, _approx))
        else:
            toks = horizon.total_steps * achieved_global_batch_tokens
            logger.info(format_passes_hint(toks, manifest.total_train_tokens, _approx))

    # Clamp the raw logging/eval/save intervals (from prepare_run_and_steps) to
    # the token-budget total_steps now that the horizon is known.
    logging_steps = max(1, min(total_steps, logging_steps))
    eval_steps = max(1, min(total_steps, eval_steps))
    save_steps = max(1, min(total_steps, save_steps))

    if global_batch_warmup_steps > 0:
        logger.warning(
            "global_batch_warmup_steps=%d > 0: the token budget is approximate "
            "under batch-size warmup (the ceil token->step conversion slightly "
            "undercounts steps during the ramp).",
            global_batch_warmup_steps,
        )

    # ---- Recycling scheduler (no-op when recycling_loop_count == 1) ----
    from nanoplm.pretraining.recycling import RecyclingScheduler
    from nanoplm.pretraining.recycling.logging import (
        RecyclingMetricsBuffer,
        per_step_payload,
        logging_steps_payload,
        inloop_eval_payload,
    )
    recycling_scheduler = None
    recycling_rng = None
    recycling_metrics_buffer = None
    if model.model_config.recycling_loop_count > 1:
        # Resume semantics for the loop-count curriculum (spec §17):
        #   - `continue` keeps the curriculum on the ORIGINAL horizon, read from
        #     the checkpoint's training_state.json (not the locally recomputed
        #     total_steps), so the loop-count ramp does not shift on resume.
        #   - `decay` anneals a trunk that already finished the curriculum, so it
        #     runs at the TARGET loop count from step 0 (curriculum_complete).
        _is_resume = bool(resume_config and resume_config.is_resume)
        _resume_mode = (resume_config.mode or "continue").lower() if _is_resume else None
        _curriculum_complete = _is_resume and _resume_mode == "decay"
        if _is_resume and _resume_mode == "continue":
            _recycling_total_steps = int(
                _read_training_state(resume_config.checkpoint_dir).get("total_steps", total_steps)
            )
        else:
            _recycling_total_steps = total_steps
        recycling_scheduler = RecyclingScheduler(
            pretrain_config,
            target_loop_count=model.model_config.recycling_loop_count,
            total_training_steps=_recycling_total_steps,
            curriculum_complete=_curriculum_complete,
        )
        # +999 offset keeps the r-sampling RNG stream distinct from the seed
        # used elsewhere (model init, data shuffling, dropout).
        recycling_rng = torch.Generator(device=device).manual_seed(int(pretrain_config.seed) + 999)
        recycling_metrics_buffer = RecyclingMetricsBuffer(window=100)
    _is_wsd = pretrain_config.lr_schedule.lower() == "wsd"
    scheduler = _create_scheduler(
        optimizer, warmup_steps, total_steps,
        pretrain_config.adam_learning_rate, pretrain_config.lr_decay_to_fraction,
        pretrain_config.lr_schedule,
        stable_steps=horizon.stable_steps if _is_wsd else None,
        decay_steps=horizon.decay_steps if _is_wsd else None,
        decay_shape=pretrain_config.decay_shape,
    )

    # ---- Resume ----
    start_step, start_epoch = 0, 0
    resume_data_step, resume_sub_batch_step = 0, 0
    resume_batch_split_divisor = 1
    resume_epoch = 0
    # When a `decay` resume anneals a `warmup_stable` trunk, the loaded
    # pretrain_config still says lr_schedule="warmup_stable", which would make
    # the terminal save write "stable-end". Force "final" for decay resumes.
    resume_force_final = False
    if resume_config and resume_config.is_resume:
        _resume_mode = (resume_config.mode or "continue").lower()
        logger.info(
            f"Resuming from checkpoint: {resume_config.checkpoint_dir} "
            f"(mode={_resume_mode})"
        )
        start_step, start_epoch = _load_checkpoint(
            orig_model, optimizer, scheduler, resume_config.checkpoint_dir,
            device, distributed, load_scheduler=False, distributed_mode=distributed_mode,
        )
        logger.info(f"Resumed at global_step={start_step}, epoch={start_epoch}")

        # Detect dataset change: if the current dataset differs from the
        # checkpoint's dataset, start from batch 0 instead of fast-forwarding.
        _ckp_state_path = Path(resume_config.checkpoint_dir) / "training_state.json"
        _dataset_changed = False
        _ckp_state = None
        if _ckp_state_path.exists():
            _ckp_state = json.loads(_ckp_state_path.read_text(encoding="utf-8"))
            _ckp_fp = _ckp_state.get("dataset_fingerprint")
            if _ckp_fp is not None and _ckp_fp != _dataset_fingerprint:
                _dataset_changed = True
                logger.warning(
                    f"Dataset changed since checkpoint (fingerprint {_ckp_fp} → "
                    f"{_dataset_fingerprint}). Dataloader will start from batch 0 "
                    f"instead of fast-forwarding."
                )
            elif _ckp_fp is None and is_main:
                logger.info(
                    "Checkpoint has no dataset fingerprint (pre-upgrade checkpoint). "
                    "Assuming same dataset for dataloader positioning."
                )

        if _resume_mode == "decay":
            # Anneal a saved trunk: the decay phase is a fresh, short run from
            # global_step 0 over `decay_steps`. Peak LR is the checkpoint's
            # current (stable) LR already loaded into the optimizer.
            if resume_config.decay_steps is not None:
                resume_decay_steps = max(1, int(resume_config.decay_steps))
            elif resume_config.decay_tokens is not None:
                resume_decay_steps = tokens_to_steps(
                    int(resume_config.decay_tokens), achieved_global_batch_tokens
                )
            else:
                raise ValueError(
                    "resume mode='decay' requires decay_tokens or decay_steps"
                )
            # decay_steps (hence the LR curve) depends on the achieved per-step
            # token throughput. If this resume reconfigured batch size / world
            # size vs the trunk, the decay won't match a single-run wsd decay.
            _trunk_gbt = (_ckp_state or {}).get("achieved_global_batch_tokens")
            if (
                is_main and _trunk_gbt is not None
                and int(_trunk_gbt) != int(achieved_global_batch_tokens)
            ):
                logger.warning(
                    "Decay resume: achieved_global_batch_tokens changed since the trunk "
                    "(%d -> %d); decay_steps and the LR curve will differ from a "
                    "single-run wsd decay.",
                    int(_trunk_gbt), int(achieved_global_batch_tokens),
                )
            if is_main:
                logger.info(
                    "Resume decay anneals the trunk over %d steps. NOTE: the decay phase "
                    "restarts the dataloader from the start of a pass, so weight "
                    "equivalence with a single-run wsd decay holds only when the "
                    "stable->decay boundary is data-pass aligned (spec sections 7/13).",
                    resume_decay_steps,
                )
            scheduler = _rebuild_scheduler_for_resume(
                optimizer=optimizer,
                resume_config=resume_config,
                pretrain_config=pretrain_config,
                checkpoint_dir=resume_config.checkpoint_dir,
                decay_steps=resume_decay_steps,
                is_main=is_main,
            )
            # The decay phase runs from step 0 to `decay_steps`; override the
            # loop horizon so `while global_step < total_steps` runs exactly the
            # decay, and reset the start so global_step/tokens_seen anchor at 0.
            total_steps = resume_decay_steps
            warmup_steps = 0
            start_step = 0
            start_epoch = 0
            horizon = StepHorizon(total_steps=resume_decay_steps)
            resume_force_final = True
            # Re-clamp logging/eval/save to the (shorter) decay horizon.
            logging_steps = max(1, min(total_steps, logging_steps))
            eval_steps = max(1, min(total_steps, eval_steps))
            save_steps = max(1, min(total_steps, save_steps))
            # Fresh phase: no dataloader fast-forward.
            resume_epoch = 0
            resume_data_step = 0
            resume_sub_batch_step = 0
            resume_batch_split_divisor = 1
            if is_main:
                logger.info(
                    "Resume decay: total_steps=%d (decay), peak LR from checkpoint, "
                    "warmup=0, terminal checkpoint -> checkpoint-final",
                    total_steps,
                )
        else:
            # continue: finish to the ORIGINAL budget with the ORIGINAL schedule,
            # fast-forwarding the dataloader from the saved within-epoch position.
            scheduler = _rebuild_scheduler_for_resume(
                optimizer=optimizer,
                resume_config=resume_config,
                pretrain_config=pretrain_config,
                checkpoint_dir=resume_config.checkpoint_dir,
                start_step=start_step,
                total_steps=total_steps,
                stable_steps=horizon.stable_steps,
                decay_steps=horizon.decay_steps,
                is_main=is_main,
            )
            resume_epoch = start_epoch
            if (
                not _dataset_changed
                and _ckp_state is not None
                and "epoch_data_step" in _ckp_state
            ):
                # Restore the exact within-pass position saved at checkpoint time.
                # (Boundary/periodic saves persist epoch_data_step/sub_batch/
                # divisor; the dataloader fast-forwards to this point.)
                resume_data_step = int(_ckp_state.get("epoch_data_step", 0))
                resume_sub_batch_step = int(_ckp_state.get("epoch_sub_batch_step", 0))
                resume_batch_split_divisor = max(
                    1, int(_ckp_state.get("batch_split_divisor", 1))
                )
            else:
                # Dataset changed, pre-upgrade checkpoint, or a terminal/role
                # checkpoint without a saved within-pass position: replay the
                # DataLoader from the top of the current pass (spec §11 honest
                # fallback). The LR schedule still fast-forwards via global_step.
                resume_data_step = 0
                resume_sub_batch_step = 0
                resume_batch_split_divisor = 1
            if resume_data_step >= synced_len:
                # Saved position lands at/after the end of a pass; start the next
                # pass cleanly from the top.
                resume_data_step = 0
                resume_sub_batch_step = 0
                resume_batch_split_divisor = 1
                resume_epoch = start_epoch + 1
                start_epoch = resume_epoch
            if is_main:
                logger.info(
                    "Resume continue: total_steps=%d, start_step=%d, "
                    "fast-forward to epoch=%d data_step=%d sub_batch=%d divisor=%d",
                    total_steps, start_step, resume_epoch, resume_data_step,
                    resume_sub_batch_step, resume_batch_split_divisor,
                )

    # ---- W&B ----
    wandb_enabled = False
    if is_main:
        try:
            wandb.init(
                project=pretrain_config.project_name,
                name=wandb_run_name,
                mode="offline",
                config={"pretrain": pretrain_config.__dict__, "model": model.model_config.__dict__, "total_steps": total_steps, "warmup_steps": warmup_steps},
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
    if use_bf16 and (not distributed or distributed_mode == "ddp"):
        amp_dtype = torch.bfloat16
        if distributed_mode == "ddp":
            logger.warning(
                "DDP bf16 uses fp32 params + torch.autocast: the residual "
                "stream and norms run in fp32, which differs numerically from "
                "the FSDP path (bf16 forward params, no autocast) and "
                "disables bf16-gated kernels such as the CuTeDSL canon "
                "backend. Loss curves are not directly comparable with FSDP "
                "runs."
            )
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
    _raw = orig_model
    _cfg = _raw.config
    # Truncated-BPTT depth K: backward is retained only for the last K recurrent
    # iters (see _drive_recurrence / _recycling_flops_per_token), so the FLOPs
    # estimator must know K or it overcounts backward and inflates MFU for r > K.
    _bptt = max(1, int(getattr(pretrain_config, "recycling_bptt", 1)))
    _flops_per_token = _estimate_model_flops_per_token(_cfg, manifest.max_seq_len, bptt=_bptt)
    # Components for per-step FLOPs at the *actual* loop count (curriculum-accurate
    # MFU). _flops_per_token above is the nominal estimate at the target loop count.
    _flops_components = _recycling_flops_components(_cfg, manifest.max_seq_len)
    _static_loop_count = max(1, int(getattr(_cfg, "recycling_loop_count", 1)))
    _peak_flops = H100_PEAK_TFLOPS * 1e12
    logger.info(f"MFU estimation: {_flops_per_token:,} training FLOPs/token, H100 peak = {H100_PEAK_TFLOPS} TFLOPS")

    _run_t0 = time.perf_counter()
    logger.info(
        f"Starting pure-torch training: total_steps={total_steps} (token-budget driven), "
        f"warmup_steps={warmup_steps}, global_batch_warmup_steps={global_batch_warmup_steps}, "
        f"grad_accum={inferred_grad_accum_steps}, "
        f"achieved_global_batch_tokens={achieved_global_batch_tokens:,} tokens"
    )
    logger.info(
        f"Precision config: bf16={use_bf16}, fp16={use_fp16}, "
        f"tf32={(pretrain_config.tf32 and device.type == 'cuda')}, fp8={pretrain_config.fp8}"
    )
    logger.info(
        "Distributed config: multi_gpu=%s mode=%s%s",
        distributed,
        distributed_mode,
        f" ddp_bucket_cap_mb={ddp_bucket_cap_mb}" if distributed_mode == "ddp" else "",
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

    # ProRes: set up progressive residual warmup step tracking.
    _use_prores = getattr(_cfg, "use_prores", False)
    _prores_model = _raw.model if hasattr(_raw, "model") else None
    if _use_prores and _prores_model is not None and hasattr(_prores_model, "update_prores_alphas"):
        _prores_model.update_prores_alphas(start_step)
        logger.info(
            f"ProRes: T={_cfg.prores_T}, "
            f"last_layer_warmup_done_at_step={_cfg.prores_T * _cfg.num_hidden_layers}"
        )
    else:
        _prores_model = None

    # ---- Training loop ----
    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    # Running token counter for the schedule descriptor in training_state.json.
    # Anchored at start_step so resume stays consistent. When
    # global_batch_warmup_steps > 0 the per-step token count ramps; we use the
    # steady-state achieved_global_batch_tokens here, so tokens_seen is an
    # approximation in that regime (noted in the warmup log above).
    tokens_seen = int(start_step) * int(achieved_global_batch_tokens)
    accum_micro_count = 0
    effective_grad_accum, effective_micro_batch, mbs_divisor = _get_batch_warmup_params(
        global_step,
        inferred_grad_accum_steps,
        pretrain_config.micro_batch_seqs,
        global_batch_warmup_steps,
    )
    pending_compile_rebuild = False
    pending_compile_target_divisor = mbs_divisor
    accum_loss = torch.zeros((), device=device)
    # MoE routing bias update state.
    _moe_layers: list = []
    _moe_bias_lr = float(getattr(_cfg, "moe_bias_update_rate", 1e-3))
    if getattr(orig_model.config, "use_moe", False) and getattr(orig_model.config, "moe_use_bias_correction", False):
        from nanoplm.pretraining.models.modern_bert.moe import MoELayer as _MoELayer
        _moe_layers = [
            m for m in orig_model.modules() if isinstance(m, _MoELayer)
        ]
        if _moe_layers:
            _moe_num_experts = _moe_layers[0].num_experts
            _moe_count_accum = torch.zeros(
                len(_moe_layers), _moe_num_experts, device=device,
            )
            logger.info(
                f"MoE routing bias correction enabled: "
                f"{len(_moe_layers)} MoE layers, {_moe_num_experts} experts"
            )
    window_loss = torch.zeros((), device=device)
    window_steps = 0
    discard_accumulation = False
    token_count = torch.zeros((), dtype=torch.long, device=device)
    raw_token_count = torch.zeros((), dtype=torch.long, device=device)
    # Sum over the logging window of flops_per_token(actual r) * valid_tokens, so
    # MFU reflects the real compute done under the recycling curriculum rather than
    # the target loop count. float64 to hold ~1e15+ FLOPs/window without precision loss.
    window_flops = torch.zeros((), dtype=torch.float64, device=device)
    log_window_t0 = time.perf_counter()
    first_step_of_run = True
    debug_non_finite_params = bool(getattr(pretrain_config, "debug_non_finite_params", True))
    loss_finite_check = _DeferredFiniteCheck(device) if debug_non_finite_params else None
    param_finite_check = _DeferredFiniteCheck(device) if debug_non_finite_params else None

    def _raise_on_nonfinite_params(context: str) -> None:
        # Consumes the async verdict submitted after the previous optimizer
        # step; detection therefore lags corruption by at most one step.
        if param_finite_check is None or not param_finite_check.poll():
            return
        bad_param = _first_nonfinite_param(orig_model)
        bad_name = bad_param[0] if bad_param is not None else "<unknown>"
        rank = dist.get_rank() if distributed and dist.is_initialized() else 0
        raise RuntimeError(
            f"Non-finite parameter detected {context} in {bad_name} on rank {rank} "
            f"(async check: corruption occurred during the preceding optimizer step)."
        )
    debug_layerwise_enabled = bool(getattr(pretrain_config, "debug_layerwise_metrics", False))
    debug_tracker: LayerwiseDebugTracker | None = None
    if debug_layerwise_enabled:
        if is_main:
            logger.warning(
                "debug_layerwise_metrics=True: forward hooks will be installed on "
                "every transformer block and the lm_head. If torch.compile is "
                "enabled, expect graph breaks at those boundaries (5-20%% throughput "
                "regression typical)."
            )
        debug_tracker = LayerwiseDebugTracker(
            model=orig_model,
            log_every=int(getattr(pretrain_config, "debug_layerwise_log_every", 20)),
            device=device,
            output_dir=output_dir,
            is_main=is_main,
            distributed=bool(distributed and dist.is_initialized()),
        )
    epoch_setter = train_ds if use_packing else train_sampler
    # [0]=loss, [1]=grad_norm (local)
    log_buf = torch.empty(2, device=device, dtype=torch.float32)
    profiler_ctx, profiler_step_cb = _make_pure_profiler(pretrain_config, output_dir, is_main)
    if global_batch_warmup_steps > 0 and is_main:
        logger.info(
            "Batch-size warmup active: steps=%d, initial_grad_accum=%d, initial_micro_batch=%d, split_divisor=%d",
            global_batch_warmup_steps,
            effective_grad_accum,
            effective_micro_batch,
            mbs_divisor,
        )

    # Step-driven outer loop: repeat the DataLoader (one pass == one "epoch")
    # until the token-budget total_steps is reached. The horizon is driven by
    # global_step, not a fixed epoch count.
    epoch = start_epoch

    # Single entry point for every checkpoint write. The common kwargs are
    # captured by reference (resolved at call time); per-call-site differences
    # (phase, and the resume data-position fields) are passed via **over.
    def _save(phase, **over):
        _save_checkpoint(
            orig_model, optimizer, scheduler, global_step, epoch, output_dir,
            logging_steps, eval_steps, save_steps,
            distributed=distributed, is_main=is_main, distributed_mode=distributed_mode,
            model_config=_model_config, manifest=_manifest, pretrain_config=pretrain_config,
            total_steps=total_steps, warmup_steps=warmup_steps,
            dataset_fingerprint=_dataset_fingerprint,
            tokens_seen=tokens_seen,
            achieved_global_batch_tokens=achieved_global_batch_tokens,
            stable_steps=horizon.stable_steps, decay_steps=horizon.decay_steps,
            phase=phase,
            **over,
        )

    with profiler_ctx:
        while global_step < total_steps:
            if epoch_setter is not None and hasattr(epoch_setter, "set_epoch"):
                epoch_setter.set_epoch(epoch)
            # global_step at the start of this pass — used only for the
            # per-pass "optimizer steps" log line below.
            epoch_start_step = global_step

            train_iter = iter(train_loader)
            # Reset timing window AFTER dataloader workers are ready so the
            # first logging window doesn't include iter(train_loader) overhead.
            if device.type == "cuda":
                torch.cuda.synchronize()
            log_window_t0 = time.perf_counter()

            epoch_ended_early = False
            epoch_sub_batch_count = 0
            for micro_step in range(synced_len):
                if pending_compile_rebuild:
                    if distributed and dist.is_initialized():
                        _dist_barrier(local_rank)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    if is_main:
                        logger.info(
                            "[step %d] Rebuilding compiled training graph for split_divisor=%d",
                            global_step,
                            pending_compile_target_divisor,
                        )
                    old_model = model
                    model = None
                    del old_model
                    gc.collect()
                    try:
                        torch.compiler.reset()
                    except Exception:
                        logger.exception(
                            "torch.compiler.reset() failed during batch-size warmup transition; continuing."
                        )
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    model = _build_compiled_training_model(
                        base_model,
                        compile_dynamic=compile_dynamic,
                        compile_mode=compile_mode,
                        distributed=distributed,
                        distributed_mode=distributed_mode,
                        ddp_bucket_cap_mb=ddp_bucket_cap_mb,
                        local_rank=local_rank,
                        device=device,
                    )
                    model.train()
                    pending_compile_rebuild = False
                    if distributed and dist.is_initialized():
                        _dist_barrier(local_rank)

                has_batch = True
                try:
                    full_batch = next(train_iter)
                except StopIteration:
                    has_batch = False
                    full_batch = None

                # When packing + num_workers > 0, greedy bin-packing can produce
                # different batch counts per rank.  Coordinate so all ranks break
                # together to avoid FSDP / NCCL deadlock.  synced_len is the
                # all-rank MIN of exact per-rank loader lengths, so exhaustion is
                # only possible near the epoch tail — gate the (blocking)
                # allreduce to that window instead of serializing every step.
                near_epoch_tail = micro_step >= synced_len - BATCH_COORD_TAIL_STEPS
                if distributed and dist.is_initialized() and near_epoch_tail:
                    has_batch_t = torch.tensor(1 if has_batch else 0, device=device, dtype=torch.int32)
                    dist.all_reduce(has_batch_t, op=dist.ReduceOp.MIN)
                    if int(has_batch_t.item()) == 0:
                        epoch_ended_early = True
                        break
                elif not has_batch:
                    if distributed and dist.is_initialized():
                        # Outside the coordinated window other ranks keep going,
                        # so a silent break would deadlock FSDP — fail loudly.
                        raise RuntimeError(
                            f"Rank {dist.get_rank()} exhausted its dataloader at micro-step "
                            f"{micro_step}/{synced_len}, before the coordinated tail window "
                            f"(last {BATCH_COORD_TAIL_STEPS} steps); train loader __len__ over-reported."
                        )
                    epoch_ended_early = True
                    break

                if resume_data_step > 0 and epoch == resume_epoch and micro_step < resume_data_step:
                    if micro_step % 1000 == 0 and is_main:
                        logger.info(
                            f"Fast-forwarding dataloader: {micro_step}/{resume_data_step} "
                            f"data steps skipped ({100 * micro_step / resume_data_step:.0f}%)"
                        )
                    continue

                resume_sub_batch_start = 0
                batch_split_divisor = mbs_divisor
                if epoch == resume_epoch and micro_step == resume_data_step and resume_sub_batch_step > 0:
                    resume_sub_batch_start = resume_sub_batch_step
                    batch_split_divisor = resume_batch_split_divisor
                elif epoch == resume_epoch and micro_step == resume_data_step and resume_data_step > 0 and is_main:
                    logger.info(
                        f"Dataloader fast-forward complete — resuming training at data_step {micro_step}"
                    )

                effective_tokens_per_split = max(1, tokens_per_micro // max(1, batch_split_divisor))
                sub_batches = _split_batch(
                    full_batch,
                    n_splits=batch_split_divisor,
                    effective_tokens_per_split=effective_tokens_per_split if "cu_seqlens" in full_batch else None,
                )
                total_sub_batches = len(sub_batches)
                if resume_sub_batch_start > 0 and is_main:
                    logger.info(
                        "Resuming at data_step=%d sub_batch=%d/%d",
                        micro_step,
                        resume_sub_batch_start,
                        total_sub_batches,
                    )

                for sub_batch_idx, batch in enumerate(sub_batches):
                    if sub_batch_idx < resume_sub_batch_start:
                        continue

                    accum_micro_count += 1
                    epoch_sub_batch_count += 1
                    at_accum_boundary = accum_micro_count >= effective_grad_accum
                    if discard_accumulation:
                        if at_accum_boundary:
                            discard_accumulation = False
                            accum_micro_count = 0
                            effective_grad_accum, effective_micro_batch, mbs_divisor = _get_batch_warmup_params(
                                global_step,
                                inferred_grad_accum_steps,
                                pretrain_config.micro_batch_seqs,
                                global_batch_warmup_steps,
                            )
                        continue

                    batch = _move_batch_to_device(batch, device)
                    upcoming_step = global_step + 1
                    sync = at_accum_boundary
                    if distributed and _is_fsdp(distributed_mode):
                        model.set_requires_gradient_sync(sync)

                    if "num_valid_tokens" in batch:
                        microstep_tokens = batch["num_valid_tokens"]
                        raw_token_count += batch["input_ids"].numel()
                    elif "attention_mask" in batch:
                        microstep_tokens = batch["attention_mask"].sum()
                        raw_token_count += batch["attention_mask"].numel()
                    else:
                        microstep_tokens = batch["input_ids"].numel()
                        raw_token_count += batch["input_ids"].numel()
                    token_count += microstep_tokens

                    # Sample r for this micro-step (locked-step across DP workers)
                    sampled_r = None
                    curriculum_k_t = None
                    if recycling_scheduler is not None:
                        if is_main:
                            r_t_tensor = torch.tensor(
                                [recycling_scheduler.sample(global_step, device, recycling_rng)],
                                dtype=torch.int64, device=device,
                            )
                        else:
                            r_t_tensor = torch.zeros(1, dtype=torch.int64, device=device)
                        if distributed and dist.is_initialized():
                            dist.broadcast(r_t_tensor, src=0)
                        sampled_r = int(r_t_tensor.item())
                        curriculum_k_t = recycling_scheduler.current_mean(global_step)
                        recycling_metrics_buffer.record_step(sampled_r)

                    # Charge FLOPs at the loop count actually executed this micro-step
                    # (sampled_r under the curriculum), not the target loop count.
                    # Cast tokens to float64 first so float32 never truncates the ~1e9
                    # FLOPs/token product (no host sync — sampled_r already synced above).
                    _loop_for_flops = sampled_r if sampled_r is not None else _static_loop_count
                    _fpt = _recycling_flops_per_token(_flops_components, _loop_for_flops, bptt=_bptt)
                    _mt64 = microstep_tokens.double() if torch.is_tensor(microstep_tokens) else float(microstep_tokens)
                    window_flops += _fpt * _mt64

                    sync_ctx = _ddp_sync_context(model, distributed_mode=distributed_mode, sync=sync)
                    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
                    fp8_ctx = te_fp8_autocast_context(
                        enabled=bool(pretrain_config.fp8 and device.type == "cuda"),
                        cfg=pretrain_config,
                        distributed=distributed,
                    )
                    with sync_ctx:
                        with amp_ctx, fp8_ctx:
                            fwd_kwargs = dict(input_ids=batch["input_ids"], labels=batch["labels"])
                            if "attention_mask" in batch:
                                fwd_kwargs["attention_mask"] = batch["attention_mask"]
                            if "cu_seqlens" in batch:
                                cu_seqlens = batch["cu_seqlens"]
                                torch._dynamo.mark_dynamic(cu_seqlens, 0)
                                fwd_kwargs["cu_seqlens"] = cu_seqlens
                                fwd_kwargs["max_seqlen"] = batch["max_seqlen"]
                            if "position_ids" in batch:
                                fwd_kwargs["position_ids"] = batch["position_ids"]
                            if sampled_r is not None:
                                # Pass as 0-d int tensors (not Python ints) so
                                # Dynamo does NOT install a value guard at the
                                # _drive_recurrence disable boundary. With Python
                                # ints the compiled segment that calls the eager
                                # driver recompiles for every unique sampled r,
                                # thrashing the compile cache under Poisson
                                # sampling.
                                fwd_kwargs["recycling_loop_count"] = torch.tensor(
                                    sampled_r, dtype=torch.int64, device=device
                                )
                                fwd_kwargs["recycling_bptt"] = torch.tensor(
                                    pretrain_config.recycling_bptt,
                                    dtype=torch.int64,
                                    device=device,
                                )
                            out = model(**fwd_kwargs)

                        loss = out["loss"] if isinstance(out, dict) else out.loss
                        if loss_finite_check is not None:
                            # Verdict is read back only after backward() below has
                            # been queued, so this never drains the GPU stream.
                            loss_finite_check.submit(~torch.isfinite(loss.detach()).all())
                        loss = loss / effective_grad_accum

                        if scaler is not None and scaler.is_enabled():
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    if loss_finite_check is not None and loss_finite_check.poll():
                        rank = dist.get_rank() if distributed and dist.is_initialized() else 0
                        logger.error(
                            "Skipping optimizer step %d due to non-finite loss on rank %d "
                            "(epoch=%d micro_step=%d sub_batch=%d).",
                            upcoming_step, rank, epoch, micro_step, sub_batch_idx,
                        )
                        # backward() already ran on the non-finite loss; zero_grad
                        # below discards those gradients along with any accumulation.
                        optimizer.zero_grad(set_to_none=True)
                        accum_loss.zero_()
                        if _moe_layers:
                            _moe_count_accum.zero_()
                        if at_accum_boundary:
                            accum_micro_count = 0
                        else:
                            discard_accumulation = True
                        continue

                    accum_loss = accum_loss + loss.detach()

                    # Accumulate MoE expert counts for routing bias update.
                    if _moe_layers:
                        for j, moe_layer in enumerate(_moe_layers):
                            if moe_layer.last_expert_counts is not None:
                                _moe_count_accum[j] += moe_layer.last_expert_counts.float()

                    if not at_accum_boundary:
                        continue

                    if scaler is not None and scaler.is_enabled():
                        if isinstance(optimizer, MuonAdamWGroup):
                            # optimizer.muon may be an OptimizerGroup (muonh_scope=mlp:
                            # NorMuon + Hyperball); GradScaler needs each real optimizer.
                            for _o in (optimizer.muon.optimizers if isinstance(optimizer.muon, OptimizerGroup) else [optimizer.muon]):
                                scaler.unscale_(_o)
                            scaler.unscale_(optimizer.adamw)
                        else:
                            scaler.unscale_(optimizer)
                    if debug_tracker is not None and debug_tracker.should_flush(upcoming_step):
                        debug_tracker.record_step_boundary()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        orig_model.parameters(),
                        max_norm=max_grad_norm,
                        error_if_nonfinite=False,
                    )

                    step_skipped = False
                    if scaler is not None and scaler.is_enabled():
                        old_scale = scaler.get_scale()
                        if isinstance(optimizer, MuonAdamWGroup):
                            for _o in (optimizer.muon.optimizers if isinstance(optimizer.muon, OptimizerGroup) else [optimizer.muon]):
                                scaler.step(_o)
                            scaler.step(optimizer.adamw)
                        else:
                            scaler.step(optimizer)
                        scaler.update()
                        step_skipped = scaler.get_scale() < old_scale
                    else:
                        optimizer.step()
                    if param_finite_check is not None:
                        # Raise on the previous step's async verdict (its event has
                        # long completed), then enqueue this step's check — replaces
                        # a per-step `.item()` that drained the whole GPU stream.
                        _raise_on_nonfinite_params(f"at optimizer step {upcoming_step}")
                        nonfinite_t = _nonfinite_check_tensor([param for param in orig_model.parameters()])
                        if nonfinite_t is not None:
                            param_finite_check.submit(nonfinite_t)
                    if isinstance(optimizer, MuonAdamWGroup):
                        muon_lr = optimizer.muon.param_groups[0]["lr"]
                        learning_rate = optimizer.adamw.param_groups[0]["lr"]
                    elif _DION_OPTIMIZER_TYPES and isinstance(optimizer, _DION_OPTIMIZER_TYPES):
                        muon_lr = optimizer.param_groups[0]["lr"]
                        learning_rate = optimizer.param_groups[1]["lr"]
                    else:
                        learning_rate = optimizer.param_groups[0]["lr"]
                        muon_lr = None

                    if not step_skipped:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # MoE routing bias correction: update after each optimizer step.
                    if _moe_layers:
                        if not step_skipped:
                            with torch.no_grad():
                                # All-reduce counts across ranks for global view.
                                if distributed and dist.is_initialized():
                                    dist.all_reduce(_moe_count_accum, op=dist.ReduceOp.SUM)
                                for j, moe_layer in enumerate(_moe_layers):
                                    counts_j = _moe_count_accum[j]
                                    # Keep the update branchless so we never materialize
                                    # a CUDA scalar on CPU via `if total > 0`.
                                    total = counts_j.sum().clamp_min(1.0)
                                    target = total / _moe_num_experts
                                    # Nudge bias: positive for underused, negative for overused.
                                    # correction_bias is a registered buffer (not a
                                    # parameter), so FSDP won't shard it.
                                    delta = _moe_bias_lr * (target - counts_j) / target
                                    # Zero-mean: keep bias vector centered (Ling-V2 style).
                                    # When counts are all zero, this naturally becomes all-zero.
                                    delta = delta - delta.mean()
                                    moe_layer.router.correction_bias.add_(
                                        delta.to(moe_layer.router.correction_bias.device)
                                    )
                        _moe_count_accum.zero_()

                    global_step += 1
                    tokens_seen += int(achieved_global_batch_tokens)
                    profiler_step_cb(global_step)

                    if _prores_model is not None:
                        _prores_model.update_prores_alphas(global_step)

                    if _use_repo and global_step == repo_rope_warmup_steps:
                        _repo_model.repo_active = True
                        if is_main:
                            logger.info(f"[step {global_step}] RePO activated (repo warmup complete)")

                    prev_warmup_state = (
                        effective_grad_accum,
                        effective_micro_batch,
                        mbs_divisor,
                    )
                    accum_micro_count = 0
                    effective_grad_accum, effective_micro_batch, mbs_divisor = _get_batch_warmup_params(
                        global_step,
                        inferred_grad_accum_steps,
                        pretrain_config.micro_batch_seqs,
                        global_batch_warmup_steps,
                    )
                    if prev_warmup_state[2] != mbs_divisor:
                        pending_compile_rebuild = True
                        pending_compile_target_divisor = mbs_divisor
                    if (
                        global_batch_warmup_steps > 0
                        and is_main
                        and prev_warmup_state != (effective_grad_accum, effective_micro_batch, mbs_divisor)
                    ):
                        logger.info(
                            "[step %d] Batch-size warmup -> grad_accum=%d micro_batch=%d split_divisor=%d",
                            global_step,
                            effective_grad_accum,
                            effective_micro_batch,
                            mbs_divisor,
                        )

                    window_loss += accum_loss.detach()
                    window_steps += 1
                    accum_loss.zero_()

                    should_log = global_step % logging_steps == 0
                    vram_log = ""
                    tps = mfu = tok = raw_tok = 0.0
                    step_tok = step_raw_tok = 0.0
                    real_tps = real_tps_log = 0.0
                    avg_step_ms = 0.0
                    if should_log:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        window_dt = max(1e-9, t1 - log_window_t0)
                        log_window_t0 = t1
                        avg_step_ms = (window_dt * 1000.0) / max(1, window_steps)
                        log_buf[0] = window_loss / max(1, window_steps)
                        log_buf[1] = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else (grad_norm if isinstance(grad_norm, torch.Tensor) else float(grad_norm))
                        log_vals = log_buf.cpu()
                        avg_loss = float(log_vals[0])
                        grad_norm_val = float(log_vals[1])
                        # Reduce valid tokens, raw tokens, and the actual-loop-count
                        # FLOPs for this window in a single collective. window_flops
                        # already weights each micro-step by its sampled loop count,
                        # so MFU = achieved FLOP/s / peak reflects real compute — not
                        # the target loop count, which overstates MFU early in the
                        # recycling curriculum (when r is still ~1).
                        reduce_buf = torch.stack([
                            token_count.double(),
                            raw_token_count.double(),
                            window_flops,
                        ])
                        if distributed and dist.is_initialized():
                            dist.all_reduce(reduce_buf, op=dist.ReduceOp.SUM)
                        tok = float(reduce_buf[0].item())
                        raw_tok = float(reduce_buf[1].item())
                        window_flops_total = float(reduce_buf[2].item())
                        tps = int(raw_tok / window_dt)
                        step_tok = tok / max(1, window_steps)
                        step_raw_tok = raw_tok / max(1, window_steps)
                        real_tps = tok / window_dt
                        real_tps_log = int(real_tps)
                        mfu = window_flops_total / window_dt / (_peak_flops * max(effective_world_size, 1))

                        token_count.zero_()
                        raw_token_count.zero_()
                        window_flops.zero_()

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
                        epoch_progress = epoch + (
                            micro_step + ((sub_batch_idx + 1) / max(1, total_sub_batches))
                        ) / max(1, synced_len)
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
                            "train/epoch": epoch_progress,
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
                                if recycling_scheduler is not None and sampled_r is not None:
                                    payload.update(per_step_payload(sampled_r, curriculum_k_t))
                                if recycling_scheduler is not None and should_log:
                                    adapter = getattr(orig_model.model, "adapter", None) if hasattr(orig_model, "model") else None
                                    payload.update(logging_steps_payload(adapter, recycling_metrics_buffer))
                                wandb.log(payload, step=global_step)
                            except Exception as exc:
                                wandb_enabled = False
                                logger.warning(f"W&B log failed; disabling. Error: {exc}")

                    if debug_tracker is not None and debug_tracker.should_flush(global_step):
                        debug_payload = debug_tracker.flush(global_step)
                        if is_main and debug_payload is not None and wandb_enabled and wandb.run is not None:
                            try:
                                wandb.log({**debug_payload, "train/global_step": global_step}, step=global_step)
                            except Exception as exc:
                                logger.warning(f"W&B debug log failed: {exc}")

                    if first_step_of_run:
                        first_step_of_run = False
                        gc.collect(); gc.freeze(); gc.disable()
                    elif global_step % 5000 == 0:
                        gc.collect()

                    if should_log:
                        window_loss.zero_()
                        window_steps = 0

                    # ---- Phase-boundary detection (role-named checkpoints) ----
                    # Terminal role per schedule (spec §6):
                    #   wsd            -> stable-end at stable_steps (intermediate,
                    #                     pre-decay) + final at total_steps.
                    #   warmup_stable  -> stable-end at total_steps (the reusable
                    #                     trunk; no separate final).
                    #   cosine/linear  -> final at total_steps.
                    # The terminal checkpoint (at total_steps) is written exactly
                    # once by the post-loop save below; the in-loop branch only
                    # handles the intermediate wsd stable-end.
                    _sched = pretrain_config.lr_schedule.lower()
                    # The wsd intermediate stable-end fires in-loop at stable_steps
                    # (always < total_steps, pre-decay). Every schedule's TERMINAL
                    # checkpoint (at total_steps) is the post-loop save's
                    # responsibility, so the in-loop booleans are wsd-intermediate
                    # only.
                    at_wsd_stable_end = (
                        _sched == "wsd" and horizon.stable_steps is not None
                        and global_step == horizon.stable_steps
                    )
                    at_terminal = (global_step == total_steps)
                    do_periodic = global_step % save_steps == 0

                    if global_step % eval_steps == 0 or at_wsd_stable_end:
                        _raise_on_nonfinite_params(f"before eval at step {global_step}")
                        if recycling_scheduler is not None and pretrain_config.recycling_pretraining_eval_loop_counts:
                            eval_loop_counts = pretrain_config.recycling_pretraining_eval_loop_counts
                        elif recycling_scheduler is not None:
                            eval_loop_counts = [model.model_config.recycling_loop_count]
                        else:
                            eval_loop_counts = [None]

                        losses_by_r: dict[int, float] = {}
                        eval_loss = None
                        for r_value in eval_loop_counts:
                            eval_loss = _evaluate(
                                orig_model, eval_loader, device, distributed, amp_dtype,
                                recycling_loop_count=r_value,
                                recycling_bptt=pretrain_config.recycling_bptt if r_value is not None else None,
                            )
                            if r_value is not None:
                                losses_by_r[r_value] = eval_loss

                        if is_main:
                            if losses_by_r:
                                losses_str = ", ".join(f"r={r}: {l:.4f}" for r, l in sorted(losses_by_r.items()))
                                logger.info(f"[step {global_step}] in-loop eval {losses_str}")
                                if wandb_enabled and wandb.run is not None:
                                    try:
                                        p = {"train/global_step": global_step}
                                        p.update(inloop_eval_payload(losses_by_r))
                                        target = model.model_config.recycling_loop_count
                                        if target in losses_by_r:
                                            p["eval/loss"] = losses_by_r[target]
                                        wandb.log(p)
                                    except Exception as exc:
                                        wandb_enabled = False
                                        logger.warning(f"W&B log failed; disabling. Error: {exc}")
                            else:
                                logger.info(f"[step {global_step}] eval_loss={eval_loss:.4f}")
                                if wandb_enabled and wandb.run is not None:
                                    try:
                                        wandb.log({"train/global_step": global_step, "eval/loss": eval_loss})
                                    except Exception as exc:
                                        wandb_enabled = False
                                        logger.warning(f"W&B log failed; disabling. Error: {exc}")
                        model.train()

                    # In-loop saves: the forced wsd stable-end boundary and the
                    # periodic save_steps cadence. Skip the periodic save at the
                    # terminal step -- the post-loop save writes the role-named
                    # terminal checkpoint there for every schedule.
                    if at_wsd_stable_end or (do_periodic and not at_terminal):
                        _raise_on_nonfinite_params(f"before checkpoint at step {global_step}")
                        next_data_step = micro_step
                        next_sub_batch_step = sub_batch_idx + 1
                        next_batch_split_divisor = batch_split_divisor
                        if next_sub_batch_step >= total_sub_batches:
                            next_data_step = micro_step + 1
                            next_sub_batch_step = 0
                            next_batch_split_divisor = mbs_divisor
                        _pos = dict(
                            epoch_data_step=next_data_step,
                            epoch_sub_batch_step=next_sub_batch_step,
                            batch_split_divisor=next_batch_split_divisor,
                        )
                        _save("stable-end" if at_wsd_stable_end else "periodic", **_pos)
                        # When the wsd stable-end coincides with the periodic
                        # cadence, also write the periodic checkpoint so the
                        # save_steps trail is unbroken.
                        if at_wsd_stable_end and do_periodic:
                            _save("periodic", **_pos)

                if epoch == resume_epoch and micro_step == resume_data_step:
                    resume_data_step = 0
                    resume_sub_batch_step = 0
                    resume_batch_split_divisor = 1

                # Token-budget horizon reached: stop after the current
                # optimizer step's logging/eval/save have fired.
                if global_step >= total_steps:
                    break

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
            epoch_fwd_count = epoch_sub_batch_count
            partial_discarded = accum_micro_count
            if is_main:
                logger.info(
                    "Data pass %d complete (global_step %d/%d): %d effective micro-steps, %d optimizer steps%s",
                    epoch + 1, global_step, total_steps, epoch_fwd_count,
                    global_step - epoch_start_step,
                    f" ({partial_discarded} trailing micro-step(s) discarded)"
                    if partial_discarded else "",
                )
            optimizer.zero_grad(set_to_none=True)
            accum_loss.zero_()
            accum_micro_count = 0
            discard_accumulation = False
            token_count.zero_()
            raw_token_count.zero_()
            window_flops.zero_()
            window_loss.zero_()
            window_steps = 0

            if (resume_data_step > 0 or resume_sub_batch_step > 0) and epoch == resume_epoch:
                resume_data_step = 0
                resume_sub_batch_step = 0
                resume_batch_split_divisor = 1

            # Stop once the token-budget horizon is met; otherwise replay the
            # DataLoader for another pass.
            if global_step >= total_steps:
                break
            epoch += 1

    # ---- Finalize ----
    _raise_on_nonfinite_params(f"before final checkpoint at step {global_step}")
    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    # Terminal role checkpoint, written exactly once at total_steps:
    #   resume decay  -> final (anneal output; overrides the trunk's schedule)
    #   warmup_stable -> stable-end (the reusable trunk a later run anneals)
    #   wsd / cosine / linear -> final
    _terminal_phase = (
        "final"
        if resume_force_final
        else (
            "stable-end"
            if pretrain_config.lr_schedule.lower() == "warmup_stable"
            else "final"
        )
    )
    _save(_terminal_phase)

    if debug_tracker is not None:
        debug_tracker.close()

    if is_main and wandb_enabled and wandb.run is not None:
        try:
            (Path(output_dir) / "wandb_run_id.txt").write_text(wandb.run.id, encoding="utf-8")
            wandb.finish()
        except Exception as exc:
            logger.warning(f"W&B finalize failed. Error: {exc}")

    if distributed and dist.is_initialized():
        _dist_barrier(local_rank)

    logger.info("Pure-torch pretraining complete.")
