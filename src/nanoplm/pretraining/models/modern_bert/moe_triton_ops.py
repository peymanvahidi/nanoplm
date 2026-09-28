"""Torch custom ops for MoE dispatch/combine Triton kernels (see
``moe_triton_kernels.py``).

Provides custom ops registered via ``torch.library``:
  - ``scatter_dispatch``: fuses repeat_interleave + permute into a single gather
  - ``scatter_add_bwd``: Triton atomic scatter-add for scatter_dispatch backward
  - ``gather_combine``: fuses unpermute + mul(weights) + sum(top_k)
  - ``gather_combine_bwd``: fused backward producing both grad_expert_sorted
    and grad_weights in a single pass (loads grad_out once)

All backward ops are registered as separate custom ops (not direct Triton
launches) so that ``torch.compile`` can trace through FakeTensors.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any

import torch
import torch.library

from ._globals import PORTABLE as _PORTABLE

_lib = torch.library.Library("nanoplm_moe", "DEF")

_lib.define("scatter_dispatch(Tensor x, Tensor token_idx) -> Tensor")
_lib.define("scatter_add_bwd(Tensor grad_sorted, Tensor token_idx, int T) -> Tensor")
_lib.define(
    "gather_combine(Tensor expert_out_sorted, Tensor inv_map, "
    "Tensor weights, float scale) -> Tensor"
)
_lib.define(
    "gather_combine_bwd(Tensor grad_out, Tensor expert_out_sorted, "
    "Tensor inv_map, Tensor weights, float scale, int M, int eo_dtype) "
    "-> (Tensor, Tensor)"
)

# Encode dtype as int so it can pass through the op schema.
_DTYPE_TO_INT = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2, torch.float64: 3}
_INT_TO_DTYPE = {v: k for k, v in _DTYPE_TO_INT.items()}


# ── Autotune control ─────────────────────────────────────────────────────────

_AUTOTUNE_CONTROL = threading.local()


def _autotune_disable_depth() -> int:
    return int(getattr(_AUTOTUNE_CONTROL, "disable_depth", 0))


@contextmanager
def disable_moe_autotune_temporarily():
    """Temporarily force the heuristic (non-autotune) launch path."""
    _AUTOTUNE_CONTROL.disable_depth = _autotune_disable_depth() + 1
    try:
        yield
    finally:
        _AUTOTUNE_CONTROL.disable_depth = max(0, _autotune_disable_depth() - 1)


def _moe_autotune_enabled() -> bool:
    v = os.getenv("NANOPLM_MOE_TRITON_AUTOTUNE", "1").strip().lower()
    return v not in {"0", "false", "off", "no"} and _autotune_disable_depth() == 0


def _moe_autotune_status_enabled() -> bool:
    v = os.getenv("NANOPLM_MOE_TRITON_AUTOTUNE_STATUS", "1").strip().lower()
    return v not in {"0", "false", "off", "no"}


_MOE_AUTOTUNE_STATUS_SEEN: set[tuple[str, tuple[Any, ...]]] = set()


def _moe_autotune_key(autotuner: Any, args_by_name: dict[str, Any]) -> tuple[Any, ...]:
    key: list[Any] = []
    for name in getattr(autotuner, "keys", ()):
        if name in args_by_name:
            key.append(args_by_name[name])
    for name in getattr(autotuner, "arg_names", ()):
        if name in args_by_name:
            arg = args_by_name[name]
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
    return tuple(key)


def _moe_autotune_status_begin(
    *,
    kernel_name: str,
    autotuner: Any,
    args_by_name: dict[str, Any],
) -> tuple[tuple[Any, ...], bool, Any] | None:
    if not _moe_autotune_status_enabled():
        return None
    key = _moe_autotune_key(autotuner, args_by_name)
    tag = (kernel_name, key)
    if tag in _MOE_AUTOTUNE_STATUS_SEEN:
        return None
    _MOE_AUTOTUNE_STATUS_SEEN.add(tag)

    in_mem_hit = key in getattr(autotuner, "cache", {})
    bench_before = getattr(autotuner, "bench_time", None)
    if in_mem_hit:
        cfg = autotuner.cache[key]
        print(
            f"[nanoplm][triton] autotune cache hit: {kernel_name} key={key} cfg={cfg}",
            flush=True,
        )
    else:
        num_cfgs = len(getattr(autotuner, "configs", ()))
        print(
            f"[nanoplm][triton] resolving autotune config: {kernel_name} key={key} "
            f"(may benchmark up to {num_cfgs} configs on first run)",
            flush=True,
        )
    return key, in_mem_hit, bench_before


def _moe_autotune_status_end(
    *,
    kernel_name: str,
    autotuner: Any,
    state: tuple[tuple[Any, ...], bool, Any] | None,
) -> None:
    if not _moe_autotune_status_enabled() or state is None:
        return
    key, in_mem_hit, bench_before = state
    if in_mem_hit:
        return
    bench_after = getattr(autotuner, "bench_time", None)
    ran_benchmark = bench_after is not None and bench_after != bench_before
    cfg = getattr(autotuner, "cache", {}).get(key, getattr(autotuner, "best_config", None))
    if ran_benchmark:
        print(
            f"[nanoplm][triton] autotune finished: {kernel_name} selected={cfg}",
            flush=True,
        )
    else:
        print(
            f"[nanoplm][triton] autotune loaded from disk cache: {kernel_name} selected={cfg}",
            flush=True,
        )


# ── Eager fallbacks (CPU, MPS, and CUDA in portable mode) ────────────────────
#
# In portable mode each CUDA impl below returns its CPU impl before touching
# ``moe_triton_kernels``, so no Triton import ever happens.


def _gather_combine_fwd_eager(expert_out_sorted, inv_map, weights, scale):
    T, top_k = inv_map.shape
    C = expert_out_sorted.shape[1]
    flat_idx = inv_map.long().reshape(-1)
    rows = expert_out_sorted[flat_idx].view(T, top_k, C)
    if expert_out_sorted.dtype in (torch.bfloat16, torch.float16):
        rows = rows.float()
    return (rows * weights.unsqueeze(-1) * scale).sum(dim=1)


def _gather_combine_bwd_eager(
    grad_out, expert_out_sorted, inv_map, weights, scale, M, out_dtype,
):
    T, top_k = inv_map.shape
    C = grad_out.shape[1]
    fp32_accum = out_dtype in (torch.bfloat16, torch.float16)

    # grad_expert_sorted
    grad_eo = torch.empty(M, C, dtype=out_dtype, device=grad_out.device)
    for k in range(top_k):
        positions = inv_map[:, k].long()
        val = grad_out * weights[:, k : k + 1] * scale
        if fp32_accum:
            val = val.to(out_dtype)
        grad_eo[positions] = val

    # grad_weights
    grad_w = torch.empty(T, top_k, dtype=grad_out.dtype, device=grad_out.device)
    for k in range(top_k):
        positions = inv_map[:, k].long()
        eo_k = expert_out_sorted[positions]
        if fp32_accum:
            eo_k = eo_k.float()
        grad_w[:, k] = (grad_out * eo_k).sum(dim=-1) * scale

    return grad_eo, grad_w


# ── build_inverse_map helper ─────────────────────────────────────────────────


def build_inverse_map(
    token_idx: torch.Tensor,
    slot_idx: torch.Tensor,
    T: int,
    top_k: int,
) -> torch.Tensor:
    """Build inv_map[token_idx[i], slot_idx[i]] = i for all i."""
    inv_map = torch.empty(T, top_k, dtype=torch.int32, device=token_idx.device)
    positions = torch.arange(
        token_idx.shape[0], dtype=torch.int32, device=token_idx.device,
    )
    inv_map[token_idx.long(), slot_idx.long()] = positions
    return inv_map


# ── scatter_dispatch: fuses repeat_interleave + permute ──────────────────────


@torch.library.register_fake("nanoplm_moe::scatter_dispatch")
def _scatter_dispatch_fake(x, token_idx):
    return torch.empty(
        token_idx.shape[0], x.shape[1], dtype=x.dtype, device=x.device,
    )


@torch.library.impl(_lib, "scatter_dispatch", "CUDA")
def _scatter_dispatch_cuda(x, token_idx):
    if _PORTABLE:
        return _scatter_dispatch_cpu(x, token_idx)
    from . import moe_triton_kernels as k

    x = x.contiguous()
    token_idx = token_idx.contiguous()
    M = token_idx.shape[0]
    D = x.shape[1]
    y = torch.empty(M, D, dtype=x.dtype, device=x.device)
    grid = (k.triton.cdiv(M, 64), k.triton.cdiv(D, 128))
    k._moe_gather_kernel[grid](
        x, token_idx, y, M, D,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_D=128, num_warps=4, num_stages=2,
    )
    return y


@torch.library.impl(_lib, "scatter_dispatch", "CPU")
def _scatter_dispatch_cpu(x, token_idx):
    return x[token_idx.long()]


@torch.library.impl(_lib, "scatter_dispatch", "MPS")
def _scatter_dispatch_mps(x, token_idx):
    # No fused MPS kernel exists; this is a plain index_select, which Metal
    # handles natively. MPS is an inference target, never a training one, so
    # only the forward op needs a registration here (see scatter_add_bwd
    # below, which stays CPU/CUDA-only).
    return x[token_idx.long()]


# ── scatter_add_bwd ──────────────────────────────────────────────────────────


@torch.library.register_fake("nanoplm_moe::scatter_add_bwd")
def _scatter_add_bwd_fake(grad_sorted, token_idx, T):
    return torch.empty(T, grad_sorted.shape[1], dtype=grad_sorted.dtype, device=grad_sorted.device)


@torch.library.impl(_lib, "scatter_add_bwd", "CUDA")
def _scatter_add_bwd_cuda(grad_sorted, token_idx, T):
    if _PORTABLE:
        return _scatter_add_bwd_cpu(grad_sorted, token_idx, T)
    from . import moe_triton_kernels as k

    grad_sorted = grad_sorted.contiguous()
    token_idx = token_idx.contiguous()
    M, D = grad_sorted.shape
    grad_x = torch.zeros(T, D, dtype=grad_sorted.dtype, device=grad_sorted.device)
    grid = (k.triton.cdiv(M, 64), k.triton.cdiv(D, 128))
    k._moe_scatter_add_kernel[grid](
        grad_sorted, token_idx, grad_x,
        M, D,
        grad_sorted.stride(0), grad_sorted.stride(1),
        grad_x.stride(0), grad_x.stride(1),
        BLOCK_M=64, BLOCK_D=128,
        num_warps=4, num_stages=2,
    )
    return grad_x


@torch.library.impl(_lib, "scatter_add_bwd", "CPU")
def _scatter_add_bwd_cpu(grad_sorted, token_idx, T):
    D = grad_sorted.shape[1]
    grad_x = torch.zeros(T, D, dtype=grad_sorted.dtype, device=grad_sorted.device)
    grad_x.index_add_(0, token_idx.long(), grad_sorted)
    return grad_x


def _scatter_dispatch_setup_context(ctx, inputs, output):
    x, token_idx = inputs
    ctx.save_for_backward(token_idx)
    ctx.num_tokens = x.shape[0]


def _scatter_dispatch_backward(ctx, grad_sorted):
    (token_idx,) = ctx.saved_tensors
    return torch.ops.nanoplm_moe.scatter_add_bwd(
        grad_sorted, token_idx, ctx.num_tokens,
    ), None


torch.library.register_autograd(
    "nanoplm_moe::scatter_dispatch",
    _scatter_dispatch_backward,
    setup_context=_scatter_dispatch_setup_context,
)


def moe_scatter_dispatch(x: torch.Tensor, token_idx: torch.Tensor) -> torch.Tensor:
    return torch.ops.nanoplm_moe.scatter_dispatch(x, token_idx)


# ── gather_combine: fuses unpermute + mul(weights) + sum(top_k) ──────────────


@torch.library.register_fake("nanoplm_moe::gather_combine")
def _gather_combine_fake(expert_out_sorted, inv_map, weights, scale):
    T = inv_map.shape[0]
    C = expert_out_sorted.shape[1]
    out_dtype = (
        torch.float32
        if expert_out_sorted.dtype != torch.float64
        else torch.float64
    )
    return torch.empty(T, C, dtype=out_dtype, device=expert_out_sorted.device)


def _launch_gather_combine_triton(expert_out_sorted, inv_map, weights, scale):
    from . import moe_triton_kernels as k

    T, top_k = inv_map.shape
    C = expert_out_sorted.shape[1]
    fp32_accum = expert_out_sorted.dtype != torch.float64
    out_dtype = torch.float32 if fp32_accum else torch.float64

    output = torch.empty(T, C, dtype=out_dtype, device=expert_out_sorted.device)

    if _moe_autotune_enabled():
        autotuner = k._moe_gather_combine_fwd_at
        status = _moe_autotune_status_begin(
            kernel_name="moe_gc_fwd",
            autotuner=autotuner,
            args_by_name={"C": C, "eo_ptr": expert_out_sorted},
        )
        autotuner[(T,)](
            expert_out_sorted, inv_map, weights, output,
            scale, C,
            expert_out_sorted.stride(0), expert_out_sorted.stride(1),
            inv_map.stride(0), inv_map.stride(1),
            weights.stride(0), weights.stride(1),
            output.stride(0), output.stride(1),
            TOP_K=top_k, FP32_ACCUM=fp32_accum,
        )
        _moe_autotune_status_end(kernel_name="moe_gc_fwd", autotuner=autotuner, state=status)
    else:
        cc_major = torch.cuda.get_device_capability()[0]
        if C < 256:
            BLOCK_D = k.triton.next_power_of_2(C)
            num_warps = 4
        elif cc_major >= 9:  # SM90 (Hopper), SM120 (Blackwell)
            BLOCK_D, num_warps = 256, 8
        else:  # SM80 (Ampere) and older
            BLOCK_D, num_warps = 256, 4
        k._moe_gather_combine_fwd_kernel[(T,)](
            expert_out_sorted, inv_map, weights, output,
            scale, C,
            expert_out_sorted.stride(0), expert_out_sorted.stride(1),
            inv_map.stride(0), inv_map.stride(1),
            weights.stride(0), weights.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_D=BLOCK_D, TOP_K=top_k, FP32_ACCUM=fp32_accum,
            num_warps=num_warps, num_stages=2,
        )
    return output


@torch.library.impl(_lib, "gather_combine", "CUDA")
def _gather_combine_cuda(expert_out_sorted, inv_map, weights, scale):
    if _PORTABLE:
        return _gather_combine_cpu(expert_out_sorted, inv_map, weights, scale)
    return _launch_gather_combine_triton(expert_out_sorted, inv_map, weights, scale)


@torch.library.impl(_lib, "gather_combine", "CPU")
def _gather_combine_cpu(expert_out_sorted, inv_map, weights, scale):
    return _gather_combine_fwd_eager(expert_out_sorted, inv_map, weights, scale)


@torch.library.impl(_lib, "gather_combine", "MPS")
def _gather_combine_mps(expert_out_sorted, inv_map, weights, scale):
    # Same reasoning as scatter_dispatch's MPS impl above: the eager
    # (gather + weighted sum) path is plain torch ops with no Triton
    # dependency, so it runs unchanged on MPS. Only the forward op is
    # registered; gather_combine_bwd stays CPU/CUDA-only (inference-only
    # target).
    return _gather_combine_fwd_eager(expert_out_sorted, inv_map, weights, scale)


# ── gather_combine fused backward ────────────────────────────────────────────


@torch.library.register_fake("nanoplm_moe::gather_combine_bwd")
def _gather_combine_bwd_fake(
    grad_out, expert_out_sorted, inv_map, weights, scale, M, eo_dtype,
):
    T, top_k = inv_map.shape
    C = grad_out.shape[1]
    return (
        torch.empty(M, C, dtype=_INT_TO_DTYPE[eo_dtype], device=grad_out.device),
        torch.empty(T, top_k, dtype=grad_out.dtype, device=grad_out.device),
    )


@torch.library.impl(_lib, "gather_combine_bwd", "CUDA")
def _gather_combine_bwd_cuda(
    grad_out, expert_out_sorted, inv_map, weights, scale, M, eo_dtype,
):
    if _PORTABLE:
        return _gather_combine_bwd_cpu(
            grad_out, expert_out_sorted, inv_map, weights, scale, M, eo_dtype
        )
    from . import moe_triton_kernels as k

    T, top_k = inv_map.shape
    C = grad_out.shape[1]
    out_dtype = _INT_TO_DTYPE[eo_dtype]
    fp32_accum = out_dtype != torch.float64
    TOP_K_PAD = k.triton.next_power_of_2(top_k)

    grad_eo = torch.empty(M, C, dtype=out_dtype, device=grad_out.device)
    grad_w = torch.empty(T, top_k, dtype=grad_out.dtype, device=grad_out.device)

    if _moe_autotune_enabled():
        autotuner = k._moe_gather_combine_bwd_at
        status = _moe_autotune_status_begin(
            kernel_name="moe_gc_bwd",
            autotuner=autotuner,
            args_by_name={"C": C, "go_ptr": grad_out},
        )
        autotuner[(T,)](
            grad_out, expert_out_sorted, inv_map, weights, grad_eo, grad_w,
            scale, C,
            grad_out.stride(0), grad_out.stride(1),
            expert_out_sorted.stride(0), expert_out_sorted.stride(1),
            inv_map.stride(0), inv_map.stride(1),
            weights.stride(0), weights.stride(1),
            grad_eo.stride(0), grad_eo.stride(1),
            grad_w.stride(0), grad_w.stride(1),
            TOP_K=top_k, TOP_K_PAD=TOP_K_PAD, FP32_ACCUM=fp32_accum,
        )
        _moe_autotune_status_end(kernel_name="moe_gc_bwd", autotuner=autotuner, state=status)
    else:
        cc_major = torch.cuda.get_device_capability()[0]
        if C < 128:
            BLOCK_D = k.triton.next_power_of_2(C)
            num_warps = 2
        elif cc_major >= 9:  # SM90 (Hopper), SM120 (Blackwell)
            BLOCK_D, num_warps = 128, 4
        else:  # SM80 (Ampere) and older
            BLOCK_D, num_warps = 128, 4
        k._moe_gather_combine_bwd_fused_kernel[(T,)](
            grad_out, expert_out_sorted, inv_map, weights, grad_eo, grad_w,
            scale, C,
            grad_out.stride(0), grad_out.stride(1),
            expert_out_sorted.stride(0), expert_out_sorted.stride(1),
            inv_map.stride(0), inv_map.stride(1),
            weights.stride(0), weights.stride(1),
            grad_eo.stride(0), grad_eo.stride(1),
            grad_w.stride(0), grad_w.stride(1),
            BLOCK_D=BLOCK_D, TOP_K=top_k, TOP_K_PAD=TOP_K_PAD, FP32_ACCUM=fp32_accum,
            num_warps=num_warps, num_stages=2,
        )
    return grad_eo, grad_w


@torch.library.impl(_lib, "gather_combine_bwd", "CPU")
def _gather_combine_bwd_cpu(
    grad_out, expert_out_sorted, inv_map, weights, scale, M, eo_dtype,
):
    return _gather_combine_bwd_eager(
        grad_out, expert_out_sorted, inv_map, weights, scale, M,
        _INT_TO_DTYPE[eo_dtype],
    )


# ── gather_combine autograd ──────────────────────────────────────────────────


def _gather_combine_setup_context(ctx, inputs, output):
    expert_out_sorted, inv_map, weights, scale = inputs
    ctx.save_for_backward(expert_out_sorted, inv_map, weights)
    ctx.scale = float(scale)
    ctx.eo_dtype = expert_out_sorted.dtype


def _gather_combine_backward(ctx, grad_out):
    expert_out_sorted, inv_map, weights = ctx.saved_tensors
    scale = ctx.scale
    M = expert_out_sorted.shape[0]
    eo_dtype_int = _DTYPE_TO_INT[ctx.eo_dtype]
    grad_out = grad_out.contiguous()

    grad_eo, grad_w = torch.ops.nanoplm_moe.gather_combine_bwd(
        grad_out, expert_out_sorted, inv_map, weights, scale, M, eo_dtype_int,
    )
    return grad_eo, None, grad_w, None


torch.library.register_autograd(
    "nanoplm_moe::gather_combine",
    _gather_combine_backward,
    setup_context=_gather_combine_setup_context,
)


def moe_gather_combine(
    expert_out_sorted: torch.Tensor,
    inv_map: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.ops.nanoplm_moe.gather_combine(
        expert_out_sorted, inv_map, weights, scale,
    )
