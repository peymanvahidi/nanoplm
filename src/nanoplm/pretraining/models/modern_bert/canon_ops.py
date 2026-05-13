"""Triton-accelerated varlen Canon depthwise convolution.

Registers three ``torch.library`` ops under the ``nanoplm_canon`` namespace:

  varlen_conv_fwd      — forward pass
  varlen_conv_bwd_dx   — backward: grad_x (transpose conv, flipped weights)
  varlen_conv_bwd_dw_db — backward: grad_weight + grad_bias (reduction over T)

The ops are opaque to Inductor — it cannot fuse them with neighbouring
LayerNorm, which is the root cause of the pathological backward kernel
explosion in the original roll-based Python implementation.  Activation
checkpointing (``USE_ACTIVATION_CHECKPOINTING_CANON``) continues to work
because ``torch.library.register_autograd`` hooks into the same
``saved_tensors_default_hooks`` mechanism that ``checkpoint()`` relies on.
"""

from __future__ import annotations

import math
import os
import threading
from contextlib import contextmanager
from typing import Any

import torch
import torch.library
import triton

# ── Library + op signatures ───────────────────────────────────────────────

_lib = torch.library.Library("nanoplm_canon", "DEF")

_lib.define(
    "varlen_conv_fwd(Tensor x, Tensor seq_id, Tensor weight, Tensor bias, int radius) -> Tensor"
)
_lib.define(
    "varlen_conv_bwd_dx(Tensor grad_out, Tensor seq_id, Tensor weight, int radius) -> Tensor"
)
_lib.define(
    "varlen_conv_bwd_dw_db(Tensor grad_out, Tensor x, Tensor seq_id, int radius) -> (Tensor, Tensor)"
)

# Fused LayerNorm + Conv ops
_lib.define(
    "varlen_ln_conv_fwd(Tensor x, Tensor seq_id, Tensor ln_weight, float ln_eps, "
    "Tensor conv_weight, Tensor conv_bias, int radius) -> (Tensor, Tensor, Tensor)"
)
_lib.define(
    "varlen_ln_conv_bwd_dw_db(Tensor grad_out, Tensor x, Tensor seq_id, "
    "Tensor ln_weight, Tensor mean, Tensor rstd, int radius) -> (Tensor, Tensor)"
)
_lib.define(
    "varlen_ln_bwd_dx_dgamma(Tensor grad_ln_out, Tensor x, "
    "Tensor mean, Tensor rstd, Tensor ln_weight) -> (Tensor, Tensor)"
)
_lib.define(
    "varlen_ln_conv_bwd_dx_dgamma(Tensor grad_out, Tensor x, Tensor seq_id, "
    "Tensor mean, Tensor rstd, Tensor ln_weight, Tensor conv_weight, int radius) -> (Tensor, Tensor)"
)

# Fused RMSNorm + Conv ops
_lib.define(
    "varlen_rms_conv_fwd(Tensor x, Tensor seq_id, Tensor rms_weight, float rms_eps, "
    "Tensor conv_weight, Tensor conv_bias, int radius) -> (Tensor, Tensor)"
)
_lib.define(
    "varlen_rms_conv_bwd_dw_db(Tensor grad_out, Tensor x, Tensor seq_id, "
    "Tensor rms_weight, Tensor inv_rms, int radius) -> (Tensor, Tensor)"
)
_lib.define(
    "varlen_rms_bwd_dx_dgamma(Tensor grad_rms_out, Tensor x, "
    "Tensor inv_rms, Tensor rms_weight) -> (Tensor, Tensor)"
)
_lib.define(
    "varlen_rms_conv_bwd_dx_dgamma(Tensor grad_out, Tensor x, Tensor seq_id, "
    "Tensor inv_rms, Tensor rms_weight, Tensor conv_weight, int radius) -> (Tensor, Tensor)"
)

# ── FakeTensor implementations (for torch.compile shape inference) ────────


@torch.library.register_fake("nanoplm_canon::varlen_conv_fwd")
def _fwd_fake(x, seq_id, weight, bias, radius):
    return torch.empty_like(x)


@torch.library.register_fake("nanoplm_canon::varlen_conv_bwd_dx")
def _bwd_dx_fake(grad_out, seq_id, weight, radius):
    return torch.empty_like(grad_out)


@torch.library.register_fake("nanoplm_canon::varlen_conv_bwd_dw_db")
def _bwd_dw_db_fake(grad_out, x, seq_id, radius):
    C = x.shape[1]
    K = 2 * radius + 1
    grad_w = torch.empty(C, K, dtype=torch.float32, device=x.device)
    grad_b = torch.empty(C, dtype=torch.float32, device=x.device)
    return grad_w, grad_b


@torch.library.register_fake("nanoplm_canon::varlen_ln_conv_fwd")
def _ln_conv_fwd_fake(x, seq_id, ln_weight, ln_eps, conv_weight, conv_bias, radius):
    T = x.shape[0]
    out = torch.empty_like(x)
    acc_dtype = torch.float32 if _needs_fp32_accum(x.dtype) else x.dtype
    mean = torch.empty(T, dtype=acc_dtype, device=x.device)
    rstd = torch.empty(T, dtype=acc_dtype, device=x.device)
    return out, mean, rstd


@torch.library.register_fake("nanoplm_canon::varlen_ln_conv_bwd_dw_db")
def _ln_conv_bwd_dw_db_fake(grad_out, x, seq_id, ln_weight, mean, rstd, radius):
    C = x.shape[1]
    K = 2 * radius + 1
    grad_w = torch.empty(C, K, dtype=torch.float32, device=x.device)
    grad_b = torch.empty(C, dtype=torch.float32, device=x.device)
    return grad_w, grad_b


@torch.library.register_fake("nanoplm_canon::varlen_ln_bwd_dx_dgamma")
def _ln_bwd_dx_dgamma_fake(grad_ln_out, x, mean, rstd, ln_weight):
    grad_x = torch.empty_like(x)
    grad_gamma = torch.empty_like(ln_weight)
    return grad_x, grad_gamma


@torch.library.register_fake("nanoplm_canon::varlen_ln_conv_bwd_dx_dgamma")
def _ln_conv_bwd_dx_dgamma_fake(
    grad_out, x, seq_id, mean, rstd, ln_weight, conv_weight, radius
):
    grad_x = torch.empty_like(x)
    grad_gamma = torch.empty_like(ln_weight)
    return grad_x, grad_gamma


# FakeTensor implementations for RMSNorm ops


@torch.library.register_fake("nanoplm_canon::varlen_rms_conv_fwd")
def _rms_conv_fwd_fake(x, seq_id, rms_weight, rms_eps, conv_weight, conv_bias, radius):
    T = x.shape[0]
    out = torch.empty_like(x)
    acc_dtype = torch.float32 if _needs_fp32_accum(x.dtype) else x.dtype
    inv_rms = torch.empty(T, dtype=acc_dtype, device=x.device)
    return out, inv_rms


@torch.library.register_fake("nanoplm_canon::varlen_rms_conv_bwd_dw_db")
def _rms_conv_bwd_dw_db_fake(grad_out, x, seq_id, rms_weight, inv_rms, radius):
    C = x.shape[1]
    K = 2 * radius + 1
    grad_w = torch.empty(C, K, dtype=torch.float32, device=x.device)
    grad_b = torch.empty(C, dtype=torch.float32, device=x.device)
    return grad_w, grad_b


@torch.library.register_fake("nanoplm_canon::varlen_rms_bwd_dx_dgamma")
def _rms_bwd_dx_dgamma_fake(grad_rms_out, x, inv_rms, rms_weight):
    grad_x = torch.empty_like(x)
    grad_gamma = torch.empty_like(rms_weight)
    return grad_x, grad_gamma


@torch.library.register_fake("nanoplm_canon::varlen_rms_conv_bwd_dx_dgamma")
def _rms_conv_bwd_dx_dgamma_fake(
    grad_out, x, seq_id, inv_rms, rms_weight, conv_weight, radius
):
    grad_x = torch.empty_like(x)
    grad_gamma = torch.empty_like(rms_weight)
    return grad_x, grad_gamma


# ── Autotune control ─────────────────────────────────────────────────────

_AUTOTUNE_CONTROL = threading.local()


def _autotune_disable_depth() -> int:
    return int(getattr(_AUTOTUNE_CONTROL, "disable_depth", 0))


@contextmanager
def disable_autotune_temporarily():
    """Temporarily force the fallback (hardcoded block-size) launch path."""
    _AUTOTUNE_CONTROL.disable_depth = _autotune_disable_depth() + 1
    try:
        yield
    finally:
        _AUTOTUNE_CONTROL.disable_depth = max(0, _autotune_disable_depth() - 1)


def _autotune_enabled() -> bool:
    v = os.getenv("NANOPLM_CANON_TRITON_AUTOTUNE", "1").strip().lower()
    return v not in {"0", "false", "off", "no"} and _autotune_disable_depth() == 0


def _autotune_status_enabled() -> bool:
    v = os.getenv("NANOPLM_CANON_TRITON_AUTOTUNE_STATUS", "1").strip().lower()
    return v not in {"0", "false", "off", "no"}


_AUTOTUNE_STATUS_SEEN: set[tuple[str, tuple[Any, ...]]] = set()


def _autotune_status_begin(
    *,
    kernel_name: str,
    autotuner: Any,
    args_by_name: dict[str, Any],
) -> tuple[tuple[Any, ...], bool, Any] | None:
    if not _autotune_status_enabled():
        return None
    key: list[Any] = []
    for name in getattr(autotuner, "keys", ()):
        if name in args_by_name:
            key.append(args_by_name[name])
    tkey = tuple(key)
    tag = (kernel_name, tkey)
    if tag in _AUTOTUNE_STATUS_SEEN:
        return None
    _AUTOTUNE_STATUS_SEEN.add(tag)

    in_mem_hit = tkey in getattr(autotuner, "cache", {})
    bench_before = getattr(autotuner, "bench_time", None)
    if in_mem_hit:
        cfg = autotuner.cache[tkey]
        print(
            f"[nanoplm][triton] autotune cache hit: {kernel_name} key={tkey} cfg={cfg}",
            flush=True,
        )
    else:
        num_cfgs = len(getattr(autotuner, "configs", ()))
        print(
            f"[nanoplm][triton] resolving autotune config: {kernel_name} key={tkey} "
            f"(may benchmark up to {num_cfgs} configs on first run)",
            flush=True,
        )
    return tkey, in_mem_hit, bench_before


def _autotune_status_end(
    *,
    kernel_name: str,
    autotuner: Any,
    state: tuple[tuple[Any, ...], bool, Any] | None,
) -> None:
    if not _autotune_status_enabled() or state is None:
        return
    tkey, in_mem_hit, bench_before = state
    if in_mem_hit:
        return
    bench_after = getattr(autotuner, "bench_time", None)
    ran_benchmark = bench_after is not None and bench_after != bench_before
    cfg = getattr(autotuner, "cache", {}).get(
        tkey, getattr(autotuner, "best_config", None)
    )
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


# ── CUDA implementations ─────────────────────────────────────────────────

# Fallback per-kernel block sizes tuned on RTX 5090 (T=65536, C=768, K=7, bf16).
_FWD_BLOCK_T, _FWD_BLOCK_C = 64, 128  # 78% roofline
_BWD_DX_BLOCK_T, _BWD_DX_BLOCK_C = 32, 128  # 83% roofline
_BWD_DW_BLOCK_T, _BWD_DW_BLOCK_C = 256, 64  # 48% roofline (atomic_add bound)


def _needs_fp32_accum(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16)


@torch.library.impl(_lib, "varlen_conv_fwd", "CUDA")
def _fwd_cuda(x, seq_id, weight, bias, radius):
    from . import canon_triton_kernels as k

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    T, C = x.shape
    fp32_acc = _needs_fp32_accum(x.dtype)

    out = torch.empty_like(x)
    if _autotune_enabled():
        autotuner = k._canon_fwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_fwd",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (
            triton.cdiv(T, META["BLOCK_T"]),
            triton.cdiv(C, META["BLOCK_C"]),
        )
        autotuner[grid](
            x,
            seq_id,
            weight,
            bias,
            out,
            T,
            C,
            x.stride(0),
            x.stride(1),
            RADIUS=radius,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(kernel_name="canon_fwd", autotuner=autotuner, state=status)
    else:
        grid = (triton.cdiv(T, _FWD_BLOCK_T), triton.cdiv(C, _FWD_BLOCK_C))
        k._canon_fwd_kernel[grid](
            x,
            seq_id,
            weight,
            bias,
            out,
            T,
            C,
            x.stride(0),
            x.stride(1),
            RADIUS=radius,
            BLOCK_T=_FWD_BLOCK_T,
            BLOCK_C=_FWD_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    return out


@torch.library.impl(_lib, "varlen_conv_bwd_dx", "CUDA")
def _bwd_dx_cuda(grad_out, seq_id, weight, radius):
    from . import canon_triton_kernels as k

    grad_out = grad_out.contiguous()
    weight = weight.contiguous()
    T, C = grad_out.shape
    fp32_acc = _needs_fp32_accum(grad_out.dtype)

    grad_x = torch.empty_like(grad_out)
    if _autotune_enabled():
        autotuner = k._canon_bwd_dx_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_bwd_dx",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (
            triton.cdiv(T, META["BLOCK_T"]),
            triton.cdiv(C, META["BLOCK_C"]),
        )
        autotuner[grid](
            grad_out,
            seq_id,
            weight,
            grad_x,
            T,
            C,
            grad_out.stride(0),
            grad_out.stride(1),
            RADIUS=radius,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_bwd_dx", autotuner=autotuner, state=status
        )
    else:
        grid = (triton.cdiv(T, _BWD_DX_BLOCK_T), triton.cdiv(C, _BWD_DX_BLOCK_C))
        k._canon_bwd_dx_kernel[grid](
            grad_out,
            seq_id,
            weight,
            grad_x,
            T,
            C,
            grad_out.stride(0),
            grad_out.stride(1),
            RADIUS=radius,
            BLOCK_T=_BWD_DX_BLOCK_T,
            BLOCK_C=_BWD_DX_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    return grad_x


@torch.library.impl(_lib, "varlen_conv_bwd_dw_db", "CUDA")
def _bwd_dw_db_cuda(grad_out, x, seq_id, radius):
    from . import canon_triton_kernels as k

    grad_out = grad_out.contiguous()
    x = x.contiguous()
    T, C = x.shape
    K = 2 * radius + 1
    fp32_acc = _needs_fp32_accum(x.dtype)

    # Accumulate weight/bias grads in fp32 for half-precision, native dtype otherwise.
    acc_dtype = torch.float32 if fp32_acc else x.dtype
    grad_w = torch.zeros(C, K, dtype=acc_dtype, device=x.device)
    grad_b = torch.zeros(C, dtype=acc_dtype, device=x.device)

    if _autotune_enabled():
        autotuner = k._canon_bwd_dw_db_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_bwd_dw_db",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (
            triton.cdiv(T, META["BLOCK_T"]),
            triton.cdiv(C, META["BLOCK_C"]),
        )
        autotuner[grid](
            grad_out,
            x,
            seq_id,
            grad_w,
            grad_b,
            T,
            C,
            grad_out.stride(0),
            grad_out.stride(1),
            RADIUS=radius,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_bwd_dw_db", autotuner=autotuner, state=status
        )
    else:
        grid = (triton.cdiv(T, _BWD_DW_BLOCK_T), triton.cdiv(C, _BWD_DW_BLOCK_C))
        k._canon_bwd_dw_db_kernel[grid](
            grad_out,
            x,
            seq_id,
            grad_w,
            grad_b,
            T,
            C,
            grad_out.stride(0),
            grad_out.stride(1),
            RADIUS=radius,
            BLOCK_T=_BWD_DW_BLOCK_T,
            BLOCK_C=_BWD_DW_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    return grad_w, grad_b


# ── Fused LN+Conv CUDA implementations ────────────────────────────────────

_LN_FWD_BLOCK_T, _LN_FWD_BLOCK_C = 64, 128
_LN_BWD_DW_BLOCK_T, _LN_BWD_DW_BLOCK_C = 256, 64


@torch.library.impl(_lib, "varlen_ln_conv_fwd", "CUDA")
def _ln_conv_fwd_cuda(x, seq_id, ln_weight, ln_eps, conv_weight, conv_bias, radius):
    from . import canon_triton_kernels as k

    x = x.contiguous()
    ln_weight = ln_weight.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    T, C = x.shape
    fp32_acc = _needs_fp32_accum(x.dtype)

    # One-pass stats kernel: single Triton kernel replaces x.to(fp32) + mean + var
    BLOCK_N = triton.next_power_of_2(C)
    acc_dtype = torch.float32 if fp32_acc else x.dtype
    mean = torch.empty(T, dtype=acc_dtype, device=x.device)
    rstd = torch.empty(T, dtype=acc_dtype, device=x.device)
    if _autotune_enabled():
        autotuner = k._ln_stats_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_ln_stats",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        autotuner[(T,)](
            x,
            mean,
            rstd,
            T,
            C,
            x.stride(0),
            x.stride(1),
            ln_eps,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_ln_stats", autotuner=autotuner, state=status
        )
    else:
        k._ln_stats_kernel[(T,)](
            x,
            mean,
            rstd,
            T,
            C,
            x.stride(0),
            x.stride(1),
            ln_eps,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
        )

    out = torch.empty_like(x)
    if _autotune_enabled():
        autotuner = k._canon_ln_fwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_ln_fwd",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (
            triton.cdiv(T, META["BLOCK_T"]),
            triton.cdiv(C, META["BLOCK_C"]),
        )
        autotuner[grid](
            x,
            seq_id,
            mean,
            rstd,
            ln_weight,
            conv_weight,
            conv_bias,
            out,
            T,
            C,
            x.stride(0),
            x.stride(1),
            RADIUS=radius,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_ln_fwd", autotuner=autotuner, state=status
        )
    else:
        grid = (triton.cdiv(T, _LN_FWD_BLOCK_T), triton.cdiv(C, _LN_FWD_BLOCK_C))
        k._canon_ln_fwd_kernel[grid](
            x,
            seq_id,
            mean,
            rstd,
            ln_weight,
            conv_weight,
            conv_bias,
            out,
            T,
            C,
            x.stride(0),
            x.stride(1),
            RADIUS=radius,
            BLOCK_T=_LN_FWD_BLOCK_T,
            BLOCK_C=_LN_FWD_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    return out, mean, rstd


@torch.library.impl(_lib, "varlen_ln_conv_bwd_dw_db", "CUDA")
def _ln_conv_bwd_dw_db_cuda(grad_out, x, seq_id, ln_weight, mean, rstd, radius):
    from . import canon_triton_kernels as k

    grad_out = grad_out.contiguous()
    x = x.contiguous()
    ln_weight = ln_weight.contiguous()
    T, C = x.shape
    K = 2 * radius + 1
    fp32_acc = _needs_fp32_accum(x.dtype)

    acc_dtype = torch.float32 if fp32_acc else x.dtype
    # Partial-buffer reduction: each T-block writes its partials, host sums.
    _MIN_BLOCK_T = 64  # smallest BLOCK_T in autotune configs
    max_t_blocks = triton.cdiv(T, _MIN_BLOCK_T)
    partial_w = torch.zeros(max_t_blocks, C * K, dtype=acc_dtype, device=x.device)
    partial_b = torch.zeros(max_t_blocks, C, dtype=acc_dtype, device=x.device)

    if _autotune_enabled():
        autotuner = k._canon_ln_bwd_dw_db_partial_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_ln_bwd_dw_db",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (
            triton.cdiv(T, META["BLOCK_T"]),
            triton.cdiv(C, META["BLOCK_C"]),
        )
        autotuner[grid](
            grad_out,
            x,
            seq_id,
            mean,
            rstd,
            ln_weight,
            partial_w,
            partial_b,
            T,
            C,
            grad_out.stride(0),
            grad_out.stride(1),
            RADIUS=radius,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_ln_bwd_dw_db", autotuner=autotuner, state=status
        )
    else:
        grid = (triton.cdiv(T, _LN_BWD_DW_BLOCK_T), triton.cdiv(C, _LN_BWD_DW_BLOCK_C))
        k._canon_ln_bwd_dw_db_partial_kernel[grid](
            grad_out,
            x,
            seq_id,
            mean,
            rstd,
            ln_weight,
            partial_w,
            partial_b,
            T,
            C,
            grad_out.stride(0),
            grad_out.stride(1),
            RADIUS=radius,
            BLOCK_T=_LN_BWD_DW_BLOCK_T,
            BLOCK_C=_LN_BWD_DW_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    # Reduce partial buffers
    grad_w = partial_w.view(max_t_blocks, C, K).sum(0)
    grad_b = partial_b.sum(0)
    return grad_w, grad_b


@torch.library.impl(_lib, "varlen_ln_bwd_dx_dgamma", "CUDA")
def _ln_bwd_dx_dgamma_cuda(grad_ln_out, x, mean, rstd, ln_weight):
    from . import canon_triton_kernels as k

    grad_ln_out = grad_ln_out.contiguous()
    x = x.contiguous()
    ln_weight = ln_weight.contiguous()
    T, C = x.shape
    BLOCK_N = triton.next_power_of_2(C)
    fp32_acc = _needs_fp32_accum(x.dtype)
    acc_dtype = torch.float32 if fp32_acc else x.dtype

    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count * 8
    rows_per_program = math.ceil(T / sm_count)

    grad_x = torch.empty_like(x)
    partial_dgamma = torch.zeros(sm_count, C, dtype=acc_dtype, device=x.device)

    if _autotune_enabled():
        autotuner = k._ln_bwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_ln_bwd",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        autotuner[(sm_count,)](
            grad_ln_out,
            x,
            mean,
            rstd,
            ln_weight,
            grad_x,
            partial_dgamma,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rows_per_program,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_ln_bwd", autotuner=autotuner, state=status
        )
    else:
        k._ln_bwd_kernel[(sm_count,)](
            grad_ln_out,
            x,
            mean,
            rstd,
            ln_weight,
            grad_x,
            partial_dgamma,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rows_per_program,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
        )

    grad_gamma = partial_dgamma.sum(0).to(ln_weight.dtype)
    return grad_x, grad_gamma


@torch.library.impl(_lib, "varlen_ln_conv_bwd_dx_dgamma", "CUDA")
def _ln_conv_bwd_dx_dgamma_cuda(
    grad_out, x, seq_id, mean, rstd, ln_weight, conv_weight, radius
):
    """Fused conv_bwd_dx + add + ln_bwd in a single kernel launch."""
    from . import canon_triton_kernels as k

    grad_out = grad_out.contiguous()
    x = x.contiguous()
    ln_weight = ln_weight.contiguous()
    conv_weight = conv_weight.contiguous()
    T, C = x.shape
    BLOCK_N = triton.next_power_of_2(C)
    fp32_acc = _needs_fp32_accum(x.dtype)
    acc_dtype = torch.float32 if fp32_acc else x.dtype

    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count * 8
    rows_per_program = math.ceil(T / sm_count)

    grad_x = torch.empty_like(x)
    partial_dgamma = torch.zeros(sm_count, C, dtype=acc_dtype, device=x.device)

    if _autotune_enabled():
        autotuner = k._fused_conv_bwd_dx_ln_bwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_fused_conv_ln_bwd",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        autotuner[(sm_count,)](
            grad_out,
            x,
            seq_id,
            mean,
            rstd,
            ln_weight,
            conv_weight,
            grad_x,
            partial_dgamma,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rows_per_program,
            RADIUS=radius,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_fused_conv_ln_bwd", autotuner=autotuner, state=status
        )
    else:
        k._fused_conv_bwd_dx_ln_bwd_kernel[(sm_count,)](
            grad_out,
            x,
            seq_id,
            mean,
            rstd,
            ln_weight,
            conv_weight,
            grad_x,
            partial_dgamma,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rows_per_program,
            RADIUS=radius,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
        )

    grad_gamma = partial_dgamma.sum(0).to(ln_weight.dtype)
    return grad_x, grad_gamma


# ── Fused RMSNorm+Conv CUDA implementations ───────────────────────────────

_RMS_FWD_BLOCK_T, _RMS_FWD_BLOCK_C = 64, 128
_RMS_BWD_DW_BLOCK_T, _RMS_BWD_DW_BLOCK_C = 256, 64


@torch.library.impl(_lib, "varlen_rms_conv_fwd", "CUDA")
def _rms_conv_fwd_cuda(x, seq_id, rms_weight, rms_eps, conv_weight, conv_bias, radius):
    from . import canon_triton_kernels as k

    x = x.contiguous()
    rms_weight = rms_weight.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    T, C = x.shape
    fp32_acc = _needs_fp32_accum(x.dtype)

    # One-pass RMS stats kernel
    BLOCK_N = triton.next_power_of_2(C)
    acc_dtype = torch.float32 if fp32_acc else x.dtype
    inv_rms = torch.empty(T, dtype=acc_dtype, device=x.device)
    if _autotune_enabled():
        autotuner = k._rms_stats_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_rms_stats",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        autotuner[(T,)](
            x,
            inv_rms,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rms_eps,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_rms_stats", autotuner=autotuner, state=status
        )
    else:
        k._rms_stats_kernel[(T,)](
            x,
            inv_rms,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rms_eps,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
        )

    out = torch.empty_like(x)
    if _autotune_enabled():
        autotuner = k._canon_rms_fwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_rms_fwd",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (
            triton.cdiv(T, META["BLOCK_T"]),
            triton.cdiv(C, META["BLOCK_C"]),
        )
        autotuner[grid](
            x,
            seq_id,
            inv_rms,
            rms_weight,
            conv_weight,
            conv_bias,
            out,
            T,
            C,
            x.stride(0),
            x.stride(1),
            RADIUS=radius,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_rms_fwd", autotuner=autotuner, state=status
        )
    else:
        grid = (triton.cdiv(T, _RMS_FWD_BLOCK_T), triton.cdiv(C, _RMS_FWD_BLOCK_C))
        k._canon_rms_fwd_kernel[grid](
            x,
            seq_id,
            inv_rms,
            rms_weight,
            conv_weight,
            conv_bias,
            out,
            T,
            C,
            x.stride(0),
            x.stride(1),
            RADIUS=radius,
            BLOCK_T=_RMS_FWD_BLOCK_T,
            BLOCK_C=_RMS_FWD_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    return out, inv_rms


@torch.library.impl(_lib, "varlen_rms_conv_bwd_dw_db", "CUDA")
def _rms_conv_bwd_dw_db_cuda(grad_out, x, seq_id, rms_weight, inv_rms, radius):
    from . import canon_triton_kernels as k

    grad_out = grad_out.contiguous()
    x = x.contiguous()
    rms_weight = rms_weight.contiguous()
    T, C = x.shape
    K = 2 * radius + 1
    fp32_acc = _needs_fp32_accum(x.dtype)

    acc_dtype = torch.float32 if fp32_acc else x.dtype
    _MIN_BLOCK_T = 64
    max_t_blocks = triton.cdiv(T, _MIN_BLOCK_T)
    partial_w = torch.zeros(max_t_blocks, C * K, dtype=acc_dtype, device=x.device)
    partial_b = torch.zeros(max_t_blocks, C, dtype=acc_dtype, device=x.device)

    if _autotune_enabled():
        autotuner = k._canon_rms_bwd_dw_db_partial_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_rms_bwd_dw_db",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        grid = lambda META: (
            triton.cdiv(T, META["BLOCK_T"]),
            triton.cdiv(C, META["BLOCK_C"]),
        )
        autotuner[grid](
            grad_out,
            x,
            seq_id,
            inv_rms,
            rms_weight,
            partial_w,
            partial_b,
            T,
            C,
            grad_out.stride(0),
            grad_out.stride(1),
            RADIUS=radius,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_rms_bwd_dw_db", autotuner=autotuner, state=status
        )
    else:
        grid = (
            triton.cdiv(T, _RMS_BWD_DW_BLOCK_T),
            triton.cdiv(C, _RMS_BWD_DW_BLOCK_C),
        )
        k._canon_rms_bwd_dw_db_partial_kernel[grid](
            grad_out,
            x,
            seq_id,
            inv_rms,
            rms_weight,
            partial_w,
            partial_b,
            T,
            C,
            grad_out.stride(0),
            grad_out.stride(1),
            RADIUS=radius,
            BLOCK_T=_RMS_BWD_DW_BLOCK_T,
            BLOCK_C=_RMS_BWD_DW_BLOCK_C,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
            num_stages=2,
        )
    grad_w = partial_w.view(max_t_blocks, C, K).sum(0)
    grad_b = partial_b.sum(0)
    return grad_w, grad_b


@torch.library.impl(_lib, "varlen_rms_bwd_dx_dgamma", "CUDA")
def _rms_bwd_dx_dgamma_cuda(grad_rms_out, x, inv_rms, rms_weight):
    from . import canon_triton_kernels as k

    grad_rms_out = grad_rms_out.contiguous()
    x = x.contiguous()
    rms_weight = rms_weight.contiguous()
    T, C = x.shape
    BLOCK_N = triton.next_power_of_2(C)
    fp32_acc = _needs_fp32_accum(x.dtype)
    acc_dtype = torch.float32 if fp32_acc else x.dtype

    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count * 8
    rows_per_program = math.ceil(T / sm_count)

    grad_x = torch.empty_like(x)
    partial_dgamma = torch.zeros(sm_count, C, dtype=acc_dtype, device=x.device)

    if _autotune_enabled():
        autotuner = k._rms_bwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_rms_bwd",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        autotuner[(sm_count,)](
            grad_rms_out,
            x,
            inv_rms,
            rms_weight,
            grad_x,
            partial_dgamma,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rows_per_program,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_rms_bwd", autotuner=autotuner, state=status
        )
    else:
        k._rms_bwd_kernel[(sm_count,)](
            grad_rms_out,
            x,
            inv_rms,
            rms_weight,
            grad_x,
            partial_dgamma,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rows_per_program,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
        )

    grad_gamma = partial_dgamma.sum(0).to(rms_weight.dtype)
    return grad_x, grad_gamma


@torch.library.impl(_lib, "varlen_rms_conv_bwd_dx_dgamma", "CUDA")
def _rms_conv_bwd_dx_dgamma_cuda(
    grad_out, x, seq_id, inv_rms, rms_weight, conv_weight, radius
):
    """Fused conv_bwd_dx + add + rms_bwd in a single kernel launch."""
    from . import canon_triton_kernels as k

    grad_out = grad_out.contiguous()
    x = x.contiguous()
    rms_weight = rms_weight.contiguous()
    conv_weight = conv_weight.contiguous()
    T, C = x.shape
    BLOCK_N = triton.next_power_of_2(C)
    fp32_acc = _needs_fp32_accum(x.dtype)
    acc_dtype = torch.float32 if fp32_acc else x.dtype

    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count * 8
    rows_per_program = math.ceil(T / sm_count)

    grad_x = torch.empty_like(x)
    partial_dgamma = torch.zeros(sm_count, C, dtype=acc_dtype, device=x.device)

    if _autotune_enabled():
        autotuner = k._fused_conv_bwd_dx_rms_bwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="canon_fused_conv_rms_bwd",
            autotuner=autotuner,
            args_by_name={"T": T, "C": C},
        )
        autotuner[(sm_count,)](
            grad_out,
            x,
            seq_id,
            inv_rms,
            rms_weight,
            conv_weight,
            grad_x,
            partial_dgamma,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rows_per_program,
            RADIUS=radius,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
        )
        _autotune_status_end(
            kernel_name="canon_fused_conv_rms_bwd", autotuner=autotuner, state=status
        )
    else:
        k._fused_conv_bwd_dx_rms_bwd_kernel[(sm_count,)](
            grad_out,
            x,
            seq_id,
            inv_rms,
            rms_weight,
            conv_weight,
            grad_x,
            partial_dgamma,
            T,
            C,
            x.stride(0),
            x.stride(1),
            rows_per_program,
            RADIUS=radius,
            BLOCK_N=BLOCK_N,
            FP32_ACCUM=fp32_acc,
            num_warps=4,
        )

    grad_gamma = partial_dgamma.sum(0).to(rms_weight.dtype)
    return grad_x, grad_gamma


def _setup_context(ctx, inputs, output):
    x, seq_id, weight, _bias, radius = inputs
    ctx.save_for_backward(x, seq_id, weight)
    ctx.radius = int(radius)


def _backward(ctx, grad_out):
    x, seq_id, weight = ctx.saved_tensors
    radius = ctx.radius
    grad_out = grad_out.contiguous()

    grad_x = torch.ops.nanoplm_canon.varlen_conv_bwd_dx(
        grad_out, seq_id, weight, radius
    )
    grad_w, grad_b = torch.ops.nanoplm_canon.varlen_conv_bwd_dw_db(
        grad_out, x, seq_id, radius
    )
    return grad_x, None, grad_w, grad_b, None


torch.library.register_autograd(
    "nanoplm_canon::varlen_conv_fwd",
    _backward,
    setup_context=_setup_context,
)


# ── Fused LN+Conv autograd glue ──────────────────────────────────────────


def _ln_conv_setup_context(ctx, inputs, output):
    x, seq_id, ln_weight, ln_eps, conv_weight, _conv_bias, radius = inputs
    _out, mean, rstd = output
    ctx.save_for_backward(x, seq_id, ln_weight, conv_weight, mean, rstd)
    ctx.radius = int(radius)
    ctx.ln_eps = float(ln_eps)


def _ln_conv_backward(ctx, grad_out, _grad_mean, _grad_rstd):
    x, seq_id, ln_weight, conv_weight, mean, rstd = ctx.saved_tensors
    radius = ctx.radius
    grad_out = grad_out.contiguous()

    # 1+2+3 fused: conv_bwd_dx + residual add + LN backward (single kernel)
    grad_x, grad_gamma = torch.ops.nanoplm_canon.varlen_ln_conv_bwd_dx_dgamma(
        grad_out, x, seq_id, mean, rstd, ln_weight, conv_weight, radius
    )

    # 4. Conv weight/bias grads (partial-buffer Triton kernel, no atomics)
    grad_conv_w, grad_conv_b = torch.ops.nanoplm_canon.varlen_ln_conv_bwd_dw_db(
        grad_out, x, seq_id, ln_weight, mean, rstd, radius
    )

    # Return grads for: x, seq_id, ln_weight, ln_eps, conv_weight, conv_bias, radius
    return grad_x, None, grad_gamma, None, grad_conv_w, grad_conv_b, None


torch.library.register_autograd(
    "nanoplm_canon::varlen_ln_conv_fwd",
    _ln_conv_backward,
    setup_context=_ln_conv_setup_context,
)


# ── Fused RMS+Conv autograd glue ──────────────────────────────────────────


def _rms_conv_setup_context(ctx, inputs, output):
    x, seq_id, rms_weight, rms_eps, conv_weight, _conv_bias, radius = inputs
    _out, inv_rms = output
    ctx.save_for_backward(x, seq_id, rms_weight, conv_weight, inv_rms)
    ctx.radius = int(radius)
    ctx.rms_eps = float(rms_eps)


def _rms_conv_backward(ctx, grad_out, _grad_inv_rms):
    x, seq_id, rms_weight, conv_weight, inv_rms = ctx.saved_tensors
    radius = ctx.radius
    grad_out = grad_out.contiguous()

    # 1+2+3 fused: conv_bwd_dx + residual add + RMS backward (single kernel)
    grad_x, grad_gamma = torch.ops.nanoplm_canon.varlen_rms_conv_bwd_dx_dgamma(
        grad_out, x, seq_id, inv_rms, rms_weight, conv_weight, radius
    )

    # 4. Conv weight/bias grads (partial-buffer Triton kernel, no atomics)
    grad_conv_w, grad_conv_b = torch.ops.nanoplm_canon.varlen_rms_conv_bwd_dw_db(
        grad_out, x, seq_id, rms_weight, inv_rms, radius
    )

    # Return grads for: x, seq_id, rms_weight, rms_eps, conv_weight, conv_bias, radius
    return grad_x, None, grad_gamma, None, grad_conv_w, grad_conv_b, None


torch.library.register_autograd(
    "nanoplm_canon::varlen_rms_conv_fwd",
    _rms_conv_backward,
    setup_context=_rms_conv_setup_context,
)


# ── Public API ────────────────────────────────────────────────────────────


def varlen_canon_conv(
    x: torch.Tensor,
    seq_id: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Triton-accelerated varlen depthwise Canon convolution.

    Drop-in replacement for ``_varlen_canon_inner``.

    Args:
        x: ``(T, C)`` packed token features.
        seq_id: ``(T,)`` int tensor mapping each token to its sequence index.
        weight: ``(C, K)`` depthwise conv weights.
        bias: ``(C,)`` conv bias.
        radius: Conv half-width (``kernel_size // 2``).
    """
    return torch.ops.nanoplm_canon.varlen_conv_fwd(x, seq_id, weight, bias, radius)


def varlen_ln_canon_conv(
    x: torch.Tensor,
    seq_id: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_eps: float,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Fused LayerNorm + Canon varlen depthwise convolution.

    Computes ``LN(x) + conv(LN(x))`` without materializing the intermediate
    LN output tensor, saving one (T, C) write+read.

    Args:
        x: ``(T, C)`` raw (un-normalized) packed token features.
        seq_id: ``(T,)`` int tensor mapping each token to its sequence index.
        ln_weight: ``(C,)`` LayerNorm gamma (scale parameter).
        ln_eps: LayerNorm epsilon.
        conv_weight: ``(C, K)`` depthwise conv weights.
        conv_bias: ``(C,)`` conv bias.
        radius: Conv half-width (``kernel_size // 2``).
    """
    out, _mean, _rstd = torch.ops.nanoplm_canon.varlen_ln_conv_fwd(
        x, seq_id, ln_weight, ln_eps, conv_weight, conv_bias, radius
    )
    return out


def varlen_rms_canon_conv(
    x: torch.Tensor,
    seq_id: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Fused RMSNorm + Canon varlen depthwise convolution.

    Computes ``RMS(x) + conv(RMS(x))`` without materializing the intermediate
    RMS output tensor, saving one (T, C) write+read.

    Args:
        x: ``(T, C)`` raw (un-normalized) packed token features.
        seq_id: ``(T,)`` int tensor mapping each token to its sequence index.
        rms_weight: ``(C,)`` RMSNorm gamma (scale parameter).
        rms_eps: RMSNorm epsilon.
        conv_weight: ``(C, K)`` depthwise conv weights.
        conv_bias: ``(C,)`` conv bias.
        radius: Conv half-width (``kernel_size // 2``).
    """
    out, _inv_rms = torch.ops.nanoplm_canon.varlen_rms_conv_fwd(
        x, seq_id, rms_weight, rms_eps, conv_weight, conv_bias, radius
    )
    return out
