"""Triton kernels for MoE dispatch and combine.

Split out of ``moe_triton_ops.py`` so that module can import cleanly without
Triton installed. Triton ships no macOS wheels and is absent on CPU-only
installs, so nothing at module scope here may be reached unless a CUDA path
actually runs. ``moe_triton_ops.py`` reaches this module only via
``from . import moe_triton_kernels as k`` inside CUDA-only function bodies
(mirrors the ``mhc_triton_ops.py`` / ``mhc_triton_kernels.py`` split).
"""

from __future__ import annotations

import triton
import triton.language as tl


# Forward: register-light (load bf16 row, fp32 accumulate, store fp32).
_MOE_GC_FWD_CONFIGS = [
    triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=2),   # SM90: large tile, deep pipeline
    triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=2),   # SM80: same tile, fewer warps
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=2),   # SM120: validated on RTX 5090
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
]

# Backward: heavier (go + go_acc + row + row_acc + val + grad_w_acc live).
_MOE_GC_BWD_CONFIGS = [
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=2),   # SM90: aggressive
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),   # SM120: validated on RTX 5090
    triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=2),    # SM80: conservative
    triton.Config({"BLOCK_D": 64}, num_warps=2, num_stages=1),
]


# ── Triton kernels ────────────────────────────────────────────────────────────


@triton.jit
def _moe_gather_kernel(
    x_ptr,
    idx_ptr,
    y_ptr,
    m,
    d,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather rows: y[i] = x[idx[i]].  Used by scatter_dispatch forward."""
    pid_m = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < m) & (offs_d[None, :] < d)

    src_rows = tl.load(idx_ptr + offs_m, mask=offs_m < m, other=0).to(tl.int64)
    x_ptrs = x_ptr + src_rows[:, None] * stride_xm + offs_d[None, :] * stride_xd
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd
    vals = tl.load(x_ptrs, mask=mask, other=0)
    tl.store(y_ptrs, vals, mask=mask)


@triton.jit
def _moe_scatter_add_kernel(
    grad_sorted_ptr,
    token_idx_ptr,
    grad_x_ptr,
    M,
    D,
    stride_gs_m,
    stride_gs_d,
    stride_gx_m,
    stride_gx_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Scatter-add: grad_x[token_idx[i]] += grad_sorted[i] via tl.atomic_add."""
    pid_m = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)

    dst_rows = tl.load(token_idx_ptr + offs_m, mask=offs_m < M, other=0).to(tl.int64)
    gs_ptrs = grad_sorted_ptr + offs_m[:, None] * stride_gs_m + offs_d[None, :] * stride_gs_d
    gx_ptrs = grad_x_ptr + dst_rows[:, None] * stride_gx_m + offs_d[None, :] * stride_gx_d

    vals = tl.load(gs_ptrs, mask=mask, other=0.0)
    tl.atomic_add(gx_ptrs, vals, mask=mask)


@triton.jit
def _moe_gather_combine_fwd_kernel(
    eo_ptr, inv_ptr, w_ptr, out_ptr,
    scale,
    C,
    stride_eo_m, stride_eo_d,
    stride_inv_t, stride_inv_k,
    stride_w_t, stride_w_k,
    stride_out_t, stride_out_d,
    BLOCK_D: tl.constexpr,
    TOP_K: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    t = tl.program_id(0)

    for d_tile in range(0, tl.cdiv(C, BLOCK_D)):
        offs_d = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < C
        acc = tl.zeros([BLOCK_D], dtype=tl.float32 if FP32_ACCUM else tl.float64)

        for k in range(TOP_K):
            pos_k = tl.load(inv_ptr + t * stride_inv_t + k * stride_inv_k).to(tl.int64)
            weight_k = tl.load(w_ptr + t * stride_w_t + k * stride_w_k)
            row = tl.load(
                eo_ptr + pos_k * stride_eo_m + offs_d * stride_eo_d,
                mask=mask_d, other=0.0,
            )
            if FP32_ACCUM:
                row = row.to(tl.float32)
            acc += weight_k * row

        acc = acc * scale
        tl.store(
            out_ptr + t * stride_out_t + offs_d * stride_out_d, acc, mask=mask_d,
        )


@triton.jit
def _moe_gather_combine_bwd_fused_kernel(
    go_ptr, eo_ptr, inv_ptr, w_ptr, ge_ptr, gw_ptr,
    scale,
    C,
    stride_go_t, stride_go_d,
    stride_eo_m, stride_eo_d,
    stride_inv_t, stride_inv_k,
    stride_w_t, stride_w_k,
    stride_ge_m, stride_ge_d,
    stride_gw_t, stride_gw_k,
    BLOCK_D: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    FP32_ACCUM: tl.constexpr,
):
    """Fused backward: computes both grad_expert_sorted and grad_weights.

    Loads grad_out[t] once, then for each slot k:
      - grad_eo[inv_map[t,k]] = grad_out[t] * weights[t,k] * scale
      - grad_w[t,k] = dot(grad_out[t], expert_out[inv_map[t,k]]) * scale
    """
    t = tl.program_id(0)
    offs_k = tl.arange(0, TOP_K_PAD)
    mask_k = offs_k < TOP_K
    grad_w_acc = tl.zeros([TOP_K_PAD], dtype=tl.float32 if FP32_ACCUM else tl.float64)

    for d_tile in range(0, tl.cdiv(C, BLOCK_D)):
        offs_d = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < C

        # Load grad_out[t] once per D tile and reuse it for all top-k slots.
        go = tl.load(
            go_ptr + t * stride_go_t + offs_d * stride_go_d, mask=mask_d, other=0.0,
        )
        go_acc = go.to(tl.float32) if FP32_ACCUM else go

        for k in range(TOP_K):
            pos_k = tl.load(inv_ptr + t * stride_inv_t + k * stride_inv_k).to(tl.int64)
            weight_k = tl.load(w_ptr + t * stride_w_t + k * stride_w_k)
            row = tl.load(
                eo_ptr + pos_k * stride_eo_m + offs_d * stride_eo_d,
                mask=mask_d, other=0.0,
            )
            row_acc = row.to(tl.float32) if FP32_ACCUM else row

            # grad_eo[pos] = grad_out * weight * scale.
            val = go * weight_k * scale
            tl.store(
                ge_ptr + pos_k * stride_ge_m + offs_d * stride_ge_d,
                val,
                mask=mask_d,
            )

            # Accumulate the grad_w dot product across D tiles without leaving
            # the fused backward kernel.
            dot = tl.sum(go_acc * row_acc, axis=0)
            grad_w_acc += tl.where(offs_k == k, dot, 0.0)

    tl.store(
        gw_ptr + t * stride_gw_t + offs_k * stride_gw_k,
        grad_w_acc * scale,
        mask=mask_k,
    )


# ── Autotuned kernel wrappers ────────────────────────────────────────────────
# Wrap the same @triton.jit kernels with triton.autotune to benchmark
# BLOCK_D / num_warps / num_stages.  Other constexprs (TOP_K, FP32_ACCUM,
# TOP_K_PAD) are passed through unchanged at call time.

_moe_gather_combine_fwd_at = triton.autotune(
    configs=_MOE_GC_FWD_CONFIGS, key=["C"], cache_results=True,
)(_moe_gather_combine_fwd_kernel)

_moe_gather_combine_bwd_at = triton.autotune(
    configs=_MOE_GC_BWD_CONFIGS, key=["C"], cache_results=True,
)(_moe_gather_combine_bwd_fused_kernel)
