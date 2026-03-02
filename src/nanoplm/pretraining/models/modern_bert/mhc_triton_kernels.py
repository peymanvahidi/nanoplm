"""Fused Triton kernels for mHC-lite residual connections.

Implements fused kernels (K3, K4) that replace separate PyTorch ops.
K2 (coefficient computation) stays in PyTorch since it operates on only
32 values per token — the overhead is in launch, not compute.

Design follows the same persistent-scheduling pattern as triton_kernels.py.

K4 backward uses a single fused kernel; split backward kernels were removed.
"""

import torch
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════════
# Hardware detection (shared with triton_kernels.py)
# ═══════════════════════════════════════════════════════════════════════════
_HW_CONFIG = None

def _get_hw_config():
    global _HW_CONFIG
    if _HW_CONFIG is None:
        props = torch.cuda.get_device_properties("cuda")
        num_sms = props.multi_processor_count
        cc = (props.major, props.minor)
        
        # SM90 (Hopper) has 227KB shared memory per block
        if cc == (9, 0):
            # (num_sms, num_warps, num_stages)
            _HW_CONFIG = (num_sms, 8, 4)
        elif cc[0] >= 12 or cc == (8, 9):
            # SM120 (Blackwell consumer/workstation: RTX 5090, RTX 6000 Blackwell) 
            # and SM89 (Ada Lovelace) have strict ~99-100KB shared memory limits per block.
            _HW_CONFIG = (num_sms, 4, 2)
        else:
            # Safe default for older/untested architectures
            _HW_CONFIG = (num_sms, 4, 2)
            
    return _HW_CONFIG


def _autotune_configs_block_k(
    *,
    block_ts: tuple[int, ...],
    block_ks: tuple[int, ...],
    warps: tuple[int, ...],
    stages: tuple[int, ...],
    max_configs: int = 32,
) -> list[triton.Config]:
    cfgs = [
        triton.Config(
            {"BLOCK_T": bt, "BLOCK_K": bk},
            num_warps=nw,
            num_stages=ns,
        )
        for bt in block_ts
        for bk in block_ks
        for nw in warps
        for ns in stages
    ]
    return cfgs[:max_configs]


def _autotune_configs_block_c(
    *,
    block_ts: tuple[int, ...],
    block_cs: tuple[int, ...],
    warps: tuple[int, ...],
    stages: tuple[int, ...],
    max_configs: int = 32,
) -> list[triton.Config]:
    cfgs = [
        triton.Config(
            {"BLOCK_T": bt, "BLOCK_C": bc},
            num_warps=nw,
            num_stages=ns,
        )
        for bt in block_ts
        for bc in block_cs
        for nw in warps
        for ns in stages
    ]
    return cfgs[:max_configs]


_AUTOTUNE_K1_FWD_CONFIGS = _autotune_configs_block_k(
    block_ts=(64, 128),
    block_ks=(32, 64, 128, 256),
    warps=(4, 8),
    stages=(2, 3),
)
_AUTOTUNE_K1_BWD_DX_CONFIGS = _autotune_configs_block_k(
    block_ts=(64, 128),
    block_ks=(32, 64, 128, 256),
    warps=(4, 8),
    stages=(2, 3),
)
_AUTOTUNE_K3_FWD_CONFIGS = _autotune_configs_block_c(
    block_ts=(32, 64, 128, 256),
    block_cs=(64, 128),
    warps=(4, 8),
    stages=(2, 3),
)
_AUTOTUNE_K3_BWD_FUSED_CONFIGS = _autotune_configs_block_c(
    block_ts=(32, 64, 128),
    block_cs=(64, 128, 256),
    warps=(4, 8),
    stages=(2, 3, 4),
)
_AUTOTUNE_K4_FWD_CONFIGS = _autotune_configs_block_c(
    block_ts=(16, 32, 64, 128),
    block_cs=(64, 128),
    warps=(4, 8),
    stages=(2, 3, 4),
)
_AUTOTUNE_K4_BWD_FUSED_CONFIGS = _autotune_configs_block_c(
    block_ts=(16, 32, 64),
    block_cs=(64, 128, 256),
    warps=(4, 8),
    stages=(2, 3),
)


# ═══════════════════════════════════════════════════════════════════════════
# Kernel 1: fused_rmsnorm_project
#
# proj_out[t, d] = (x_flat[t] / rms[t]) @ W[d]
# Dimensions: x_flat is (T, nC), W is (D_out, nC), proj_out is (T, D_out)
# D_out is tiny (e.g. 32), nC is large (e.g. 8192). 
# This is a vectorized dot-product kernel, not a standard GEMM.
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_rmsnorm_project_fwd_kernel(
    x_ptr, W_ptr,
    out_ptr, inv_rms_ptr,
    T, nC: tl.constexpr, D_out: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    t_start = pid * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    
    d_offs = tl.arange(0, D_out)
    
    sum_sq = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, D_out), dtype=tl.float32)
    
    for k_start in tl.range(0, nC, BLOCK_K):
        # Load x_tile: (BLOCK_T, BLOCK_K) via TMA
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(T, nC),
            strides=(nC, 1),
            offsets=(t_start, k_start),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0)
        )
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1))
        x_tile_f32 = x_tile.to(tl.float32)
        sum_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)
        
        # NOTE: TMA descriptors cannot be transposed. Load W in its native layout
        # (D_out, nC) and transpose in registers for the dot.
        w_block_ptr = tl.make_block_ptr(
            base=W_ptr,
            shape=(D_out, nC),
            strides=(nC, 1),
            offsets=(0, k_start),
            block_shape=(D_out, BLOCK_K),
            order=(1, 0),
        )
        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1))

        # Tensor Core matmul: (BLOCK_T, BLOCK_K) @ (BLOCK_K, D_out)
        acc += tl.dot(x_tile, w_tile.T)

    inv_rms = tl.rsqrt(sum_sq / nC + 1e-6)
    tl.store(inv_rms_ptr + t_offs, inv_rms, mask=t_mask)

    # proj_out = (x @ W_T) / rms = (x @ W_T) * inv_rms
    out_vals = acc * inv_rms[:, None]

    out_ptrs = out_ptr + t_offs[:, None] * D_out + d_offs[None, :]
    tl.store(out_ptrs, out_vals.to(tl.bfloat16), mask=t_mask[:, None])


@triton.jit
def _fused_rmsnorm_project_bwd_dx_kernel(
    x_ptr, W_ptr, grad_out_ptr, proj_out_ptr, inv_rms_ptr,
    grad_x_ptr,
    T, nC: tl.constexpr, D_out: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute ∂L/∂x_flat in a single pass!
    dx = c1 * dx_norm - scale * x
    where dx_norm = grad_proj @ W
    c1 = 1/rms.
    dot_term = sum_d(grad_proj_d * proj_out_d) * rms
    scale = dot_term / (rms^3 * nC) = sum_d(grad_proj_d * proj_out_d) / (rms^2 * nC)
    """
    pid = tl.program_id(0)
    t_start = pid * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T

    inv_rms = tl.load(inv_rms_ptr + t_offs, mask=t_mask, other=1.0)
    
    d_offs = tl.arange(0, D_out)
    go_ptrs = grad_out_ptr + t_offs[:, None] * D_out + d_offs[None, :]
    proj_ptrs = proj_out_ptr + t_offs[:, None] * D_out + d_offs[None, :]
    
    go = tl.load(go_ptrs, mask=t_mask[:, None], other=0.0)
    proj_out = tl.load(proj_ptrs, mask=t_mask[:, None], other=0.0)

    # Compute scale directly from outputs without iterating x again.
    # scale = sum_d(go_d * proj_out_d) / (rms^2 * nC) = dot * inv_rms^2 / nC
    dot_term_fast = tl.sum(go.to(tl.float32) * proj_out.to(tl.float32), axis=1)  # (BLOCK_T,)
    scale = dot_term_fast * (inv_rms * inv_rms) / nC

    # Single pass to compute dx
    for k_start in tl.range(0, nC, BLOCK_K):
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(T, nC),
            strides=(nC, 1),
            offsets=(t_start, k_start),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0)
        )
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        
        # Load W_tile: need (D_out, BLOCK_K)
        w_block_ptr = tl.make_block_ptr(
            base=W_ptr,
            shape=(D_out, nC),
            strides=(nC, 1),
            offsets=(0, k_start),
            block_shape=(D_out, BLOCK_K),
            order=(1, 0)
        )
        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1))
        
        # Tensor Core matmul: (BLOCK_T, D_out) @ (D_out, BLOCK_K)
        dx_norm_tile = tl.dot(go, w_tile)  # (BLOCK_T, BLOCK_K)

        dx_tile = inv_rms[:, None] * dx_norm_tile - scale[:, None] * x_tile
        
        gx_block_ptr = tl.make_block_ptr(
            base=grad_x_ptr,
            shape=(T, nC),
            strides=(nC, 1),
            offsets=(t_start, k_start),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0)
        )
        tl.store(gx_block_ptr, dx_tile.to(tl.bfloat16), boundary_check=(0, 1))

# ═══════════════════════════════════════════════════════════════════════════
# Kernel 4: fused_post_res — THE CRITICAL KERNEL
#
# output[t, i, c] = Σ_j H_merged[t,i,j] * x[t,j,c]  +  h_post[t,i] * lo[t,c]
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_post_res_fwd_kernel_n4(
    x_ptr, lo_ptr, H_ptr, hp_ptr, out_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """n=4 specialization that reuses each x[j] tile across all 4 outputs."""
    tl.static_assert(n == 4)
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    num_tiles = num_t_tiles * num_c_tiles

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_t = tile_id // num_c_tiles
        pid_c = tile_id % num_c_tiles

        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        lo_ptrs = tl.make_block_ptr(
            base=lo_ptr,
            shape=(T, C),
            strides=(C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        lo = tl.load(lo_ptrs, boundary_check=(0, 1)).to(tl.float32)

        hp0 = tl.load(hp_ptr + t_offs * n + 0, mask=t_mask, other=0.0).to(tl.float32)
        hp1 = tl.load(hp_ptr + t_offs * n + 1, mask=t_mask, other=0.0).to(tl.float32)
        hp2 = tl.load(hp_ptr + t_offs * n + 2, mask=t_mask, other=0.0).to(tl.float32)
        hp3 = tl.load(hp_ptr + t_offs * n + 3, mask=t_mask, other=0.0).to(tl.float32)

        acc0 = hp0[:, None] * lo
        acc1 = hp1[:, None] * lo
        acc2 = hp2[:, None] * lo
        acc3 = hp3[:, None] * lo

        for j in tl.static_range(4):
            x_ptrs = tl.make_block_ptr(
                base=x_ptr + j * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            xj = tl.load(x_ptrs, boundary_check=(0, 1)).to(tl.float32)

            h0j = tl.load(H_ptr + t_offs * (n * n) + 0 * n + j, mask=t_mask, other=0.0).to(tl.float32)
            h1j = tl.load(H_ptr + t_offs * (n * n) + 1 * n + j, mask=t_mask, other=0.0).to(tl.float32)
            h2j = tl.load(H_ptr + t_offs * (n * n) + 2 * n + j, mask=t_mask, other=0.0).to(tl.float32)
            h3j = tl.load(H_ptr + t_offs * (n * n) + 3 * n + j, mask=t_mask, other=0.0).to(tl.float32)

            acc0 += h0j[:, None] * xj
            acc1 += h1j[:, None] * xj
            acc2 += h2j[:, None] * xj
            acc3 += h3j[:, None] * xj

        out0 = tl.make_block_ptr(
            base=out_ptr + 0 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        out1 = tl.make_block_ptr(
            base=out_ptr + 1 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        out2 = tl.make_block_ptr(
            base=out_ptr + 2 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        out3 = tl.make_block_ptr(
            base=out_ptr + 3 * C,
            shape=(T, C),
            strides=(n * C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0),
        )
        tl.store(out0, acc0.to(tl.bfloat16), boundary_check=(0, 1))
        tl.store(out1, acc1.to(tl.bfloat16), boundary_check=(0, 1))
        tl.store(out2, acc2.to(tl.bfloat16), boundary_check=(0, 1))
        tl.store(out3, acc3.to(tl.bfloat16), boundary_check=(0, 1))


# ---------- K4 Backward: Fused kernel (grad_x, grad_lo, grad_H, grad_hp) ----------
# Single backward kernel for K4, tiled over T with an inner C loop.

@triton.jit
def _fused_post_res_bwd_fused_kernel_n4(
    x_ptr, lo_ptr, H_ptr, hp_ptr, grad_out_ptr,
    grad_x_ptr, grad_lo_ptr, grad_H_ptr, grad_hp_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Fused K4 backward: computes all 4 gradient tensors in a single pass.

    Tiling: persistent grid over T-tiles, inner loop over C.
    - grad_H, grad_hp: reduce over C (accumulated across C-tiles, stored after sweep)
    - grad_x, grad_lo: no reduction (computed and stored per C-tile)

    Register-pressure-optimized phased streaming:
      Phase 1: Load 4 go tiles (stay live for all subsequent phases)
      Phase 2a: Compute grad_lo = sum_i hp_i * go_i, store immediately
      Phase 2b: Load lo, accumulate ghp, release lo
      Phase 3 (per j): Load x_j, accumulate gH_ij, compute+store grad_x_j, release x_j
    Max live 2D tiles: 5 (4 go + 1 streaming x/lo)
    """
    tl.static_assert(n == 4)
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_offs = tile_id * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        # Load H[t, i, j] and hp[t, i] scalars once per T-tile (don't depend on c)
        H_00 = tl.load(H_ptr + t_offs * (n * n) + 0,  mask=t_mask, other=0.0).to(tl.float32)
        H_01 = tl.load(H_ptr + t_offs * (n * n) + 1,  mask=t_mask, other=0.0).to(tl.float32)
        H_02 = tl.load(H_ptr + t_offs * (n * n) + 2,  mask=t_mask, other=0.0).to(tl.float32)
        H_03 = tl.load(H_ptr + t_offs * (n * n) + 3,  mask=t_mask, other=0.0).to(tl.float32)
        H_10 = tl.load(H_ptr + t_offs * (n * n) + 4,  mask=t_mask, other=0.0).to(tl.float32)
        H_11 = tl.load(H_ptr + t_offs * (n * n) + 5,  mask=t_mask, other=0.0).to(tl.float32)
        H_12 = tl.load(H_ptr + t_offs * (n * n) + 6,  mask=t_mask, other=0.0).to(tl.float32)
        H_13 = tl.load(H_ptr + t_offs * (n * n) + 7,  mask=t_mask, other=0.0).to(tl.float32)
        H_20 = tl.load(H_ptr + t_offs * (n * n) + 8,  mask=t_mask, other=0.0).to(tl.float32)
        H_21 = tl.load(H_ptr + t_offs * (n * n) + 9,  mask=t_mask, other=0.0).to(tl.float32)
        H_22 = tl.load(H_ptr + t_offs * (n * n) + 10, mask=t_mask, other=0.0).to(tl.float32)
        H_23 = tl.load(H_ptr + t_offs * (n * n) + 11, mask=t_mask, other=0.0).to(tl.float32)
        H_30 = tl.load(H_ptr + t_offs * (n * n) + 12, mask=t_mask, other=0.0).to(tl.float32)
        H_31 = tl.load(H_ptr + t_offs * (n * n) + 13, mask=t_mask, other=0.0).to(tl.float32)
        H_32 = tl.load(H_ptr + t_offs * (n * n) + 14, mask=t_mask, other=0.0).to(tl.float32)
        H_33 = tl.load(H_ptr + t_offs * (n * n) + 15, mask=t_mask, other=0.0).to(tl.float32)

        hp_0 = tl.load(hp_ptr + t_offs * n + 0, mask=t_mask, other=0.0).to(tl.float32)
        hp_1 = tl.load(hp_ptr + t_offs * n + 1, mask=t_mask, other=0.0).to(tl.float32)
        hp_2 = tl.load(hp_ptr + t_offs * n + 2, mask=t_mask, other=0.0).to(tl.float32)
        hp_3 = tl.load(hp_ptr + t_offs * n + 3, mask=t_mask, other=0.0).to(tl.float32)

        # 20 reduction accumulators: 4 ghp + 16 gH, each (BLOCK_T,) f32
        ghp_0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp_1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp_2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp_3 = tl.zeros((BLOCK_T,), dtype=tl.float32)

        gH_00 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_01 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_02 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_03 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_10 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_11 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_12 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_13 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_20 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_21 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_22 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_23 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_30 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_31 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_32 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gH_33 = tl.zeros((BLOCK_T,), dtype=tl.float32)

        for c_start in tl.range(0, C, BLOCK_C):
            # --- Phase 1: load all 4 grad_out tiles (stay live) ---
            go0_ptr = tl.make_block_ptr(base=grad_out_ptr + 0 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            go1_ptr = tl.make_block_ptr(base=grad_out_ptr + 1 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            go2_ptr = tl.make_block_ptr(base=grad_out_ptr + 2 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            go3_ptr = tl.make_block_ptr(base=grad_out_ptr + 3 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))

            go0 = tl.load(go0_ptr, boundary_check=(0, 1)).to(tl.float32)
            go1 = tl.load(go1_ptr, boundary_check=(0, 1)).to(tl.float32)
            go2 = tl.load(go2_ptr, boundary_check=(0, 1)).to(tl.float32)
            go3 = tl.load(go3_ptr, boundary_check=(0, 1)).to(tl.float32)

            # --- Phase 2a: compute grad_lo for this C-tile and store immediately ---
            # grad_lo[t, c] = sum_i hp[t, i] * go[t, i, c]  (no reduction over C)
            glo = hp_0[:, None] * go0 + hp_1[:, None] * go1 + hp_2[:, None] * go2 + hp_3[:, None] * go3
            glo_ptr = tl.make_block_ptr(base=grad_lo_ptr, shape=(T, C), strides=(C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            tl.store(glo_ptr, glo.to(tl.bfloat16), boundary_check=(0, 1))

            # --- Phase 2b: load lo, accumulate ghp, release lo ---
            lo_block_ptr = tl.make_block_ptr(base=lo_ptr, shape=(T, C), strides=(C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            lo = tl.load(lo_block_ptr, boundary_check=(0, 1)).to(tl.float32)
            ghp_0 += tl.sum(go0 * lo, axis=1)
            ghp_1 += tl.sum(go1 * lo, axis=1)
            ghp_2 += tl.sum(go2 * lo, axis=1)
            ghp_3 += tl.sum(go3 * lo, axis=1)

            # --- Phase 3: stream x_j one at a time, accumulate gH + compute+store grad_x_j ---
            # x stream 0
            x0_ptr = tl.make_block_ptr(base=x_ptr + 0 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            x0 = tl.load(x0_ptr, boundary_check=(0, 1)).to(tl.float32)
            gH_00 += tl.sum(go0 * x0, axis=1)
            gH_10 += tl.sum(go1 * x0, axis=1)
            gH_20 += tl.sum(go2 * x0, axis=1)
            gH_30 += tl.sum(go3 * x0, axis=1)
            # grad_x[t, 0, c] = sum_i H[t, i, 0] * go[t, i, c]
            gx0 = H_00[:, None] * go0 + H_10[:, None] * go1 + H_20[:, None] * go2 + H_30[:, None] * go3
            gx0_ptr = tl.make_block_ptr(base=grad_x_ptr + 0 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            tl.store(gx0_ptr, gx0.to(tl.bfloat16), boundary_check=(0, 1))

            # x stream 1
            x1_ptr = tl.make_block_ptr(base=x_ptr + 1 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            x1 = tl.load(x1_ptr, boundary_check=(0, 1)).to(tl.float32)
            gH_01 += tl.sum(go0 * x1, axis=1)
            gH_11 += tl.sum(go1 * x1, axis=1)
            gH_21 += tl.sum(go2 * x1, axis=1)
            gH_31 += tl.sum(go3 * x1, axis=1)
            gx1 = H_01[:, None] * go0 + H_11[:, None] * go1 + H_21[:, None] * go2 + H_31[:, None] * go3
            gx1_ptr = tl.make_block_ptr(base=grad_x_ptr + 1 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            tl.store(gx1_ptr, gx1.to(tl.bfloat16), boundary_check=(0, 1))

            # x stream 2
            x2_ptr = tl.make_block_ptr(base=x_ptr + 2 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            x2 = tl.load(x2_ptr, boundary_check=(0, 1)).to(tl.float32)
            gH_02 += tl.sum(go0 * x2, axis=1)
            gH_12 += tl.sum(go1 * x2, axis=1)
            gH_22 += tl.sum(go2 * x2, axis=1)
            gH_32 += tl.sum(go3 * x2, axis=1)
            gx2 = H_02[:, None] * go0 + H_12[:, None] * go1 + H_22[:, None] * go2 + H_32[:, None] * go3
            gx2_ptr = tl.make_block_ptr(base=grad_x_ptr + 2 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            tl.store(gx2_ptr, gx2.to(tl.bfloat16), boundary_check=(0, 1))

            # x stream 3
            x3_ptr = tl.make_block_ptr(base=x_ptr + 3 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            x3 = tl.load(x3_ptr, boundary_check=(0, 1)).to(tl.float32)
            gH_03 += tl.sum(go0 * x3, axis=1)
            gH_13 += tl.sum(go1 * x3, axis=1)
            gH_23 += tl.sum(go2 * x3, axis=1)
            gH_33 += tl.sum(go3 * x3, axis=1)
            gx3 = H_03[:, None] * go0 + H_13[:, None] * go1 + H_23[:, None] * go2 + H_33[:, None] * go3
            gx3_ptr = tl.make_block_ptr(base=grad_x_ptr + 3 * C, shape=(T, C), strides=(n * C, 1), offsets=(tile_id * BLOCK_T, c_start), block_shape=(BLOCK_T, BLOCK_C), order=(1, 0))
            tl.store(gx3_ptr, gx3.to(tl.bfloat16), boundary_check=(0, 1))

        # Store reduction outputs: grad_hp[t, i] and grad_H[t, i, j]
        tl.store(grad_hp_ptr + t_offs * n + 0, ghp_0, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 1, ghp_1, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 2, ghp_2, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 3, ghp_3, mask=t_mask)

        H_base = t_offs * (n * n)
        tl.store(grad_H_ptr + H_base + 0,  gH_00, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 1,  gH_01, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 2,  gH_02, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 3,  gH_03, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 4,  gH_10, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 5,  gH_11, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 6,  gH_12, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 7,  gH_13, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 8,  gH_20, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 9,  gH_21, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 10, gH_22, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 11, gH_23, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 12, gH_30, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 13, gH_31, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 14, gH_32, mask=t_mask)
        tl.store(grad_H_ptr + H_base + 15, gH_33, mask=t_mask)


# ═══════════════════════════════════════════════════════════════════════════
# Kernel 3: fused_pre_map
# layer_input[t, c] = Σ_j h_pre[t, j] * x_streams[t, j, c]
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_pre_map_fwd_kernel(
    x_ptr, h_pre_ptr, out_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    num_tiles = num_t_tiles * num_c_tiles

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=True):
        pid_t = tile_id // num_c_tiles
        pid_c = tile_id % num_c_tiles
        t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        t_mask = t_offs < T
        tc_mask = t_mask[:, None] & (c_offs[None, :] < C)

        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(n):
            hp_j = tl.load(h_pre_ptr + t_offs * n + j, mask=t_mask, other=0.0)
            
            x_base_offset = j * C
            x_block_ptr = tl.make_block_ptr(
                base=x_ptr + x_base_offset,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0)
            )
            x_j = tl.load(x_block_ptr, boundary_check=(0, 1)).to(tl.float32)
            acc += hp_j[:, None] * x_j

        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(T, C),
            strides=(C, 1),
            offsets=(pid_t * BLOCK_T, pid_c * BLOCK_C),
            block_shape=(BLOCK_T, BLOCK_C),
            order=(1, 0)
        )
        tl.store(out_block_ptr, acc.to(tl.bfloat16), boundary_check=(0, 1))


@triton.jit
def _fused_pre_map_bwd_fused_kernel_n4(
    x_ptr, h_pre_ptr, grad_out_ptr,
    grad_x_ptr, grad_hp_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Fused K3 backward: computes (grad_x_streams, grad_h_pre) in one pass.

    Saves a full grad_out (T,C) reread versus a two-pass backward by:
      - loading grad_out once per C-tile
      - writing grad_x for all 4 streams
      - streaming x_j to accumulate grad_h_pre reductions over C
    """
    tl.static_assert(n == 4)
    pid = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)

    for tile_id in tl.range(pid, num_t_tiles, NUM_SMS, flatten=True):
        t_offs = tile_id * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        hp0 = tl.load(h_pre_ptr + t_offs * n + 0, mask=t_mask, other=0.0).to(tl.float32)
        hp1 = tl.load(h_pre_ptr + t_offs * n + 1, mask=t_mask, other=0.0).to(tl.float32)
        hp2 = tl.load(h_pre_ptr + t_offs * n + 2, mask=t_mask, other=0.0).to(tl.float32)
        hp3 = tl.load(h_pre_ptr + t_offs * n + 3, mask=t_mask, other=0.0).to(tl.float32)

        ghp0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        ghp3 = tl.zeros((BLOCK_T,), dtype=tl.float32)

        for c_start in tl.range(0, C, BLOCK_C):
            go_ptr = tl.make_block_ptr(
                base=grad_out_ptr,
                shape=(T, C),
                strides=(C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            go = tl.load(go_ptr, boundary_check=(0, 1)).to(tl.float32)

            gx0 = hp0[:, None] * go
            gx1 = hp1[:, None] * go
            gx2 = hp2[:, None] * go
            gx3 = hp3[:, None] * go

            gx0_ptr = tl.make_block_ptr(
                base=grad_x_ptr + 0 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            gx1_ptr = tl.make_block_ptr(
                base=grad_x_ptr + 1 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            gx2_ptr = tl.make_block_ptr(
                base=grad_x_ptr + 2 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            gx3_ptr = tl.make_block_ptr(
                base=grad_x_ptr + 3 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            tl.store(gx0_ptr, gx0.to(tl.bfloat16), boundary_check=(0, 1))
            tl.store(gx1_ptr, gx1.to(tl.bfloat16), boundary_check=(0, 1))
            tl.store(gx2_ptr, gx2.to(tl.bfloat16), boundary_check=(0, 1))
            tl.store(gx3_ptr, gx3.to(tl.bfloat16), boundary_check=(0, 1))

            x0_ptr = tl.make_block_ptr(
                base=x_ptr + 0 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            x0 = tl.load(x0_ptr, boundary_check=(0, 1)).to(tl.float32)
            ghp0 += tl.sum(x0 * go, axis=1)

            x1_ptr = tl.make_block_ptr(
                base=x_ptr + 1 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            x1 = tl.load(x1_ptr, boundary_check=(0, 1)).to(tl.float32)
            ghp1 += tl.sum(x1 * go, axis=1)

            x2_ptr = tl.make_block_ptr(
                base=x_ptr + 2 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            x2 = tl.load(x2_ptr, boundary_check=(0, 1)).to(tl.float32)
            ghp2 += tl.sum(x2 * go, axis=1)

            x3_ptr = tl.make_block_ptr(
                base=x_ptr + 3 * C,
                shape=(T, C),
                strides=(n * C, 1),
                offsets=(tile_id * BLOCK_T, c_start),
                block_shape=(BLOCK_T, BLOCK_C),
                order=(1, 0),
            )
            x3 = tl.load(x3_ptr, boundary_check=(0, 1)).to(tl.float32)
            ghp3 += tl.sum(x3 * go, axis=1)

        tl.store(grad_hp_ptr + t_offs * n + 0, ghp0, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 1, ghp1, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 2, ghp2, mask=t_mask)
        tl.store(grad_hp_ptr + t_offs * n + 3, ghp3, mask=t_mask)


# Autotuned launch wrappers used by mhc_triton_ops.py runtime path.
# The search space for each kernel is capped to at most 32 configs.

@triton.autotune(configs=_AUTOTUNE_K1_FWD_CONFIGS, key=["T", "nC", "D_out"], cache_results=True)
@triton.jit
def _fused_rmsnorm_project_fwd_kernel_autotuned(
    x_ptr, W_ptr,
    out_ptr, inv_rms_ptr,
    T, nC: tl.constexpr, D_out: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    _fused_rmsnorm_project_fwd_kernel(
        x_ptr,
        W_ptr,
        out_ptr,
        inv_rms_ptr,
        T,
        nC,
        D_out,
        BLOCK_T,
        BLOCK_K,
    )


@triton.autotune(configs=_AUTOTUNE_K1_BWD_DX_CONFIGS, key=["T", "nC", "D_out"], cache_results=True)
@triton.jit
def _fused_rmsnorm_project_bwd_dx_kernel_autotuned(
    x_ptr, W_ptr, grad_out_ptr, proj_out_ptr, inv_rms_ptr,
    grad_x_ptr,
    T, nC: tl.constexpr, D_out: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    _fused_rmsnorm_project_bwd_dx_kernel(
        x_ptr,
        W_ptr,
        grad_out_ptr,
        proj_out_ptr,
        inv_rms_ptr,
        grad_x_ptr,
        T,
        nC,
        D_out,
        BLOCK_T,
        BLOCK_K,
    )


@triton.autotune(configs=_AUTOTUNE_K3_FWD_CONFIGS, key=["T", "C", "n"], cache_results=True)
@triton.jit
def _fused_pre_map_fwd_kernel_autotuned(
    x_ptr, h_pre_ptr, out_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    _fused_pre_map_fwd_kernel(
        x_ptr,
        h_pre_ptr,
        out_ptr,
        T,
        C,
        n,
        BLOCK_T,
        BLOCK_C,
        NUM_SMS,
    )


@triton.autotune(configs=_AUTOTUNE_K3_BWD_FUSED_CONFIGS, key=["T", "C", "n"], cache_results=True)
@triton.jit
def _fused_pre_map_bwd_fused_kernel_n4_autotuned(
    x_ptr, h_pre_ptr, grad_out_ptr,
    grad_x_ptr, grad_hp_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    _fused_pre_map_bwd_fused_kernel_n4(
        x_ptr,
        h_pre_ptr,
        grad_out_ptr,
        grad_x_ptr,
        grad_hp_ptr,
        T,
        C,
        n,
        BLOCK_T,
        BLOCK_C,
        NUM_SMS,
    )


@triton.autotune(configs=_AUTOTUNE_K4_FWD_CONFIGS, key=["T", "C", "n"], cache_results=True)
@triton.jit
def _fused_post_res_fwd_kernel_n4_autotuned(
    x_ptr, lo_ptr, H_ptr, hp_ptr, out_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    _fused_post_res_fwd_kernel_n4(
        x_ptr,
        lo_ptr,
        H_ptr,
        hp_ptr,
        out_ptr,
        T,
        C,
        n,
        BLOCK_T,
        BLOCK_C,
        NUM_SMS,
    )


@triton.autotune(configs=_AUTOTUNE_K4_BWD_FUSED_CONFIGS, key=["T", "C", "n"], cache_results=True)
@triton.jit
def _fused_post_res_bwd_fused_kernel_n4_autotuned(
    x_ptr, lo_ptr, H_ptr, hp_ptr, grad_out_ptr,
    grad_x_ptr, grad_lo_ptr, grad_H_ptr, grad_hp_ptr,
    T, C: tl.constexpr, n: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    _fused_post_res_bwd_fused_kernel_n4(
        x_ptr,
        lo_ptr,
        H_ptr,
        hp_ptr,
        grad_out_ptr,
        grad_x_ptr,
        grad_lo_ptr,
        grad_H_ptr,
        grad_hp_ptr,
        T,
        C,
        n,
        BLOCK_T,
        BLOCK_C,
        NUM_SMS,
    )
