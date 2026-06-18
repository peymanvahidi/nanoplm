"""CuTeDSL varlen Canon RMSNorm + depthwise conv (K=7) for sm90.

Registers ``torch.ops.nanoplm_canon_cute``:

  rms_conv_fwd(x, seq_id, rms_weight, rms_eps, conv_weight, conv_bias)
      -> (out, inv_rms)
  rms_conv_bwd_dx_dgamma(go, x, seq_id, inv_rms, rms_weight, conv_weight)
      -> (dx, dgamma)
  rms_conv_bwd_dw_db(go, x, seq_id, inv_rms)
      -> (dw_nogamma, db)   # dw is pre-gamma; autograd glue folds gamma

Drop-in faster backend for the Triton ``nanoplm_canon`` RMS ops when
`is_compatible()` holds (CUDA sm90, bf16, C in ``_CFG`` (512-3072, C%256==0),
K == 7, T % 64 == 0).
Measured on GH200 @ T=65536, C=1024 vs the Triton ops:
fwd 445->113us, bwd dx+dgamma 446->189us, bwd dw+db 417->92us (kernel time).

Design (one CTA stages its row-tile + conv halo into smem via cp.async, so
every gmem element is read exactly once per kernel):
  fwd : warp-per-row RMS stats from smem -> per-(row,tap) scale table
        (inv_rms[nbr] * seq-valid) -> branchless K-tap conv from smem with
        gamma folded into the weights; writes out + inv_rms.
  bwd : persistent CTAs over NITER consecutive tiles; conv-transpose of go
        from smem, per-row <x, wdy> via warp shuffle + smem combine, dx
        written inline; dgamma / dw / db accumulated in registers across
        tiles and written once as per-CTA partials, reduced with torch.

NOTE: no ``from __future__ import annotations`` here - PEP 563 string
annotations break CuTeDSL's Constexpr detection.
"""

import os

import torch
import torch.library

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import from_dlpack, make_ptr
from cuda.bindings import driver as _cuda_drv

VEC = 8
K = 7
R = 3
HPAD = 4             # staged halo rows each side (>= R)

# Per-C tile configs, tuned on GH200 (sm90); see agent_space/canon_cute/PLAN.md
# and agent_space/canon_cute/tune_c.py. LANES = C // VEC. The backward kernels
# launch with NTHR == LANES (one thread per vec8 lane) so per-lane register
# accumulators (dgamma/dw/db partials) stay one-thread-per-lane; the forward
# kernel has no per-lane partials and may use NTHR = FWD_ROWG * LANES for
# occupancy at small/large C. Requires C % 256 == 0 (LANES % 32 == 0 for the
# warp-per-row stats sweep). BT bounded by 227KB smem/CTA.
#
# bwd_dx is resident-warp bound (ncu: per-warp issue cadence ~4 cycles for
# every C; throughput scales with warps/SM, needs >= ~8). Two kernel shapes:
#   BWD_XS=0 (hybrid): x rows in per-row register fragments, gro in smem.
#     ~242 regs -> 2 CTAs/SM. Best when warps/CTA is pow2 (2/4/8 -> 8/SM).
#   BWD_XS=1 (all-smem): x staged to smem too, no fragment arrays -> ~190
#     regs natural; combined with BWD_MINB (launch-bounds min CTAs/SM, caps
#     regs so extra CTAs fit, tiny stack spill) reaches 9-12 warps/SM. Used
#     for non-pow2 LANES (768/1536/3072) where 2 CTAs x 3/6 warps = 6/SM
#     starved the SM (the old 44-49% plateau).
# At BWD_MINB=m the grid must have >= m*132 CTAs or SMs sit half-empty:
# keep T/(BWD_BT*BWD_NITER) >= 528 at MINB=4.
# Measured SOL at T=65536 vs the 3545 GB/s ceiling (fwd / bwd_dx / dwdb):
#   512: 65/75/87   768: 59/65/71   1024: 74/76/91   1536: 62/67/65
#   2048: 60/71/76  3072: 62/60/67
# C=4096 is excluded: at 8KB/row the smem budget caps every kernel at
# 1 CTA/SM (and the register file caps ~128 regs/thread at NTHR=512),
# leaving all three kernels under 60% with no config to fix it; it falls
# back to Triton.
# FWD_MINB: min_blocks_per_mp launch bound for the fwd kernel. Only C=1024
# benefits: at 172 regs it lands at 2.97 CTAs/SM -> 2 (12% occ); FWD_MINB=3
# trims to 168 regs -> exactly 3 CTAs/SM (18% occ), no spill, 148->136us, DRAM
# 42->45% SOL, bit-identical. Other C already hit ~18% occ or spill if forced.
#   C: (FWD_BT, BWD_BT, BWD_NITER, DWDB_BT, DWDB_NITER, FWD_ROWG,
#       BWD_XS, BWD_MINB, FWD_MINB)
_CFG = {
    512:  (16, 8, 16, 8, 16, 1, 0, 0, 0),
    768:  (8, 4, 16, 8, 16, 1, 1, 4, 0),
    1024: (16, 8, 16, 8, 32, 1, 0, 0, 3),
    1536: (8, 8, 16, 16, 8, 1, 1, 2, 0),
    2048: (16, 8, 16, 8, 32, 1, 0, 0, 0),
    3072: (16, 8, 32, 8, 32, 1, 1, 1, 0),
}


def _lanes(C):
    return C // VEC


def _row_lane(tidx, c, NTHR, LANES):
    """(row, lane) for staging copy c. With NTHR == LANES each thread owns
    exactly one lane, so the decomposition is free. NTHR is always a
    multiple of LANES, so the generic path's div/mod of tidx is
    loop-invariant (CSE'd across the unrolled staging loop)."""
    if cutlass.const_expr(NTHR == LANES):
        return c, tidx
    ROWG = NTHR // LANES
    return tidx // LANES + c * ROWG, tidx % LANES


# ════════════════════════════════════════════════════════════════════════
# Kernels
# ════════════════════════════════════════════════════════════════════════


@cute.kernel
def _fwd_kernel(
    mX: cute.Tensor,    # (T, LANES, VEC) bf16
    mO: cute.Tensor,    # (T, LANES, VEC) bf16
    mSeq: cute.Tensor,  # (T,) int32
    mInv: cute.Tensor,  # (T,) fp32 out
    mWf: cute.Tensor,   # (K, LANES, VEC) bf16: (w * gamma) transposed
    mG: cute.Tensor,    # (LANES, VEC) bf16 gamma (raw rms_weight)
    mB: cute.Tensor,    # (LANES, VEC) bf16 conv bias (raw)
    eps: cutlass.Float32,
    T: cutlass.Int32,
    BT: cutlass.Constexpr,
    NTHR: cutlass.Constexpr,
    LANES: cutlass.Constexpr,
    HAS_NORM: cutlass.Constexpr,
):
    HB = BT + 2 * HPAD
    ROWG = NTHR // LANES
    RT = BT // ROWG
    NW = NTHR // 32

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    t0 = bidx * BT

    alloc = cutlass.utils.SmemAllocator()
    s_x = alloc.allocate_tensor(
        cute.BFloat16,
        cute.make_layout((HB, LANES, VEC), stride=(LANES * VEC, VEC, 1)), 128)
    s_seq = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(HB), 16)
    s_inv = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(HB), 16)
    s_scl = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(BT * K), 16)

    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cute.BFloat16,
                               num_bits_per_copy=128)
    nch = HB * LANES // NTHR
    for c in cutlass.range_constexpr(nch):
        row, ln = _row_lane(tidx, c, NTHR, LANES)
        gt = t0 - HPAD + row
        if gt < 0:
            gt = 0
        if gt > T - 1:
            gt = T - 1
        cute.copy(atom, mX[gt, ln, None], s_x[row, ln, None])
    cute.arch.cp_async_commit_group()

    if tidx < HB:
        gt = t0 - HPAD + tidx
        sq = cutlass.Int32(-1)
        if gt >= 0 and gt < T:
            sq = mSeq[gt]
        s_seq[tidx] = sq

    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # phase A: warp-per-row RMS stats from smem (skipped for plain conv)
    warp = tidx // 32
    lane32 = tidx % 32
    fa = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
    n_iters = (HB + NW - 1) // NW if HAS_NORM else 0
    for it in cutlass.range_constexpr(n_iters):
        h = warp + it * NW
        if cutlass.const_expr((it + 1) * NW <= HB):
            pred = True
        else:
            pred = h < HB
        if pred:
            acc = cute.make_fragment(cute.make_layout(VEC), cutlass.Float32)
            acc.fill(0.0)
            av = acc.load()
            for q in cutlass.range_constexpr(LANES // 32):
                cute.autovec_copy(s_x[h, lane32 + 32 * q, None], fa)
                xf = fa.load().to(cutlass.Float32)
                av = av + xf * xf
            ss = av.reduce(cute.ReductionOp.ADD, 0.0, 0)
            for off in cutlass.range_constexpr(5):
                ss = ss + cute.arch.shuffle_sync_bfly(ss, 1 << off)
            if lane32 == 0:
                inv = cute.math.rsqrt(ss / (LANES * VEC) + eps)
                s_inv[h] = inv
                gt = t0 - HPAD + h
                if h >= HPAD and h < HPAD + BT:
                    mInv[gt] = inv
    cute.arch.barrier()

    # phase A2: per-(row,tap) scale = inv_rms[nbr] * seq-valid
    n2 = (BT * K + NTHR - 1) // NTHR
    for it in cutlass.range_constexpr(n2):
        idx = tidx + it * NTHR
        if idx < BT * K:
            row = idx // K
            tap = idx % K
            hc = row + HPAD
            hn = hc + tap - R
            sc = cutlass.Float32(0.0)
            if s_seq[hn] == s_seq[hc]:
                if cutlass.const_expr(HAS_NORM):
                    sc = s_inv[hn]
                else:
                    sc = cutlass.Float32(1.0)
            s_scl[idx] = sc
    cute.arch.barrier()

    # phase B: branchless conv from smem
    rg, lane = _row_lane(tidx, 0, NTHR, LANES)

    # Folded weights stay a host-side transform here: fwd CTAs are NOT
    # persistent (one tile each), so per-CTA scalar weight gathers don't
    # amortize (measured 5-17 SOL points lost across C). The bwd kernels are
    # persistent and DO read the raw (C, K) layout in-kernel. Gamma/bias are
    # read raw bf16 (no host float()/contiguous() kernels).
    wf = []
    for j in cutlass.range_constexpr(K):
        f = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
        cute.autovec_copy(mWf[j, lane, None], f)
        wf.append(f.load())  # keep bf16 in regs; convert per use
    if cutlass.const_expr(HAS_NORM):
        gam_f16 = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
        cute.autovec_copy(mG[lane, None], gam_f16)
        gam = gam_f16.load().to(cutlass.Float32)
    bias_f16 = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
    cute.autovec_copy(mB[lane, None], bias_f16)
    bias = bias_f16.load().to(cutlass.Float32)

    xfr = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
    out_frag = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)

    for j in cutlass.range_constexpr(RT):
        row = rg * RT + j
        hc = row + HPAD
        srow = row * K
        inv_c = s_scl[srow + R]
        cute.autovec_copy(s_x[hc, lane, None], xfr)
        if cutlass.const_expr(HAS_NORM):
            acc = bias + xfr.load().to(cutlass.Float32) * (gam * inv_c)
        else:
            acc = bias + xfr.load().to(cutlass.Float32)
        for s in cutlass.range_constexpr(K):
            cute.autovec_copy(s_x[hc - R + s, lane, None], xfr)
            acc = acc + xfr.load().to(cutlass.Float32) * (
                wf[s].to(cutlass.Float32) * s_scl[srow + s])
        out_frag.store(acc.to(cute.BFloat16))
        cute.autovec_copy(out_frag, mO[t0 + row, lane, None])


@cute.kernel
def _pconv_fwd_persist_kernel(
    mX: cute.Tensor,    # (T, LANES, VEC) bf16
    mO: cute.Tensor,    # (T, LANES, VEC) bf16
    mSeq: cute.Tensor,  # (T,) int32
    mWf: cute.Tensor,   # (K, LANES, VEC) bf16
    mB: cute.Tensor,    # (LANES, VEC) bf16
    T: cutlass.Int32,
    BT: cutlass.Constexpr,
    LANES: cutlass.Constexpr,
    NITER: cutlass.Constexpr,
):
    HB = BT + 2 * HPAD
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx

    alloc = cutlass.utils.SmemAllocator()
    s_x = alloc.allocate_tensor(
        cute.BFloat16,
        cute.make_layout((HB, LANES, VEC), stride=(LANES * VEC, VEC, 1)), 128)
    s_seq = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(HB), 16)
    s_scl = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(BT * K), 16)

    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cute.BFloat16,
                               num_bits_per_copy=128)

    wf = []
    for j in cutlass.range_constexpr(K):
        f = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
        cute.autovec_copy(mWf[j, lane, None], f)
        wf.append(f.load())
    bias_f16 = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
    cute.autovec_copy(mB[lane, None], bias_f16)
    bias = bias_f16.load().to(cutlass.Float32)

    xfr = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
    out_frag = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)

    for tile in cutlass.range_constexpr(NITER):
        t0 = (bidx * NITER + tile) * BT

        for c in cutlass.range_constexpr(HB):
            gt = t0 - HPAD + c
            if gt < 0:
                gt = 0
            if gt > T - 1:
                gt = T - 1
            cute.copy(atom, mX[gt, lane, None], s_x[c, lane, None])
        cute.arch.cp_async_commit_group()

        if tidx < HB:
            gt = t0 - HPAD + tidx
            sq = cutlass.Int32(-1)
            if gt >= 0 and gt < T:
                sq = mSeq[gt]
            s_seq[tidx] = sq

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        n2 = (BT * K + LANES - 1) // LANES
        for it in cutlass.range_constexpr(n2):
            idx = tidx + it * LANES
            if idx < BT * K:
                row = idx // K
                tap = idx % K
                hc = row + HPAD
                hn = hc + tap - R
                sc = cutlass.Float32(0.0)
                if s_seq[hn] == s_seq[hc]:
                    sc = cutlass.Float32(1.0)
                s_scl[idx] = sc
        cute.arch.barrier()

        for row in cutlass.range_constexpr(BT):
            hc = row + HPAD
            srow = row * K
            cute.autovec_copy(s_x[hc, lane, None], xfr)
            acc = bias + xfr.load().to(cutlass.Float32)
            for s in cutlass.range_constexpr(K):
                cute.autovec_copy(s_x[hc - R + s, lane, None], xfr)
                acc = acc + xfr.load().to(cutlass.Float32) * (
                    wf[s].to(cutlass.Float32) * s_scl[srow + s])
            out_frag.store(acc.to(cute.BFloat16))
            cute.autovec_copy(out_frag, mO[t0 + row, lane, None])

        cute.arch.barrier()


@cute.kernel
def _bwd_dx_kernel(
    mGo: cute.Tensor,    # (T, LANES, VEC) bf16
    mX: cute.Tensor,     # (T, LANES, VEC) bf16
    mSeq: cute.Tensor,   # (T,) int32
    mInv: cute.Tensor,   # (T,) fp32
    mW: cute.Tensor,     # (C, K) bf16: raw conv weight (flip via tap index)
    mG: cute.Tensor,     # (LANES, VEC) bf16 gamma (raw rms_weight)
    mDx: cute.Tensor,    # (T, LANES, VEC) bf16 out
    mDgP: cute.Tensor,   # (nGrid, LANES, VEC) fp32 out dgamma partials
    T: cutlass.Int32,
    BT: cutlass.Constexpr,
    NTHR: cutlass.Constexpr,
    NITER: cutlass.Constexpr,
    LANES: cutlass.Constexpr,
    HAS_NORM: cutlass.Constexpr,
    XSMEM: cutlass.Constexpr,
):
    HB = BT + 2 * HPAD
    ROWG = NTHR // LANES
    RT = BT // ROWG
    NL32 = LANES // 32   # sub-warps per row (per-row reduction partials)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    alloc = cutlass.utils.SmemAllocator()
    s_go = alloc.allocate_tensor(
        cute.BFloat16,
        cute.make_layout((HB, LANES, VEC), stride=(LANES * VEC, VEC, 1)), 128)
    if cutlass.const_expr(HAS_NORM):
        s_gro = alloc.allocate_tensor(
            cute.BFloat16,
            cute.make_layout((BT, LANES, VEC), stride=(LANES * VEC, VEC, 1)), 128)
    if cutlass.const_expr(XSMEM):
        s_x = alloc.allocate_tensor(
            cute.BFloat16,
            cute.make_layout((BT, LANES, VEC), stride=(LANES * VEC, VEC, 1)),
            128)
    s_seq = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(HB), 16)
    if cutlass.const_expr(HAS_NORM):
        s_inv = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(BT), 16)
        s_msk = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(BT * K), 16)
        s_red = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout(BT * NL32), 16)

    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cute.BFloat16,
                               num_bits_per_copy=128)

    rg, lane = _row_lane(tidx, 0, NTHR, LANES)
    warp4 = lane // 32
    lane32 = tidx % 32

    # raw (C, K) weight, flipped at load time (conv-transpose); scalar
    # gathers once per persistent CTA, L1/L2-resident -> no host transform.
    wfl = []
    for j in cutlass.range_constexpr(K):
        fw16 = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
        for v in cutlass.range_constexpr(VEC):
            fw16[v] = mW[lane * VEC + v, K - 1 - j]
        wfl.append(fw16.load())  # bf16 in regs; convert per use
    if cutlass.const_expr(HAS_NORM):
        gam_f16 = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
        cute.autovec_copy(mG[lane, None], gam_f16)
        gam = gam_f16.load().to(cutlass.Float32)

    if cutlass.const_expr(HAS_NORM):
        dg_acc_f = cute.make_fragment(cute.make_layout(VEC), cutlass.Float32)
        dg_acc_f.fill(0.0)
        dg = dg_acc_f.load()

    fr = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
    fr2 = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
    out_frag = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
    # Hybrid (XSMEM=0): x rows live in per-row register fragments instead of
    # smem (x has no halo) -> smem is (2*BT+8) rows instead of (3*BT+8). gro
    # stays in smem: holding both in registers exceeds the per-thread budget
    # at large LANES (regfile/NTHR shrinks as C grows) and spills.
    # All-smem (XSMEM=1): x cp.async-staged to smem like go; no fragment
    # arrays -> ~190 regs vs ~242, so MINB launch bounds can fit 3-4 CTAs/SM
    # (the kernel is resident-warp bound; non-pow2 LANES needs this to get
    # past 6 warps/SM).
    nfx = RT if (HAS_NORM and not XSMEM) else 0
    fx = [cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
          for _ in range(nfx)]

    for it in cutlass.range(NITER):
        t0 = (bidx * NITER + it) * BT

        nch = HB * LANES // NTHR
        for c in cutlass.range_constexpr(nch):
            row, ln = _row_lane(tidx, c, NTHR, LANES)
            gt = t0 - HPAD + row
            if gt < 0:
                gt = 0
            if gt > T - 1:
                gt = T - 1
            cute.copy(atom, mGo[gt, ln, None], s_go[row, ln, None])
        if cutlass.const_expr(XSMEM and HAS_NORM):
            nchx = BT * LANES // NTHR
            for c in cutlass.range_constexpr(nchx):
                row, ln = _row_lane(tidx, c, NTHR, LANES)
                cute.copy(atom, mX[t0 + row, ln, None], s_x[row, ln, None])
        cute.arch.cp_async_commit_group()
        # prefetch this tile's x rows into registers (independent loads,
        # issued back-to-back so they overlap)
        for j in cutlass.range_constexpr(nfx):
            cute.autovec_copy(mX[t0 + rg * RT + j, lane, None], fx[j])

        if tidx < HB:
            gt = t0 - HPAD + tidx
            sq = cutlass.Int32(-1)
            if gt >= 0 and gt < T:
                sq = mSeq[gt]
            s_seq[tidx] = sq
        if cutlass.const_expr(HAS_NORM):
            if tidx < BT:
                s_inv[tidx] = mInv[t0 + tidx]

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        if cutlass.const_expr(HAS_NORM):
            n2 = (BT * K + NTHR - 1) // NTHR
            for c in cutlass.range_constexpr(n2):
                idx = tidx + c * NTHR
                if idx < BT * K:
                    row = idx // K
                    tap = idx % K
                    hc = row + HPAD
                    m = cutlass.Float32(0.0)
                    if s_seq[hc + tap - R] == s_seq[hc]:
                        m = cutlass.Float32(1.0)
                    s_msk[idx] = m
            cute.arch.barrier()

        # B1: conv-T (+ gro stash + dgamma + per-warp partial <x, wdy>)
        for j in cutlass.range_constexpr(RT):
            row = rg * RT + j
            hc = row + HPAD
            cute.autovec_copy(s_go[hc, lane, None], fr)
            acc = fr.load().to(cutlass.Float32)
            for s in cutlass.range_constexpr(K):
                hn = hc - R + s
                cute.autovec_copy(s_go[hn, lane, None], fr)
                gv = fr.load().to(cutlass.Float32)
                if cutlass.const_expr(HAS_NORM):
                    srow = row * K
                    acc = acc + gv * (wfl[s].to(cutlass.Float32) * s_msk[srow + s])
                else:
                    if s_seq[hn] == s_seq[hc]:
                        acc = acc + gv * wfl[s].to(cutlass.Float32)
            if cutlass.const_expr(HAS_NORM):
                inv_c = s_inv[row]
                fr2.store(acc.to(cute.BFloat16))
                cute.autovec_copy(fr2, s_gro[row, lane, None])
                if cutlass.const_expr(XSMEM):
                    cute.autovec_copy(s_x[row, lane, None], fr)
                    xv = fr.load().to(cutlass.Float32)
                else:
                    xv = fx[j].load().to(cutlass.Float32)
                dg = dg + acc * (xv * inv_c)
                p = (xv * (acc * gam)).reduce(cute.ReductionOp.ADD, 0.0, 0)
                for off in cutlass.range_constexpr(5):
                    p = p + cute.arch.shuffle_sync_bfly(p, 1 << off)
                if lane32 == 0:
                    s_red[row * NL32 + warp4] = p
            else:
                # plain conv-T: dx = go + convT(go) directly
                out_frag.store(acc.to(cute.BFloat16))
                cute.autovec_copy(out_frag, mDx[t0 + row, lane, None])
        cute.arch.barrier()

        # B2: dx (RMS backward; skipped for plain conv)
        if cutlass.const_expr(HAS_NORM):
            for j in cutlass.range_constexpr(RT):
                row = rg * RT + j
                inv_c = s_inv[row]
                srw = s_red[row * NL32]
                for wsub in cutlass.range_constexpr(1, NL32):
                    srw = srw + s_red[row * NL32 + wsub]
                cute.autovec_copy(s_gro[row, lane, None], fr)
                gro = fr.load().to(cutlass.Float32)
                if cutlass.const_expr(XSMEM):
                    cute.autovec_copy(s_x[row, lane, None], fr2)
                    xv = fr2.load().to(cutlass.Float32)
                else:
                    xv = fx[j].load().to(cutlass.Float32)
                c1 = inv_c * inv_c * srw / (LANES * VEC)
                dx = (gro * gam - xv * c1) * inv_c
                out_frag.store(dx.to(cute.BFloat16))
                cute.autovec_copy(out_frag, mDx[t0 + row, lane, None])
            cute.arch.barrier()  # protect smem before next tile's staging

    if cutlass.const_expr(HAS_NORM):
        dg_acc_f.store(dg)
        cute.autovec_copy(dg_acc_f, mDgP[bidx, lane, None])


@cute.kernel
def _dwdb_kernel(
    mGo: cute.Tensor,    # (T, LANES, VEC) bf16
    mX: cute.Tensor,     # (T, LANES, VEC) bf16
    mSeq: cute.Tensor,   # (T,) int32
    mInv: cute.Tensor,   # (T,) fp32
    mDwP: cute.Tensor,   # (nGrid, K+1, LANES, VEC) fp32: K dw taps + db
    T: cutlass.Int32,
    BT: cutlass.Constexpr,
    NTHR: cutlass.Constexpr,
    NITER: cutlass.Constexpr,
    LANES: cutlass.Constexpr,
    HAS_NORM: cutlass.Constexpr,
):
    HB = BT + 2 * HPAD
    ROWG = NTHR // LANES
    RT = BT // ROWG

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    alloc = cutlass.utils.SmemAllocator()
    s_x = alloc.allocate_tensor(
        cute.BFloat16,
        cute.make_layout((HB, LANES, VEC), stride=(LANES * VEC, VEC, 1)), 128)
    s_go = alloc.allocate_tensor(
        cute.BFloat16,
        cute.make_layout((BT, LANES, VEC), stride=(LANES * VEC, VEC, 1)), 128)
    s_seq = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(HB), 16)
    if cutlass.const_expr(HAS_NORM):
        s_inv = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(HB), 16)
        s_scl = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(BT * K), 16)

    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cute.BFloat16,
                               num_bits_per_copy=128)

    rg, lane = _row_lane(tidx, 0, NTHR, LANES)

    dw_f = []
    for s in cutlass.range_constexpr(K):
        f = cute.make_fragment(cute.make_layout(VEC), cutlass.Float32)
        f.fill(0.0)
        dw_f.append(f)
    dw = [f.load() for f in dw_f]
    db_f = cute.make_fragment(cute.make_layout(VEC), cutlass.Float32)
    db_f.fill(0.0)
    db = db_f.load()

    fr = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)
    fr2 = cute.make_fragment(cute.make_layout(VEC), cute.BFloat16)

    for it in cutlass.range(NITER):
        t0 = (bidx * NITER + it) * BT

        nch = HB * LANES // NTHR
        for c in cutlass.range_constexpr(nch):
            row, ln = _row_lane(tidx, c, NTHR, LANES)
            gt = t0 - HPAD + row
            if gt < 0:
                gt = 0
            if gt > T - 1:
                gt = T - 1
            cute.copy(atom, mX[gt, ln, None], s_x[row, ln, None])
        nchg = BT * LANES // NTHR
        for c in cutlass.range_constexpr(nchg):
            row, ln = _row_lane(tidx, c, NTHR, LANES)
            cute.copy(atom, mGo[t0 + row, ln, None], s_go[row, ln, None])
        cute.arch.cp_async_commit_group()

        if tidx < HB:
            gt = t0 - HPAD + tidx
            sq = cutlass.Int32(-1)
            if gt >= 0 and gt < T:
                sq = mSeq[gt]
            s_seq[tidx] = sq
            if cutlass.const_expr(HAS_NORM):
                iv = cutlass.Float32(0.0)
                if gt >= 0 and gt < T:
                    iv = mInv[gt]
                s_inv[tidx] = iv

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        if cutlass.const_expr(HAS_NORM):
            n2 = (BT * K + NTHR - 1) // NTHR
            for c in cutlass.range_constexpr(n2):
                idx = tidx + c * NTHR
                if idx < BT * K:
                    row = idx // K
                    tap = idx % K
                    hc = row + HPAD
                    hn = hc + tap - R
                    sc = cutlass.Float32(0.0)
                    if s_seq[hn] == s_seq[hc]:
                        sc = s_inv[hn]
                    s_scl[idx] = sc
            cute.arch.barrier()

        for j in cutlass.range_constexpr(RT):
            row = rg * RT + j
            hc = row + HPAD
            srow = row * K
            cute.autovec_copy(s_go[row, lane, None], fr)
            gov = fr.load().to(cutlass.Float32)
            db = db + gov
            for s in cutlass.range_constexpr(K):
                hn = hc - R + s
                if cutlass.const_expr(HAS_NORM):
                    cute.autovec_copy(s_x[hn, lane, None], fr2)
                    xv = fr2.load().to(cutlass.Float32)
                    dw[s] = dw[s] + gov * (xv * s_scl[srow + s])
                else:
                    cute.autovec_copy(s_x[hn, lane, None], fr2)
                    xv = fr2.load().to(cutlass.Float32)
                    if s_seq[hn] == s_seq[hc]:
                        dw[s] = dw[s] + gov * xv
        cute.arch.barrier()

    for s in cutlass.range_constexpr(K):
        dw_f[s].store(dw[s])
        cute.autovec_copy(dw_f[s], mDwP[bidx, s, lane, None])
    db_f.store(db)
    cute.autovec_copy(db_f, mDwP[bidx, K, lane, None])


# ════════════════════════════════════════════════════════════════════════
# Launchers + compile cache
# ════════════════════════════════════════════════════════════════════════


@cute.jit
def _fwd_launch(
    pX: cute.Pointer, pO: cute.Pointer, pSeq: cute.Pointer, pInv: cute.Pointer,
    pWf: cute.Pointer, pG: cute.Pointer, pB: cute.Pointer,
    eps: cutlass.Float32, T: cutlass.Int32, nblk: cutlass.Int32,
    stream: _cuda_drv.CUstream,
    BT: cutlass.Constexpr, NTHR: cutlass.Constexpr, LANES: cutlass.Constexpr,
    HAS_NORM: cutlass.Constexpr, MINB: cutlass.Constexpr,
):
    l3 = cute.make_layout((T, LANES, VEC), stride=(LANES * VEC, VEC, 1))
    mX = cute.make_tensor(pX, l3)
    mO = cute.make_tensor(pO, l3)
    mSeq = cute.make_tensor(pSeq, cute.make_layout(T))
    mInv = cute.make_tensor(pInv, cute.make_layout(T))
    mWf = cute.make_tensor(
        pWf, cute.make_layout((K, LANES, VEC), stride=(LANES * VEC, VEC, 1)))
    mG = cute.make_tensor(pG, cute.make_layout((LANES, VEC), stride=(VEC, 1)))
    mB = cute.make_tensor(pB, cute.make_layout((LANES, VEC), stride=(VEC, 1)))
    _fwd_kernel(mX, mO, mSeq, mInv, mWf, mG, mB, eps, T, BT, NTHR,
                LANES, HAS_NORM).launch(
        grid=(nblk, 1, 1), block=(NTHR, 1, 1), stream=stream,
        min_blocks_per_mp=MINB,
    )


@cute.jit
def _pconv_fwd_persist_launch(
    pX: cute.Pointer, pO: cute.Pointer, pSeq: cute.Pointer,
    pWf: cute.Pointer, pB: cute.Pointer,
    T: cutlass.Int32, nblk: cutlass.Int32, stream: _cuda_drv.CUstream,
    BT: cutlass.Constexpr, LANES: cutlass.Constexpr, NITER: cutlass.Constexpr,
):
    l3 = cute.make_layout((T, LANES, VEC), stride=(LANES * VEC, VEC, 1))
    mX = cute.make_tensor(pX, l3)
    mO = cute.make_tensor(pO, l3)
    mSeq = cute.make_tensor(pSeq, cute.make_layout(T))
    mWf = cute.make_tensor(
        pWf, cute.make_layout((K, LANES, VEC), stride=(LANES * VEC, VEC, 1)))
    mB = cute.make_tensor(pB, cute.make_layout((LANES, VEC), stride=(VEC, 1)))
    _pconv_fwd_persist_kernel(mX, mO, mSeq, mWf, mB, T, BT, LANES, NITER).launch(
        grid=(nblk, 1, 1), block=(LANES, 1, 1), stream=stream
    )


@cute.jit
def _bwd_dx_launch(
    pGo: cute.Pointer, pX: cute.Pointer, pSeq: cute.Pointer, pInv: cute.Pointer,
    pW: cute.Pointer, pG: cute.Pointer, pDx: cute.Pointer, pDgP: cute.Pointer,
    T: cutlass.Int32, nblk: cutlass.Int32, stream: _cuda_drv.CUstream,
    BT: cutlass.Constexpr, NTHR: cutlass.Constexpr, NITER: cutlass.Constexpr,
    LANES: cutlass.Constexpr, HAS_NORM: cutlass.Constexpr,
    XSMEM: cutlass.Constexpr, MINB: cutlass.Constexpr,
):
    l3 = cute.make_layout((T, LANES, VEC), stride=(LANES * VEC, VEC, 1))
    mGo = cute.make_tensor(pGo, l3)
    mX = cute.make_tensor(pX, l3)
    mDx = cute.make_tensor(pDx, l3)
    mSeq = cute.make_tensor(pSeq, cute.make_layout(T))
    mInv = cute.make_tensor(pInv, cute.make_layout(T))
    mW = cute.make_tensor(
        pW, cute.make_layout((LANES * VEC, K), stride=(K, 1)))
    mG = cute.make_tensor(pG, cute.make_layout((LANES, VEC), stride=(VEC, 1)))
    mDgP = cute.make_tensor(
        pDgP, cute.make_layout((nblk, LANES, VEC), stride=(LANES * VEC, VEC, 1)))
    _bwd_dx_kernel(mGo, mX, mSeq, mInv, mW, mG, mDx, mDgP,
                   T, BT, NTHR, NITER, LANES, HAS_NORM, XSMEM).launch(
        grid=(nblk, 1, 1), block=(NTHR, 1, 1), stream=stream,
        min_blocks_per_mp=MINB,
    )


@cute.jit
def _dwdb_launch(
    pGo: cute.Pointer, pX: cute.Pointer, pSeq: cute.Pointer, pInv: cute.Pointer,
    pDwP: cute.Pointer,
    T: cutlass.Int32, nblk: cutlass.Int32, stream: _cuda_drv.CUstream,
    BT: cutlass.Constexpr, NTHR: cutlass.Constexpr, NITER: cutlass.Constexpr,
    LANES: cutlass.Constexpr, HAS_NORM: cutlass.Constexpr,
):
    l3 = cute.make_layout((T, LANES, VEC), stride=(LANES * VEC, VEC, 1))
    mGo = cute.make_tensor(pGo, l3)
    mX = cute.make_tensor(pX, l3)
    mSeq = cute.make_tensor(pSeq, cute.make_layout(T))
    mInv = cute.make_tensor(pInv, cute.make_layout(T))
    mDwP = cute.make_tensor(
        pDwP, cute.make_layout((nblk, K + 1, LANES, VEC),
                               stride=((K + 1) * LANES * VEC, LANES * VEC, VEC, 1)))
    _dwdb_kernel(mGo, mX, mSeq, mInv, mDwP, T, BT, NTHR, NITER,
                 LANES, HAS_NORM).launch(
        grid=(nblk, 1, 1), block=(NTHR, 1, 1), stream=stream
    )


_compile_cache = {}


def _compiled(name, launcher, args, constexprs):
    key = (name,) + constexprs
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(launcher, *args, *constexprs)
    return _compile_cache[key]


def _stream():
    return _cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)


# Debug aid: cute kernels launch via the CUDA driver API, which
# CUDA_LAUNCH_BLOCKING does not synchronize. With NANOPLM_CANON_CUTE_SYNC=1
# every cute launch is followed by a device sync so faults are attributed
# to the exact op instead of a later (unrelated) launch.
_DEBUG_SYNC = os.environ.get("NANOPLM_CANON_CUTE_SYNC", "0") == "1"


def _maybe_sync(tag):
    if _DEBUG_SYNC:
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(f"canon cute kernel fault at {tag}") from exc


def _pbf16(t):
    return make_ptr(cute.BFloat16, t.data_ptr(), cute.AddressSpace.gmem,
                    assumed_align=16)


def _pf32(t, align=4):
    return make_ptr(cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem,
                    assumed_align=align)


def _pi32(t):
    return make_ptr(cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem,
                    assumed_align=4)


# NOTE: no host-side memoization of derived tensors here. Keying on
# (data_ptr, _version) is unsafe in training: FSDP2 unshard buffers and
# activation allocations recycle addresses with _version=0, so a cache
# returns stale (and possibly smaller) tensors -> garbage masks / OOB.


_dummy_f32 = {}


def _dummy(device, C):
    # C fp32 zeros; stands in for unused gamma/inv/partial pointers on the
    # plain (HAS_NORM=False) path. All reads of these are gated on HAS_NORM
    # in-kernel; the buffer is a safety placeholder, >= C in every dtype view.
    buf = _dummy_f32.get((device, C))
    if buf is None:
        buf = torch.zeros(C, dtype=torch.float32, device=device)
        _dummy_f32[(device, C)] = buf
    return buf


def _pick_niter(T, bt, pref):
    """Largest NITER <= pref with T % (bt * NITER) == 0 (persistent grids)."""
    ni = pref
    while ni > 1 and T % (bt * ni) != 0:
        ni //= 2
    return ni


def _seq_i32(seq_id):
    if seq_id.dtype == torch.int32 and seq_id.is_contiguous():
        return seq_id
    return seq_id.detach().to(torch.int32).contiguous()


def _bf16c(t):
    """Raw bf16 contiguous view of a param for in-kernel native-layout reads.

    No-op (no kernel launched) in the standard training setup where params
    are already contiguous bf16."""
    t = t.detach()
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


# ════════════════════════════════════════════════════════════════════════
# torch.library ops
# ════════════════════════════════════════════════════════════════════════

_lib = torch.library.Library("nanoplm_canon_cute", "DEF")

_lib.define(
    "rms_conv_fwd(Tensor x, Tensor seq_id, Tensor rms_weight, float rms_eps, "
    "Tensor conv_weight, Tensor conv_bias) -> (Tensor, Tensor)"
)
_lib.define(
    "rms_conv_bwd_dx_dgamma(Tensor go, Tensor x, Tensor seq_id, Tensor inv_rms, "
    "Tensor rms_weight, Tensor conv_weight) -> (Tensor, Tensor)"
)
_lib.define(
    "rms_conv_bwd_dw_db(Tensor go, Tensor x, Tensor seq_id, Tensor inv_rms) "
    "-> (Tensor, Tensor)"
)


@torch.library.register_fake("nanoplm_canon_cute::rms_conv_fwd")
def _fwd_fake(x, seq_id, rms_weight, rms_eps, conv_weight, conv_bias):
    T = x.shape[0]
    return torch.empty_like(x), torch.empty(T, dtype=torch.float32,
                                            device=x.device)


@torch.library.register_fake("nanoplm_canon_cute::rms_conv_bwd_dx_dgamma")
def _bwd_dx_fake(go, x, seq_id, inv_rms, rms_weight, conv_weight):
    return torch.empty_like(x), torch.empty_like(rms_weight,
                                                 dtype=torch.float32)


@torch.library.register_fake("nanoplm_canon_cute::rms_conv_bwd_dw_db")
def _bwd_dwdb_fake(go, x, seq_id, inv_rms):
    C = x.shape[1]
    dw = torch.empty(C, K, dtype=torch.float32, device=x.device)
    db = torch.empty(C, dtype=torch.float32, device=x.device)
    return dw, db


@torch.library.impl(_lib, "rms_conv_fwd", "CUDA")
@torch.no_grad()
def _fwd_cuda(x, seq_id, rms_weight, rms_eps, conv_weight, conv_bias):
    T, C = x.shape
    x = x.detach().contiguous()
    out = torch.empty_like(x)
    inv = torch.empty(T, dtype=torch.float32, device=x.device)

    # bf16 fold+transpose: 2 small kernels (vs 4-5 before — float converts
    # and bias/gamma copies are gone; those are read raw bf16 in-kernel).
    # torch bf16 mul computes in fp32 internally, so (w * g) here matches the
    # old fp32-fold bit-for-bit after the bf16 round.
    g16 = _bf16c(rms_weight)
    b16 = _bf16c(conv_bias)
    wf = (conv_weight.detach() * g16.unsqueeze(1)).t().contiguous()
    seq = _seq_i32(seq_id)
    lanes = _lanes(C)

    fwd_bt, fwd_nthr = _CFG[C][0], _CFG[C][5] * lanes
    args = (
        _pbf16(x), _pbf16(out), _pi32(seq), _pf32(inv),
        _pbf16(wf), _pbf16(g16), _pbf16(b16),
        cutlass.Float32(rms_eps), cutlass.Int32(T),
        cutlass.Int32(T // fwd_bt), _stream(),
    )
    fn = _compiled("fwd", _fwd_launch, args,
                   (fwd_bt, fwd_nthr, lanes, True, _CFG[C][8]))
    fn(*args)
    _maybe_sync("rms_conv_fwd")
    return out, inv


@torch.library.impl(_lib, "rms_conv_bwd_dx_dgamma", "CUDA")
@torch.no_grad()
def _bwd_dx_cuda(go, x, seq_id, inv_rms, rms_weight, conv_weight):
    T, C = x.shape
    go = go.detach().contiguous()
    x = x.detach().contiguous()
    inv_rms = inv_rms.detach().contiguous()
    lanes = _lanes(C)
    bwd_bt, bwd_niter_pref = _CFG[C][1], _CFG[C][2]
    bwd_xs, bwd_minb = _CFG[C][6], _CFG[C][7]
    niter = _pick_niter(T, bwd_bt, bwd_niter_pref)
    nG = T // (bwd_bt * niter)
    dx = torch.empty_like(x)
    dgp = torch.empty(nG, C, dtype=torch.float32, device=x.device)

    # raw weight; the kernel flips taps at load (conv-transpose) and reads
    # bf16 gamma directly -> no flip/t/float kernels on the host path.
    w16 = _bf16c(conv_weight)
    g16 = _bf16c(rms_weight)
    seq = _seq_i32(seq_id)
    args = (
        _pbf16(go), _pbf16(x), _pi32(seq), _pf32(inv_rms),
        _pbf16(w16), _pbf16(g16), _pbf16(dx), _pf32(dgp, 16),
        cutlass.Int32(T), cutlass.Int32(nG), _stream(),
    )
    fn = _compiled("bwd_dx", _bwd_dx_launch, args,
                   (bwd_bt, lanes, niter, lanes, True, bool(bwd_xs), bwd_minb))
    fn(*args)
    _maybe_sync("rms_conv_bwd_dx_dgamma")
    dgamma = dgp.sum(0)
    return dx, dgamma


@torch.library.impl(_lib, "rms_conv_bwd_dw_db", "CUDA")
@torch.no_grad()
def _bwd_dwdb_cuda(go, x, seq_id, inv_rms):
    T, C = x.shape
    go = go.detach().contiguous()
    x = x.detach().contiguous()
    inv_rms = inv_rms.detach().contiguous()
    lanes = _lanes(C)
    dwdb_bt, dwdb_niter_pref = _CFG[C][3], _CFG[C][4]
    niter = _pick_niter(T, dwdb_bt, dwdb_niter_pref)
    nG = T // (dwdb_bt * niter)
    dwp = torch.empty(nG, K + 1, C, dtype=torch.float32, device=x.device)
    seq = _seq_i32(seq_id)
    args = (
        _pbf16(go), _pbf16(x), _pi32(seq), _pf32(inv_rms), _pf32(dwp, 16),
        cutlass.Int32(T), cutlass.Int32(nG), _stream(),
    )
    fn = _compiled("dwdb", _dwdb_launch, args,
                   (dwdb_bt, lanes, niter, lanes, True))
    fn(*args)
    _maybe_sync("rms_conv_bwd_dw_db")
    red = dwp.sum(0)               # (K+1, C)
    dw_nogamma = red[:K].t().contiguous()   # (C, K), pre-gamma
    db = red[K]
    return dw_nogamma, db


_lib.define(
    "conv_fwd(Tensor x, Tensor seq_id, Tensor conv_weight, Tensor conv_bias) "
    "-> Tensor"
)
_lib.define(
    "conv_bwd_dx(Tensor go, Tensor seq_id, Tensor conv_weight) -> Tensor"
)
_lib.define(
    "conv_bwd_dw_db(Tensor go, Tensor x, Tensor seq_id) -> (Tensor, Tensor)"
)


@torch.library.register_fake("nanoplm_canon_cute::conv_fwd")
def _pconv_fwd_fake(x, seq_id, conv_weight, conv_bias):
    return torch.empty_like(x)


@torch.library.register_fake("nanoplm_canon_cute::conv_bwd_dx")
def _pconv_bwd_dx_fake(go, seq_id, conv_weight):
    return torch.empty_like(go)


@torch.library.register_fake("nanoplm_canon_cute::conv_bwd_dw_db")
def _pconv_bwd_dwdb_fake(go, x, seq_id):
    C = x.shape[1]
    dw = torch.empty(C, K, dtype=torch.float32, device=x.device)
    db = torch.empty(C, dtype=torch.float32, device=x.device)
    return dw, db


@torch.library.impl(_lib, "conv_fwd", "CUDA")
@torch.no_grad()
def _pconv_fwd_cuda(x, seq_id, conv_weight, conv_bias):
    """out = x + bias + conv(x) (residual included, matches plain canon)."""
    T, C = x.shape
    x = x.detach().contiguous()
    out = torch.empty_like(x)
    wf = conv_weight.detach().t().contiguous()      # (K, C) bf16, unfolded
    b16 = _bf16c(conv_bias)
    dummy = _dummy(x.device, C)
    seq = _seq_i32(seq_id)
    lanes = _lanes(C)
    fwd_bt, fwd_nthr = _CFG[C][0], _CFG[C][5] * lanes
    if C == 3072 and fwd_nthr == lanes and T % (fwd_bt * 2) == 0:
        niter = 2
        args = (
            _pbf16(x), _pbf16(out), _pi32(seq),
            _pbf16(wf), _pbf16(b16),
            cutlass.Int32(T), cutlass.Int32(T // (fwd_bt * niter)),
            _stream(),
        )
        fn = _compiled("pconv_fwd_persist", _pconv_fwd_persist_launch, args,
                       (fwd_bt, lanes, niter))
        fn(*args)
        _maybe_sync("conv_fwd_persist")
        return out

    args = (
        _pbf16(x), _pbf16(out), _pi32(seq), _pf32(dummy),
        _pbf16(wf), _pbf16(dummy), _pbf16(b16),
        cutlass.Float32(0.0), cutlass.Int32(T),
        cutlass.Int32(T // fwd_bt), _stream(),
    )
    fn = _compiled("pconv_fwd", _fwd_launch, args,
                   (fwd_bt, fwd_nthr, lanes, False, _CFG[C][8]))
    fn(*args)
    _maybe_sync("conv_fwd")
    return out


@torch.library.impl(_lib, "conv_bwd_dx", "CUDA")
@torch.no_grad()
def _pconv_bwd_dx_cuda(go, seq_id, conv_weight):
    T, C = go.shape
    go = go.detach().contiguous()
    dx = torch.empty_like(go)
    w16 = _bf16c(conv_weight)
    dummy = _dummy(go.device, C)
    lanes = _lanes(C)
    bwd_bt, bwd_niter_pref = _CFG[C][1], _CFG[C][2]
    # Plain Canon-B has no RMS reduction/dgamma work, so a longer persistent
    # span amortizes launch/grid overhead better at QKV width. RMS mode keeps
    # the shared _CFG value.
    if C == 3072:
        bwd_niter_pref = 64
    niter = _pick_niter(T, bwd_bt, bwd_niter_pref)
    nG = T // (bwd_bt * niter)
    seq = _seq_i32(seq_id)
    args = (
        _pbf16(go), _pbf16(go), _pi32(seq), _pf32(dummy),
        _pbf16(w16), _pbf16(dummy), _pbf16(dx), _pf32(dummy, 16),
        cutlass.Int32(T), cutlass.Int32(nG), _stream(),
    )
    # plain path: no x / no reduction -> register pressure is far lower, the
    # hybrid shape already reaches 8+ warps/SM; keep XSMEM/MINB off.
    fn = _compiled("pconv_bwd_dx", _bwd_dx_launch, args,
                   (bwd_bt, lanes, niter, lanes, False, False, 0))
    fn(*args)
    _maybe_sync("conv_bwd_dx")
    return dx


@torch.library.impl(_lib, "conv_bwd_dw_db", "CUDA")
@torch.no_grad()
def _pconv_bwd_dwdb_cuda(go, x, seq_id):
    T, C = x.shape
    go = go.detach().contiguous()
    x = x.detach().contiguous()
    lanes = _lanes(C)
    dwdb_bt, dwdb_niter_pref = _CFG[C][3], _CFG[C][4]
    # Plain Canon-B skips the RMS scale-table phase below. At C=3072 that
    # shifts the best persistence point: fewer partial CTAs wins without
    # hurting resident CTA count. Keep the shared _CFG value for RMS mode.
    if C == 3072:
        dwdb_niter_pref = 64
    niter = _pick_niter(T, dwdb_bt, dwdb_niter_pref)
    nG = T // (dwdb_bt * niter)
    dwp = torch.empty(nG, K + 1, C, dtype=torch.float32, device=x.device)
    dummy = _dummy(x.device, C)
    seq = _seq_i32(seq_id)
    args = (
        _pbf16(go), _pbf16(x), _pi32(seq), _pf32(dummy), _pf32(dwp, 16),
        cutlass.Int32(T), cutlass.Int32(nG), _stream(),
    )
    fn = _compiled("pconv_dwdb", _dwdb_launch, args,
                   (dwdb_bt, lanes, niter, lanes, False))
    fn(*args)
    _maybe_sync("conv_bwd_dw_db")
    red = dwp.sum(0)
    return red[:K].t().contiguous(), red[K]


def _pconv_setup_context(ctx, inputs, output):
    x, seq_id, conv_weight, _conv_bias = inputs
    ctx.save_for_backward(x, seq_id, conv_weight)


def _pconv_backward(ctx, grad_out):
    x, seq_id, conv_weight = ctx.saved_tensors
    grad_out = grad_out.contiguous()
    dx = torch.ops.nanoplm_canon_cute.conv_bwd_dx(grad_out, seq_id, conv_weight)
    dw, db = torch.ops.nanoplm_canon_cute.conv_bwd_dw_db(grad_out, x, seq_id)
    return dx, None, dw, db


torch.library.register_autograd(
    "nanoplm_canon_cute::conv_fwd",
    _pconv_backward,
    setup_context=_pconv_setup_context,
)


# ── autograd glue ────────────────────────────────────────────────────────


def _setup_context(ctx, inputs, output):
    x, seq_id, rms_weight, rms_eps, conv_weight, _conv_bias = inputs
    _out, inv_rms = output
    ctx.save_for_backward(x, seq_id, rms_weight, conv_weight, inv_rms)


def _backward(ctx, grad_out, _grad_inv):
    x, seq_id, rms_weight, conv_weight, inv_rms = ctx.saved_tensors
    grad_out = grad_out.contiguous()
    dx, dgamma = torch.ops.nanoplm_canon_cute.rms_conv_bwd_dx_dgamma(
        grad_out, x, seq_id, inv_rms, rms_weight, conv_weight)
    dw_nogamma, db = torch.ops.nanoplm_canon_cute.rms_conv_bwd_dw_db(
        grad_out, x, seq_id, inv_rms)
    # dw kernel accumulates go * x * inv (pre-gamma); true conv input is
    # RMS(x) = x * inv * gamma, so fold gamma per channel here.
    dw = dw_nogamma * rms_weight.float()[:, None]
    return dx, None, dgamma.to(rms_weight.dtype), None, dw, db


torch.library.register_autograd(
    "nanoplm_canon_cute::rms_conv_fwd",
    _backward,
    setup_context=_setup_context,
)


# ════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════


def is_compatible(x: torch.Tensor, radius: int) -> bool:
    """True when the CuTeDSL backend can handle this call."""
    if radius != R:
        return False
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2:
        return False
    T, C = x.shape
    if C not in _CFG:
        return False
    # All tile configs need T % 64 == 0 (fwd grid is exact; persistent
    # backward NITER relaxes via _pick_niter down to BT*1).
    if T % 64 != 0 or T % _CFG[C][0] != 0:
        return False
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        return False
    return True


def varlen_rms_canon_conv(
    x: torch.Tensor,
    seq_id: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Fused RMSNorm + Canon conv, CuTeDSL backend (sm90, bf16, C in 512-3072, K=7).

    Same contract as ``canon_ops.varlen_rms_canon_conv``.
    """
    assert radius == R, "CuTeDSL canon backend is compiled for K=7 only"
    out, _inv = torch.ops.nanoplm_canon_cute.rms_conv_fwd(
        x, seq_id, rms_weight, rms_eps, conv_weight, conv_bias)
    return out


def varlen_canon_conv_residual(
    x: torch.Tensor,
    seq_id: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Plain (un-normalized) varlen Canon conv, residual included:
    ``x + bias + conv(x)``. CuTeDSL backend, K=7 only."""
    assert radius == R, "CuTeDSL canon backend is compiled for K=7 only"
    return torch.ops.nanoplm_canon_cute.conv_fwd(x, seq_id, conv_weight,
                                                 conv_bias)
