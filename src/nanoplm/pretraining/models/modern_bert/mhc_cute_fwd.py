"""CuTeDSL mHC-lite pre-side forward (n=4, C=1024, sm90).

One 4-CTA cluster per 16-token tile row, persistent over tiles; CTA q owns column
quarter q of every stream (4 x 256 columns of the 4C row) and keeps that slice of
W (32 x 1024) resident in shared memory.
  * a loader warp TMA-loads the x tile (4 streams x 16 x 256, 128B-swizzled) into an
    NX-stage ring;
  * 8 producer warps compute the 32 partial projections (mma.sync) and the partial
    sums of squares, and push them to every CTA of the cluster (st.async + mbarrier);
  * consumer group A reduces them: h_pre for the pre-map first, then the full
    coefficient set (RMS, sigmoids, softmax over the 24 permutations, H) of the
    tokens this CTA owns;
  * consumer group B forms li = sum_j h_pre_j x_j from the staged tile (ldmatrix),
    writes it through stmatrix and stores it by TMA.
x is read from HBM once.

NOTE: no ``from __future__ import annotations``.
"""
import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.nvgpu import warpgroup
from cuda.bindings import driver as _cuda_drv

from .mhc_cute_common import (B_EMPTY, B_FULL, B_XFREE, B_XREADY, C, K, NO, NS, PAD, PSTR5, PV5, RS6, VEC, _fake, _ffi_compile, _max_clusters, _psrc)


NQ = 4
QC = C // NQ
KQ = NS * QC
QV = QC // VEC
SROW = KQ + PAD
BT = 16
NPR = 128                 # coefficient consumer threads (group A)
NPB = 128                 # pre-map consumer threads (group B)
B_HPF, B_HPE = 10, 12      # h_pre slot full / empty (A -> B handshake)
NPP = 256                 # producer threads (8 warps: half a stream's K each)
NX = 3
PFD = 2                   # producer ldmatrix prefetch distance (k-steps)
TX_X = NS * BT * QC * 2


@cute.jit
def _tma_x(ax, tXs, tXg, bar_x, xs, rt, lane, q):
    """Whole warp: TMA the 4 stream tiles of row-tile rt into x stage xs."""
    if lane == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar_x, TX_X)
    for u in cutlass.range_constexpr(NS):
        cute.copy(ax, tXg[(None, rt, u * NQ + q)], tXs[(None, xs * NS + u)], tma_bar_ptr=bar_x)


@cute.jit
def _recv_sum(sRecv, sl: cutlass.Int32, tt: cutlass.Int32, a: cutlass.Int32, b: cutlass.Int32,
              NQ: cutlass.Constexpr):
    """(projection partial sum [a, b], sum of squares) of token tt over the NQ column quarters."""
    v = sRecv[sl, 0, tt, a, b]
    fs = sRecv[sl, 0, tt, NO // 4, 0]
    for r in cutlass.range_constexpr(1, NQ):
        v = v + sRecv[sl, r, tt, a, b]
        fs = fs + sRecv[sl, r, tt, NO // 4, 0]
    return v, fs


@cute.jit
def _coeffs_from(v, fs, sCo, mProj, mHm, mHpre, mHpost, mRinv, psrc, t: cutlass.Int32,
                 lane: cutlass.Int32, eps: cutlass.Float32, merge: cutlass.Constexpr):
    """Full coefficient set of one owned token from its reduced sums (one warp, lane = output o)."""
    rinv = cute.math.rsqrt(fs / K + eps)
    pr = v * rinv
    al = sCo[NO + 2]
    if lane < 2 * NS:
        al = sCo[NO + 1]
    if lane < NS:
        al = sCo[NO]
    z = pr * al + sCo[lane]
    sig = 1.0 / (1.0 + cute.math.exp(-z))
    zr = z
    if lane < 2 * NS:
        zr = cutlass.Float32(-1.0e30)
    m = cute.arch.warp_reduction(zr, cute.arch.fmax)
    e = cute.math.exp(zr - m)
    ssum = cute.arch.warp_reduction(e, lambda a_, b_: a_ + b_)
    hres = cute.arch.shuffle_sync(e, 8 + psrc[0])
    for u in cutlass.range_constexpr(1, 6):
        hres = hres + cute.arch.shuffle_sync(e, 8 + psrc[u])
    hpo_i = cute.arch.shuffle_sync(sig, NS + (lane % 16) // NS)
    hpr_j = cute.arch.shuffle_sync(sig, lane % NS)
    hm = hres * (1.0 / ssum)
    if cutlass.const_expr(merge):
        # H_merged = H_res - h_post h_pre^T: the wrapped layer returns li + f (its own residual)
        hm = hm - (2.0 * hpo_i) * hpr_j
    mProj[t, lane] = pr
    if lane < 16:
        mHm[t, lane] = hm
    if lane < NS:
        mHpre[t, lane] = sig
    if lane >= NS and lane < 2 * NS:
        mHpost[t, lane - NS] = 2.0 * sig
    if lane == 0:
        mRinv[t] = rinv


@cute.kernel
def _pre_fwd7_kernel(
    ax: cute.CopyAtom, tma_x: cute.Tensor, ali: cute.CopyAtom, tma_li: cute.Tensor,
    slX: cute.ComposedLayout, slLi: cute.ComposedLayout,
    mWv: cute.Tensor, mAlpha: cute.Tensor, mBias: cute.Tensor,
    mHm: cute.Tensor, mHpost: cute.Tensor, mHpre: cute.Tensor,
    mProj: cute.Tensor, mRinv: cute.Tensor, mPsrc: cute.Tensor,
    eps: cutlass.Float32, ntiles: cutlass.Int32, ncl: cutlass.Int32,
    tiled_mma: cute.TiledMma, merge: cutlass.Constexpr,
):
    KG = NPP // 32
    KW = KQ // KG
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    q, _, _ = cute.arch.block_in_cluster_idx()
    cl = bidx // NQ
    warp = tidx // 32
    lane = tidx % 32

    alloc = cutlass.utils.SmemAllocator()
    sX = alloc.allocate_tensor(cute.BFloat16, slX.outer, 1024, swizzle=slX.inner)      # (BT, QC, NX*NS)
    sLi = alloc.allocate_tensor(cute.BFloat16, slLi.outer, 1024, swizzle=slLi.inner)   # (BT, QC, 2)
    sW = alloc.allocate_tensor(cute.BFloat16, cute.make_layout((NO, SROW), stride=(SROW, 1)), 128)
    sRecv = alloc.allocate_tensor(cutlass.Float32,
                                  cute.make_layout((2, NQ, BT, PV5, 4),
                                                   stride=(NQ * BT * PSTR5, BT * PSTR5, PSTR5, 4, 1)), 16)
    sBar = alloc.allocate_tensor(cutlass.Int64, cute.make_layout(14), 16)
    sRed = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((KG, BT, NO), stride=(BT * RS6, RS6, 1)), 16)
    sSsq = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((KG, BT), stride=(BT, 1)), 16)
    sHp = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((2, BT, NS), stride=(BT * NS, NS, 1)), 16)
    sCo = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(NO + 3), 16)
    sWv = cute.make_tensor(sW.iterator, cute.make_layout((NO, SROW // VEC, VEC), stride=(SROW, VEC, 1)))
    bar = sBar.iterator
    TXB = NQ * BT * PSTR5 * 4
    ntl = (ntiles - cl + ncl - 1) // ncl

    if tidx < NO:
        sCo[tidx] = mBias[tidx]
    if tidx < 3:
        sCo[NO + tidx] = mAlpha[tidx]
    if tidx == 0:
        for i in cutlass.range_constexpr(2):
            cute.arch.mbarrier_init(bar + (B_FULL + i), 1)
            cute.arch.mbarrier_init(bar + (B_EMPTY + i), NQ)
        for i in cutlass.range_constexpr(NX):
            cute.arch.mbarrier_init(bar + (B_XREADY + i), 1)
            cute.arch.mbarrier_init(bar + (B_XFREE + i), 1)
        for i in cutlass.range_constexpr(2):
            cute.arch.mbarrier_init(bar + (B_HPF + i), 1)
            cute.arch.mbarrier_init(bar + (B_HPE + i), 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.mbarrier_arrive_and_expect_tx(bar + B_FULL, TXB)
        if ntl > 1:
            cute.arch.mbarrier_arrive_and_expect_tx(bar + (B_FULL + 1), TXB)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    tXs, tXg = cpasync.tma_partition(ax, 0, cute.make_layout(1), cute.group_modes(sX, 0, 2),
                                     cute.group_modes(cute.local_tile(tma_x, (BT, QC), (None, None)), 0, 2))
    ldN = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cute.BFloat16)
    thr_mma = tiled_mma.get_slice(lane)
    cpA = cute.make_tiled_copy_A(ldN, tiled_mma).get_slice(lane)

    if warp < NPP // 32:
        # ================================================= producers
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cute.BFloat16, num_bits_per_copy=128)
        st_atom = cute.make_copy_atom(cpasync.CopyDsmemStoreOp(), cutlass.Float32, num_bits_per_copy=128)
        kg = warp                           # this warp's K slice: stream kg // 2, half kg % 2
        for c in cutlass.range_constexpr(NO // (NPP // (NS * QV))):
            o = tidx // (NS * QV) + c * (NPP // (NS * QV))
            vi = tidx % (NS * QV)
            cute.copy(atom, mWv[o, (vi // QV) * (C // VEC) + q * QV + vi % QV, None], sWv[o, vi, None])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(barrier_id=1, number_of_threads=NPP)

        cpB = cute.make_tiled_copy_B(ldN, tiled_mma).get_slice(lane)
        gB = cute.local_tile(sW, (NO, KW), (0, kg))
        tCsB = thr_mma.partition_B(gB)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tsB = cpB.partition_S(gB)
        trB = cpB.retile(tCrB)
        cCc = cute.make_identity_tensor((16, NO))
        tCcC = thr_mma.partition_C(cCc)
        tCrC = tiled_mma.make_fragment_C(tCcC.shape)
        fp = cute.make_fragment(cute.make_layout(4), cutlass.Float32)

        for n in cutlass.range(ntl):
            xs = n % NX
            sl = n % 2
            cute.arch.mbarrier_wait(bar + (B_XREADY + xs), (n // NX) % 2)
            gA = cute.local_tile(sX[None, None, xs * NS + kg // 2], (BT, KW), (0, kg % 2))
            tCsA = thr_mma.partition_A(gA)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tsA = cpA.partition_S(gA)
            trA = cpA.retile(tCrA)
            tCrC.fill(0.0)
            nkb = cute.size(tCrA, mode=[2])
            # sums of squares from the A fragments already in registers (elements 0,1,4,5: row g;
            # 2,3,6,7: row g+8), instead of a second ldmatrix of x + diag(X X^T) HMMAs
            sq0 = cutlass.Float32(0.0)
            sq1 = cutlass.Float32(0.0)
            for kb in cutlass.range_constexpr(min(PFD, nkb)):
                cute.copy(cpA, tsA[None, None, kb], trA[None, None, kb])
                cute.copy(cpB, tsB[None, None, kb], trB[None, None, kb])
            for kb in cutlass.range_constexpr(nkb):
                if cutlass.const_expr(kb + PFD < nkb):
                    cute.copy(cpA, tsA[None, None, kb + PFD], trA[None, None, kb + PFD])
                    cute.copy(cpB, tsB[None, None, kb + PFD], trB[None, None, kb + PFD])
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC)
                av = tCrA[None, None, kb].load().to(cutlass.Float32)
                sq0 = sq0 + av[0] * av[0] + av[1] * av[1] + av[4] * av[4] + av[5] * av[5]
                sq1 = sq1 + av[2] * av[2] + av[3] * av[3] + av[6] * av[6] + av[7] * av[7]
            for i in cutlass.range_constexpr(cute.size(tCrC)):
                rr, cc = tCcC[i]
                sRed[kg, rr, cc] = tCrC[i]
            sq0 = sq0 + cute.arch.shuffle_sync_bfly(sq0, 1)
            sq0 = sq0 + cute.arch.shuffle_sync_bfly(sq0, 2)
            sq1 = sq1 + cute.arch.shuffle_sync_bfly(sq1, 1)
            sq1 = sq1 + cute.arch.shuffle_sync_bfly(sq1, 2)
            if lane % 4 == 0:
                sSsq[kg, lane // 4] = sq0
                sSsq[kg, lane // 4 + 8] = sq1
            cute.arch.barrier(barrier_id=1, number_of_threads=NPP)
            if n >= 2:
                cute.arch.mbarrier_wait(bar + (B_EMPTY + sl), ((n - 2) // 2) % 2)

            for rnd in cutlass.range_constexpr((BT * PV5 + NPP - 1) // NPP):
                it = tidx + rnd * NPP
                if it < BT * PV5:
                    prow = it // PV5
                    cv = it % PV5
                    if cv < NO // 4:
                        for u in cutlass.range_constexpr(4):
                            acc_p = sRed[0, prow, cv * 4 + u]
                            for g in cutlass.range_constexpr(1, KG):
                                acc_p = acc_p + sRed[g, prow, cv * 4 + u]
                            fp[u] = acc_p
                    else:
                        ssq_p = sSsq[0, prow]
                        for g in cutlass.range_constexpr(1, KG):
                            ssq_p = ssq_p + sSsq[g, prow]
                        fp[0] = ssq_p
                        fp[1] = cutlass.Float32(0.0)
                        fp[2] = cutlass.Float32(0.0)
                        fp[3] = cutlass.Float32(0.0)
                    dloc = sRecv[sl, q, prow, cv, None]
                    for r in cutlass.range_constexpr(NQ):
                        dst = cute.make_tensor(cute.arch.map_dsmem_ptr(dloc.iterator, r), cute.make_layout(4))
                        cute.copy(st_atom, fp, dst, mbar_ptr=cute.arch.map_dsmem_ptr(bar + (B_FULL + sl), r))
            cute.arch.barrier(barrier_id=1, number_of_threads=NPP)
    elif warp < (NPP + NPR) // 32:
        # ================================================= consumers A: coefficients of tile n
        ctid = tidx - NPP
        cw = ctid // 32
        psrc = [mPsrc[lane % 16, u] for u in range(6)]
        for n in cutlass.range(ntl):
            sl = n % 2
            t0 = (cl + n * ncl) * BT
            cute.arch.mbarrier_wait(bar + (B_FULL + sl), (n // 2) % 2)
            # 1) everything this tile needs from the receive slot, into registers; then free the slot
            # h_pre inputs: only ctid < BT * 4 use them; the rest read an in-bounds
            # duplicate rather than tokens BT..2BT-1, past the slot (into sRed)
            ht = (ctid // 4) % BT
            hj = ctid % 4
            hv, hfs = _recv_sum(sRecv, sl, ht, 0, hj, NQ)
            tto = NQ * cw + q                                          # the owned token of this warp
            ov, ofs = _recv_sum(sRecv, sl, tto, lane // 4, lane % 4, NQ)
            cute.arch.barrier(barrier_id=2, number_of_threads=NPR)
            if ctid == 0:
                if n + 2 < ntl:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar + (B_FULL + sl), TXB)
                for r in cutlass.range_constexpr(NQ):
                    cute.arch.mbarrier_arrive(bar + (B_EMPTY + sl), r)
            # 2) h_pre for the pre-map warps
            if n >= 2:
                cute.arch.mbarrier_wait(bar + (B_HPE + sl), ((n - 2) // 2) % 2)
            if ctid < BT * 4:
                z = hv * cute.math.rsqrt(hfs / K + eps) * sCo[NO] + sCo[hj]
                sHp[sl, ht, hj] = 1.0 / (1.0 + cute.math.exp(-z))
            cute.arch.barrier(barrier_id=4, number_of_threads=NPR)
            if ctid == 0:
                cute.arch.mbarrier_arrive(bar + (B_HPF + sl))
            # 3) the owned tokens' full coefficient sets: global outputs only
            _coeffs_from(ov, ofs, sCo, mProj, mHm, mHpre, mHpost, mRinv, psrc, t0 + tto, lane, eps, merge)
    elif warp < (NPP + NPR + NPB) // 32:
        # ================================================= consumers B: pre-map of tile n
        btid = tidx - NPP - NPR
        cw = btid // 32
        stA = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), cute.BFloat16)
        cpS = cute.make_tiled_copy_C(stA, tiled_mma).get_slice(lane)
        cL = cute.make_identity_tensor((BT, 16))
        tCcL = thr_mma.partition_C(cL)
        fob = cute.make_fragment_like(tiled_mma.make_fragment_C(tCcL.shape), cute.BFloat16)
        tLs, tLg = cpasync.tma_partition(ali, 0, cute.make_layout(1), cute.group_modes(sLi, 0, 2),
                                         cute.group_modes(cute.local_tile(tma_li, (BT, QC), (None, None)), 0, 2))
        NB = QC // 16
        for n in cutlass.range(ntl):
            xs = n % NX
            sl = n % 2
            lb = n % 2
            if cw == 0:
                # the li staging buffer lb was last stored two tiles ago
                cute.arch.cp_async_bulk_wait_group(1, read=True)
            cute.arch.barrier(barrier_id=3, number_of_threads=NPB)
            cute.arch.mbarrier_wait(bar + (B_HPF + sl), (n // 2) % 2)
            cute.arch.mbarrier_wait(bar + (B_XREADY + xs), (n // NX) % 2)
            for bb in cutlass.range_constexpr(NB // (NPB // 32)):
                blk = cw * (NB // (NPB // 32)) + bb
                xf = []
                for j in cutlass.range_constexpr(NS):
                    g = cute.local_tile(sX[None, None, xs * NS + j], (BT, 16), (0, blk))
                    fr = tiled_mma.make_fragment_A(thr_mma.partition_A(g))
                    cute.copy(cpA, cpA.partition_S(g), cpA.retile(fr))
                    xf.append(fr.load().to(cutlass.Float32))
                for pv in cutlass.range_constexpr(4):
                    rr, cc = tCcL[2 * pv]
                    h = [sHp[sl, rr, j] for j in range(NS)]
                    for w2 in cutlass.range_constexpr(2):
                        e = 2 * pv + w2
                        acc = h[0] * xf[0][e]
                        for j in cutlass.range_constexpr(1, NS):
                            acc = acc + h[j] * xf[j][e]
                        fob[e] = acc.to(cute.BFloat16)
                gD = cute.local_tile(sLi[None, None, lb], (BT, 16), (0, blk))
                cute.copy(cpS, cpS.retile(fob), cpS.partition_D(gD))
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=3, number_of_threads=NPB)
            if btid == 0:
                cute.arch.mbarrier_arrive(bar + (B_XFREE + xs))
                cute.arch.mbarrier_arrive(bar + (B_HPE + sl))
            if cw == 0:
                cute.copy(ali, tLs[(None, lb)], tLg[(None, cl + n * ncl, q)])
                cute.arch.cp_async_bulk_commit_group()
        if cw == 0:
            cute.arch.cp_async_bulk_wait_group(0)
    else:
        # ================================================= loader warp
        # runs up to NX tiles ahead; only it ever waits for a stage to be freed
        for n in cutlass.range(ntl):
            xs = n % NX
            if n >= NX:
                cute.arch.mbarrier_wait(bar + (B_XFREE + xs), ((n - NX) // NX) % 2)
            _tma_x(ax, tXs, tXg, bar + (B_XREADY + xs), xs, cl + n * ncl, lane, q)

    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()


@cute.jit
def _pre_fwd7_launch(
    tX: cute.Tensor, tW: cute.Tensor, tAlpha: cute.Tensor, tBias: cute.Tensor,
    tLi: cute.Tensor, tHm: cute.Tensor, tHpost: cute.Tensor, tHpre: cute.Tensor,
    tProj: cute.Tensor, tRinv: cute.Tensor, tPsrc: cute.Tensor,
    eps: cutlass.Float32, T: cutlass.Int32, ntiles: cutlass.Int32, ncl: cutlass.Int32,
    stream: _cuda_drv.CUstream, merge: cutlass.Constexpr,
):
    atom_sw = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, cute.BFloat16)
    slX = cute.tile_to_shape(atom_sw, (BT, QC, NX * NS), order=(0, 1, 2))
    slLi = cute.tile_to_shape(atom_sw, (BT, QC, 2), order=(0, 1, 2))
    mX2 = cute.make_tensor(tX.iterator, cute.make_layout((T, K), stride=(K, 1)))
    mLi2 = cute.make_tensor(tLi.iterator, cute.make_layout((T, C), stride=(C, 1)))
    ax, tma_x = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mX2,
                                            cute.slice_(slX, (None, None, 0)), (BT, QC))
    ali, tma_li = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mLi2,
                                              cute.slice_(slLi, (None, None, 0)), (BT, QC))
    mWv = cute.make_tensor(tW.iterator, cute.make_layout((NO, K // VEC, VEC), stride=(K, VEC, 1)))
    mAlpha = cute.make_tensor(tAlpha.iterator, cute.make_layout(3))
    mBias = cute.make_tensor(tBias.iterator, cute.make_layout(NO))
    mHm = cute.make_tensor(tHm.iterator, cute.make_layout((T, 16), stride=(16, 1)))
    mHpost = cute.make_tensor(tHpost.iterator, cute.make_layout((T, NS), stride=(NS, 1)))
    mHpre = cute.make_tensor(tHpre.iterator, cute.make_layout((T, NS), stride=(NS, 1)))
    mProj = cute.make_tensor(tProj.iterator, cute.make_layout((T, NO), stride=(NO, 1)))
    mRinv = cute.make_tensor(tRinv.iterator, cute.make_layout(T))
    mPsrc = cute.make_tensor(tPsrc.iterator, cute.make_layout((16, 6), stride=(6, 1)))
    op = cute.nvgpu.warp.MmaF16BF16Op(cute.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 16, 16))
    _pre_fwd7_kernel(ax, tma_x, ali, tma_li, slX, slLi, mWv, mAlpha, mBias, mHm, mHpost, mHpre, mProj, mRinv,
                     mPsrc, eps, ntiles, ncl, tiled_mma, merge).launch(
        grid=(ncl * NQ, 1, 1), block=(NPP + NPR + NPB + 32, 1, 1), cluster=(NQ, 1, 1), stream=stream)


_ffi_cache = {}
def pre_fwd7(x, W, alpha3, bias, P=None, eps=1e-6, merge=True):
    """merge=True: Hm = H_res - h_post h_pre^T (layer returns li + f); False: Hm = H_res (layer returns f)."""
    T = x.shape[0]
    assert x.shape[1:] == (NS, C) and x.dtype == torch.bfloat16 and T % BT == 0
    x = x.detach().contiguous()
    Wb = W.detach().to(torch.bfloat16).contiguous()
    alpha3 = alpha3.detach().float().contiguous()
    bias = bias.detach().float().contiguous()
    dev = x.device
    li = torch.empty(T, C, device=dev, dtype=torch.bfloat16)
    Hm = torch.empty(T, 16, device=dev, dtype=torch.float32)
    hpost = torch.empty(T, NS, device=dev, dtype=torch.float32)
    hpre = torch.empty(T, NS, device=dev, dtype=torch.float32)
    proj = torch.empty(T, NO, device=dev, dtype=torch.float32)
    rinv = torch.empty(T, device=dev, dtype=torch.float32)
    ntiles = T // BT
    ncl = min(ntiles, _max_clusters(NQ))
    fn = _ffi_cache.get(("pre_fwd7", bool(merge)))
    if fn is None:
        bf, f32 = cute.BFloat16, cutlass.Float32
        fakes = (_fake(bf, (None, NS, C)), _fake(bf, (NO, K)), _fake(f32, (3,), 4), _fake(f32, (NO,), 4),
                 _fake(bf, (None, C)), _fake(f32, (None, 16)), _fake(f32, (None, NS)), _fake(f32, (None, NS)),
                 _fake(f32, (None, NO)), _fake(f32, (None,), 4), _fake(cutlass.Int32, (16, 6), 4))
        fn = _ffi_compile("pre_fwd7", _pre_fwd7_launch, fakes,
                          (cutlass.Float32(eps), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)), (bool(merge),))
        _ffi_cache[("pre_fwd7", bool(merge))] = fn
    fn(x, Wb, alpha3, bias, li, Hm, hpost, hpre, proj, rinv, _psrc(dev), eps, T, ntiles, ncl)
    return li, Hm, hpost, hpre, proj, rinv
