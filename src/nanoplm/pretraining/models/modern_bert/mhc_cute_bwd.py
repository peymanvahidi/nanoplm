"""CuTeDSL mHC-lite pre-side backward (n=4, C=1024, sm90).

Same 4-CTA cluster split as the forward (column quarter per CTA, resident W slice).
Per 16-token tile:
  * x, go (4 streams x 16 x 256 each), g_li and Hm arrive by TMA (issued by a loader
    warp, which also TMA-stores gx and refills the stages);
  * producer warps form <g_li, x_j> on tensor cores, exchange it (owner token only),
    and run the coefficient backward of their owned tokens in registers, giving
    g_proj * r and the RMSNorm term s, pushed back to every CTA;
  * gx warps write gx = H^T go + h_pre g_li + (g_proj r) W - s x in one pass, reading
    go / x / g_li as ldmatrix A-fragments that line up with the MMA C-fragments.
The W gradient is one bf16 GEMM (g_proj r)^T x outside the kernel.

NOTE: no ``from __future__ import annotations``.
"""
import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.nvgpu import warpgroup
from cuda.bindings import driver as _cuda_drv

from .mhc_cute_common import (BT, C, GPS, GRS, K, KQ, NGT, NGW, NO, NPT, NPW, NQ, NS, QC, QV, SGS, SROW, VEC, perm_gsrc_table, _coef_bwd_a, _coef_bwd_b, _fake, _ffi_compile, _max_clusters)


B_XR, B_GOR, B_GMF, B_GME, B_GF, B_GE, B_ED = 0, 2, 4, 6, 8, 10, 12
LDW = 1                   # loader warp: stage refills + gx TMA stores
TX_X = NS * BT * QC * 2 + BT * QC * 2          # x (4 streams) + g_li
TX_GO = NS * BT * QC * 2 + BT * 16 * 4         # go (4 streams) + Hm


@cute.jit
def _tma_x(ax, tXs, tXg, agl, tLs, tLg, bar_x, stg, rt, lane, q):
    """Whole warp: TMA x (4 streams) + g_li of row-tile rt into stage stg."""
    if lane == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar_x, TX_X)
    for u in cutlass.range_constexpr(NS):
        cute.copy(ax, tXg[(None, rt, u * NQ + q)], tXs[(None, stg * NS + u)], tma_bar_ptr=bar_x)
    cute.copy(agl, tLg[(None, rt, q)], tLs[(None, stg)], tma_bar_ptr=bar_x)


@cute.jit
def _tma_go(ago, tGs, tGg, ahm, tHs, tHg, bar_go, stg, rt, lane, q):
    """Whole warp: TMA go (4 streams) + Hm of row-tile rt into stage stg."""
    if lane == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(bar_go, TX_GO)
    for u in cutlass.range_constexpr(NS):
        cute.copy(ago, tGg[(None, rt, u * NQ + q)], tGs[(None, stg * NS + u)], tma_bar_ptr=bar_go)
    cute.copy(ahm, tHg[(None, rt, 0)], tHs[(None, stg)], tma_bar_ptr=bar_go)


@cute.kernel
def _pre_bwd3_kernel(
    ax: cute.CopyAtom, tma_x: cute.Tensor, ago: cute.CopyAtom, tma_go: cute.Tensor,
    agl: cute.CopyAtom, tma_gl: cute.Tensor, ahm: cute.CopyAtom, tma_hm: cute.Tensor,
    aox: cute.CopyAtom, tma_gx: cute.Tensor,
    slX: cute.ComposedLayout, slGl: cute.ComposedLayout, slHm: cute.Layout,
    mW: cute.Tensor, mAlpha: cute.Tensor, mBias: cute.Tensor,
    mProj: cute.Tensor, mRinv: cute.Tensor, mHpre: cute.Tensor,
    mGhm: cute.Tensor, mGhp: cute.Tensor, mGsrc: cute.Tensor,
    mGx: cute.Tensor, mGpr: cute.Tensor, mGab: cute.Tensor,
    ntiles: cutlass.Int32, ncl: cutlass.Int32,
    tiled_mma: cute.TiledMma, merge: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    q, _, _ = cute.arch.block_in_cluster_idx()
    cl = bidx // NQ
    warp = tidx // 32
    lane = tidx % 32
    ntl = (ntiles - cl + ncl - 1) // ncl

    alloc = cutlass.utils.SmemAllocator()
    sX = alloc.allocate_tensor(cute.BFloat16, slX.outer, 1024, swizzle=slX.inner)      # (BT, QC, 2*NS)
    sGo = alloc.allocate_tensor(cute.BFloat16, slX.outer, 1024, swizzle=slX.inner)
    sGl = alloc.allocate_tensor(cute.BFloat16, slGl.outer, 1024, swizzle=slGl.inner)   # (BT, QC, 2)
    sHm = alloc.allocate_tensor(cutlass.Float32, slHm, 128)                           # (BT, 16, 2)
    sW = alloc.allocate_tensor(cute.BFloat16, cute.make_layout((NO, SROW), stride=(SROW, 1)), 128)
    sHp = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((2, BT, NS), stride=(BT * NS, NS, 1)), 16)
    recvGm = alloc.allocate_tensor(cutlass.Float32,
                                   cute.make_layout((2, NQ, BT // NQ, NS), stride=(NQ * (BT // NQ) * NS, (BT // NQ) * NS, NS, 1)), 16)
    recvG = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((2, BT, GRS), stride=(BT * GRS, GRS, 1)), 16)
    sGmP = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((BT, NS), stride=(NS, 1)), 16)
    sG = alloc.allocate_tensor(cute.BFloat16, cute.make_layout((BT, SGS, 2), stride=(SGS, 1, BT * SGS)), 128)
    sS = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((BT, 2), stride=(1, BT)), 16)
    sCo = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(NO + 3), 16)
    sBar = alloc.allocate_tensor(cutlass.Int64, cute.make_layout(14), 16)
    bar = sBar.iterator
    TX_GM = NQ * (BT // NQ) * NS * 4
    TX_G = BT * GPS * 4

    if tidx < NO:
        sCo[tidx] = mBias[tidx]
    if tidx < 3:
        sCo[NO + tidx] = mAlpha[tidx]
    if tidx == 0:
        for i in cutlass.range_constexpr(2):
            cute.arch.mbarrier_init(bar + (B_XR + i), 1)
            cute.arch.mbarrier_init(bar + (B_GOR + i), 1)
            cute.arch.mbarrier_init(bar + (B_GMF + i), 1)
            cute.arch.mbarrier_init(bar + (B_GME + i), NQ)
            cute.arch.mbarrier_init(bar + (B_GF + i), 1)
            cute.arch.mbarrier_init(bar + (B_GE + i), NQ)
            cute.arch.mbarrier_init(bar + (B_ED + i), NGT)
        cute.arch.mbarrier_init_fence()
        cute.arch.mbarrier_arrive_and_expect_tx(bar + B_GMF, TX_GM)
        cute.arch.mbarrier_arrive_and_expect_tx(bar + B_GF, TX_G)
        if ntl > 1:
            cute.arch.mbarrier_arrive_and_expect_tx(bar + (B_GMF + 1), TX_GM)
            cute.arch.mbarrier_arrive_and_expect_tx(bar + (B_GF + 1), TX_G)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    # TMA partitions
    tXs, tXg = cpasync.tma_partition(ax, 0, cute.make_layout(1), cute.group_modes(sX, 0, 2),
                                     cute.group_modes(cute.local_tile(tma_x, (BT, QC), (None, None)), 0, 2))
    tGs, tGg = cpasync.tma_partition(ago, 0, cute.make_layout(1), cute.group_modes(sGo, 0, 2),
                                     cute.group_modes(cute.local_tile(tma_go, (BT, QC), (None, None)), 0, 2))
    tLs, tLg = cpasync.tma_partition(agl, 0, cute.make_layout(1), cute.group_modes(sGl, 0, 2),
                                     cute.group_modes(cute.local_tile(tma_gl, (BT, QC), (None, None)), 0, 2))
    tHs, tHg = cpasync.tma_partition(ahm, 0, cute.make_layout(1), cute.group_modes(sHm, 0, 2),
                                     cute.group_modes(cute.local_tile(tma_hm, (BT, 16), (None, None)), 0, 2))

    ldN = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cute.BFloat16)
    thr_mma = tiled_mma.get_slice(lane)

    if warp < NPW:
        # ====================================================== producers
        j = warp
        st16 = cute.make_copy_atom(cpasync.CopyDsmemStoreOp(), cutlass.Float32, num_bits_per_copy=128)
        st4 = cute.make_copy_atom(cpasync.CopyDsmemStoreOp(), cutlass.Float32, num_bits_per_copy=32)
        gsrc = [mGsrc[(lane + 16) % 24, u] for u in range(NS)]
        cpA = cute.make_tiled_copy_A(ldN, tiled_mma).get_slice(lane)
        cpB = cute.make_tiled_copy_B(ldN, tiled_mma).get_slice(lane)
        cD = cute.make_identity_tensor((16, 16))
        tCcD = thr_mma.partition_C(cD)
        tCrD = tiled_mma.make_fragment_C(tCcD.shape)
        tCrD1 = tiled_mma.make_fragment_C(tCcD.shape)
        tCrD2 = tiled_mma.make_fragment_C(tCcD.shape)
        tCrD3 = tiled_mma.make_fragment_C(tCcD.shape)
        accs = [tCrD, tCrD1, tCrD2, tCrD3]
        f4 = cute.make_fragment(cute.make_layout(4), cutlass.Float32)
        f1 = cute.make_fragment(cute.make_layout(1), cutlass.Float32)
        acc_gz = cutlass.Float32(0.0)
        acc_ga = cutlass.Float32(0.0)
        tt = NS * warp + q
        # per-token coefficient inputs, prefetched one tile ahead
        nx_pr = mProj[cl * BT + tt, lane]
        nx_r = mRinv[cl * BT + tt]
        nx_ghm = mGhm[cl * BT + tt, lane % 16]
        nx_ghp = mGhp[cl * BT + tt, lane % NS]
        for n in cutlass.range(ntl):
            stg = n % 2
            sl = n % 2
            t0 = (cl + n * ncl) * BT
            tg = t0 + tt
            in_pr = nx_pr
            in_r = nx_r
            in_ghm = nx_ghm
            in_ghp = nx_ghp
            if n + 1 < ntl:
                nx_pr = mProj[tg + ncl * BT, lane]
                nx_r = mRinv[tg + ncl * BT]
                nx_ghm = mGhm[tg + ncl * BT, lane % 16]
                nx_ghp = mGhp[tg + ncl * BT, lane % NS]
            # everything of the coefficient backward that does not need gm: off the x-wait chain
            al, sig, acc_pre, gz0 = _coef_bwd_a(sCo, in_pr, in_ghm, in_ghp, gsrc, lane, merge)
            cute.arch.mbarrier_wait(bar + (B_XR + stg), (n // 2) % 2)
            gA = sGl[None, None, stg]
            gB = sX[None, None, stg * NS + j]
            tCsA = thr_mma.partition_A(gA)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tsA = cpA.partition_S(gA)
            trA = cpA.retile(tCrA)
            tCsB = thr_mma.partition_B(gB)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tsB = cpB.partition_S(gB)
            trB = cpB.retile(tCrB)
            for a_ in accs:
                a_.fill(0.0)
            # 4 independent accumulator chains (a single one is HMMA-latency bound)
            for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                cute.copy(cpA, tsA[None, None, kb], trA[None, None, kb])
                cute.copy(cpB, tsB[None, None, kb], trB[None, None, kb])
                cute.gemm(tiled_mma, accs[kb % 4], tCrA[None, None, kb], tCrB[None, None, kb], accs[kb % 4])
            for i in cutlass.range_constexpr(cute.size(tCrD)):
                tCrD[i] = (tCrD[i] + tCrD1[i]) + (tCrD2[i] + tCrD3[i])
            for i in cutlass.range_constexpr(cute.size(tCrD)):
                rr, cc = tCcD[i]
                if rr == cc:
                    sGmP[rr, j] = tCrD[i]
            cute.arch.barrier(barrier_id=1, number_of_threads=NPT)
            if n >= 2:
                cute.arch.mbarrier_wait(bar + (B_GME + sl), ((n - 2) // 2) % 2)
            if tidx < BT:
                for u in cutlass.range_constexpr(NS):
                    f4[u] = sGmP[tidx, u]
                dloc = recvGm[sl, q, tidx // NQ, None]
                own = tidx % NQ
                dst = cute.make_tensor(cute.arch.map_dsmem_ptr(dloc.iterator, own), cute.make_layout(4))
                cute.copy(st16, f4, dst, mbar_ptr=cute.arch.map_dsmem_ptr(bar + (B_GMF + sl), own))
            cute.arch.mbarrier_wait(bar + (B_GMF + sl), (n // 2) % 2)

            gm = recvGm[sl, 0, warp, lane % NS]
            for rr in cutlass.range_constexpr(1, NQ):
                gm = gm + recvGm[sl, rr, warp, lane % NS]
            gpr, s_t, gz = _coef_bwd_b(gm, al, sig, acc_pre, gz0, in_pr, in_r, lane)
            acc_gz = acc_gz + gz
            acc_ga = acc_ga + gz * in_pr
            mGpr[tg, lane] = gpr
            cute.arch.barrier(barrier_id=1, number_of_threads=NPT)
            if tidx == 0:
                if n + 2 < ntl:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar + (B_GMF + sl), TX_GM)
                for rr in cutlass.range_constexpr(NQ):
                    cute.arch.mbarrier_arrive(bar + (B_GME + sl), rr)
            if n >= 2:
                cute.arch.mbarrier_wait(bar + (B_GE + sl), ((n - 2) // 2) % 2)
            f1[0] = gpr
            for rr in cutlass.range_constexpr(NQ):
                mb = cute.arch.map_dsmem_ptr(bar + (B_GF + sl), rr)
                dst = cute.make_tensor(cute.arch.map_dsmem_ptr(
                    recvG.iterator + (sl * (BT * GRS) + tt * GRS + lane), rr), cute.make_layout(1))
                cute.copy(st4, f1, dst, mbar_ptr=mb)
            if lane == 0:
                f1[0] = s_t
                for rr in cutlass.range_constexpr(NQ):
                    mb = cute.arch.map_dsmem_ptr(bar + (B_GF + sl), rr)
                    dst = cute.make_tensor(cute.arch.map_dsmem_ptr(
                        recvG.iterator + (sl * (BT * GRS) + tt * GRS + NO), rr), cute.make_layout(1))
                    cute.copy(st4, f1, dst, mbar_ptr=mb)
        cute.arch.atomic_add(mGab.iterator + (3 + lane), acc_gz)
        v0 = cutlass.Float32(0.0)
        v1 = cutlass.Float32(0.0)
        v2 = acc_ga
        if lane < 2 * NS:
            v2 = cutlass.Float32(0.0)
            v1 = acc_ga
            if lane < NS:
                v1 = cutlass.Float32(0.0)
                v0 = acc_ga
        g0 = cute.arch.warp_reduction(v0, lambda a_, b_: a_ + b_)
        g1 = cute.arch.warp_reduction(v1, lambda a_, b_: a_ + b_)
        g2 = cute.arch.warp_reduction(v2, lambda a_, b_: a_ + b_)
        if lane == 0:
            cute.arch.atomic_add(mGab.iterator + 0, g0)
            cute.arch.atomic_add(mGab.iterator + 1, g1)
            cute.arch.atomic_add(mGab.iterator + 2, g2)
    elif warp < NPW + NGW:
        # ====================================================== gx warps
        gt = tidx - NPT
        gw = warp - NPW
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cute.BFloat16, num_bits_per_copy=128)
        sWv = cute.make_tensor(sW.iterator, cute.make_layout((NO, SROW // VEC, VEC), stride=(SROW, VEC, 1)))
        # resident W slice (natural column order), once
        for w in cutlass.range_constexpr(NO // (NGT // (NS * QV))):
            o = gt // (NS * QV) + w * (NGT // (NS * QV))
            vi = gt % (NS * QV)
            cute.copy(atom, mW[o, (vi // QV) * (C // VEC) + q * QV + vi % QV, None], sWv[o, vi, None])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(barrier_id=2, number_of_threads=NGT)

        sGv = cute.make_tensor(sG.iterator, cute.make_layout((BT, NO, 2), stride=(SGS, 1, BT * SGS)))
        sWT = cute.make_tensor(sW.iterator, cute.make_layout((KQ, NO), stride=(1, SROW)))
        ldT = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), cute.BFloat16)
        cpA = cute.make_tiled_copy_A(ldN, tiled_mma).get_slice(lane)
        cpB = cute.make_tiled_copy_B(ldT, tiled_mma).get_slice(lane)
        tCsA = thr_mma.partition_A(sGv[None, None, 0])
        tCrA = tiled_mma.make_fragment_A(tCsA)
        trA = cpA.retile(tCrA)
        CW = QC // NGW                      # 32 columns per warp, all 4 streams
        NCH = 16
        cC = cute.make_identity_tensor((BT, NCH))
        tCcC = thr_mma.partition_C(cC)
        tC0 = tiled_mma.make_fragment_C(tCcC.shape)
        stA = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), cute.BFloat16)
        cpS = cute.make_tiled_copy_C(stA, tiled_mma).get_slice(lane)
        fob = cute.make_fragment_like(tC0, cute.BFloat16)
        f4h = cute.make_fragment(cute.make_layout(4), cutlass.Float32)
        sHm4 = cute.make_tensor(sHm.iterator, cute.make_layout((BT, NS, NS, 2), stride=(16, NS, 1, BT * 16)))
        sHp4 = cute.make_tensor(sHp.iterator, cute.make_layout((2, BT, NS), stride=(BT * NS, NS, 1)))
        # hpre rows are prefetched one tile ahead (a global load on the critical path otherwise)
        hpf = cute.make_fragment(cute.make_layout(NS), cutlass.Float32)
        if gt < BT:
            for u in cutlass.range_constexpr(NS):
                hpf[u] = mHpre[cl * BT + gt, u]
        for n in cutlass.range(ntl):
            stg = n % 2
            sl = n % 2
            t0 = (cl + n * ncl) * BT
            cute.arch.mbarrier_wait(bar + (B_GF + sl), (n // 2) % 2)
            cute.arch.mbarrier_wait(bar + (B_GOR + stg), (n // 2) % 2)
            cute.arch.mbarrier_wait(bar + (B_XR + stg), (n // 2) % 2)
            for e in cutlass.range_constexpr((BT * NO) // NGT):
                idx = gt + e * NGT
                sG[idx // NO, idx % NO, stg] = cutlass.BFloat16(recvG[sl, idx // NO, idx % NO])
            if gt < BT:
                sS[gt, stg] = recvG[sl, gt, NO]
                for u in cutlass.range_constexpr(NS):
                    sHp[stg, gt, u] = hpf[u]
                if n + 1 < ntl:
                    for u in cutlass.range_constexpr(NS):
                        hpf[u] = mHpre[t0 + ncl * BT + gt, u]
            cute.arch.barrier(barrier_id=2, number_of_threads=NGT)
            if gt == 0:
                if n + 2 < ntl:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar + (B_GF + sl), TX_G)
                for rr in cutlass.range_constexpr(NQ):
                    cute.arch.mbarrier_arrive(bar + (B_GE + sl), rr)
            tsA = cpA.partition_S(sGv[None, None, stg])
            for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                cute.copy(cpA, tsA[None, None, kb], trA[None, None, kb])
            # this lane's two token rows (g, g+8): all per-row scalars once per tile
            rws = [tCcC[0][0], tCcC[2][0]]
            hmv = []
            hpv = []
            svv = []
            for h2 in cutlass.range_constexpr(2):
                rr0 = rws[h2]
                hrow = []
                for u in cutlass.range_constexpr(NS):
                    cute.autovec_copy(sHm4[rr0, u, None, stg], f4h)
                    hrow.append(f4h.load())
                hmv.append(hrow)
                cute.autovec_copy(sHp4[stg, rr0, None], f4h)
                hpv.append(f4h.load())
                svv.append(sS[rr0, stg])
            for ch in cutlass.range_constexpr(CW // NCH):
                cq0 = gw * CW + ch * NCH
                # operands of this 16 x 16 block as A-fragments: element v lines up with the C fragment's v
                def _afrag(src2d):
                    g = cute.local_tile(src2d, (BT, NCH), (0, cq0 // NCH))
                    fr = tiled_mma.make_fragment_A(thr_mma.partition_A(g))
                    cute.copy(cpA, cpA.partition_S(g), cpA.retile(fr))
                    return fr.load().to(cutlass.Float32)
                gov = [_afrag(sGo[None, None, stg * NS + u]) for u in range(NS)]
                gl = _afrag(sGl[None, None, stg])
                for jj in cutlass.range_constexpr(NS):
                    # one stream at a time: a single live accumulator
                    gB = cute.local_tile(sWT, (NCH, NO), ((jj * QC + cq0) // NCH, 0))
                    tCsB = thr_mma.partition_B(gB)
                    tCrB = tiled_mma.make_fragment_B(tCsB)
                    tsB = cpB.partition_S(gB)
                    trB = cpB.retile(tCrB)
                    tC0.fill(0.0)
                    for kb in cutlass.range_constexpr(cute.size(tCrB, mode=[2])):
                        cute.copy(cpB, tsB[None, None, kb], trB[None, None, kb])
                        cute.gemm(tiled_mma, tC0, tCrA[None, None, kb], tCrB[None, None, kb], tC0)
                    xv = _afrag(sX[None, None, stg * NS + jj])
                    for pv in cutlass.range_constexpr(4):
                        e0 = 2 * pv
                        h2 = pv % 2                      # elements (0,1),(4,5): row g ; (2,3),(6,7): row g+8
                        hp = hpv[h2][jj]
                        sv = svv[h2]
                        hmr = [hmv[h2][u][jj] for u in range(NS)]
                        for w2 in cutlass.range_constexpr(2):
                            e = e0 + w2
                            acc = tC0[e]
                            for u in cutlass.range_constexpr(NS):
                                acc = acc + hmr[u] * gov[u][e]
                            fob[e] = (acc + hp * gl[e] - sv * xv[e]).to(cute.BFloat16)
                    # this chunk's go values are already in registers (gov): its
                    # stage block is reused as the output staging buffer
                    gD = cute.local_tile(sGo[None, None, stg * NS + jj], (BT, NCH), (0, cq0 // NCH))
                    cute.copy(cpS, cpS.retile(fob), cpS.partition_D(gD))
            # staged gx visible to the async proxy, stage done with: hand it to the loader warp
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(bar + (B_ED + stg))
    else:
        # ====================================================== loader warpgroup
        if warp == NPW + NGW:
            _tma_x(ax, tXs, tXg, agl, tLs, tLg, bar + B_XR, 0, cl, lane, q)
            _tma_go(ago, tGs, tGg, ahm, tHs, tHg, bar + B_GOR, 0, cl, lane, q)
            if ntl > 1:
                _tma_x(ax, tXs, tXg, agl, tLs, tLg, bar + (B_XR + 1), 1, cl + ncl, lane, q)
                _tma_go(ago, tGs, tGg, ahm, tHs, tHg, bar + (B_GOR + 1), 1, cl + ncl, lane, q)
            tOs, tOg = cpasync.tma_partition(aox, 0, cute.make_layout(1), cute.group_modes(sGo, 0, 2),
                                             cute.group_modes(cute.local_tile(tma_gx, (BT, QC), (None, None)), 0, 2))
            for n in cutlass.range(ntl):
                stg = n % 2
                cute.arch.mbarrier_wait(bar + (B_ED + stg), (n // 2) % 2)
                # x / g_li of tile n+2 first: the producer chain waits on them
                if n + 2 < ntl:
                    _tma_x(ax, tXs, tXg, agl, tLs, tLg, bar + (B_XR + stg), stg, cl + (n + 2) * ncl, lane, q)
                for u in cutlass.range_constexpr(NS):
                    cute.copy(aox, tOs[(None, stg * NS + u)], tOg[(None, cl + n * ncl, u * NQ + q)])
                cute.arch.cp_async_bulk_commit_group()
                # go stage doubles as the gx staging buffer: refill once the store has read it
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                if n + 2 < ntl:
                    _tma_go(ago, tGs, tGg, ahm, tHs, tHg, bar + (B_GOR + stg), stg, cl + (n + 2) * ncl, lane, q)
            cute.arch.cp_async_bulk_wait_group(0)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()


@cute.jit
def _pre_bwd3_launch(
    tX: cute.Tensor, tGo: cute.Tensor, tGl: cute.Tensor, tW: cute.Tensor,
    tAlpha: cute.Tensor, tBias: cute.Tensor, tProj: cute.Tensor, tRinv: cute.Tensor,
    tHm: cute.Tensor, tHpre: cute.Tensor, tGhm: cute.Tensor, tGhp: cute.Tensor, tGsrc: cute.Tensor,
    tGx: cute.Tensor, tGpr: cute.Tensor, tGab: cute.Tensor,
    T: cutlass.Int32, ntiles: cutlass.Int32, ncl: cutlass.Int32, stream: _cuda_drv.CUstream,
    merge: cutlass.Constexpr,
):
    mX2 = cute.make_tensor(tX.iterator, cute.make_layout((T, K), stride=(K, 1)))
    mGo2 = cute.make_tensor(tGo.iterator, cute.make_layout((T, K), stride=(K, 1)))
    mGl2 = cute.make_tensor(tGl.iterator, cute.make_layout((T, C), stride=(C, 1)))
    mHm2 = cute.make_tensor(tHm.iterator, cute.make_layout((T, 16), stride=(16, 1)))
    atom_sw = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, cute.BFloat16)
    slX = cute.tile_to_shape(atom_sw, (BT, QC, 2 * NS), order=(0, 1, 2))
    slGl = cute.tile_to_shape(atom_sw, (BT, QC, 2), order=(0, 1, 2))
    slHm = cute.make_layout((BT, 16, 2), stride=(16, 1, BT * 16))
    g2s = cpasync.CopyBulkTensorTileG2SOp()
    ax, tma_x = cpasync.make_tiled_tma_atom(g2s, mX2, cute.slice_(slX, (None, None, 0)), (BT, QC))
    ago, tma_go = cpasync.make_tiled_tma_atom(g2s, mGo2, cute.slice_(slX, (None, None, 0)), (BT, QC))
    agl, tma_gl = cpasync.make_tiled_tma_atom(g2s, mGl2, cute.slice_(slGl, (None, None, 0)), (BT, QC))
    ahm, tma_hm = cpasync.make_tiled_tma_atom(g2s, mHm2, cute.slice_(slHm, (None, None, 0)), (BT, 16))
    mGxT = cute.make_tensor(tGx.iterator, cute.make_layout((T, K), stride=(K, 1)))
    aox, tma_gx = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mGxT,
                                              cute.slice_(slX, (None, None, 0)), (BT, QC))
    mW = cute.make_tensor(tW.iterator, cute.make_layout((NO, K // VEC, VEC), stride=(K, VEC, 1)))
    mAlpha = cute.make_tensor(tAlpha.iterator, cute.make_layout(3))
    mBias = cute.make_tensor(tBias.iterator, cute.make_layout(NO))
    mProj = cute.make_tensor(tProj.iterator, cute.make_layout((T, NO), stride=(NO, 1)))
    mRinv = cute.make_tensor(tRinv.iterator, cute.make_layout(T))
    mHpre = cute.make_tensor(tHpre.iterator, cute.make_layout((T, NS), stride=(NS, 1)))
    mGhm = cute.make_tensor(tGhm.iterator, cute.make_layout((T, 16), stride=(16, 1)))
    mGhp = cute.make_tensor(tGhp.iterator, cute.make_layout((T, NS), stride=(NS, 1)))
    mGsrc = cute.make_tensor(tGsrc.iterator, cute.make_layout((24, NS), stride=(NS, 1)))
    mGx = cute.make_tensor(tGx.iterator, cute.make_layout((T, NS, C), stride=(K, C, 1)))
    mGpr = cute.make_tensor(tGpr.iterator, cute.make_layout((T, NO), stride=(NO, 1)))
    mGab = cute.make_tensor(tGab.iterator, cute.make_layout(3 + NO))
    op = cute.nvgpu.warp.MmaF16BF16Op(cute.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 16, 16))
    _pre_bwd3_kernel(ax, tma_x, ago, tma_go, agl, tma_gl, ahm, tma_hm, aox, tma_gx, slX, slGl, slHm,
                     mW, mAlpha, mBias, mProj, mRinv, mHpre, mGhm, mGhp, mGsrc, mGx, mGpr, mGab,
                     ntiles, ncl, tiled_mma, merge).launch(
        grid=(ncl * NQ, 1, 1), block=(NPT + NGT + 32 * LDW, 1, 1), cluster=(NQ, 1, 1), stream=stream)


_GSRC = {}
_ffi_cache = {}
def pre_bwd(x, go, g_li, W, alpha3, bias, proj, rinv, Hm, hpre, g_Hm, g_hpost, merge=True):
    """-> gx (T,4,C) bf16, gpr (T,32) fp32 (= g_proj * r, for the dW GEMM), galpha (3,), gbias (32,).
    merge must match the forward: True when Hm = H_res - h_post h_pre^T, False when Hm = H_res."""
    T = x.shape[0]
    assert T % BT == 0
    dev = x.device
    x = x.detach().contiguous()
    go = go.detach().contiguous()
    g_li = g_li.detach().contiguous()
    Wb = W.detach().to(torch.bfloat16).contiguous()
    alpha3 = alpha3.detach().float().contiguous()
    bias = bias.detach().float().contiguous()
    g_Hm = g_Hm.detach().reshape(T, 16).float().contiguous()
    g_hpost = g_hpost.detach().float().contiguous()
    if dev not in _GSRC:
        _GSRC[dev] = perm_gsrc_table(dev)
    gx = torch.empty_like(x)
    gpr = torch.empty(T, NO, device=dev, dtype=torch.float32)
    gab = torch.zeros(3 + NO, device=dev, dtype=torch.float32)
    ntiles = T // BT
    ncl = min(ntiles, _max_clusters(NQ))
    fn = _ffi_cache.get(("pre_bwd3", bool(merge)))
    if fn is None:
        bf, f32 = cute.BFloat16, cutlass.Float32
        fakes = (_fake(bf, (None, NS, C)), _fake(bf, (None, NS, C)), _fake(bf, (None, C)), _fake(bf, (NO, K)),
                 _fake(f32, (3,), 4), _fake(f32, (NO,), 4), _fake(f32, (None, NO)), _fake(f32, (None,), 4),
                 _fake(f32, (None, 16)), _fake(f32, (None, NS)), _fake(f32, (None, 16)), _fake(f32, (None, NS)),
                 _fake(cutlass.Int32, (24, NS), 4),
                 _fake(bf, (None, NS, C)), _fake(f32, (None, NO)), _fake(f32, (3 + NO,), 4))
        fn = _ffi_compile("pre_bwd3", _pre_bwd3_launch, fakes,
                          (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)), (bool(merge),),
                          )
        _ffi_cache[("pre_bwd3", bool(merge))] = fn
    fn(x, go, g_li, Wb, alpha3, bias, proj, rinv, Hm, hpre, g_Hm, g_hpost, _GSRC[dev], gx, gpr, gab,
       T, ntiles, ncl)
    return gx, gpr, gab[:3], gab[3:]
