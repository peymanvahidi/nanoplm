"""v2 fused mHC-lite (n=4) kernels: fewer passes over the 4-wide residual stream.

Forward
  pre_fwd   one kernel: RMS over the 4C row, 32-way projection, all
            coefficients (h_pre, h_post, softmax over 24 permutations,
            H_merged = H_res - h_post (x) h_pre) and the pre-map
            layer_input = sum_j h_pre_j x_j. x is read from HBM once (the
            pre-map pass re-reads the CTA's own tile, which is L2-resident).
  post_fwd  existing kernel: out = H_merged @ x + h_post (x) layer_output.
Backward
  post_bwd  existing kernel: g_lo, gx_post = H_merged^T go, g_Hm, g_hpost.
  pre_bwd   one kernel: pre-map grad, the whole coefficient backward in
            registers, and the RMSNorm+projection input grad, written as
            gx = gx_post + h_pre_j g_li + r (g_xn - x_n (g_proj . proj)/K)
            in a single pass. The RMSNorm reduction g_xn . x_n is replaced by
            g_proj . proj (32 values), so no extra pass over x.
  W grad    one bf16 GEMM (g_proj * r)^T @ x.
gx_post reaches pre_bwd as the gradient of an alias of x that pre returns and
post consumes, so autograd never materialises a separate gx sum.

Math matches MHCLiteBlock's PyTorch path (RMSNorm eps 1e-6).
"""
import itertools

import torch
import triton
import triton.language as tl


def perm_matrix_padded(n: int = 4, device="cuda") -> torch.Tensor:
    """(32, 16) fp32: rows 8..31 are the 24 flattened permutation matrices,
    rows 0..7 zero, so a (T,32) coefficient row can dot it directly."""
    perms = list(itertools.permutations(range(n)))
    P = torch.zeros(2 * n + len(perms), n * n, device=device, dtype=torch.float32)
    for i, p in enumerate(perms):
        for row, col in enumerate(p):
            P[2 * n + i, row * n + col] = 1.0
    return P


@triton.jit
def _coeffs(proj, o, P_ptr, alpha_ptr, bias_ptr, MERGE: tl.constexpr):
    """From proj (BT,32) = x_norm @ W^T: z, sigmoid, softmax, H_merged, and the
    per-stream h_pre / h_post scalars. Returns everything backward needs."""
    a_pre = tl.load(alpha_ptr + 0)
    a_post = tl.load(alpha_ptr + 1)
    a_res = tl.load(alpha_ptr + 2)
    b = tl.load(bias_ptr + o)
    alpha_vec = tl.where(o < 4, a_pre, tl.where(o < 8, a_post, a_res))
    z = proj * alpha_vec[None, :] + b[None, :]
    sig = 1.0 / (1.0 + tl.exp(-z))
    is_res = o >= 8
    zr = tl.where(is_res[None, :], z, float("-inf"))
    m = tl.max(zr, axis=1)
    e = tl.where(is_res[None, :], tl.exp(zr - m[:, None]), 0.0)
    a = tl.where(is_res[None, :], e / tl.sum(e, axis=1)[:, None], 0.0)
    q = tl.arange(0, 16)
    pp = tl.load(P_ptr + o[:, None] * 16 + q[None, :])
    Hres = tl.dot(a, pp, input_precision="ieee")
    hpre0 = tl.sum(tl.where(o[None, :] == 0, sig, 0.0), axis=1)
    hpre1 = tl.sum(tl.where(o[None, :] == 1, sig, 0.0), axis=1)
    hpre2 = tl.sum(tl.where(o[None, :] == 2, sig, 0.0), axis=1)
    hpre3 = tl.sum(tl.where(o[None, :] == 3, sig, 0.0), axis=1)
    hpo0 = 2.0 * tl.sum(tl.where(o[None, :] == 4, sig, 0.0), axis=1)
    hpo1 = 2.0 * tl.sum(tl.where(o[None, :] == 5, sig, 0.0), axis=1)
    hpo2 = 2.0 * tl.sum(tl.where(o[None, :] == 6, sig, 0.0), axis=1)
    hpo3 = 2.0 * tl.sum(tl.where(o[None, :] == 7, sig, 0.0), axis=1)
    qi = q // 4
    qj = q % 4
    hpost_q = tl.where(qi[None, :] == 0, hpo0[:, None], tl.where(qi[None, :] == 1, hpo1[:, None],
              tl.where(qi[None, :] == 2, hpo2[:, None], hpo3[:, None])))
    hpre_q = tl.where(qj[None, :] == 0, hpre0[:, None], tl.where(qj[None, :] == 1, hpre1[:, None],
             tl.where(qj[None, :] == 2, hpre2[:, None], hpre3[:, None])))
    if MERGE:
        # H_merged = H_res - h_post (x) h_pre: the wrapped layer returns li + f (its own residual)
        Hm = Hres - hpost_q * hpre_q
    else:
        Hm = Hres
    return (alpha_vec, sig, a, pp, q, Hm, hpre_q, hpost_q,
            hpre0, hpre1, hpre2, hpre3, hpo0, hpo1, hpo2, hpo3, is_res)


@triton.jit
def _pre_fwd_kernel(x_ptr, W_ptr, P_ptr, alpha_ptr, bias_ptr,
                    li_ptr, hm_ptr, hpost_ptr, proj_ptr, rinv_ptr, hpre_ptr,
                    T, eps: tl.constexpr, C: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
                    MERGE: tl.constexpr):
    K: tl.constexpr = 4 * C
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    o = tl.arange(0, 32)
    sumsq = tl.zeros((BT,), tl.float32)
    acc = tl.zeros((BT, 32), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        xt = tl.load(x_ptr + t[:, None] * K + kk[None, :], mask=tm[:, None], other=0.0)
        xf = xt.to(tl.float32)
        sumsq += tl.sum(xf * xf, axis=1)
        wt = tl.load(W_ptr + o[:, None] * K + kk[None, :])
        acc += tl.dot(xt, tl.trans(wt))
    r = tl.rsqrt(sumsq / K + eps)
    proj = acc * r[:, None]
    tl.store(proj_ptr + t[:, None] * 32 + o[None, :], proj, mask=tm[:, None])
    tl.store(rinv_ptr + t, r, mask=tm)
    (alpha_vec, sig, a, pp, q, Hm, hpre_q, hpost_q,
     hpre0, hpre1, hpre2, hpre3, hpo0, hpo1, hpo2, hpo3, is_res) = _coeffs(proj, o, P_ptr, alpha_ptr, bias_ptr, MERGE)
    tl.store(hm_ptr + t[:, None] * 16 + q[None, :], Hm, mask=tm[:, None])
    tl.store(hpost_ptr + t * 4 + 0, hpo0, mask=tm)
    tl.store(hpost_ptr + t * 4 + 1, hpo1, mask=tm)
    tl.store(hpost_ptr + t * 4 + 2, hpo2, mask=tm)
    tl.store(hpost_ptr + t * 4 + 3, hpo3, mask=tm)
    tl.store(hpre_ptr + t * 4 + 0, hpre0, mask=tm)
    tl.store(hpre_ptr + t * 4 + 1, hpre1, mask=tm)
    tl.store(hpre_ptr + t * 4 + 2, hpre2, mask=tm)
    tl.store(hpre_ptr + t * 4 + 3, hpre3, mask=tm)
    for c0 in range(0, C, BC):
        cc = c0 + tl.arange(0, BC)
        base = x_ptr + t[:, None] * K + cc[None, :]
        x0 = tl.load(base + 0 * C, mask=tm[:, None], other=0.0).to(tl.float32)
        x1 = tl.load(base + 1 * C, mask=tm[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(base + 2 * C, mask=tm[:, None], other=0.0).to(tl.float32)
        x3 = tl.load(base + 3 * C, mask=tm[:, None], other=0.0).to(tl.float32)
        li = hpre0[:, None] * x0 + hpre1[:, None] * x1 + hpre2[:, None] * x2 + hpre3[:, None] * x3
        tl.store(li_ptr + t[:, None] * C + cc[None, :], li.to(tl.bfloat16), mask=tm[:, None])



@triton.jit
def _hm_col(Hm, q, k: tl.constexpr):
    return tl.sum(tl.where(q[None, :] == k, Hm, 0.0), axis=1)


@triton.jit
def _post_fwd_kernel(x_ptr, lo_ptr, hm_ptr, hp_ptr, out_ptr, T,
                     C: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """out_i = sum_j Hm_ij x_j + hpost_i lo. 2-D grid over (token, channel) tiles."""
    K: tl.constexpr = 4 * C
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    cc = tl.program_id(1) * BC + tl.arange(0, BC)
    tm = t < T
    m2 = tm[:, None] & (cc < C)[None, :]
    base = x_ptr + t[:, None] * K + cc[None, :]
    x0 = tl.load(base + 0 * C, mask=m2, other=0.0).to(tl.float32)
    x1 = tl.load(base + 1 * C, mask=m2, other=0.0).to(tl.float32)
    x2 = tl.load(base + 2 * C, mask=m2, other=0.0).to(tl.float32)
    x3 = tl.load(base + 3 * C, mask=m2, other=0.0).to(tl.float32)
    lo = tl.load(lo_ptr + t[:, None] * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    ob = out_ptr + t[:, None] * K + cc[None, :]
    for i in tl.static_range(4):
        h0 = tl.load(hm_ptr + t * 16 + i * 4 + 0, mask=tm, other=0.0)
        h1 = tl.load(hm_ptr + t * 16 + i * 4 + 1, mask=tm, other=0.0)
        h2 = tl.load(hm_ptr + t * 16 + i * 4 + 2, mask=tm, other=0.0)
        h3 = tl.load(hm_ptr + t * 16 + i * 4 + 3, mask=tm, other=0.0)
        hp = tl.load(hp_ptr + t * 4 + i, mask=tm, other=0.0)
        oi = h0[:, None] * x0 + h1[:, None] * x1 + h2[:, None] * x2 + h3[:, None] * x3 + hp[:, None] * lo
        tl.store(ob + i * C, oi.to(tl.bfloat16), mask=m2)


@triton.jit
def _post_bwd_kernel(go_ptr, x_ptr, lo_ptr, hp_ptr, glo_ptr, ghm_ptr, ghp_ptr, T,
                     C: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """g_lo = sum_i hpost_i go_i ; gHm_ij = <go_i, x_j> ; g_hpost_i = <go_i, lo>.
    gx_post = Hm^T go is NOT formed here; pre_bwd computes it from go."""
    K: tl.constexpr = 4 * C
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    tm = t < T
    m2 = tm[:, None]
    q = tl.arange(0, 16)
    hp0 = tl.load(hp_ptr + t * 4 + 0, mask=tm, other=0.0)
    hp1 = tl.load(hp_ptr + t * 4 + 1, mask=tm, other=0.0)
    hp2 = tl.load(hp_ptr + t * 4 + 2, mask=tm, other=0.0)
    hp3 = tl.load(hp_ptr + t * 4 + 3, mask=tm, other=0.0)
    acc = tl.zeros((BT, 16), tl.float32)
    ghp = tl.zeros((BT, 4), tl.float32)
    q4 = tl.arange(0, 4)
    for c0 in range(0, C, BC):
        cc = c0 + tl.arange(0, BC)
        gb = go_ptr + t[:, None] * K + cc[None, :]
        xb = x_ptr + t[:, None] * K + cc[None, :]
        g0 = tl.load(gb + 0 * C, mask=m2, other=0.0).to(tl.float32)
        g1 = tl.load(gb + 1 * C, mask=m2, other=0.0).to(tl.float32)
        g2 = tl.load(gb + 2 * C, mask=m2, other=0.0).to(tl.float32)
        g3 = tl.load(gb + 3 * C, mask=m2, other=0.0).to(tl.float32)
        x0 = tl.load(xb + 0 * C, mask=m2, other=0.0).to(tl.float32)
        x1 = tl.load(xb + 1 * C, mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(xb + 2 * C, mask=m2, other=0.0).to(tl.float32)
        x3 = tl.load(xb + 3 * C, mask=m2, other=0.0).to(tl.float32)
        lo = tl.load(lo_ptr + t[:, None] * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
        glo = hp0[:, None] * g0 + hp1[:, None] * g1 + hp2[:, None] * g2 + hp3[:, None] * g3
        tl.store(glo_ptr + t[:, None] * C + cc[None, :], glo.to(tl.bfloat16), mask=m2)
        acc += tl.where(q[None, :] == 0, tl.sum(g0 * x0, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 1, tl.sum(g0 * x1, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 2, tl.sum(g0 * x2, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 3, tl.sum(g0 * x3, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 4, tl.sum(g1 * x0, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 5, tl.sum(g1 * x1, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 6, tl.sum(g1 * x2, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 7, tl.sum(g1 * x3, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 8, tl.sum(g2 * x0, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 9, tl.sum(g2 * x1, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 10, tl.sum(g2 * x2, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 11, tl.sum(g2 * x3, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 12, tl.sum(g3 * x0, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 13, tl.sum(g3 * x1, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 14, tl.sum(g3 * x2, axis=1)[:, None], 0.0)
        acc += tl.where(q[None, :] == 15, tl.sum(g3 * x3, axis=1)[:, None], 0.0)
        ghp += tl.where(q4[None, :] == 0, tl.sum(g0 * lo, axis=1)[:, None], 0.0)
        ghp += tl.where(q4[None, :] == 1, tl.sum(g1 * lo, axis=1)[:, None], 0.0)
        ghp += tl.where(q4[None, :] == 2, tl.sum(g2 * lo, axis=1)[:, None], 0.0)
        ghp += tl.where(q4[None, :] == 3, tl.sum(g3 * lo, axis=1)[:, None], 0.0)
    tl.store(ghm_ptr + t[:, None] * 16 + q[None, :], acc, mask=m2)
    tl.store(ghp_ptr + t[:, None] * 4 + q4[None, :], ghp, mask=m2)


@triton.jit
def _pre_bwd_coef_kernel(gli_ptr, x_ptr, ghm_ptr, ghp_ptr, proj_ptr, rinv_ptr,
                         P_ptr, alpha_ptr, bias_ptr, gproj_ptr, s_ptr, galpha_ptr, gbias_ptr,
                         T, C: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, MERGE: tl.constexpr):
    """Phase 1 (<g_li, x_j>) and the whole coefficient backward. Writes
    gpr = g_proj * r (T,32) and s = r^2 (g_proj . proj)/K (T,), which is all
    the gx pass needs besides Hm and h_pre from the forward."""
    K: tl.constexpr = 4 * C
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    m2 = tm[:, None]
    o = tl.arange(0, 32)
    proj = tl.load(proj_ptr + t[:, None] * 32 + o[None, :], mask=m2, other=0.0)
    r = tl.load(rinv_ptr + t, mask=tm, other=0.0)
    (alpha_vec, sig, a, pp, q, Hm, hpre_q, hpost_q,
     hpre0, hpre1, hpre2, hpre3, hpo0, hpo1, hpo2, hpo3, is_res) = _coeffs(proj, o, P_ptr, alpha_ptr, bias_ptr, MERGE)
    # phase 1: g_hpre_j (pre-map part) = <g_li, x_j>
    gm0 = tl.zeros((BT,), tl.float32)
    gm1 = tl.zeros((BT,), tl.float32)
    gm2 = tl.zeros((BT,), tl.float32)
    gm3 = tl.zeros((BT,), tl.float32)
    for c0 in range(0, C, BC):
        cc = c0 + tl.arange(0, BC)
        gl = tl.load(gli_ptr + t[:, None] * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
        base = x_ptr + t[:, None] * K + cc[None, :]
        gm0 += tl.sum(gl * tl.load(base + 0 * C, mask=m2, other=0.0).to(tl.float32), axis=1)
        gm1 += tl.sum(gl * tl.load(base + 1 * C, mask=m2, other=0.0).to(tl.float32), axis=1)
        gm2 += tl.sum(gl * tl.load(base + 2 * C, mask=m2, other=0.0).to(tl.float32), axis=1)
        gm3 += tl.sum(gl * tl.load(base + 3 * C, mask=m2, other=0.0).to(tl.float32), axis=1)
    gHm = tl.load(ghm_ptr + t[:, None] * 16 + q[None, :], mask=m2, other=0.0)
    qi = q // 4
    qj = q % 4
    if MERGE:
        # the -h_post (x) h_pre part of Hm sends its gradient to h_pre and h_post
        t1 = gHm * hpost_q
        t2 = gHm * hpre_q
    else:
        t1 = tl.zeros_like(gHm)
        t2 = tl.zeros_like(gHm)
    ghpre0 = gm0 - tl.sum(tl.where(qj[None, :] == 0, t1, 0.0), axis=1)
    ghpre1 = gm1 - tl.sum(tl.where(qj[None, :] == 1, t1, 0.0), axis=1)
    ghpre2 = gm2 - tl.sum(tl.where(qj[None, :] == 2, t1, 0.0), axis=1)
    ghpre3 = gm3 - tl.sum(tl.where(qj[None, :] == 3, t1, 0.0), axis=1)
    ghpo0 = tl.load(ghp_ptr + t * 4 + 0, mask=tm, other=0.0) - tl.sum(tl.where(qi[None, :] == 0, t2, 0.0), axis=1)
    ghpo1 = tl.load(ghp_ptr + t * 4 + 1, mask=tm, other=0.0) - tl.sum(tl.where(qi[None, :] == 1, t2, 0.0), axis=1)
    ghpo2 = tl.load(ghp_ptr + t * 4 + 2, mask=tm, other=0.0) - tl.sum(tl.where(qi[None, :] == 2, t2, 0.0), axis=1)
    ghpo3 = tl.load(ghp_ptr + t * 4 + 3, mask=tm, other=0.0) - tl.sum(tl.where(qi[None, :] == 3, t2, 0.0), axis=1)
    g_a = tl.dot(gHm, tl.trans(pp), input_precision="ieee")
    # softmax backward over the 24 permutation lanes only: P's zero rows make g_a
    # 0 * gHm on the other 8, which is NaN once gHm is, and must not reach h_pre / h_post
    ag = tl.where(is_res[None, :], a * g_a, 0.0)
    g_zres = tl.where(is_res[None, :], ag - a * tl.sum(ag, axis=1)[:, None], 0.0)
    g_h = tl.where(o[None, :] == 0, ghpre0[:, None], tl.where(o[None, :] == 1, ghpre1[:, None],
          tl.where(o[None, :] == 2, ghpre2[:, None], tl.where(o[None, :] == 3, ghpre3[:, None],
          tl.where(o[None, :] == 4, 2.0 * ghpo0[:, None], tl.where(o[None, :] == 5, 2.0 * ghpo1[:, None],
          tl.where(o[None, :] == 6, 2.0 * ghpo2[:, None], tl.where(o[None, :] == 7, 2.0 * ghpo3[:, None], 0.0))))))))
    g_z = g_zres + g_h * sig * (1.0 - sig)
    g_z = tl.where(m2, g_z, 0.0)
    gproj = g_z * alpha_vec[None, :]
    tl.store(gproj_ptr + t[:, None] * 32 + o[None, :], gproj * r[:, None], mask=m2)
    gzp = g_z * proj
    tl.atomic_add(galpha_ptr + 0, tl.sum(tl.sum(tl.where(o[None, :] < 4, gzp, 0.0), axis=1), axis=0))
    tl.atomic_add(galpha_ptr + 1, tl.sum(tl.sum(tl.where((o[None, :] >= 4) & (o[None, :] < 8), gzp, 0.0), axis=1), axis=0))
    tl.atomic_add(galpha_ptr + 2, tl.sum(tl.sum(tl.where(o[None, :] >= 8, gzp, 0.0), axis=1), axis=0))
    tl.atomic_add(gbias_ptr + o, tl.sum(g_z, axis=0))
    d = tl.sum(gproj * proj, axis=1) / K
    tl.store(s_ptr + t, r * r * d, mask=tm)


@triton.jit
def _gx_j(j: tl.constexpr, Hm_t, q, hpj, g0, g1, g2, g3, gl, gpr, gpr_b, sv, wj, xjb, off, m2, gx_ptr,
          BF16DOT: tl.constexpr):
    xj = xjb.to(tl.float32)
    if BF16DOT:
        gxn = tl.dot(gpr_b, wj)
    else:
        gxn = tl.dot(gpr, wj.to(tl.float32), input_precision="tf32")
    gx = (_hm_col(Hm_t, q, 0 * 4 + j)[:, None] * g0 + _hm_col(Hm_t, q, 1 * 4 + j)[:, None] * g1
          + _hm_col(Hm_t, q, 2 * 4 + j)[:, None] * g2 + _hm_col(Hm_t, q, 3 * 4 + j)[:, None] * g3
          + hpj[:, None] * gl + gxn - xj * sv[:, None])
    tl.store(gx_ptr + off, gx.to(tl.bfloat16), mask=m2)
    return tl.dot(tl.trans(gpr_b), xjb)


@triton.jit
def _pre_bwd_gx_kernel(gli_ptr, x_ptr, go_ptr, hm_ptr, hpre_ptr, gpr_ptr, s_ptr, W_ptr, gx_ptr, gwp_ptr,
                       T, C: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, S: tl.constexpr,
                       BF16DOT: tl.constexpr = True):
    """gx_j = sum_i Hm_ij go_i + h_pre_j g_li + gpr @ W_j - s x_j, and the
    W-grad partial gpr^T @ x for this column block over this token split.
    Grid (C // BC, S); the W slice stays in registers across the token loop."""
    K: tl.constexpr = 4 * C
    pc = tl.program_id(0)
    ps = tl.program_id(1)
    cc = pc * BC + tl.arange(0, BC)
    o = tl.arange(0, 32)
    q = tl.arange(0, 16)
    q4 = tl.arange(0, 4)
    w0 = tl.load(W_ptr + o[:, None] * K + (0 * C + cc)[None, :])
    w1 = tl.load(W_ptr + o[:, None] * K + (1 * C + cc)[None, :])
    w2 = tl.load(W_ptr + o[:, None] * K + (2 * C + cc)[None, :])
    w3 = tl.load(W_ptr + o[:, None] * K + (3 * C + cc)[None, :])
    a0 = tl.zeros((32, BC), tl.float32)
    a1 = tl.zeros((32, BC), tl.float32)
    a2 = tl.zeros((32, BC), tl.float32)
    a3 = tl.zeros((32, BC), tl.float32)
    for t0 in range(ps * BT, T, S * BT):
        t = t0 + tl.arange(0, BT)
        tm = t < T
        m2 = tm[:, None]
        Hm_t = tl.load(hm_ptr + t[:, None] * 16 + q[None, :], mask=m2, other=0.0)
        hpre = tl.load(hpre_ptr + t[:, None] * 4 + q4[None, :], mask=m2, other=0.0)
        gpr = tl.load(gpr_ptr + t[:, None] * 32 + o[None, :], mask=m2, other=0.0)
        gpr_b = gpr.to(tl.bfloat16)
        sv = tl.load(s_ptr + t, mask=tm, other=0.0)
        gb = go_ptr + t[:, None] * K + cc[None, :]
        g0 = tl.load(gb + 0 * C, mask=m2, other=0.0).to(tl.float32)
        g1 = tl.load(gb + 1 * C, mask=m2, other=0.0).to(tl.float32)
        g2 = tl.load(gb + 2 * C, mask=m2, other=0.0).to(tl.float32)
        g3 = tl.load(gb + 3 * C, mask=m2, other=0.0).to(tl.float32)
        gl = tl.load(gli_ptr + t[:, None] * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
        off = t[:, None] * K + cc[None, :]
        x0 = tl.load(x_ptr + off + 0 * C, mask=m2, other=0.0)
        x1 = tl.load(x_ptr + off + 1 * C, mask=m2, other=0.0)
        x2 = tl.load(x_ptr + off + 2 * C, mask=m2, other=0.0)
        x3 = tl.load(x_ptr + off + 3 * C, mask=m2, other=0.0)
        a0 += _gx_j(0, Hm_t, q, _hm_col(hpre, q4, 0), g0, g1, g2, g3, gl, gpr, gpr_b, sv, w0, x0, off + 0 * C, m2, gx_ptr, BF16DOT)
        a1 += _gx_j(1, Hm_t, q, _hm_col(hpre, q4, 1), g0, g1, g2, g3, gl, gpr, gpr_b, sv, w1, x1, off + 1 * C, m2, gx_ptr, BF16DOT)
        a2 += _gx_j(2, Hm_t, q, _hm_col(hpre, q4, 2), g0, g1, g2, g3, gl, gpr, gpr_b, sv, w2, x2, off + 2 * C, m2, gx_ptr, BF16DOT)
        a3 += _gx_j(3, Hm_t, q, _hm_col(hpre, q4, 3), g0, g1, g2, g3, gl, gpr, gpr_b, sv, w3, x3, off + 3 * C, m2, gx_ptr, BF16DOT)
    pb = gwp_ptr + ps * 32 * K + o[:, None] * K + cc[None, :]
    tl.store(pb + 0 * C, a0)
    tl.store(pb + 1 * C, a1)
    tl.store(pb + 2 * C, a2)
    tl.store(pb + 3 * C, a3)


CFG = dict(pre_fwd=(64, 128, 64, 4), post_fwd=(8, 512, 4), post_bwd=(16, 128, 8),
           pre_coef=(32, 64, 4), pre_gx=(16, 64, 16, 4, 2))


class MHCPreFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, alpha3, bias, P, eps, merge=True):
        T, n, C = x.shape
        xf = x.contiguous()
        Wb = W.to(torch.bfloat16).contiguous()
        li = torch.empty(T, C, device=x.device, dtype=torch.bfloat16)
        Hm = torch.empty(T, 16, device=x.device, dtype=torch.float32)
        hpost = torch.empty(T, 4, device=x.device, dtype=torch.float32)
        proj = torch.empty(T, 32, device=x.device, dtype=torch.float32)
        rinv = torch.empty(T, device=x.device, dtype=torch.float32)
        hpre = torch.empty(T, 4, device=x.device, dtype=torch.float32)
        BT, BK, BC, nw = CFG["pre_fwd"]
        _pre_fwd_kernel[(triton.cdiv(T, BT),)](xf, Wb, P, alpha3, bias, li, Hm, hpost, proj, rinv, hpre,
                                               T, eps=eps, C=C, BT=BT, BK=BK, BC=BC, MERGE=bool(merge), num_warps=nw)
        ctx.save_for_backward(xf, Wb, alpha3, bias, P, proj, rinv, Hm, hpre)
        ctx.merge = bool(merge)
        ctx.W_dtype = W.dtype
        # dense and never read: a stride-0 placeholder breaks AOT autograd's
        # tangent coercion when it crosses a compiled-graph boundary
        tok = torch.empty_like(xf)
        return li, Hm, hpost, tok

    @staticmethod
    def backward(ctx, g_li, g_Hm, g_hpost, go):
        # `go` arrives as the "gradient" of `tok`, an uninitialised placeholder that
        # MHCPostFused consumes and hands the block's output gradient back
        # through, so gx_post = Hm^T go is formed in-kernel here and autograd
        # never adds two (T, n, C) gradients for x.
        xf, Wb, alpha3, bias, P, proj, rinv, Hm, hpre = ctx.saved_tensors
        T, n, C = xf.shape
        dev = xf.device
        g_li = torch.zeros(T, C, device=dev, dtype=torch.bfloat16) if g_li is None else g_li.contiguous()
        g_Hm = torch.zeros(T, 16, device=dev, dtype=torch.float32) if g_Hm is None else g_Hm.float().contiguous()
        g_hpost = torch.zeros(T, 4, device=dev, dtype=torch.float32) if g_hpost is None else g_hpost.float().contiguous()
        go = torch.zeros_like(xf) if go is None else go.contiguous()
        gx = torch.empty_like(xf)
        gpr = torch.empty(T, 32, device=dev, dtype=torch.float32)
        sv = torch.empty(T, device=dev, dtype=torch.float32)
        galpha = torch.zeros(3, device=dev, dtype=torch.float32)
        gbias = torch.zeros(32, device=dev, dtype=torch.float32)
        BT, BC, nw = CFG["pre_coef"]
        _pre_bwd_coef_kernel[(triton.cdiv(T, BT),)](g_li, xf, g_Hm, g_hpost, proj, rinv, P, alpha3, bias,
                                                    gpr, sv, galpha, gbias, T, C=C, BT=BT, BC=BC, MERGE=ctx.merge,
                                                    num_warps=nw)
        BT, BC, S, nw, ns = CFG["pre_gx"]
        gwp = torch.empty(S, 32, 4 * C, device=dev, dtype=torch.float32)
        _pre_bwd_gx_kernel[(C // BC, S)](g_li, xf, go, Hm, hpre, gpr, sv, Wb, gx, gwp,
                                         T, C=C, BT=BT, BC=BC, S=S, BF16DOT=True, num_warps=nw, num_stages=ns)
        gW = gwp.sum(0).to(ctx.W_dtype)
        return gx, gW, galpha, gbias, None, None, None


class MHCPostFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tok, x, lo, Hm, hpost):
        T, n, C = x.shape
        lo = lo.contiguous()
        out = torch.empty_like(x)
        BT, BC, nw = CFG["post_fwd"]
        _post_fwd_kernel[(triton.cdiv(T, BT), triton.cdiv(C, BC))](x, lo, Hm, hpost, out, T, C=C, BT=BT, BC=BC,
                                                        num_warps=nw)
        ctx.save_for_backward(x, lo, hpost)
        return out

    @staticmethod
    def backward(ctx, go):
        x, lo, hpost = ctx.saved_tensors
        T, n, C = x.shape
        go = go.contiguous()
        glo = torch.empty_like(lo)
        gHm = torch.empty(T, 16, device=x.device, dtype=torch.float32)
        ghp = torch.empty(T, 4, device=x.device, dtype=torch.float32)
        BT, BC, nw = CFG["post_bwd"]
        _post_bwd_kernel[(triton.cdiv(T, BT),)](go, x, lo, hpost, glo, gHm, ghp, T, C=C, BT=BT, BC=BC,
                                                num_warps=nw)
        return go, None, glo, gHm, ghp


def supports(x: torch.Tensor) -> bool:
    """Shapes these kernels handle: (T, 4, C) with C a multiple of the 128-wide
    channel tiles, and 32-bit element offsets (T * 4 * C < 2^31)."""
    return x.dim() == 3 and x.shape[1] == 4 and x.shape[2] % 128 == 0 and x.numel() < 2 ** 31


def mhc_fused_block(x, layer, W, alpha3, bias, P, eps=1e-6, merge=True, **kw):
    """merge=True: layer returns li + f and Hm = H_res - h_post h_pre^T cancels the li;
    merge=False: layer returns f and Hm = H_res (the mHC paper / DeepSeek-V4 form)."""
    assert supports(x), f"mhc_fused_v2: unsupported x of shape {tuple(x.shape)}"
    # the kernels index x densely: pre and post must see the same dense copy
    x = x.contiguous()
    li, Hm, hpost, tok = MHCPreFused.apply(x, W, alpha3, bias, P, eps, merge)
    lo = layer(li, **kw)
    return MHCPostFused.apply(tok, x, lo, Hm, hpost)
