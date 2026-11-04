"""Shared pieces of the CuTeDSL mHC-lite kernels (n=4 streams, C=1024, sm90):
problem constants, the cluster split constants of the backward, host helpers
(TVM-FFI compile cache, fake tensors, max active clusters) and the coefficient
backward device helpers.

NOTE: no ``from __future__ import annotations`` (breaks CuTeDSL Constexpr detection).
"""
import itertools

import torch
import cutlass
import cutlass.cute as cute

NS = 4            # streams


C = 1024          # hidden


K = NS * C        # 4096, projection input width


VEC = 8           # bf16 per 16-byte vector


PAD = 8           # row padding (elements): rows shift 16 B per row -> conflict-free ldmatrix


NO = 32           # projection outputs: 4 pre + 4 post + 24 res


_PERMS = list(itertools.permutations(range(NS)))


_cache = {}


_NCL = {}


def _max_clusters(nq):
    if nq not in _NCL:
        _NCL[nq] = cutlass.utils.HardwareInfo().get_max_active_clusters(nq)
    return _NCL[nq]


PSTR5 = 36        # exchange row: 32 proj + sumsq + 3 pad = 9 x 16 B


PV5 = PSTR5 // 4


def perm_src_table(device="cuda"):
    """(16, 6) int32: for H_res entry k = 4i + j, the 6 permutation indices p
    with perm_p(i) == j (their logits sit in warp lane 8 + p)."""
    rows = [[p for p, perm in enumerate(_PERMS) if perm[k // NS] == k % NS] for k in range(NS * NS)]
    return torch.tensor(rows, dtype=torch.int32, device=device)


RS6 = NO + 8      # sRed row stride: 40 floats -> conflict-free 64-bit C-fragment stores


B_FULL, B_EMPTY, B_XREADY, B_XFREE = 0, 2, 4, 7


def _fake(dtype, shape, align=16):
    """Compile-time stand-in for a row-major torch tensor; None in shape = symbolic."""
    from cutlass.cute.runtime import make_fake_compact_tensor
    shp = tuple(cute.sym_int32() if d is None else d for d in shape)
    return make_fake_compact_tensor(dtype, shp, stride_order=tuple(reversed(range(len(shape)))),
                                    assumed_align=align)


def _ffi_compile(name, launcher, fakes, scalars, constexprs, ptxas=""):
    """cute.compile with TVM-FFI: the returned callable takes torch tensors and
    python scalars directly and launches on torch's current stream."""
    from cutlass.cute.runtime import make_fake_stream
    key = (name, launcher.__module__, id(launcher)) + tuple(constexprs)
    if key not in _cache:
        _cache[key] = cute.compile(launcher, *fakes, *scalars, make_fake_stream(use_tvm_ffi_env_stream=True),
                                   *constexprs, options="--enable-tvm-ffi" + (f" --ptxas-options={ptxas}" if ptxas else ""))
    return _cache[key]


_PSRC = {}


def _psrc(dev):
    if dev not in _PSRC:
        _PSRC[dev] = perm_src_table(dev)
    return _PSRC[dev]


NQ = 4


QC = C // NQ              # 256 columns per stream per CTA


KQ = NS * QC              # 1024


QV = QC // VEC            # 32


SROW = KQ + PAD           # x / go stage row (elements)


BT = 16


NPW = 4                   # producer warps


NGW = 8                   # gx warps


NPT = NPW * 32


NGT = NGW * 32


GPS = NO + 1              # gpr row pushed per token: 32 gpr + s


GRS = 36                  # its padded row in the receive buffer


SGS = NO + 8              # bf16 gpr tile row (conflict-free ldmatrix)


def perm_gsrc_table(device="cuda"):
    """(24, 4) int32: for permutation p, the gHm entries 4i + perm_p(i)."""
    rows = [[4 * i + perm[i] for i in range(NS)] for perm in _PERMS]
    return torch.tensor(rows, dtype=torch.int32, device=device)


@cute.jit
def _coef_bwd_a(sCo, pr, ghm_l, ghp_l, gsrc, lane: cutlass.Int32, merge: cutlass.Constexpr):
    """Coefficient backward, everything that does not need gm (lane = output o).
    Returns (al, sig, acc_pre, g_hpost, gz_res) with gz_res final for lanes >= 4."""
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
    a = e * (1.0 / ssum)                  # not e / ssum: e == 0 lanes fail FCHK -> slow IEEE-div subroutine
    li_ = lane % NS
    acc_pre = cutlass.Float32(0.0)
    acc_post = cutlass.Float32(0.0)
    if cutlass.const_expr(merge):
        # Hm = H_res - h_post h_pre^T: its gradient reaches h_pre and h_post too
        for u in cutlass.range_constexpr(NS):
            g_uj = cute.arch.shuffle_sync(ghm_l, NS * u + li_)
            g_iu = cute.arch.shuffle_sync(ghm_l, NS * li_ + u)
            hpo_u = 2.0 * cute.arch.shuffle_sync(sig, NS + u)
            hpr_u = cute.arch.shuffle_sync(sig, u)
            acc_pre = acc_pre + g_uj * hpo_u
            acc_post = acc_post + g_iu * hpr_u
    g_hpost = ghp_l - acc_post
    ga = cutlass.Float32(0.0)
    for u in cutlass.range_constexpr(NS):
        ga = ga + cute.arch.shuffle_sync(ghm_l, gsrc[u])
    sag = cute.arch.warp_reduction(a * ga, lambda a_, b_: a_ + b_)
    gz = a * (ga - sag)
    if lane < 2 * NS:
        gz = (2.0 * g_hpost) * sig * (1.0 - sig)
    return al, sig, acc_pre, gz


@cute.jit
def _coef_bwd_b(gm, al, sig, acc_pre, gz, pr, r, lane: cutlass.Int32):
    """The gm-dependent tail: g_hpre for lanes 0-3, then g_proj, d, s."""
    if lane < NS:
        gz = (gm - acc_pre) * sig * (1.0 - sig)
    gproj = gz * al
    d = cute.arch.warp_reduction(gproj * pr, lambda a_, b_: a_ + b_) / K
    return gproj * r, r * r * d, gz
