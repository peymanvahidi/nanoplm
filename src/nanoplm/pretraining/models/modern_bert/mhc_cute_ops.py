"""mHC-lite (n=4, C=1024) with the CuTeDSL cluster kernels for the pre side
(mhc_cute_fwd / mhc_cute_bwd) and the v2 Triton post kernels.

Selected in MHCLiteBlock (see MHCLiteBlock._backend). The CuTe launches are
wrapped as torch.library custom ops so torch.compile treats them as opaque.
dW is a separate bf16 GEMM on the returned g_proj * r (as v2 does).
"""
import torch

from . import mhc_cute_fwd as _FK
from . import mhc_cute_bwd as _BK
from .mhc_fused_v2 import MHCPostFused, mhc_fused_block as _v2_block, supports as _v2_supports

_EPS = 1e-6
_DEBUG = None
_NS, _C = 4, 1024


@torch.library.custom_op("nanoplm_mhc_cute::pre_fwd", mutates_args=())
def pre_fwd_op(x: torch.Tensor, W: torch.Tensor, alpha3: torch.Tensor, bias: torch.Tensor, merge: bool
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _FK.pre_fwd7(x, W, alpha3, bias, eps=_EPS, merge=merge)


@pre_fwd_op.register_fake
def _(x, W, alpha3, bias, merge):
    T = x.shape[0]
    f = dict(device=x.device, dtype=torch.float32)
    return (x.new_empty(T, _C), torch.empty(T, 16, **f), torch.empty(T, _NS, **f), torch.empty(T, _NS, **f),
            torch.empty(T, 32, **f), torch.empty(T, **f))


@torch.library.custom_op("nanoplm_mhc_cute::pre_bwd", mutates_args=())
def pre_bwd_op(x: torch.Tensor, go: torch.Tensor, g_li: torch.Tensor, W: torch.Tensor, alpha3: torch.Tensor,
               bias: torch.Tensor, proj: torch.Tensor, rinv: torch.Tensor, Hm: torch.Tensor, hpre: torch.Tensor,
               g_Hm: torch.Tensor, g_hpost: torch.Tensor, merge: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gx, gpr, ga, gb = _BK.pre_bwd(x, go, g_li, W, alpha3, bias, proj, rinv, Hm, hpre, g_Hm, g_hpost, merge=merge)
    return gx, gpr, torch.cat([ga, gb])


@pre_bwd_op.register_fake
def _(x, go, g_li, W, alpha3, bias, proj, rinv, Hm, hpre, g_Hm, g_hpost, merge):
    T = x.shape[0]
    return (torch.empty_like(x), torch.empty(T, 32, device=x.device, dtype=torch.float32),
            torch.empty(35, device=x.device, dtype=torch.float32))


def _tma_ready(t):
    """Dense bf16 with a 16-byte aligned base, as the backward TMA loads need."""
    t = t.contiguous()
    return t if t.storage_offset() % 8 == 0 else t.clone()


class MHCPreCute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, alpha3, bias, merge):
        li, Hm, hpost, hpre, proj, rinv = pre_fwd_op(x, W, alpha3, bias, merge)
        ctx.save_for_backward(x, W, alpha3, bias, proj, rinv, Hm, hpre)
        ctx.merge = merge
        # dense, never read: carries the block output gradient back (see mhc_fused_v2)
        tok = torch.empty_like(x)
        return li, Hm, hpost, tok

    @staticmethod
    def backward(ctx, g_li, g_Hm, g_hpost, go):
        x, W, alpha3, bias, proj, rinv, Hm, hpre = ctx.saved_tensors
        T = x.shape[0]
        g_li = torch.zeros(T, _C, device=x.device, dtype=torch.bfloat16) if g_li is None else _tma_ready(g_li)
        g_Hm = torch.zeros(T, 16, device=x.device, dtype=torch.float32) if g_Hm is None else g_Hm.float().contiguous()
        g_hpost = (torch.zeros(T, _NS, device=x.device, dtype=torch.float32) if g_hpost is None
                   else g_hpost.float().contiguous())
        go = torch.zeros_like(x) if go is None else _tma_ready(go)
        gx, gpr, gab = pre_bwd_op(x, go, g_li, W, alpha3, bias, proj, rinv, Hm, hpre, g_Hm, g_hpost, ctx.merge)
        if _DEBUG is not None:
            _DEBUG.append(dict(x=x, go=go, g_li=g_li, W=W, alpha3=alpha3, bias=bias, proj=proj, rinv=rinv, Hm=Hm,
                               hpre=hpre, g_Hm=g_Hm, g_hpost=g_hpost, gab=gab.clone(), gpr=gpr.clone()))
        gW = torch.mm(gpr.to(torch.bfloat16).t(), x.reshape(T, _NS * _C)).to(W.dtype)
        return gx, gW, gab[:3], gab[3:], None


def mhc_cute_block(x, layer, W, alpha3, bias, P, merge=True, **kw):
    """merge: see mhc_fused_v2.mhc_fused_block."""
    assert x.shape[1:] == (_NS, _C) and _v2_supports(x), f"mhc_cute: unsupported x of shape {tuple(x.shape)}"
    # the kernels index x densely: pre and post must see the same dense copy
    x = x.contiguous()
    # TMA needs whole 16-token tiles and a 16-byte aligned base (allocations are
    # 512-byte aligned, so only a view's offset can break it); v2 takes the rest
    if x.shape[0] == 0 or x.shape[0] % 16 != 0 or x.storage_offset() % 8 != 0:
        return _v2_block(x, layer, W, alpha3, bias, P, merge=merge, **kw)
    li, Hm, hpost, tok = MHCPreCute.apply(x, W, alpha3, bias, merge)
    lo = layer(li, **kw)
    return MHCPostFused.apply(tok, x, lo, Hm, hpost)
