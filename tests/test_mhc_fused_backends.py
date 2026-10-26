"""mHC-lite fused backends (v2 Triton, CuTeDSL) and the DeepSeek-V4-style defaults.

* every MHCLiteBlock backend against an fp32 reference of the same math, with the
  merged post-map (layer wrapping: H_res - h_post h_pre^T) and without it
  (sublayer wrapping: H_res, the branch returns f alone);
* the whole model in sublayer mode: init values, the stream copy at the embedding,
  f-only branches, and loss / grad agreement between the backends.
"""
import os

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

C, K = 1024, 4096


def _is_sm90():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _cute_runs():
    from nanoplm.pretraining.models.modern_bert.modeling import _cute_available

    return _is_sm90() and _cute_available()


def _backends():
    out = [("pytorch", None, False), ("v2", "v2", True)]
    if _cute_runs():
        out.append(("cute", "cute", True))
    return out


@pytest.fixture
def backend_env():
    old = os.environ.get("NANOPLM_MHC_BACKEND")
    yield
    if old is None:
        os.environ.pop("NANOPLM_MHC_BACKEND", None)
    else:
        os.environ["NANOPLM_MHC_BACKEND"] = old


def _set_backend(name):
    if name is None:
        os.environ.pop("NANOPLM_MHC_BACKEND", None)
    else:
        os.environ["NANOPLM_MHC_BACKEND"] = name


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("merge", [True, False])
def test_block_backends_vs_fp32_reference(merge, backend_env):
    from nanoplm.pretraining.models.modern_bert import modeling as mod
    from nanoplm.pretraining.models.modern_bert.mhc_fused_v2 import perm_matrix_padded

    T = 4096
    torch.manual_seed(0)
    Wl = torch.randn(C, C, device="cuda") / C ** 0.5
    P = perm_matrix_padded(device="cuda")

    class Lin(nn.Module):
        def forward(self, li, **kw):
            return (li.float() @ Wl).to(torch.bfloat16)

    def ref(x, W, a3, b, eps=1e-6):
        xf = x.float().reshape(T, K)
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        proj = (xf * r) @ W.t()
        av = torch.cat([a3[0].expand(4), a3[1].expand(4), a3[2].expand(24)])
        z = proj * av + b
        hpre, hpost = torch.sigmoid(z[:, :4]), 2 * torch.sigmoid(z[:, 4:8])
        H = (torch.softmax(z[:, 8:], -1) @ P[8:]).view(T, 4, 4)
        if merge:
            H = H - hpost[:, :, None] * hpre[:, None, :]
        lo = (hpre[:, :, None] * x.float()).sum(1) @ Wl
        return H @ x.float() + hpost[:, :, None] * lo[:, None, :]

    x0 = torch.randn(T, 4, C, device="cuda").bfloat16()
    g = torch.randn_like(x0)
    W0 = torch.randn(32, K, device="cuda") * 0.02
    b0 = torch.randn(32, device="cuda") * 0.1
    xr, Wr = x0.float().requires_grad_(True), W0.clone().requires_grad_(True)
    ar = torch.tensor([0.5, 0.6, 0.7], device="cuda", requires_grad=True)
    br = b0.clone().requires_grad_(True)
    ref(xr, Wr, ar, br).backward(g.float())
    yr = ref(x0.float(), W0, ar.detach(), b0)

    for name, env, fused in _backends():
        _set_backend(env)
        blk = mod.MHCLiteBlock(4, C, Lin(), triton_fused=fused, merge_residual=merge).cuda()
        with torch.no_grad():
            blk.W_all.weight.copy_(W0)
            blk.W_all.bias.copy_(b0)
            blk.alpha_pre.fill_(0.5)
            blk.alpha_post.fill_(0.6)
            blk.alpha_res.fill_(0.7)
        if fused:
            assert blk._backend(x0) == name
        x = x0.clone().requires_grad_(True)
        y = blk(x)
        y.backward(g)
        ga = torch.cat([blk.alpha_pre.grad, blk.alpha_post.grad, blk.alpha_res.grad])
        errs = dict(out=_rel(y, yr), gx=_rel(x.grad, xr.grad), gW=_rel(blk.W_all.weight.grad, Wr.grad),
                    galpha=_rel(ga, ar.grad), gbias=_rel(blk.W_all.bias.grad, br.grad))
        # the pytorch path computes the coefficients in bf16
        tol = 1e-2 if name in ("v2", "cute") else 4e-2
        assert all(v < tol for v in errs.values()), (name, merge, errs)


def _ref_block(x, W, a3, b, P, layer, merge, eps=1e-6):
    """fp64 oracle of an MHCLiteBlock for any (T, 4, C); W, li and the layer output
    are rounded to bf16 as the kernels' projection GEMM and the layer see them."""
    bf = lambda t: t.to(torch.bfloat16).to(t.dtype)
    T, _, C = x.shape
    xf = x.reshape(T, 4 * C)
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    proj = (xf * r) @ bf(W).t()
    av = torch.cat([a3[0].expand(4), a3[1].expand(4), a3[2].expand(24)])
    z = proj * av + b
    hpre, hpost = torch.sigmoid(z[:, :4]), 2 * torch.sigmoid(z[:, 4:8])
    H = (torch.softmax(z[:, 8:], -1) @ P[8:]).view(T, 4, 4)
    if merge:
        H = H - hpost[:, :, None] * hpre[:, None, :]
    lo = bf(layer(bf((hpre[:, :, None] * x).sum(1))))
    return H @ x + hpost[:, :, None] * lo[:, None, :]


def _row_err(a, b):
    """Worst per-token max error over the global scale: one wrong tile can hide in a norm."""
    return ((a.double() - b.double()).abs().flatten(1).amax(1).max() / b.double().abs().max()).item()


def _view(x, layout):
    """x as the block sees it, still differentiable back to the leaf x."""
    if layout == "misaligned":  # dense, but 2 bytes past an aligned base
        return torch.cat([x.new_zeros(1), x.flatten()])[1:].view(x.shape)
    if layout == "strided":  # a channel slice of a wider tensor
        return torch.cat([x, torch.zeros_like(x)], -1)[..., : x.shape[-1]]
    return x


def _check_block(T, C, name, merge=True, layout="dense", ignore_li=False, device="cuda"):
    from nanoplm.pretraining.models.modern_bert import modeling as mod
    from nanoplm.pretraining.models.modern_bert.mhc_fused_v2 import perm_matrix_padded

    _set_backend(name)
    torch.manual_seed(0)
    K = 4 * C
    P = perm_matrix_padded(device=device)
    Wl = torch.randn(C, C, device=device, dtype=torch.float64) / C ** 0.5
    const = torch.randn(T, C, device=device, dtype=torch.float64)

    def lo_of(li):  # ignore_li: li gets no gradient, the backward sees a zero g_li
        return const if ignore_li else li.double() @ Wl

    class Lin(nn.Module):
        def forward(self, li, **kw):
            return lo_of(li).to(torch.bfloat16)

    x0 = torch.randn(T, 4, C, device=device).bfloat16()
    g = torch.randn_like(x0)
    W0 = torch.randn(32, K, device=device) * 0.02
    b0 = torch.randn(32, device=device) * 0.1
    xr, Wr = x0.double().requires_grad_(True), W0.double().requires_grad_(True)
    ar = torch.tensor([0.5, 0.6, 0.7], device=device, dtype=torch.float64, requires_grad=True)
    br = b0.double().requires_grad_(True)
    yr = _ref_block(xr, Wr, ar, br, P.double(), lo_of, merge)
    yr.backward(g.double())

    blk = mod.MHCLiteBlock(4, C, Lin(), triton_fused=True, merge_residual=merge).to(device)
    with torch.no_grad():
        blk.W_all.weight.copy_(W0)
        blk.W_all.bias.copy_(b0)
        blk.alpha_pre.fill_(0.5)
        blk.alpha_post.fill_(0.6)
        blk.alpha_res.fill_(0.7)
    x = x0.clone().requires_grad_(True)
    xv = _view(x, layout)
    assert blk._backend(xv) == name
    y = blk(xv)
    y.backward(g)
    ga = torch.cat([blk.alpha_pre.grad, blk.alpha_post.grad, blk.alpha_res.grad])
    errs = dict(out=_rel(y, yr), gx=_rel(x.grad, xr.grad), gW=_rel(blk.W_all.weight.grad, Wr.grad),
                galpha=_rel(ga, ar.grad), gbias=_rel(blk.W_all.bias.grad, br.grad),
                out_row=_row_err(y, yr), gx_row=_row_err(x.grad, xr.grad))
    assert all(torch.isfinite(t).all() for t in (y, x.grad, blk.W_all.weight.grad)), (name, T, C)
    # galpha is 3 sums over few tokens at small T: bf16-level noise cancels less
    tol = dict(out_row=3e-2, gx_row=3e-2, galpha=5e-2)
    assert all(v < tol.get(k, 1e-2) for k, v in errs.items()), (name, T, C, errs)
    return errs


# 16-token tile edges (T % 16, the CuTe cluster counts ntl in {1, 2}), the v2
# channel mask (C < 1024) and a single token
_SHAPES = [(16, 1024), (32, 1024), (48, 1024), (4104, 1024), (4112, 1024), (37, 384), (129, 256), (1, 128)]


def _shape_cases():
    for T, C in _SHAPES:
        yield pytest.param(T, C, "v2", id=f"v2-{T}x{C}")
        if C == 1024 and T % 16 == 0:
            yield pytest.param(T, C, "cute", id=f"cute-{T}x{C}")


def _skip_unrunnable(name):
    if name == "cute" and not _cute_runs():
        pytest.skip("CuTe mHC needs sm90 and nvidia-cutlass-dsl >= 4.5")


@pytest.mark.parametrize("merge", [True, False])
@pytest.mark.parametrize("T,C,name", list(_shape_cases()))
def test_block_shapes_vs_fp64_reference(T, C, name, merge, backend_env):
    _skip_unrunnable(name)
    _check_block(T, C, name, merge=merge)


@pytest.mark.parametrize("layout", ["misaligned", "strided"])
@pytest.mark.parametrize("name", ["v2", "cute"])
def test_block_view_inputs(name, layout, backend_env):
    _skip_unrunnable(name)
    _check_block(48, 1024, name, layout=layout)


@pytest.mark.parametrize("name", ["v2", "cute"])
def test_block_layer_ignores_li(name, backend_env):
    _skip_unrunnable(name)
    _check_block(32, 1024, name, ignore_li=True)


def _model(backend, seed=0):
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig, ModernBertForMaskedLM

    _set_backend(backend)
    torch.manual_seed(seed)
    cfg = ModernBertConfig(vocab_size=32, hidden_size=C, intermediate_size=2048, num_hidden_layers=2,
                           num_attention_heads=8, use_mhc_lite=True, mhc_n_streams=4,
                           mhc_triton_fused=backend is not None, mhc_lite_wrapping_level="sublayers",
                           no_mlp_on_first_layer=False, use_canon_layers=False, use_repo=False)
    return ModernBertForMaskedLM(cfg).to("cuda", torch.bfloat16).train()


def test_sublayer_model_defaults(backend_env):
    m = _model(None)
    layer = m.model.layers[0]
    for blk in (layer.mhc_attn, layer.mhc_mlp):
        assert blk.merge_residual is False
        assert abs(blk.W_all.weight.float().std().item() - 0.02) < 2e-3
        assert blk.W_all.bias.abs().max().item() == 0
        assert [p.item() for p in (blk.alpha_pre, blk.alpha_post, blk.alpha_res)] == [1.0, 1.0, 1.0]
    head = m.model.mhc_head
    assert abs(head.hc_fn.float().std().item() - 0.02) < 2e-3
    assert head.hc_base.abs().max().item() == 0 and head.hc_scale.item() == 1.0

    ids = torch.randint(4, 32, (512,), device="cuda")
    e = m.model.embeddings(ids)
    xs = m.model._mhc_expand(e, 4)
    assert all(torch.equal(xs[:, j], e) for j in range(4)), "embedding is copied into every stream"
    f = layer.mhc_mlp.layer(e)
    assert not torch.allclose(f.float(), (e + f).float()), "the branch returns f alone"


def test_sublayer_model_backends_agree(backend_env):
    B, S = 4, 512
    torch.manual_seed(1)
    ids = torch.randint(4, 32, (B * S,), device="cuda")
    labels = torch.full_like(ids, -100)
    msk = torch.rand(B * S, device="cuda") < 0.15
    labels[msk] = ids[msk]
    cu = torch.arange(0, B * S + 1, S, device="cuda", dtype=torch.int32)
    pos = torch.arange(S, device="cuda").repeat(B)
    keys = ("layers.0.mhc_attn.W_all.weight", "layers.1.mhc_mlp.W_all.weight", "mhc_head.hc_fn",
            "tok_embeddings.weight", "layers.0.enc.attn.Wqkv.weight")
    res = {}
    for name, env, _ in _backends():
        m = _model(env)
        # at init every stream holds the same embedding and the zero-init output
        # projections make f = 0, so the mHC coefficient grads are exactly 0: move
        # every weight off that point, identically for each backend
        with torch.no_grad():
            torch.manual_seed(2)
            for p in m.parameters():
                p.add_((torch.randn(p.shape, device="cuda") * 0.02).to(p.dtype))
        out = m(input_ids=ids, labels=labels, cu_seqlens=cu, max_seqlen=S, position_ids=pos)
        out["loss"].backward()
        grads = {k: p.grad.float().clone() for k, p in m.named_parameters()
                 if p.grad is not None and any(k.endswith(s) for s in keys)}
        assert len(grads) == len(keys) and all(torch.isfinite(v).all() for v in grads.values()), name
        res[name] = (out["loss"].item(), grads)
    l0, g0 = res["pytorch"]
    for name, (l, g) in res.items():
        assert abs(l - l0) < 5e-3, (name, l, l0)
        for k in g0:
            assert _rel(g[k], g0[k]) < 5e-2, (name, k, _rel(g[k], g0[k]))
