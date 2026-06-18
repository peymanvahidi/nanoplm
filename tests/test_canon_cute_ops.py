"""CuTeDSL canon backend vs the Triton reference ops (sm90 GPU required)."""
import pytest
import torch

cuda_sm90 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (9, 0)
)
pytestmark = pytest.mark.skipif(
    not cuda_sm90, reason="requires CUDA sm90 (H100/GH200)"
)

C = 1024
K = 7
R = 3
EPS = 1e-5


def _seqid(T, avg=64, mn=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    sc = float(avg - mn)
    L, tot = [], 0
    while tot < T:
        n = mn + int(torch.empty(1).exponential_(1.0 / sc, generator=g).item())
        n = min(max(mn, n), T - tot)
        L.append(n)
        tot += n
    return torch.repeat_interleave(
        torch.arange(len(L), dtype=torch.int32), torch.tensor(L)
    ).cuda()


def _inputs(T, seed=0):
    torch.manual_seed(seed)
    dt = torch.bfloat16
    return dict(
        x=torch.randn(T, C, device="cuda", dtype=dt),
        go=torch.randn(T, C, device="cuda", dtype=dt),
        w=torch.randn(C, K, device="cuda", dtype=dt) * 0.1,
        b=torch.randn(C, device="cuda", dtype=dt) * 0.1,
        g=torch.randn(C, device="cuda", dtype=dt) * 0.1 + 1.0,
        sid=_seqid(T, seed=seed),
    )


@pytest.fixture(scope="module")
def ops():
    import nanoplm.pretraining.models.modern_bert.canon_ops  # noqa: F401
    import nanoplm.pretraining.models.modern_bert.canon_cute_ops as cute_ops
    return torch.ops.nanoplm_canon, torch.ops.nanoplm_canon_cute, cute_ops


@pytest.mark.parametrize("T", [4096, 8192, 65536])
def test_fwd_matches_triton(ops, T):
    tri, cu, _ = ops
    i = _inputs(T)
    out_t, inv_t = tri.varlen_rms_conv_fwd(i["x"], i["sid"], i["g"], EPS,
                                           i["w"], i["b"], R)
    out_c, inv_c = cu.rms_conv_fwd(i["x"], i["sid"], i["g"], EPS,
                                   i["w"], i["b"])
    assert torch.allclose(inv_c, inv_t, atol=1e-5, rtol=1e-4)
    err = (out_c.float() - out_t.float()).abs().max().item()
    assert err < 0.05, f"fwd max err {err}"


@pytest.mark.parametrize("T", [4096, 65536])
def test_backward_matches_triton(ops, T):
    tri, _, cute_ops = ops
    i = _inputs(T)

    def run(fwd_fn):
        x = i["x"].clone().requires_grad_(True)
        g = i["g"].clone().requires_grad_(True)
        w = i["w"].clone().requires_grad_(True)
        b = i["b"].clone().requires_grad_(True)
        out = fwd_fn(x, i["sid"], g, EPS, w, b, R)
        out.backward(i["go"])
        return out, x.grad, g.grad, w.grad, b.grad

    from nanoplm.pretraining.models.modern_bert.canon_ops import (
        varlen_rms_canon_conv as tri_fn,
    )
    o_t, dx_t, dg_t, dw_t, db_t = run(tri_fn)
    o_c, dx_c, dg_c, dw_c, db_c = run(cute_ops.varlen_rms_canon_conv)

    assert (o_c.float() - o_t.float()).abs().max().item() < 0.05
    assert (dx_c.float() - dx_t.float()).abs().max().item() < 0.05
    scale = dg_t.float().abs().max().clamp_min(1.0).item()
    assert (dg_c.float() - dg_t.float()).abs().max().item() < 2e-2 * scale
    scale = dw_t.float().abs().max().clamp_min(1.0).item()
    assert (dw_c.float() - dw_t.float()).abs().max().item() < 2e-2 * scale
    scale = db_t.float().abs().max().clamp_min(1.0).item()
    assert (db_c.float() - db_t.float()).abs().max().item() < 2e-2 * scale


def test_activation_checkpointing(ops):
    _, _, cute_ops = ops
    from torch.utils.checkpoint import checkpoint
    i = _inputs(4096)
    x = i["x"].clone().requires_grad_(True)
    g = i["g"].clone().requires_grad_(True)

    def fn(x_):
        return cute_ops.varlen_rms_canon_conv(x_, i["sid"], g, EPS,
                                              i["w"], i["b"], R)

    out_ckpt = checkpoint(fn, x, use_reentrant=False)
    out_ckpt.backward(i["go"])
    dx_ckpt = x.grad.clone()
    x.grad = None
    out = fn(x)
    out.backward(i["go"])
    assert torch.equal(out_ckpt, out)
    assert torch.allclose(dx_ckpt.float(), x.grad.float(), atol=1e-3)


def test_single_sequence(ops):
    tri, cu, _ = ops
    T = 4096
    i = _inputs(T)
    sid = torch.zeros(T, dtype=torch.int32, device="cuda")
    out_t, _ = tri.varlen_rms_conv_fwd(i["x"], sid, i["g"], EPS, i["w"], i["b"], R)
    out_c, _ = cu.rms_conv_fwd(i["x"], sid, i["g"], EPS, i["w"], i["b"])
    assert (out_c.float() - out_t.float()).abs().max().item() < 0.05


def test_compatibility_gate(ops):
    _, _, cute_ops = ops
    x = torch.randn(4096, C, device="cuda", dtype=torch.bfloat16)
    assert cute_ops.is_compatible(x, R)
    assert not cute_ops.is_compatible(x, 2)                      # K != 7
    assert not cute_ops.is_compatible(x.float(), R)              # not bf16
    assert not cute_ops.is_compatible(x[:100], R)                # T % 64 != 0
    for c_ok in (512, 768, 1536, 2048, 3072):                    # supported Cs
        assert cute_ops.is_compatible(
            torch.randn(4096, c_ok, device="cuda", dtype=torch.bfloat16), R)
    for c_bad in (320, 1100, 4096, 8192):    # unsupported Cs (4096: smem wall)
        assert not cute_ops.is_compatible(
            torch.randn(4096, c_bad, device="cuda", dtype=torch.bfloat16), R)


def test_canon_layer_dispatch(ops):
    """ModernBertCanonLayer fused-RMS path routes through the cute backend."""
    from nanoplm.pretraining.models.modern_bert.canon_layer import (
        ModernBertCanonLayer,
    )
    T = 4096
    i = _inputs(T)
    layer = ModernBertCanonLayer(C, kernel_size=K).cuda().to(torch.bfloat16)
    cu_seqlens = torch.tensor([0, 1000, 2500, T], device="cuda",
                              dtype=torch.int32)
    out = layer(
        i["x"], cu_seqlens=cu_seqlens,
        fuse_ln_weight=i["g"], fuse_ln_eps=EPS, fuse_norm_type="rmsnorm",
    )
    # reference through the Triton public fn with the same seq_id derivation
    positions = torch.arange(T, device="cuda", dtype=torch.int32)
    seq_id = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    from nanoplm.pretraining.models.modern_bert.canon_ops import (
        varlen_rms_canon_conv as tri_fn,
    )
    ref = tri_fn(i["x"], seq_id, i["g"], EPS, layer.conv.weight[:, 0, :],
                 layer.conv.bias, R)
    assert (out.float() - ref.float()).abs().max().item() < 0.05


@pytest.mark.parametrize("T", [4096, 65536])
def test_plain_conv_matches_triton(ops, T):
    """Plain (no-norm) path: layer-0 canon with Identity attn_norm."""
    tri, cu, _ = ops
    i = _inputs(T)
    mixed = tri.varlen_conv_fwd(i["x"], i["sid"], i["w"], i["b"], R)
    ref = i["x"] + mixed
    out = cu.conv_fwd(i["x"], i["sid"], i["w"], i["b"])
    assert (out.float() - ref.float()).abs().max().item() < 0.05


def test_plain_conv_backward(ops):
    _, _, cute_ops = ops
    from nanoplm.pretraining.models.modern_bert.canon_ops import (
        varlen_canon_conv as tri_fn,
    )
    i = _inputs(4096)

    def run(fwd):
        x = i["x"].clone().requires_grad_(True)
        w = i["w"].clone().requires_grad_(True)
        b = i["b"].clone().requires_grad_(True)
        out = fwd(x, w, b)
        out.backward(i["go"])
        return out, x.grad, w.grad, b.grad

    o_t, dx_t, dw_t, db_t = run(
        lambda x, w, b: x + tri_fn(x, i["sid"], w, b, R))
    o_c, dx_c, dw_c, db_c = run(
        lambda x, w, b: cute_ops.varlen_canon_conv_residual(
            x, i["sid"], w, b, R))
    assert (o_c.float() - o_t.float()).abs().max().item() < 0.05
    assert (dx_c.float() - dx_t.float()).abs().max().item() < 0.05
    for got, ref in ((dw_c, dw_t), (db_c, db_t)):
        scale = ref.float().abs().max().clamp_min(1.0).item()
        assert (got.float() - ref.float()).abs().max().item() < 2e-2 * scale


def test_canon_layer_plain_dispatch(ops):
    """Identity-norm layer (layer 0) routes the plain path through cute."""
    from nanoplm.pretraining.models.modern_bert.canon_layer import (
        ModernBertCanonLayer,
    )
    T = 4096
    i = _inputs(T)
    layer = ModernBertCanonLayer(C, kernel_size=K).cuda().to(torch.bfloat16)
    cu_seqlens = torch.tensor([0, 900, 2600, T], device="cuda",
                              dtype=torch.int32)
    out = layer(i["x"], cu_seqlens=cu_seqlens)  # no fuse args -> plain path
    positions = torch.arange(T, device="cuda", dtype=torch.int32)
    seq_id = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    from nanoplm.pretraining.models.modern_bert.canon_ops import (
        varlen_canon_conv as tri_fn,
    )
    ref = i["x"] + tri_fn(i["x"], seq_id, layer.conv.weight[:, 0, :],
                          layer.conv.bias, R)
    assert (out.float() - ref.float()).abs().max().item() < 0.05


@pytest.mark.parametrize("Cx", [512, 768, 1536, 2048, 3072])
def test_other_channel_sizes(ops, Cx):
    """Generalized-C path: fwd + all grads vs Triton at C != 1024."""
    tri, cu, cute_ops = ops
    T = 8192
    torch.manual_seed(11)
    dt = torch.bfloat16
    x0 = torch.randn(T, Cx, device="cuda", dtype=dt)
    go = torch.randn(T, Cx, device="cuda", dtype=dt)
    w0 = torch.randn(Cx, K, device="cuda", dtype=dt) * 0.1
    b0 = torch.randn(Cx, device="cuda", dtype=dt) * 0.1
    g0 = torch.randn(Cx, device="cuda", dtype=dt) * 0.1 + 1.0
    sid = _seqid(T, seed=11)

    assert cute_ops.is_compatible(x0, R)

    def run(fwd_fn):
        x = x0.clone().requires_grad_(True)
        g = g0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = fwd_fn(x, sid, g, EPS, w, b, R)
        out.backward(go)
        return out, x.grad, g.grad, w.grad, b.grad

    from nanoplm.pretraining.models.modern_bert.canon_ops import (
        varlen_rms_canon_conv as tri_fn,
    )
    o_t, dx_t, dg_t, dw_t, db_t = run(tri_fn)
    o_c, dx_c, dg_c, dw_c, db_c = run(cute_ops.varlen_rms_canon_conv)

    assert (o_c.float() - o_t.float()).abs().max().item() < 0.05
    assert (dx_c.float() - dx_t.float()).abs().max().item() < 0.05
    for got, ref in ((dg_c, dg_t), (dw_c, dw_t), (db_c, db_t)):
        scale = ref.float().abs().max().clamp_min(1.0).item()
        assert (got.float() - ref.float()).abs().max().item() < 2e-2 * scale

    # plain (no-norm) path at this C
    mixed = tri.varlen_conv_fwd(x0, sid, w0, b0, R)
    out_p = cu.conv_fwd(x0, sid, w0, b0)
    assert (out_p.float() - (x0 + mixed).float()).abs().max().item() < 0.05


@pytest.mark.parametrize("Cx", [2048, 3072])
def test_canon_b_shaped_plain_dispatch(ops, Cx):
    """canon_b path: plain (no-norm) layer at qkv widths routes through cute
    with correct grads (2048 = GQAr2, 3072 = MHA at hidden=1024)."""
    from nanoplm.pretraining.models.modern_bert.canon_layer import (
        ModernBertCanonLayer,
    )
    from nanoplm.pretraining.models.modern_bert.canon_ops import (
        varlen_canon_conv as tri_fn,
    )
    T = 8192
    torch.manual_seed(0)
    layer = ModernBertCanonLayer(Cx, kernel_size=K).cuda().to(torch.bfloat16)
    x = torch.randn(T, Cx, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    go = torch.randn(T, Cx, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 900, 2600, 5000, T], device="cuda",
                              dtype=torch.int32)
    out = layer(x, cu_seqlens=cu_seqlens)  # no fuse args -> plain path
    out.backward(go)

    positions = torch.arange(T, device="cuda", dtype=torch.int32)
    seq_id = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    xr = x.detach().clone().requires_grad_(True)
    wr = layer.conv.weight[:, 0, :].detach().clone().requires_grad_(True)
    br = layer.conv.bias.detach().clone().requires_grad_(True)
    ref = xr + tri_fn(xr, seq_id, wr, br, R)
    ref.backward(go)

    assert (out.float() - ref.float()).abs().max().item() < 0.05
    assert (x.grad.float() - xr.grad.float()).abs().max().item() < 0.05
    scale = wr.grad.float().abs().max().clamp_min(1.0).item()
    assert (layer.conv.weight.grad[:, 0, :].float()
            - wr.grad.float()).abs().max().item() < 2e-2 * scale
    assert (layer.conv.bias.grad.float()
            - br.grad.float()).abs().max().item() < 2e-2
