"""Regression tests for the fused MoE router (top-k + expert histogram + padded gate).

These cover three optimisations that must be *numerically inert*:

* ``nanoplm_router::topk_counts`` replaces ``torch.topk`` in ``Router.forward``.
  It must return the same expert set, in the same (descending-score) order, and
  the values gathered with those indices must be bit-identical to
  ``torch.topk(...).values``.
* The same kernel emits the per-expert assignment histogram that used to be
  built with ``ones_like`` / ``zeros`` / ``scatter_add_`` in ``MoELayer.forward``
  and is consumed by the expert-bias load-balancing update in
  ``pure_pipeline.py``. It must match ``scatter_add_`` exactly.
* The router gate GEMM is padded to a multiple of 8 output features so cuBLAS
  picks an sm90 kernel instead of an Ampere ``align1`` fallback. The padded
  columns must not leak into logits, losses, or ``gate.weight.grad``.
"""
from __future__ import annotations

import pytest
import torch

from nanoplm.pretraining.models.modern_bert.moe import (
    Router,
    _router_topk_indices,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available")),
]


def _ref_counts(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    flat = indices.reshape(-1)
    return torch.zeros(
        num_experts, device=flat.device, dtype=torch.int64,
    ).scatter_add_(0, flat, torch.ones_like(flat))


# ── fused top-k: exact equivalence with torch.topk ───────────────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(37, 8, 2), (1024, 47, 3), (4096, 47, 3), (64, 256, 4)])
def test_topk_indices_match_torch_topk(device, shape):
    m, e, k = shape
    torch.manual_seed(shape[0])
    x = torch.randn(m, e, device=device, dtype=torch.float32)

    idx, cnt = _router_topk_indices(x, k, True)
    ref = torch.topk(x, k, dim=-1)

    assert idx.dtype == torch.int64
    assert idx.shape == (m, k)
    assert torch.equal(idx, ref.indices)
    # The values the router actually uses are gathered, so this is the
    # property that has to hold bit-exactly.
    assert torch.equal(x.gather(-1, idx), ref.values)
    assert torch.equal(cnt, _ref_counts(ref.indices, e))
    assert int(cnt.sum()) == m * k


@pytest.mark.parametrize("device", DEVICES)
def test_topk_counts_can_be_skipped(device):
    x = torch.randn(128, 47, device=device)
    idx, cnt = _router_topk_indices(x, 3, False)
    assert cnt is None
    assert idx.shape == (128, 3)


@pytest.mark.parametrize("device", DEVICES)
def test_topk_tie_break_is_lowest_index(device):
    """Deterministic tie-break: among equal scores the lowest expert wins."""
    x = torch.zeros(4, 8, device=device)
    idx, _ = _router_topk_indices(x, 3, False)
    expected = torch.tensor([0, 1, 2], device=device).expand(4, 3)
    assert torch.equal(idx, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_topk_handles_masked_experts(device):
    """Group-limited routing masks with finfo.min; those must never be picked."""
    e, k = 16, 3
    x = torch.randn(64, e, device=device)
    x[:, :8] = torch.finfo(x.dtype).min
    idx, cnt = _router_topk_indices(x, k, True)
    assert int(idx.min()) >= 8
    assert int(cnt[:8].sum()) == 0


@CUDA
def test_topk_outputs_are_not_differentiable():
    x = torch.randn(256, 47, device="cuda", requires_grad=True)
    idx, cnt = _router_topk_indices(x, 3, True)
    assert not idx.requires_grad
    assert not cnt.requires_grad


# ── Router: end-to-end equivalence against the pre-optimisation formulation ──


def _reference_router_forward(router: Router, x: torch.Tensor):
    """The router as it was written before the fused top-k / padded gate."""
    raw_logits = torch.nn.functional.linear(x, router.gate.weight).float()
    logits = raw_logits.sigmoid()
    z_loss = raw_logits.float().square().mean()
    selection = logits if router.correction_bias is None else logits + router.correction_bias
    weights, indices = torch.topk(selection, router.top_k, dim=-1)
    if router.correction_bias is not None:
        weights = logits.gather(dim=-1, index=indices)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return weights / denom, indices, z_loss, logits


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bias_correction", [True, False])
def test_router_matches_reference(device, bias_correction):
    torch.manual_seed(7)
    h, e, k, t = 64, 47, 3, 512
    router = Router(h, e, k, bias_correction=bias_correction).to(device)
    if bias_correction:
        with torch.no_grad():
            router.correction_bias.normal_(0, 0.05)
    x = torch.randn(t, h, device=device)

    w, idx, z, logits, counts = router(x)
    rw, ridx, rz, rlogits = _reference_router_forward(router, x)

    # Expert *selection* must be identical. The logits themselves are produced
    # by a differently-tiled GEMM (the gate is padded 47 -> 48), so they agree
    # only to fp32 rounding, not bit-for-bit.
    assert torch.equal(idx, ridx)
    torch.testing.assert_close(logits, rlogits, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(w, rw, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(z, rz, rtol=1e-6, atol=1e-6)
    if bias_correction:
        assert torch.equal(counts, _ref_counts(ridx, e))
    else:
        assert counts is None


@pytest.mark.parametrize("device", DEVICES)
def test_router_gate_padding_is_transparent(device):
    """The padded gate must produce the same logits and the same weight grad."""
    torch.manual_seed(11)
    h, e, k, t = 64, 47, 3, 256
    router = Router(h, e, k, bias_correction=True).to(device)
    assert router._gate_pad == 1, "E=47 must be padded to 48"

    x = torch.randn(t, h, device=device)
    _, _, _, logits, _ = router(x)
    ref = torch.nn.functional.linear(x, router.gate.weight).float().sigmoid()
    torch.testing.assert_close(logits, ref, rtol=2e-5, atol=2e-5)

    router.zero_grad(set_to_none=True)
    logits.sum().backward()
    g_fused = router.gate.weight.grad.clone()
    assert g_fused.shape == (e, h)
    assert torch.isfinite(g_fused).all()

    router.zero_grad(set_to_none=True)
    ref2 = torch.nn.functional.linear(x, router.gate.weight).float().sigmoid()
    ref2.sum().backward()
    torch.testing.assert_close(g_fused, router.gate.weight.grad, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_router_no_pad_when_already_aligned(device):
    router = Router(32, 48, 3, bias_correction=True).to(device)
    assert router._gate_pad == 0
    x = torch.randn(64, 32, device=device)
    _, _, _, logits, counts = router(x)
    assert logits.shape == (64, 48)
    assert int(counts.sum()) == 64 * 3


@pytest.mark.parametrize("device", DEVICES)
def test_router_gradients_flow_to_gate_and_weights(device):
    torch.manual_seed(3)
    router = Router(32, 47, 3, bias_correction=True).to(device)
    x = torch.randn(128, 32, device=device, requires_grad=True)
    w, _, z, _, _ = router(x)
    (w.sum() + z).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert router.gate.weight.grad is not None
    assert torch.isfinite(router.gate.weight.grad).all()
    assert router.gate.weight.grad.abs().sum() > 0


# ── MoELayer bookkeeping ─────────────────────────────────────────────────────


def _moe_cfg(**overrides):
    from nanoplm.pretraining.models.modern_bert.config import ModernBertConfig

    cfg = ModernBertConfig(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_moe=True,
        moe_num_experts=47,
        moe_top_k=3,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.parametrize("bias_correction", [True, False])
def test_moe_layer_tracks_counts_only_when_consumed(bias_correction):
    from nanoplm.pretraining.models.modern_bert.moe import MoELayer

    for backend in ("cutlass", "sonicmoe"):
        cfg = _moe_cfg(
            moe_use_bias_correction=bias_correction,
            moe_kernel_backend=backend,
        )
        layer = MoELayer(cfg)
        assert layer._track_expert_counts is bias_correction
        # Only the sonicmoe path needs the router to emit the histogram; the
        # cutlass path builds `counts` for the grouped GEMM anyway.
        assert layer.router.track_expert_counts is (
            bias_correction and backend == "sonicmoe"
        )
        assert layer.last_expert_counts is None


# ── routed-expert Wi gate/up layout version ──────────────────────────────────
#
# `Wi`'s last dim is stored interleaved gate-first (layout v2). A v1
# (`[up | gate]` block-concat) checkpoint is a *different function* under v2
# code, so it must fail loudly rather than resume into silent divergence.


def test_gated_moe_layer_carries_a_persistent_layout_marker():
    from nanoplm.pretraining.models.modern_bert.moe import (
        WI_GATED_LAYOUT_VERSION, MoELayer,
    )

    sd = MoELayer(_moe_cfg()).state_dict()
    assert "_wi_gated_layout_version" in sd, (
        "gated MoE layers must persist the Wi layout version so a v1 "
        "checkpoint cannot be resumed silently"
    )
    assert int(sd["_wi_gated_layout_version"][0]) == WI_GATED_LAYOUT_VERSION
    # 1-D, not 0-dim: DCP's _distribute_state_dict forces 0-dim buffers to CPU.
    assert sd["_wi_gated_layout_version"].ndim == 1


def test_srelu_moe_layer_has_no_layout_marker():
    """sReLU's Wi is ungated — no convention to get wrong, stays compatible."""
    from nanoplm.pretraining.models.modern_bert.moe import MoELayer

    layer = MoELayer(_moe_cfg(mlp_activation="srelu"))
    sd = layer.state_dict()
    assert "_wi_gated_layout_version" not in sd
    layer.load_state_dict(sd)  # round-trips with no marker at all


def test_current_layout_round_trips():
    from nanoplm.pretraining.models.modern_bert.moe import MoELayer

    src, dst = MoELayer(_moe_cfg()), MoELayer(_moe_cfg())
    # A bare MoELayer leaves Wi as torch.empty (init_weights lives on the full
    # model), and uninitialized memory can hold NaN, which never compares equal.
    torch.manual_seed(0)
    with torch.no_grad():
        src.Wi.copy_(torch.randn_like(src.Wi))
    dst.load_state_dict(src.state_dict())
    assert torch.equal(dst.Wi, src.Wi)


def test_v1_checkpoint_is_rejected_with_the_conversion_recipe():
    from nanoplm.pretraining.models.modern_bert.moe import MoELayer

    layer = MoELayer(_moe_cfg())
    v1_sd = {k: v for k, v in layer.state_dict().items()
             if k != "_wi_gated_layout_version"}
    with pytest.raises(RuntimeError) as exc:
        layer.load_state_dict(v1_sd)
    msg = str(exc.value)
    assert "layout mismatch" in msg
    assert "convert_wi_state_dict_v1_to_v2" in msg
    # strict=False must not downgrade this to a silent partial load: a wrong
    # gate/up pairing is a correctness fault, not a missing-key nuisance.
    with pytest.raises(RuntimeError):
        layer.load_state_dict(v1_sd, strict=False)


def test_convert_v1_to_v2_preserves_the_function_and_loads():
    from nanoplm.pretraining.models.modern_bert.moe import (
        MoELayer, convert_wi_state_dict_v1_to_v2,
    )
    import torch.nn.functional as F

    layer = MoELayer(_moe_cfg())
    E, H, two_i = layer.Wi.shape
    half = two_i // 2
    torch.manual_seed(0)
    wi_v1 = torch.randn(E, H, two_i)

    v1_sd = {k: v for k, v in layer.state_dict().items()
             if k != "_wi_gated_layout_version"}
    v1_sd["Wi"] = wi_v1
    # Optimizer state carries the same column order and must be permuted too.
    momentum = torch.randn_like(wi_v1)
    opt_sd = {"state": {"Wi": {"momentum_buffer": momentum,
                               "step": torch.tensor(7)}}}

    converted = convert_wi_state_dict_v1_to_v2(v1_sd, opt_sd)
    assert converted == ["Wi"]
    layer.load_state_dict(v1_sd)  # no longer raises

    for W_v1, W_v2 in ((wi_v1, layer.Wi.detach()),
                       (momentum, opt_sd["state"]["Wi"]["momentum_buffer"])):
        # gate: v1 second half -> v2 even columns; up: v1 first half -> odd.
        assert torch.equal(W_v2[..., 0::2], W_v1[..., half:])
        assert torch.equal(W_v2[..., 1::2], W_v1[..., :half])

    # ...and therefore the same function: apply both conventions to the SAME
    # preactivation, permuted the same way as its weight columns.
    h1 = torch.randn(5, two_i)
    h2 = torch.empty_like(h1)
    h2[..., 0::2], h2[..., 1::2] = h1[..., half:], h1[..., :half]
    # `.contiguous()` on both sides so the comparison is of the *pairing*, not
    # of ATen's vectorized-vs-strided silu, which differ in the last ulp.
    v1 = F.silu(h1[..., half:].contiguous()) * h1[..., :half].contiguous()
    v2 = F.silu(h2[..., 0::2].contiguous()) * h2[..., 1::2].contiguous()
    assert torch.equal(v1, v2)
    assert opt_sd["state"]["Wi"]["step"].item() == 7  # non-Wi state untouched


def test_convert_v1_to_v2_leaves_srelu_alone():
    from nanoplm.pretraining.models.modern_bert.moe import (
        MoELayer, convert_wi_state_dict_v1_to_v2,
    )

    sd = MoELayer(_moe_cfg(mlp_activation="srelu")).state_dict()
    with pytest.raises(ValueError, match="no gated MoE routed-expert"):
        convert_wi_state_dict_v1_to_v2(dict(sd))
