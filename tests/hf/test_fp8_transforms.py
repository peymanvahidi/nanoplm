"""Tensor math for the FP8 -> dense restore. No Transformer Engine needed.

The forward conversion (``_copy_mlp_branch_to_te_layernorm_mlp`` in fp8.py) has
three fc1/fc2 layouts, keyed by the activation string
``_te_layernorm_mlp_activation`` returns:

- "swiglu" (ModernBertSwiGLUMLP): fc1 = [gate, up] from native Wi = [up, gate]
  -- halves swap, for weight and bias alike. Getting the inverse backwards
  silently computes silu(up) * gate instead of silu(gate) * x_proj -- a model
  that loads, embeds plausibly, and is wrong.
- "geglu" (ModernBertMLP with act=F.gelu): fc1 = Wi unchanged (TE gates on the
  first half, which is exactly what ModernBertMLP already does).
- "srelu" (ModernBertSReluMLP): fc1 = Wi unchanged; fc2 = Wo_weight.T.

fc2 is unchanged for the gated forms (swiglu/geglu); only srelu transposes.
fc2 biases carry over unchanged in all three layouts.
"""

import pytest
import torch

from nanoplm.pretraining.fp8 import (
    native_wi_to_te_fc1,
    native_wo_to_te_fc2,
    te_fc1_to_native_wi,
    te_fc2_to_native_wo,
)

# ---------------------------------------------------------------------------
# fc1 (Wi): known layouts with distinguishable halves.
# ---------------------------------------------------------------------------


def _make_halves(inter: int, hidden: int) -> tuple[torch.Tensor, torch.Tensor]:
    up = torch.arange(inter * hidden, dtype=torch.float32).reshape(inter, hidden)
    gate = up + 1000.0
    return up, gate


def test_swiglu_native_wi_to_te_fc1_swaps_halves():
    up, gate = _make_halves(8, 4)
    native_wi = torch.cat([up, gate], dim=0)  # native order: [up, gate]
    te_fc1 = torch.cat([gate, up], dim=0)  # TE order:     [gate, up]
    torch.testing.assert_close(native_wi_to_te_fc1(native_wi, "swiglu"), te_fc1)


def test_swiglu_te_fc1_to_native_wi_swaps_halves_back():
    up, gate = _make_halves(8, 4)
    native_wi = torch.cat([up, gate], dim=0)
    te_fc1 = torch.cat([gate, up], dim=0)
    torch.testing.assert_close(te_fc1_to_native_wi(te_fc1, "swiglu"), native_wi)


def test_swiglu_fc1_bias_swaps_halves_too():
    up, gate = _make_halves(8, 1)
    up_bias, gate_bias = up.squeeze(1), gate.squeeze(1)
    native_bias = torch.cat([up_bias, gate_bias], dim=0)
    te_bias = torch.cat([gate_bias, up_bias], dim=0)
    torch.testing.assert_close(native_wi_to_te_fc1(native_bias, "swiglu"), te_bias)
    torch.testing.assert_close(te_fc1_to_native_wi(te_bias, "swiglu"), native_bias)


@pytest.mark.parametrize("activation", ["geglu", "srelu"])
def test_non_swiglu_fc1_unchanged(activation):
    up, gate = _make_halves(8, 4)
    wi = torch.cat([up, gate], dim=0)
    torch.testing.assert_close(native_wi_to_te_fc1(wi, activation), wi)
    torch.testing.assert_close(te_fc1_to_native_wi(wi, activation), wi)


# ---------------------------------------------------------------------------
# fc2 (Wo): only srelu transposes.
# ---------------------------------------------------------------------------


def test_srelu_native_wo_to_te_fc2_transposes():
    wo_weight = torch.randn(8, 4)  # (intermediate, hidden)
    fc2 = wo_weight.T  # (hidden, intermediate)
    result = native_wo_to_te_fc2(wo_weight, "srelu")
    torch.testing.assert_close(result, fc2)
    assert result.shape == fc2.shape


def test_srelu_te_fc2_to_native_wo_transposes_back():
    wo_weight = torch.randn(8, 4)  # (intermediate, hidden)
    fc2 = wo_weight.T  # (hidden, intermediate)
    result = te_fc2_to_native_wo(fc2, "srelu")
    torch.testing.assert_close(result, wo_weight)
    assert result.shape == wo_weight.shape


@pytest.mark.parametrize("activation", ["swiglu", "geglu"])
def test_gated_fc2_unchanged(activation):
    w = torch.randn(4, 8)
    torch.testing.assert_close(native_wo_to_te_fc2(w, activation), w)
    torch.testing.assert_close(te_fc2_to_native_wo(w, activation), w)


def test_srelu_fc2_rejects_non_2d_input():
    with pytest.raises(ValueError):
        native_wo_to_te_fc2(torch.randn(8), "srelu")
    with pytest.raises(ValueError):
        te_fc2_to_native_wo(torch.randn(8), "srelu")


# ---------------------------------------------------------------------------
# Round trips.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["swiglu", "geglu", "srelu"])
def test_fc1_weight_roundtrips(activation):
    wi = torch.randn(16, 4)
    torch.testing.assert_close(
        te_fc1_to_native_wi(native_wi_to_te_fc1(wi, activation), activation), wi
    )


@pytest.mark.parametrize("activation", ["swiglu", "geglu", "srelu"])
def test_fc1_bias_roundtrips(activation):
    bias = torch.randn(16)
    torch.testing.assert_close(
        te_fc1_to_native_wi(native_wi_to_te_fc1(bias, activation), activation), bias
    )


@pytest.mark.parametrize("activation", ["swiglu", "geglu", "srelu"])
def test_fc2_weight_roundtrips(activation):
    w = torch.randn(8, 4)
    torch.testing.assert_close(
        te_fc2_to_native_wo(native_wo_to_te_fc2(w, activation), activation), w
    )


# ---------------------------------------------------------------------------
# Error handling.
# ---------------------------------------------------------------------------


def test_swiglu_fc1_rejects_odd_first_dimension():
    with pytest.raises(ValueError, match="even"):
        native_wi_to_te_fc1(torch.randn(7, 4), "swiglu")
    with pytest.raises(ValueError, match="even"):
        te_fc1_to_native_wi(torch.randn(7, 4), "swiglu")


def test_fc1_unknown_activation_raises():
    with pytest.raises(ValueError, match="bogus"):
        native_wi_to_te_fc1(torch.randn(8, 4), "bogus")
    with pytest.raises(ValueError, match="bogus"):
        te_fc1_to_native_wi(torch.randn(8, 4), "bogus")


def test_fc2_unknown_activation_raises():
    with pytest.raises(ValueError, match="bogus"):
        native_wo_to_te_fc2(torch.randn(8, 4), "bogus")
    with pytest.raises(ValueError, match="bogus"):
        te_fc2_to_native_wo(torch.randn(8, 4), "bogus")
