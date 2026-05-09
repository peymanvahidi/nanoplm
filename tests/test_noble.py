"""Tests for NOBLE (Nonlinear lOw-rank Branch for Linear Enhancement).

Verifies CosNet, NOBLELinear, init properties, and model integration.
"""

import math

import pytest
import torch
import torch.nn as nn

from nanoplm.pretraining.models.modern_bert.modeling import (
    CosNet,
    ModernBertConfig,
    NOBLELinear,
    _build_linear,
)


# ---- Step 1: CosNet unit tests ----


class TestCosNet:
    """Test CosNet module in isolation."""

    def test_output_shape(self):
        rank = 64
        cosnet = CosNet(rank)
        x = torch.randn(8, 32, rank)  # (batch, seq, rank)
        out = cosnet(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_output_bounded(self):
        """CosNet output must be in [-1, 1] since it ends with cos()."""
        rank = 32
        cosnet = CosNet(rank)
        x = torch.randn(100, rank) * 10  # large inputs
        out = cosnet(x)
        assert out.min() >= -1.0, f"Min {out.min()} < -1"
        assert out.max() <= 1.0, f"Max {out.max()} > 1"

    def test_init_ranges(self):
        """Verify init values match paper specs."""
        rank = 128
        cosnet = CosNet(rank, omega_range=(0.8, 1.2), phi_std=0.1)
        # Frequencies should be in [0.8, 1.2]
        assert cosnet.omega1.min() >= 0.8
        assert cosnet.omega1.max() <= 1.2
        assert cosnet.omega2.min() >= 0.8
        assert cosnet.omega2.max() <= 1.2
        # Phases: normal(0, 0.1) — check std is roughly right
        assert cosnet.phi1.mean().abs() < 0.1  # mean near 0
        assert cosnet.phi2.mean().abs() < 0.1
        # M: Xavier — check it's not zero or identity
        assert cosnet.M.abs().sum() > 0
        assert not torch.allclose(cosnet.M, torch.eye(rank))

    def test_gradients_flow(self):
        """All CosNet params must receive gradients."""
        rank = 16
        cosnet = CosNet(rank)
        x = torch.randn(4, rank, requires_grad=True)
        out = cosnet(x)
        loss = out.sum()
        loss.backward()
        for name, p in cosnet.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ---- Step 1: NOBLELinear unit tests ----


class TestNOBLELinear:
    """Test NOBLELinear module in isolation."""

    def test_output_shape(self):
        noble = NOBLELinear(64, 128, rank=16)
        x = torch.randn(4, 10, 64)
        out = noble(x)
        assert out.shape == (4, 10, 128)

    def test_matches_linear_at_init(self):
        """At init, NOBLE branch should be near-silent (W_up ≈ 0).

        Output should be very close to main linear output.
        """
        torch.manual_seed(42)
        in_f, out_f, rank = 128, 256, 32
        noble = NOBLELinear(in_f, out_f, rank, alpha=0.01)
        x = torch.randn(4, 10, in_f)

        full_out = noble(x)
        main_out = noble.linear(x)
        branch_out = noble.noble_w_up(noble.cosnet(noble.W_down(x)))

        # Branch magnitude should be much smaller than main
        main_norm = main_out.norm()
        branch_norm = branch_out.norm()
        ratio = branch_norm / main_norm
        assert ratio < 0.1, f"Branch/main ratio {ratio:.4f} too large at init"

    def test_weight_property(self):
        """The .weight property must return the SAME tensor as linear.weight."""
        noble = NOBLELinear(64, 128, rank=16)
        assert noble.weight is noble.linear.weight
        assert noble.weight.data_ptr() == noble.linear.weight.data_ptr()

    def test_bias_property_no_bias(self):
        noble = NOBLELinear(64, 128, rank=16, bias=False)
        assert noble.bias is None

    def test_bias_property_with_bias(self):
        noble = NOBLELinear(64, 128, rank=16, bias=True)
        assert noble.bias is noble.linear.bias
        assert noble.bias is not None

    def test_weight_inplace_init(self):
        """nn.init functions must work through the .weight property."""
        noble = NOBLELinear(64, 128, rank=16)
        # This is exactly what init_weights() does
        nn.init.zeros_(noble.weight)
        assert noble.linear.weight.abs().sum() == 0

        nn.init.uniform_(noble.weight, -0.1, 0.1)
        assert noble.linear.weight.abs().sum() > 0

    def test_w_up_near_zero(self):
        """W_up init should be near-zero with std = alpha/sqrt(rank)."""
        rank = 64
        alpha = 0.01
        noble = NOBLELinear(768, 768, rank=rank, alpha=alpha)
        expected_std = alpha / math.sqrt(rank)
        actual_std = noble.noble_w_up.weight.std().item()
        # Allow 3x tolerance for random init
        assert actual_std < expected_std * 3, (
            f"W_up std {actual_std:.6f} too far from expected {expected_std:.6f}"
        )

    def test_all_params_have_grad(self):
        """Every parameter in NOBLELinear must receive gradients."""
        noble = NOBLELinear(64, 128, rank=16, bias=True)
        x = torch.randn(2, 5, 64, requires_grad=True)
        out = noble(x)
        loss = out.sum()
        loss.backward()

        for name, p in noble.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_param_count(self):
        """Verify exact param count matches expectation."""
        in_f, out_f, rank = 768, 768, 64
        noble = NOBLELinear(in_f, out_f, rank, bias=False)

        expected = (
            in_f * out_f          # linear.weight
            + in_f * rank         # W_down.weight
            + 4 * rank            # omega1, phi1, omega2, phi2
            + rank * rank         # M
            + rank * out_f        # noble_w_up.weight
        )
        actual = sum(p.numel() for p in noble.parameters())
        assert actual == expected, f"Param count: expected {expected}, got {actual}"

    def test_named_parameters_structure(self):
        """Verify param names for optimizer routing."""
        noble = NOBLELinear(64, 128, rank=16, bias=True)
        names = {name for name, _ in noble.named_parameters()}
        expected = {
            "linear.weight",
            "linear.bias",
            "W_down.weight",
            "cosnet.omega1",
            "cosnet.phi1",
            "cosnet.M",
            "cosnet.omega2",
            "cosnet.phi2",
            "noble_w_up.weight",
        }
        assert names == expected, f"Unexpected param names: {names - expected}"


# ---- Step 1: Factory function test ----


class TestBuildLinear:
    """Test the _build_linear factory."""

    def test_returns_linear_when_disabled(self):
        config = ModernBertConfig(use_noble=False)
        layer = _build_linear(64, 128, config, bias=False)
        assert isinstance(layer, nn.Linear)
        assert not isinstance(layer, NOBLELinear)

    def test_returns_noble_when_enabled(self):
        config = ModernBertConfig(use_noble=True, noble_rank=32)
        layer = _build_linear(64, 128, config, bias=False)
        assert isinstance(layer, NOBLELinear)
        assert layer.rank == 32

    def test_bias_forwarded(self):
        config = ModernBertConfig(use_noble=True, noble_rank=16)
        layer = _build_linear(64, 128, config, bias=True)
        assert isinstance(layer, NOBLELinear)
        assert layer.bias is not None

    def test_config_params_forwarded(self):
        config = ModernBertConfig(
            use_noble=True,
            noble_rank=32,
            noble_alpha=0.05,
            noble_phi_std=0.2,
        )
        layer = _build_linear(64, 128, config, bias=False)
        assert layer.rank == 32
        # Check noble_w_up init reflects alpha=0.05
        expected_std = 0.05 / math.sqrt(32)
        actual_std = layer.noble_w_up.weight.std().item()
        assert actual_std < expected_std * 3
