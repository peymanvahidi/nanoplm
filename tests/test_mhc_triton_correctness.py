"""Correctness tests for mHC-lite Triton kernels vs paper-spec reference.

Reference implementations follow the precision specification from the
DeepSeek mHC paper (Eq 1-10, Section 4.1 Kernel Fusion):
  - x input: bfloat16
  - Weight (phi): bfloat16 (stored), tensor-core matmul
  - Matmul output (x @ phi): float32 accumulation
  - RMS norm factor: float32
  - Division by norm, alpha scaling, bias: float32
  - Sigmoid/softmax: float32
  - Key optimization: matmul FIRST, then divide by RMS norm (Eq 5 before Eq 6-7)

The mHC-lite paper inherits this precision spec, changing only
Sinkhorn-Knopp -> BvN convex combination for H^res.

Tests individual kernels (K1, K3, K4) forward + backward, and the full
MHCLiteBlock end-to-end (triton path vs pytorch path).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# Tolerances for bf16 arithmetic.
#
# bf16 has 7 mantissa bits, so the ULP at magnitude M is M * 2^-7 ≈ M * 0.0078.
# Kernel vs reference may differ by up to ~2 ULP due to different rounding
# points (tensor-core tiling boundaries, intermediate f32 vs bf16 truncation).
# We use rtol=0.016 (≈2 ULP) with a small atol for near-zero values.
#
# For K1 (rmsnorm_project): output magnitudes up to ~64, so max_abs ≈ 0.5
# is expected bf16 rounding noise at that magnitude.
# For K4 (post_res): output magnitudes up to ~16, so max_abs ≈ 0.125.
FWD_ATOL = 1e-2
FWD_RTOL = 1.6e-2   # ~2 bf16 ULPs
BWD_ATOL = 5e-2
BWD_RTOL = 2e-2


# ── helpers ──────────────────────────────────────────────────────────────────

def _import_kernels():
    from nanoplm.pretraining.models.modern_bert import mhc_triton_kernels as k
    # trigger op registration
    from nanoplm.pretraining.models.modern_bert import mhc_triton_ops  # noqa: F401
    return k


def _rand(shape, *, dtype=torch.bfloat16, device="cuda", requires_grad=False):
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def _assert_close(name, actual, expected, atol, rtol):
    if actual.dtype != expected.dtype:
        actual = actual.to(expected.dtype)
    diff = (actual.float() - expected.float()).abs()
    ref = expected.float().abs().clamp(min=1e-8)
    max_abs = diff.max().item()
    max_rel = (diff / ref).max().item()
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    assert ok, (
        f"{name}: max_abs={max_abs:.4e} max_rel={max_rel:.4e} "
        f"(atol={atol}, rtol={rtol})"
    )


# ── K1: fused_rmsnorm_project ────────────────────────────────────────────────

class TestK1FusedRMSNormProject:
    """K1: proj = (x_bf16 @ W_bf16.T) * inv_rms  [paper Eq 5-7]

    Paper spec (DeepSeek mHC Eq 5-7):
      Eq 5: proj_raw = x_bf16 @ phi  (bf16 tensor-core matmul, f32 accum)
      Eq 6: r = ||x||_2 / sqrt(nC)   (f32)
      Eq 7: proj = (1/r) * proj_raw   (f32, then cast to bf16 for storage)
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.k = _import_kernels()
        self.T, self.n, self.C = 2048, 4, 256
        self.nC = self.n * self.C
        self.D_out = 2 * self.n + math.factorial(self.n)  # 32

    def _ref_fwd(self, x_flat, W):
        """Paper-spec reference: matmul first, then normalize.

        x_flat: (T, nC) bf16
        W: (D_out, nC) bf16

        Simulates tensor-core bf16 matmul (bf16 multiply, f32 accumulate)
        using PyTorch's native bf16 matmul on CUDA (same hardware path).
        """
        # Eq 5: bf16 matmul -> f32 accum (on CUDA this uses tensor cores)
        # We do x_bf16 @ W_bf16.T and get a bf16 result, then upcast.
        # But to truly match the Triton kernel which accumulates in f32,
        # we simulate: multiply in bf16 precision, accumulate in f32.
        # PyTorch's torch.matmul(bf16, bf16) on CUDA uses TF32/bf16 tensor cores
        # with f32 accumulation, matching the Triton kernel.
        proj_raw = torch.matmul(x_flat, W.T)  # bf16 @ bf16 -> bf16 (tensor core)

        # Eq 6: RMS in f32
        x_f32 = x_flat.float()
        rms = (x_f32 * x_f32).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()

        # Eq 7: scale in f32
        proj_out = proj_raw.float() * rms

        return proj_out.to(x_flat.dtype), rms.squeeze(-1)

    def test_forward(self):
        x = _rand((self.T, self.nC))
        W = _rand((self.D_out, self.nC))
        ref, _ = self._ref_fwd(x, W)
        out, inv_rms = torch.ops.nanoplm_mhc.fused_rmsnorm_project(x, W)
        _assert_close("K1_fwd", out, ref, FWD_ATOL, FWD_RTOL)

    def test_backward(self):
        x = _rand((self.T, self.nC), requires_grad=True)
        W = _rand((self.D_out, self.nC), requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        W_ref = W.detach().clone().requires_grad_(True)

        # Triton path
        out, _ = torch.ops.nanoplm_mhc.fused_rmsnorm_project(x, W)
        loss = out.float().sum()
        loss.backward()

        # Reference: use the same paper-spec forward for autograd
        ref, _ = self._ref_fwd(x_ref, W_ref)
        ref.float().sum().backward()

        _assert_close("K1_grad_x", x.grad, x_ref.grad, BWD_ATOL, BWD_RTOL)
        _assert_close("K1_grad_W", W.grad, W_ref.grad, BWD_ATOL, BWD_RTOL)


# ── K3: fused_pre_map ────────────────────────────────────────────────────────

class TestK3FusedPreMap:
    """K3: layer_input[t,c] = sum_j h_pre[t,j] * x[t,j,c]

    x is bf16, h_pre is f32. Kernel loads x as bf16->f32, multiplies by
    h_pre (f32), accumulates in f32, stores as bf16.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.k = _import_kernels()
        self.T, self.n, self.C = 2048, 4, 256

    def _ref_fwd(self, x_streams, h_pre):
        """h_pre: (T, n) f32, x_streams: (T, n, C) bf16 -> (T, C) bf16"""
        # Load x as bf16, upcast to f32 for multiply+accumulate
        result = (h_pre.unsqueeze(-1) * x_streams.float()).sum(dim=-2)
        return result.to(x_streams.dtype)

    def test_forward(self):
        x = _rand((self.T, self.n, self.C))
        h = torch.randn(self.T, self.n, device="cuda", dtype=torch.float32)
        ref = self._ref_fwd(x, h)
        out = torch.ops.nanoplm_mhc.fused_pre_map(x, h)
        _assert_close("K3_fwd", out, ref, FWD_ATOL, FWD_RTOL)

    def test_backward(self):
        x = _rand((self.T, self.n, self.C), requires_grad=True)
        h = torch.randn(self.T, self.n, device="cuda", dtype=torch.float32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        h_ref = h.detach().clone().requires_grad_(True)

        out = torch.ops.nanoplm_mhc.fused_pre_map(x, h)
        out.float().sum().backward()

        ref = self._ref_fwd(x_ref, h_ref)
        ref.float().sum().backward()

        _assert_close("K3_grad_x", x.grad, x_ref.grad, BWD_ATOL, BWD_RTOL)
        _assert_close("K3_grad_h", h.grad, h_ref.grad, BWD_ATOL, BWD_RTOL)


# ── K4: fused_post_res ───────────────────────────────────────────────────────

class TestK4FusedPostRes:
    """K4: out[t,i,c] = sum_j H[t,i,j]*x[t,j,c] + h_post[t,i]*lo[t,c]

    x, lo are bf16 (loaded to f32 in kernel). H, h_post are f32.
    Accumulation in f32, store as bf16.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.k = _import_kernels()
        self.T, self.n, self.C = 2048, 4, 256

    def _ref_fwd(self, x_streams, layer_output, H_merged, h_post):
        """Matches Triton kernel precision: load bf16->f32, compute f32, store bf16."""
        # H_merged: (T, n, n) f32, x_streams: (T, n, C) bf16
        # h_post: (T, n) f32, layer_output: (T, C) bf16
        out = torch.matmul(H_merged, x_streams.float())
        out += h_post.unsqueeze(-1) * layer_output.float().unsqueeze(-2)
        return out.to(x_streams.dtype)

    def test_forward(self):
        x = _rand((self.T, self.n, self.C))
        lo = _rand((self.T, self.C))
        H = torch.randn(self.T, self.n, self.n, device="cuda", dtype=torch.float32)
        hp = torch.randn(self.T, self.n, device="cuda", dtype=torch.float32)

        ref = self._ref_fwd(x, lo, H, hp)
        out = torch.ops.nanoplm_mhc.fused_post_res(x, lo, H, hp)
        _assert_close("K4_fwd", out, ref, FWD_ATOL, FWD_RTOL)

    def test_backward(self):
        x = _rand((self.T, self.n, self.C), requires_grad=True)
        lo = _rand((self.T, self.C), requires_grad=True)
        H = torch.randn(self.T, self.n, self.n, device="cuda", dtype=torch.float32, requires_grad=True)
        hp = torch.randn(self.T, self.n, device="cuda", dtype=torch.float32, requires_grad=True)

        x_ref = x.detach().clone().requires_grad_(True)
        lo_ref = lo.detach().clone().requires_grad_(True)
        H_ref = H.detach().clone().requires_grad_(True)
        hp_ref = hp.detach().clone().requires_grad_(True)

        out = torch.ops.nanoplm_mhc.fused_post_res(x, lo, H, hp)
        out.float().sum().backward()

        ref = self._ref_fwd(x_ref, lo_ref, H_ref, hp_ref)
        ref.float().sum().backward()

        _assert_close("K4_grad_x", x.grad, x_ref.grad, BWD_ATOL, BWD_RTOL)
        _assert_close("K4_grad_lo", lo.grad, lo_ref.grad, BWD_ATOL, BWD_RTOL)
        _assert_close("K4_grad_H", H.grad, H_ref.grad, BWD_ATOL, BWD_RTOL)
        _assert_close("K4_grad_hp", hp.grad, hp_ref.grad, BWD_ATOL, BWD_RTOL)


# ── Full MHCLiteBlock: triton vs pytorch ─────────────────────────────────────

class TestMHCLiteBlockEndToEnd:
    """Compare _forward_triton vs _forward_pytorch on the same block.

    Note: _forward_pytorch does normalize-then-project (standard RMSNorm),
    while _forward_triton uses K1 which does project-then-normalize (paper
    Eq 5-7 optimization). These are mathematically equivalent but
    numerically different in bf16.

    The E2E test validates that both paths produce results within acceptable
    bf16 tolerance of each other.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        _import_kernels()
        from nanoplm.pretraining.models.modern_bert.modeling import MHCLiteBlock

        self.T, self.n, self.C = 1024, 4, 256
        self.layer = _IdentityLinear(self.C).cuda().bfloat16()
        self.block = MHCLiteBlock(
            n_streams=self.n,
            hidden_size=self.C,
            layer=self.layer,
            triton_fused=True,
        ).cuda().bfloat16()

    def test_forward_match(self):
        x = _rand((self.T, self.n, self.C))
        with torch.no_grad():
            ref = self.block._forward_pytorch(x.clone())
            tri = self.block._forward_triton(x.clone())
        # Wider tolerance for E2E because of K1 computation-order difference
        # (normalize-then-project vs project-then-normalize).
        _assert_close("E2E_fwd", tri, ref, atol=5e-2, rtol=2e-2)

    def test_backward_match(self):
        x_tri = _rand((self.T, self.n, self.C), requires_grad=True)
        x_ref = x_tri.detach().clone().requires_grad_(True)

        out_tri = self.block._forward_triton(x_tri)
        out_tri.float().sum().backward()

        out_ref = self.block._forward_pytorch(x_ref)
        out_ref.float().sum().backward()

        _assert_close("E2E_grad_x", x_tri.grad, x_ref.grad, atol=1e-1, rtol=5e-2)

    def test_param_grads_match(self):
        """Check that parameter gradients (W_all, alpha_*) agree."""
        x = _rand((self.T, self.n, self.C))

        # Triton path
        self.block.zero_grad()
        out_tri = self.block._forward_triton(x.clone().requires_grad_(True))
        out_tri.float().sum().backward()
        grads_tri = {n: p.grad.clone() for n, p in self.block.named_parameters() if p.grad is not None}

        # PyTorch path
        self.block.zero_grad()
        out_ref = self.block._forward_pytorch(x.clone().requires_grad_(True))
        out_ref.float().sum().backward()
        grads_ref = {n: p.grad.clone() for n, p in self.block.named_parameters() if p.grad is not None}

        for name in grads_ref:
            assert name in grads_tri, f"Missing triton grad for {name}"
            _assert_close(f"E2E_param_{name}", grads_tri[name], grads_ref[name],
                          atol=1e-1, rtol=5e-2)


class _IdentityLinear(nn.Module):
    """Minimal stand-in for a transformer layer: linear + gelu.
    Deterministic, no attention, works for both paths identically."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        return F.gelu(self.proj(x))
