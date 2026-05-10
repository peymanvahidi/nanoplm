"""Tests for fused MoE scatter_dispatch and gather_combine ops."""
from __future__ import annotations

import pytest
import torch

from nanoplm.pretraining.models.modern_bert.moe_triton_ops import (
    build_inverse_map,
    moe_gather_combine,
    moe_scatter_dispatch,
)


def _make_test_data(T, top_k, C, num_experts, dtype, device):
    """Create realistic MoE dispatch metadata + tensors."""
    x = torch.randn(T, C, device=device, dtype=dtype)
    # Random expert assignments
    indices = torch.randint(0, num_experts, (T, top_k), device=device)
    expert_flat = indices.reshape(-1)
    sorted_idx = expert_flat.argsort(stable=True)

    token_idx = (sorted_idx // top_k).to(torch.int32)
    slot_idx = (sorted_idx % top_k).to(torch.int32)
    inv_map = build_inverse_map(token_idx, slot_idx, T, top_k)

    weights = torch.randn(T, top_k, device=device, dtype=torch.float32).softmax(dim=-1)
    return x, token_idx, slot_idx, inv_map, weights, sorted_idx


# ── scatter_dispatch tests ───────────────────────────────────────────────────


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available")),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scatter_dispatch_forward(device, dtype):
    T, top_k, C, E = 128, 2, 64, 8
    x, token_idx, *_ , sorted_idx = _make_test_data(T, top_k, C, E, dtype, device)

    # Fused
    out = moe_scatter_dispatch(x, token_idx)

    # Reference: repeat_interleave + permute via sorted_idx
    x_expanded = x.repeat_interleave(top_k, dim=0)
    ref = x_expanded[sorted_idx]

    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=0, rtol=0)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available")),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scatter_dispatch_backward(device, dtype):
    T, top_k, C, E = 128, 2, 64, 8
    x, token_idx, *_, sorted_idx = _make_test_data(T, top_k, C, E, dtype, device)

    # Fused path
    x1 = x.clone().detach().requires_grad_(True)
    out1 = moe_scatter_dispatch(x1, token_idx)
    out1.float().square().mean().backward()

    # Reference path
    x2 = x.clone().detach().requires_grad_(True)
    x_expanded = x2.repeat_interleave(top_k, dim=0)
    ref = x_expanded[sorted_idx]
    ref.float().square().mean().backward()

    assert torch.allclose(out1.float(), ref.float(), atol=0, rtol=0)
    assert torch.allclose(x1.grad.float(), x2.grad.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scatter_dispatch_gradcheck():
    T, top_k, C, E = 8, 2, 16, 4
    x, token_idx, *_ = _make_test_data(T, top_k, C, E, torch.float64, "cuda")
    x = x.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda inp: moe_scatter_dispatch(inp, token_idx),
        (x,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


# ── gather_combine tests ────────────────────────────────────────────────────


def _gather_combine_reference(expert_out_sorted, inv_map, weights, scale):
    """Eager reference that matches the original MoE combine semantics."""
    T, top_k = inv_map.shape
    C = expert_out_sorted.shape[1]
    flat_idx = inv_map.long().reshape(-1)
    rows = expert_out_sorted[flat_idx].view(T, top_k, C)
    if expert_out_sorted.dtype in (torch.bfloat16, torch.float16):
        rows = rows.float()
    return (rows * weights.unsqueeze(-1) * scale).sum(dim=1)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available")),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_combine_forward(device, dtype):
    T, top_k, C, E = 128, 2, 64, 8
    scale = 1.0
    x, token_idx, slot_idx, inv_map, weights, sorted_idx = _make_test_data(
        T, top_k, C, E, dtype, device,
    )
    expert_out_sorted = torch.randn(
        T * top_k, C, device=device, dtype=dtype,
    )

    out = moe_gather_combine(expert_out_sorted, inv_map, weights, scale)
    ref = _gather_combine_reference(expert_out_sorted, inv_map, weights, scale)

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available")),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_combine_backward(device, dtype):
    T, top_k, C, E = 128, 2, 64, 8
    scale = 1.5
    *_, inv_map, weights, _ = _make_test_data(T, top_k, C, E, dtype, device)
    eo = torch.randn(T * top_k, C, device=device, dtype=dtype)

    # Fused path
    eo1 = eo.clone().detach().requires_grad_(True)
    w1 = weights.clone().detach().requires_grad_(True)
    out1 = moe_gather_combine(eo1, inv_map, w1, scale)
    out1.square().mean().backward()

    # Reference path
    eo2 = eo.clone().detach().requires_grad_(True)
    w2 = weights.clone().detach().requires_grad_(True)
    ref = _gather_combine_reference(eo2, inv_map, w2, scale)
    ref.square().mean().backward()

    assert torch.allclose(out1, ref, atol=1e-4, rtol=1e-4)

    atol_grad = 1e-3 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(eo1.grad.float(), eo2.grad.float(), atol=atol_grad, rtol=1e-3)
    assert torch.allclose(w1.grad.float(), w2.grad.float(), atol=atol_grad, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gather_combine_gradcheck():
    T, top_k, C, E = 8, 2, 16, 4
    scale = 1.0
    *_, inv_map, weights, _ = _make_test_data(T, top_k, C, E, torch.float64, "cuda")
    eo = torch.randn(T * top_k, C, device="cuda", dtype=torch.float64, requires_grad=True)
    w = weights.double().requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda e, ww: moe_gather_combine(e, inv_map, ww, scale),
        (eo, w),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


# ── gather_combine output dtype contract ─────────────────────────────────────


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available")),
])
def test_gather_combine_output_dtype(device):
    """bf16 expert_out + fp32 weights → fp32 output (matches eager promotion)."""
    T, top_k, C, E = 32, 2, 64, 4
    *_, inv_map, weights, _ = _make_test_data(T, top_k, C, E, torch.bfloat16, device)
    eo = torch.randn(T * top_k, C, device=device, dtype=torch.bfloat16)
    out = moe_gather_combine(eo, inv_map, weights, 1.0)
    assert out.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gather_combine_large_hidden_dim_cuda():
    """Fused CUDA path should handle research-scale hidden dims without eager fallback."""
    T, top_k, C, E = 64, 2, 4096, 8
    scale = 1.25
    *_, inv_map, weights, _ = _make_test_data(T, top_k, C, E, torch.bfloat16, "cuda")
    eo = torch.randn(T * top_k, C, device="cuda", dtype=torch.bfloat16)

    eo1 = eo.clone().detach().requires_grad_(True)
    w1 = weights.clone().detach().requires_grad_(True)
    out1 = moe_gather_combine(eo1, inv_map, w1, scale)
    out1.square().mean().backward()

    eo2 = eo.clone().detach().requires_grad_(True)
    w2 = weights.clone().detach().requires_grad_(True)
    ref = _gather_combine_reference(eo2, inv_map, w2, scale)
    ref.square().mean().backward()

    assert out1.shape == ref.shape
    assert out1.dtype == ref.dtype == torch.float32
    assert torch.allclose(out1, ref, atol=2e-3, rtol=2e-3)
    assert torch.allclose(eo1.grad.float(), eo2.grad.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(w1.grad.float(), w2.grad.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gather_combine_non_power_of_two_topk_cuda():
    """Fused backward must support MoE top-k values like 3, not just powers of two."""
    T, top_k, C, E = 64, 3, 1024, 8
    scale = 1.0
    *_, inv_map, weights, _ = _make_test_data(T, top_k, C, E, torch.bfloat16, "cuda")
    eo = torch.randn(T * top_k, C, device="cuda", dtype=torch.bfloat16)

    eo1 = eo.clone().detach().requires_grad_(True)
    w1 = weights.clone().detach().requires_grad_(True)
    out1 = moe_gather_combine(eo1, inv_map, w1, scale)
    out1.square().mean().backward()

    eo2 = eo.clone().detach().requires_grad_(True)
    w2 = weights.clone().detach().requires_grad_(True)
    ref = _gather_combine_reference(eo2, inv_map, w2, scale)
    ref.square().mean().backward()

    assert torch.allclose(out1, ref, atol=2e-3, rtol=2e-3)
    assert torch.allclose(eo1.grad.float(), eo2.grad.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(w1.grad.float(), w2.grad.float(), atol=2e-3, rtol=2e-3)


# ── build_inverse_map test ───────────────────────────────────────────────────


def test_build_inverse_map():
    T, top_k, E = 32, 2, 4
    device = "cpu"
    indices = torch.randint(0, E, (T, top_k), device=device)
    expert_flat = indices.reshape(-1)
    sorted_idx = expert_flat.argsort(stable=True)
    token_idx = (sorted_idx // top_k).to(torch.int32)
    slot_idx = (sorted_idx % top_k).to(torch.int32)

    inv_map = build_inverse_map(token_idx, slot_idx, T, top_k)

    # Verify: inv_map[token_idx[i], slot_idx[i]] == i for all i
    for i in range(T * top_k):
        t = token_idx[i].item()
        k = slot_idx[i].item()
        assert inv_map[t, k].item() == i


# ── Integration test ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_moe_layer_integration():
    """Full MoE layer forward+backward matches eager reference."""
    from nanoplm.pretraining.models.modern_bert.moe_grouped_gemm_ops import (
        moe_grouped_gemm,
    )

    T, C, E, top_k, inter = 256, 64, 4, 2, 32
    scale = 1.0

    # Shared inputs
    x = torch.randn(T, C, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, E, (T, top_k), device="cuda")
    weights = torch.randn(T, top_k, device="cuda", dtype=torch.float32).softmax(dim=-1)
    expert_flat = indices.reshape(-1)
    sorted_idx = expert_flat.argsort(stable=True)
    counts = expert_flat.bincount(minlength=E).to(torch.int64)

    Wi = torch.randn(E, C, 2 * inter, device="cuda", dtype=torch.bfloat16)
    Wo = torch.randn(E, inter, C, device="cuda", dtype=torch.bfloat16)

    # --- Fused path ---
    token_idx = (sorted_idx // top_k).to(torch.int32)
    slot_idx = (sorted_idx % top_k).to(torch.int32)
    inv_map = build_inverse_map(token_idx, slot_idx, T, top_k)

    x1 = x.clone().detach().requires_grad_(True)
    w1 = weights.clone().detach().requires_grad_(True)
    x_sorted1 = moe_scatter_dispatch(x1, token_idx)
    wi1 = moe_grouped_gemm(x_sorted1, Wi, counts)
    proj1, gate1 = wi1.chunk(2, dim=-1)
    act1 = torch.nn.functional.silu(gate1) * proj1
    eo_sorted1 = moe_grouped_gemm(act1, Wo, counts)
    out1 = moe_gather_combine(eo_sorted1, inv_map, w1, scale)
    out1.float().square().mean().backward()

    # --- Eager reference path (pure PyTorch, no custom ops) ---
    x2 = x.clone().detach().requires_grad_(True)
    w2 = weights.clone().detach().requires_grad_(True)
    x_expanded = x2.repeat_interleave(top_k, dim=0)
    x_sorted2 = x_expanded[sorted_idx]
    wi2 = moe_grouped_gemm(x_sorted2, Wi, counts)
    proj2, gate2 = wi2.chunk(2, dim=-1)
    act2 = torch.nn.functional.silu(gate2) * proj2
    eo_sorted2 = moe_grouped_gemm(act2, Wo, counts)
    # Unsort: place each sorted row back at its original position
    expert_out2 = torch.empty_like(eo_sorted2)
    expert_out2[sorted_idx] = eo_sorted2
    expert_out2 = expert_out2.view(T, top_k, C)
    out2 = (expert_out2 * w2.unsqueeze(-1) * scale).sum(dim=1)
    out2.float().square().mean().backward()

    assert torch.allclose(out1, out2, atol=1e-3, rtol=1e-3)
    assert torch.allclose(x1.grad.float(), x2.grad.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(w1.grad.float(), w2.grad.float(), atol=1e-2, rtol=1e-2)
