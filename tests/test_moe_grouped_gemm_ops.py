from __future__ import annotations

import pytest
import torch

from nanoplm.pretraining.models.modern_bert.moe_grouped_gemm_ops import (
    _reference_grouped_gemm,
    moe_grouped_gemm,
)


def _make_counts(device: torch.device) -> torch.Tensor:
    return torch.tensor([0, 2, 3], device=device, dtype=torch.int64)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
@pytest.mark.parametrize(
    ("trans_a", "trans_b", "a_shape", "b_shape"),
    [
        (False, False, (5, 8), (3, 8, 16)),
        (False, True, (5, 16), (3, 8, 16)),
        (True, False, (5, 8), (5, 16)),
    ],
)
def test_grouped_gemm_matches_reference(device, trans_a, trans_b, a_shape, b_shape):
    torch_device = torch.device(device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    counts = _make_counts(torch_device)
    a = torch.randn(a_shape, device=torch_device, dtype=dtype)
    b = torch.randn(b_shape, device=torch_device, dtype=dtype)

    out = moe_grouped_gemm(a, b, counts, trans_a=trans_a, trans_b=trans_b)
    ref = _reference_grouped_gemm(a, b, counts, trans_a, trans_b)

    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("trans_a", "trans_b", "a_shape", "b_shape"),
    [
        (False, False, (5, 8), (3, 8, 16)),
        (False, True, (5, 16), (3, 8, 16)),
        (True, False, (5, 8), (5, 16)),
    ],
)
def test_grouped_gemm_backward_matches_reference(trans_a, trans_b, a_shape, b_shape):
    counts = _make_counts(torch.device("cuda"))

    a = torch.randn(a_shape, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(b_shape, device="cuda", dtype=torch.bfloat16)

    a_cutlass = a.clone().detach().requires_grad_(True)
    b_cutlass = b.clone().detach().requires_grad_(True)
    y_cutlass = moe_grouped_gemm(
        a_cutlass, b_cutlass, counts, trans_a=trans_a, trans_b=trans_b
    )
    y_cutlass.float().square().mean().backward()

    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    y_ref = _reference_grouped_gemm(a_ref, b_ref, counts, trans_a, trans_b)
    y_ref.float().square().mean().backward()

    assert torch.allclose(y_cutlass.float(), y_ref.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(
        a_cutlass.grad.float(), a_ref.grad.float(), atol=2e-2, rtol=2e-2
    )
    assert torch.allclose(
        b_cutlass.grad.float(), b_ref.grad.float(), atol=2e-2, rtol=2e-2
    )
