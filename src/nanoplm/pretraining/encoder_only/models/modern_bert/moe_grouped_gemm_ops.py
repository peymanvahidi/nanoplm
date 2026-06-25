from __future__ import annotations

import torch
import torch.library

from .cutlass_grouped_gemm import grouped_gemm as cutlass_grouped_gemm


_lib = torch.library.Library("nanoplm_moe", "FRAGMENT")
_lib.define(
    "grouped_gemm(Tensor a, Tensor b, Tensor batch_sizes, bool trans_a=False, bool trans_b=False) -> Tensor"
)


def _validate_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> None:
    if trans_a and trans_b:
        raise ValueError("nanoplm_moe::grouped_gemm does not support trans_a and trans_b together")
    if a.ndim != 2:
        raise ValueError(f"expected a 2D lhs tensor, got {tuple(a.shape)}")
    if batch_sizes.ndim != 1:
        raise ValueError(f"expected a 1D batch_sizes tensor, got {tuple(batch_sizes.shape)}")
    if batch_sizes.dtype != torch.int64:
        raise ValueError(f"expected int64 batch_sizes, got {batch_sizes.dtype}")
    if trans_a:
        if b.ndim != 2:
            raise ValueError(f"expected a 2D rhs tensor when trans_a=True, got {tuple(b.shape)}")
    else:
        if b.ndim != 3:
            raise ValueError(f"expected a 3D rhs tensor when trans_a=False, got {tuple(b.shape)}")


def _output_shape(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> tuple[int, ...]:
    _validate_inputs(a, b, batch_sizes, trans_a, trans_b)
    if trans_a:
        return (batch_sizes.shape[0], a.shape[1], b.shape[1])
    if trans_b:
        return (a.shape[0], b.shape[1])
    return (a.shape[0], b.shape[2])


def _reference_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    _validate_inputs(a, b, batch_sizes, trans_a, trans_b)
    sizes = [int(x) for x in batch_sizes.detach().to(device="cpu").tolist()]

    if trans_a:
        out = torch.empty(
            (len(sizes), a.shape[1], b.shape[1]),
            device=a.device,
            dtype=a.dtype,
        )
        offset = 0
        for idx, rows in enumerate(sizes):
            a_slice = a[offset : offset + rows]
            b_slice = b[offset : offset + rows]
            if rows == 0:
                out[idx].zero_()
            else:
                out[idx] = a_slice.transpose(0, 1) @ b_slice
            offset += rows
        return out

    out_cols = b.shape[1] if trans_b else b.shape[2]
    out = torch.empty((a.shape[0], out_cols), device=a.device, dtype=a.dtype)
    offset = 0
    for idx, rows in enumerate(sizes):
        a_slice = a[offset : offset + rows]
        b_slice = b[idx].transpose(0, 1) if trans_b else b[idx]
        out[offset : offset + rows] = a_slice @ b_slice
        offset += rows
    return out


@torch.library.register_fake("nanoplm_moe::grouped_gemm")
def _grouped_gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
):
    return torch.empty(
        _output_shape(a, b, batch_sizes, trans_a, trans_b),
        device=a.device,
        dtype=a.dtype,
    )


@torch.library.impl(_lib, "grouped_gemm", "CUDA")
def _grouped_gemm_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
):
    _validate_inputs(a, b, batch_sizes, trans_a, trans_b)
    if a.dtype not in {torch.float16, torch.bfloat16}:
        return _reference_grouped_gemm(a, b, batch_sizes, trans_a, trans_b)
    return cutlass_grouped_gemm(a, b, batch_sizes, trans_a=trans_a, trans_b=trans_b)


@torch.library.impl(_lib, "grouped_gemm", "CPU")
def _grouped_gemm_cpu(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
):
    return _reference_grouped_gemm(a, b, batch_sizes, trans_a, trans_b)


def _setup_context(ctx, inputs, output):
    a, b, batch_sizes, trans_a, trans_b = inputs
    ctx.save_for_backward(a, b, batch_sizes)
    ctx.trans_a = bool(trans_a)
    ctx.trans_b = bool(trans_b)


def _backward(ctx, grad_out: torch.Tensor):
    grad_out = grad_out.contiguous()
    a, b, batch_sizes = ctx.saved_tensors
    trans_a = ctx.trans_a
    trans_b = ctx.trans_b

    grad_a = grad_b = None
    if trans_a:
        if ctx.needs_input_grad[0]:
            grad_a = torch.ops.nanoplm_moe.grouped_gemm(
                b,
                grad_out,
                batch_sizes,
                False,
                True,
            )
        if ctx.needs_input_grad[1]:
            grad_b = torch.ops.nanoplm_moe.grouped_gemm(
                a,
                grad_out,
                batch_sizes,
                False,
                False,
            )
    else:
        if ctx.needs_input_grad[0]:
            grad_a = torch.ops.nanoplm_moe.grouped_gemm(
                grad_out,
                b,
                batch_sizes,
                False,
                not trans_b,
            )
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad_out, a) if trans_b else (a, grad_out)
            grad_b = torch.ops.nanoplm_moe.grouped_gemm(
                lhs,
                rhs,
                batch_sizes,
                True,
                False,
            )
    return grad_a, grad_b, None, None, None


torch.library.register_autograd(
    "nanoplm_moe::grouped_gemm",
    _backward,
    setup_context=_setup_context,
)


def moe_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    return torch.ops.nanoplm_moe.grouped_gemm(a, b, batch_sizes, trans_a, trans_b)
