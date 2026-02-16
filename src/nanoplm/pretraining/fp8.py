"""Minimal FP8 Linear training utilities (ported from nanochat).

This module replaces eligible ``nn.Linear`` layers with ``Float8Linear``.
Weights stay in their original dtype; only matmuls run via ``torch._scaled_mm``.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

EPS = 1e-12
_ALIGN = 16


@torch.no_grad()
def _to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 with tensorwise dynamic scaling."""
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().max()
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    x_fp8 = (x.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale.reciprocal()


def _to_col_major(x: torch.Tensor) -> torch.Tensor:
    """Return a 2D tensor with column-major memory layout for _scaled_mm arg B."""
    return x.t().contiguous().t()


def _pad16(x: torch.Tensor) -> torch.Tensor:
    """Zero-pad a 2D tensor so both dims are multiples of 16.

    Padding is done in the original dtype *before* FP8 conversion so that
    ``torch.nn.functional.pad`` always operates on a supported type.
    """
    m, n = x.shape
    pm = (-m) % _ALIGN
    pn = (-n) % _ALIGN
    if pm == 0 and pn == 0:
        return x
    return torch.nn.functional.pad(x, (0, pn, 0, pm))


@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """FP8 matmul autograd for Linear forward/backward GEMMs.

    All operands are padded to multiples of 16 before ``_scaled_mm`` because
    that kernel requires 16-byte aligned dimensions.  The padding is applied
    *before* FP8 quantisation (zero-pad in the source dtype) so that
    ``F.pad`` never sees an FP8 tensor.
    """

    @staticmethod
    def forward(ctx, input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_2d, weight)
        M, K = input_2d.shape
        N = weight.shape[0]

        inp_p = _pad16(input_2d)
        w_p = _pad16(weight)

        input_fp8, input_inv = _to_fp8(inp_p, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(w_p, torch.float8_e4m3fn)

        out = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            use_fast_accum=True,
        )
        return out[:M, :N]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_2d, weight = ctx.saved_tensors
        M, K = input_2d.shape
        N = weight.shape[0]

        # --- grad_input = grad_output @ weight ---
        go_p = _pad16(grad_output)
        w_p = _pad16(weight)
        go_fp8, go_inv = _to_fp8(go_p, torch.float8_e5m2)
        w_fp8, w_inv = _to_fp8(w_p, torch.float8_e4m3fn)
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )[:M, :K]

        # --- grad_weight = grad_output^T @ input_2d ---
        go_p2 = _pad16(grad_output)
        in_p = _pad16(input_2d)
        go_fp8_2, go_inv_2 = _to_fp8(go_p2, torch.float8_e5m2)
        in_fp8, in_inv = _to_fp8(in_p, torch.float8_e4m3fn)
        go_t = go_fp8_2.t().contiguous()
        in_col = _to_col_major(in_fp8)
        grad_weight = torch._scaled_mm(
            go_t,
            in_col,
            scale_a=go_inv_2,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )[:N, :K]

        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """Drop-in ``nn.Linear`` replacement using FP8 matmuls."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            input = input.to(torch.get_autocast_gpu_dtype())

        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod: nn.Linear) -> "Float8Linear":
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    """API-compatible minimal config (tensorwise recipe only)."""

    @staticmethod
    def from_recipe_name(recipe_name: str) -> "Float8LinearConfig":
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'."
            )
        return Float8LinearConfig()


def convert_to_float8_training(
    module: nn.Module,
    *,
    config: Float8LinearConfig | None = None,
    module_filter_fn: Callable[[nn.Module, str], bool] | None = None,
) -> nn.Module:
    """Replace eligible ``nn.Linear`` modules with ``Float8Linear``."""
    if config is not None and not isinstance(config, Float8LinearConfig):
        raise TypeError("config must be Float8LinearConfig or None")

    def _convert(mod: nn.Module, prefix: str = "") -> None:
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module

