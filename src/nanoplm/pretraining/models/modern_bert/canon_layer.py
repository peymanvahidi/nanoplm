"""Depthwise separable 1D convolution ("Canon") layer for ModernBERT.

Provides ``ModernBertCanonLayer`` — a bidirectional depthwise conv1d residual
block with multiple forward paths: pure PyTorch (padded and varlen), fused
LayerNorm+Conv Triton, and fused RMSNorm+Conv Triton kernels.
"""

from __future__ import annotations

import weakref
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from ._globals import (
    USE_ACTIVATION_CHECKPOINTING_CANON,
    _canon_cute_ops,
    _canon_ops,
    _varlen_canon_conv,
    _varlen_ln_canon_conv,
    _varlen_rms_canon_conv,
)
from .config import ModernBertConfig


def _make_canon_layer(channels: int, config: ModernBertConfig) -> nn.Module:
    """Factory: returns a symmetric (bidirectional) Canon layer."""
    if config.canon_layers_kernel_size is None:
        raise ValueError(
            "canon_layers_kernel_size was not resolved in ModernBertConfig.__post_init__"
        )
    return ModernBertCanonLayer(channels, kernel_size=config.canon_layers_kernel_size)


def _canon_accum_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return x.dtype


# seq_id is identical for every canon site in a micro-batch (all layers get
# the same cu_seqlens tensor object), but used to be re-derived per call --
# 2-3 kernel launches x num_layers x sites per forward, a real cost on
# power-capped CPUs. Cache it per cu_seqlens *object identity*; the
# weakref.finalize evicts the entry when the tensor is collected, before its
# id() can be reused. This is safe where (data_ptr, _version) keying is not:
# allocator address reuse creates a new python object with a fresh id only
# after the old entry is gone. (Not a WeakKeyDictionary: weakref.ref.__eq__
# compares referents with `==`, which is elementwise for tensors.)
_seq_id_cache: dict = {}


def _derive_seq_id(cu_seqlens: torch.Tensor, T: int) -> torch.Tensor:
    """Per-token sequence index via binary search over cu_seqlens.

    Deliberately NOT the cumsum-of-resets over position_ids: inductor fuses
    that cumsum into a split-scan (decoupled-lookback) Triton kernel that
    intermittently triggers illegal memory accesses in training
    (torch 2.12 + max-autotune, sm90).
    """
    if torch.compiler.is_compiling():
        # No caching under torch.compile: dynamo guards on the cache/weakref
        # internals (tracing weakref.finalize even guards on the length of
        # its class-level _registry, which the call itself mutates -> hard
        # "guard failed on the same frame" error). Inside a compiled graph
        # the two ops are fused and launch-free anyway.
        positions = torch.arange(
            T, device=cu_seqlens.device, dtype=cu_seqlens.dtype
        )
        return torch.searchsorted(
            cu_seqlens[1:], positions, right=True, out_int32=True
        )
    key = id(cu_seqlens)
    cached = _seq_id_cache.get(key)
    if cached is not None and cached.shape[0] == T:
        return cached
    positions = torch.arange(T, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    seq_id = torch.searchsorted(
        cu_seqlens[1:], positions, right=True, out_int32=True
    )
    if key not in _seq_id_cache:
        weakref.finalize(cu_seqlens, _seq_id_cache.pop, key, None)
    _seq_id_cache[key] = seq_id
    return seq_id


class ModernBertCanonLayer(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self.radius = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.radius,
            groups=channels,
            bias=True,
        )

    def _forward_varlen(self, x, cu_seqlens, position_ids=None):
        T, C = x.shape
        n_seqs = cu_seqlens.shape[0] - 1

        if n_seqs <= 1:
            acc_dtype = _canon_accum_dtype(x)
            x_acc = x.to(dtype=acc_dtype)
            mixed = (
                F.conv1d(
                    x_acc.T.unsqueeze(0),
                    self.conv.weight.to(dtype=acc_dtype),
                    bias=(
                        self.conv.bias.to(dtype=acc_dtype)
                        if self.conv.bias is not None
                        else None
                    ),
                    stride=1,
                    padding=self.radius,
                    groups=self.conv.groups,
                )
                .squeeze(0)
                .T.to(dtype=x.dtype)
            )
            return x + mixed

        seq_id = _derive_seq_id(cu_seqlens, T)
        weight = self.conv.weight[:, 0, :]
        bias = self.conv.bias
        # CuTeDSL backend (sm90, bf16, C in 512-3072, K=7); returns x + conv(x).
        if _canon_cute_ops is not None and _canon_cute_ops.is_compatible(
            x, self.radius
        ):
            return _canon_cute_ops.varlen_canon_conv_residual(
                x, seq_id, weight, bias, self.radius
            )
        mixed = _varlen_canon_conv(x, seq_id, weight, bias, self.radius)
        return x + mixed

    def _forward_padded(self, x, attention_mask=None):
        acc_dtype = _canon_accum_dtype(x)
        x_acc = x.to(dtype=acc_dtype)
        token_mask = None
        if attention_mask is not None:
            # Treat any nonzero as "valid token". This avoids accidental
            # NaNs if a caller passes an additive mask (e.g. 0 / -inf).
            token_mask = attention_mask.ne(0).unsqueeze(-1).to(dtype=acc_dtype)
            x_acc = x_acc * token_mask
        mixed = F.conv1d(
            x_acc.transpose(1, 2),
            self.conv.weight.to(dtype=acc_dtype),
            bias=(
                self.conv.bias.to(dtype=acc_dtype)
                if self.conv.bias is not None
                else None
            ),
            stride=1,
            padding=self.radius,
            groups=self.conv.groups,
        ).transpose(1, 2)
        out = x_acc + mixed
        if token_mask is not None:
            out = out * token_mask
        return out.to(dtype=x.dtype)

    def _forward_varlen_fused_ln(
        self, x, cu_seqlens, ln_weight, ln_eps, position_ids=None
    ):
        """Fused LN + Canon varlen path: computes LN(x) + conv(LN(x)) without
        materializing the intermediate LN output."""
        T, C = x.shape
        n_seqs = cu_seqlens.shape[0] - 1

        # Single-sequence: fall back to separate LN + F.conv1d (no Triton path)
        if n_seqs <= 1:
            acc_dtype = _canon_accum_dtype(x)
            # Compute LN manually to get normalized x
            x_fp32 = x.float()
            mean = x_fp32.mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(x_fp32.var(dim=-1, keepdim=True, correction=0) + ln_eps)
            ln_out = ((x_fp32 - mean) * rstd * ln_weight.float()).to(dtype=acc_dtype)
            mixed = (
                F.conv1d(
                    ln_out.T.unsqueeze(0),
                    self.conv.weight.to(dtype=acc_dtype),
                    bias=(
                        self.conv.bias.to(dtype=acc_dtype)
                        if self.conv.bias is not None
                        else None
                    ),
                    stride=1,
                    padding=self.radius,
                    groups=self.conv.groups,
                )
                .squeeze(0)
                .T.to(dtype=x.dtype)
            )
            return (ln_out + mixed).to(dtype=x.dtype)

        seq_id = _derive_seq_id(cu_seqlens, T)

        weight = self.conv.weight[:, 0, :]
        bias = self.conv.bias
        return _varlen_ln_canon_conv(
            x, seq_id, ln_weight, ln_eps, weight, bias, self.radius
        )

    def _forward_varlen_fused_rms(
        self, x, cu_seqlens, rms_weight, rms_eps, position_ids=None
    ):
        """Fused RMS + Canon varlen path: computes RMS(x) + conv(RMS(x)) without
        materializing the intermediate RMS output."""
        T, C = x.shape
        n_seqs = cu_seqlens.shape[0] - 1

        # Single-sequence: fall back to separate RMS + F.conv1d (no Triton path)
        if n_seqs <= 1:
            acc_dtype = _canon_accum_dtype(x)
            # Compute RMS manually to get normalized x
            x_fp32 = x.float()
            inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
            rms_out = (x_fp32 * inv_rms * rms_weight.float()).to(dtype=acc_dtype)
            mixed = (
                F.conv1d(
                    rms_out.T.unsqueeze(0),
                    self.conv.weight.to(dtype=acc_dtype),
                    bias=(
                        self.conv.bias.to(dtype=acc_dtype)
                        if self.conv.bias is not None
                        else None
                    ),
                    stride=1,
                    padding=self.radius,
                    groups=self.conv.groups,
                )
                .squeeze(0)
                .T.to(dtype=x.dtype)
            )
            return (rms_out + mixed).to(dtype=x.dtype)

        seq_id = _derive_seq_id(cu_seqlens, T)

        weight = self.conv.weight[:, 0, :]
        bias = self.conv.bias
        # CuTeDSL backend (sm90, bf16, C in 512-3072, K=7) — ~2-4x the Triton kernels.
        if _canon_cute_ops is not None and _canon_cute_ops.is_compatible(
            x, self.radius
        ):
            return _canon_cute_ops.varlen_rms_canon_conv(
                x, seq_id, rms_weight, rms_eps, weight, bias, self.radius
            )
        return _varlen_rms_canon_conv(
            x, seq_id, rms_weight, rms_eps, weight, bias, self.radius
        )

    def forward(
        self,
        x,
        position_ids=None,
        cu_seqlens=None,
        attention_mask=None,
        fuse_ln_weight=None,
        fuse_ln_eps=None,
        fuse_norm_type="layernorm",
    ):
        # Checkpoint at the module boundary so all Canon insertion sites (A/B/C/D)
        # and both varlen/padded paths are covered by the same flag.
        use_ckpt = (
            USE_ACTIVATION_CHECKPOINTING_CANON
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
        )
        # Eval/inference should avoid autotune warmup latency.
        # _canon_ops is None on CPU/MPS (no CUDA) — skip the Triton autotune guard.
        autotune_ctx = (
            nullcontext()
            if self.training or _canon_ops is None
            else _canon_ops.disable_autotune_temporarily()
        )

        with autotune_ctx:
            return self._forward_dispatch(
                x,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                fuse_ln_weight=fuse_ln_weight,
                fuse_ln_eps=fuse_ln_eps,
                fuse_norm_type=fuse_norm_type,
                use_ckpt=use_ckpt,
            )

    def _forward_dispatch(
        self,
        x,
        *,
        position_ids,
        cu_seqlens,
        attention_mask,
        fuse_ln_weight,
        fuse_ln_eps,
        fuse_norm_type,
        use_ckpt,
    ):
        # Fused Norm+Conv path (varlen multi-sequence only)
        if fuse_ln_weight is not None and cu_seqlens is not None:
            if fuse_norm_type == "rmsnorm":
                # RMSNorm fusion path
                if use_ckpt:
                    if position_ids is None:
                        return _checkpoint(
                            lambda x_, cu_: self._forward_varlen_fused_rms(
                                x_,
                                cu_seqlens=cu_,
                                rms_weight=fuse_ln_weight,
                                rms_eps=fuse_ln_eps,
                                position_ids=None,
                            ),
                            x,
                            cu_seqlens,
                            use_reentrant=False,
                        )
                    return _checkpoint(
                        lambda x_, cu_, pos_: self._forward_varlen_fused_rms(
                            x_,
                            cu_seqlens=cu_,
                            rms_weight=fuse_ln_weight,
                            rms_eps=fuse_ln_eps,
                            position_ids=pos_,
                        ),
                        x,
                        cu_seqlens,
                        position_ids,
                        use_reentrant=False,
                    )
                return self._forward_varlen_fused_rms(
                    x,
                    cu_seqlens=cu_seqlens,
                    rms_weight=fuse_ln_weight,
                    rms_eps=fuse_ln_eps,
                    position_ids=position_ids,
                )
            else:
                # LayerNorm fusion path
                if use_ckpt:
                    if position_ids is None:
                        return _checkpoint(
                            lambda x_, cu_: self._forward_varlen_fused_ln(
                                x_,
                                cu_seqlens=cu_,
                                ln_weight=fuse_ln_weight,
                                ln_eps=fuse_ln_eps,
                                position_ids=None,
                            ),
                            x,
                            cu_seqlens,
                            use_reentrant=False,
                        )
                    return _checkpoint(
                        lambda x_, cu_, pos_: self._forward_varlen_fused_ln(
                            x_,
                            cu_seqlens=cu_,
                            ln_weight=fuse_ln_weight,
                            ln_eps=fuse_ln_eps,
                            position_ids=pos_,
                        ),
                        x,
                        cu_seqlens,
                        position_ids,
                        use_reentrant=False,
                    )
                return self._forward_varlen_fused_ln(
                    x,
                    cu_seqlens=cu_seqlens,
                    ln_weight=fuse_ln_weight,
                    ln_eps=fuse_ln_eps,
                    position_ids=position_ids,
                )

        if cu_seqlens is not None:
            if use_ckpt:
                if position_ids is None:
                    return _checkpoint(
                        lambda x_, cu_: self._forward_varlen(
                            x_, cu_seqlens=cu_, position_ids=None
                        ),
                        x,
                        cu_seqlens,
                        use_reentrant=False,
                    )
                return _checkpoint(
                    lambda x_, cu_, pos_: self._forward_varlen(
                        x_, cu_seqlens=cu_, position_ids=pos_
                    ),
                    x,
                    cu_seqlens,
                    position_ids,
                    use_reentrant=False,
                )
            return self._forward_varlen(
                x, cu_seqlens=cu_seqlens, position_ids=position_ids
            )
        if x.dim() == 3:
            if use_ckpt:
                if attention_mask is None:
                    return _checkpoint(
                        lambda x_: self._forward_padded(x_, attention_mask=None),
                        x,
                        use_reentrant=False,
                    )
                return _checkpoint(
                    lambda x_, mask_: self._forward_padded(x_, attention_mask=mask_),
                    x,
                    attention_mask,
                    use_reentrant=False,
                )
            return self._forward_padded(x, attention_mask=attention_mask)
        raise ValueError(f"Expected padded input [B, S, C], got shape={tuple(x.shape)}")


def _varlen_canon_inner(x, seq_id, weight, bias, radius):
    """Depthwise conv with boundary masking for varlen Canon mixing."""
    acc_dtype = _canon_accum_dtype(x)
    x_acc = x.to(dtype=acc_dtype)
    weight_acc = weight.to(dtype=acc_dtype)
    bias_acc = bias.to(dtype=acc_dtype) if bias is not None else None
    out = torch.zeros_like(x_acc)
    for k, offset in enumerate(range(-radius, radius + 1)):
        rolled_x = torch.roll(x_acc, shifts=-offset, dims=0)
        rolled_id = torch.roll(seq_id, shifts=-offset, dims=0)
        # IMPORTANT: avoid `0 * NaN = NaN` propagation across sequence boundaries.
        valid = (rolled_id == seq_id).unsqueeze(-1)
        rolled_x = rolled_x.masked_fill(~valid, 0)
        out = out + rolled_x * weight_acc[:, k]
    if bias_acc is not None:
        out = out + bias_acc
    return out.to(dtype=x.dtype)
