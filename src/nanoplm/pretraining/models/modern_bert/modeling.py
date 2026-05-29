"""Pure PyTorch ModernBERT for masked language modeling.

The model is intentionally small and readable:
- pre-norm transformer blocks
- RoPE attention (full + sliding-window layers)
- GLU MLP (or SwiGLU replacement)
- explicit, centralized initialization
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from ._globals import (  # noqa: E402 — flags must load after _globals side-effects
    _HAS_FLASH_VARLEN,
    _flash_varlen_fn,
    _FLASH_HAS_DROPOUT,
    USE_ACTIVATION_CHECKPOINTING_CANON,
    USE_ACTIVATION_CHECKPOINTING_MHC,
    USE_TRITON_SRELU,
    _mhc_triton_ops,
    _canon_ops,
    _varlen_canon_conv,
    _varlen_ln_canon_conv,
    _varlen_rms_canon_conv,
)

from .config import (
    ModernBertConfig,
    _create_norm,
    _get_activation,
    _parse_canon_layers_mode,
    _resolve_canon_kernel_size,
    _NOBLE_TARGET_ROLES,
)


from .rope import (
    ModernBertRotaryEmbedding,
    RePOModule,
    _apply_paired_rope,
    _apply_rope,
    _apply_rope_repo,
    _apply_rope_repo_sdpa,
    _rotate_half,
)

from .noble import CosNet, NOBLELinear, _build_linear

from .canon_layer import (
    ModernBertCanonLayer,
    _make_canon_layer,
    _varlen_canon_inner,
)


def _pair_sdpa_attention_mask(
    attn_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attn_mask is None:
        return None
    if attn_mask.shape[-2] == 1:
        return attn_mask.repeat_interleave(2, dim=-1)
    return attn_mask.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)


def _pair_flash_window_size(window_size: tuple[int, int]) -> tuple[int, int]:
    return tuple(-1 if w < 0 else 2 * w for w in window_size)


def _diff_attn_v2(
    attn1: torch.Tensor,
    attn2: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """Differential Attention V2 subtraction.

    attn1, attn2: attention outputs from paired heads (same shape).
    lam: per-token, per-head scalar (broadcastable to attn shape).
    Returns attn1 - sigmoid(lam) * attn2.
    """
    return attn1 - torch.sigmoid(lam).unsqueeze(-1) * attn2


def _full_attention_mask(
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.bool().all():
        return None
    return attention_mask[:, None, None, :].bool()


def _sliding_attention_mask(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    sliding_window: int,
    device: torch.device,
) -> torch.Tensor:
    q = torch.arange(seq_len, device=device)[:, None]
    kv = torch.arange(seq_len, device=device)[None, :]
    mask = (q - kv).abs() <= sliding_window
    mask = mask[None, None, :, :]

    if attention_mask is not None and not attention_mask.bool().all():
        mask = mask & attention_mask[:, None, None, :].bool()

    return mask


# ---------------------------------------------------------------------------
# Unpadding helpers for flash_attn_varlen_func
# ---------------------------------------------------------------------------


def _unpad_input(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive unpadding metadata from a (batch, seq_len) attention mask.

    Returns:
        indices:    (total_tokens,) – flat indices of non-padding positions.
        cu_seqlens: (batch + 1,)    – cumulative sequence lengths (int32).
        max_seqlen: 0-D int32 tensor – longest sequence in the batch.
    """
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens.max()  # keep as tensor to avoid torch.compile graph break
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def _pad_output(
    hidden: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Scatter a flat (total_tokens, …) tensor back to (batch, seqlen, …)."""
    out = torch.zeros(
        (batch * seqlen, *hidden.shape[1:]),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    out[indices] = hidden
    return out.view(batch, seqlen, *hidden.shape[1:])


def _position_ids_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    total: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert cu_seqlens to per-token position IDs (reset to 0 per sequence).

    Example: cu_seqlens=[0,3,5,9] → [0,1,2, 0,1, 0,1,2,3]
    """
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    offsets = cu_seqlens[:-1].repeat_interleave(seq_lens)
    return torch.arange(total, device=device, dtype=torch.int32) - offsets


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.norm = _create_norm(
            config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
        )
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.fused_up_gate = config.fused_up_gate
        if self.fused_up_gate:
            self.Wi = _build_linear(
                config.hidden_size,
                2 * config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
            self.W_up = None
            self.W_gate = None
        else:
            self.Wi = None
            self.W_up = _build_linear(
                config.hidden_size,
                config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
            self.W_gate = _build_linear(
                config.hidden_size,
                config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
        self.Wo = _build_linear(
            config.intermediate_size,
            config.hidden_size,
            config,
            bias=config.mlp_bias,
            role="ffn_wo",
        )
        self.drop = nn.Dropout(config.mlp_dropout)
        self.act = _get_activation(config.hidden_activation)
        self.canon_d = (
            _make_canon_layer(2 * config.intermediate_size, config)
            if "d" in config.canon_layer_set
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.fused_up_gate:
            wi = self.Wi(x)
        else:
            wi = torch.cat([self.W_up(x), self.W_gate(x)], dim=-1)
        if self.canon_d is not None:
            wi = self.canon_d(
                wi,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )
        x_proj, gate = wi.chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(x_proj) * gate))


class ModernBertSReluMLP(nn.Module):
    """MLP using relu(x)^2 activation (no gating).

    When USE_TRITON_SRELU is True, uses a fused Triton kernel for
    relu(x @ Wi.T)^2 in a single pass.  Wo is stored as (intermediate, hidden)
    to match the kernel layout (post @ Wo).

    When USE_TRITON_SRELU is False, uses plain PyTorch ops for benchmarking.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.Wo_weight = nn.Parameter(
            torch.empty(config.intermediate_size, config.hidden_size)
        )
        self.Wo_bias = (
            nn.Parameter(torch.zeros(config.hidden_size)) if config.mlp_bias else None
        )
        self.drop = nn.Dropout(config.mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if USE_TRITON_SRELU:
            from nanoplm.pretraining.models.modern_bert.triton_kernels import (
                FusedLinearReLUSquare,
            )

            out = FusedLinearReLUSquare.apply(x, self.Wi.weight, self.Wo_weight)
        else:
            h = F.relu(self.Wi(x))
            h = h * h
            out = h @ self.Wo_weight
        if self.Wo_bias is not None:
            out = out + self.Wo_bias
        return self.drop(out)


class ModernBertSwiGLUMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.fused_up_gate = config.fused_up_gate
        if self.fused_up_gate:
            self.Wi = _build_linear(
                config.hidden_size,
                2 * config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
            self.W_up = None
            self.W_gate = None
        else:
            self.Wi = None
            self.W_up = _build_linear(
                config.hidden_size,
                config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
            self.W_gate = _build_linear(
                config.hidden_size,
                config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
        self.Wo = _build_linear(
            config.intermediate_size,
            config.hidden_size,
            config,
            bias=config.mlp_bias,
            role="ffn_wo",
        )
        self.drop = nn.Dropout(config.mlp_dropout)
        self.canon_d = (
            _make_canon_layer(2 * config.intermediate_size, config)
            if "d" in config.canon_layer_set
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.fused_up_gate:
            wi = self.Wi(x)
        else:
            wi = torch.cat([self.W_up(x), self.W_gate(x)], dim=-1)
        if self.canon_d is not None:
            wi = self.canon_d(
                wi,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )
        x_proj, gate = wi.chunk(2, dim=-1)
        return self.Wo(self.drop(F.silu(gate) * x_proj))


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int = 0):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.use_qk_norm = config.use_qk_norm
        self.dropout = config.attention_dropout
        self.scale = self.head_dim**-0.5
        self.use_diff_attn_v2 = config.use_diff_attn_v2
        self.use_paired_head_attention = config.use_paired_head_attention

        # GQA: num_kv_heads <= num_heads. MHA when num_kv_heads == num_heads.
        self.num_kv_heads = config.num_kv_heads

        if self.use_diff_attn_v2:
            # DiffV2 doubles Q heads on top of whatever GQA config is set.
            # Each original head splits into a pair for differential subtraction.
            self.num_q_heads = 2 * self.num_heads
            # Lambda: per-token, per-head scalar controlling subtraction weight.
            self.lambda_proj = nn.Linear(
                config.hidden_size,
                self.num_heads,
                bias=False,
            )
        else:
            self.num_q_heads = self.num_heads
            self.lambda_proj = None

        self.fused_qkv = config.fused_qkv
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        if self.fused_qkv:
            qkv_dim = q_dim + 2 * kv_dim
            self.Wqkv = _build_linear(
                config.hidden_size,
                qkv_dim,
                config,
                bias=config.attention_bias,
                role="attn_qkv",
            )
            self.Wq = None
            self.Wq2 = None
            self.Wk = None
            self.Wv = None
        else:
            self.Wqkv = None
            if self.use_diff_attn_v2:
                # DiffV2 doubles Q heads into two groups for differential
                # subtraction.  Split them into Wq (group 1) and Wq2 (group 2)
                # so Muon orthogonalizes each group independently.
                half_q_dim = self.num_heads * self.head_dim
                self.Wq = _build_linear(
                    config.hidden_size, half_q_dim, config,
                    bias=config.attention_bias, role="attn_qkv",
                )
                self.Wq2 = _build_linear(
                    config.hidden_size, half_q_dim, config,
                    bias=config.attention_bias, role="attn_qkv",
                )
            else:
                self.Wq = _build_linear(
                    config.hidden_size, q_dim, config,
                    bias=config.attention_bias, role="attn_qkv",
                )
                self.Wq2 = None
            self.Wk = _build_linear(
                config.hidden_size, kv_dim, config,
                bias=config.attention_bias, role="attn_qkv",
            )
            self.Wv = _build_linear(
                config.hidden_size, kv_dim, config,
                bias=config.attention_bias, role="attn_qkv",
            )

        self.Wo = _build_linear(
            config.hidden_size,
            config.hidden_size,
            config,
            bias=config.attention_bias,
            role="attn_out",
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )
        qkv_out_dim = q_dim + 2 * kv_dim
        self.canon_b = (
            _make_canon_layer(qkv_out_dim, config)
            if "b" in config.canon_layer_set
            else None
        )

        # RePO: learned per-head positions replacing fixed RoPE indices
        self.repo = None
        if config.use_repo and layer_idx >= config.repo_after_n_layers:
            self.repo = RePOModule(
                config.hidden_size,
                config.num_attention_heads,
                config.head_dim,
            )
            theta = (
                config.global_rope_theta
                if config.layer_types[layer_idx] == "full_attention"
                else config.local_rope_theta
            )
            channel = torch.arange(0, config.head_dim, 2, dtype=torch.float32)
            self.register_buffer(
                "repo_inv_freq",
                1.0 / (theta ** (channel / config.head_dim)),
                persistent=False,
            )

    def _pair_varlen_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int | torch.Tensor
    ]:
        cos, sin = cos_sin
        q = q.reshape(q.shape[0], self.num_heads // 2, 2 * self.head_dim)
        k = k.reshape(k.shape[0], self.num_kv_heads // 2, 2 * self.head_dim)
        q, k = _apply_paired_rope(q, k, cos, sin)
        q = q.reshape(q.shape[0] * 2, self.num_heads // 2, self.head_dim)
        k = k.reshape(k.shape[0] * 2, self.num_kv_heads // 2, self.head_dim)
        v = v.reshape(v.shape[0] * 2, self.num_kv_heads // 2, self.head_dim)
        paired_cu = cu_seqlens * 2
        paired_max = max_seqlen * 2
        return q, k, v, paired_cu, paired_max

    def _pair_sdpa_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _, seq_len, _ = q.shape
        cos, sin = cos_sin
        q = q.transpose(1, 2).reshape(
            bsz, seq_len, self.num_heads // 2, 2 * self.head_dim
        )
        k = k.transpose(1, 2).reshape(
            bsz, seq_len, self.num_kv_heads // 2, 2 * self.head_dim
        )
        q, k = _apply_paired_rope(q, k, cos[0], sin[0])
        q = q.reshape(bsz, 2 * seq_len, self.num_heads // 2, self.head_dim).transpose(
            1, 2
        )
        k = k.reshape(
            bsz, 2 * seq_len, self.num_kv_heads // 2, self.head_dim
        ).transpose(1, 2)
        v = (
            v.transpose(1, 2)
            .reshape(bsz, 2 * seq_len, self.num_kv_heads // 2, self.head_dim)
            .transpose(1, 2)
        )
        return q, k, v

    # -- varlen (flash-attention) path -----------------------------------------

    def _forward_varlen(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        window_size: tuple[int, int],
        position_ids: Optional[torch.Tensor] = None,
        repo_active: bool = False,
    ) -> torch.Tensor:
        total = x.shape[0]  # (total_tokens, hidden)
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        if self.fused_qkv:
            qkv = self.Wqkv(x)
            if self.canon_b is not None:
                qkv = self.canon_b(
                    qkv, position_ids=position_ids, cu_seqlens=cu_seqlens
                )
            q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        else:
            if self.Wq2 is not None:
                # DiffV2 split: interleave the two Q groups so paired heads
                # (0::2, 1::2) are contiguous as the rest of the code expects.
                q1 = self.Wq(x).view(total, self.num_heads, self.head_dim)
                q2 = self.Wq2(x).view(total, self.num_heads, self.head_dim)
                q = torch.stack([q1, q2], dim=2).view(total, -1)
            else:
                q = self.Wq(x)
            k = self.Wk(x)
            v = self.Wv(x)
            if self.canon_b is not None:
                qkv = torch.cat([q, k, v], dim=-1)
                qkv = self.canon_b(
                    qkv, position_ids=position_ids, cu_seqlens=cu_seqlens
                )
                q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(total, self.num_q_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)

        if self.repo is not None and repo_active:
            positions = self.repo(x)  # (T, num_heads)
            q, k = _apply_rope_repo(q, k, positions, self.repo_inv_freq)
        elif not self.use_paired_head_attention:
            cos, sin = cos_sin
            q, k = _apply_rope(q, k, cos, sin)
        if self.use_qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        if self.use_paired_head_attention:
            q, k, v, cu_seqlens, max_seqlen = self._pair_varlen_qkv(
                q,
                k,
                v,
                cos_sin,
                cu_seqlens,
                max_seqlen,
            )
            window_size = _pair_flash_window_size(window_size)

        # When max_seqlen is already a plain int (static-shape mode) skip .item()
        # to avoid a graph break.  For tensor values (dynamic mode) .item() is
        # fine — flash_attn is an opaque C extension at a graph boundary.
        max_s = max_seqlen if isinstance(max_seqlen, int) else max_seqlen.item()
        kwargs = dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            softmax_scale=self.scale,
            window_size=window_size,
        )
        if _FLASH_HAS_DROPOUT:
            kwargs["dropout_p"] = self.dropout if self.training else 0.0

        y = _flash_varlen_fn(q, k, v, **kwargs)
        if isinstance(y, tuple):
            y = y[0]

        if self.use_paired_head_attention:
            y = y.reshape(total, self.num_heads, self.head_dim)

        if self.use_diff_attn_v2:
            # y: (total, 2*num_heads, head_dim) — paired heads in same GQA group
            # are contiguous, so 0::2 and 1::2 select the two halves.
            lam = self.lambda_proj(x)  # (total, num_heads)
            y = _diff_attn_v2(y[:, 0::2], y[:, 1::2], lam)
            # y: (total, num_heads, head_dim)

        y = y.contiguous().view(total, -1)  # (total, hidden)
        return self.out_drop(self.Wo(y))

    # -- SDPA (fallback) path --------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        repo_active: bool = False,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            return self._forward_varlen(
                x,
                cos_sin,
                cu_seqlens,
                max_seqlen,
                window_size,
                position_ids=position_ids,
                repo_active=repo_active,
            )

        bsz, seq_len, _ = x.shape
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        if self.fused_qkv:
            qkv = self.Wqkv(x)
            if self.canon_b is not None:
                qkv = self.canon_b(qkv, attention_mask=token_mask)
            q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        else:
            if self.Wq2 is not None:
                q1 = self.Wq(x).view(bsz, seq_len, self.num_heads, self.head_dim)
                q2 = self.Wq2(x).view(bsz, seq_len, self.num_heads, self.head_dim)
                q = torch.stack([q1, q2], dim=3).view(bsz, seq_len, -1)
            else:
                q = self.Wq(x)
            k = self.Wk(x)
            v = self.Wv(x)
            if self.canon_b is not None:
                qkv = torch.cat([q, k, v], dim=-1)
                qkv = self.canon_b(qkv, attention_mask=token_mask)
                q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(bsz, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.repo is not None and repo_active:
            positions = self.repo(x)  # (B, S, num_heads)
            q, k = _apply_rope_repo_sdpa(q, k, positions, self.repo_inv_freq)
        elif not self.use_paired_head_attention:
            cos, sin = cos_sin
            q, k = _apply_rope(q, k, cos, sin)
        if self.use_qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        if self.use_paired_head_attention:
            q, k, v = self._pair_sdpa_qkv(q, k, v, cos_sin)
            attn_mask = _pair_sdpa_attention_mask(attn_mask)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=(self.dropout if self.training else 0.0),
            scale=self.scale,
            enable_gqa=(self.num_q_heads != self.num_kv_heads),
        )

        if self.use_paired_head_attention:
            y = (
                y.transpose(1, 2)
                .reshape(bsz, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        if self.use_diff_attn_v2:
            # y: (B, 2H, S, D) -> differential subtraction
            lam = self.lambda_proj(x)  # (B, S, H)
            lam = lam.transpose(1, 2)  # (B, H, S)
            y = _diff_attn_v2(y[:, 0::2], y[:, 1::2], lam)
            # y: (B, H, S, D)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_drop(self.Wo(y))


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.has_mlp = (layer_idx != 0) or (not config.no_mlp_on_first_layer)
        self.norm_type = config.norm_type  # Store for fused Canon path
        self.activation_checkpointing = bool(
            getattr(config, "activation_checkpointing", False)
        )
        self.activation_checkpointing_mode = (
            str(getattr(config, "activation_checkpointing_mode", "layer"))
            .strip()
            .lower()
        )
        self.attn_norm = (
            nn.Identity()
            if layer_idx == 0
            else _create_norm(
                config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
            )
        )
        self.canon_a = (
            _make_canon_layer(config.hidden_size, config)
            if "a" in config.canon_layer_set
            else None
        )
        self.attn = ModernBertAttention(config, layer_idx=layer_idx)
        if self.has_mlp:
            self.mlp_norm = _create_norm(
                config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
            )
            self.canon_c = (
                _make_canon_layer(config.hidden_size, config)
                if "c" in config.canon_layer_set
                else None
            )
            if config.use_moe and config.moe_layer_flags[layer_idx]:
                from .moe import MoELayer
                self.mlp = MoELayer(config)
            else:
                # For dense layers in MoE mode, use the full active intermediate
                # size so per-token capacity matches the MoE layers.
                mlp_config = config
                if config.use_moe and not config.moe_layer_flags[layer_idx]:
                    import copy
                    mlp_config = copy.copy(config)
                    mlp_config.intermediate_size = config.moe_dense_intermediate_size
                if mlp_config.mlp_activation == "srelu":
                    self.mlp = ModernBertSReluMLP(mlp_config)
                elif mlp_config.mlp_activation == "swiglu":
                    self.mlp = ModernBertSwiGLUMLP(mlp_config)
                else:
                    self.mlp = ModernBertMLP(mlp_config)
        else:
            self.mlp_norm = nn.Identity()
            self.canon_c = None
            self.mlp = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        repo_active: bool = False,
        prores_alpha: "torch.Tensor | float" = 1.0,
    ) -> torch.Tensor:
        do_ckpt_attn = (
            self.activation_checkpointing
            and self.training
            and self.activation_checkpointing_mode in {"attn", "attn+mlp"}
        )
        if do_ckpt_attn:
            _attn_mask = attn_mask
            _cos_sin = cos_sin
            _cu_seqlens = cu_seqlens
            _max_seqlen = max_seqlen
            _window_size = window_size
            _position_ids = position_ids
            _token_mask = token_mask
            _repo_active = repo_active

            _fuse_ln_a = (
                self.canon_a is not None
                and isinstance(self.attn_norm, (nn.LayerNorm, nn.RMSNorm))
                and _cu_seqlens is not None
            )

            def _attn_branch(
                x_in: torch.Tensor,
                *,
                _layer: "ModernBertEncoderLayer" = self,
            ) -> torch.Tensor:
                if _fuse_ln_a:
                    attn_in = _layer.canon_a(
                        x_in,
                        position_ids=_position_ids,
                        cu_seqlens=_cu_seqlens,
                        fuse_ln_weight=_layer.attn_norm.weight,
                        fuse_ln_eps=_layer.attn_norm.eps,
                        fuse_norm_type=_layer.norm_type,
                    )
                else:
                    attn_in = _layer.attn_norm(x_in)
                    if _layer.canon_a is not None:
                        attn_in = _layer.canon_a(
                            attn_in,
                            position_ids=_position_ids,
                            cu_seqlens=_cu_seqlens,
                            attention_mask=_token_mask,
                        )
                return _layer.attn(
                    attn_in,
                    cos_sin=_cos_sin,
                    attn_mask=_attn_mask,
                    cu_seqlens=_cu_seqlens,
                    max_seqlen=_max_seqlen,
                    window_size=_window_size,
                    position_ids=_position_ids,
                    token_mask=_token_mask,
                    repo_active=_repo_active,
                )

            try:
                attn_out = _checkpoint(_attn_branch, x, use_reentrant=False)
            except TypeError:  # older torch checkpoint API
                attn_out = _checkpoint(_attn_branch, x)
            x = x + prores_alpha * attn_out
        else:
            fuse_ln_a = (
                self.canon_a is not None
                and isinstance(self.attn_norm, (nn.LayerNorm, nn.RMSNorm))
                and cu_seqlens is not None
            )
            if fuse_ln_a:
                attn_in = self.canon_a(
                    x,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    fuse_ln_weight=self.attn_norm.weight,
                    fuse_ln_eps=self.attn_norm.eps,
                    fuse_norm_type=self.norm_type,
                )
            else:
                attn_in = self.attn_norm(x)
                if self.canon_a is not None:
                    attn_in = self.canon_a(
                        attn_in,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                        attention_mask=token_mask,
                    )
            attn_out = self.attn(
                attn_in,
                cos_sin=cos_sin,
                attn_mask=attn_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                window_size=window_size,
                position_ids=position_ids,
                token_mask=token_mask,
                repo_active=repo_active,
            )
            x = x + prores_alpha * attn_out
        if self.mlp is not None:
            do_ckpt_mlp = (
                self.activation_checkpointing
                and self.training
                and self.activation_checkpointing_mode == "attn+mlp"
            )
            if do_ckpt_mlp:
                _position_ids = position_ids
                _cu_seqlens = cu_seqlens
                _token_mask = token_mask
                _fuse_ln_c = (
                    self.canon_c is not None
                    and isinstance(self.mlp_norm, (nn.LayerNorm, nn.RMSNorm))
                    and _cu_seqlens is not None
                )

                def _mlp_branch(
                    x_in: torch.Tensor,
                    *,
                    _layer: "ModernBertEncoderLayer" = self,
                ) -> torch.Tensor:
                    if _fuse_ln_c:
                        mlp_in = _layer.canon_c(
                            x_in,
                            position_ids=_position_ids,
                            cu_seqlens=_cu_seqlens,
                            fuse_ln_weight=_layer.mlp_norm.weight,
                            fuse_ln_eps=_layer.mlp_norm.eps,
                            fuse_norm_type=_layer.norm_type,
                        )
                    else:
                        mlp_in = _layer.mlp_norm(x_in)
                        if _layer.canon_c is not None:
                            mlp_in = _layer.canon_c(
                                mlp_in,
                                position_ids=_position_ids,
                                cu_seqlens=_cu_seqlens,
                                attention_mask=_token_mask,
                            )
                    return _layer.mlp(
                        mlp_in,
                        position_ids=_position_ids,
                        cu_seqlens=_cu_seqlens,
                        attention_mask=_token_mask,
                    )

                try:
                    mlp_out = _checkpoint(_mlp_branch, x, use_reentrant=False)
                except TypeError:  # older torch checkpoint API
                    mlp_out = _checkpoint(_mlp_branch, x)
                x = x + prores_alpha * mlp_out
            else:
                fuse_ln_c = (
                    self.canon_c is not None
                    and isinstance(self.mlp_norm, (nn.LayerNorm, nn.RMSNorm))
                    and cu_seqlens is not None
                )
                if fuse_ln_c:
                    mlp_in = self.canon_c(
                        x,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                        fuse_ln_weight=self.mlp_norm.weight,
                        fuse_ln_eps=self.mlp_norm.eps,
                        fuse_norm_type=self.norm_type,
                    )
                else:
                    mlp_in = self.mlp_norm(x)
                    if self.canon_c is not None:
                        mlp_in = self.canon_c(
                            mlp_in,
                            position_ids=position_ids,
                            cu_seqlens=cu_seqlens,
                            attention_mask=token_mask,
                        )
                mlp_out = self.mlp(
                    mlp_in,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    attention_mask=token_mask,
                )
                x = x + prores_alpha * mlp_out
        return x


class ModernBertAttnResidual(nn.Module):
    """Attention residual branch: x -> x + attn(attn_norm(x))."""

    def __init__(self, encoder: ModernBertEncoderLayer):
        super().__init__()
        # Store encoder without registering as a submodule (MHCLiteSublayersLayer owns it).
        self.__dict__["_encoder"] = encoder
        self.attention_type = encoder.attention_type

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        repo_active: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        enc: ModernBertEncoderLayer = self.__dict__["_encoder"]
        attn_in = enc.attn_norm(x)
        if enc.canon_a is not None:
            attn_in = enc.canon_a(
                attn_in,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=token_mask,
            )
        return x + enc.attn(
            attn_in,
            cos_sin=cos_sin,
            attn_mask=attn_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            window_size=window_size,
            position_ids=position_ids,
            token_mask=token_mask,
            repo_active=repo_active,
        )


class ModernBertMLPResidual(nn.Module):
    """MLP residual branch: x -> x + mlp(mlp_norm(x))."""

    def __init__(self, encoder: ModernBertEncoderLayer):
        super().__init__()
        # Store encoder without registering as a submodule (MHCLiteSublayersLayer owns it).
        self.__dict__["_encoder"] = encoder
        self.attention_type = encoder.attention_type

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        repo_active: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        enc: ModernBertEncoderLayer = self.__dict__["_encoder"]
        if enc.mlp is None:
            return x
        mlp_in = enc.mlp_norm(x)
        if enc.canon_c is not None:
            mlp_in = enc.canon_c(
                mlp_in,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=token_mask,
            )
        return x + enc.mlp(
            mlp_in,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            attention_mask=token_mask,
        )


def _build_permutation_matrices(n: int) -> tuple[torch.Tensor, int]:
    """Pre-compute all n! permutation matrices (flattened) and identity index."""
    from itertools import permutations

    perms = list(permutations(range(n)))
    identity_idx = 0  # (0,1,...,n-1) is first in lexicographic order
    P = torch.zeros(len(perms), n * n)
    for i, perm in enumerate(perms):
        for row, col in enumerate(perm):
            P[i, row * n + col] = 1.0
    return P, identity_idx


class MHCLiteBlock(nn.Module):
    """mHC-lite: wraps a transformer layer with n residual streams
    and doubly stochastic mixing via convex combination of permutation matrices.

    Forward: x_streams (..., n, C) -> (..., n, C)
    x_{l+1} = H^res_l @ x_l + H^post_l * f(H^pre_l @ x_l)
    where f is the wrapped transformer layer (without residual).

    Optimized I/O: fused projection, merged H_res-h_post application.
    Optional Triton kernels for further fusion (set triton_fused=True).
    """

    def __init__(
        self,
        n_streams: int,
        hidden_size: int,
        layer: nn.Module,
        triton_fused: bool = False,
    ):
        super().__init__()
        self.n = n_streams
        self.C = hidden_size
        self.nC = n_streams * hidden_size
        self.layer = layer
        n_fact = math.factorial(n_streams)
        self.n_fact = n_fact
        self.triton_fused = triton_fused

        self.alpha_pre = nn.Parameter(torch.tensor([0.01]))
        self.alpha_post = nn.Parameter(torch.tensor([0.01]))
        self.alpha_res = nn.Parameter(torch.tensor([0.01]))

        # Fused single projection: pre(n) + post(n) + res(n!) outputs
        total_out = n_streams + n_streams + n_fact
        self.W_all = nn.Linear(self.nC, total_out, bias=True)

        perm_flat, self._identity_idx = _build_permutation_matrices(n_streams)
        self.register_buffer("perm_mat", perm_flat)  # (n!, n*n)

    @property
    def attention_type(self):
        return self.layer.attention_type

    def _mhc_coeffs_pytorch(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute h_pre/h_post/H_merged (PyTorch path)."""
        n = self.n
        dt = x_streams.dtype

        x_flat = x_streams.reshape(*x_streams.shape[:-2], self.nC)
        x_norm = F.rms_norm(x_flat, (self.nC,))

        all_proj = F.linear(x_norm, self.W_all.weight.to(dt), None)
        pre_proj, post_proj, res_proj = all_proj.split([n, n, self.n_fact], dim=-1)

        bias = self.W_all.bias.to(dt)
        pre_bias = bias[:n]
        post_bias = bias[n : 2 * n]
        res_bias = bias[2 * n :]

        h_pre = torch.sigmoid(self.alpha_pre.to(dt) * pre_proj + pre_bias)
        h_post = 2.0 * torch.sigmoid(self.alpha_post.to(dt) * post_proj + post_bias)
        a_res = F.softmax(self.alpha_res.to(dt) * res_proj + res_bias, dim=-1)

        H_res = torch.matmul(a_res, self.perm_mat.to(dt)).unflatten(-1, (n, n))
        H_merged = H_res - h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)
        return h_pre, h_post, H_merged

    def _mhc_pre_map_pytorch(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-map bundle: layer_input + coefficients for post-res."""
        h_pre, h_post, H_merged = self._mhc_coeffs_pytorch(x_streams)
        layer_input = torch.matmul(h_pre.unsqueeze(-2), x_streams).squeeze(-2)
        return layer_input, H_merged, h_post

    def _mhc_post_res_pytorch(
        self,
        x_streams: torch.Tensor,
        layer_output: torch.Tensor,
        H_merged: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(H_merged, x_streams) + h_post.unsqueeze(
            -1
        ) * layer_output.unsqueeze(-2)

    def _forward_pytorch(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        """Pure PyTorch forward path (always correct, works everywhere)."""
        pre_map = (
            self._compiled_mhc_pre_map_pytorch
            if self.training and hasattr(self, "_compiled_mhc_pre_map_pytorch")
            else self._mhc_pre_map_pytorch
        )
        post_res = (
            self._compiled_mhc_post_res_pytorch
            if self.training and hasattr(self, "_compiled_mhc_post_res_pytorch")
            else self._mhc_post_res_pytorch
        )
        use_ckpt = (
            USE_ACTIVATION_CHECKPOINTING_MHC
            and self.training
            and torch.is_grad_enabled()
            and x_streams.requires_grad
        )

        if use_ckpt:
            layer_input, H_merged, h_post = _checkpoint(
                lambda x_: pre_map(x_),
                x_streams,
                use_reentrant=False,
            )
        else:
            layer_input, H_merged, h_post = pre_map(x_streams)
        layer_output = self.layer(layer_input, **kwargs)

        if use_ckpt:
            return _checkpoint(
                lambda x_, lo_, H_, hp_: post_res(x_, lo_, H_, hp_),
                x_streams,
                layer_output,
                H_merged,
                h_post,
                use_reentrant=False,
            )
        return post_res(x_streams, layer_output, H_merged, h_post)

    def _mhc_coeffs_triton(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute h_pre/h_post/H_merged using fused K1 + PyTorch tiny ops."""
        n = self.n
        T = x_streams.shape[0]
        dt = x_streams.dtype

        x_flat = x_streams.reshape(T, self.nC)
        all_proj, _inv_rms = torch.ops.nanoplm_mhc.fused_rmsnorm_project(
            x_flat, self.W_all.weight.to(dt)
        )

        pre_proj, post_proj, res_proj = all_proj.split([n, n, self.n_fact], dim=-1)

        bias = self.W_all.bias.to(dt)
        h_pre = torch.sigmoid(self.alpha_pre.to(dt) * pre_proj + bias[:n])
        h_post = 2.0 * torch.sigmoid(
            self.alpha_post.to(dt) * post_proj + bias[n : 2 * n]
        )
        a_res = F.softmax(self.alpha_res.to(dt) * res_proj + bias[2 * n :], dim=-1)

        H_res = torch.matmul(a_res, self.perm_mat.to(dt)).unflatten(-1, (n, n))
        H_merged = H_res - h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)
        return h_pre, h_post, H_merged

    def _mhc_pre_map_triton(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-map bundle for Triton path: layer_input + post-res coefficients."""
        h_pre, h_post, H_merged = self._mhc_coeffs_triton(x_streams)
        layer_input = torch.ops.nanoplm_mhc.fused_pre_map(x_streams, h_pre.float())
        return layer_input, H_merged.float(), h_post.float()

    def _mhc_post_res_triton(
        self,
        x_streams: torch.Tensor,
        layer_output: torch.Tensor,
        H_merged: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.nanoplm_mhc.fused_post_res(
            x_streams, layer_output, H_merged, h_post
        )

    def _forward_triton(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        """Triton-fused forward path. Requires (T, n, C) bf16 on CUDA.

        Uses Triton for the memory-heavy stream operations (K3: pre-map,
        K4: post-res) and PyTorch for the coefficient computation (tiny ops).
        """
        pre_map = (
            self._compiled_mhc_pre_map_triton
            if self.training and hasattr(self, "_compiled_mhc_pre_map_triton")
            else self._mhc_pre_map_triton
        )
        post_res = (
            self._compiled_mhc_post_res_triton
            if self.training and hasattr(self, "_compiled_mhc_post_res_triton")
            else self._mhc_post_res_triton
        )
        use_ckpt = (
            USE_ACTIVATION_CHECKPOINTING_MHC
            and self.training
            and torch.is_grad_enabled()
            and x_streams.requires_grad
        )
        # Eval/inference should avoid autotune warmup latency.
        autotune_ctx = (
            nullcontext()
            if self.training
            else _mhc_triton_ops.disable_autotune_temporarily()
        )

        with autotune_ctx:
            if use_ckpt:
                layer_input, H_merged, h_post = _checkpoint(
                    lambda x_: pre_map(x_),
                    x_streams,
                    use_reentrant=False,
                )
            else:
                layer_input, H_merged, h_post = pre_map(x_streams)

            # Transformer layer
            layer_output = self.layer(layer_input, **kwargs)

            # K4: Triton fused post-res (H_merged @ x + h_post * layer_output)
            if use_ckpt:
                return _checkpoint(
                    lambda x_, lo_, H_, hp_: post_res(x_, lo_, H_, hp_),
                    x_streams,
                    layer_output,
                    H_merged,
                    h_post,
                    use_reentrant=False,
                )
            return post_res(x_streams, layer_output, H_merged, h_post)

    def forward(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        """x_streams: (..., n, C).  Returns (..., n, C)."""
        # Use Triton path when: triton_fused=True, CUDA, bf16, 2D token dim
        use_triton = (
            self.triton_fused
            and x_streams.is_cuda
            and x_streams.dtype == torch.bfloat16
            and x_streams.dim() == 3  # (T, n, C) — no batch dim
        )

        if use_triton:
            return self._forward_triton(x_streams, **kwargs)
        return self._forward_pytorch(x_streams, **kwargs)


class MHCLiteSublayersLayer(nn.Module):
    """Transformer layer with mHC-lite applied to attention and MLP sublayers separately."""

    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.enc = ModernBertEncoderLayer(config, layer_idx)
        self.mhc_attn = MHCLiteBlock(
            config.mhc_n_streams,
            config.hidden_size,
            ModernBertAttnResidual(self.enc),
            triton_fused=config.mhc_triton_fused,
        )
        self.mhc_mlp = (
            MHCLiteBlock(
                config.mhc_n_streams,
                config.hidden_size,
                ModernBertMLPResidual(self.enc),
                triton_fused=config.mhc_triton_fused,
            )
            if self.enc.mlp is not None
            else None
        )

    @property
    def attention_type(self):
        return self.enc.attention_type

    def forward(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        x_streams = self.mhc_attn(x_streams, **kwargs)
        if self.mhc_mlp is not None:
            x_streams = self.mhc_mlp(x_streams, **kwargs)
        return x_streams


class ModernBertModel(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        if config.use_mhc_lite:
            if config.mhc_lite_wrapping_level == "layer":
                self.layers = nn.ModuleList(
                    [
                        MHCLiteBlock(
                            config.mhc_n_streams,
                            config.hidden_size,
                            ModernBertEncoderLayer(config, i),
                            triton_fused=config.mhc_triton_fused,
                        )
                        for i in range(config.num_hidden_layers)
                    ]
                )
            else:
                self.layers = nn.ModuleList(
                    [
                        MHCLiteSublayersLayer(config, i)
                        for i in range(config.num_hidden_layers)
                    ]
                )
        else:
            self.layers = nn.ModuleList(
                [
                    ModernBertEncoderLayer(config, i)
                    for i in range(config.num_hidden_layers)
                ]
            )
        # ---- Huginn-style recycling: split into prelude/recurrent/coda ----
        # Plain Python lists, NOT ModuleLists: the modules already live in
        # self.layers and are registered there. Re-registering them under
        # ModuleLists would double every key in state_dict (e.g.
        # `layers.0.attn.Wo.weight` AND `prelude_layers.0.attn.Wo.weight`
        # pointing to the same tensor), bloating checkpoints and breaking
        # strict load_state_dict.
        P = config.recycling_num_prelude_layers
        C = config.recycling_num_coda_layers
        L = config.num_hidden_layers
        self.prelude_layers: list[nn.Module] = list(self.layers[:P])
        self.recurrent_layers: list[nn.Module] = list(self.layers[P:L - C]) if (L - P - C) > 0 else []
        self.coda_layers: list[nn.Module] = list(self.layers[L - C:]) if C > 0 else []

        # Adapter only when recycling is active.
        if config.recycling_loop_count > 1:
            self.adapter = nn.Linear(
                2 * config.hidden_size,
                config.hidden_size,
                bias=config.recycling_adapter_bias,
            )
        else:
            self.adapter = None

        if config.use_resid_lambdas:
            self.resid_lambdas = nn.Parameter(torch.ones(config.num_hidden_layers))
        else:
            self.register_parameter("resid_lambdas", None)
        if config.use_x0_lambdas:
            self.x0_lambdas = nn.Parameter(torch.zeros(config.num_hidden_layers))
        else:
            self.register_parameter("x0_lambdas", None)
        self.final_norm = _create_norm(
            config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
        )
        self.rotary_emb = ModernBertRotaryEmbedding(config)
        # RePO: disabled at init; enabled by the training loop after
        # repo_rope_warmup_steps (or warmup_steps fallback).
        self.repo_active = False
        # ProRes: progressive residual warmup. The training loop calls
        # update_prores_alphas(step) once per optimizer step.
        # Stored as a non-persistent buffer so torch.compile treats values as
        # dynamic (no graph-break / recompilation per unique float).
        self.use_prores = config.use_prores
        self._prores_T = config.prores_T
        _init_val = 0.0 if config.use_prores else 1.0
        self.register_buffer(
            "_prores_alphas",
            torch.full((config.num_hidden_layers,), _init_val),
            persistent=False,
        )

    def update_prores_alphas(self, step: int) -> None:
        """Recompute per-layer ProRes alphas. Call once per optimizer step."""
        T = self._prores_T
        vals = [
            min(step / (T * (l + 1)), 1.0) for l in range(self.config.num_hidden_layers)
        ]
        self._prores_alphas.copy_(torch.tensor(vals, dtype=self._prores_alphas.dtype))

    def _recurrent_step_varlen(
        self,
        e: torch.Tensor,
        s_prev: torch.Tensor,
        *,
        rope_full,
        rope_sliding,
        cu_seqlens: torch.Tensor,
        max_seqlen,
        window_full,
        window_sliding,
        position_ids: torch.Tensor,
        repo_active: bool,
        prores_alphas: torch.Tensor,
    ) -> torch.Tensor:
        """One iteration of the recurrent block (varlen / flash path).

        Inlines the per-layer call so the for-loop unrolls inside a single
        compiled frame instead of dispatching through a closure (which would
        re-enter Dynamo once per layer and specialize on ``layer_idx``,
        blowing ``cache_size_limit`` for R > cache_size). All path-specific
        tensors arrive as explicit kwargs — no closure capture — so
        ``torch.compile`` guards stay stable across calls.
        """
        P = self.config.recycling_num_prelude_layers
        u = self.adapter(torch.cat([s_prev, e], dim=-1))
        for i, layer in enumerate(self.recurrent_layers):
            lt = layer.attention_type  # str, constant per layer
            if lt == "full_attention":
                cos_sin = rope_full
                window_size = window_full
            else:
                cos_sin = rope_sliding
                window_size = window_sliding
            u = layer(
                u,
                cos_sin=cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                window_size=window_size,
                position_ids=position_ids,
                repo_active=repo_active,
                prores_alpha=prores_alphas[P + i],
            )
        return u

    def _recurrent_step_sdpa(
        self,
        e: torch.Tensor,
        s_prev: torch.Tensor,
        *,
        rope_full,
        rope_sliding,
        attn_mask_full,
        attn_mask_sliding,
        token_mask: torch.Tensor,
        repo_active: bool,
        prores_alphas: torch.Tensor,
    ) -> torch.Tensor:
        """One iteration of the recurrent block (SDPA fallback path).

        Same no-closure design as ``_recurrent_step_varlen`` — see that
        docstring for rationale.
        """
        P = self.config.recycling_num_prelude_layers
        u = self.adapter(torch.cat([s_prev, e], dim=-1))
        for i, layer in enumerate(self.recurrent_layers):
            lt = layer.attention_type
            if lt == "full_attention":
                cos_sin = rope_full
                attn_mask = attn_mask_full
            else:
                cos_sin = rope_sliding
                attn_mask = attn_mask_sliding
            u = layer(
                u,
                attn_mask=attn_mask,
                cos_sin=cos_sin,
                position_ids=None,
                token_mask=token_mask,
                repo_active=repo_active,
                prores_alpha=prores_alphas[P + i],
            )
        return u

    @torch._dynamo.disable(recursive=False)
    def _drive_recurrence(
        self,
        e: torch.Tensor,
        s: torch.Tensor,
        r,
        K,
        step_fn,
    ) -> torch.Tensor:
        """Run r iterations of the recurrent block in eager Python.

        Why: the trip count r is sampled per micro-step (Poisson). Without
        this disable, r would flow into ``for _ in range(r):`` inside a
        torch.compile-traced graph; Dynamo specializes on Python-int loop
        bounds, so every unique r causes a guard miss + fresh trace +
        autotune pass, thrashing the compile cache and stalling training.
        Disabling Dynamo here keeps the variable trip count confined to
        eager Python while the shape-stable prelude and coda regions still
        compile normally on either side of this call (graph break).

        Decorator is ``@torch._dynamo.disable(recursive=False)``: the
        driver body itself runs eager (so ``range(r)`` stays in Python),
        but each ``step_fn(e, s)`` call re-enters Dynamo and compiles the
        shape-stable recurrent step as its own cached graph. This recovers
        torch.compile fusion on the recurrent block (~75% of compute) —
        matching Huginn's reference (seal-rg/recurrent-pretraining
        ``model_dynamic.iterate_forward``).

        Critical precondition: ``step_fn`` must call a method that does NOT
        dispatch per-layer through a closure with an int ``layer_idx`` arg.
        We tried that and it produced R separately-traced frames (one per
        ``layer_idx`` value) that blow ``cache_size_limit`` when R > 8.
        Instead, ``_recurrent_step_varlen`` / ``_recurrent_step_sdpa``
        inline the per-layer call directly — the for-loop unrolls inside
        one compiled frame.

        r and K may arrive as Python ints (eager callers) or as 0-d int
        tensors (compiled training loop — passing tensors prevents Dynamo
        from installing an equality guard on the int value at the
        graph-break boundary, which would also force a recompile per r).

        Splits into a no-grad outer phase (r - K iterations) and an in-graph
        phase (K iterations) so truncated BPTT only retains gradients for
        the last K iters. ``e`` is NOT detached so prelude params still
        receive gradients through the in-graph iters.
        """
        r = int(r.item()) if isinstance(r, torch.Tensor) else int(r)
        K = int(K.item()) if isinstance(K, torch.Tensor) else int(K)
        no_grad_steps = max(0, r - K)
        with torch.no_grad():
            for _ in range(no_grad_steps):
                s = step_fn(e, s)
        if no_grad_steps > 0:
            s = s.detach()
        for _ in range(r - no_grad_steps):
            s = step_fn(e, s)
        return s

    def _apply_takase_init(self) -> None:
        """Huginn §4.1 Takase spike init applied to the recurrent block + adapter.

        Effective depth l = P + r_bar * R + C with r_bar = max(1, loop_count - 1).
        Recurrent residual out-projections: std = sqrt(1 / (5 * h * l))
        Recurrent non-residual linears + adapter: std = sqrt(2 / (5 * h))
        All draws from truncated normal at ±3σ.

        Only runs when config.recycling_use_takase_init AND config.recycling_loop_count > 1.
        Prelude/coda layers are NOT touched; they keep whatever wo_init produced.
        """
        cfg = self.config
        if not cfg.recycling_use_takase_init or cfg.recycling_loop_count <= 1:
            return

        h = cfg.hidden_size
        P = cfg.recycling_num_prelude_layers
        C = cfg.recycling_num_coda_layers
        L = cfg.num_hidden_layers
        R = L - P - C
        r_bar = max(1, cfg.recycling_loop_count - 1)
        l = P + r_bar * R + C

        std_general = math.sqrt(2.0 / (5.0 * h))
        std_out = math.sqrt(1.0 / (5.0 * h * l))

        def _trunc_init(tensor: torch.Tensor, std: float) -> None:
            torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)

        for layer in self.recurrent_layers:
            attn = getattr(layer, "attn", None)
            if attn is not None:
                if getattr(attn, "Wo", None) is not None:
                    _trunc_init(attn.Wo.weight, std_out)
                # Attention input projections: fused Wqkv OR separate Wq/Wq2/Wk/Wv
                if getattr(attn, "Wqkv", None) is not None:
                    _trunc_init(attn.Wqkv.weight, std_general)
                else:
                    for name in ("Wq", "Wq2", "Wk", "Wv"):
                        proj = getattr(attn, name, None)
                        if proj is not None:
                            _trunc_init(proj.weight, std_general)

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                # MLP input projections: fused Wi OR separate W_up/W_gate
                if getattr(mlp, "Wi", None) is not None:
                    _trunc_init(mlp.Wi.weight, std_general)
                elif getattr(mlp, "W_up", None) is not None:
                    _trunc_init(mlp.W_up.weight, std_general)
                    if getattr(mlp, "W_gate", None) is not None:
                        _trunc_init(mlp.W_gate.weight, std_general)
                # MLP output: Linear `Wo` OR raw Parameter `Wo_weight` (SReLU MLP)
                if getattr(mlp, "Wo", None) is not None:
                    _trunc_init(mlp.Wo.weight, std_out)
                elif getattr(mlp, "Wo_weight", None) is not None:
                    _trunc_init(mlp.Wo_weight, std_out)

        if self.adapter is not None:
            _trunc_init(self.adapter.weight, std_general)
            if self.adapter.bias is not None:
                torch.nn.init.zeros_(self.adapter.bias)

        from nanoplm.utils.logger import logger
        logger.info(
            "[recycling] Takase spike init applied: l=%d (P=%d, r_bar=%d, R=%d, C=%d), "
            "std_general=%.6f, std_out=%.6f",
            l, P, r_bar, R, C, std_general, std_out,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        _cu_seqlens: Optional[torch.Tensor] = None,
        _max_seqlen: Optional[int] = None,
        _position_ids: Optional[torch.Tensor] = None,
        recycling_loop_count: Optional[int] = None,
        recycling_bptt: Optional[int] = None,
    ) -> torch.Tensor:
        # ---- varlen (flash-attention) path --------------------------------
        if _cu_seqlens is not None:
            device = input_ids.device
            x = self.embeddings(input_ids)  # (total_tokens, hidden)
            if self.config.use_mhc_lite:
                n = self.config.mhc_n_streams
                x = F.pad(x.unsqueeze(-2), (0, 0, 0, n - 1))  # (T, n, C)
            x0 = x if self.x0_lambdas is not None else None
            if _position_ids is None:
                _position_ids = _position_ids_from_cu_seqlens(
                    _cu_seqlens, x.shape[0], x.device
                )

            # Build RoPE tables only up to the current batch's max sequence
            # length when available. This trims needless work for short-sequence
            # packed batches while preserving a safe fallback for manual callers.
            rope_len = (
                _max_seqlen
                if _max_seqlen is not None
                else self.config.max_position_embeddings
            )
            cos_f, sin_f = self.rotary_emb(rope_len, device, x.dtype, "full_attention")
            cos_s, sin_s = self.rotary_emb(
                rope_len, device, x.dtype, "sliding_attention"
            )
            rope = {
                "full_attention": (
                    self.rotary_emb.paired_from_positions(
                        _position_ids,
                        device,
                        x.dtype,
                        "full_attention",
                    )
                    if self.config.use_paired_head_attention
                    else (cos_f[0, _position_ids], sin_f[0, _position_ids])
                ),
                "sliding_attention": (
                    self.rotary_emb.paired_from_positions(
                        _position_ids,
                        device,
                        x.dtype,
                        "sliding_attention",
                    )
                    if self.config.use_paired_head_attention
                    else (cos_s[0, _position_ids], sin_s[0, _position_ids])
                ),
            }
            windows = {
                "full_attention": (-1, -1),
                "sliding_attention": (
                    self.config.sliding_window,
                    self.config.sliding_window,
                ),
            }

            repo_active = self.repo_active
            prores_alphas = self._prores_alphas  # (num_layers,) tensor buffer

            _recycle = (
                self.config.recycling_loop_count > 1 or recycling_loop_count is not None
            )
            if _recycle:
                if self.adapter is None:
                    raise ValueError(
                        "Recycling forward requested (recycling_loop_count > 1) but the "
                        "adapter is None — the model was constructed with "
                        "config.recycling_loop_count == 1. Rebuild the model with "
                        "config.recycling_loop_count > 1 before passing the kwarg."
                    )
                r = recycling_loop_count if recycling_loop_count is not None else self.config.recycling_loop_count
                K = recycling_bptt if recycling_bptt is not None else 8

            # Path-specific tensors hoisted once per forward so the unrolled
            # prelude/coda loops below see the same captures Dynamo can
            # constant-fold; no per-call closure means no layer_idx guard.
            _rope_full = rope["full_attention"]
            _rope_sliding = rope["sliding_attention"]
            _window_full = windows["full_attention"]
            _window_sliding = windows["sliding_attention"]

            if _recycle:
                h_pre = x
                # Inlined prelude: per-layer call directly in the unrolled
                # for-loop. The previous _run_layer_varlen closure made
                # Dynamo trace this as a separate frame and specialize on
                # layer_idx (P specializations).
                for i, layer in enumerate(self.prelude_layers):
                    lt = layer.attention_type
                    if lt == "full_attention":
                        _cos_sin = _rope_full
                        _window_size = _window_full
                    else:
                        _cos_sin = _rope_sliding
                        _window_size = _window_sliding
                    h_pre = layer(
                        h_pre,
                        cos_sin=_cos_sin,
                        cu_seqlens=_cu_seqlens,
                        max_seqlen=_max_seqlen,
                        window_size=_window_size,
                        position_ids=_position_ids,
                        repo_active=repo_active,
                        prores_alpha=prores_alphas[i],
                    )
                e = h_pre

                s = torch.randn_like(e) * self.config.recycling_s0_noise_sigma

                def _step_varlen(e_in, s_in):
                    return self._recurrent_step_varlen(
                        e_in,
                        s_in,
                        rope_full=_rope_full,
                        rope_sliding=_rope_sliding,
                        cu_seqlens=_cu_seqlens,
                        max_seqlen=_max_seqlen,
                        window_full=_window_full,
                        window_sliding=_window_sliding,
                        position_ids=_position_ids,
                        repo_active=repo_active,
                        prores_alphas=prores_alphas,
                    )
                s = self._drive_recurrence(e, s, r, K, _step_varlen)

                P_count = self.config.recycling_num_prelude_layers
                R_count = len(self.recurrent_layers)
                # Inlined coda — same reason as the prelude block above.
                for i, layer in enumerate(self.coda_layers):
                    lt = layer.attention_type
                    if lt == "full_attention":
                        _cos_sin = _rope_full
                        _window_size = _window_full
                    else:
                        _cos_sin = _rope_sliding
                        _window_size = _window_sliding
                    s = layer(
                        s,
                        cos_sin=_cos_sin,
                        cu_seqlens=_cu_seqlens,
                        max_seqlen=_max_seqlen,
                        window_size=_window_size,
                        position_ids=_position_ids,
                        repo_active=repo_active,
                        prores_alpha=prores_alphas[P_count + R_count + i],
                    )
                x = s
            else:
                for i, layer in enumerate(self.layers):
                    if self.resid_lambdas is not None:
                        x = self.resid_lambdas[i] * x
                    if self.x0_lambdas is not None:
                        x = x + self.x0_lambdas[i] * x0
                    alpha = prores_alphas[i]
                    lt = layer.attention_type
                    if (
                        self.config.activation_checkpointing
                        and self.training
                        and str(self.config.activation_checkpointing_mode).strip().lower()
                        == "layer"
                    ):
                        cos_sin = rope[lt]
                        cu_seqlens = _cu_seqlens
                        max_seqlen = _max_seqlen
                        window_size = windows[lt]
                        position_ids = _position_ids

                        def _layer_forward(
                            x_in: torch.Tensor,
                            *,
                            _layer: nn.Module = layer,
                            _cos_sin=cos_sin,
                            _cu_seqlens=cu_seqlens,
                            _max_seqlen=max_seqlen,
                            _window_size=window_size,
                            _position_ids=position_ids,
                            _repo_active: bool = repo_active,
                            _prores_alpha=alpha,
                        ) -> torch.Tensor:
                            return _layer(
                                x_in,
                                cos_sin=_cos_sin,
                                cu_seqlens=_cu_seqlens,
                                max_seqlen=_max_seqlen,
                                window_size=_window_size,
                                position_ids=_position_ids,
                                repo_active=_repo_active,
                                prores_alpha=_prores_alpha,
                            )

                        try:
                            x = _checkpoint(_layer_forward, x, use_reentrant=False)
                        except TypeError:  # older torch checkpoint API
                            x = _checkpoint(_layer_forward, x)
                    else:
                        x = layer(
                            x,
                            cos_sin=rope[lt],
                            cu_seqlens=_cu_seqlens,
                            max_seqlen=_max_seqlen,
                            window_size=windows[lt],
                            position_ids=_position_ids,
                            repo_active=repo_active,
                            prores_alpha=alpha,
                        )

            if self.config.use_mhc_lite:
                x = x[..., 0, :]  # compress: take stream 0
            return self.final_norm(x)

        # ---- SDPA (fallback) path -----------------------------------------
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}"
            )
        device = input_ids.device

        x = self.embeddings(input_ids)
        if self.config.use_mhc_lite:
            n = self.config.mhc_n_streams
            x = F.pad(x.unsqueeze(-2), (0, 0, 0, n - 1))  # (B, S, n, C)
        x0 = x if self.x0_lambdas is not None else None

        attn_masks = {
            "full_attention": _full_attention_mask(attention_mask),
            "sliding_attention": _sliding_attention_mask(
                attention_mask,
                seq_len=seq_len,
                sliding_window=self.config.sliding_window,
                device=device,
            ),
        }

        rope = {
            "full_attention": (
                self.rotary_emb.paired(
                    seq_len=seq_len,
                    device=device,
                    dtype=x.dtype,
                    layer_type="full_attention",
                )
                if self.config.use_paired_head_attention
                else self.rotary_emb(
                    seq_len=seq_len,
                    device=device,
                    dtype=x.dtype,
                    layer_type="full_attention",
                )
            ),
            "sliding_attention": (
                self.rotary_emb.paired(
                    seq_len=seq_len,
                    device=device,
                    dtype=x.dtype,
                    layer_type="sliding_attention",
                )
                if self.config.use_paired_head_attention
                else self.rotary_emb(
                    seq_len=seq_len,
                    device=device,
                    dtype=x.dtype,
                    layer_type="sliding_attention",
                )
            ),
        }

        repo_active = self.repo_active
        prores_alphas = self._prores_alphas  # (num_layers,) tensor buffer

        _recycle = (
            self.config.recycling_loop_count > 1 or recycling_loop_count is not None
        )
        if _recycle:
            if self.adapter is None:
                raise ValueError(
                    "Recycling forward requested (recycling_loop_count > 1) but the "
                    "adapter is None — the model was constructed with "
                    "config.recycling_loop_count == 1. Rebuild the model with "
                    "config.recycling_loop_count > 1 before passing the kwarg."
                )
            r = recycling_loop_count if recycling_loop_count is not None else self.config.recycling_loop_count
            K = recycling_bptt if recycling_bptt is not None else 8

        # Path-specific tensors hoisted once per forward — see varlen path
        # for the rationale (avoids per-layer closure -> layer_idx guard).
        _rope_full = rope["full_attention"]
        _rope_sliding = rope["sliding_attention"]
        _attn_mask_full = attn_masks["full_attention"]
        _attn_mask_sliding = attn_masks["sliding_attention"]

        if _recycle:
            h_pre = x
            # Inlined prelude.
            for i, layer in enumerate(self.prelude_layers):
                lt = layer.attention_type
                if lt == "full_attention":
                    _cos_sin = _rope_full
                    _attn_mask = _attn_mask_full
                else:
                    _cos_sin = _rope_sliding
                    _attn_mask = _attn_mask_sliding
                h_pre = layer(
                    h_pre,
                    attn_mask=_attn_mask,
                    cos_sin=_cos_sin,
                    position_ids=None,
                    token_mask=attention_mask,
                    repo_active=repo_active,
                    prores_alpha=prores_alphas[i],
                )
            e = h_pre

            s = torch.randn_like(e) * self.config.recycling_s0_noise_sigma

            def _step_sdpa(e_in, s_in):
                return self._recurrent_step_sdpa(
                    e_in,
                    s_in,
                    rope_full=_rope_full,
                    rope_sliding=_rope_sliding,
                    attn_mask_full=_attn_mask_full,
                    attn_mask_sliding=_attn_mask_sliding,
                    token_mask=attention_mask,
                    repo_active=repo_active,
                    prores_alphas=prores_alphas,
                )
            s = self._drive_recurrence(e, s, r, K, _step_sdpa)

            P_count = self.config.recycling_num_prelude_layers
            R_count = len(self.recurrent_layers)
            # Inlined coda.
            for i, layer in enumerate(self.coda_layers):
                lt = layer.attention_type
                if lt == "full_attention":
                    _cos_sin = _rope_full
                    _attn_mask = _attn_mask_full
                else:
                    _cos_sin = _rope_sliding
                    _attn_mask = _attn_mask_sliding
                s = layer(
                    s,
                    attn_mask=_attn_mask,
                    cos_sin=_cos_sin,
                    position_ids=None,
                    token_mask=attention_mask,
                    repo_active=repo_active,
                    prores_alpha=prores_alphas[P_count + R_count + i],
                )
            x = s
        else:
            for i, layer in enumerate(self.layers):
                if self.resid_lambdas is not None:
                    x = self.resid_lambdas[i] * x
                if self.x0_lambdas is not None:
                    x = x + self.x0_lambdas[i] * x0
                alpha = prores_alphas[i]
                layer_type = layer.attention_type
                if (
                    self.config.activation_checkpointing
                    and self.training
                    and str(self.config.activation_checkpointing_mode).strip().lower()
                    == "layer"
                ):
                    attn_mask = attn_masks[layer_type]
                    cos_sin = rope[layer_type]
                    token_mask = attention_mask

                    def _layer_forward(
                        x_in: torch.Tensor,
                        *,
                        _layer: nn.Module = layer,
                        _attn_mask=attn_mask,
                        _cos_sin=cos_sin,
                        _token_mask=token_mask,
                        _repo_active: bool = repo_active,
                        _prores_alpha=alpha,
                    ) -> torch.Tensor:
                        return _layer(
                            x_in,
                            attn_mask=_attn_mask,
                            cos_sin=_cos_sin,
                            position_ids=None,
                            token_mask=_token_mask,
                            repo_active=_repo_active,
                            prores_alpha=_prores_alpha,
                        )

                    try:
                        x = _checkpoint(_layer_forward, x, use_reentrant=False)
                    except TypeError:  # older torch checkpoint API
                        x = _checkpoint(_layer_forward, x)
                else:
                    x = layer(
                        x,
                        attn_mask=attn_masks[layer_type],
                        cos_sin=rope[layer_type],
                        position_ids=None,
                        token_mask=attention_mask,
                        repo_active=repo_active,
                        prores_alpha=alpha,
                    )

        if self.config.use_mhc_lite:
            x = x[..., 0, :]  # compress: take stream 0
        return self.final_norm(x)


@dataclass
class MLMOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor

    def __getitem__(self, key: str):
        return getattr(self, key)


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.classifier_bias,
        )
        self.norm = _create_norm(
            config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
        )
        self.act = _get_activation(config.classifier_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(x)))


class ModernBertForMaskedLM(nn.Module):
    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

        self.sparse_prediction = config.sparse_prediction
        self.sparse_pred_ignore_index = config.sparse_pred_ignore_index

        self.init_weights()

        if config.tie_word_embeddings:
            self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    @torch.no_grad()
    def init_weights(self) -> None:

        width = self.config.hidden_size
        bound = math.sqrt(3.0 / width)
        # Residual-branch out-projections (attn.Wo, mlp.Wo) are configurable.
        # For MuonH, zero-init is invalid because ||W_init||_F defines a hard
        # hyperball radius and a zero-initialized matrix would be locked at zero.
        wo_init = self.config.wo_init
        mlp_out_bound = math.sqrt(3.0 / self.config.intermediate_size)
        # NOBLE half-Kaiming: halve init scale for layers that use Kaiming
        # (Wqkv, Wi).
        noble_kaiming = (
            self.config.use_noble and self.config.noble_half_kaiming
        )
        kaiming_bound = bound / 2 if noble_kaiming else bound

        def _noble_bound(mod: nn.Module) -> float:
            return kaiming_bound if isinstance(mod, NOBLELinear) else bound

        def _init_wo(tensor: torch.Tensor, init_bound: float) -> None:
            if wo_init == "zero":
                nn.init.zeros_(tensor)
            else:
                nn.init.uniform_(tensor, -init_bound, init_bound)

        embedding_std = 0.02 if self.config.tie_word_embeddings else 1.0

        nn.init.normal_(
            self.model.embeddings.tok_embeddings.weight,
            mean=0.0,
            std=embedding_std,
        )

        for module in self.modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        for layer in self.model.layers:
            if isinstance(layer, MHCLiteBlock):
                enc = layer.layer
                mhc_blocks = [layer]
            elif isinstance(layer, MHCLiteSublayersLayer):
                enc = layer.enc
                mhc_blocks = [
                    b for b in (layer.mhc_attn, layer.mhc_mlp) if b is not None
                ]
            else:
                enc = layer
                mhc_blocks = []
            # QKV weight init (NOBLE modules get half-Kaiming bound)
            if enc.attn.Wqkv is not None:
                b = _noble_bound(enc.attn.Wqkv)
                nn.init.uniform_(enc.attn.Wqkv.weight, -b, b)
            else:
                bq = _noble_bound(enc.attn.Wq)
                nn.init.uniform_(enc.attn.Wq.weight, -bq, bq)
                if enc.attn.Wq2 is not None:
                    bq2 = _noble_bound(enc.attn.Wq2)
                    nn.init.uniform_(enc.attn.Wq2.weight, -bq2, bq2)
                bk = _noble_bound(enc.attn.Wk)
                nn.init.uniform_(enc.attn.Wk.weight, -bk, bk)
                bv = _noble_bound(enc.attn.Wv)
                nn.init.uniform_(enc.attn.Wv.weight, -bv, bv)
            _init_wo(enc.attn.Wo.weight, bound)

            # MLP weight init (NOBLE modules get half-Kaiming bound)
            if enc.mlp is not None:
                is_moe_mlp = False
                if getattr(self.config, "use_moe", False):
                    from .moe import MoELayer as _MoELayer
                    is_moe_mlp = isinstance(enc.mlp, _MoELayer)

                if is_moe_mlp:
                    # Routed expert stacked weights (nn.Parameter, not nn.Linear).
                    nn.init.uniform_(enc.mlp.Wi, -bound, bound)
                    _init_wo(enc.mlp.Wo, mlp_out_bound)
                    nn.init.uniform_(enc.mlp.router.gate.weight, -bound, bound)
                    shared = enc.mlp.shared_expert
                    if shared.Wi is not None:
                        bi = _noble_bound(shared.Wi)
                        nn.init.uniform_(shared.Wi.weight, -bi, bi)
                    elif hasattr(shared, "W_up") and shared.W_up is not None:
                        bu = _noble_bound(shared.W_up)
                        nn.init.uniform_(shared.W_up.weight, -bu, bu)
                        bg = _noble_bound(shared.W_gate)
                        nn.init.uniform_(shared.W_gate.weight, -bg, bg)
                    _init_wo(shared.Wo.weight, mlp_out_bound)
                elif enc.mlp.Wi is not None:
                    bi = _noble_bound(enc.mlp.Wi)
                    nn.init.uniform_(enc.mlp.Wi.weight, -bi, bi)
                elif hasattr(enc.mlp, "W_up") and enc.mlp.W_up is not None:
                    bu = _noble_bound(enc.mlp.W_up)
                    nn.init.uniform_(enc.mlp.W_up.weight, -bu, bu)
                    bg = _noble_bound(enc.mlp.W_gate)
                    nn.init.uniform_(enc.mlp.W_gate.weight, -bg, bg)
                if not is_moe_mlp:
                    if hasattr(enc.mlp, "Wo"):
                        _init_wo(enc.mlp.Wo.weight, mlp_out_bound)
                    else:
                        _init_wo(enc.mlp.Wo_weight, mlp_out_bound)

            # QKV bias init
            if enc.attn.Wqkv is not None:
                if enc.attn.Wqkv.bias is not None:
                    nn.init.zeros_(enc.attn.Wqkv.bias)
            else:
                if enc.attn.Wq.bias is not None:
                    nn.init.zeros_(enc.attn.Wq.bias)
                if enc.attn.Wq2 is not None and enc.attn.Wq2.bias is not None:
                    nn.init.zeros_(enc.attn.Wq2.bias)
                if enc.attn.Wk.bias is not None:
                    nn.init.zeros_(enc.attn.Wk.bias)
                if enc.attn.Wv.bias is not None:
                    nn.init.zeros_(enc.attn.Wv.bias)
            if enc.attn.Wo.bias is not None:
                nn.init.zeros_(enc.attn.Wo.bias)

            # MLP bias init
            if enc.mlp is not None:
                is_moe_mlp = False
                if getattr(self.config, "use_moe", False):
                    from .moe import MoELayer as _MoELayer
                    is_moe_mlp = isinstance(enc.mlp, _MoELayer)

                if is_moe_mlp:
                    shared = enc.mlp.shared_expert
                    if shared.Wi is not None:
                        if shared.Wi.bias is not None:
                            nn.init.zeros_(shared.Wi.bias)
                    elif hasattr(shared, "W_up") and shared.W_up is not None:
                        if shared.W_up.bias is not None:
                            nn.init.zeros_(shared.W_up.bias)
                        if shared.W_gate.bias is not None:
                            nn.init.zeros_(shared.W_gate.bias)
                    if shared.Wo.bias is not None:
                        nn.init.zeros_(shared.Wo.bias)
                elif enc.mlp.Wi is not None:
                    if enc.mlp.Wi.bias is not None:
                        nn.init.zeros_(enc.mlp.Wi.bias)
                elif hasattr(enc.mlp, "W_up") and enc.mlp.W_up is not None:
                    if enc.mlp.W_up.bias is not None:
                        nn.init.zeros_(enc.mlp.W_up.bias)
                    if enc.mlp.W_gate.bias is not None:
                        nn.init.zeros_(enc.mlp.W_gate.bias)
                if not is_moe_mlp:
                    if hasattr(enc.mlp, "Wo"):
                        if enc.mlp.Wo.bias is not None:
                            nn.init.zeros_(enc.mlp.Wo.bias)
                    elif enc.mlp.Wo_bias is not None:
                        nn.init.zeros_(enc.mlp.Wo_bias)

            # DiffV2: zero-init lambda_proj so sigmoid(0)=0.5 at start.
            if enc.attn.lambda_proj is not None:
                nn.init.zeros_(enc.attn.lambda_proj.weight)

            # RePO: zero-init W_z so positions start at zero (NoPE-like).
            # W_g and W_c keep default Kaiming uniform init.
            if enc.attn.repo is not None:
                nn.init.zeros_(enc.attn.repo.W_z.weight)

            # mHC-lite: zero-init fused projection, set biases for identity behavior
            def _init_mhc_block(block: MHCLiteBlock) -> None:
                nn.init.zeros_(block.W_all.weight)
                n_s = block.n
                bias = block.W_all.bias.data
                # pre bias: first n values
                bias[:n_s].fill_(-1.0)
                bias[0] = 1.0
                # post bias: next n values
                bias[n_s : 2 * n_s].fill_(-1.0)
                bias[n_s] = 1.0
                # res bias: last n! values
                bias[2 * n_s :].fill_(-8.0)
                bias[2 * n_s + block._identity_idx] = 0.0
                block.alpha_pre.fill_(0.01)
                block.alpha_post.fill_(0.01)
                block.alpha_res.fill_(0.01)

            for block in mhc_blocks:
                _init_mhc_block(block)

        nn.init.uniform_(self.head.dense.weight, -bound, bound)
        if self.head.dense.bias is not None:
            nn.init.zeros_(self.head.dense.bias)

        decoder_std = embedding_std if self.config.tie_word_embeddings else 0.001
        nn.init.normal_(self.decoder.weight, mean=0.0, std=decoder_std)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        if self.model.resid_lambdas is not None:
            self.model.resid_lambdas.fill_(self.config.resid_lambda_init)
        if self.model.x0_lambdas is not None:
            self.model.x0_lambdas.fill_(self.config.x0_lambda_init)
        self.model._apply_takase_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embeddings.tok_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
        recycling_loop_count: Optional[int] = None,
        recycling_bptt: Optional[int] = None,
    ) -> MLMOutput:
        # ---- static packed path (pre-flattened by collator) ---------------
        # When both cu_seqlens and position_ids are provided and input_ids is
        # already 1-D (flat), the collator has done all unpadding / position-id
        # computation.  No data-dependent ops here → dynamic=False safe.
        if cu_seqlens is not None and position_ids is not None and input_ids.dim() == 1:
            if not _HAS_FLASH_VARLEN:
                raise RuntimeError(
                    "Sequence packing requires flash attention (flash_attn or "
                    "flash_attn_interface)."
                )
            x = self.model(
                input_ids,
                _cu_seqlens=cu_seqlens,
                _max_seqlen=max_seqlen,  # pass int directly — no tensor
                _position_ids=position_ids,
                recycling_loop_count=recycling_loop_count,
                recycling_bptt=recycling_bptt,
            )
            # x: (F, hidden) where F is fixed flat length

            if labels is not None:
                logits = self.decoder(self.head(x))
                loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    ignore_index=self.sparse_pred_ignore_index,
                )
                # MoE auxiliary losses (balance + z-loss, coefficients baked in).
                if self.config.use_moe:
                    from .moe import MoELayer as _MoELayer
                    aux_losses = [
                        module.last_aux_loss
                        for module in self.modules()
                        if isinstance(module, _MoELayer)
                        and module.last_aux_loss is not None
                    ]
                    if aux_losses:
                        loss = loss + sum(aux_losses) / len(aux_losses)
            else:
                logits = self.decoder(self.head(x))
                loss = None

            return MLMOutput(loss=loss, logits=logits)

        use_varlen = (
            _HAS_FLASH_VARLEN and input_ids.is_cuda and attention_mask is not None
        )

        # ---- varlen (flash-attention) path --------------------------------
        if use_varlen:
            batch, seq_len = input_ids.shape
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

            if cu_seqlens is not None:
                # Packed path: cu_seqlens provided by the packing collator.
                # max_seqlen may be an int or a tensor; keep consistent.
                if isinstance(max_seqlen, int):
                    max_seqlen_t = torch.tensor(max_seqlen, dtype=torch.int32)
                else:
                    max_seqlen_t = max_seqlen
            else:
                # Unpacked path: derive cu_seqlens from attention_mask.
                _indices, cu_seqlens, max_seqlen_t = _unpad_input(attention_mask)
                indices = _indices

            flat_ids = input_ids.view(-1)[indices]  # (total_tokens,)
            position_ids = _position_ids_from_cu_seqlens(
                cu_seqlens, flat_ids.shape[0], flat_ids.device
            )

            x = self.model(
                flat_ids,
                _cu_seqlens=cu_seqlens,
                _max_seqlen=max_seqlen_t,
                _position_ids=position_ids,
                recycling_loop_count=recycling_loop_count,
                recycling_bptt=recycling_bptt,
            )
            # x: (total_tokens, hidden) — flat, no padding

            if self.sparse_prediction and labels is not None:
                flat_labels = labels.view(-1)[indices]
                keep = flat_labels != self.sparse_pred_ignore_index
                logits = self.decoder(self.head(x[keep]))
                loss = F.cross_entropy(logits.float(), flat_labels[keep])
            elif labels is not None:
                flat_labels = labels.view(-1)[indices]
                logits = self.decoder(self.head(x))
                loss = F.cross_entropy(
                    logits.float(),
                    flat_labels,
                    ignore_index=self.sparse_pred_ignore_index,
                )
            else:
                logits = self.decoder(self.head(x))
                logits = _pad_output(logits, indices, batch, seq_len)
                loss = None

            return MLMOutput(loss=loss, logits=logits)

        # ---- SDPA (fallback) path -----------------------------------------
        x = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            recycling_loop_count=recycling_loop_count,
            recycling_bptt=recycling_bptt,
        )

        if self.sparse_prediction and labels is not None:
            flat_labels = labels.view(-1)
            x = x.view(flat_labels.shape[0], -1)
            keep = flat_labels != self.sparse_pred_ignore_index
            x = x[keep]
            flat_labels = flat_labels[keep]
        else:
            flat_labels = labels

        logits = self.decoder(self.head(x))

        loss = None
        if labels is not None:
            if self.sparse_prediction:
                loss = F.cross_entropy(logits.float(), flat_labels)
            else:
                loss = F.cross_entropy(
                    logits.float().view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=self.sparse_pred_ignore_index,
                )

        return MLMOutput(loss=loss, logits=logits)

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not only_trainable)
        )


def map_hf_state_dict_to_pure(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {k: v for k, v in hf_state_dict.items() if not k.startswith("_")}


def map_pure_state_dict_to_hf(
    pure_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return dict(pure_state_dict)
