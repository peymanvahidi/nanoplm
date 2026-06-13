"""Rotary Position Embedding (RoPE) and RePO learned positions for ModernBERT.

Includes standard RoPE, paired-head staggered RoPE, and RePO
(Re-Positioning) — learned per-head continuous positions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModernBertConfig


# ── Core RoPE operations ───────────────────────────────────────────────────


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: (B, H, S, D) or (T, H, D), cos/sin: (1, S, rope_dim) or (T, rope_dim)
    # Head dim broadcasts — works for different Q/KV head counts.
    # Partial RoPE: when rope_dim < head_dim, only the first rope_dim dims are rotated.
    q_dtype = q.dtype
    k_dtype = k.dtype
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rope_dim = cos.shape[-1]

    if rope_dim < q.shape[-1]:
        q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        qf, kf = q_rot.float(), k_rot.float()
        q_rot = (qf * cos + _rotate_half(qf) * sin).to(dtype=q_dtype)
        k_rot = (kf * cos + _rotate_half(kf) * sin).to(dtype=k_dtype)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    # Full RoPE fast path (original)
    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


def _apply_paired_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply staggered RoPE to paired heads.

    q/k last dim is 2*D, where the first D channels belong to the first head in
    the pair and the second D channels belong to the second head in the pair.
    Each half is rotated independently with positions (2p) and (2p+1).

    Partial RoPE: cos/sin have last dim = 2*rope_dim (even + odd halves).
    When rope_dim < head_dim, only the first rope_dim dims of each head's half
    are rotated; the rest pass through.
    """

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    head_dim = q.shape[-1] // 2
    rope_dim = cos.shape[-1] // 2

    q1, q2 = q[..., :head_dim], q[..., head_dim:]
    k1, k2 = k[..., :head_dim], k[..., head_dim:]
    cos1, cos2 = cos[..., :rope_dim], cos[..., rope_dim:]
    sin1, sin2 = sin[..., :rope_dim], sin[..., rope_dim:]

    if rope_dim < head_dim:
        # Partial RoPE: split each head's dims into rotary and passthrough
        q1_rot, q1_pass = q1[..., :rope_dim], q1[..., rope_dim:]
        q2_rot, q2_pass = q2[..., :rope_dim], q2[..., rope_dim:]
        k1_rot, k1_pass = k1[..., :rope_dim], k1[..., rope_dim:]
        k2_rot, k2_pass = k2[..., :rope_dim], k2[..., rope_dim:]

        q1f, q2f = q1_rot.float(), q2_rot.float()
        k1f, k2f = k1_rot.float(), k2_rot.float()

        q1_rot = (q1f * cos1 + _rotate_half(q1f) * sin1).to(dtype=q1.dtype)
        q2_rot = (q2f * cos2 + _rotate_half(q2f) * sin2).to(dtype=q2.dtype)
        k1_rot = (k1f * cos1 + _rotate_half(k1f) * sin1).to(dtype=k1.dtype)
        k2_rot = (k2f * cos2 + _rotate_half(k2f) * sin2).to(dtype=k2.dtype)

        q1 = torch.cat([q1_rot, q1_pass], dim=-1)
        q2 = torch.cat([q2_rot, q2_pass], dim=-1)
        k1 = torch.cat([k1_rot, k1_pass], dim=-1)
        k2 = torch.cat([k2_rot, k2_pass], dim=-1)
        return torch.cat((q1, q2), dim=-1), torch.cat((k1, k2), dim=-1)

    # Full RoPE fast path (original)
    q1_dtype, q2_dtype = q1.dtype, q2.dtype
    k1_dtype, k2_dtype = k1.dtype, k2.dtype
    q1f, q2f = q1.float(), q2.float()
    k1f, k2f = k1.float(), k2.float()

    q1 = (q1f * cos1 + _rotate_half(q1f) * sin1).to(dtype=q1_dtype)
    q2 = (q2f * cos2 + _rotate_half(q2f) * sin2).to(dtype=q2_dtype)
    k1 = (k1f * cos1 + _rotate_half(k1f) * sin1).to(dtype=k1_dtype)
    k2 = (k2f * cos2 + _rotate_half(k2f) * sin2).to(dtype=k2_dtype)
    return torch.cat((q1, q2), dim=-1), torch.cat((k1, k2), dim=-1)


# ── RePO: learned per-head positions ──────────────────────────────────────


class RePOModule(nn.Module):
    """RePO (Re-Positioning): predicts continuous per-head positions from hidden states.

    Replaces fixed integer RoPE positions with learned content-dependent positions.
    Architecture: SwiGLU position representation + linear per-head position assignment.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, d_p: Optional[int] = None
    ):
        super().__init__()
        d_p = d_p or hidden_size // 8
        self.W_g = nn.Linear(hidden_size, d_p, bias=False)
        self.W_c = nn.Linear(hidden_size, d_p, bias=False)
        self.W_z = nn.Linear(d_p, num_heads, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., hidden_size) -> (..., num_heads) scalar positions per head."""
        r = F.silu(self.W_g(h)) * self.W_c(h)
        return self.W_z(r)


def _apply_rope_repo(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using learned per-head positions (varlen path).

    q, k:      (T, H, D)
    positions: (T, H)     – learned scalar position per head
    inv_freq:  (rope_dim//2,)

    Partial RoPE: when rope_dim < head_dim, only the first rope_dim dims
    of q/k are rotated.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    # (T, H, 1) * (1, 1, rope_dim//2) -> (T, H, rope_dim//2)
    freqs = positions.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, H, rope_dim)
    cos = emb.cos()
    sin = emb.sin()
    rope_dim = cos.shape[-1]

    if rope_dim < q.shape[-1]:
        q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        qf, kf = q_rot.float(), k_rot.float()
        q_rot = (qf * cos + _rotate_half(qf) * sin).to(dtype=q_dtype)
        k_rot = (kf * cos + _rotate_half(kf) * sin).to(dtype=k_dtype)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


def _apply_rope_repo_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using learned per-head positions (SDPA/padded path).

    q, k:      (B, H, S, D)
    positions: (B, S, H)    – learned scalar position per head
    inv_freq:  (rope_dim//2,)

    Partial RoPE: when rope_dim < head_dim, only the first rope_dim dims
    of q/k are rotated.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    # (B, S, H) -> (B, H, S, 1)
    pos = positions.permute(0, 2, 1).unsqueeze(-1).float()
    freqs = pos * inv_freq.view(1, 1, 1, -1)  # (B, H, S, rope_dim//2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (B, H, S, rope_dim)
    cos = emb.cos()
    sin = emb.sin()
    rope_dim = cos.shape[-1]

    if rope_dim < q.shape[-1]:
        q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        qf, kf = q_rot.float(), k_rot.float()
        q_rot = (qf * cos + _rotate_half(qf) * sin).to(dtype=q_dtype)
        k_rot = (kf * cos + _rotate_half(kf) * sin).to(dtype=k_dtype)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


# ── Rotary embedding module ────────────────────────────────────────────────


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            "inv_freq_full",
            self._build_inv_freq(config.global_rope_theta),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_sliding",
            self._build_inv_freq(config.local_rope_theta),
            persistent=False,
        )

    def _build_inv_freq(self, theta: float) -> torch.Tensor:
        rope_dim = self.config.rope_dim
        channel = torch.arange(0, rope_dim, 2, dtype=torch.float32)
        return 1.0 / (theta ** (channel / rope_dim))

    def _select_inv_freq(self, layer_type: str, device: torch.device) -> torch.Tensor:
        inv_freq = (
            self.inv_freq_full
            if layer_type == "full_attention"
            else self.inv_freq_sliding
        )
        return inv_freq.to(device=device)

    def paired_from_positions(
        self,
        positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._select_inv_freq(layer_type, device)
        pos = positions.to(device=device, dtype=torch.float32)
        even_freqs = torch.outer(2.0 * pos, inv_freq)
        odd_freqs = torch.outer(2.0 * pos + 1.0, inv_freq)
        even = torch.cat((even_freqs, even_freqs), dim=-1)
        odd = torch.cat((odd_freqs, odd_freqs), dim=-1)
        emb = torch.cat((even, odd), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def paired(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        cos, sin = self.paired_from_positions(positions, device, dtype, layer_type)
        return cos[None], sin[None]

    def forward(
        self,
        seq_len: int | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_type == "full_attention":
            inv_freq = self.inv_freq_full
        else:
            inv_freq = self.inv_freq_sliding

        inv_freq = inv_freq.to(device=device)
        if isinstance(seq_len, torch.Tensor):
            # Dynamic tensor lengths cannot be passed to torch.arange without
            # materializing a Python int. Slice from the model cap instead so
            # compiled varlen paths stay graph-friendly.
            positions = torch.arange(
                self.config.max_position_embeddings,
                device=device,
                dtype=torch.float32,
            )[:seq_len]
        else:
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None].to(dtype=dtype), emb.sin()[None].to(dtype=dtype)
