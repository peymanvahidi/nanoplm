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
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

# Registers torch.ops.nanoplm_mhc::* used by MHCLiteBlock's Triton path.
from . import mhc_triton_ops as _mhc_triton_ops  # noqa: F401

_HAS_FLASH_VARLEN = False
_flash_varlen_fn = None
_FLASH_HAS_DROPOUT = False
USE_ACTIVATION_CHECKPOINTING_CANON = True
# mHC-lite selective recompute (paper-inspired):
# checkpoint only mHC pre/post kernels, keep heavy layer function outside.
USE_ACTIVATION_CHECKPOINTING_MHC = False

if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
    try:
        # FA3 (H100 / sm90)
        from kernels import get_kernel

        _fa3 = get_kernel("varunneal/flash-attention-3")
        _fa3 = getattr(_fa3, "flash_attn_interface", _fa3)
        _flash_varlen_fn = _fa3.flash_attn_varlen_func
        _HAS_FLASH_VARLEN = True
        _FLASH_HAS_DROPOUT = False  # FA3 removed dropout_p
    except ImportError:
        pass

USE_TRITON_SRELU = False

if torch.cuda.is_available() and (torch.cuda.get_device_capability() == (9, 0) or torch.cuda.get_device_capability() == (12, 0)):
    USE_TRITON_SRELU = True


if not _HAS_FLASH_VARLEN:
    try:
        # FA2 (Ampere+, RTX 30xx/40xx/50xx)
        from flash_attn import flash_attn_varlen_func as _flash_varlen_fn

        _HAS_FLASH_VARLEN = True
        _FLASH_HAS_DROPOUT = True  # FA2 supports dropout_p
    except ImportError:
        pass


def _parse_canon_layers_mode(mode: str) -> frozenset[str]:
    if not isinstance(mode, str):
        raise ValueError(f"canon_layers_mode must be a string, got {type(mode).__name__}")
    normalized = mode.strip().lower()
    if normalized in {"", "none", "off"}:
        return frozenset()

    allowed = {"a", "b", "c", "d"}
    separators = {" ", "+", "-", "_", "/", "|", ","}
    selected: set[str] = set()
    for char in normalized:
        if char in separators:
            continue
        if char not in allowed:
            raise ValueError(
                f"Invalid canon_layers_mode={mode!r}. "
                "Use a subset of ABCD (e.g., 'abcd', 'ac', 'bcd')."
            )
        selected.add(char)
    return frozenset(selected)


def _resolve_canon_kernel_size(
    canon_layers_kernel_size: Optional[int],
) -> int:
    allowed = frozenset({3, 5, 7})
    if canon_layers_kernel_size is None:
        return 5
    if isinstance(canon_layers_kernel_size, bool) or not isinstance(canon_layers_kernel_size, int):
        raise ValueError(
            "canon_layers_kernel_size must be an integer or null/None "
            f"(auto default). Got {canon_layers_kernel_size!r}."
        )
    if canon_layers_kernel_size not in allowed:
        allowed_str = ", ".join(str(v) for v in sorted(allowed))
        raise ValueError(
            "Invalid canon_layers_kernel_size="
            f"{canon_layers_kernel_size}. Allowed values: {allowed_str}."
        )
    return canon_layers_kernel_size


@dataclass
class ModernBertConfig:
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    num_kv_heads: Optional[int] = None  # GQA: None means MHA (num_kv_heads = num_attention_heads)
    mlp_activation: str = "swiglu"
    hidden_activation: str = "gelu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0
    norm_eps: float = 1e-5
    norm_bias: bool = False
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: Optional[int] = None
    unk_token_id: int = 2
    mask_token_id: int = 3
    attention_bias: bool = False
    attention_dropout: float = 0.0
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    embedding_dropout: float = 0.0
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    no_mlp_on_first_layer: bool = True
    decoder_bias: bool = True
    classifier_bias: bool = False
    classifier_activation: str = "gelu"
    sparse_prediction: bool = False
    sparse_pred_ignore_index: int = -100
    tie_word_embeddings: bool = True
    global_rope_theta: float = 160_000.0
    local_rope_theta: float = 10_000.0
    use_resid_lambdas: bool = False
    use_x0_lambdas: bool = False
    use_qk_norm: bool = False
    use_canon_layers: bool = False
    canon_layers_mode: str = "abcd"
    canon_layers_kernel_size: Optional[int] = None
    resid_lambda_init: float = 1.0
    x0_lambda_init: float = 0.1
    use_repo: bool = False
    repo_after_n_layers: int = 3
    use_prores: bool = False
    prores_T: int = 1000
    gradient_checkpointing: bool = False
    gradient_checkpointing_mode: Literal["layer", "attn", "attn+mlp"] = "layer"
    use_mhc_lite: bool = False
    mhc_n_streams: int = 4
    mhc_triton_fused: bool = False
    mhc_lite_wrapping_level: Literal["layer", "sublayers"] = "layer"
    use_diff_attn_v2: bool = False
    attn_layer_pattern: Optional[str] = None

    head_dim: int = field(init=False)
    sliding_window: int = field(init=False)
    layer_types: list[str] = field(init=False)
    canon_layer_set: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.mhc_lite_wrapping_level = str(self.mhc_lite_wrapping_level).strip().lower()  # type: ignore[assignment]
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"{self.hidden_size} vs {self.num_attention_heads}"
            )
        # GQA: resolve num_kv_heads (None = MHA, i.e. same as num_attention_heads).
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_kv_heads for GQA: "
                f"{self.num_attention_heads} % {self.num_kv_heads} != 0"
            )
        if self.use_diff_attn_v2 and (2 * self.num_attention_heads) % self.num_kv_heads != 0:
            raise ValueError(
                "With DiffV2, 2*num_attention_heads must be divisible by num_kv_heads: "
                f"2*{self.num_attention_heads} % {self.num_kv_heads} != 0"
            )
        if self.mhc_lite_wrapping_level not in {"layer", "sublayers"}:
            raise ValueError(
                "mhc_lite_wrapping_level must be one of {'layer', 'sublayers'}, "
                f"got {self.mhc_lite_wrapping_level!r}"
            )
        if not self.use_mhc_lite and self.mhc_lite_wrapping_level != "layer":
            raise ValueError(
                "mhc_lite_wrapping_level != 'layer' requires use_mhc_lite=true "
                "(to avoid a silent no-op configuration)."
            )
        if self.use_mhc_lite and self.use_resid_lambdas:
            raise ValueError(
                "use_mhc_lite=true is not compatible with use_resid_lambdas=true. "
                "resid_lambdas scales the hidden state before each layer, which breaks "
                "mHC-lite's stability guarantees (doubly-stochastic mixing)."
            )
        attn_stride = max(1, int(self.global_attn_every_n_layers))
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.sliding_window = self.local_attention // 2
        self.mlp_activation = self.mlp_activation.lower()
        if self.mlp_activation not in {"swiglu", "glu", "srelu"}:
            raise ValueError(
                f"Unsupported mlp_activation: {self.mlp_activation}. Supported: ['swiglu', 'glu', 'srelu']"
            )
        self.canon_layers_kernel_size = _resolve_canon_kernel_size(
            self.canon_layers_kernel_size,
        )
        self.canon_layer_set = _parse_canon_layers_mode(self.canon_layers_mode)
        if not self.use_canon_layers:
            self.canon_layer_set = frozenset()
        elif not self.canon_layer_set:
            raise ValueError(
                "use_canon_layers=True requires non-empty canon_layers_mode "
                "(for example: 'abcd' or 'ac')."
            )
        # Build layer_types from explicit pattern or stride.
        if self.attn_layer_pattern is not None:
            pattern = self.attn_layer_pattern.upper().strip()
            if not pattern:
                raise ValueError("attn_layer_pattern must not be empty when provided.")
            _map = {"F": "full_attention", "S": "sliding_attention"}
            for ch in pattern:
                if ch not in _map:
                    raise ValueError(
                        f"Invalid character '{ch}' in attn_layer_pattern. "
                        "Use 'F' for full attention and 'S' for sliding attention."
                    )
            # Tile pattern to cover all layers.
            self.layer_types = [
                _map[pattern[i % len(pattern)]]
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                "full_attention" if i % attn_stride == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.use_diff_attn_v2 and self.use_repo:
            raise ValueError(
                "use_diff_attn_v2 is not compatible with use_repo. "
                "Differential attention V2 changes Q/K head counts which is "
                "incompatible with RePO's per-head position prediction."
            )
        if self.use_repo and self.num_kv_heads != self.num_attention_heads:
            raise ValueError(
                "GQA (num_kv_heads != num_attention_heads) is not compatible with use_repo. "
                "RePO predicts per-head positions for Q and K jointly, which requires "
                "equal Q/K head counts."
            )


def _get_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "srelu":
        return lambda x: F.relu(x).square()
    if name == "silu":
        return F.silu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: (B, H, S, D) or (T, H, D), cos/sin: (1, S, D) or (T, D)
    # Head dim broadcasts — works for different Q/KV head counts.
    q_dtype = q.dtype
    k_dtype = k.dtype
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


@torch.compile
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


class RePOModule(nn.Module):
    """RePO (Re-Positioning): predicts continuous per-head positions from hidden states.

    Replaces fixed integer RoPE positions with learned content-dependent positions.
    Architecture: SwiGLU position representation + linear per-head position assignment.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, d_p: Optional[int] = None):
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
    inv_freq:  (D//2,)
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    # (T, H, 1) * (1, 1, D//2) -> (T, H, D//2)
    freqs = positions.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, H, D)
    cos = emb.cos()
    sin = emb.sin()
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
    inv_freq:  (D//2,)
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    # (B, S, H) -> (B, H, S, 1)
    pos = positions.permute(0, 2, 1).unsqueeze(-1).float()
    freqs = pos * inv_freq.view(1, 1, 1, -1)  # (B, H, S, D//2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (B, H, S, D)
    cos = emb.cos()
    sin = emb.sin()
    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


def _full_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
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


def _make_canon_layer(
    channels: int, config: ModernBertConfig
) -> nn.Module:
    """Factory: returns a symmetric (bidirectional) Canon layer."""
    if config.canon_layers_kernel_size is None:
        raise ValueError("canon_layers_kernel_size was not resolved in ModernBertConfig.__post_init__")
    return ModernBertCanonLayer(channels, kernel_size=config.canon_layers_kernel_size)


def _canon_accum_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return x.dtype


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
            mixed = F.conv1d(
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
            ).squeeze(0).T.to(dtype=x.dtype)
            return x + mixed

        if position_ids is not None and position_ids.shape[0] == T:
            seq_start = (position_ids == 0).to(dtype=cu_seqlens.dtype)
            seq_start = seq_start.clone()
            seq_start[0] = 1
            seq_id = torch.cumsum(seq_start, dim=0) - 1
        else:
            positions = torch.arange(T, device=x.device, dtype=cu_seqlens.dtype)
            seq_id = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        weight = self.conv.weight[:, 0, :]
        bias = self.conv.bias
        mixed = _varlen_canon_inner(x, seq_id, weight, bias, self.radius)
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

    def forward(self, x, position_ids=None, cu_seqlens=None, attention_mask=None):
        # Checkpoint at the module boundary so all Canon insertion sites (A/B/C/D)
        # and both varlen/padded paths are covered by the same flag.
        use_ckpt = (
            USE_ACTIVATION_CHECKPOINTING_CANON
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
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
            return self._forward_varlen(x, cu_seqlens=cu_seqlens, position_ids=position_ids)
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


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


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
        channel = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32)
        return 1.0 / (theta ** (channel / self.config.head_dim))

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_type == "full_attention":
            inv_freq = self.inv_freq_full
        else:
            inv_freq = self.inv_freq_sliding

        inv_freq = inv_freq.to(device=device)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None].to(dtype=dtype), emb.sin()[None].to(dtype=dtype)


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.Wo = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
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
        wi = self.Wi(x)
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
            nn.Parameter(torch.zeros(config.hidden_size))
            if config.mlp_bias
            else None
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
            from nanoplm.pretraining.models.modern_bert.triton_kernels import FusedLinearReLUSquare
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
        self.Wi = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.Wo = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
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
        wi = self.Wi(x)
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
        self.scale = self.head_dim ** -0.5
        self.use_diff_attn_v2 = config.use_diff_attn_v2

        # GQA: num_kv_heads <= num_heads. MHA when num_kv_heads == num_heads.
        self.num_kv_heads = config.num_kv_heads

        if self.use_diff_attn_v2:
            # DiffV2 doubles Q heads on top of whatever GQA config is set.
            # Each original head splits into a pair for differential subtraction.
            self.num_q_heads = 2 * self.num_heads
            # Lambda: per-token, per-head scalar controlling subtraction weight.
            self.lambda_proj = nn.Linear(
                config.hidden_size, self.num_heads, bias=False,
            )
        else:
            self.num_q_heads = self.num_heads
            self.lambda_proj = None

        qkv_dim = (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
        self.Wqkv = nn.Linear(
            config.hidden_size, qkv_dim, bias=config.attention_bias,
        )

        self.Wo = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )
        qkv_out_dim = (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
        self.canon_b = (
            _make_canon_layer(qkv_out_dim, config)
            if "b" in config.canon_layer_set
            else None
        )

        # RePO: learned per-head positions replacing fixed RoPE indices
        self.repo = None
        if config.use_repo and layer_idx >= config.repo_after_n_layers:
            self.repo = RePOModule(
                config.hidden_size, config.num_attention_heads, config.head_dim,
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
        qkv = self.Wqkv(x)
        if self.canon_b is not None:
            qkv = self.canon_b(qkv, position_ids=position_ids, cu_seqlens=cu_seqlens)

        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(total, self.num_q_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)

        if self.repo is not None and repo_active:
            positions = self.repo(x)  # (T, num_heads)
            q, k = _apply_rope_repo(q, k, positions, self.repo_inv_freq)
        else:
            cos, sin = cos_sin
            q, k = _apply_rope(q, k, cos, sin)
        if self.use_qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

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
        qkv = self.Wqkv(x)
        if self.canon_b is not None:
            qkv = self.canon_b(qkv, attention_mask=token_mask)

        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(bsz, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.repo is not None and repo_active:
            positions = self.repo(x)  # (B, S, num_heads)
            q, k = _apply_rope_repo_sdpa(q, k, positions, self.repo_inv_freq)
        else:
            cos, sin = cos_sin
            q, k = _apply_rope(q, k, cos, sin)
        if self.use_qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=(self.dropout if self.training else 0.0),
            scale=self.scale,
            enable_gqa=(self.num_q_heads != self.num_kv_heads),
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
        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", False))
        self.gradient_checkpointing_mode = str(
            getattr(config, "gradient_checkpointing_mode", "layer")
        ).strip().lower()
        self.attn_norm = (
            nn.Identity()
            if layer_idx == 0
            else nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        )
        self.canon_a = (
            _make_canon_layer(config.hidden_size, config)
            if "a" in config.canon_layer_set
            else None
        )
        self.attn = ModernBertAttention(config, layer_idx=layer_idx)
        if self.has_mlp:
            self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
            self.canon_c = (
                _make_canon_layer(config.hidden_size, config)
                if "c" in config.canon_layer_set
                else None
            )
            if config.mlp_activation == "srelu":
                self.mlp = ModernBertSReluMLP(config)
            elif config.mlp_activation == "swiglu":
                self.mlp = ModernBertSwiGLUMLP(config)
            else:
                self.mlp = ModernBertMLP(config)
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
            self.gradient_checkpointing
            and self.training
            and self.gradient_checkpointing_mode in {"attn", "attn+mlp"}
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

            def _attn_branch(
                x_in: torch.Tensor,
                *,
                _layer: "ModernBertEncoderLayer" = self,
            ) -> torch.Tensor:
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
                self.gradient_checkpointing
                and self.training
                and self.gradient_checkpointing_mode == "attn+mlp"
            )
            if do_ckpt_mlp:
                _position_ids = position_ids
                _cu_seqlens = cu_seqlens
                _token_mask = token_mask

                def _mlp_branch(
                    x_in: torch.Tensor,
                    *,
                    _layer: "ModernBertEncoderLayer" = self,
                ) -> torch.Tensor:
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

    def __init__(self, n_streams: int, hidden_size: int, layer: nn.Module,
                 triton_fused: bool = False):
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
        pre_proj, post_proj, res_proj = all_proj.split(
            [n, n, self.n_fact], dim=-1
        )

        bias = self.W_all.bias.to(dt)
        pre_bias = bias[:n]
        post_bias = bias[n:2 * n]
        res_bias = bias[2 * n:]

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
        return (
            torch.matmul(H_merged, x_streams)
            + h_post.unsqueeze(-1) * layer_output.unsqueeze(-2)
        )

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

        pre_proj, post_proj, res_proj = all_proj.split(
            [n, n, self.n_fact], dim=-1
        )

        bias = self.W_all.bias.to(dt)
        h_pre = torch.sigmoid(self.alpha_pre.to(dt) * pre_proj + bias[:n])
        h_post = 2.0 * torch.sigmoid(self.alpha_post.to(dt) * post_proj + bias[n:2*n])
        a_res = F.softmax(self.alpha_res.to(dt) * res_proj + bias[2*n:], dim=-1)

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
                    [MHCLiteSublayersLayer(config, i) for i in range(config.num_hidden_layers)]
                )
        else:
            self.layers = nn.ModuleList(
                [ModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)]
            )
        if config.use_resid_lambdas:
            self.resid_lambdas = nn.Parameter(torch.ones(config.num_hidden_layers))
        else:
            self.register_parameter("resid_lambdas", None)
        if config.use_x0_lambdas:
            self.x0_lambdas = nn.Parameter(torch.zeros(config.num_hidden_layers))
        else:
            self.register_parameter("x0_lambdas", None)
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
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
        vals = [min(step / (T * (l + 1)), 1.0) for l in range(self.config.num_hidden_layers)]
        self._prores_alphas.copy_(torch.tensor(vals, dtype=self._prores_alphas.dtype))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        _cu_seqlens: Optional[torch.Tensor] = None,
        _max_seqlen: Optional[int] = None,
        _position_ids: Optional[torch.Tensor] = None,
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

            # Pre-compute RoPE tables up to max_position_embeddings (fixed size
            # avoids graph breaks / recompilation) and index by _position_ids.
            rope_len = self.config.max_position_embeddings
            cos_f, sin_f = self.rotary_emb(
                rope_len, device, x.dtype, "full_attention"
            )
            cos_s, sin_s = self.rotary_emb(
                rope_len, device, x.dtype, "sliding_attention"
            )
            rope = {
                "full_attention": (
                    cos_f[0, _position_ids],
                    sin_f[0, _position_ids],
                ),
                "sliding_attention": (
                    cos_s[0, _position_ids],
                    sin_s[0, _position_ids],
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

            for i, layer in enumerate(self.layers):
                if self.resid_lambdas is not None:
                    x = self.resid_lambdas[i] * x
                if self.x0_lambdas is not None:
                    x = x + self.x0_lambdas[i] * x0
                alpha = prores_alphas[i]
                lt = layer.attention_type
                if (
                    self.config.gradient_checkpointing
                    and self.training
                    and str(self.config.gradient_checkpointing_mode).strip().lower() == "layer"
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
            "full_attention": self.rotary_emb(
                seq_len=seq_len,
                device=device,
                dtype=x.dtype,
                layer_type="full_attention",
            ),
            "sliding_attention": self.rotary_emb(
                seq_len=seq_len,
                device=device,
                dtype=x.dtype,
                layer_type="sliding_attention",
            ),
        }

        repo_active = self.repo_active
        prores_alphas = self._prores_alphas  # (num_layers,) tensor buffer

        for i, layer in enumerate(self.layers):
            if self.resid_lambdas is not None:
                x = self.resid_lambdas[i] * x
            if self.x0_lambdas is not None:
                x = x + self.x0_lambdas[i] * x0
            alpha = prores_alphas[i]
            layer_type = layer.attention_type
            if (
                self.config.gradient_checkpointing
                and self.training
                and str(self.config.gradient_checkpointing_mode).strip().lower() == "layer"
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


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.classifier_bias,
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
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
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = config.sparse_prediction
        self.sparse_pred_ignore_index = config.sparse_pred_ignore_index

        self.init_weights()

        if config.tie_word_embeddings:
            self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    @torch.no_grad()
    def init_weights(self) -> None:
        
        width = self.config.hidden_size
        bound = math.sqrt(3.0 / width)
        embedding_std = 0.02 if self.config.tie_word_embeddings else 1.0

        nn.init.normal_(
            self.model.embeddings.tok_embeddings.weight,
            mean=0.0,
            std=embedding_std,
        )

        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for layer in self.model.layers:
            if isinstance(layer, MHCLiteBlock):
                enc = layer.layer
                mhc_blocks = [layer]
            elif isinstance(layer, MHCLiteSublayersLayer):
                enc = layer.enc
                mhc_blocks = [b for b in (layer.mhc_attn, layer.mhc_mlp) if b is not None]
            else:
                enc = layer
                mhc_blocks = []
            nn.init.uniform_(enc.attn.Wqkv.weight, -bound, bound)
            nn.init.zeros_(enc.attn.Wo.weight)
            if enc.mlp is not None:
                nn.init.uniform_(enc.mlp.Wi.weight, -bound, bound)
                if hasattr(enc.mlp, "Wo"):
                    nn.init.zeros_(enc.mlp.Wo.weight)
                else:
                    nn.init.zeros_(enc.mlp.Wo_weight)

            if enc.attn.Wqkv.bias is not None:
                nn.init.zeros_(enc.attn.Wqkv.bias)
            if enc.attn.Wo.bias is not None:
                nn.init.zeros_(enc.attn.Wo.bias)
            if enc.mlp is not None:
                if enc.mlp.Wi.bias is not None:
                    nn.init.zeros_(enc.mlp.Wi.bias)
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
                bias[n_s:2 * n_s].fill_(-1.0)
                bias[n_s] = 1.0
                # res bias: last n! values
                bias[2 * n_s:].fill_(-8.0)
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
    ) -> dict[str, Optional[torch.Tensor]]:
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
            )
            # x: (F, hidden) where F is fixed flat length

            if labels is not None:
                logits = self.decoder(self.head(x))
                loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    ignore_index=self.sparse_pred_ignore_index,
                )
            else:
                logits = self.decoder(self.head(x))
                loss = None

            return {"loss": loss, "logits": logits}

        use_varlen = (
            _HAS_FLASH_VARLEN
            and input_ids.is_cuda
            and attention_mask is not None
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

            return {"loss": loss, "logits": logits}

        # ---- SDPA (fallback) path -----------------------------------------
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)

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

        return {"loss": loss, "logits": logits}

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not only_trainable)
        )


def map_hf_state_dict_to_pure(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in hf_state_dict.items() if not k.startswith("_")}


def map_pure_state_dict_to_hf(pure_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return dict(pure_state_dict)
