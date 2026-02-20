"""Pure PyTorch ModernBERT for masked language modeling.

The model is intentionally small and readable:
- pre-norm transformer blocks
- RoPE attention (full + sliding-window layers)
- GLU MLP (or SwiGLU replacement)
- explicit, centralized initialization
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_FLASH_VARLEN = False
_flash_varlen_fn = None
_FLASH_HAS_DROPOUT = False

try:
    # FA3 (Hopper / Blackwell)
    from kernels import get_kernel
    _flash_varlen_fn = get_kernel('varunneal/flash-attention-3').flash_attn_interface.flash_attn_varlen_func
    _HAS_FLASH_VARLEN = True
    _FLASH_HAS_DROPOUT = False  # FA3 removed dropout_p
except ImportError:
    try:
        # FA2 (Ampere+, RTX 30xx/40xx/50xx)
        from flash_attn import flash_attn_varlen_func as _flash_varlen_fn
        _HAS_FLASH_VARLEN = True
        _FLASH_HAS_DROPOUT = True  # FA2 supports dropout_p
    except ImportError:
        pass


@dataclass
class ModernBertConfig:
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
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
    resid_lambda_init: float = 1.0
    x0_lambda_init: float = 0.1

    head_dim: int = field(init=False)
    sliding_window: int = field(init=False)
    layer_types: list[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"{self.hidden_size} vs {self.num_attention_heads}"
            )

        attn_stride = max(1, int(self.global_attn_every_n_layers))
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.sliding_window = self.local_attention // 2
        self.mlp_activation = self.mlp_activation.lower()
        if self.mlp_activation not in {"swiglu", "glu"}:
            raise ValueError(
                f"Unsupported mlp_activation: {self.mlp_activation}. Supported: ['swiglu', 'glu']"
            )
        self.layer_types = [
            "full_attention" if i % attn_stride == 0 else "sliding_attention"
            for i in range(self.num_hidden_layers)
        ]


def _get_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
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
    # q/k: (B, H, S, D), cos/sin: (1, S, D)
    q_dtype = q.dtype
    k_dtype = k.dtype
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(x_proj) * gate))


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(self.drop(F.silu(gate) * x_proj))


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.use_qk_norm = config.use_qk_norm
        self.dropout = config.attention_dropout
        self.scale = self.head_dim ** -0.5

        self.Wqkv = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.attention_bias,
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

    # -- varlen (flash-attention) path -----------------------------------------

    def _forward_varlen(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        window_size: tuple[int, int],
    ) -> torch.Tensor:
        total = x.shape[0]  # (total_tokens, hidden)
        qkv = self.Wqkv(x).view(total, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # each: (total, H, D)

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
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            return self._forward_varlen(x, cos_sin, cu_seqlens, max_seqlen, window_size)

        bsz, seq_len, _ = x.shape
        qkv = self.Wqkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_drop(self.Wo(y))


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.attn_norm = (
            nn.Identity()
            if layer_idx == 0
            else nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        )
        self.attn = ModernBertAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = (
            ModernBertSwiGLUMLP(config)
            if config.mlp_activation == "swiglu"
            else ModernBertMLP(config)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            cos_sin=cos_sin,
            attn_mask=attn_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            window_size=window_size,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ModernBertModel(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
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
            x0 = x if self.x0_lambdas is not None else None

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

            for i, layer in enumerate(self.layers):
                if self.resid_lambdas is not None:
                    x = self.resid_lambdas[i] * x
                if self.x0_lambdas is not None:
                    x = x + self.x0_lambdas[i] * x0
                lt = layer.attention_type
                x = layer(
                    x,
                    cos_sin=rope[lt],
                    cu_seqlens=_cu_seqlens,
                    max_seqlen=_max_seqlen,
                    window_size=windows[lt],
                )

            return self.final_norm(x)

        # ---- SDPA (fallback) path -----------------------------------------
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}"
            )
        device = input_ids.device

        x = self.embeddings(input_ids)
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

        for i, layer in enumerate(self.layers):
            if self.resid_lambdas is not None:
                x = self.resid_lambdas[i] * x
            if self.x0_lambdas is not None:
                x = x + self.x0_lambdas[i] * x0
            layer_type = layer.attention_type
            x = layer(x, attn_mask=attn_masks[layer_type], cos_sin=rope[layer_type])

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
            nn.init.uniform_(layer.attn.Wqkv.weight, -bound, bound)
            nn.init.zeros_(layer.attn.Wo.weight)
            nn.init.uniform_(layer.mlp.Wi.weight, -bound, bound)
            nn.init.zeros_(layer.mlp.Wo.weight)

            if layer.attn.Wqkv.bias is not None:
                nn.init.zeros_(layer.attn.Wqkv.bias)
            if layer.attn.Wo.bias is not None:
                nn.init.zeros_(layer.attn.Wo.bias)
            if layer.mlp.Wi.bias is not None:
                nn.init.zeros_(layer.mlp.Wi.bias)
            if layer.mlp.Wo.bias is not None:
                nn.init.zeros_(layer.mlp.Wo.bias)

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
