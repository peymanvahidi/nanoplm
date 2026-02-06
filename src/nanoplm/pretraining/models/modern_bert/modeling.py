"""
Pure-torch reimplementation of ModernBERT for masked language modeling.

This module replicates the HuggingFace `ModernBertForMaskedLM` architecture
in pure PyTorch (no `transformers` dependency). Given identical weights and
inputs, the forward pass produces numerically identical outputs.

Architecture highlights (ModernBERT vs classic BERT):
- Pre-norm (LayerNorm before attention/MLP, not after)
- Rotary Position Embeddings (RoPE) instead of learned absolute positions
- GLU MLP (Gated Linear Unit with GELU activation)
- Hybrid attention: alternating full and sliding-window layers

Reference: HF transformers/models/modernbert/modeling_modernbert.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModernBertConfig:
    """Configuration for the pure-torch ModernBERT model."""

    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
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

    # Derived (computed in __post_init__)
    head_dim: int = field(init=False)
    sliding_window: int = field(init=False)
    layer_types: list[str] = field(init=False)
    rope_theta: dict[str, float] = field(init=False)

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.sliding_window = self.local_attention // 2
        self.layer_types = [
            "sliding_attention" if bool(i % self.global_attn_every_n_layers) else "full_attention"
            for i in range(self.num_hidden_layers)
        ]
        self.rope_theta = {
            "full_attention": self.global_rope_theta,
            "sliding_attention": self.local_rope_theta,
        }


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_ACT_FN = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _get_act(name: str) -> nn.Module:
    if name not in _ACT_FN:
        raise ValueError(f"Unsupported activation: {name!r}")
    return _ACT_FN[name]()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class ModernBertEmbeddings(nn.Module):
    """Token embeddings → LayerNorm → Dropout. No positional encoding here."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id,
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------

class ModernBertRotaryEmbedding(nn.Module):
    """
    Computes RoPE cos/sin tables per layer type (full vs sliding).

    Each layer type has its own theta and thus its own inv_freq.
    Matches HF's ``ModernBertRotaryEmbedding`` for the *default* rope_type.
    """

    def __init__(self, config: ModernBertConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.unique_layer_types = list(set(config.layer_types))

        for layer_type in self.unique_layer_types:
            theta = config.rope_theta[layer_type]
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float, device=device) / config.head_dim)
            )
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return ``(cos, sin)`` of shape ``(B, S, head_dim)`` cast to ``x.dtype``.

        Internally, RoPE freqs are computed with shape ``(B, S, head_dim//2)``
        and concatenated to ``(B, S, head_dim)`` before applying ``cos``/``sin``.
        """
        inv_freq: torch.Tensor = getattr(self, f"{layer_type}_inv_freq")

        # inv_freq: (head_dim//2,) → (1, head_dim//2, 1)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        # position_ids: (B, S) → (B, 1, S)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 for RoPE computation
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # (B, S, head_dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, S, head_dim)
        cos = emb.cos()  # attention_scaling = 1.0 for default rope_type
        sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# RoPE application
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors. Matches HF exactly (float32 intermediates)."""
    original_dtype = q.dtype
    cos = cos.unsqueeze(unsqueeze_dim)  # (B, 1, S, head_dim) for (B, H, S, D) inputs
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(original_dtype), k_embed.to(original_dtype)


# ---------------------------------------------------------------------------
# Attention mask construction
# ---------------------------------------------------------------------------

def create_full_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Build the mask for full (bidirectional) attention layers.

    For SDPA: returns a bool mask where ``True`` = attend, ``False`` = ignore.
    If there is no padding (all 1s in attention_mask, or mask is None), returns
    ``None`` — SDPA handles unmasked full attention natively.
    """
    if attention_mask is None:
        return None

    # attention_mask: (B, S) — 1 for real, 0 for pad
    # If no token is masked, skip materialization (allows flash-attn kernel path)
    if attention_mask.all():
        return None

    # Expand to 4-D: (B, 1, 1, S) — broadcasts over heads and query positions
    return attention_mask[:, None, None, :].bool()


def create_sliding_window_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    sliding_window: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the mask for sliding-window (bidirectional) attention layers.

    Returns a bool mask of shape ``(1, 1, S, S)`` (or ``(B, 1, S, S)`` when padding
    is present) where ``True`` = attend.

    The window condition is ``|q_idx - kv_idx| <= sliding_window``  (inclusive).
    This matches HF's ``sliding_window_bidirectional_overlay``.
    """
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)   # (S, 1)
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
    window_mask = (q_idx - kv_idx).abs() <= sliding_window       # (S, S)
    window_mask = window_mask.unsqueeze(0).unsqueeze(0)          # (1, 1, S, S)

    if attention_mask is not None and not attention_mask.all():
        # Combine with padding: (B, 1, 1, S)
        pad_mask = attention_mask[:, None, None, :].bool()
        window_mask = window_mask & pad_mask  # (B, 1, S, S)

    return window_mask


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class ModernBertMLP(nn.Module):
    """GLU MLP: Wi projects to 2×intermediate then splits into input + gate."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = _get_act(config.hidden_activation)
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class ModernBertAttention(nn.Module):
    """
    Multi-head self-attention using PyTorch SDPA.

    Supports both full and sliding-window attention via the pre-computed mask.
    """

    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim ** -0.5

        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()

        # Sliding window: used by flash-attn path; for SDPA the mask is pre-computed.
        if config.layer_types[layer_idx] == "sliding_attention":
            self.sliding_window = config.sliding_window + 1  # +1 for inclusive bounds (matches HF)
        else:
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        qkv = self.Wqkv(hidden_states)                              # (B, S, 3*H)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)     # (B, S, 3, nh, hd)
        q, k, v = qkv.unbind(dim=2)                                 # each (B, S, nh, hd)

        q = q.transpose(1, 2)  # (B, nh, S, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        dropout_p = self.attention_dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=self.scaling,
        )  # (B, nh, S, hd)

        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1).contiguous()
        attn_output = self.out_drop(self.Wo(attn_output))
        return attn_output


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------

class ModernBertEncoderLayer(nn.Module):
    """Pre-norm encoder layer with residual connections."""

    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Layer 0 has Identity instead of LayerNorm for attn_norm (matches HF)
        if layer_idx == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

        self.attn = ModernBertAttention(config=config, layer_idx=layer_idx)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        attn_output = self.attn(
            self.attn_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Encoder (backbone)
# ---------------------------------------------------------------------------

class ModernBertModel(nn.Module):
    """
    The core ModernBERT encoder: embeddings → N encoder layers → final norm.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertEncoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.rotary_emb = ModernBertRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S) token IDs.
            attention_mask: (B, S) with 1 for real tokens and 0 for padding.

        Returns:
            Hidden states of shape (B, S, hidden_size).
        """
        B, S = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(S, device=device).unsqueeze(0)  # (1, S)

        hidden_states = self.embeddings(input_ids)

        # Build attention masks per layer type
        attention_mask_mapping: dict[str, Optional[torch.Tensor]] = {
            "full_attention": create_full_attention_mask(attention_mask, B, S, device),
            "sliding_attention": create_sliding_window_attention_mask(
                attention_mask, B, S, self.config.sliding_window, device,
            ),
        }

        # Pre-compute RoPE per layer type
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask_mapping[layer.attention_type],
                position_embeddings=position_embeddings[layer.attention_type],
            )

        hidden_states = self.final_norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Prediction head
# ---------------------------------------------------------------------------

class ModernBertPredictionHead(nn.Module):
    """Dense → activation → LayerNorm (used before the decoder projection)."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.classifier_bias)
        self.act = _get_act(config.classifier_activation)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


# ---------------------------------------------------------------------------
# MLM model
# ---------------------------------------------------------------------------

class ModernBertForMaskedLM(nn.Module):
    """
    ModernBERT with a masked-language-model head.

    Combines the backbone, prediction head, and output decoder (with optional
    weight tying to the input embeddings).
    """

    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config

        # 1. Create backbone
        self.model = ModernBertModel(config)

        # 2. First init pass: backbone only.
        #    Replicates HF's ModernBertModel.post_init() which runs
        #    apply(_init_weights) on the backbone, then marks every module
        #    with _is_hf_initialized so the second pass won't re-init them.
        self.model.apply(self._init_weights)
        for m in self.model.modules():
            m._is_hf_initialized = True

        # 3. Create head and decoder (their default nn.Linear.__init__
        #    kaiming_uniform_ runs at the same RNG state as in HF, because
        #    the backbone init advanced the RNG identically).
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = config.sparse_prediction
        self.sparse_pred_ignore_index = config.sparse_pred_ignore_index

        # 4. Second init pass: only head, decoder, and self are initialized
        #    (backbone modules are skipped via _is_hf_initialized flag).
        #    This matches HF's ModernBertForMaskedLM.post_init().
        self.apply(self._initialize_weights)

        # Weight tying (after init so decoder gets its own init, then tied)
        if config.tie_word_embeddings:
            self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embeddings.tok_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.decoder

    # ----- weight initialization (matches HF _init_weights exactly) --------

    def _initialize_weights(self, module: nn.Module) -> None:
        """Wrapper matching HF's ``_initialize_weights``: skip modules already
        initialized in a prior ``post_init`` pass (marked with ``_is_hf_initialized``)."""
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def _init_weight(mod: nn.Module, std: float) -> None:
            nn.init.trunc_normal_(
                mod.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                nn.init.zeros_(mod.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size ** -0.5,
        }

        if isinstance(module, ModernBertEmbeddings):
            _init_weight(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, ModernBertMLP):
            _init_weight(module.Wi, stds["in"])
            _init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertAttention):
            _init_weight(module.Wqkv, stds["in"])
            _init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertPredictionHead):
            _init_weight(module.dense, stds["out"])
        elif isinstance(module, ModernBertForMaskedLM):
            _init_weight(module.decoder, stds["out"])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, ModernBertRotaryEmbedding):
            # Re-compute inv_freq (matches HF's _init_weights for RoPE)
            for layer_type in module.unique_layer_types:
                theta = module.config.rope_theta[layer_type]
                buf = getattr(module, f"{layer_type}_inv_freq")
                inv_freq = 1.0 / (
                    theta ** (
                        torch.arange(
                            0,
                            module.config.head_dim,
                            2,
                            dtype=torch.float,
                            device=buf.device,
                        ) / module.config.head_dim
                    )
                )
                buf.copy_(inv_freq)
                getattr(module, f"{layer_type}_original_inv_freq").copy_(inv_freq)

    # ----- forward ----------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, S) token IDs.
            attention_mask: (B, S) with 1 for real tokens, 0 for padding.
            labels: (B, S) target token IDs for MLM loss. Positions with
                ``-100`` are ignored by the cross-entropy loss.

        Returns:
            Dictionary with keys ``"loss"`` (scalar or None) and ``"logits"``
            (B, S, vocab_size) or (N_masked, vocab_size) if sparse_prediction.
        """
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.sparse_prediction and labels is not None:
            # Flatten and filter to masked positions only
            labels_flat = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels_flat.shape[0], -1)
            mask_tokens = labels_flat != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels_flat = labels_flat[mask_tokens]
        else:
            labels_flat = labels

        logits = self.decoder(self.head(last_hidden_state))

        loss = None
        if labels is not None:
            # Upcast logits to float32 for loss computation (matches HF ForMaskedLMLoss)
            if self.sparse_prediction:
                loss = F.cross_entropy(logits.float(), labels_flat)
            else:
                loss = F.cross_entropy(
                    logits.float().view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=self.sparse_pred_ignore_index,
                )

        return {"loss": loss, "logits": logits}

    # ----- utility: state dict mapping from/to HF -------------------------

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Return total number of (trainable) parameters."""
        return sum(p.numel() for p in self.parameters() if not only_trainable or p.requires_grad)


# ---------------------------------------------------------------------------
# Weight mapping utilities (HF ↔ pure-torch)
# ---------------------------------------------------------------------------

def map_hf_state_dict_to_pure(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Map a HuggingFace ``ModernBertForMaskedLM`` state dict to the pure-torch
    naming convention.  The architectures are intentionally named identically,
    so this is an identity mapping for all keys.
    """
    # The module hierarchy is the same:
    #   HF: model.embeddings.tok_embeddings.weight  →  model.embeddings.tok_embeddings.weight
    #   HF: model.layers.0.attn.Wqkv.weight         →  model.layers.0.attn.Wqkv.weight
    #   HF: decoder.weight                           →  decoder.weight
    #   etc.
    #
    # The only difference is that HF wraps things in PreTrainedModel which adds
    # some metadata keys.  We simply filter to known parameter names.
    return {k: v for k, v in hf_state_dict.items() if not k.startswith("_")}


def map_pure_state_dict_to_hf(pure_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Map a pure-torch ``ModernBertForMaskedLM`` state dict back to HuggingFace
    naming.  Both naming conventions are identical by design.
    """
    return dict(pure_state_dict)
