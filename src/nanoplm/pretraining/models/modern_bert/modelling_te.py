"""Transformer Engine ModernBERT path for masked language modeling."""

from __future__ import annotations

import math
from typing import Optional

from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.pytorch import attention as te_attention
from transformer_engine.common.recipe import DelayedScaling, Format,NVFP4BlockScaling,Float8CurrentScaling,Float8BlockScaling

from nanoplm.pretraining.models.modern_bert.modeling import (
    ModernBertConfig,
    _get_activation,
    _unpad_input,
)

USE_FP8 = True
FULL_ATTN_EVERY_N_LAYER = 3

# FP8 recipe: delayed scaling with HYBRID format (E4M3 forward, E5M2 backward).
# amax_history_len=16 is a reasonable default for stability; increase for smoother scaling.
FP8_RECIPE = Float8CurrentScaling()

# TE requires window_size=(-1, -1) for full attention, not None.
_FULL_ATTN_WINDOW: tuple[int, int] = (-1, -1)


class TEModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.norm = te.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


class TEModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()

        if FULL_ATTN_EVERY_N_LAYER == 0:
            self.is_full_attention = True
        else:
            self.is_full_attention = layer_idx % FULL_ATTN_EVERY_N_LAYER == 0

        # TE requires (-1, -1) for full attention (not None).
        self.window_size: tuple[int, int] = (
            _FULL_ATTN_WINDOW
            if self.is_full_attention
            else (config.sliding_window, config.sliding_window)
        )

        bound = math.sqrt(3.0 / config.hidden_size)

        def _init(t: torch.Tensor) -> None:
            nn.init.uniform_(t, -bound, bound)

        def _output_init(t: torch.Tensor) -> None:
            nn.init.zeros_(t)

        # Fused LayerNorm + QKV projection + attention + output projection.
        # qkv_format='thd': flat (total_tokens, heads, dim) — required for varlen with
        # correct per-sequence RoPE position resets via cu_seqlens.
        # attn_mask_type='padding': required by TE when qkv_format='thd'.
        self.attn = te.MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            layernorm_epsilon=config.norm_eps,
            bias=config.attention_bias,
            normalization="LayerNorm",
            input_layernorm=layer_idx != 0,  # layer 0: embedding already normed
            attn_mask_type="padding",         # required for qkv_format=thd
            window_size=self.window_size,
            fuse_qkv_params=True,
            qkv_format="thd",
            return_layernorm_output=False,
            return_bias=False,
            init_method=_init,
            output_layer_init_method=_output_init,
        )

        # Fused LayerNorm + FC1 + SwiGLU + FC2.
        self.mlp = te.LayerNormMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            eps=config.norm_eps,
            bias=config.mlp_bias,
            normalization="LayerNorm",
            activation="swiglu",
            init_method=_init,
            output_layer_init_method=_output_init,
        )

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        # x: (total_tokens, hidden) — flat thd layout
        attn_out = self.attn(
            hidden_states=x,
            rotary_pos_emb=rotary_pos_emb,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            attn_mask_type="padding",
            window_size=self.window_size,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + attn_out

        mlp_out = self.mlp(x)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        return x + mlp_out


class TEModernBertModel(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = TEModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [TEModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = te.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self._rope_full_emb = te_attention.RotaryPositionEmbedding(
            config.head_dim, rotary_base=config.global_rope_theta
        )
        self._rope_sliding_emb = te_attention.RotaryPositionEmbedding(
            config.head_dim, rotary_base=config.local_rope_theta
        )
        self._max_pos = config.max_position_embeddings
        # Lazily cached on first forward (needs CUDA device).
        self._rope_full_freqs: Optional[torch.Tensor] = None
        self._rope_sliding_freqs: Optional[torch.Tensor] = None

    def _get_rope_freqs(self, max_seqlen: int):
        """Return cached RoPE frequency tensors, computing on first call."""
        if self._rope_full_freqs is None:
            self._rope_full_freqs = self._rope_full_emb(self._max_pos)
            self._rope_sliding_freqs = self._rope_sliding_emb(self._max_pos)
        return self._rope_full_freqs[:max_seqlen], self._rope_sliding_freqs[:max_seqlen]

    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Always runs in thd (flat/varlen) mode.

        input_ids: (total_tokens,) — already unpadded by TEModernBertForMaskedLM.
        cu_seqlens: cumulative sequence lengths (batch+1,) int32.
        max_seqlen: length of the longest sequence in the batch.
        """
        x = self.embeddings(input_ids)  # (total_tokens, hidden)

        # FP8 GEMMs require total_tokens divisible by 16 (forward needs %8, wgrad needs %16).
        # Pad x with zeros and extend cu_seqlens with a dummy entry for the pad tokens.
        real_total = x.shape[0]
        pad = (-real_total) % 16
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))  # (real_total + pad, hidden)
            cu_seqlens = torch.cat([
                cu_seqlens,
                cu_seqlens[-1:] + pad,
            ])

        rope_full_freqs, rope_sliding_freqs = self._get_rope_freqs(max_seqlen)

        for layer in self.layers:
            rope = rope_full_freqs if layer.is_full_attention else rope_sliding_freqs
            x = layer(x, rotary_pos_emb=rope, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Trim padding before final norm and return.
        if pad > 0:
            x = x[:real_total]

        return self.final_norm(x)


class TEModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.classifier_bias,
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.act = _get_activation(config.classifier_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(x)))


class TEModernBertForMaskedLM(nn.Module):
    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.model = TEModernBertModel(config)
        self.head = TEModernBertPredictionHead(config)
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

        nn.init.normal_(self.model.embeddings.tok_embeddings.weight, mean=0.0, std=embedding_std)

        for module in self.modules():
            if module.__class__.__name__ in {"LayerNorm"}:
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

        nn.init.uniform_(self.head.dense.weight, -bound, bound)
        if self.head.dense.bias is not None:
            nn.init.zeros_(self.head.dense.bias)

        decoder_std = embedding_std if self.config.tie_word_embeddings else 0.001
        nn.init.normal_(self.decoder.weight, mean=0.0, std=decoder_std)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

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
    ) -> dict[str, Optional[torch.Tensor]]:
        # Always unpad to flat (total_tokens,) before the encoder.
        indices = None  # set when we need to index into flattened input_ids/labels
        if cu_seqlens is not None and attention_mask is None:
            # Static packed input: already flat (F,) with cu_seqlens provided.
            # No attention_mask means all tokens are valid (padding is explicit
            # in cu_seqlens as zero-length sequences at the end).
            flat_ids = input_ids.view(-1)
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        elif attention_mask is not None:
            if cu_seqlens is not None:
                # Packed input: cu_seqlens already provided by the packing collator.
                indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
                if max_seqlen is None:
                    max_seqlen = int(attention_mask.sum(dim=-1).max().item())
            else:
                # Unpacked padded input: derive cu_seqlens from attention_mask.
                indices, cu_seqlens, max_seqlen_t = _unpad_input(attention_mask)
                max_seqlen = int(max_seqlen_t.item())
            flat_ids = input_ids.view(-1)[indices]
        else:
            # No mask, no cu_seqlens: treat each row as a full sequence.
            total = input_ids.numel()
            flat_ids = input_ids.view(-1)
            cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=input_ids.device)
            max_seqlen = input_ids.shape[-1]
        x = self.model(flat_ids, cu_seqlens=cu_seqlens, max_seqlen=int(max_seqlen))

        if self.sparse_prediction and labels is not None:
            flat_labels = labels.view(-1)[indices] if indices is not None else labels.view(-1)
            keep = flat_labels != self.sparse_pred_ignore_index
            logits = self.decoder(self.head(x[keep]))
            loss = F.cross_entropy(logits.float(), flat_labels[keep])
        elif labels is not None:
            flat_labels = labels.view(-1)[indices] if indices is not None else labels.view(-1)
            logits = self.decoder(self.head(x))
            loss = F.cross_entropy(
                logits.float(),
                flat_labels,
                ignore_index=self.sparse_pred_ignore_index,
            )
        else:
            logits = self.decoder(self.head(x))
            loss = None

        return {"loss": loss, "logits": logits}

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not only_trainable)
        )
