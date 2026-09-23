"""Hugging Face model classes for nanoPLM.

These are a shell, not a reimplementation: they build the same ModernBertModel /
ModernBertPredictionHead / decoder that PureProtModernBertMLM builds, under the
same attribute names. That makes state-dict keys identical between the native
model and this one, so export copies weights with no renaming table.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

from nanoplm.hf.configuration_nanoplm import NanoPLMConfig
from nanoplm.pretraining.models.modern_bert.modeling import (
    ModernBertAttention,
    ModernBertModel,
    ModernBertPredictionHead,
    ModernBertRotaryEmbedding,
    _repo_inv_freq,
    _reset_moe_bookkeeping,
)
from nanoplm.pretraining.models.modern_bert.pure_model import modernbert_config_from


class NanoPLMPreTrainedModel(PreTrainedModel):
    config_class = NanoPLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        """Materialize the non-persistent runtime-state buffers.

        transformers 5.x's `from_pretrained` builds the model on the meta
        device, loads the checkpoint, then recreates every *non-persistent*
        buffer with `torch.empty_like` (uninitialized memory) and calls this
        hook to fill them back in (`_move_missing_keys_from_meta_to_device` /
        `_initialize_missing_keys` in transformers' modeling_utils.py). This
        hook also runs once after a direct (non-`from_pretrained`)
        construction, via `post_init()` -> `init_weights()`.

        Three buffer sites are affected: `ModernBertModel._prores_alphas`
        (every model), `ModernBertRotaryEmbedding.inv_freq_full` /
        `.inv_freq_sliding` (every model -- the standard RoPE tables), and
        `ModernBertAttention.repo_inv_freq` (RePO layers only). All three need
        recomputing here -- the first two from the frozen `NanoPLMConfig`
        state (`use_prores` / `prores_step`, `use_repo` / `repo_active`)
        rather than whatever a training loop last wrote; the RoPE tables are
        config-derived constants with no training-time state at all. A fourth
        site, `MoELayer`'s four dispatch-workspace buffers (`moe.py`), is
        deliberately NOT handled here: they are scratch space the native model
        also constructs via plain `torch.empty` and fully overwrites before
        any read (see `moe.py`'s `_ensure_dispatch_buffers`), so garbage after
        reload is exactly as harmless as garbage after a fresh `torch.empty`
        call in a directly constructed model.

        Everything else -- every `nn.Linear` / `nn.Embedding` weight, every
        norm -- is either loaded from the checkpoint or already has a fine
        default init from construction; this function never writes to a
        `Parameter`.
        """
        if isinstance(module, ModernBertModel):
            if self.config.use_prores:
                module.update_prores_alphas(int(self.config.prores_step))
            else:
                module._prores_alphas.fill_(1.0)
            if self.config.use_repo:
                module.repo_active = bool(self.config.repo_active)
        elif isinstance(module, ModernBertAttention) and module.repo is not None:
            module.repo_inv_freq.copy_(
                _repo_inv_freq(module._repo_theta, module._repo_rope_dim).to(
                    device=module.repo_inv_freq.device,
                    dtype=module.repo_inv_freq.dtype,
                )
            )
        elif isinstance(module, ModernBertRotaryEmbedding):
            module.inv_freq_full.copy_(
                module._build_inv_freq(module.config.global_rope_theta).to(
                    device=module.inv_freq_full.device,
                    dtype=module.inv_freq_full.dtype,
                )
            )
            module.inv_freq_sliding.copy_(
                module._build_inv_freq(module.config.local_rope_theta).to(
                    device=module.inv_freq_sliding.device,
                    dtype=module.inv_freq_sliding.dtype,
                )
            )


class NanoPLMModel(NanoPLMPreTrainedModel):
    """Bare encoder. `AutoModel` resolves here."""

    def __init__(self, config: NanoPLMConfig):
        super().__init__(config)
        self._inner_config = modernbert_config_from(config.to_model_config())
        self.model = ModernBertModel(self._inner_config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embeddings.tok_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embeddings.tok_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput:
        _reset_moe_bookkeeping(self.model, self._inner_config.use_moe)
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=None)


class NanoPLMForMaskedLM(NanoPLMPreTrainedModel):
    """MLM head. `AutoModelForMaskedLM` resolves here."""

    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: NanoPLMConfig):
        super().__init__(config)
        inner = modernbert_config_from(config.to_model_config())
        self._inner_config = inner
        self.model = ModernBertModel(inner)
        self.head = ModernBertPredictionHead(inner)
        self.decoder = nn.Linear(
            inner.hidden_size, inner.vocab_size, bias=inner.decoder_bias
        )
        if inner.tie_word_embeddings:
            self.decoder.weight = self.model.embeddings.tok_embeddings.weight
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embeddings.tok_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embeddings.tok_embeddings = value

    def get_output_embeddings(self) -> nn.Module:
        return self.decoder

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.decoder = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MaskedLMOutput:
        _reset_moe_bookkeeping(self.model, self._inner_config.use_moe)
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.decoder(self.head(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1),
                ignore_index=self._inner_config.sparse_pred_ignore_index,
            )
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=None)
