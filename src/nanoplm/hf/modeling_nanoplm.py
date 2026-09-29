"""Hugging Face model classes for nanoPLM.

These are a shell, not a reimplementation: they build the same ModernBertModel /
ModernBertPredictionHead / decoder that PureProtModernBertMLM builds, under the
same attribute names. That makes state-dict keys identical between the native
model and this one, so export copies weights with no renaming table.
"""

from __future__ import annotations

from contextlib import contextmanager
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
    _init_modernbert_weights,
    _repo_inv_freq,
    _reset_moe_bookkeeping,
)
from nanoplm.pretraining.models.modern_bert.pure_model import modernbert_config_from


@contextmanager
def _collect_hidden_states(encoder: ModernBertModel, enabled: bool):
    """Capture per-layer outputs without touching the compiled forward.

    Threading an `output_hidden_states` flag through ModernBertModel.forward
    would add a branch to a torch.compile-guarded hot path for zero training
    benefit. Hooks are registered only when asked for and removed on exit
    (`finally`, so they come off even if the wrapped forward raises).

    Captured tensors are NOT detached: HF `hidden_states` are part of the
    autograd graph, and detaching would silently break gradients from any
    loss computed on an intermediate layer.

    Under recycling (`encoder.adapter is not None`) the recurrent block's
    layers fire once per iteration rather than once total, so entries would
    otherwise balloon with `recycling_loop_count`. Per the design spec, one
    entry is reported per recurrent ITERATION instead of per layer firing:
    the embeddings output, each prelude layer's output, the output of the
    recurrent block's LAST layer (`recurrent_layers[-1]`, which -- because
    each iteration inlines the recurrent layers in one call -- already fires
    exactly once per iteration) once per iteration, and each coda layer's
    output. Outside recycling, every layer in `encoder.layers` is hooked once.

    Under mHC-lite (`use_mhc_lite=True`) the embeddings entry is (B, S, C)
    but every layer entry is (B, S, n, C) with n = `config.mhc_n_streams`:
    mHC-lite carries multiple residual streams per layer, and that shape is
    reported as-is rather than reshaped/compressed away.
    """
    captured: list[torch.Tensor] = []
    if not enabled:
        yield captured
        return

    handles = []

    def _hook(_module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured.append(tensor)

    handles.append(encoder.embeddings.register_forward_hook(_hook))
    if encoder.adapter is not None:
        for layer in encoder.prelude_layers:
            handles.append(layer.register_forward_hook(_hook))
        handles.append(encoder.recurrent_layers[-1].register_forward_hook(_hook))
        for layer in encoder.coda_layers:
            handles.append(layer.register_forward_hook(_hook))
    else:
        for layer in encoder.layers:
            handles.append(layer.register_forward_hook(_hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class NanoPLMPreTrainedModel(PreTrainedModel):
    config_class = NanoPLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize parameters with nanoPLM's native scheme, and
        materialize the non-persistent runtime-state buffers.

        transformers 5.x's `initialize_weights()` walks the module tree
        children-first and calls this hook once per module not already
        flagged `_is_hf_initialized`, root last. For *custom* model code
        (anything outside the `transformers` package -- true here), it also
        SKIPS the call entirely for a module whose own direct parameters and
        buffers (`recurse=False`) are all already flagged, which is
        vacuously true for a module with none of either. `self` (this class)
        is exactly such a module: `NanoPLMModel` / `NanoPLMForMaskedLM` own
        no parameter or buffer directly -- everything lives under
        `.model` / `.head` / `.decoder` -- so hooking `module is self` would
        never actually fire. `root.model` (`ModernBertModel`) always carries
        the non-persistent `_prores_alphas` buffer directly (see below), so
        its visit is guaranteed and used as the trigger instead; `self`
        (bound to this method regardless of which `module` is being
        visited) gives it the outer shell to initialize.

        On direct construction nothing is flagged, so every parameter gets a
        fresh native-scheme draw. On `from_pretrained`, loaded params/buffers
        (and tied weights) are flagged *before* this runs, and transformers
        swaps in guarded `nn.init.*` functions that silently skip any
        flagged tensor -- so only the keys the checkpoint didn't supply
        (e.g. a missing MoE expert) actually get (re)initialized; every
        loaded key is left untouched. `_init_modernbert_weights`
        (modeling.py) is the one function both this hook and the native
        `ModernBertForMaskedLM.init_weights` call, and it additionally
        guards its few direct tensor writes (`.fill_`, slice assignment)
        that bypass that automatic guard.

        This hook also runs once after a direct (non-`from_pretrained`)
        construction, via `post_init()` -> `init_weights()`.

        Separately from parameter init, three buffer sites always need
        recomputing here: `ModernBertModel._prores_alphas`
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
        """
        if isinstance(module, ModernBertModel):
            _init_modernbert_weights(self, self._inner_config)
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

    # Every export is a NanoPLMForMaskedLM checkpoint, so loading it as the bare
    # encoder (as the model card's quickstart does) always leaves the MLM head's
    # weights over. That is expected; without this, transformers prints a LOAD
    # REPORT flagging each of them UNEXPECTED. Anchored, so an unexpected key
    # anywhere else is still reported.
    _keys_to_ignore_on_load_unexpected = [r"^head\.", r"^decoder\."]

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
        want_hidden = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        with _collect_hidden_states(self.model, want_hidden) as captured:
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return BaseModelOutput(
            last_hidden_state=hidden,
            hidden_states=tuple(captured) if want_hidden else None,
        )


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
        want_hidden = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        with _collect_hidden_states(self.model, want_hidden) as captured:
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.decoder(self.head(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1),
                ignore_index=self._inner_config.sparse_pred_ignore_index,
            )
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=tuple(captured) if want_hidden else None,
        )
