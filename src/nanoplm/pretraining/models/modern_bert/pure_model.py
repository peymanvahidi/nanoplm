"""
Pure-torch protein ModernBERT MLM wrapper.

This is the pure-torch counterpart to ``model.py`` (which wraps HF's
``ModernBertForMaskedLM``).  It wraps the pure-torch
``ModernBertForMaskedLM`` from ``modeling.py`` instead.

The existing HF-based ``ProtModernBertMLM`` in ``model.py`` is left
completely untouched.
"""

from dataclasses import dataclass

from nanoplm.pretraining.models.modern_bert.modeling import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertSwiGLUMLP,
)
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig


class PureProtModernBertMLM(ModernBertForMaskedLM):
    """Pure-torch ``ProtModernBertMLM`` (no HF ``transformers`` dependency)."""

    def __init__(self, config: ProtModernBertMLMConfig):
        self.tokenizer = ProtModernBertTokenizer()

        mb_config = ModernBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=1024,  # hardcoded (matches HF wrapper)
            mlp_dropout=config.mlp_dropout,
            mlp_bias=config.mlp_bias,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            classifier_activation=config.classifier_activation,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=None,
            unk_token_id=self.tokenizer.unk_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
        )

        super().__init__(mb_config)

        # Apply SwiGLU activation to MLP layers if specified
        # (matches the monkey-patching in ProtModernBertMLM)
        if config.mlp_activation.lower() == "swiglu":
            for layer in self.model.layers:
                layer.mlp = ModernBertSwiGLUMLP(mb_config)
