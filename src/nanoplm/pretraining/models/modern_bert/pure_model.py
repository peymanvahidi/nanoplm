"""
Pure-torch protein ModernBERT MLM wrapper.

This is the pure-torch counterpart to ``model.py`` (which wraps HF's
``ModernBertForMaskedLM``).  It wraps the pure-torch
``ModernBertForMaskedLM`` from ``modeling.py`` instead.

The existing HF-based ``ProtModernBertMLM`` in ``model.py`` is left
completely untouched.
"""

from nanoplm.pretraining.models.modern_bert.modeling import (
    ModernBertConfig,
    ModernBertForMaskedLM,
)
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig

_TE_IMPORT_ERROR = None
try:
    from nanoplm.pretraining.models.modern_bert.modelling_te import TEModernBertForMaskedLM
except Exception as exc:  # pragma: no cover - depends on TE availability
    _TE_IMPORT_ERROR = exc

    class TEModernBertForMaskedLM:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "Transformer Engine model requested but unavailable. "
                "Install `transformer-engine` to use --pure-te."
            ) from _TE_IMPORT_ERROR


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
            mlp_activation=config.mlp_activation,
            # Keep this comfortably above common dataset max_seq_len values.
            # Position embeddings are RoPE frequencies (not learned tables), so a
            # larger cap avoids runtime index asserts with minimal overhead.
            max_position_embeddings=8192,
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
            use_resid_lambdas=config.use_resid_lambdas,
            use_x0_lambdas=config.use_x0_lambdas,
            resid_lambda_init=config.resid_lambda_init,
            x0_lambda_init=config.x0_lambda_init,
        )

        super().__init__(mb_config)


class TEProtModernBertMLM(TEModernBertForMaskedLM):
    """Transformer-Engine ``ProtModernBertMLM`` wrapper."""

    def __init__(self, config: ProtModernBertMLMConfig):
        self.tokenizer = ProtModernBertTokenizer()

        mb_config = ModernBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            mlp_activation=config.mlp_activation,
            # Keep this comfortably above common dataset max_seq_len values.
            max_position_embeddings=8192,
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
            use_resid_lambdas=config.use_resid_lambdas,
            use_x0_lambdas=config.use_x0_lambdas,
            resid_lambda_init=config.resid_lambda_init,
            x0_lambda_init=config.x0_lambda_init,
        )

        super().__init__(mb_config)
