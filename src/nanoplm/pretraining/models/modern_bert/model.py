from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertForMaskedLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer


class SwiGLU(nn.Module):
    def forward(self, x, gate):
        return F.silu(gate) * x


class ModernBertMLPSwiGLU(nn.Module):
    """Replacement MLP that applies SwiGLU to each ModernBERT layer."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=config.mlp_bias)
        self.drop = nn.Dropout(config.mlp_dropout)
        self.act = SwiGLU()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states):
        x, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(x, gate)))

@dataclass
class ProtModernBertMLMConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int = 32
    mlp_activation: str = "swiglu"
    mlp_dropout: float = 0.0
    mlp_bias: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    classifier_activation: str = "gelu"
    use_resid_lambdas: bool = False
    use_x0_lambdas: bool = False
    resid_lambda_init: float = 1.0
    x0_lambda_init: float = 0.1

class ProtModernBertMLM(ModernBertForMaskedLM):

    def __init__(
        self,
        config: ProtModernBertMLMConfig
    ):
        self.tokenizer = ProtModernBertTokenizer()

        self.config = ModernBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            # Keep this comfortably above common dataset max_seq_len values.
            # RoPE frequencies are generated from this bound at runtime.
            max_position_embeddings=8192,
            mlp_dropout=config.mlp_dropout,
            mlp_bias=config.mlp_bias,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            classifier_activation=config.classifier_activation,
            # Set correct token IDs from our tokenizer
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=None,  # Not used in our tokenizer
            unk_token_id=self.tokenizer.unk_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            loss_type="ForMaskedLM",
        )

        super().__init__(self.config)
        # PreTrainedModel.__init__ auto-infers loss_type from class name via
        # regex against LOSS_MAPPING keys. "ProtModernBertMLM" doesn't contain
        # "ForMaskedLM", so it falls back to None → ForCausalLMLoss (which
        # shifts labels left by 1 — wrong for MLM). Override it here.
        self.loss_type = "ForMaskedLM"

        # Apply SwiGLU activation to MLP layers if specified
        if config.mlp_activation.lower() == "swiglu":
            for layer in self.model.layers:
                layer.mlp = ModernBertMLPSwiGLU(self.config)
