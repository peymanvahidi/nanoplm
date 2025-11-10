from dataclasses import dataclass

from transformers import ModernBertConfig, ModernBertForMaskedLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

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
            max_position_embeddings=1024, # this is hardcoded
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
            mask_token_id=self.tokenizer.mask_token_id
        )

        super().__init__(self.config)
