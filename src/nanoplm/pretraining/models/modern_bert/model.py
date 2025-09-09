from transformers import ModernBertConfig, ModernBertForMaskedLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

class ProtModernBertMLM(ModernBertForMaskedLM):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        vocab_size: int = 29,
        mlp_activation: str = "swiglu",
        mlp_dropout: float = 0.0,
        mlp_bias: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        classifier_activation: str = "gelu",
    ):
        self.tokenizer = ProtModernBertTokenizer()

        self.config = ModernBertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            mlp_dropout=mlp_dropout,
            mlp_bias=mlp_bias,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            classifier_activation=classifier_activation,
            # Set correct token IDs from our tokenizer
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=None,  # Not used in our tokenizer
            unk_token_id=self.tokenizer.unk_token_id,
            mask_token_id=self.tokenizer.mask_token_id
        )

        super().__init__(self.config)
