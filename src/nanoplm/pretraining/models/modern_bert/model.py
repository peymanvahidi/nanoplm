import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import torch.nn.functional as F

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    ModernBertModel,
    ModernBertConfig,
)
from transformers.modeling_outputs import MaskedLMOutput


class ModernBertForMaskedLMConfig(PretrainedConfig):
    """
    Configuration for ModernBertForMaskedLM.

    Exposes key hyperparameters for convenient control from code:
    - hidden_size (embedding dimension)
    - intermediate_size (MLP size)
    - num_hidden_layers
    - num_attention_heads
    - vocab_size

    Also carries tokenizer special token ids.
    """

    model_type = "modernbert_mlm"

    def __init__(
        self,
        vocab_size: int = 29,
        hidden_size: int = 512,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 1024,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        unk_token_id: int = 2,
        mask_token_id: int = 3,
        mlp_activation: str = "swiglu",
        attention_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        mlp_bias: bool = False,
        attention_bias: bool = False,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.unk_token_id = unk_token_id
        self.mask_token_id = mask_token_id

        self.mlp_activation = mlp_activation
        self.attention_dropout = attention_dropout
        self.mlp_dropout = mlp_dropout
        self.mlp_bias = mlp_bias
        self.attention_bias = attention_bias


class ModernBertForMaskedLM(PreTrainedModel):
    """
    ModernBERT backbone with a Masked-LM head.

    This mirrors HuggingFace's ForMaskedLM pattern and allows configuring
    heads, layers, embedding size, and intermediate size via config.
    """

    config_class = ModernBertForMaskedLMConfig

    def __init__(self, config: ModernBertForMaskedLMConfig):
        super().__init__(config)
        self.config = config

        # Build underlying ModernBERT config from our higher-level config
        backbone_config = ModernBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            attention_dropout=config.attention_dropout,
            mlp_dropout=config.mlp_dropout,
            mlp_bias=config.mlp_bias,
            attention_bias=config.attention_bias,
        )

        self.backbone = ModernBertModel(backbone_config)

        # Optional: switch MLP to SwiGLU like student model if requested
        if getattr(config, "mlp_activation", "").lower() == "swiglu":
            try:
                for layer in self.backbone.layers:
                    layer.mlp = ModernBertMLPSwiGLU(backbone_config)
            except Exception:
                pass

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and tie if requested
        self.post_init()
        if getattr(config, "tie_word_embeddings", True):
            self.tie_weights()

    # Weight tying helpers
    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.embeddings.tok_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.embeddings.tok_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Ignore index -100 positions
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_modernbert_mlm(
    *,
    vocab_size: int = 29,
    hidden_size: int = 512,
    intermediate_size: int = 1024,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 8,
    max_position_embeddings: int = 1024,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    unk_token_id: int = 2,
    mask_token_id: int = 3,
    mlp_activation: str = "swiglu",
    attention_dropout: float = 0.0,
    mlp_dropout: float = 0.0,
    mlp_bias: bool = False,
    attention_bias: bool = False,
) -> ModernBertForMaskedLM:
    """Factory helper to build a configurable ModernBertForMaskedLM instance."""

    config = ModernBertForMaskedLMConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        unk_token_id=unk_token_id,
        mask_token_id=mask_token_id,
        mlp_activation=mlp_activation,
        attention_dropout=attention_dropout,
        mlp_dropout=mlp_dropout,
        mlp_bias=mlp_bias,
        attention_bias=attention_bias,
    )

    return ModernBertForMaskedLM(config)


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
