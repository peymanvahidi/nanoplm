import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    ModernBertModel,
    ModernBertConfig,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.t5.modeling_t5 import T5LayerNorm

from .tokenizer import ProtXTokenizer

class ProtX(nn.Module):
    """Student model for ProtX"""

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_activation: str = "swiglu",
    ):
        super().__init__()

        self.tokenizer = ProtXTokenizer()

        self.config = ModernBertConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=embed_dim,
            intermediate_size=embed_dim * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            mlp_bias=False,
            attention_bias=False,
        )

        self.model = ModernBertModel(self.config)

        if mlp_activation.lower() == "swiglu":
            for layer in self.model.layers:
                layer.mlp = ModernBertMLPSwiGLU(self.config)

        self.proj = nn.Linear(embed_dim, 1024, bias=False)
        self.proj_norm = T5LayerNorm(1024)

    def forward(self, input_ids, attention_mask, teacher_embeddings=None):
        student_out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        student_repr = student_out.last_hidden_state
        
        loss = None

        if teacher_embeddings is not None:
            projected_repr = self.proj(student_repr)  # (batch_size, seq_len, 1024)
            projected_repr = self.proj_norm(projected_repr)
            
            mask = attention_mask.unsqueeze(-1).float()
            teacher_embeddings = teacher_embeddings.to(projected_repr.device)
            diff = ((projected_repr - teacher_embeddings) ** 2) * mask
            loss = diff.sum() / mask.sum().clamp(min=1)
        
            return MaskedLMOutput(
                loss=loss,
                logits=student_repr,
                # hidden_states=student_out.hidden_states,
                # attentions=student_out.attentions
            )
        
        else:
            # During inference, no loss is calculated
            return student_out

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
