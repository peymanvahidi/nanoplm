import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    ModernBertModel,
    ModernBertConfig,
)

from .tokenizer import ProtXTokenizer

class ProtX(nn.Module):
    """Student model for ProtX"""

    def __init__(
        self,
        student_dim: int = 512,
        student_layers: int = 6,
        student_heads: int = 8,
        activation: str = "swiglu",
    ):
        super().__init__()

        self.tokenizer = ProtXTokenizer()

        self.student_config = ModernBertConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=student_dim,
            intermediate_size=student_dim * 2,
            num_hidden_layers=student_layers,
            num_attention_heads=student_heads,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            mlp_bias=False,
            attention_bias=False,
        )

        self.student = ModernBertModel(self.student_config)

        # swap in SwiGLU blocks if requested
        if activation.lower() == "swiglu":
            for layer in self.student.layers:
                layer.mlp = ModernBertMLPSwiGLU(self.student_config)

        # Project student 512‑>1024 to match teacher hidden size
        self.proj = nn.Linear(student_dim, 1024, bias=False)
        self.mse_loss = nn.MSELoss(reduction="none")

    # ---------------------------------------------------------------------
    # forward pass
    # ---------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, target_repr):
        """Return (loss, student_repr?) so it fits easily into training loops."""

        student_out = self.student(input_ids=input_ids, attention_mask=attention_mask)
        student_repr = self.proj(student_out.last_hidden_state)  # (B,L,1024)

        # compute MSE masked over non‑padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        # Manually calculate MSE to avoid issues with inference tensors
        diff = ((student_repr - target_repr) ** 2) * mask
        loss = diff.sum() / mask.sum().clamp(min=1)

        return loss

    # optional convenience method
    def save_student(self, path: str):
        """Save student weights + projection only."""
        torch.save(
            {
                "student_state_dict": self.student.state_dict(),
                "proj_state_dict": self.proj.state_dict(),
                "student_config": self.student_config.to_dict(),
            },
            path,
        )

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
