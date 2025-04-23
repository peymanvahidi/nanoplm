import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    ModernBertModel,
    ModernBertConfig,
    PreTrainedTokenizer,
)

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab=None, unk_token="<unk>", pad_token="<pad>", eos_token="</s>"):
        
        # Define vocabulary mapping amino acids & special tokens
        self.vocab = {
            "A": 3, "L": 4, "G": 5, "V": 6, "S": 7, "R": 8, "E": 9, "D": 10,
            "T": 11, "I": 12, "P": 13, "K": 14, "F": 15, "Q": 16, "N": 17,
            "Y": 18, "M": 19, "H": 20, "W": 21, "C": 22, "X": 23, "B": 24,
            "O": 25, "U": 26, "Z": 27, pad_token: 0, eos_token: 1, unk_token: 2
        }

        # Reverse vocabulary for decoding
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Set special tokens explicitly
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token


        # Initialize parent class properly
        super().__init__(
            unk_token=unk_token, 
            pad_token=pad_token, 
            eos_token=eos_token,
            vocab =vocab
        )

    def get_vocab(self):
        """ Returns the vocabulary dictionary. """
        return self.vocab 

    def _tokenize(self, text):
        return list(text)  
    
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))  
    
    def _convert_id_to_token(self, index):
        # Reverse the vocab to convert id back to token
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return reverse_vocab.get(index, self.unk_token)

    def save_vocabulary(self, save_directory):
        # Optionally save your vocab to a file
        pass

    def encode(self, text, add_special_tokens=True):
        # Optionally add special tokens like [PAD], [EOS]
        return [self.convert_token_to_id(token) for token in self._tokenize(text)]
    
    def decode(self, token_ids, skip_special_tokens=False):
        # Optionally decode the token IDs back to string
        return ''.join([self.convert_id_to_token(id) for id in token_ids])


# ---------------------------------------------------------------------------
# Student MLP block with SwiGLU (optional)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Distillation wrapper
# ---------------------------------------------------------------------------

class DistilProtT5(nn.Module):
    """Distil *token‑level* representations from ProtT5 into a 512‑d ModernBERT student."""

    def __init__(
        self,
        teacher_name: str = "Rostlab/prot_t5_xl_uniref50",
        student_dim: int = 512,
        student_layers: int = 6,
        student_heads: int = 8,
        intermediate_size: int | None = None,
        activation: str = "swiglu",
    ):
        super().__init__()

        # ---------------- teacher (frozen) ----------------
        self.teacher = T5EncoderModel.from_pretrained(teacher_name)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(teacher_name, do_lower_case=False, use_fast=False)
        teacher_dim = self.teacher.config.d_model  # 1024 for ProtT5

        # ---------------- student ----------------
        if intermediate_size is None:
            intermediate_size = student_dim * 4

        self.student_config = ModernBertConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=student_dim,
            intermediate_size=intermediate_size,
            num_hidden_layers=student_layers,
            num_attention_heads=student_heads,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id or 0,
            max_position_embeddings=4096,
            attention_dropout=0.1,
            mlp_dropout=0.1,
            mlp_bias=True,
            attention_bias=True,
        )

        self.student = ModernBertModel(self.student_config)

        # swap in SwiGLU blocks if requested
        if activation.lower() == "swiglu":
            for layer in self.student.layers:
                layer.mlp = ModernBertMLPSwiGLU(self.student_config)

        # Project student 512‑>1024 to match teacher hidden size
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

        self._mse = nn.MSELoss(reduction="none")

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def _teacher_forward(self, input_ids, attention_mask):
        with torch.no_grad():
            return self.teacher(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.detach()

    # ---------------------------------------------------------------------
    # forward pass
    # ---------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, output_student_repr: bool = False):
        """Return (loss, student_repr?) so it fits easily into training loops."""
        teacher_repr = self._teacher_forward(input_ids, attention_mask)  # (B,L,1024)

        student_out = self.student(input_ids=input_ids, attention_mask=attention_mask)
        student_repr = self.proj(student_out.last_hidden_state)  # (B,L,1024)

        # compute MSE masked over non‑padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        # Manually calculate MSE to avoid issues with inference tensors
        diff = ((student_repr - teacher_repr) ** 2) * mask
        loss = diff.sum() / mask.sum().clamp(min=1)

        if output_student_repr:
            return loss, student_repr
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


# ---------------------------------------------------------------------------
# quick sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    seqs = [
        "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVIKVETSVHLTPE",
        "MENLFQAGVSNLSR",
    ]

    model = DistilProtT5().to(device)
    batch = tokenize_sequences(model.tokenizer, seqs, device=device)

    loss = model(**batch)
    print(f"Initial distillation loss: {loss.item():.4f}")
