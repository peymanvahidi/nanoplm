import torch.nn as nn
import torch.nn.functional as F
import torch
from safetensors.torch import load_file
from transformers import (
    ModernBertModel,
    ModernBertConfig,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.modeling_t5 import T5LayerNorm
from typing import Iterator, Union, List, Generator, Tuple

from .tokenizer import ProtXTokenizer

class ProtX(nn.Module):
    """Student model for ProtX"""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
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

    def forward(self, input_ids, attention_mask, training_mode = False, teacher_embeddings=None):
        student_out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        if training_mode:
            projected_repr = self.proj(student_out.last_hidden_state)  # (batch_size, seq_len, 1024)
            projected_repr = self.proj_norm(projected_repr)
            # training mode
            return BaseModelOutput(
                last_hidden_state=projected_repr,
                hidden_states=student_out.hidden_states,
                attentions=student_out.attentions
            )
        else:
            # Inference mode
            return BaseModelOutput(
                last_hidden_state=student_out.last_hidden_state,
                hidden_states=student_out.hidden_states,
                attentions=student_out.attentions
            )

    def load_and_generate_embeddings(
        self,
        checkpoint_path: str,
        sequences: Union[List[str], Iterator[str]],
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "cuda",
        per_seq_embeddings: bool = True  # True for pooled, False for per-token
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """
        Load model from checkpoint and generate embeddings for sequences.
        
        Args:
            checkpoint_path: Path to the model.safetensors file
            sequences: Iterator or list of protein sequences
            batch_size: Number of sequences to process at once
            max_length: Maximum sequence length for tokenization
            device: Device to run inference on
            per_seq_embeddings: If True, return pooled sequence-level embeddings [embed_dim].
                               If False, return per-token embeddings [sequence_length, embed_dim]
            
        Yields:
            Tuple of (sequence, embedding_tensor) for each input sequence
            - If per_seq_embeddings=True: embedding shape is [embed_dim]
            - If per_seq_embeddings=False: embedding shape is [sequence_length, embed_dim] 
        """
        # Load the checkpoint
        try:
            state_dict = load_file(checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
        
        # Move model to device and set to eval mode
        self.to(device)
        self.eval()
        
        # Convert sequences to list if it's an iterator
        if not isinstance(sequences, list):
            sequences = list(sequences)
        
        # Process sequences in batches
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]
                
                # Tokenize the batch
                tokenized = self.tokenizer.batch_encode_plus(
                    batch_sequences,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized["attention_mask"].to(device)
                
                # Generate embeddings
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    training_mode=False
                )
                
                # Extract embeddings based on per_seq_embeddings setting
                embeddings = outputs.last_hidden_state  # (batch_size, seq_len, embed_dim)
                
                if per_seq_embeddings:
                    # Return pooled sequence-level embeddings (mean pooling)
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    masked_embeddings = embeddings * mask_expanded
                    summed = torch.sum(masked_embeddings, dim=1)
                    summed_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    mean_pooled = summed / summed_mask
                    
                    # Yield each sequence with its pooled embedding
                    for j, (seq, embedding) in enumerate(zip(batch_sequences, mean_pooled)):
                        yield seq, embedding.cpu()
                else:
                    # Return per-token embeddings (remove padding)
                    for j, (seq, seq_embeddings, seq_mask) in enumerate(zip(batch_sequences, embeddings, attention_mask)):
                        # Get actual sequence length (excluding padding)
                        actual_length = seq_mask.sum().item()
                        yield seq, seq_embeddings[:actual_length].cpu()

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
