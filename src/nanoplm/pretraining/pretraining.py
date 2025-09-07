import torch
import random
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any, Iterator
from pathlib import Path
from Bio import SeqIO
from torch.utils.data import Dataset
from dataclasses import dataclass
from transformers import (
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
import math

from nanoplm.pretraining.models.modern_bert import (
    ModernBertTokenizer,
    ModernBertForMaskedLM,
    create_modernbert_mlm,
)


"""Pretraining utilities using ModernBERT for Masked Language Modeling."""


@dataclass
class MLMDataCollator:
    """Data collator for masked language modeling"""
    
    tokenizer: PreTrainedTokenizer
    mlm_probability: float = 0.15
    mask_token_probability: float = 0.8
    random_token_probability: float = 0.1
    leave_unchanged_probability: float = 0.1
    
    def __post_init__(self):
        # Verify probabilities sum to 1
        total_prob = self.mask_token_probability + self.random_token_probability + self.leave_unchanged_probability
        if not math.isclose(total_prob, 1.0, rel_tol=1e-5):
            raise ValueError(f"Masking probabilities must sum to 1.0, got {total_prob}")
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Apply MLM masking to a batch of examples
        
        Args:
            examples: List of dictionaries with 'input_ids' and 'attention_mask'
            
        Returns:
            Dictionary with masked input_ids, attention_mask, and labels
        """
        # Stack inputs
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        
        # Clone input_ids for labels (original unmasked tokens)
        labels = input_ids.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = self._get_special_tokens_mask(input_ids)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Don't mask padding tokens
        probability_matrix.masked_fill_(~attention_mask.bool(), value=0.0)
        
        # Sample tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels to -100 for non-masked tokens (ignore in loss)
        labels[~masked_indices] = -100
        
        # Apply different masking strategies
        self._apply_masking_strategies(input_ids, masked_indices)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get mask for special tokens that shouldn't be masked"""
        special_tokens = {
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.unk_token_id
        }
        
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_tokens:
            if token_id is not None:
                mask |= (input_ids == token_id)
        
        return mask
    
    def _apply_masking_strategies(self, input_ids: torch.Tensor, masked_indices: torch.Tensor):
        """Apply different masking strategies to selected tokens"""
        # Get amino acid token IDs (excluding special tokens)
        amino_acid_tokens = [
            token_id for token_id in self.tokenizer.vocab.values()
            if token_id not in {
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id, 
                self.tokenizer.unk_token_id,
                self.tokenizer.mask_token_id
            }
        ]
        
        # 80% of the time: replace with [MASK] token
        mask_token_indices = masked_indices.clone()
        mask_prob = torch.rand(input_ids.shape) < self.mask_token_probability
        mask_token_indices &= mask_prob
        input_ids[mask_token_indices] = self.tokenizer.mask_token_id
        
        # 10% of the time: replace with random amino acid token  
        random_token_indices = masked_indices.clone()
        random_token_indices &= ~mask_token_indices
        random_prob = torch.rand(input_ids.shape) < (
            self.random_token_probability / (self.random_token_probability + self.leave_unchanged_probability)
        )
        random_token_indices &= random_prob
        
        if random_token_indices.any():
            random_tokens = torch.tensor(
                np.random.choice(amino_acid_tokens, size=random_token_indices.sum().item()),
                device=input_ids.device
            )
            input_ids[random_token_indices] = random_tokens
        
        # 10% of the time: leave unchanged (already done implicitly)


class ProteinMLMDataset(Dataset):
    """Dataset for MLM pretraining on protein sequences"""
    
    def __init__(
        self,
        fasta_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        min_length: int = 20,
        subsample_ratio: float = 1.0,
        seed: int = 42
    ):
        self.fasta_path = Path(fasta_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.subsample_ratio = subsample_ratio
        
        # Load and filter sequences
        self.sequences = self._load_sequences(seed)
        
    def _load_sequences(self, seed: int) -> List[str]:
        """Load sequences from FASTA file with optional subsampling"""
        sequences = []
        
        for record in SeqIO.parse(self.fasta_path, "fasta"):
            seq = str(record.seq).upper()
            
            # Filter by length
            if self.min_length <= len(seq) <= self.max_length:
                # Replace non-standard amino acids
                seq = self._clean_sequence(seq)
                sequences.append(seq)
        
        # Subsample if requested
        if self.subsample_ratio < 1.0:
            random.seed(seed)
            n_samples = int(len(sequences) * self.subsample_ratio)
            sequences = random.sample(sequences, n_samples)
        
        print(f"Loaded {len(sequences)} sequences for MLM pretraining")
        return sequences
    
    def _clean_sequence(self, sequence: str) -> str:
        """Clean sequence by replacing non-standard amino acids"""
        # Replace ambiguous/non-standard amino acids with X
        replacements = {"B": "X", "Z": "X", "U": "X", "O": "X"}
        for old, new in replacements.items():
            sequence = sequence.replace(old, new)
        return sequence
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Tokenize sequence
        encoded = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


    


class MLMTrainer(Trainer):
    """Custom trainer for MLM pretraining, following the same pattern as DistillationTrainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute MLM loss"""
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def create_training_args(
    output_dir: str = "./mlm_checkpoints",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    max_steps: int = -1,
    logging_steps: int = 100,
    eval_steps: int = 500,
    save_steps: int = 1000,
    save_total_limit: int = 3,
    fp16: bool = False,  # Default to False for CPU compatibility
    dataloader_num_workers: int = 0,  # Default to 0 for macOS compatibility
    remove_unused_columns: bool = False,
    report_to: list = None,
    **kwargs
) -> TrainingArguments:
    """Create TrainingArguments for MLM pretraining, following distillation patterns"""
    
    # Create base training arguments
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "save_strategy": "steps",
        "fp16": fp16,
        "dataloader_num_workers": dataloader_num_workers,
        "remove_unused_columns": remove_unused_columns,
        "report_to": report_to,
    }
    
    # Add evaluation strategy if we have eval steps
    if eval_steps > 0:
        training_args_dict["eval_steps"] = eval_steps
        training_args_dict["eval_strategy"] = "steps"
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = "eval_loss"
        training_args_dict["greater_is_better"] = False
    
    # Merge with any additional kwargs
    training_args_dict.update(kwargs)
    
    return TrainingArguments(**training_args_dict)


def create_trainer(
    model: ModernBertForMaskedLM,
    training_args: TrainingArguments,
    train_dataset: ProteinMLMDataset,
    eval_dataset: Optional[ProteinMLMDataset] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    data_collator: Optional[MLMDataCollator] = None
) -> MLMTrainer:
    """Create MLM Trainer for pretraining, following distillation patterns"""
    
    if tokenizer is None:
        tokenizer = ModernBertTokenizer()
    
    if data_collator is None:
        data_collator = MLMDataCollator(tokenizer=tokenizer)
    
    return MLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )


def create_mlm_model_from_config(
    embed_dim: int = 512,
    num_layers: int = 12,
    num_heads: int = 8,
    intermediate_size: Optional[int] = None,
    mlp_activation: str = "swiglu",
    vocab_size: int = 29,
    **kwargs
) -> ModernBertForMaskedLM:
    """Create ModernBERT MLM model from configuration parameters"""
    inter_size = intermediate_size if intermediate_size is not None else embed_dim * 2

    return create_modernbert_mlm(
        hidden_size=embed_dim,
        intermediate_size=inter_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        vocab_size=vocab_size,
        mlp_activation=mlp_activation,
        **kwargs,
    )


def create_mlm_model_from_checkpoint(
    checkpoint_path: str,
    mlp_activation: str = "swiglu",
    vocab_size: int = 29,
) -> ModernBertForMaskedLM:
    """Create ModernBERT MLM model and load from a distillation checkpoint.

    This function maps student checkpoint keys from 'model.' to 'backbone.' and drops
    projection-related parameters that do not exist in the MLM model.
    """
    # Detect architecture from checkpoint
    embed_dim, num_layers, num_heads = inspect_checkpoint_architecture(checkpoint_path)

    # Create ModernBERT MLM model
    model = create_modernbert_mlm(
        hidden_size=embed_dim,
        intermediate_size=embed_dim * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        vocab_size=vocab_size,
        mlp_activation=mlp_activation,
    )

    # Load weights from distillation checkpoint (partial loading)
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)

    # Filter out projection layer weights and rename model.* -> backbone.*
    remapped_state_dict = {}
    for k, v in state_dict.items():
        # Skip projection layers and their norms
        if k.startswith("proj.") or k.startswith("proj_norm."):
            continue
        # Rename 'model.' prefix (student) to 'backbone.' (MLM model)
        if k.startswith("model."):
            new_key = "backbone." + k[len("model."):]
        else:
            new_key = k
        remapped_state_dict[new_key] = v

    # Handle vocabulary size mismatch for token embeddings
    tok_embed_key = "backbone.embeddings.tok_embeddings.weight"
    if tok_embed_key in remapped_state_dict:
        tensor = remapped_state_dict[tok_embed_key]
        ckpt_vocab = tensor.shape[0]
        model_vocab = model.config.vocab_size
        if ckpt_vocab == model_vocab:
            pass  # OK
        elif ckpt_vocab > model_vocab:
            # Truncate checkpoint to model's vocab size
            remapped_state_dict[tok_embed_key] = tensor[: model_vocab]
        else:
            # Checkpoint vocab smaller than model vocab (e.g., no mask token); keep model's init
            del remapped_state_dict[tok_embed_key]

    # Load compatible weights
    model.load_state_dict(remapped_state_dict, strict=False)
    print(f"Loaded base model weights from {checkpoint_path}")

    return model


def save_mlm_model_for_downstream(
    model: ModernBertForMaskedLM,
    output_path: str,
    save_config: bool = True
):
    """Save MLM model without the MLM head for downstream use"""
    
    # Extract base model state dict (without MLM head)
    base_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('lm_head.')}

    # For compatibility with code expecting 'model.' prefix, remap 'backbone.' -> 'model.'
    compatible_state_dict = {}
    for k, v in base_state_dict.items():
        if k.startswith('backbone.'):
            new_key = 'model.' + k[len('backbone.'):]
        else:
            new_key = k
        compatible_state_dict[new_key] = v
    
    # Save as safetensors for compatibility with existing code
    try:
        from safetensors.torch import save_file
        save_file(compatible_state_dict, output_path)
        print(f"Base model saved to {output_path}")
        
        if save_config:
            config_path = output_path.replace('.safetensors', '_config.json')
            model.config.save_pretrained(Path(config_path).parent)
            print(f"Config saved to {config_path}")
            
    except ImportError:
        torch.save(compatible_state_dict, output_path.replace('.safetensors', '.pt'))
        print(f"Base model saved to {output_path.replace('.safetensors', '.pt')} (safetensors not available)") 


def inspect_checkpoint_architecture(checkpoint_path: str) -> Tuple[int, int, int]:
    """
    Inspect model architecture from a safetensors checkpoint file.

    Returns a tuple (embed_dim, num_layers, num_heads).
    """
    from safetensors.torch import load_file

    try:
        state_dict = load_file(checkpoint_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading checkpoint from {checkpoint_path}: {e}")

    embed_dim = None
    num_layers = 0
    num_heads = None

    # Try to infer embed_dim from embeddings or early layer weights
    for key, tensor in state_dict.items():
        if 'embeddings.tok_embeddings.weight' in key:
            vocab_size, embed_dim = tensor.shape
            break
        elif 'model.layers.0.attn.Wo.weight' in key or 'backbone.layers.0.attn.Wo.weight' in key:
            embed_dim = tensor.shape[0]
            break
        elif 'model.layers.0.mlp.Wi.weight' in key or 'backbone.layers.0.mlp.Wi.weight' in key:
            embed_dim = tensor.shape[1]
            break

    if embed_dim is None:
        raise ValueError("Could not determine embed_dim from checkpoint")

    # Count number of layers by scanning keys
    layer_indices = set()
    for key in state_dict.keys():
        if '.layers.' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        layer_indices.add(layer_idx)
                    except ValueError:
                        continue
    num_layers = len(layer_indices)
    if num_layers == 0:
        raise ValueError("Could not determine num_layers from checkpoint")

    # Infer number of attention heads from Wqkv matrix of the first layer
    for key, tensor in state_dict.items():
        if 'layers.0.attn.Wqkv.weight' in key:
            qkv_dim, model_dim = tensor.shape
            if qkv_dim == 3 * embed_dim:
                # Try common head dimensions
                for head_dim in [64, 32, 128, 16]:
                    if embed_dim % head_dim == 0:
                        num_heads = embed_dim // head_dim
                        break
            break

    if num_heads is None:
        # Fallback: try to infer from separate query weights if present
        for key, tensor in state_dict.items():
            if ('layers.0' in key) and ('query' in key.lower()) and ('weight' in key) and len(tensor.shape) == 2:
                out_dim, in_dim = tensor.shape
                if in_dim == embed_dim:
                    for head_dim in [64, 32, 128, 16]:
                        if out_dim % head_dim == 0:
                            num_heads = out_dim // head_dim
                            break
                break

    if num_heads is None:
        raise ValueError("Could not determine num_heads from checkpoint")

    return embed_dim, num_layers, num_heads