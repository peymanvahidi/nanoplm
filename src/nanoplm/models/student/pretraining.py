import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any, Iterator
from pathlib import Path
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from transformers import (
    PreTrainedTokenizer, 
    PreTrainedModel, 
    PretrainedConfig,
    Trainer,
    TrainingArguments
)
from transformers.modeling_outputs import MaskedLMOutput
import math

from nanoplm.models.student.model import ProtX
from nanoplm.models.student.tokenizer import ProtXTokenizer


class ProtXMLMTokenizer(ProtXTokenizer):
    """Extended tokenizer with MASK token for MLM pretraining"""
    
    def __init__(self, vocab=None, unk_token="<unk>", pad_token="<pad>", eos_token="</s>", mask_token="<mask>"):
        # Define vocabulary mapping amino acids & special tokens including MASK
        self.vocab = {
            "A": 4, "L": 5, "G": 6, "V": 7, "S": 8, "R": 9, "E": 10, "D": 11,
            "T": 12, "I": 13, "P": 14, "K": 15, "F": 16, "Q": 17, "N": 18,
            "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23, "X": 24, "B": 25,
            "O": 26, "U": 27, "Z": 28, 
            pad_token: 0, eos_token: 1, unk_token: 2, mask_token: 3
        }

        # Call PreTrainedTokenizer's __init__ directly
        from transformers import PreTrainedTokenizer
        PreTrainedTokenizer.__init__(
            self,
            unk_token=unk_token, 
            pad_token=pad_token, 
            eos_token=eos_token,
            mask_token=mask_token
        )

        # Set up token ID attributes
        self.unk_token_id = self.vocab.get(unk_token)
        self.pad_token_id = self.vocab.get(pad_token)
        self.eos_token_id = self.vocab.get(eos_token)
        self.mask_token_id = self.vocab.get(mask_token)
        
        self.model_input_names = ["input_ids", "attention_mask"]


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
        total_prob = (self.mask_token_probability + 
                     self.random_token_probability + 
                     self.leave_unchanged_probability)
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


class ProtXMLMConfig(PretrainedConfig):
    """Configuration class for ProtXMLM model"""
    
    model_type = "protx_mlm"
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_activation: str = "swiglu",
        vocab_size: int = 29,
        max_position_embeddings: int = 1024,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        mask_token_id: int = 3,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_activation = mlp_activation
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.mask_token_id = mask_token_id


class ProtXMLM(PreTrainedModel):
    """ProtX model with MLM head for pretraining"""
    
    config_class = ProtXMLMConfig
    
    def __init__(self, config: ProtXMLMConfig):
        super().__init__(config)
        
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Create base ProtX model but modify config for MLM
        from transformers import ModernBertConfig
        from transformers.models.t5.modeling_t5 import T5LayerNorm
        from nanoplm.models.student.model import ModernBertMLPSwiGLU
        
        self.bert_config = ModernBertConfig(
            vocab_size=self.vocab_size,
            hidden_size=config.embed_dim,
            intermediate_size=config.embed_dim * 2,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            mlp_bias=False,
            attention_bias=False,
        )
        
        from transformers import ModernBertModel
        self.model = ModernBertModel(self.bert_config)
        
        if config.mlp_activation.lower() == "swiglu":
            for layer in self.model.layers:
                layer.mlp = ModernBertMLPSwiGLU(self.bert_config)
        
        # MLM prediction head
        self.mlm_head = nn.Linear(config.embed_dim, self.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def _init_weights(self, module):
        """Initialize weights for MLM head"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[MaskedLMOutput, torch.Tensor]:
        """
        Forward pass for MLM training or inference
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            labels: Labels for MLM loss calculation (optional)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return ModelOutput object
            
        Returns:
            MaskedLMOutput if labels provided, else logits tensor
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get hidden states from base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Apply MLM head
        logits = self.mlm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Calculate MLM loss
            loss_fct = nn.CrossEntropyLoss()
            # Flatten for loss calculation
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


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
    model: ProtXMLM,
    training_args: TrainingArguments,
    train_dataset: ProteinMLMDataset,
    eval_dataset: Optional[ProteinMLMDataset] = None,
    tokenizer: Optional[ProtXMLMTokenizer] = None,
    data_collator: Optional[MLMDataCollator] = None
) -> MLMTrainer:
    """Create MLM Trainer for pretraining, following distillation patterns"""
    
    if tokenizer is None:
        tokenizer = ProtXMLMTokenizer()
    
    if data_collator is None:
        data_collator = MLMDataCollator(tokenizer=tokenizer)
    
    return MLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


def create_mlm_model_from_config(
    embed_dim: int = 512,
    num_layers: int = 12,
    num_heads: int = 8,
    mlp_activation: str = "swiglu",
    vocab_size: int = 29,
    **kwargs
) -> ProtXMLM:
    """Create MLM model from configuration parameters"""
    config = ProtXMLMConfig(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_activation=mlp_activation,
        vocab_size=vocab_size,
        **kwargs
    )
    
    model = ProtXMLM(config)
    return model


def create_mlm_model_from_checkpoint(
    checkpoint_path: str,
    mlp_activation: str = "swiglu"
) -> ProtXMLM:
    """Create MLM model and load from distillation checkpoint"""
    # Detect architecture from checkpoint
    embed_dim, num_layers, num_heads = ProtX.inspect_checkpoint_architecture(checkpoint_path)
    
    # Create MLM model config
    config = ProtXMLMConfig(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_activation=mlp_activation
    )
    
    # Create MLM model
    mlm_model = ProtXMLM(config)
    
    # Load weights from distillation checkpoint (partial loading)
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)
    
    # Filter out projection layer weights since MLM model doesn't have them
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if not k.startswith(('proj.', 'proj_norm.'))
    }
    
    # Load compatible weights
    mlm_model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded base model weights from {checkpoint_path}")
    
    return mlm_model


def save_mlm_model_for_downstream(
    model: ProtXMLM,
    output_path: str,
    save_config: bool = True
):
    """Save MLM model without the MLM head for downstream use"""
    
    # Extract base model state dict (without MLM head)
    base_state_dict = {}
    for key, value in model.state_dict().items():
        if not key.startswith('mlm_head.'):
            base_state_dict[key] = value
    
    # Save as safetensors for compatibility with existing code
    try:
        from safetensors.torch import save_file
        save_file(base_state_dict, output_path)
        print(f"Base model saved to {output_path}")
        
        if save_config:
            config_path = output_path.replace('.safetensors', '_config.json')
            model.config.save_pretrained(Path(config_path).parent)
            print(f"Config saved to {config_path}")
            
    except ImportError:
        torch.save(base_state_dict, output_path.replace('.safetensors', '.pt'))
        print(f"Base model saved to {output_path.replace('.safetensors', '.pt')} (safetensors not available)") 