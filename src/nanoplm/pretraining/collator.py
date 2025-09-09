import math
import torch
from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer
from dataclasses import dataclass


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

        # Precompute allowed token ids for random replacement (exclude specials and [MASK])
        vocab_ids = list(self.tokenizer.get_vocab().values())
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        if getattr(self.tokenizer, "mask_token_id", None) is not None:
            special_ids.add(self.tokenizer.mask_token_id)
        self._allowed_random_token_ids = torch.tensor(
            [tid for tid in vocab_ids if tid not in special_ids], dtype=torch.long
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Apply MLM masking to a batch of examples

        Args:
            examples: List of dictionaries with 'input_ids' and 'attention_mask'

        Returns:
            Dictionary with masked input_ids, attention_mask, and labels
        """
        # Dynamic padding using tokenizer for variable-length sequences
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Clone input_ids for labels (original unmasked tokens)
        labels = input_ids.clone()

        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens (use tokenizer utility)
        special_tokens_mask = self._get_special_tokens_mask(input_ids)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Don't mask padding tokens
        probability_matrix.masked_fill_(~attention_mask.bool(), value=0.0)

        # Sample tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens AND padding tokens (ignore in loss)
        labels[~masked_indices] = -100
        # Also set padding positions to -100 (they should always be ignored)
        labels[~attention_mask.bool()] = -100

        # Apply different masking strategies
        self._apply_masking_strategies(input_ids, masked_indices)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get mask of special tokens that shouldn't be masked using the tokenizer's logic."""
        seq_masks: List[List[int]] = [
            self.tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True)
            for seq in input_ids
        ]
        return torch.tensor(seq_masks, dtype=torch.bool, device=input_ids.device)

    def _apply_masking_strategies(self, input_ids: torch.Tensor, masked_indices: torch.Tensor):
        """Apply different masking strategies to selected tokens"""
        # Allowed random replacement token ids (move to current device lazily)
        allowed_ids = self._allowed_random_token_ids.to(input_ids.device)

        # 80% of the time: replace with [MASK] token
        mask_token_indices = masked_indices.clone()
        mask_prob = torch.rand(input_ids.shape, device=input_ids.device) < self.mask_token_probability
        mask_token_indices &= mask_prob
        input_ids[mask_token_indices] = self.tokenizer.mask_token_id

        # 10% of the time: replace with random amino acid token
        random_token_indices = masked_indices.clone()
        random_token_indices &= ~mask_token_indices
        random_prob = torch.rand(input_ids.shape, device=input_ids.device) < (
            self.random_token_probability / (self.random_token_probability + self.leave_unchanged_probability)
        )
        random_token_indices &= random_prob

        if random_token_indices.any():
            num_random = random_token_indices.sum().item()
            # Sample uniformly from allowed token ids
            idxs = torch.randint(low=0, high=allowed_ids.numel(), size=(num_random,), device=input_ids.device)
            random_tokens = allowed_ids[idxs]
            input_ids[random_token_indices] = random_tokens

        # 10% of the time: leave unchanged (already done implicitly)
