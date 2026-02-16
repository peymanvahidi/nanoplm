"""Sequence-packing and MLM collators for varlen flash attention.

Key classes:
    ``DataCollatorWithFlattening`` -- production packing collator. Flattens inputs for flash-attention.
    ``ProtDataCollatorForLM`` -- standard padding-based MLM collator used for masking.
"""

import logging
import torch
from dataclasses import dataclass
from typing import Iterable, Optional, List, Any, Dict, Union
from transformers import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------- #
#  ProtDataCollatorForLM -- padding-based MLM collator
# -------------------------------------------------------------------- #


class ProtDataCollatorForLM(DataCollatorForLanguageModeling):
    """Protein-aware MLM collator.

    - Custom (mask, random, keep) proportions.
    - Random replacements drawn only from non-special tokens.
    - Never masks at padding (``attention_mask == 0``).
    """

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        mask_token_probability: float = 0.80,
        random_token_probability: float = 0.10,
        keep_probability: float = 0.10,
        *,
        extra_excluded_token_ids: Optional[Iterable[int]] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability, **kwargs,
        )

        total = mask_token_probability + random_token_probability + keep_probability
        self.p_mask = mask_token_probability / total
        self.p_rand = random_token_probability / total
        self.p_keep = keep_probability / total

        if getattr(self.tokenizer, "mask_token_id", None) is None:
            raise ValueError("Tokenizer must define a mask_token_id for MLM.")

        vocab_ids = list(self.tokenizer.get_vocab().values())
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        special_ids.add(self.tokenizer.mask_token_id)
        if extra_excluded_token_ids:
            special_ids.update(extra_excluded_token_ids)

        allowed = [tid for tid in vocab_ids if tid not in special_ids]
        if not allowed:
            raise ValueError(
                "No allowable token ids for random replacement after exclusions."
            )
        self.allowed_random_token_ids = torch.tensor(allowed, dtype=torch.long)

    def torch_mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None,
    ):
        """Mirrors HF logic but uses custom p_mask/p_rand/p_keep and a restricted random pool."""
        labels = inputs.clone()

        probability_matrix = torch.full(
            labels.shape, self.mlm_probability, device=inputs.device,
        )

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val.tolist(), already_has_special_tokens=True,
                )
                for val in labels
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool, device=inputs.device,
            )

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).to(torch.bool)
        labels[~masked_indices] = -100

        if masked_indices.any():
            dice = torch.rand(size=inputs.shape, device=inputs.device)

            mask_choice = (dice < self.p_mask) & masked_indices
            rand_choice = (
                (dice >= self.p_mask)
                & (dice < self.p_mask + self.p_rand)
                & masked_indices
            )

            inputs[mask_choice] = self.tokenizer.mask_token_id

            if rand_choice.any():
                pool = self.allowed_random_token_ids.to(inputs.device)
                n = rand_choice.sum().item()
                idxs = torch.randint(
                    low=0, high=pool.numel(), size=(n,), device=inputs.device,
                )
                inputs[rand_choice] = pool[idxs]

        return inputs, labels

    def __call__(self, examples: List[dict]):
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch["input_ids"] = batch["input_ids"].long()
        if "token_type_ids" in batch:
            batch["token_type_ids"] = batch["token_type_ids"].long()
        batch["attention_mask"] = batch["attention_mask"].long()

        input_ids = batch["input_ids"]

        if "special_tokens_mask" in batch:
            special = batch["special_tokens_mask"].bool()
        else:
            special = [
                self.tokenizer.get_special_tokens_mask(
                    v.tolist(), already_has_special_tokens=True,
                )
                for v in input_ids
            ]
            special = torch.tensor(special, dtype=torch.bool, device=input_ids.device)

        if "attention_mask" in batch:
            special |= ~batch["attention_mask"].bool()

        inputs, labels = self.torch_mask_tokens(input_ids, special_tokens_mask=special)
        batch["input_ids"] = inputs
        batch["labels"] = labels

        return batch


# -------------------------------------------------------------------- #
#  DataCollatorWithFlattening -- production collator
# -------------------------------------------------------------------- #


@dataclass
class DataCollatorWithFlattening:
    """Data collator that wraps a DataCollatorForLanguageModeling and flattens inputs for flash-attention.

    This collator enables efficient training on batches containing variable-length sequences, by first flattening
    (packing) multiple input sequences into a single contiguous tensor without padding between sequences. Then, it
    applies masked language modeling (MLM) masking using the provided DataCollatorForLanguageModeling instance.

    The collator also generates metadata required for Flash Attention or context-parallel attention:
      - `cu_seqlens` tensors, denoting cumulative sequence lengths so that sequence boundaries
        within the packed tensor are known during attention computation.
    """

    collator: DataCollatorForLanguageModeling
    return_position_ids: bool = False
    pad_to_multiple_of: int | None = None
    pad_sequences_to_be_divisible_by: int | None = None
    separator_id: int | None = None

    def __post_init__(self):
        """Ensure padding options are not used together."""
        if (
            self.pad_sequences_to_be_divisible_by is not None
            and self.pad_to_multiple_of is not None
        ):
            raise ValueError(
                "pad_sequences_to_be_divisible_by and pad_to_multiple_of cannot be used together"
            )

    def __call__(self, features, return_tensors=None):
        """Process a batch of variable-length sequences for Flash Attention with MLM.

        This method performs the following steps:
        1. Flattens multiple sequences into a single packed tensor with Flash Attention metadata
        2. Applies MLM masking by delegating to the underlying collator
        3. Optionally pads to a multiple of a specified number
        """
        # 1. Perform masking with the underlying collator (padding locally to apply mask)
        # Note: We rely on the collator to handle tokenization/masking.
        bshd_batch = self.collator(features)

        # 2. Create the flattened batch structure and compute cu_seqlens based on effective lengths
        #    Wait, we need to be careful. The collator pads the batch. We want the UNPADDED lengths
        #    for cu_seqlens. But we also want the MASKED input_ids.
        #
        #    Strategy:
        #    - Use bshd_batch['attention_mask'] to identify valid tokens.
        #    - Select valid tokens from bshd_batch['input_ids'] (which are masked).
        #    - Select valid tokens from bshd_batch['labels'].
        
        mask = bshd_batch["attention_mask"].bool()
        
        # Flattened tensors (only valid tokens)
        flat_input_ids = bshd_batch["input_ids"][mask]
        flat_labels = bshd_batch["labels"][mask]
        
        # Reconstruct sample lengths from the mask (sum-rows)
        # bshd_batch is [Batch, MaxSeqLen]
        # lengths is [Batch]
        seq_lengths = mask.sum(dim=1).to(torch.int32)
        
        # Verify
        assert flat_input_ids.numel() == seq_lengths.sum().item()

        # Build packet batch
        packed_batch = {}
        packed_batch["input_ids"] = flat_input_ids
        packed_batch["labels"] = flat_labels
        
        # Compute cu_seqlens
        cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device=flat_input_ids.device)
        cu_seqlens[1:] = torch.cumsum(seq_lengths, dim=0)
        
        packed_batch["cu_seqlens"] = cu_seqlens
        packed_batch["max_seqlen"] = seq_lengths.max().item() if len(seq_lengths) > 0 else 0
        packed_batch["num_valid_tokens"] = flat_input_ids.numel()

        if self.return_position_ids:
             # position_ids [0..len-1] for each sequence, concatenated
             # We can generate this easily
             position_ids_list = [torch.arange(slen, device=flat_input_ids.device) for slen in seq_lengths]
             packed_batch["position_ids"] = torch.cat(position_ids_list)

        if self.pad_to_multiple_of is not None:
             packed_batch = self._pad_batch_to_multiple_of(packed_batch)

        return packed_batch

    def _pad_batch_to_multiple_of(self, batch):
        """Add a mock sequence to make the total number of tokens divisible by pad_to_multiple_of."""
        pad_token_id = self.collator.tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            pad_token_id = 0

        assert self.pad_to_multiple_of is not None

        return _pt_pad_to_multiple_of(
            batch,
            self.pad_to_multiple_of,
            token_pad=pad_token_id,
            label_pad=-100,
        )


def _pt_pad_to_multiple_of(batch: Dict[str, Any], pad_to_multiple_of: int, token_pad: int, label_pad: int):
    """Pad a batch to a multiple of pad_to_multiple_of."""
    total_tokens = batch["input_ids"].numel()
    remainder = (-total_tokens) % pad_to_multiple_of

    if remainder == 0:
        return batch

    # Check device
    device = batch["input_ids"].device

    # Extend input_ids
    batch["input_ids"] = torch.cat(
        [batch["input_ids"], torch.full((remainder,), token_pad, dtype=batch["input_ids"].dtype, device=device)], dim=0
    )

    # Extend labels
    if "labels" in batch:
        batch["labels"] = torch.cat(
            [batch["labels"], torch.full((remainder,), label_pad, dtype=batch["labels"].dtype, device=device)], dim=0
        )

    # Handling cu_seqlens with padding in flattened batch:
    # Option 1: Treat padding as a new dummy sequence.
    # Option 2: Append to the last sequence (messy for attention).
    # Option 3: Do nothing to cu_seqlens if Flash Attention implementation ignores trailing tokens not in cu_seqlens ranges.
    #
    # ESM2 adds a new entry to cu_seq_lens.
    
    if "cu_seqlens" in batch:
        current_cu = batch["cu_seqlens"]
        # Add a new "sequence" for the padding
        new_end = current_cu[-1] + remainder
        batch["cu_seqlens"] = torch.cat([current_cu, new_end.unsqueeze(0)])
    
    if "position_ids" in batch:
        # Extend position ids for padding (usually 0 or continue?)
        # ESM2 uses arange(remainder)
        batch["position_ids"] = torch.cat(
            [batch["position_ids"], torch.arange(remainder, dtype=batch["position_ids"].dtype, device=device)], dim=0
        )
        
    return batch
