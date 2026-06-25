from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

@dataclass
class DataCollatorForCLM:

    pad_token_id: int = 0
    ignore_index: int = -100
    pack_sequences: bool = False
    max_length: Optional[int] = None

    def __call__(
        self, examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        seqs: List[List[int]] = []
        for ex in examples:
            if isinstance(ex, dict):
                ids = ex["input_ids"]
            else:
                ids = ex
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if self.max_length is not None:
                ids = ids[: self.max_length]
            seqs.append(ids)

        if self.pack_sequences:
            return self._pack(seqs)
        return self._pad(seqs)

    # Padded path                                                          

    def _pad(self, seqs: List[List[int]]) -> Dict[str, Any]:
        max_len = max(len(s) for s in seqs)
        input_ids_list: List[List[int]] = []
        attention_mask_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        for ids in seqs:
            n = len(ids)
            pad_len = max_len - n

            # Build pre-shifted labels: label[i] = ids[i+1] within the sequence.
            shifted = ids[1:] + [self.ignore_index]  # last position has no target
            shifted = shifted + [self.ignore_index] * pad_len  # mask padding

            input_ids_list.append(ids + [self.pad_token_id] * pad_len)
            attention_mask_list.append([1] * n + [0] * pad_len)
            labels_list.append(shifted)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }

    # Packed (varlen) path                                                 

    def _pack(self, seqs: List[List[int]]) -> Dict[str, Any]:
        flat_ids: List[int] = []
        flat_labels: List[int] = []
        seq_lengths: List[int] = []

        for ids in seqs:
            n = len(ids)
            seq_lengths.append(n)
            flat_ids.extend(ids)

            for j in range(n - 1):
                flat_labels.append(ids[j + 1])
            flat_labels.append(self.ignore_index)  # boundary: no next token

        # cu_seqlens: cumulative sequence lengths, shape (batch+1,), int32.
        cu = [0]
        for length in seq_lengths:
            cu.append(cu[-1] + length)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32)
        max_seqlen = max(seq_lengths)

        return {
            "input_ids": torch.tensor(flat_ids, dtype=torch.long),
            "labels": torch.tensor(flat_labels, dtype=torch.long),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
