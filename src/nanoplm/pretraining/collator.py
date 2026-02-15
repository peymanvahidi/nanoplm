import torch
import numpy as np
from typing import Iterable, Optional, List
from transformers import DataCollatorForLanguageModeling


class ProtDataCollatorForLM(DataCollatorForLanguageModeling):
    """
    Protein-aware MLM collator:
      - custom (mask, random, keep) proportions
      - random replacements drawn only from non-special tokens
      - never masks at padding (attention_mask == 0)
    """

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.03,
        mask_token_probability: float = 0.80,
        random_token_probability: float = 0.10,
        keep_probability: float = 0.10,
        *,
        extra_excluded_token_ids: Optional[Iterable[int]] = None,
        **kwargs,
    ):
        # parent handles dynamic padding, tensor type, etc.
        super().__init__(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability, **kwargs
        )

        # normalize split (robust to slight mis-specified sums)
        total = mask_token_probability + random_token_probability + keep_probability
        self.p_mask = mask_token_probability / total
        self.p_rand = random_token_probability / total
        self.p_keep = keep_probability / total

        if getattr(self.tokenizer, "mask_token_id", None) is None:
            raise ValueError("Tokenizer must define a mask_token_id for MLM.")

        # Build the pool for random replacements: all vocab ids minus specials (and optional extras)
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
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ):
        """
        Mirrors HF logic but uses custom p_mask/p_rand/p_keep and a restricted random pool.
        """
        labels = inputs.clone()

        # base Bernoulli for whether a position is subject to any corruption
        probability_matrix = torch.full(
            labels.shape, self.mlm_probability, device=inputs.device
        )

        if special_tokens_mask is None:
            # compute via tokenizer
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val.tolist(), already_has_special_tokens=True
                )
                for val in labels
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool, device=inputs.device
            )

        # never corrupt special positions
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # sample which positions to corrupt at all
        masked_indices = torch.bernoulli(probability_matrix).to(torch.bool)
        labels[~masked_indices] = -100  # ignore in loss

        if masked_indices.any():
            # Within the selected positions, decide mask / random / keep
            dice = torch.rand(size=inputs.shape, device=inputs.device)

            mask_choice = (dice < self.p_mask) & masked_indices
            rand_choice = (
                (dice >= self.p_mask)
                & (dice < self.p_mask + self.p_rand)
                & masked_indices
            )
            # keep_choice is implicit

            # 1) replace with [MASK]
            inputs[mask_choice] = self.tokenizer.mask_token_id

            # 2) replace with random non-special token
            if rand_choice.any():
                pool = self.allowed_random_token_ids.to(inputs.device)
                n = rand_choice.sum().item()
                idxs = torch.randint(
                    low=0, high=pool.numel(), size=(n,), device=inputs.device
                )
                inputs[rand_choice] = pool[idxs]

            # 3) keep_choice -> unchanged
        return inputs, labels

    def __call__(self, examples: List[dict]):
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Ensure tensors are in long dtype (what HF models expect)
        batch["input_ids"] = batch["input_ids"].long()
        batch["attention_mask"] = batch["attention_mask"].long()

        input_ids = batch["input_ids"]

        # Build/augment special mask
        if "special_tokens_mask" in batch:
            special = batch["special_tokens_mask"].bool()
        else:
            special = [
                self.tokenizer.get_special_tokens_mask(
                    v.tolist(), already_has_special_tokens=True
                )
                for v in input_ids
            ]
            special = torch.tensor(special, dtype=torch.bool, device=input_ids.device)

        # Forbid masking where attention_mask == 0 (padding, packed slack, etc.)
        if "attention_mask" in batch:
            special |= ~batch["attention_mask"].bool()

        inputs, labels = self.torch_mask_tokens(input_ids, special_tokens_mask=special)
        batch["input_ids"] = inputs
        batch["labels"] = labels

        return batch


class PackingCollator:
    """Sequence-packing MLM collator for varlen flash attention.

    Packs multiple protein sequences into flat 1-D tensors to eliminate padding
    waste. Each batch contains several sequences back-to-back; the collator
    emits ``cu_seqlens``, ``position_ids``, and ``max_seqlen`` so the model can
    pass them directly to ``flash_attn_varlen_func`` without re-deriving
    sequence boundaries from ``attention_mask``.

    Algorithm (first-fit-decreasing):
        1. Sort incoming examples by length (descending).
        2. Greedily assign each sequence to the first row that has room.
        3. Apply MLM masking on the packed result.
        4. Flatten to 1-D, stripping all padding.

    When ``pad_to`` is set, output tensors are padded to a fixed size
    ``F = pad_to`` and ``cu_seqlens`` is padded to a fixed length, enabling
    ``torch.compile(dynamic=False)``.

    Output dict:
        ``input_ids``        – (T,) or (F,)  flat token IDs
        ``labels``           – (T,) or (F,)  MLM targets (-100 for pad/unmasked)
        ``position_ids``     – (T,) or (F,)  per-token positions (reset per seq)
        ``cu_seqlens``       – variable or fixed length
        ``max_seqlen``       – int
        ``num_valid_tokens`` – int    = T (total real tokens)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        mlm_probability: float = 0.03,
        mask_token_probability: float = 0.80,
        random_token_probability: float = 0.10,
        keep_probability: float = 0.10,
        *,
        pad_to: Optional[int] = None,
        max_batch_sequences: Optional[int] = None,
        extra_excluded_token_ids: Optional[Iterable[int]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.mlm_probability = mlm_probability
        self.pad_to = pad_to
        if pad_to is not None:
            self.F = pad_to
            # Fixed cu_seqlens length: real seqs + 1 (leading 0) + padding seqs.
            # Padding tokens (F - T) are split into chunks ≤ max_seq_len.
            if max_batch_sequences is None:
                raise ValueError(
                    "max_batch_sequences must be provided when pad_to is set."
                )
            micro_batch_size = pad_to // max_seq_len
            self.cu_len = max_batch_sequences + 1 + micro_batch_size
        else:
            self.F = None
            self.cu_len = None

        # Normalise mask/random/keep proportions
        total = mask_token_probability + random_token_probability + keep_probability
        self.p_mask = mask_token_probability / total
        self.p_rand = random_token_probability / total
        self.p_keep = keep_probability / total

        if getattr(self.tokenizer, "mask_token_id", None) is None:
            raise ValueError("Tokenizer must define a mask_token_id for MLM.")

        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

        # Build random-replacement pool (same logic as ProtDataCollatorForLM)
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

    # --------------------------------------------------------------------- #
    # Packing
    # --------------------------------------------------------------------- #

    def _pack_sequences(
        self, examples: List[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Pack *examples* into fixed-width rows using first-fit-decreasing.

        Returns:
            input_ids      – (R, max_seq_len)
            attention_mask – (R, max_seq_len)
            cu_seqlens     – (total_seqs + 1,) int32
            max_seqlen     – int
        """
        # Extract per-example token tensors (variable length)
        seqs: List[torch.Tensor] = []
        for ex in examples:
            ids = ex["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            # Trim trailing padding (datasets may store fixed-width rows)
            mask = ex.get("attention_mask", None)
            if mask is not None:
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask)
                length = int(mask.sum().item())
                ids = ids[:length]
            seqs.append(ids)

        # Sort descending by length for better bin-packing
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
        seqs = [seqs[i] for i in order]

        W = self.max_seq_len  # row width

        # Greedy first-fit-decreasing into rows
        rows: List[List[torch.Tensor]] = []     # per-row list of sequences
        row_fill: List[int] = []                 # current fill of each row

        for seq in seqs:
            slen = len(seq)
            if slen > W:
                seq = seq[:W]
                slen = W
            placed = False
            for r_idx in range(len(rows)):
                if row_fill[r_idx] + slen <= W:
                    rows[r_idx].append(seq)
                    row_fill[r_idx] += slen
                    placed = True
                    break
            if not placed:
                rows.append([seq])
                row_fill.append(slen)

        R = len(rows)  # number of packed rows

        # Build output tensors
        input_ids = torch.full((R, W), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((R, W), dtype=torch.long)

        seq_lengths: List[int] = []
        max_seqlen = 0

        for r_idx, row_seqs in enumerate(rows):
            pos = 0
            for seq in row_seqs:
                slen = len(seq)
                input_ids[r_idx, pos : pos + slen] = seq.long()
                attention_mask[r_idx, pos : pos + slen] = 1
                seq_lengths.append(slen)
                if slen > max_seqlen:
                    max_seqlen = slen
                pos += slen

        cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(
            torch.tensor(seq_lengths, dtype=torch.int32), dim=0
        )

        return input_ids, attention_mask, cu_seqlens, max_seqlen

    # --------------------------------------------------------------------- #
    # MLM masking (operates on the packed 2-D tensors)
    # --------------------------------------------------------------------- #

    def _apply_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MLM masking, respecting padding and special tokens."""
        labels = input_ids.clone()

        probability_matrix = torch.full(
            labels.shape, self.mlm_probability, device=input_ids.device
        )

        # Build special-token mask per row (each row may contain multiple seqs)
        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for row_idx in range(input_ids.size(0)):
            row_ids = input_ids[row_idx].tolist()
            stm = self.tokenizer.get_special_tokens_mask(
                row_ids, already_has_special_tokens=True
            )
            special[row_idx] = torch.tensor(stm, dtype=torch.bool)

        # Never mask padding positions
        special |= ~attention_mask.bool()

        probability_matrix.masked_fill_(special, 0.0)

        masked_indices = torch.bernoulli(probability_matrix).to(torch.bool)
        labels[~masked_indices] = -100

        if masked_indices.any():
            dice = torch.rand(size=input_ids.shape, device=input_ids.device)

            mask_choice = (dice < self.p_mask) & masked_indices
            rand_choice = (
                (dice >= self.p_mask)
                & (dice < self.p_mask + self.p_rand)
                & masked_indices
            )

            input_ids[mask_choice] = self.tokenizer.mask_token_id

            if rand_choice.any():
                pool = self.allowed_random_token_ids.to(input_ids.device)
                n = rand_choice.sum().item()
                idxs = torch.randint(
                    low=0, high=pool.numel(), size=(n,), device=input_ids.device
                )
                input_ids[rand_choice] = pool[idxs]

        return input_ids, labels

    # --------------------------------------------------------------------- #
    # __call__
    # --------------------------------------------------------------------- #

    def __call__(self, examples: List[dict]) -> dict:
        input_ids_2d, attention_mask_2d, cu_seqlens, max_seqlen = self._pack_sequences(
            examples
        )

        input_ids_2d, labels_2d = self._apply_mlm(input_ids_2d, attention_mask_2d)

        # Flatten to 1-D (zero waste): strip all padding from the 2-D packed
        # rows and emit flat tensors with cu_seqlens + position_ids — the model
        # takes the varlen flash-attention path directly.
        from nanoplm.pretraining.models.modern_bert.modeling import (
            _position_ids_from_cu_seqlens,
        )

        real_mask = attention_mask_2d.flatten().bool()
        flat_ids = input_ids_2d.flatten()[real_mask]    # (T,)
        flat_labels = labels_2d.flatten()[real_mask]     # (T,)
        T = flat_ids.shape[0]

        position_ids = _position_ids_from_cu_seqlens(
            cu_seqlens, T, cu_seqlens.device
        )

        if self.pad_to is None:
            return {
                "input_ids": flat_ids,          # (T,)
                "labels": flat_labels,           # (T,)
                "position_ids": position_ids,    # (T,)
                "cu_seqlens": cu_seqlens,        # (num_seqs + 1,)
                "max_seqlen": max_seqlen,        # int
                "num_valid_tokens": T,           # real token count
            }

        # ---- static-size mode: pad to fixed F for dynamic=False compile ----
        F = self.F
        W = self.max_seq_len

        padded_ids = torch.full((F,), self.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((F,), -100, dtype=torch.long)
        padded_pos = torch.zeros(F, dtype=torch.int32)
        padded_ids[:T] = flat_ids
        padded_labels[:T] = flat_labels
        padded_pos[:T] = position_ids

        # Build fixed-length cu_seqlens: real entries, then padding "sequences"
        # of ≤ max_seq_len each (so max_seqlen stays ≤ max_seq_len and flash
        # attention tiles optimally), then fill remaining slots with F.
        padded_cu = torch.full((self.cu_len,), F, dtype=torch.int32)
        N_real = len(cu_seqlens)
        padded_cu[:N_real] = cu_seqlens

        pad_tokens = F - T
        pos = T
        idx = N_real
        while pad_tokens > 0 and idx < self.cu_len:
            chunk = min(W, pad_tokens)
            pos += chunk
            padded_cu[idx] = pos
            pad_tokens -= chunk
            idx += 1

        return {
            "input_ids": padded_ids,        # (F,) fixed
            "labels": padded_labels,         # (F,) fixed
            "position_ids": padded_pos,      # (F,) fixed
            "cu_seqlens": padded_cu,         # (cu_len,) fixed
            "max_seqlen": W,                 # constant = max_seq_len
            "num_valid_tokens": T,           # real token count
        }
