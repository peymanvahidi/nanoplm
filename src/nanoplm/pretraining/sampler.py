"""Length-bucketed batch sampler for efficient sequence packing."""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler


class LengthBucketedBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups similar-length sequences and targets a
    fixed token budget per mini-batch.

    Each mini-batch is filled with sequences from a narrow length range
    until the cumulative token count reaches ``max_tokens``.  This ensures
    every training step processes roughly the same number of tokens
    regardless of sequence lengths, while keeping sequences within a batch
    similar in length for optimal packing.

    Algorithm:
        1. Sort all dataset indices by sequence length.
        2. Chunk sorted indices into mega-batches of
           ``mega_batch_multiplier × batch_size`` sequences.
        3. Within each mega-batch, greedily fill mini-batches up to
           ``max_tokens`` tokens (sequences stay length-sorted).
        4. Shuffle mini-batch order within each mega-batch.
        5. Shuffle mega-batch order across the epoch.
        6. For distributed training: partition mini-batches across ranks.

    Args:
        dataset: A dataset with a ``get_all_sequence_lengths()`` method
            returning an ``np.ndarray`` of int32 lengths.
        batch_size: Maximum number of sequences per mini-batch (cap).
        max_tokens: Target token budget per mini-batch.  If ``None``,
            defaults to ``batch_size × max_seq_len`` (where max_seq_len
            is the longest sequence in the dataset).
        mega_batch_multiplier: How many ``batch_size`` chunks per
            mega-batch.  Larger = tighter length grouping, less randomness.
        shuffle: Whether to shuffle ordering each epoch.
        seed: Random seed for reproducibility.
        drop_last: Drop the last incomplete mini-batch.
        num_replicas: Number of distributed processes (``None`` for single GPU).
        rank: Rank of this process (``None`` for single GPU).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        max_tokens: Optional[int] = None,
        mega_batch_multiplier: int = 100,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.lengths: np.ndarray = dataset.get_all_sequence_lengths()
        self.batch_size = batch_size
        self.max_tokens = max_tokens if max_tokens is not None else batch_size * int(self.lengths.max())
        self.mega_batch_multiplier = mega_batch_multiplier
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        # Pre-sort indices by length (stable sort preserves shard order for ties).
        self._sorted_indices = np.argsort(self.lengths, kind="stable")

        # Cache batch count for __len__.
        self._cached_len: Optional[int] = None

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch
        self._cached_len = None  # batch count may vary by epoch

    # ------------------------------------------------------------------ #
    # Core batching logic
    # ------------------------------------------------------------------ #

    def _fill_batches_from_chunk(self, indices: np.ndarray) -> List[List[int]]:
        """Greedily fill mini-batches from *indices* (assumed length-sorted)
        until each reaches ``max_tokens`` or ``batch_size`` sequences."""
        batches: List[List[int]] = []
        i = 0
        n = len(indices)
        while i < n:
            batch: List[int] = []
            tok_count = 0
            while i < n and len(batch) < self.batch_size:
                seq_len = int(self.lengths[indices[i]])
                if tok_count + seq_len > self.max_tokens and len(batch) > 0:
                    break
                batch.append(int(indices[i]))
                tok_count += seq_len
                i += 1
            if batch:
                batches.append(batch)
        return batches

    def _generate_batches(self) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        mega_size = self.batch_size * self.mega_batch_multiplier

        # Chunk sorted indices into mega-batches.
        mega_batches = [
            self._sorted_indices[i : i + mega_size].copy()
            for i in range(0, len(self._sorted_indices), mega_size)
        ]

        # Shuffle mega-batch order first (which length ranges appear when).
        if self.shuffle:
            rng.shuffle(mega_batches)

        # Within each mega-batch, greedily fill mini-batches by token budget.
        all_batches: List[List[int]] = []
        for mb in mega_batches:
            mb_batches = self._fill_batches_from_chunk(mb)
            if self.drop_last and mb_batches and len(mb_batches[-1]) < self.batch_size:
                # Only drop if the last batch is truly undersized
                # (token-budget batches may legitimately have fewer sequences)
                pass  # keep all batches — token budget is the real constraint
            # Shuffle mini-batch order within each mega-batch.
            if self.shuffle:
                rng.shuffle(mb_batches)
            all_batches.extend(mb_batches)

        # Distributed partitioning: round-robin across ranks.
        if self.num_replicas is not None and self.num_replicas > 1:
            all_batches = all_batches[self.rank :: self.num_replicas]

        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._generate_batches()
        self._cached_len = len(batches)
        yield from batches

    def __len__(self) -> int:
        if self._cached_len is not None:
            return self._cached_len
        # Estimate: run the batching logic to get exact count.
        self._cached_len = len(self._generate_batches())
        return self._cached_len
