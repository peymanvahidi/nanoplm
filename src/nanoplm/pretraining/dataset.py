import os
import torch
import bisect
import numpy as np
from itertools import islice
from math import ceil
from typing import List, Dict, Optional, Any
from pathlib import Path
from torch.utils.data import Dataset

from nanoplm.utils import logger

# Deprecation shim: ShardWriter moved to nanoplm.data.sharding.pretrain_shard_writer
from nanoplm.data.sharding.pretrain_shard_writer import ShardWriter as ShardWriter  # noqa: F401


class ShardedDataset(Dataset):
    """Dataset for loading pre-tokenized sequences from flat binary shards via memmap.

    Each shard consists of:
    - shard_NNNN.bin: concatenated uint8 tokens (all sequences back-to-back)
    - shard_NNNN.idx.npy: numpy .npy file containing int32 array of sequence lengths

    Uses np.memmap for zero-copy reads. Memmaps are created lazily so that
    spawn-based DataLoader workers (macOS default) only pickle lightweight
    metadata (paths + offsets), not the mapped data itself.
    """

    def __init__(self, data_dir: str, pad_token_id: int = 0) -> None:
        """
        Args:
            data_dir: Directory containing binary shard files (*.bin + *.idx.npy)
            pad_token_id: Token ID used for padding (default 0).
        """
        self._init_dataset(str(data_dir), pad_token_id=pad_token_id, log=True)

    def _init_dataset(self, data_dir: str, pad_token_id: int = 0, log: bool = False) -> None:
        """Shared init logic (called from __init__ and __setstate__).

        Args:
            data_dir: Path to the shard directory.
            pad_token_id: Token ID used for padding.
            log: Whether to emit log messages (suppressed in DataLoader workers).
        """
        self.data_dir = Path(data_dir)
        self.pad_token_id = pad_token_id

        # Validate directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if not self.data_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.data_dir}")

        # Find all shard .bin files — store paths as strings for safe pickling
        self._bin_paths: List[str] = [
            str(p) for p in sorted(self.data_dir.glob("*.bin"))
        ]

        if len(self._bin_paths) == 0:
            raise FileNotFoundError(
                f"No binary shard files (*.bin) found in {self.data_dir}"
            )

        if log:
            logger.info(f"Found {len(self._bin_paths)} binary shards in {self.data_dir}")

        # Load index files (small arrays, always kept in memory)
        self._offsets: List[np.ndarray] = []  # per-shard cumulative byte offsets
        self.lengths: List[int] = []

        for bin_path_str in self._bin_paths:
            bin_path = Path(bin_path_str)
            idx_path = bin_path.with_name(bin_path.stem + ".idx.npy")
            if not idx_path.exists():
                raise FileNotFoundError(
                    f"Index file not found for shard: {idx_path}"
                )

            sizes = np.load(str(idx_path))  # int32 array of sequence lengths
            offsets = np.concatenate([[0], np.cumsum(sizes)])

            self._offsets.append(offsets)
            self.lengths.append(len(sizes))

        self.cum_lengths = np.cumsum(self.lengths)

        if log:
            logger.info(
                f"Loaded {int(self.cum_lengths[-1]):,} pre-tokenized sequences from {len(self._bin_paths)} shards"
            )

    # -- Pickle support for spawn-based DataLoader workers (macOS default) --
    # Without this, pickling serializes the entire memmap data to each worker.
    # With this, only the directory path is pickled; memmaps are re-created lazily.

    def __getstate__(self) -> dict:
        return {"data_dir": str(self.data_dir), "pad_token_id": self.pad_token_id}

    def __setstate__(self, state) -> None:
        # Backward compat: old pickles stored just the data_dir string.
        if isinstance(state, str):
            self._init_dataset(state, pad_token_id=0, log=False)
        else:
            self._init_dataset(state["data_dir"], pad_token_id=state.get("pad_token_id", 0), log=False)

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        # Per-worker lazy cache: rebuild if we're in a new process (fork or spawn).
        pid = os.getpid()
        if getattr(self, '_mmaps_pid', None) != pid:
            self._mmaps = [np.memmap(p, dtype=np.uint8, mode="r") for p in self._bin_paths]
            self._mmaps_pid = pid

        shard_idx, local_idx = self._get_shard(idx)

        offsets = self._offsets[shard_idx]
        start = int(offsets[local_idx])
        end = int(offsets[local_idx + 1])

        raw = self._mmaps[shard_idx][start:end].copy()

        input_ids = torch.from_numpy(raw).long() # To long since it will be concat

        # Generate attention_mask on-the-fly: 1 for non-padding tokens, 0 for padding
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _get_shard(self, idx: int) -> tuple[int, int]:
        """Return (shard_index, local_index_in_shard)"""
        shard_idx = bisect.bisect_right(self.cum_lengths, idx)
        if shard_idx > 0:
            idx -= self.cum_lengths[shard_idx - 1]
        return shard_idx, idx

    def get_all_sequence_lengths(self) -> np.ndarray:
        """Return a flat int32 array of the length of every sequence in the dataset."""
        all_lengths = []
        for offsets in self._offsets:
            all_lengths.append(np.diff(offsets).astype(np.int32))
        return np.concatenate(all_lengths)

    def fingerprint(self) -> str:
        """Return a lightweight hex digest identifying this dataset's content.

        Hashes the number of shards, number of sequences per shard, and all
        sequence lengths (from the already-loaded index arrays).  This is fast
        because it only touches metadata already in memory — no data I/O.
        """
        import hashlib
        h = hashlib.sha256()
        h.update(len(self._bin_paths).to_bytes(4, "little"))
        for offsets in self._offsets:
            lengths = np.diff(offsets).astype(np.int32)
            h.update(lengths.tobytes())
        return h.hexdigest()[:16]

    def cleanup(self):
        """Release per-worker memmap handles."""
        self._mmaps = None
        self._mmaps_pid = None

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens in the dataset."""
        if not hasattr(self, "_total_tokens"):
             self._total_tokens = sum(int(offsets[-1]) for offsets in self._offsets)
        return self._total_tokens


class TokenPackingDataset(torch.utils.data.IterableDataset):
    """Dataset that uses sequence packing to construct batches with variable length up to a maximum number of tokens."""

    def __init__(
        self,
        dataset: Dataset,
        max_tokens_per_batch: int,
        drop_last: bool = True,
        split_samples: bool = False,
        sampler: Optional[torch.utils.data.Sampler] = None,
    ):
        """
        Args:
            dataset: Dataset to pack.
            max_tokens_per_batch: Maximum number of tokens per batch.
            drop_last: Whether to drop the last batch if it's less than max_length.
            split_samples: Whether to split samples to ensure batches have exactly max_tokens_per_batch tokens.
            sampler: Optional sampler to determine the order of iteration over the dataset.
        """
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.drop_last = drop_last
        self.split_samples = split_samples
        self.sampler = sampler
        self._epoch = 0
        self._cached_len: Optional[int] = None
        self._skip_samples = 0
        self._skip_epoch: Optional[int] = None

    def _resolve_sample_lengths(self) -> list[int]:
        if hasattr(self.dataset, "get_all_sequence_lengths"):
            lengths = self.dataset.get_all_sequence_lengths()
            if self.sampler is None:
                return [int(length) for length in lengths.tolist()]
            return [int(lengths[idx]) for idx in self.sampler]

        if self.sampler is not None:
            if not hasattr(self.dataset, "__getitem__"):
                raise TypeError(
                    "Underlying dataset must support indexed access or expose "
                    "'get_all_sequence_lengths' to compute TokenPackingDataset length"
                )
            return [int(len(self.dataset[idx]["input_ids"])) for idx in self.sampler]

        if hasattr(self.dataset, "__len__") and hasattr(self.dataset, "__getitem__"):
            return [int(len(self.dataset[idx]["input_ids"])) for idx in range(len(self.dataset))]

        raise TypeError(
            "Underlying dataset must support indexed access or expose "
            "'get_all_sequence_lengths' to compute TokenPackingDataset length"
        )

    def _count_batches_from_lengths(self, lengths: list[int]) -> int:
        if not lengths:
            return 0

        batches = 0
        current_length = 0

        for sample_length in lengths:
            if sample_length <= 0:
                continue

            current_length += sample_length

            if current_length == self.max_tokens_per_batch:
                batches += 1
                current_length = 0
            elif current_length > self.max_tokens_per_batch:
                if not self.split_samples:
                    if current_length > sample_length:
                        batches += 1
                    current_length = sample_length
                else:
                    batches += 1
                    current_length -= self.max_tokens_per_batch
                    while current_length >= self.max_tokens_per_batch:
                        batches += 1
                        current_length -= self.max_tokens_per_batch

        if not self.drop_last and current_length > 0:
            batches += 1

        return batches

    def __len__(self) -> int:
        """Return the exact number of batches yielded by the dataset."""
        if self._cached_len is None:
            lengths = self._resolve_sample_lengths()
            self._cached_len = self._count_batches_from_lengths(lengths)
        return self._cached_len

    def __iter__(self):
        """Yield batches of samples, each with a variable number of tokens up to the maximum length.

        When split_samples=True, ensures each batch has exactly max_tokens_per_batch by splitting
        the final sample if needed. The remaining tokens from the split sample start the next batch.

        Returns:
            A generator of batches of samples, each with a variable number of tokens up to the maximum length.
        """
        samples = []
        current_length = 0
        skip_epoch = getattr(self, "_skip_epoch", None)
        skip_samples = getattr(self, "_skip_samples", 0) if skip_epoch == self._epoch else 0
        if skip_samples > 0:
            # Reset in the local worker copy after consuming the scheduled skip.
            self._skip_samples = 0
            self._skip_epoch = None

        if self.sampler is not None:
             # Handle DataLoader worker sharding if using multiple workers
             worker_info = torch.utils.data.get_worker_info()
             if worker_info is not None and worker_info.num_workers > 1:
                 if isinstance(self.sampler, torch.utils.data.DistributedSampler):
                     # Build a per-worker DistributedSampler instead of sharing one sampler
                     # and slicing in Python. This avoids O(dataset_size) index fan-out per worker.
                     worker_sampler = torch.utils.data.DistributedSampler(
                         self.dataset,
                         num_replicas=self.sampler.num_replicas * worker_info.num_workers,
                         rank=self.sampler.rank * worker_info.num_workers + worker_info.id,
                         shuffle=self.sampler.shuffle,
                         seed=self.sampler.seed,
                         drop_last=self.sampler.drop_last,
                     )
                     worker_sampler.set_epoch(self._epoch)
                     # When fast-forwarding, skip the corresponding per-worker share of samples.
                     # Each worker sees 1/num_workers of the total samples, so scale accordingly.
                     per_worker_skip = skip_samples // worker_info.num_workers
                     indices = islice(worker_sampler, per_worker_skip, None)
                     iterator = (self.dataset[i] for i in indices)
                 else:
                     # Generic fallback for non-distributed samplers.
                     iterator = (
                         self.dataset[i]
                         for i in islice(self.sampler, worker_info.id + skip_samples, None, worker_info.num_workers)
                     )
             else:
                 indices = islice(self.sampler, skip_samples, None)
                 iterator = (self.dataset[i] for i in indices)

        elif isinstance(self.dataset, torch.utils.data.IterableDataset):
             # For IterableDataset, we rely on the underlying dataset to handle worker sharding
             # (See PyTorch docs: underlying dataset should implement __iter__ with worker awareness)
             iterator = iter(self.dataset)
        else:
             # Map-style dataset without sampler
             worker_info = torch.utils.data.get_worker_info()
             if worker_info is not None and worker_info.num_workers > 1:
                 per_worker = int(ceil(len(self.dataset) / worker_info.num_workers))
                 start = worker_info.id * per_worker
                 end = min(start + per_worker, len(self.dataset))
                 # Only iterate the indices assigned to this worker
                 iterator = (self.dataset[i] for i in range(start, end))
             else:
                 iterator = (self.dataset[i] for i in range(len(self.dataset)))

        for sample in iterator:
            # handle case where sample might be None or invalid if dataset has holes
            if sample is None:
                continue

            current_length += len(sample["input_ids"])
            
            if current_length == self.max_tokens_per_batch:
                yield [*samples, sample]
                samples = []
                current_length = 0

            elif current_length > self.max_tokens_per_batch:
                if not self.split_samples:
                    # If we are not splitting samples, we can just yield the current batch (before this sample) and
                    # start a new one.
                    if samples: # yield only if we have accumulated samples
                        yield samples
                    samples = [sample]
                    current_length = len(sample["input_ids"])
                else:
                    # Calculate how many tokens are already in the batch from previous samples
                    # We need to recalculate tokens_in_batch because current_length includes the current sample
                    tokens_in_batch = current_length - len(sample["input_ids"])
                    
                    tokens_available = self.max_tokens_per_batch - tokens_in_batch
                    
                    first_part, remaining_part = _split_sample_by_num_tokens(sample, tokens_available)
                    yield [*samples, first_part]
                    samples = [remaining_part]
                    # current_length for next iteration is just the remaining part length
                    current_length = len(remaining_part["input_ids"])
            else:
                samples.append(sample)

        if not self.drop_last and samples:
            yield samples

    def fast_forward(self, n_batches: int, epoch: Optional[int] = None) -> int:
        """Advance the dataset state by *n_batches* without reading data.

        Replays the packing logic on sequence lengths alone (read from index
        files, no sample I/O) to determine how many underlying samples are
        consumed in *n_batches* packed batches.  On the next ``__iter__`` call
        the sampler will ``islice`` past those samples so iteration starts at
        the correct position.

        Returns the number of underlying samples that will be skipped.
        """
        if n_batches <= 0:
            self._skip_samples = 0
            self._skip_epoch = None
            return 0

        lengths = self._resolve_sample_lengths()
        batches_seen = 0
        samples_consumed = 0
        current_length = 0

        for sample_length in lengths:
            if batches_seen >= n_batches:
                break
            if sample_length <= 0:
                samples_consumed += 1
                continue

            samples_consumed += 1
            current_length += sample_length

            if current_length == self.max_tokens_per_batch:
                batches_seen += 1
                current_length = 0
            elif current_length > self.max_tokens_per_batch:
                if not self.split_samples:
                    batches_seen += 1
                    current_length = sample_length
                else:
                    batches_seen += 1
                    current_length -= self.max_tokens_per_batch
                    while current_length >= self.max_tokens_per_batch and batches_seen < n_batches:
                        batches_seen += 1
                        current_length -= self.max_tokens_per_batch

        self._skip_samples = samples_consumed
        self._skip_epoch = self._epoch if epoch is None else int(epoch)
        return samples_consumed

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset."""
        self._epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self._epoch)
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self._epoch)


def _split_sample_by_num_tokens(sample: Dict[str, Any], num_tokens: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a sample dictionary at a specified number of tokens.

    This function splits a sample into two parts: the first part contains exactly `num_tokens` tokens,
    and the second part contains the remaining tokens. All fields that are sequences (input_ids, attention_mask,
    token_type_ids, labels, etc.) are split accordingly.
    """
    sample_length = len(sample["input_ids"])
    if num_tokens >= sample_length:
        raise ValueError(
            f"num_tokens ({num_tokens}) must be less than sample length ({sample_length}) to split the sample"
        )
    if num_tokens <= 0:
        raise ValueError(f"num_tokens ({num_tokens}) must be positive")

    first_part = {}
    remaining_part = {}

    # Fields that should be split by tokens (sequence fields)
    sequence_fields = ["input_ids", "attention_mask", "token_type_ids", "token_type", "labels"]

    for key, value in sample.items():
        if key in sequence_fields:
            # Handle both list and tensor inputs
            if isinstance(value, torch.Tensor):
                first_part[key] = value[:num_tokens].clone()
                remaining_part[key] = value[num_tokens:].clone()
            elif isinstance(value, list):
                first_part[key] = value[:num_tokens]
                remaining_part[key] = value[num_tokens:]
            else:
                # For other types, try to slice if possible
                try:
                    first_part[key] = value[:num_tokens]
                    remaining_part[key] = value[num_tokens:]
                except (TypeError, IndexError):
                    first_part[key] = value
                    remaining_part[key] = value
        else:
            first_part[key] = value
            remaining_part[key] = value

    return first_part, remaining_part
