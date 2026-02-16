import os
import torch
import bisect
import numpy as np
from math import ceil
from Bio import SeqIO
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from concurrent.futures import ProcessPoolExecutor

from nanoplm.utils import logger, create_dirs


class ShardWriter:
    """Utility class to tokenize FASTA sequences and save them as flat binary shards.

    This class handles the preprocessing step: reading FASTA, tokenizing sequences,
    and saving them to binary files (.bin + .idx) for fast loading during training.

    This is NOT a Dataset - it's a preprocessing utility that creates shards.
    """

    def __init__(
        self,
        fasta_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        output_dir: str,
        samples_per_shard: int = 10000,
        max_workers: int = -1,
        force: bool = False,
    ) -> None:
        """
        Args:
            fasta_path: Path to input FASTA file
            tokenizer: Tokenizer to use for encoding sequences
            max_length: Maximum sequence length
            output_dir: Directory to save binary shards
            samples_per_shard: Number of sequences per shard file
            max_workers: Number of parallel workers (-1 = all CPUs)
            force: If True, overwrite existing shards
        """
        self.fasta_path = str(fasta_path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.output_dir = Path(output_dir)
        self.samples_per_shard = int(samples_per_shard)
        self.max_workers = os.cpu_count() if max_workers == -1 else max_workers
        self.force = bool(force)

        # Validate that the FASTA file exists and is readable
        fasta_path_obj = Path(self.fasta_path)
        if not fasta_path_obj.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_path}")

        if not fasta_path_obj.is_file():
            raise ValueError(f"Path is not a file: {self.fasta_path}")

        if not os.access(self.fasta_path, os.R_OK):
            raise PermissionError(f"FASTA file is not readable: {self.fasta_path}")

        # Check if the file has any content
        if fasta_path_obj.stat().st_size == 0:
            raise ValueError(f"FASTA file is empty: {self.fasta_path}")

        # Create or open a persistent SQLite-backed index for random access
        self._db_path = f"{self.fasta_path}.idx"
        self._index = SeqIO.index_db(self._db_path, [self.fasta_path], "fasta")
        self._keys: List[str] = list(self._index.keys())

        if len(self._keys) == 0:
            raise ValueError(f"No sequences found in FASTA: {self.fasta_path}")

        logger.info(
            f"Loaded FASTA: {self.fasta_path} with {len(self._keys):,} sequences (max_length={self.max_length})."
        )

    def create_shards(self) -> List[Path]:
        """Create binary shards from FASTA sequences.

        Returns:
            List of paths to created shard .bin files
        """
        # Check if shards already exist
        shards_exist = (
            self.output_dir.exists() and len(list(self.output_dir.glob("*.bin"))) > 0
        )

        if shards_exist and not self.force:
            raise FileExistsError(
                f"Binary shards already exist in {self.output_dir}. "
                f"Set force=True to overwrite them."
            )

        if shards_exist and self.force:
            logger.warning(f"Overwriting existing shards in {self.output_dir}")
            for shard in self.output_dir.glob("*.bin"):
                shard.unlink()
            for shard in self.output_dir.glob("*.idx.npy"):
                shard.unlink()

        # Create output directory
        create_dirs(self.output_dir)

        total_seqs = len(self._keys)
        num_shards = ceil(total_seqs / self.samples_per_shard)

        logger.info(
            f"Splitting {total_seqs:,} sequences into {num_shards} shards "
            f"({self.samples_per_shard:,} sequences per shard)"
        )

        # Split keys into shards
        sharded_keys = [
            self._keys[i * self.samples_per_shard : (i + 1) * self.samples_per_shard]
            for i in range(num_shards)
        ]

        # Prepare arguments for parallel processing
        args = [
            (
                i,
                self.fasta_path,
                self._db_path,
                keys,
                self.tokenizer,
                self.max_length,
                self.output_dir,
            )
            for i, keys in enumerate(sharded_keys)
        ]

        # Process shards in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            shard_paths = list(executor.map(process_shard, args))

        logger.info(
            f"Successfully tokenized {total_seqs:,} sequences and saved to {len(shard_paths)} binary shards in {self.output_dir}"
        )

        return [Path(p) for p in shard_paths]


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

        # Memmaps created lazily per-process (not here)
        self._mmaps: Optional[List[np.memmap]] = None

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

    # -- Lazy memmap creation --

    def _ensure_mmaps(self) -> None:
        """Create memmap objects on first access (once per process)."""
        if self._mmaps is None:
            self._mmaps = [
                np.memmap(p, dtype=np.uint8, mode="r") for p in self._bin_paths
            ]

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        self._ensure_mmaps()

        shard_idx, local_idx = self._get_shard(idx)

        offsets = self._offsets[shard_idx]
        start = int(offsets[local_idx])
        end = int(offsets[local_idx + 1])

        raw = self._mmaps[shard_idx][start:end]
        input_ids = torch.from_numpy(raw.copy()).long() # To long since it will be concat

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

    def cleanup(self):
        """Release memmap objects (public API, kept for compatibility)."""
        if self._mmaps is not None:
            for mmap in self._mmaps:
                del mmap
            self._mmaps = None

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

    def __len__(self) -> int:
        """Return the estimated number of batches in the dataset."""
        if hasattr(self.dataset, "total_tokens"):
             total_tokens = self.dataset.total_tokens
        elif hasattr(self.dataset, "__len__") and hasattr(self.dataset, "__getitem__"):
             if isinstance(self.dataset, ShardedDataset):
                 total_tokens = self.dataset.total_tokens
             else:
                 # Fallback: estimate using length * mean_seq_len ? No reliability.
                 # We will return the dataset length which is definitely wrong (too high)
                 # but prevents "not implemented" errors.
                 # Ideally we should raise specific warning.
                 # OR, we assume full packing isn't happening and return len(dataset).
                 # Raising is safer.
                 raise TypeError("Underlying dataset must have 'total_tokens' property to estimate length of TokenPackingDataset")
        else:
             raise TypeError("Underlying dataset must have 'total_tokens' property to estimate length of TokenPackingDataset")

        # Adjust for sampler (e.g. distributed training)
        # If we have a sampler, we assume it shards the dataset.
        if self.sampler is not None:
            if hasattr(self.sampler, "num_replicas") and self.sampler.num_replicas > 1:
                total_tokens = total_tokens // self.sampler.num_replicas
            elif hasattr(self.sampler, "num_samples") and self.sampler.num_samples is not None:
                # Some samplers might expose num_samples directly
                # However, for packing, we need tokens, not samples.
                # If sampler subsamples (like RandomSampler with num_samples), we can't easily know token count.
                # We'll assume proportional reduction if num_samples < len(dataset).
                if hasattr(self.dataset, "__len__"):
                     ratio = self.sampler.num_samples / len(self.dataset)
                     total_tokens = int(total_tokens * ratio)

        # length is total tokens / max tokens per batch.
        # Since packing is greedy and not optimal, we will have some fragmentation/waste.
        # A simple heuristic is to assume 95% packing efficiency.
        # But to avoid warnings, it's safer to overestimate the number of batches slightly.
        # Or, we can just return exact theoretical minimum and let DataLoader warn.
        # The warning is harmless but annoying.
        # Let's adjust by a factor of 1.1 to account for fragmentation if not splitting samples.
        factor = 1.0
        if not self.split_samples:
             # Without splitting, large samples cause more fragmentation
             factor = 1.15

        estimated_batches = int(ceil(total_tokens / self.max_tokens_per_batch * factor))
        return estimated_batches

    def __iter__(self):
        """Yield batches of samples, each with a variable number of tokens up to the maximum length.

        When split_samples=True, ensures each batch has exactly max_tokens_per_batch by splitting
        the final sample if needed. The remaining tokens from the split sample start the next batch.

        Returns:
            A generator of batches of samples, each with a variable number of tokens up to the maximum length.
        """
        samples = []
        current_length = 0
        
        if self.sampler is not None:
             # Handle DataLoader worker sharding if using multiple workers
             worker_info = torch.utils.data.get_worker_info()
             if worker_info is not None and worker_info.num_workers > 1:
                 # We need to split the sampler's indices across workers.
                 # We iterate over the sampler and pick indices corresponding to this worker.
                 # A simple round-robin or chunking approach works.
                 # Slicing the list is simplest for now.
                 all_indices = list(self.sampler)
                 # Shard indices: each worker gets a contiguous chunk
                 per_worker = int(ceil(len(all_indices) / worker_info.num_workers))
                 start = worker_info.id * per_worker
                 end = min(start + per_worker, len(all_indices))
                 indices = all_indices[start:end]
                 iterator = (self.dataset[i] for i in indices)
             else:
                 iterator = (self.dataset[i] for i in self.sampler)

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

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset."""
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


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


def process_shard(args):
    shard_idx, fasta_path, db_path, shard_keys, tokenizer, max_length, output_dir = args

    # Each process must open its own index
    index = SeqIO.index_db(db_path, [fasta_path], "fasta")

    bin_path = Path(output_dir) / f"shard_{shard_idx:04d}.bin"
    idx_path = Path(output_dir) / f"shard_{shard_idx:04d}.idx.npy"

    token_arrays = []

    for key in tqdm(shard_keys, desc=f"Tokenizing shard {shard_idx}", leave=False):
        record = index[key]
        seq = str(record.seq)

        encoding = tokenizer(
            seq,
            add_special_tokens=True,
            padding=False,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        token_arrays.append(encoding["input_ids"].squeeze(0).numpy().astype(np.uint8))

    # Build sizes array (int32)
    sizes = np.array([len(a) for a in token_arrays], dtype=np.int32)

    # Concatenate all tokens and write .bin
    all_tokens = np.concatenate(token_arrays)
    all_tokens.tofile(str(bin_path))

    # Save sizes as .npy (the .idx file)
    np.save(str(idx_path), sizes)

    index.close()
    return str(bin_path)
