import os
import torch
import bisect
import numpy as np
from math import ceil
from Bio import SeqIO
from typing import List, Dict, Optional
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from concurrent.futures import ProcessPoolExecutor

from nanoplm.utils import logger, create_dirs
from nanoplm.data.file_pool import ThreadSafeFileHandlePool, detect_file_limits


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

    def __init__(self, data_dir: str, load_all_in_memory: bool = False) -> None:
        """
        Args:
            data_dir: Directory containing binary shard files (*.bin + *.idx.npy)
            load_all_in_memory: Accepted for API compatibility but ignored
                (memmap is already the optimal strategy)
        """
        self._init_dataset(str(data_dir), log=True)

    def _init_dataset(self, data_dir: str, log: bool = False) -> None:
        """Shared init logic (called from __init__ and __setstate__).

        Args:
            data_dir: Path to the shard directory.
            log: Whether to emit log messages (suppressed in DataLoader workers).
        """
        self.data_dir = Path(data_dir)

        # Auto-detect file limits if not specified
        if max_open_files is None:
            max_open_files = detect_file_limits(num_workers=1)
        self.max_open_files = max_open_files

        # File handle pool (created per-worker or on first access)
        self._worker_pool: Optional[ThreadSafeFileHandlePool] = None

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

    def __getstate__(self) -> str:
        return str(self.data_dir)

    def __setstate__(self, state: str) -> None:
        self._init_dataset(state, log=False)

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
        input_ids = torch.from_numpy(raw.copy())  # single copy, stays uint8

        # Generate attention_mask on-the-fly: 1 for non-padding tokens, 0 for padding (pad_token_id=0)
        attention_mask = (input_ids != 0).to(torch.uint8)

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

    def cleanup(self):
        """Release memmap objects (public API, kept for compatibility)."""
        if self._mmaps is not None:
            for mmap in self._mmaps:
                del mmap
            self._mmaps = None

    def _get_worker_pool(self) -> ThreadSafeFileHandlePool:
        """
        Get or create the file handle pool for this worker.

class LazyFastaDataset(Dataset):
    """FASTA dataset that tokenizes sequences lazily for MLM pretraining.

    Uses an on-disk index for random access and defers padding to the collator.
    """

        Returns:
            ThreadSafeFileHandlePool: The file handle pool for this worker
        """
        if self._worker_pool is None:
            self._worker_pool = ThreadSafeFileHandlePool(max_open_files=self.max_open_files)
            logger.debug(f"Created file handle pool with max_open_files={self.max_open_files}")
        return self._worker_pool

    def __getstate__(self):
        """
        Prepare state for pickling (DataLoader multiprocessing).

        File handles cannot be pickled, so we clear the worker pool.
        It will be recreated per-worker via worker_init_fn.
        """
        state = self.__dict__.copy()
        state['_worker_pool'] = None  # Don't pickle file handles
        return state

    def __setstate__(self, state):
        """
        Restore state after unpickling.

        The worker pool will be recreated on first access in each worker.
        """
        self.__dict__.update(state)
        # Pool will be created via _get_worker_pool() or worker_init_fn

    def read_batch_vectorized(self, indices: List[int]) -> Dict[str, List[torch.Tensor]]:
        """
        Read multiple samples efficiently using vectorized HDF5 reads.

        For consecutive indices within the same shard, uses hyperslab selection
        (slice) for efficient batch reads. For non-consecutive or cross-shard
        access, falls back to grouped individual reads.

        Args:
            indices: List of global sample indices to read

        Returns:
            Dict with keys:
                - 'input_ids': List of tensors (one per sample)
                - 'attention_mask': List of tensors (one per sample)

        Example:
            >>> dataset = LoadShardedFastaMLMDataset("data/shards")
            >>> batch = dataset.read_batch_vectorized([0, 1, 2, 3])
            >>> print(len(batch['input_ids']))  # 4
        """
        if self._in_memory:
            # For in-memory mode, just use regular indexing
            return {
                'input_ids': [self[i]['input_ids'] for i in indices],
                'attention_mask': [self[i]['attention_mask'] for i in indices],
            }

        # Group indices by shard
        shard_groups = {}
        for idx in indices:
            shard_idx, local_idx = self._get_shard(idx)
            if shard_idx not in shard_groups:
                shard_groups[shard_idx] = []
            shard_groups[shard_idx].append((idx, local_idx))

        # Read from each shard
        results = {}
        pool = self._get_worker_pool()

        for shard_idx, idx_pairs in shard_groups.items():
            # Sort by local index to check for consecutive access
            idx_pairs.sort(key=lambda x: x[1])
            local_indices = [local_idx for _, local_idx in idx_pairs]

            file_handle = pool.get_file(self.shard_paths[shard_idx])

            # Check if indices are consecutive
            is_consecutive = all(
                local_indices[i] + 1 == local_indices[i + 1]
                for i in range(len(local_indices) - 1)
            )

        key = self._keys[idx]

        # Create index on-demand to avoid multiprocessing pickle issues
        index = SeqIO.index_db(self._db_path, [self.fasta_path], "fasta")
        try:
            record = index[key]
            sequence = str(record.seq)
        finally:
            index.close()

        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            padding=False,  # defer padding to the collator for dynamic padding
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Return results in original order
        return {
            'input_ids': [results[idx]['input_ids'] for idx in indices],
            'attention_mask': [results[idx]['attention_mask'] for idx in indices],
        }


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
