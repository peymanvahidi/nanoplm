from __future__ import annotations

import threading
import h5py
import torch
import numpy as np

from Bio import SeqIO
from pathlib import Path
from typing import TYPE_CHECKING, Union, List, Optional, Dict
from torch.utils.data import Dataset, IterableDataset
from concurrent.futures import ThreadPoolExecutor

from nanoplm.utils import logger
from nanoplm.data.infra.file_pool import ThreadSafeFileHandlePool, detect_file_limits

if TYPE_CHECKING:
    from nanoplm.distillation.models.teacher import BaseTeacher


class KDDatasetOnTheFly(IterableDataset):
    """
    On-the-fly dataset that yields raw sequences for tokenization in the collator.

    The collator will handle:
    - Student tokenization (for student model input)
    - Teacher tokenization and embedding computation (for distillation target)
    """
    def __init__(
        self,
        input_fasta: Union[str, Path],
        teacher: BaseTeacher,
        max_seq_len: int,
        device: str,
        read_batch_size: int = 32,  # Kept for potential future batched processing
    ):
        self.input_fasta = Path(input_fasta)
        self.teacher = teacher
        self.device = device
        self.max_seq_len = max_seq_len
        self.read_batch_size = read_batch_size

    def __iter__(self):
        """Yield raw sequences for tokenization in collator."""
        data_gen = (
            (record.id, str(record.seq))
            for record in SeqIO.parse(self.input_fasta, "fasta")
        )

        for _, sequence in data_gen:
            # Yield raw sequence - collator will tokenize with both student and teacher tokenizers
            yield {"raw_sequence": sequence}


class LoadKDDataset(Dataset):
    """
    Knowledge distillation dataset loader with optimized I/O for large datasets.

    Features:
    - LRU cache for shard files (memory efficient)
    - Chunked reading for better I/O performance
    - Optional prefetching with threading
    - Reduced memory footprint

    This dataset automatically detects whether the data is stored as:
    - Multiple sharded files: {h5_path.stem}_shard_*.h5
    - Single file: {h5_path}

    Args:
        h5_path: Path to the H5 file (prefix for sharded files)
        device: Device to load tensors to
        seed: Random seed for shuffling (optional)
        sharded: Deprecated - now auto-detected based on files present
        max_open_files: Maximum number of shard files to keep open simultaneously
        chunk_size: Number of samples to read in a single I/O operation
        prefetch_batches: Number of batches to prefetch in background
        use_threading: Whether to enable background prefetching threads
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        device: str,
        seed: Optional[int] = None,
        sharded: bool = False,
        max_open_files: Optional[int] = None,  # Auto-detect if None
        chunk_size: int = 32,  # Read multiple samples at once
        prefetch_batches: int = 2,  # Background prefetching
        use_threading: bool = True,  # Threading for I/O
    ):
        self.h5_path = Path(h5_path)
        self.device = device
        self.seed = seed
        self.chunk_size = chunk_size
        self.prefetch_batches = prefetch_batches
        self.use_threading = use_threading

        # Auto-detect file limits if not specified
        if max_open_files is None:
            max_open_files = detect_file_limits(num_workers=4)
        self.max_open_files = max_open_files

        # Lazy file pool (created per-worker or on first access)
        self._worker_pool = None

        # Prefetch cache for chunks (orthogonal to file pooling)
        self._prefetch_cache: Dict[int, List[Dict]] = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_executor = None
        self._prefetch_hits = 0

        # Auto-detect sharded vs single file
        self.sharded = self._detect_sharding()

        if self.sharded:
            self._load_sharded_files()
        else:
            self.shard_paths = [self.h5_path]
            self.shard_sizes = [self._get_file_size(self.h5_path)]
            self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)
            self.total_size = sum(self.shard_sizes)

        self.indices = list(range(self.total_size))

        if self.seed is not None:
            self._shuffle_indices()

        # Start prefetch thread if enabled
        if self.use_threading:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="ProtX-Prefetch"
            )

        logger.info(f"LoadKDDataset initialized:")
        logger.info(f"  - Max open files: {self.max_open_files}")
        logger.info(f"  - Chunk size: {self.chunk_size}")
        logger.info(f"  - Prefetch batches: {self.prefetch_batches}")
        logger.info(f"  - Threading enabled: {self.use_threading}")

    def _detect_sharding(self) -> bool:
        """
        Auto-detect whether the dataset is sharded or stored as a single file.

        Returns:
            True if sharded files exist, False if single file exists

        Raises:
            FileNotFoundError: If neither sharded nor single file exists
        """
        base_name = self.h5_path.stem
        parent_dir = self.h5_path.parent

        # Check for sharded files first
        shard_pattern = f"{base_name}_shard_*.h5"
        shard_files = list(parent_dir.glob(shard_pattern))

        if shard_files:
            logger.info(f"Detected sharded dataset: found {len(shard_files)} shard files")
            return True

        # Check for single file
        if self.h5_path.exists():
            logger.info(f"Detected single-file dataset: {self.h5_path.name}")
            return False

        # Neither exists - provide helpful error message
        raise FileNotFoundError(
            f"No dataset files found at {self.h5_path}.\n"
            f"Looked for:\n"
            f"  - Sharded files: {parent_dir / shard_pattern}\n"
            f"  - Single file: {self.h5_path}\n"
            f"Please ensure you have run 'nanoplm data from-yaml' with pipeline_mode: 'distillation'"
        )

    def _get_file_size(self, path: Path) -> int:
        """Get number of sequences in H5 file without keeping it open"""
        with h5py.File(path, "r") as f:
            return len(f.keys())

    def _load_sharded_files(self):
        """Load shard metadata without opening all files immediately"""
        base_name = self.h5_path.stem
        parent_dir = self.h5_path.parent

        # Find all shard files
        shard_pattern = f"{base_name}_shard_*.h5"
        self.shard_paths = sorted(parent_dir.glob(shard_pattern))

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shard files found matching pattern: {parent_dir / shard_pattern}"
            )

        logger.info(f"Found {len(self.shard_paths)} shard files")

        # Get sizes without keeping files open
        self.shard_sizes = []
        for shard_path in self.shard_paths:
            size = self._get_file_size(shard_path)
            self.shard_sizes.append(size)

        # Build cumulative index
        self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)
        self.total_size = sum(self.shard_sizes)

        logger.info(f"Total sequences across shards: {self.total_size}")
        for i, size in enumerate(self.shard_sizes):
            logger.info(f"Shard {i}: {size} sequences")

    def _get_worker_pool(self) -> ThreadSafeFileHandlePool:
        """Get or create worker-specific file handle pool"""
        if self._worker_pool is None:
            self._worker_pool = ThreadSafeFileHandlePool(
                max_open_files=self.max_open_files
            )
        return self._worker_pool

    def _shuffle_indices(self):
        """Shuffle the indices based on seed"""
        rng = np.random.RandomState(self.seed)
        rng.shuffle(self.indices)

    def _read_chunk(self, start_idx: int, end_idx: int) -> List[Dict]:
        """Read a chunk of samples efficiently"""
        chunk_data = []

        # Group indices by shard for efficient access
        shard_groups = {}
        for i, idx in enumerate(range(start_idx, end_idx)):
            if idx >= self.total_size:
                break

            actual_idx = self.indices[idx]

            if self.sharded:
                shard_idx = np.searchsorted(
                    self.cumulative_sizes[1:], actual_idx, side="right"
                )
                local_idx = actual_idx - self.cumulative_sizes[shard_idx]
            else:
                shard_idx = 0
                local_idx = actual_idx

            if shard_idx not in shard_groups:
                shard_groups[shard_idx] = []
            shard_groups[shard_idx].append((i, local_idx))

        # Read from each shard in the chunk
        result_data = [None] * (end_idx - start_idx)
        pool = self._get_worker_pool()

        for shard_idx, local_indices in shard_groups.items():
            shard_file = pool.get_file(self.shard_paths[shard_idx])

            for result_idx, local_idx in local_indices:
                if local_idx >= self.shard_sizes[shard_idx]:
                    continue

                grp = shard_file[str(local_idx)]

                input_ids = torch.tensor(grp["input_ids"][:], dtype=torch.long)
                attention_mask = torch.tensor(
                    grp["attention_mask"][:], dtype=torch.long
                )
                teacher_embeddings = torch.tensor(
                    grp["teacher_embeddings"][:], dtype=torch.float
                )

                result_data[result_idx] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "teacher_embeddings": teacher_embeddings,
                }

        return [data for data in result_data if data is not None]

    def _prefetch_chunk(self, chunk_start: int):
        """Prefetch a chunk in background thread"""
        if not self.use_threading:
            return

        chunk_end = min(chunk_start + self.chunk_size, self.total_size)

        # Check if already cached
        with self._prefetch_lock:
            if chunk_start in self._prefetch_cache:
                return

        # Read chunk
        chunk_data = self._read_chunk(chunk_start, chunk_end)

        # Store in cache
        with self._prefetch_lock:
            # Keep cache size limited
            if len(self._prefetch_cache) >= self.prefetch_batches:
                # Remove oldest entry
                oldest_key = min(self._prefetch_cache.keys())
                del self._prefetch_cache[oldest_key]

            self._prefetch_cache[chunk_start] = chunk_data

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if idx >= self.total_size or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.total_size}"
            )

        # Calculate chunk boundaries
        chunk_start = (idx // self.chunk_size) * self.chunk_size
        chunk_offset = idx - chunk_start

        # Check prefetch cache first
        if self.use_threading:
            with self._prefetch_lock:
                if chunk_start in self._prefetch_cache:
                    chunk_data = self._prefetch_cache[chunk_start]
                    if chunk_offset < len(chunk_data):
                        self._prefetch_hits += 1

                        # Prefetch next chunk in background
                        next_chunk_start = chunk_start + self.chunk_size
                        if (
                            next_chunk_start < self.total_size
                            and next_chunk_start not in self._prefetch_cache
                        ):
                            self._prefetch_executor.submit(
                                self._prefetch_chunk, next_chunk_start
                            )

                        return chunk_data[chunk_offset]

        # Fallback to direct read
        actual_idx = self.indices[idx]
        pool = self._get_worker_pool()

        if self.sharded:
            # Find which shard contains this index
            shard_idx = np.searchsorted(
                self.cumulative_sizes[1:], actual_idx, side="right"
            )
            local_idx = actual_idx - self.cumulative_sizes[shard_idx]
            shard_file = pool.get_file(self.shard_paths[shard_idx])
            grp = shard_file[str(local_idx)]
        else:
            shard_file = pool.get_file(self.shard_paths[0])
            grp = shard_file[str(actual_idx)]

        input_ids = torch.tensor(grp["input_ids"][:], dtype=torch.long)
        attention_mask = torch.tensor(grp["attention_mask"][:], dtype=torch.long)
        teacher_embeddings = torch.tensor(
            grp["teacher_embeddings"][:], dtype=torch.float
        )

        # Trigger prefetching for future accesses
        if self.use_threading:
            next_chunk_start = chunk_start + self.chunk_size
            if (
                next_chunk_start < self.total_size
                and next_chunk_start not in self._prefetch_cache
            ):
                self._prefetch_executor.submit(self._prefetch_chunk, next_chunk_start)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_embeddings": teacher_embeddings,
        }

    def get_stats(self) -> Dict[str, int]:
        """Get performance statistics"""
        stats = {
            "prefetch_hits": self._prefetch_hits,
            "prefetch_cache_size": len(self._prefetch_cache),
        }

        # Add pool stats if pool exists
        if self._worker_pool is not None:
            pool_stats = self._worker_pool.get_stats()
            stats.update({
                "pool_hits": pool_stats['hits'],
                "pool_misses": pool_stats['misses'],
                "pool_hit_rate": pool_stats['hit_rate'],
                "open_files": pool_stats['open_files'],
            })

        return stats

    def __getstate__(self):
        """Prepare state for pickling (needed for multiprocessing DataLoader)"""
        state = self.__dict__.copy()
        # Remove non-picklable objects
        state["_worker_pool"] = None  # Pool recreated per-worker
        state["_prefetch_cache"] = {}
        state["_prefetch_lock"] = None
        state["_prefetch_executor"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling"""
        self.__dict__.update(state)
        # Recreate locks and executor
        self._prefetch_lock = threading.Lock()
        self._prefetch_cache = {}
        if self.use_threading:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="ProtX-Prefetch"
            )
        # Pool recreated on first access via _get_worker_pool()

    def __del__(self):
        """Clean up file handles and threads"""
        # Close file pool
        if hasattr(self, '_worker_pool') and self._worker_pool is not None:
            self._worker_pool.close_all()

        # Shutdown thread pool
        if hasattr(self, "_prefetch_executor") and self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=False)
