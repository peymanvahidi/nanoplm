import os
import torch
import h5py
import bisect
import numpy as np
from math import ceil
from Bio import SeqIO
from typing import List, Dict, Optional
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed

from nanoplm.utils import logger, create_dirs
from nanoplm.data.file_pool import ThreadSafeFileHandlePool, detect_file_limits


class SaveShardedFastaMLMDataset:
    """Utility class to tokenize FASTA sequences and save them as HDF5 shards.

    This class handles the preprocessing step: reading FASTA, tokenizing sequences,
    and saving them to HDF5 files for fast loading during training.

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
            output_dir: Directory to save HDF5 shards
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
        """Create HDF5 shards from FASTA sequences.

        Returns:
            List of paths to created shard files
        """
        # Check if shards already exist
        shards_exist = (
            self.output_dir.exists() and len(list(self.output_dir.glob("*.h5"))) > 0
        )

        if shards_exist and not self.force:
            raise FileExistsError(
                f"HDF5 shards already exist in {self.output_dir}. "
                f"Set force=True to overwrite them."
            )

        if shards_exist and self.force:
            logger.warning(f"Overwriting existing shards in {self.output_dir}")
            for shard in self.output_dir.glob("*.h5"):
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
            f"Successfully tokenized {total_seqs:,} sequences and saved to {len(shard_paths)} HDF5 shards in {self.output_dir}"
        )

        return [Path(p) for p in shard_paths]


class LoadShardedFastaMLMDataset(Dataset):
    """Dataset for loading pre-tokenized sequences from HDF5 shards.

    Supports two modes:
    - streaming (default): Uses LRU file handle pool for efficient file access
    - load_all_in_memory: read all shards into memory (dict) at init
    """

    def __init__(
        self,
        hdf5_dir: str,
        load_all_in_memory: bool = False,
        max_open_files: Optional[int] = None,
    ) -> None:
        """
        Args:
            hdf5_dir: Directory containing HDF5 shard files (*.h5)
            load_all_in_memory: Whether to load all shards into memory (default: False)
            max_open_files: Maximum number of file handles to keep open (auto-detected if None)
        """
        self.hdf5_dir = Path(hdf5_dir)
        self._in_memory = bool(load_all_in_memory)

        # Auto-detect file limits if not specified
        if max_open_files is None:
            max_open_files = detect_file_limits(num_workers=1)
        self.max_open_files = max_open_files

        # File handle pool (created per-worker or on first access)
        self._worker_pool: Optional[ThreadSafeFileHandlePool] = None

        # Validate directory exists
        if not self.hdf5_dir.exists():
            raise FileNotFoundError(f"HDF5 directory not found: {self.hdf5_dir}")

        if not self.hdf5_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.hdf5_dir}")

        # Find all shard files
        self.shard_paths = sorted(self.hdf5_dir.glob("*.h5"))

        if len(self.shard_paths) == 0:
            raise FileNotFoundError(
                f"No HDF5 shard files (*.h5) found in {self.hdf5_dir}"
            )

        logger.info(f"Found {len(self.shard_paths)} HDF5 shards in {self.hdf5_dir}")

        # Read shard lengths without keeping files open
        self.lengths = []
        for path in self.shard_paths:
            with h5py.File(path, "r") as f:
                n = len(f["input_ids"])
            self.lengths.append(n)

        self.cum_lengths = np.cumsum(self.lengths)

        logger.info(
            f"Loaded {int(self.cum_lengths[-1]):,} pre-tokenized sequences from {len(self.shard_paths)} shards"
        )

        # In-memory option: load all shard data into Python lists
        if self._in_memory:
            logger.info("Loading all shards into memory (parallel)...")
            num_shards = len(self.shard_paths)
            # Decide number of workers: use up to half of CPUs but not more than shards
            try:
                max_workers = max(1, min((os.cpu_count() or 1) // 2, num_shards))
            except Exception:
                max_workers = 1

            # Prepare result container
            self._in_memory_shards = [None] * num_shards

            # Use ProcessPoolExecutor to let each worker open its own HDF5 file
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    futures = {
                        exe.submit(_read_shard_for_worker, str(p)): idx
                        for idx, p in enumerate(self.shard_paths)
                    }
                    for fut in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="Loading shards",
                        leave=False,
                    ):
                        idx = futures[fut]
                        try:
                            _, data = fut.result()
                            self._in_memory_shards[idx] = data
                        except Exception:
                            # If any worker fails, fall back to sequential loading
                            logger.warning(
                                "Parallel shard loading failed; falling back to sequential load."
                            )
                            raise

                # Verify none are None
                for i, entry in enumerate(self._in_memory_shards):
                    if entry is None:
                        raise RuntimeError(f"Shard {i} not loaded correctly")

                # Build flat index for O(1) global access
                self._build_flat_index()

            except Exception:
                # Fallback: sequential load with safe bulk reads
                logger.info("Falling back to sequential shard loading...")
                self._in_memory_shards = []
                for i, path in enumerate(self.shard_paths, start=1):
                    if i % 10 == 0 or i == num_shards:
                        logger.info(f"Loading shard {i}/{num_shards}...")
                    with h5py.File(path, "r") as f:
                        n = len(f["input_ids"])
                        inputs = [None] * n
                        masks = [None] * n
                        for j in range(n):
                            # Store tokens as uint8 to minimize memory; ids are cast to torch.long at training time.
                            arr = np.array(f["input_ids"][j], dtype=np.uint8)
                            inputs[j] = arr
                            # Derive attention_mask on-the-fly as uint8 as well (1 for non-padding, 0 for padding).
                            masks[j] = (arr != 0).astype(np.uint8)
                        self._in_memory_shards.append((inputs, masks))

                # Build flat index for O(1) global access after sequential fallback
                self._build_flat_index()

        else:
            logger.info("Using streaming mode.")

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        shard_idx, local_idx = self._get_shard(idx)

        # In-memory fast path
        if self._in_memory:
            # If flat indexing exists, use it for true O(1) access
            if hasattr(self, "_flat_inputs") and hasattr(self, "_flat_masks"):
                # We have both the inputs and masks in memory, so we can return them directly
                input_ids = torch.tensor(self._flat_inputs[idx], dtype=torch.uint8)
                attention_mask = torch.tensor(self._flat_masks[idx], dtype=torch.uint8)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            inputs, masks = self._in_memory_shards[shard_idx]
            input_ids = torch.tensor(inputs[local_idx], dtype=torch.uint8)
            attention_mask = torch.tensor(masks[local_idx], dtype=torch.uint8)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        # Streaming path: use LRU file pool for efficient access
        pool = self._get_worker_pool()
        file_handle = pool.get_file(self.shard_paths[shard_idx])
        input_ids = torch.tensor(file_handle["input_ids"][local_idx], dtype=torch.uint8)

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

    def _build_flat_index(self) -> None:
        """Build flat input/mask lists from self._in_memory_shards for O(1) global indexing."""
        self._flat_inputs = []
        self._flat_masks = []
        for inputs, masks in self._in_memory_shards:
            self._flat_inputs.extend(inputs)
            self._flat_masks.extend(masks)
        logger.info(f"Flat indexing enabled: {len(self._flat_inputs):,} samples")

    def _get_worker_pool(self) -> ThreadSafeFileHandlePool:
        """
        Get or create the file handle pool for this worker.

        Lazily creates the pool on first access. This handles both single-process
        (num_workers=0) and multi-process cases. For multi-process, per-worker
        pools are created via worker_init_fn.

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

            if is_consecutive and len(local_indices) > 1:
                # Efficient slice read for consecutive indices
                start = local_indices[0]
                end = local_indices[-1] + 1
                batch_data = file_handle['input_ids'][start:end]

                for i, (global_idx, _) in enumerate(idx_pairs):
                    input_ids = torch.tensor(batch_data[i], dtype=torch.uint8)
                    attention_mask = (input_ids != 0).to(torch.uint8)
                    results[global_idx] = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                    }
            else:
                # Individual reads for non-consecutive indices
                for global_idx, local_idx in idx_pairs:
                    input_ids = torch.tensor(file_handle['input_ids'][local_idx], dtype=torch.uint8)
                    attention_mask = (input_ids != 0).to(torch.uint8)
                    results[global_idx] = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                    }

        # Return results in original order
        return {
            'input_ids': [results[idx]['input_ids'] for idx in indices],
            'attention_mask': [results[idx]['attention_mask'] for idx in indices],
        }


def process_shard(args):
    shard_idx, fasta_path, db_path, shard_keys, tokenizer, max_length, output_dir = args

    # Each process must open its own index
    index = SeqIO.index_db(db_path, [fasta_path], "fasta")

    shard_path = Path(output_dir) / f"shard_{shard_idx:04d}.h5"
    input_ids_list = []

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

        input_ids_list.append(encoding["input_ids"].squeeze(0))

    # Write results
    with h5py.File(shard_path, "w") as h5f:
        total = len(input_ids_list)
        h5f.create_dataset(
            "input_ids", (total,), dtype=h5py.special_dtype(vlen=np.uint8)
        )

        for i in tqdm(range(total), desc=f"Writing Shard {shard_idx}", leave=False):
            h5f["input_ids"][i] = np.array(input_ids_list[i], dtype=np.uint8)

    index.close()
    return str(shard_path)

def _read_shard_for_worker(path_str: str):
    """Helper for parallel shard loading in separate processes.

    Returns a tuple (indexable_path_str, (inputs_list, masks_list)).
    """

    path = Path(path_str)
    inputs = []
    masks = []
    with h5py.File(path, "r") as f:
        n = len(f["input_ids"])
        # Bulk read each variable-length element into a numpy array and derive attention masks
        for i in range(n):
            # Store tokens as uint8 to minimize memory; ids are cast to torch.long at training time.
            arr = np.array(f["input_ids"][i], dtype=np.uint8)
            inputs.append(arr)
            # Derive attention_mask on-the-fly as uint8 (1 for non-padding, 0 for padding).
            masks.append((arr != 0).astype(np.uint8))

    return path_str, (inputs, masks)


def _pretraining_worker_init_fn(worker_id):
    """
    Worker initialization function for PyTorch DataLoader.

    Initializes per-worker state for multi-process data loading, including:
    - Creating separate file handle pools for each worker
    - Seeding RNGs for reproducibility

    This is a module-level function (not a closure) to support pickling.

    Args:
        worker_id: Worker ID assigned by DataLoader
    """
    import torch.utils.data as data_utils

    worker_info = data_utils.get_worker_info()
    if worker_info is None:
        # Single-process loading (num_workers=0)
        return

    # Create per-worker file handle pool
    worker_dataset = worker_info.dataset
    worker_dataset._worker_pool = ThreadSafeFileHandlePool(
        max_open_files=worker_dataset.max_open_files
    )

    # Seed RNGs for reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    logger.debug(
        f"Worker {worker_id} initialized with seed {worker_seed}, "
        f"max_open_files={worker_dataset.max_open_files}"
    )


def get_pretraining_worker_init_fn(dataset=None):
    """
    Get worker initialization function for PyTorch DataLoader.

    Returns the module-level worker initialization function that can be pickled
    for multi-process data loading.

    Args:
        dataset: Unused, kept for backward compatibility

    Returns:
        Callable: Function to pass to DataLoader's worker_init_fn parameter

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = LoadShardedFastaMLMDataset("data/shards")
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=get_pretraining_worker_init_fn()
        ... )
    """
    return _pretraining_worker_init_fn


def benchmark_read_modes(dataset, n_samples: int = 100, n_iterations: int = 3):
    """
    Compare sequential vs vectorized read performance.

    Benchmarks the performance difference between using individual __getitem__
    calls and the vectorized read_batch_vectorized method. Useful for evaluating
    whether vectorized reads provide speedup for your specific access patterns.

    Args:
        dataset: LoadShardedFastaMLMDataset instance
        n_samples: Number of samples to read per iteration
        n_iterations: Number of benchmark iterations

    Returns:
        dict: Performance statistics including:
            - sequential_avg: Average time for sequential reads (seconds)
            - vectorized_avg: Average time for vectorized reads (seconds)
            - speedup: Speedup ratio (sequential / vectorized)

    Example:
        >>> dataset = LoadShardedFastaMLMDataset("data/shards")
        >>> stats = benchmark_read_modes(dataset, n_samples=100, n_iterations=3)
        >>> print(f"Speedup: {stats['speedup']:.2f}x")

    Note:
        Speedup depends on access patterns. Consecutive indices within the same
        shard benefit most from vectorized reads (2-3x speedup). Random access
        may show less improvement due to overhead.
    """
    import time

    # Sequential reads
    seq_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        for i in range(n_samples):
            _ = dataset[i]
        seq_times.append(time.perf_counter() - start)

    # Vectorized reads (consecutive indices)
    vec_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = dataset.read_batch_vectorized(list(range(n_samples)))
        vec_times.append(time.perf_counter() - start)

    seq_avg = sum(seq_times) / len(seq_times)
    vec_avg = sum(vec_times) / len(vec_times)

    return {
        'sequential_avg': seq_avg,
        'vectorized_avg': vec_avg,
        'speedup': seq_avg / (vec_avg + 1e-9)
    }