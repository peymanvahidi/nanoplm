import re
import h5py
import torch
import threading
import numpy as np

from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Optional, Dict
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, IterableDataset
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from nanoplm.distillation.models.teacher import BaseTeacher
from nanoplm.utils import logger, get_device, create_dirs
from nanoplm.data.file_pool import ThreadSafeFileHandlePool, detect_file_limits


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


class SaveKDDataset(Dataset):
    def __init__(
        self,
        input_fasta: Union[str, Path],
        output_path: Union[str, Path],
        teacher: BaseTeacher,
        mode: str,
        max_seq_len: int,
        batch_size: int,
        device: str,
        skip_n: int = 0,
        samples_per_shard: int = 10000,
        force: bool = False,
    ):
        """
        Save knowledge distillation dataset with teacher embeddings.

        Args:
            input_fasta: Path to input FASTA file
            output_path: Path to output H5 file (or prefix for sharded files)
            teacher: Teacher model for generating embeddings
            mode: Processing mode (currently only "get_embeddings")
            max_seq_len: Maximum sequence length
            batch_size: Batch size for teacher embedding calculation
            device: Device to use ("auto", "cuda", "mps", "cpu")
            skip_n: Number of sequences to skip from the beginning
            samples_per_shard: Number of samples per shard file (-1 for single file)
            force: Force overwrite existing files
        """
        self.input_fasta = Path(input_fasta)
        self.output_path = Path(output_path)
        self.teacher = teacher
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device if device != "auto" else get_device()
        self.skip_n = skip_n

        # Validate samples_per_shard
        if samples_per_shard < -1 or samples_per_shard == 0:
            raise ValueError(
                f"samples_per_shard must be -1 (no sharding) or a positive number, got {samples_per_shard}"
            )

        self.samples_per_shard = samples_per_shard
        self.force = force
        self.data_gen = None
        self._cached_len = None

    def __len__(self):
        if self._cached_len is None:
            self._cached_len = max(
                0, sum(1 for _ in SeqIO.parse(self.input_fasta, "fasta")) - self.skip_n
            )
        return self._cached_len

    def _load(self):
        if not self.data_gen:
            raw_generator = SeqIO.parse(self.input_fasta, "fasta")

            if self.skip_n > 0:
                logger.info(
                    f"Skipping first {self.skip_n} sequences from {self.input_fasta}."
                )
                for _ in range(self.skip_n):
                    try:
                        next(raw_generator)
                    except StopIteration:
                        logger.warning(
                            f"Tried to skip {self.skip_n} sequences, but FASTA file has fewer."
                        )
                        break

            self.data_gen = ((record.id, str(record.seq)) for record in raw_generator)
            logger.info(
                f"{self.input_fasta} initialized (with skip_n={self.skip_n}). Now ready for processing."
            )

    def process_dataset(self) -> Union[Path, List[Path]]:
        self._load()

        if self.mode == "get_embeddings":
            self.teacher_model = self.teacher.encoder_model
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.teacher_tokenizer = self.teacher.tokenizer
        create_dirs(self.output_path)

        total_sequences_in_fasta = self.__len__() + self.skip_n
        total_to_process = self.__len__()

        # Log dataset summary
        skip_info = f" (skipping {self.skip_n})" if self.skip_n > 0 else ""

        logger.info(
            f"Dataset: {total_sequences_in_fasta:,} sequences → {total_to_process:,} to process{skip_info}"
        )

        # Determine if sharding is needed based on samples_per_shard
        if self.samples_per_shard == -1 or total_to_process <= self.samples_per_shard:
            # Single file mode
            logger.info(f"Single file will be created at {self.output_path}")
            return self._process_dataset_single()
        else:
            # Calculate number of shards needed
            import math
            self.n_files = math.ceil(total_to_process / self.samples_per_shard)
            logger.info(
                f"{self.n_files} sharded files will be created "
                f"(~{self.samples_per_shard:,} samples per shard) at {self.output_path.parent}"
            )
            return self._process_dataset_sharded()

    def _process_dataset_single(self) -> Path:
        """Process dataset into a single H5 file (original behavior)"""

        batch = []

        if self.output_path.exists():
            if self.force:
                logger.info(
                    f"Found existing HDF5 file at {self.output_path}. Will overwrite existing file."
                )
                self.output_path.unlink()
            else:
                raise FileExistsError(
                    f"Found existing HDF5 file at {self.output_path}. Use --force to overwrite existing file."
                )
        else:
            logger.info(
                f"No existing HDF5 file at {self.output_path}. Creating new file."
            )

        processed_sequences = 0
        with h5py.File(self.output_path, "w") as h5f:

            with tqdm(
                total=self.__len__(), desc="Generating embeddings", unit="seq"
            ) as pbar:
                for _, sequence in self.data_gen:
                    teacher_seq = self.teacher.preprocess(sequence)
                    batch.append(teacher_seq)

                    # Process the batch if it's full
                    if len(batch) == self.batch_size:
                        self._process_and_save_batch(
                            h5f=h5f, batch=batch, start_index=processed_sequences
                        )
                        processed_sequences += len(batch)
                        pbar.update(len(batch))
                        batch = []

                # Process any remaining sequences in the last batch
                if batch:
                    self._process_and_save_batch(
                        h5f=h5f, batch=batch, start_index=processed_sequences
                    )
                    processed_sequences += len(batch)
                    pbar.update(len(batch))

        logger.info(f"Processed and saved {processed_sequences} new sequences.")
        logger.info(f"Dataset: {self.output_path}.")
        return self.output_path

    def _process_dataset_sharded(self) -> List[Path]:
        """Process dataset into multiple sharded H5 files based on samples_per_shard"""

        # Generate shard file names
        base_name = self.output_path.stem
        output_dir = self.output_path.parent
        shard_paths = [
            output_dir / f"{base_name}_shard_{i}.h5" for i in range(self.n_files)
        ]

        # Check for existing files and handle them
        for shard_path in shard_paths:
            if shard_path.exists():
                if self.force:
                    logger.info(
                        f"Found existing sharded file at {shard_path}. Will overwrite existing file."
                    )
                    shard_path.unlink()
                else:
                    raise FileExistsError(
                        f"Found existing sharded file at {shard_path}. Use --force to overwrite existing file."
                    )
            else:
                logger.info(f"Creating new shard file: {shard_path.name}")

        # Process sequences and write to one shard file at a time
        batch = []
        current_shard_idx = 0
        current_shard_count = 0
        processed_sequences = 0
        current_h5_file = None

        try:
            with tqdm(
                total=self.__len__(), desc="Generating sharded embeddings", unit="seq"
            ) as pbar:
                for _, sequence in self.data_gen:
                    # Check if we need to open the first file
                    if current_h5_file is None:
                        current_h5_file = h5py.File(shard_paths[current_shard_idx], "w")
                        logger.info(
                            f"Opened shard file: {shard_paths[current_shard_idx].name}"
                        )

                    teacher_seq = self.teacher.preprocess(sequence)
                    batch.append(teacher_seq)

                    if len(batch) == self.batch_size:
                        # Check if current shard is full and we're not on the last shard
                        if (
                            current_shard_count + len(batch) > self.samples_per_shard
                            and current_shard_idx < self.n_files - 1
                        ):
                            # Close current file and move to next
                            current_h5_file.close()
                            logger.info(
                                f"Closed shard file: {shard_paths[current_shard_idx].name} with {current_shard_count} sequences"
                            )

                            current_shard_idx += 1
                            current_shard_count = 0

                            # Open next shard file
                            current_h5_file = h5py.File(
                                shard_paths[current_shard_idx], "w"
                            )
                            logger.info(
                                f"Opened shard file: {shard_paths[current_shard_idx].name}"
                            )

                        # Process the batch
                        self._process_and_save_batch(
                            h5f=current_h5_file,
                            batch=batch,
                            start_index=current_shard_count,
                        )
                        current_shard_count += len(batch)
                        processed_sequences += len(batch)
                        pbar.update(len(batch))
                        batch = []

                # Process any remaining sequences in the last batch
                if batch:
                    self._process_and_save_batch(
                        h5f=current_h5_file,
                        batch=batch,
                        start_index=current_shard_count,
                    )
                    processed_sequences += len(batch)
                    pbar.update(len(batch))

        finally:
            # Close the currently open file
            if current_h5_file is not None:
                current_h5_file.close()
                logger.info(
                    f"Closed final shard file: {shard_paths[current_shard_idx].name}"
                )

        # Log shard information with progress summary
        logger.info("Sharded processing completed! Summary:")
        total_sequences_after = 0
        created_shard_paths = []

        for i, shard_path in enumerate(shard_paths):
            if shard_path.exists():
                with h5py.File(shard_path, "r") as shard_file:
                    shard_size = len(shard_file.keys())
                    file_size_gb = shard_path.stat().st_size / (1024**3)
                    total_sequences_after += shard_size
                    logger.info(
                        f"  Shard {i:2d}: {shard_size:8,} sequences, {file_size_gb:6.1f} GB"
                    )
                    created_shard_paths.append(shard_path)
            else:
                logger.info(f"  Shard {i:2d}: 0 sequences (not created)")

        logger.info(f"Processed and saved {processed_sequences} new sequences.")
        logger.info(f"Total sequences across all shards: {total_sequences_after:,}")
        logger.info(
            f"Successfully created {len(created_shard_paths)} out of {len(shard_paths)} shard files"
        )

        return created_shard_paths

    def _process_and_save_batch(
        self,
        h5f: h5py.File,
        batch: List[str],
        start_index: int,
    ):
        """
        Process a batch of sequences and save to H5 file.

        IMPORTANT: Stores STUDENT-tokenized input_ids (for student model),
        but computes teacher embeddings using teacher tokenization.
        """
        from nanoplm.distillation.models.student.tokenizer import ProtXTokenizer

        # Teacher tokenization for computing embeddings
        teacher_encoding = self.teacher_tokenizer.batch_encode_plus(
            batch,  # Preprocessed sequences like "M K T"
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # Student tokenization for storage (what student model will receive)
        # Convert preprocessed ("M K T") back to raw ("MKT")
        raw_sequences = [seq.replace(" ", "") for seq in batch]
        student_tokenizer = ProtXTokenizer()
        student_encoding = student_tokenizer.batch_encode_plus(
            raw_sequences,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # Compute teacher embeddings using teacher tokenization
        with torch.no_grad():
            teacher_input_ids = teacher_encoding["input_ids"].to(self.device)
            teacher_attention_mask = teacher_encoding["attention_mask"].to(self.device)

            teacher_embeddings = self.teacher_model(
                input_ids=teacher_input_ids, attention_mask=teacher_attention_mask
            ).last_hidden_state

        # Save STUDENT-tokenized input_ids and attention_mask
        # Save TEACHER embeddings (target for student to match)
        for i, (seq_input_ids, seq_attention_mask, seq_teacher_embeddings) in enumerate(
            zip(
                student_encoding["input_ids"].cpu().numpy(),
                student_encoding["attention_mask"].cpu().numpy(),
                teacher_embeddings.cpu().numpy(),
            )
        ):
            grp = h5f.create_group(str(start_index + i))
            grp.create_dataset(
                "input_ids",
                data=seq_input_ids.astype(np.int8),
            )
            grp.create_dataset(
                "attention_mask",
                data=seq_attention_mask.astype(np.int8),
            )
            grp.create_dataset(
                "teacher_embeddings",
                data=seq_teacher_embeddings.astype(np.float16),
            )


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


def shard_h5_file(
    input_h5_path: Union[str, Path],
    n_sharded_files: int,
    output_dir: Optional[Union[str, Path]] = None,
    total_sequences: Optional[int] = None,
) -> List[Path]:
    """
    Split a large H5 file into smaller sharded files.

    Args:
        input_h5_path: Path to the input H5 file to shard
        n_sharded_files: Number of shard files to create
        output_dir: Directory to save sharded files (defaults to same as input file)
        total_sequences: Total number of sequences (if known, skips counting)

    Returns:
        List of paths to the created shard files
    """
    input_path = Path(input_h5_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate shard file names
    base_name = input_path.stem  # e.g., "train" from "train.h5"
    shard_paths = [
        output_dir / f"{base_name}_shard_{i}.h5" for i in range(n_sharded_files)
    ]

    # Get input file size for progress display
    input_file_size_gb = input_path.stat().st_size / (1024**3)

    logger.info(
        f"Starting to shard {input_path} ({input_file_size_gb:.1f} GB) into {n_sharded_files} files..."
    )

    # Open input file and get total size
    with h5py.File(input_path, "r") as input_h5:
        if total_sequences is not None:
            logger.info(f"Using provided sequence count: {total_sequences:,}")
            sequences_count = total_sequences
        else:
            logger.info(
                "Counting sequences in H5 file (this may take a while for large files)..."
            )
            sequences_count = len(input_h5.keys())
            logger.info(f"Total sequences: {sequences_count:,}")

        sequences_per_shard = sequences_count // n_sharded_files
        logger.info(f"Sequences per shard: {sequences_per_shard:,}")

        # Create shard files
        shard_files = [h5py.File(path, "w") for path in shard_paths]

        try:
            current_shard_idx = 0
            current_shard_count = 0

            # Enhanced progress bar with more information
            with tqdm(
                total=sequences_count,
                desc=f"Sharding {base_name}.h5",
                unit="seq",
                unit_scale=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                dynamic_ncols=True,
            ) as pbar:

                for seq_idx in range(sequences_count):
                    # Move to next shard if current one is full (except for last shard)
                    if (
                        current_shard_count >= sequences_per_shard
                        and current_shard_idx < n_sharded_files - 1
                    ):
                        current_shard_idx += 1
                        current_shard_count = 0

                    # Copy sequence data to current shard
                    source_group = input_h5[str(seq_idx)]
                    target_group = shard_files[current_shard_idx].create_group(
                        str(current_shard_count)
                    )

                    # Copy all datasets from source to target
                    for dataset_name in source_group.keys():
                        target_group.create_dataset(
                            dataset_name, data=source_group[dataset_name][:]
                        )

                    current_shard_count += 1

                    # Update progress bar with additional info
                    percentage = (seq_idx + 1) / sequences_count * 100
                    pbar.set_postfix(
                        {
                            "Shard": f"{current_shard_idx}/{n_sharded_files-1}",
                            "Progress": f"{percentage:.1f}%",
                        }
                    )
                    pbar.update(1)

        finally:
            # Close all shard files
            for shard_file in shard_files:
                shard_file.close()

    # Log shard information with progress summary
    logger.info("Sharding completed! Summary:")
    total_output_size_gb = 0
    for i, shard_path in enumerate(shard_paths):
        with h5py.File(shard_path, "r") as shard_file:
            shard_size = len(shard_file.keys())
            file_size_gb = shard_path.stat().st_size / (1024**3)
            total_output_size_gb += file_size_gb
            logger.info(
                f"  Shard {i:2d}: {shard_size:8,} sequences, {file_size_gb:6.1f} GB"
            )

    logger.info(f"Total output size: {total_output_size_gb:.1f} GB")
    logger.info(f"Successfully created {len(shard_paths)} shard files")
    return shard_paths
