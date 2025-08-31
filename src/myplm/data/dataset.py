import re
import h5py
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Optional, Dict
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, IterableDataset
from collections import OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from myplm.models.teacher import ProtT5
from myplm.utils import logger, get_device

class ProtXDataGen(IterableDataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        teacher_tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        device: str
    ):
        self.data_path = Path(data_path)
        self.tokenizer = teacher_tokenizer
        self.device = device
        self.max_seq_len = max_seq_len

    def __iter__(self):
        data_gen = (
            (record.id, str(record.seq)) 
            for record in SeqIO.parse(self.data_path, "fasta")
        )
        
        for _, sequence in data_gen:
            teacher_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
            tokenized_seq = self.tokenizer.encode_plus(
                teacher_seq,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt"
            )
            yield {
                "input_ids": tokenized_seq["input_ids"].squeeze(0),
                "attention_mask": tokenized_seq["attention_mask"].squeeze(0)
            }

class ProtXDataProcessor(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        teacher_model: ProtT5,
        max_seq_len: int,
        batch_size: int,
        device: str,
        skip_n: int = 0,
        n_files: int = 1
    ):
        self.data_path = Path(data_path)
        self.teacher = teacher_model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device if device != "auto" else get_device()
        self.skip_n = skip_n
        self.n_files = n_files
        self._loaded = False

    def _load(self):
        if not self._loaded:
            raw_generator = SeqIO.parse(self.data_path, "fasta")
            
            if self.skip_n > 0:
                logger.info(f"Skipping first {self.skip_n} sequences from {self.data_path}.")
                for _ in range(self.skip_n):
                    try:
                        next(raw_generator)
                    except StopIteration:
                        logger.warning(f"Tried to skip {self.skip_n} sequences, but FASTA file has fewer.")
                        break
            
            self.data_gen = (
                (record.id, str(record.seq)) 
                for record in raw_generator
            )
            self._loaded = True
            logger.info(f"{self.data_path} initialized (with skip_n={self.skip_n}). Now ready for processing.")

    def __len__(self):
        if not hasattr(self, "_cached_len"):
            self._cached_len = max(0, sum(1 for _ in SeqIO.parse(self.data_path, "fasta")) - self.skip_n)
        return self._cached_len
    
    def process_dataset(self, save_path: Path) -> Union[Path, List[Path]]:
        self._load()

        teacher_tokenizer = self.teacher.tokenizer
        teacher_model = self.teacher.encoder_model
        
        total_sequences_in_fasta = sum(1 for _ in SeqIO.parse(self.data_path, "fasta"))
        num_sequences_to_process = total_sequences_in_fasta - self.skip_n
        if num_sequences_to_process < 0:
            num_sequences_to_process = 0
        
        logger.info(f"Found {total_sequences_in_fasta} total sequences in {self.data_path}.")
        if self.skip_n > 0:
            logger.info(f"Skipping {self.skip_n} sequences. Attempting to process {num_sequences_to_process} sequences.")
        else:
            logger.info(f"Processing {num_sequences_to_process} sequences.")

        if self.n_files > 1:
            logger.info(f"Creating {self.n_files} sharded files during processing.")
            return self._process_dataset_sharded(save_path, teacher_tokenizer, teacher_model, num_sequences_to_process)
        else:
            return self._process_dataset_single(save_path, teacher_tokenizer, teacher_model, num_sequences_to_process)
    
    def _process_dataset_single(self, save_path: Path, teacher_tokenizer, teacher_model, num_sequences_to_process: int) -> Path:
        """Process dataset into a single H5 file (original behavior)"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        batch = []
        current_h5_offset = 0
        mode = "w"

        if save_path.exists():
            logger.info(f"Found existing HDF5 file at {save_path}. Will append new data.")
            mode = "a"
            try:
                with h5py.File(save_path, "r") as h5f_check:
                    current_h5_offset = len(h5f_check.keys())
                logger.info(f"Existing HDF5 file contains {current_h5_offset} entries. New entries will start from this index.")
            except Exception as e:
                logger.error(f"Error reading existing HDF5 file {save_path} to determine offset. Please check the file. Error: {e}")
                raise
        else:
            logger.info(f"No existing HDF5 file at {save_path}. Creating new file.")

        sequences_actually_processed_this_run = 0
        with h5py.File(save_path, mode) as h5f:
            _current_fasta_iter = SeqIO.parse(self.data_path, "fasta")
            if self.skip_n > 0:
                for _ in range(self.skip_n):
                    try: next(_current_fasta_iter)
                    except StopIteration: break
            
            processing_generator = (
                (record.id, str(record.seq)) 
                for record in _current_fasta_iter
            )

            with tqdm(total=num_sequences_to_process, desc="Generating embeddings", unit="seq") as pbar:
                for _, sequence in processing_generator:
                    teacher_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                    batch.append(teacher_seq)

                    if len(batch) == self.batch_size:
                        self._process_and_save_batch(
                            h5f, batch, teacher_tokenizer, teacher_model, current_h5_offset + sequences_actually_processed_this_run
                        )
                        sequences_actually_processed_this_run += len(batch)
                        pbar.update(len(batch))
                        batch = []
                
                if batch:
                    self._process_and_save_batch(
                        h5f, batch, teacher_tokenizer, teacher_model, current_h5_offset + sequences_actually_processed_this_run
                    )
                    sequences_actually_processed_this_run += len(batch)
                    pbar.update(len(batch))
        
        total_entries_in_h5_after_run = 0
        with h5py.File(save_path, "r") as h5f_final_check:
            total_entries_in_h5_after_run = len(h5f_final_check.keys())

        logger.info(f"Processed and saved {sequences_actually_processed_this_run} new sequences.")
        logger.info(f"Dataset now contains {total_entries_in_h5_after_run} total sequences at {save_path}.")
        return save_path
    
    def _process_dataset_sharded(self, save_path: Path, teacher_tokenizer, teacher_model, num_sequences_to_process: int) -> List[Path]:
        """Process dataset into multiple sharded H5 files"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate shard file names
        base_name = save_path.stem
        output_dir = save_path.parent
        shard_paths = [
            output_dir / f"{base_name}_shard_{i}.h5" 
            for i in range(self.n_files)
        ]
        
        # Calculate sequences per shard
        sequences_per_shard = num_sequences_to_process // self.n_files
        # Add remaining sequences to the last shard
        sequences_in_last_shard = sequences_per_shard + (num_sequences_to_process % self.n_files)
        logger.info(f"Sequences per shard: {sequences_per_shard:,}")
        if num_sequences_to_process % self.n_files > 0:
            logger.info(f"Last shard will contain: {sequences_in_last_shard:,} sequences")
        
        # Check for existing files and calculate starting offsets
        shard_offsets = []
        for i, shard_path in enumerate(shard_paths):
            if shard_path.exists():
                logger.warning(f"Found existing sharded file at {shard_path}. Will append new data.")
                try:
                    with h5py.File(shard_path, "r") as h5f_check:
                        offset = len(h5f_check.keys())
                        shard_offsets.append(offset)
                        logger.info(f"Existing shard {shard_path.name} contains {offset} entries.")
                except Exception as e:
                    logger.error(f"Error reading existing shard {shard_path}. Please check the file. Error: {e}")
                    raise
            else:
                shard_offsets.append(0)
                logger.info(f"Creating new shard file: {shard_path.name}")
        
        # Process sequences and write to one shard file at a time
        batch = []
        current_shard_idx = 0
        current_shard_count = shard_offsets[0]
        sequences_actually_processed_this_run = 0
        current_h5_file = None
        
        try:
            _current_fasta_iter = SeqIO.parse(self.data_path, "fasta")
            if self.skip_n > 0:
                for _ in range(self.skip_n):
                    try: next(_current_fasta_iter)
                    except StopIteration: break
            
            processing_generator = (
                (record.id, str(record.seq)) 
                for record in _current_fasta_iter
            )

            with tqdm(total=num_sequences_to_process, desc="Generating sharded embeddings", unit="seq") as pbar:
                for seq_idx, (_, sequence) in enumerate(processing_generator):
                    # Check if we need to open the first file
                    if current_h5_file is None:
                        # Open current shard file
                        mode = "a" if shard_offsets[current_shard_idx] > 0 else "w"
                        current_h5_file = h5py.File(shard_paths[current_shard_idx], mode)
                        logger.info(f"Opened shard file: {shard_paths[current_shard_idx].name}")
                    
                    teacher_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                    batch.append(teacher_seq)

                    if len(batch) == self.batch_size:
                        # Check if processing this batch would exceed the shard limit
                        current_shard_target = sequences_in_last_shard if current_shard_idx == self.n_files - 1 else sequences_per_shard
                        
                        # If current shard would be overfilled and we're not on the last shard, switch
                        if (current_shard_count + len(batch) > current_shard_target and 
                            current_shard_idx < self.n_files - 1):
                            
                            # Close current file and move to next
                            current_h5_file.close()
                            logger.info(f"Closed shard file: {shard_paths[current_shard_idx].name} with {current_shard_count} sequences")
                            
                            current_shard_idx += 1
                            current_shard_count = shard_offsets[current_shard_idx]
                            
                            # Open next shard file
                            mode = "a" if shard_offsets[current_shard_idx] > 0 else "w"
                            current_h5_file = h5py.File(shard_paths[current_shard_idx], mode)
                            logger.info(f"Opened shard file: {shard_paths[current_shard_idx].name}")
                        
                        # Process the batch
                        self._process_and_save_batch(
                            current_h5_file, batch, teacher_tokenizer, teacher_model, current_shard_count
                        )
                        current_shard_count += len(batch)
                        sequences_actually_processed_this_run += len(batch)
                        pbar.update(len(batch))
                        batch = []
                
                # Process any remaining sequences in the last batch
                if batch:
                    self._process_and_save_batch(
                        current_h5_file, batch, teacher_tokenizer, teacher_model, current_shard_count
                    )
                    sequences_actually_processed_this_run += len(batch)
                    pbar.update(len(batch))
        
        finally:
            # Close the currently open file
            if current_h5_file is not None:
                current_h5_file.close()
                logger.info(f"Closed final shard file: {shard_paths[current_shard_idx].name}")
        
        # Log shard information with progress summary
        logger.info("Sharded processing completed! Summary:")
        total_sequences_after = 0
        created_shard_paths = []
        
        for i, shard_path in enumerate(shard_paths):
            if shard_path.exists():
                with h5py.File(shard_path, "r") as shard_file:
                    shard_size = len(shard_file.keys())
                    file_size_gb = shard_path.stat().st_size / (1024 ** 3)
                    total_sequences_after += shard_size
                    logger.info(f"  Shard {i:2d}: {shard_size:8,} sequences, {file_size_gb:6.1f} GB")
                    created_shard_paths.append(shard_path)
            else:
                logger.info(f"  Shard {i:2d}: 0 sequences (not created)")
        
        logger.info(f"Processed and saved {sequences_actually_processed_this_run} new sequences.")
        logger.info(f"Total sequences across all shards: {total_sequences_after:,}")
        logger.info(f"Successfully created {len(created_shard_paths)} out of {len(shard_paths)} shard files")
        
        return created_shard_paths
    
    def _process_and_save_batch(
        self,
        h5f: h5py.File,
        batch: List[str],
        teacher_tokenizer: PreTrainedTokenizer,
        teacher_model,
        start_index: int
    ):
        batch_encoding = teacher_tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            input_ids = batch_encoding["input_ids"].to(self.device)
            attention_mask = batch_encoding["attention_mask"].to(self.device)
            
            teacher_embeddings = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
        
        for i, (seq_input_ids, seq_attention_mask, seq_teacher_embeddings) in enumerate(
            zip(input_ids.cpu().numpy(), attention_mask.cpu().numpy(), teacher_embeddings.cpu().numpy())
        ):
            grp = h5f.create_group(str(start_index + i))
            grp.create_dataset("input_ids", data=seq_input_ids.astype(np.int8))
            grp.create_dataset("attention_mask", data=seq_attention_mask.astype(np.int8))
            grp.create_dataset("teacher_embeddings", data=seq_teacher_embeddings.astype(np.float16))

def shard_h5_file(
    input_h5_path: Union[str, Path], 
    n_sharded_files: int,
    output_dir: Optional[Union[str, Path]] = None,
    total_sequences: Optional[int] = None
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
        output_dir / f"{base_name}_shard_{i}.h5" 
        for i in range(n_sharded_files)
    ]
    
    # Get input file size for progress display
    input_file_size_gb = input_path.stat().st_size / (1024 ** 3)
    
    logger.info(f"Starting to shard {input_path} ({input_file_size_gb:.1f} GB) into {n_sharded_files} files...")
    
    # Open input file and get total size
    with h5py.File(input_path, "r") as input_h5:
        if total_sequences is not None:
            logger.info(f"Using provided sequence count: {total_sequences:,}")
            sequences_count = total_sequences
        else:
            logger.info("Counting sequences in H5 file (this may take a while for large files)...")
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
                dynamic_ncols=True
            ) as pbar:
                
                for seq_idx in range(sequences_count):
                    # Move to next shard if current one is full (except for last shard)
                    if (current_shard_count >= sequences_per_shard and 
                        current_shard_idx < n_sharded_files - 1):
                        current_shard_idx += 1
                        current_shard_count = 0
                    
                    # Copy sequence data to current shard
                    source_group = input_h5[str(seq_idx)]
                    target_group = shard_files[current_shard_idx].create_group(str(current_shard_count))
                    
                    # Copy all datasets from source to target
                    for dataset_name in source_group.keys():
                        target_group.create_dataset(
                            dataset_name, 
                            data=source_group[dataset_name][:]
                        )
                    
                    current_shard_count += 1
                    
                    # Update progress bar with additional info
                    percentage = (seq_idx + 1) / sequences_count * 100
                    pbar.set_postfix({
                        'Shard': f"{current_shard_idx}/{n_sharded_files-1}",
                        'Progress': f"{percentage:.1f}%"
                    })
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
            file_size_gb = shard_path.stat().st_size / (1024 ** 3)
            total_output_size_gb += file_size_gb
            logger.info(f"  Shard {i:2d}: {shard_size:8,} sequences, {file_size_gb:6.1f} GB")
    
    logger.info(f"Total output size: {total_output_size_gb:.1f} GB")
    logger.info(f"Successfully created {len(shard_paths)} shard files")
    return shard_paths

class ProtXDataLoader(Dataset):
    def __init__(
        self, 
        h5_path: Union[str, Path],
        device: str,
        seed: Optional[int] = None,
        sharded: bool = False
    ):
        self.h5_path = Path(h5_path)
        self.device = device
        self.seed = seed
        self.sharded = sharded
        
        if self.sharded:
            self._load_sharded_files()
        else:
            self.h5f = h5py.File(self.h5_path, "r")
            self.total_size = len(self.h5f.keys())
        
        self.indices = list(range(self.total_size))
        
        if self.seed is not None:
            self._shuffle_indices()
    
    def _load_sharded_files(self):
        """Load multiple shard files based on the base path"""
        base_name = self.h5_path.stem
        parent_dir = self.h5_path.parent
        
        # Find all shard files
        shard_pattern = f"{base_name}_shard_*.h5"
        shard_files = sorted(parent_dir.glob(shard_pattern))
        
        if not shard_files:
            raise FileNotFoundError(f"No shard files found matching pattern: {parent_dir / shard_pattern}")
        
        logger.info(f"Found {len(shard_files)} shard files")
        
        # Open all shard files
        self.shard_files = [h5py.File(shard_path, "r") for shard_path in shard_files]
        
        # Build cumulative index to map global index to (shard_idx, local_idx)
        self.shard_sizes = [len(shard_file.keys()) for shard_file in self.shard_files]
        self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)
        self.total_size = sum(self.shard_sizes)
        
        logger.info(f"Total sequences across shards: {self.total_size}")
        for i, size in enumerate(self.shard_sizes):
            logger.info(f"Shard {i}: {size} sequences")
    
    def _shuffle_indices(self):
        """Shuffle the indices based on seed"""
        rng = np.random.RandomState(self.seed)
        rng.shuffle(self.indices)
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx >= self.total_size or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")
        
        actual_idx = self.indices[idx]
        
        if self.sharded:
            # Find which shard contains this index
            shard_idx = np.searchsorted(self.cumulative_sizes[1:], actual_idx, side='right')
            local_idx = actual_idx - self.cumulative_sizes[shard_idx]
            grp = self.shard_files[shard_idx][str(local_idx)]
        else:
            grp = self.h5f[str(actual_idx)]
        
        input_ids = torch.tensor(grp["input_ids"][:], dtype=torch.long)
        attention_mask = torch.tensor(grp["attention_mask"][:], dtype=torch.long)
        teacher_embeddings = torch.tensor(grp["teacher_embeddings"][:], dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_embeddings": teacher_embeddings
        }
    
    def __del__(self):
        """Clean up file handles"""
        if self.sharded and hasattr(self, 'shard_files'):
            for shard_file in self.shard_files:
                shard_file.close()
        elif hasattr(self, 'h5f'):
            self.h5f.close()


class ProtXDataLoaderOptimized(Dataset):
    """
    Optimized ProtX Data Loader for large datasets with:
    - LRU cache for shard files (memory efficient)
    - Chunked reading for better I/O performance  
    - Optional prefetching with threading
    - Reduced memory footprint
    """
    
    def __init__(
        self, 
        h5_path: Union[str, Path],
        device: str,
        seed: Optional[int] = None,
        sharded: bool = False,
        max_open_files: int = 5,      # NEW: Limit open files
        chunk_size: int = 32,         # NEW: Read multiple samples at once
        prefetch_batches: int = 2,    # NEW: Background prefetching  
        use_threading: bool = True    # NEW: Threading for I/O
    ):
        self.h5_path = Path(h5_path)
        self.device = device
        self.seed = seed
        self.sharded = sharded
        self.max_open_files = max_open_files
        self.chunk_size = chunk_size
        self.prefetch_batches = prefetch_batches
        self.use_threading = use_threading
        
        # LRU cache for open files
        self._file_cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()
        
        # Prefetch cache for chunks
        self._prefetch_cache: Dict[int, List[Dict]] = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_executor = None
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._prefetch_hits = 0
        
        if self.sharded:
            self._load_sharded_files_optimized()
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
            self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ProtX-Prefetch")
            
        logger.info(f"ProtXDataLoaderOptimized initialized:")
        logger.info(f"  - Max open files: {self.max_open_files}")
        logger.info(f"  - Chunk size: {self.chunk_size}")  
        logger.info(f"  - Prefetch batches: {self.prefetch_batches}")
        logger.info(f"  - Threading enabled: {self.use_threading}")
    
    def _get_file_size(self, path: Path) -> int:
        """Get number of sequences in H5 file without keeping it open"""
        with h5py.File(path, "r") as f:
            return len(f.keys())
    
    def _load_sharded_files_optimized(self):
        """Load shard metadata without opening all files immediately"""
        base_name = self.h5_path.stem
        parent_dir = self.h5_path.parent
        
        # Find all shard files
        shard_pattern = f"{base_name}_shard_*.h5"
        self.shard_paths = sorted(parent_dir.glob(shard_pattern))
        
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found matching pattern: {parent_dir / shard_pattern}")
        
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
    
    def _get_shard_file(self, shard_idx: int) -> h5py.File:
        """Get shard file with LRU caching"""
        with self._cache_lock:
            # Check if file is already open
            if shard_idx in self._file_cache:
                # Move to end (most recently used)
                self._file_cache.move_to_end(shard_idx)
                self._cache_hits += 1
                return self._file_cache[shard_idx]
            
            self._cache_misses += 1
            
            # Need to open new file
            if len(self._file_cache) >= self.max_open_files:
                # Remove least recently used file
                old_idx, old_file = self._file_cache.popitem(last=False)
                old_file.close()
                logger.debug(f"Closed LRU shard file {old_idx}")
            
            # Open new file
            new_file = h5py.File(self.shard_paths[shard_idx], "r")
            self._file_cache[shard_idx] = new_file
            logger.debug(f"Opened shard file {shard_idx}")
            
            return new_file
    
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
                shard_idx = np.searchsorted(self.cumulative_sizes[1:], actual_idx, side='right')
                local_idx = actual_idx - self.cumulative_sizes[shard_idx]
            else:
                shard_idx = 0
                local_idx = actual_idx
            
            if shard_idx not in shard_groups:
                shard_groups[shard_idx] = []
            shard_groups[shard_idx].append((i, local_idx))
        
        # Read from each shard in the chunk
        result_data = [None] * (end_idx - start_idx)
        
        for shard_idx, local_indices in shard_groups.items():
            shard_file = self._get_shard_file(shard_idx)
            
            for result_idx, local_idx in local_indices:
                if local_idx >= self.shard_sizes[shard_idx]:
                    continue
                    
                grp = shard_file[str(local_idx)]
                
                input_ids = torch.tensor(grp["input_ids"][:], dtype=torch.long)
                attention_mask = torch.tensor(grp["attention_mask"][:], dtype=torch.long)
                teacher_embeddings = torch.tensor(grp["teacher_embeddings"][:], dtype=torch.float)
                
                result_data[result_idx] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "teacher_embeddings": teacher_embeddings
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
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")
        
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
                        if (next_chunk_start < self.total_size and 
                            next_chunk_start not in self._prefetch_cache):
                            self._prefetch_executor.submit(self._prefetch_chunk, next_chunk_start)
                        
                        return chunk_data[chunk_offset]
        
        # Fallback to direct read
        actual_idx = self.indices[idx]
        
        if self.sharded:
            # Find which shard contains this index
            shard_idx = np.searchsorted(self.cumulative_sizes[1:], actual_idx, side='right')
            local_idx = actual_idx - self.cumulative_sizes[shard_idx]
            shard_file = self._get_shard_file(shard_idx)
            grp = shard_file[str(local_idx)]
        else:
            shard_file = self._get_shard_file(0)
            grp = shard_file[str(actual_idx)]
        
        input_ids = torch.tensor(grp["input_ids"][:], dtype=torch.long)
        attention_mask = torch.tensor(grp["attention_mask"][:], dtype=torch.long)
        teacher_embeddings = torch.tensor(grp["teacher_embeddings"][:], dtype=torch.float)
        
        # Trigger prefetching for future accesses
        if self.use_threading:
            next_chunk_start = chunk_start + self.chunk_size
            if (next_chunk_start < self.total_size and 
                next_chunk_start not in self._prefetch_cache):
                self._prefetch_executor.submit(self._prefetch_chunk, next_chunk_start)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_embeddings": teacher_embeddings
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get performance statistics"""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "prefetch_hits": self._prefetch_hits,
            "open_files": len(self._file_cache),
            "prefetch_cache_size": len(self._prefetch_cache)
        }
    
    def __del__(self):
        """Clean up file handles and threads"""
        # Close all cached files
        with self._cache_lock:
            for shard_file in self._file_cache.values():
                try:
                    shard_file.close()
                except:
                    pass
            self._file_cache.clear()
        
        # Shutdown thread pool
        if self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=True)
