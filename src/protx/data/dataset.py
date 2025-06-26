import gc
import re
import h5py
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Optional
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, IterableDataset

from ..models.teacher import ProtT5
from ..utils import logger, get_device

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
        skip_n: int = 0
    ):
        self.data_path = Path(data_path)
        self.teacher = teacher_model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device if device != "auto" else get_device()
        self.skip_n = skip_n
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
    
    def process_dataset(self, save_path: Path) -> Path:
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
