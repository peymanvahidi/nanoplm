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
from ..utils import logger

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
        seqs_num_per_file: int,
        batch_size: int,
        device: str
    ):
        self.data_path = Path(data_path)
        self.teacher = teacher_model
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.seqs_num_per_file = seqs_num_per_file
        self._loaded = False

    def _load(self):
        if not self._loaded:
            self.data_gen = (
                (record.id, str(record.seq)) 
                for record in SeqIO.parse(self.data_path, "fasta")
            )
            self._loaded = True
            logger.info(f"{self.data_path} initialized successfully.")

    def __len__(self):
        self._load()
        return sum(1 for _ in self.data_gen)
    
    def process_dataset(self, save_path: Path) -> List[Path]:
        self._load()
        teacher_tokenizer = self.teacher.tokenizer
        teacher_model = self.teacher.encoder_model
        
        total_sequences = sum(1 for _ in SeqIO.parse(self.data_path, "fasta"))
        
        self._loaded = False
        self._load()
        
        logger.info(f"Processing {total_sequences} sequences and generating embeddings...")
        
        save_path = Path(save_path)
        base_filename = save_path.stem
        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        batch = []
        all_input_ids = []
        all_teacher_embeddings = []
        file_count = 1
        sequence_count = 0
        saved_files = []
        
        with tqdm(total=total_sequences, desc="Generating embeddings", unit="seq") as pbar:
            for _, sequence in self.data_gen:
                teacher_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                
                batch.append(teacher_seq)

                if len(batch) == self.batch_size:
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
                    
                    all_input_ids.append(input_ids.cpu().numpy())
                    all_teacher_embeddings.append(teacher_embeddings.cpu().numpy())
                    
                    sequence_count += len(batch)
                    pbar.update(len(batch))
                    batch = []
                    
                    # Check if we need to save the file
                    if sequence_count >= self.seqs_num_per_file:
                        current_file = save_dir / f"{base_filename}{file_count}.h5"
                        self._save_file(current_file, all_input_ids, all_teacher_embeddings)
                        saved_files.append(current_file)
                        
                        # Clear memory
                        all_input_ids = []
                        all_teacher_embeddings = []
                        gc.collect()
                        
                        sequence_count = 0
                        file_count += 1
            
            # Check if there are any remaining sequences in the batch
            if batch:
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
                
                all_input_ids.append(input_ids.cpu().numpy())
                all_teacher_embeddings.append(teacher_embeddings.cpu().numpy())
                
                sequence_count += len(batch)
                pbar.update(len(batch))
            
            # Save any remaining data
            if all_input_ids:
                current_file = save_dir / f"{base_filename}{file_count}.h5"
                self._save_file(current_file, all_input_ids, all_teacher_embeddings)
                saved_files.append(current_file)
                
                # Clear memory
                all_input_ids = []
                all_teacher_embeddings = []
                gc.collect()
        
        logger.info(f"Dataset saved successfully in {len(saved_files)} files with {total_sequences} total sequences")
        return saved_files
    
    def _save_file(
        self,
        file_path: Path,
        input_ids_list: List[np.ndarray],
        teacher_embeddings_list: List[np.ndarray]
    ):
        input_ids_array = np.concatenate(input_ids_list, axis=0)
        teacher_embeddings_array = np.concatenate(teacher_embeddings_list, axis=0)
        
        logger.debug(f"Saving file {file_path}")
        logger.debug(f"Input IDs shape: {input_ids_array.shape}")
        logger.debug(f"Teacher embeddings shape: {teacher_embeddings_array.shape}")
        
        with h5py.File(file_path, "w") as f:
            f.create_dataset("input_ids", data=input_ids_array.astype(np.int8))
            f.create_dataset("teacher_embeddings", data=teacher_embeddings_array.astype(np.float16))
        
        logger.debug(f"Saved {input_ids_array.shape[0]} sequences to {file_path}")

class ProtXDataLoader(Dataset):
    def __init__(
        self, 
        h5_path: Union[str, Path, List[Path]],
        device: str,
        seed: Optional[int] = None
    ):
        self.device = device
        self.seed = seed
        
        if isinstance(h5_path, (str, Path)):
            self.h5_paths = [Path(h5_path)]
        else:
            self.h5_paths = [Path(p) for p in h5_path]
        
        self._calculate_dataset_info()
        
        if self.seed is not None:
            self._shuffle_files()
    
    def _calculate_dataset_info(self):
        """Calculate total dataset size and create mapping from global index to file+local index"""
        self.file_sizes = []
        self.cumulative_sizes = []
        cumulative = 0
        
        for path in self.h5_paths:
            with h5py.File(path, "r") as f:
                file_size = f["input_ids"].shape[0]
                self.file_sizes.append(file_size)
                cumulative += file_size
                self.cumulative_sizes.append(cumulative)
        
        self.total_size = cumulative
    
    def _shuffle_files(self):
        """Shuffle the file order based on seed"""
        rng = np.random.RandomState(self.seed)
        indices = list(range(len(self.h5_paths)))
        rng.shuffle(indices)
        
        self.h5_paths = [self.h5_paths[i] for i in indices]
        self.file_sizes = [self.file_sizes[i] for i in indices]
        
        cumulative = 0
        self.cumulative_sizes = []
        for size in self.file_sizes:
            cumulative += size
            self.cumulative_sizes.append(cumulative)
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx >= self.total_size or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")
        
        file_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                file_idx = i
                break
        
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]
        
        file_path = self.h5_paths[file_idx]
        with h5py.File(file_path, "r") as f:
            input_ids = torch.tensor(f["input_ids"][local_idx], dtype=torch.long)
            teacher_embeddings = torch.tensor(f["teacher_embeddings"][local_idx], dtype=torch.float)
            
            attention_mask = (input_ids != 0).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_embeddings": teacher_embeddings
        }
