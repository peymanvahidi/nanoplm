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
        device: str
    ):
        self.data_path = Path(data_path)
        self.teacher = teacher_model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device if device != "auto" else get_device()
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
    
    def process_dataset(self, save_path: Path) -> Path:
        self._load()
        teacher_tokenizer = self.teacher.tokenizer
        teacher_model = self.teacher.encoder_model
        
        total_sequences = sum(1 for _ in SeqIO.parse(self.data_path, "fasta"))
        
        self._loaded = False
        self._load()
        
        logger.info(f"Processing {total_sequences} sequences and generating embeddings...")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        batch = []
        sequence_index = 0
        
        with h5py.File(save_path, "w") as h5f:
            with tqdm(total=total_sequences, desc="Generating embeddings", unit="seq") as pbar:
                for _, sequence in self.data_gen:
                    teacher_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                    batch.append(teacher_seq)

                    if len(batch) == self.batch_size:
                        self._process_and_save_batch(
                            h5f, batch, teacher_tokenizer, teacher_model, sequence_index
                        )
                        sequence_index += len(batch)
                        pbar.update(len(batch))
                        batch = []
                
                # Process any remaining sequences in the batch
                if batch:
                    self._process_and_save_batch(
                        h5f, batch, teacher_tokenizer, teacher_model, sequence_index
                    )
                    pbar.update(len(batch))
        
        logger.info(f"Dataset saved successfully to {save_path} with {total_sequences} total sequences")
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
        
        # Save each sequence in the batch as a separate group
        for i, (seq_input_ids, seq_attention_mask, seq_teacher_embeddings) in enumerate(
            zip(input_ids.cpu().numpy(), attention_mask.cpu().numpy(), teacher_embeddings.cpu().numpy())
        ):
            grp = h5f.create_group(str(start_index + i))
            grp.create_dataset("input_ids", data=seq_input_ids.astype(np.int8))
            grp.create_dataset("attention_mask", data=seq_attention_mask.astype(np.int8))
            grp.create_dataset("teacher_embeddings", data=seq_teacher_embeddings.astype(np.float16))

class ProtXDataLoader(Dataset):
    def __init__(
        self, 
        h5_path: Union[str, Path],
        device: str,
        seed: Optional[int] = None
    ):
        self.h5_path = Path(h5_path)
        self.device = device
        self.seed = seed
        
        self.h5f = h5py.File(self.h5_path, "r")
        self.total_size = len(self.h5f.keys())
        
        # Create a list of indices for potential shuffling
        self.indices = list(range(self.total_size))
        
        if self.seed is not None:
            self._shuffle_indices()
    
    def _shuffle_indices(self):
        """Shuffle the indices based on seed"""
        rng = np.random.RandomState(self.seed)
        rng.shuffle(self.indices)
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx >= self.total_size or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")
        
        # Use shuffled index if shuffling is enabled
        actual_idx = self.indices[idx]
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
        if hasattr(self, 'h5f'):
            self.h5f.close()
