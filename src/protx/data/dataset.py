import re
import h5py
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from typing import Union
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, IterableDataset

from ..config import DataConfig
from ..models.teacher import ProtT5
from ..utils.common import get_device
from ..utils.logger import logger

class ProtXDataGen(IterableDataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        teacher_tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 512,
        device: str = get_device()
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
                "input_ids": tokenized_seq["input_ids"].squeeze(0).to(self.device),
                "attention_mask": tokenized_seq["attention_mask"].squeeze(0).to(self.device)
            }

class ProtXDataProcessor(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        teacher_model: ProtT5 = ProtT5(),
        max_seq_len: int = DataConfig().max_seq_len,
        batch_size: int = 32,
        device: str = get_device()
    ):
        self.data_path = Path(data_path)
        self.teacher = teacher_model
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
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
    
    def process_dataset(self, save_path: Path):
        self._load()
        teacher_tokenizer = self.teacher.tokenizer
        teacher_model = self.teacher.encoder_model

        batch = []
        all_input_ids = []
        all_attention_masks = []
        all_teacher_embeddings = []
        
        # Calculate total number of sequences for progress bar
        total_sequences = sum(1 for _ in SeqIO.parse(self.data_path, "fasta"))
        
        # Reset data generator after counting
        self._loaded = False
        self._load()
        
        logger.info(f"Processing {total_sequences} sequences and generating embeddings...")
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
                    all_attention_masks.append(attention_mask.cpu().numpy())
                    all_teacher_embeddings.append(teacher_embeddings.cpu().numpy())
                    
                    pbar.update(len(batch))
                    batch = []
            
            # Check if there are any remaining sequences
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
                all_attention_masks.append(attention_mask.cpu().numpy())
                all_teacher_embeddings.append(teacher_embeddings.cpu().numpy())
                
                pbar.update(len(batch))
        
        input_ids_array = np.concatenate(all_input_ids, axis=0)
        attention_mask_array = np.concatenate(all_attention_masks, axis=0)
        teacher_embeddings_array = np.concatenate(all_teacher_embeddings, axis=0)

        logger.info(f"Input IDs shape: {input_ids_array.shape}")
        logger.info(f"Attention mask shape: {attention_mask_array.shape}")
        logger.info(f"Teacher embeddings shape: {teacher_embeddings_array.shape}")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(save_path, "w") as f:
            f.create_dataset("input_ids", data=input_ids_array)
            f.create_dataset("attention_mask", data=attention_mask_array)
            f.create_dataset("teacher_embeddings", data=teacher_embeddings_array)
        
        logger.info(f"Dataset saved successfully with {input_ids_array.shape[0]} sequences")
        return save_path


class ProtXDataLoader(Dataset):
    """Dataset class for loading data saved by DataProcessor"""
    def __init__(
        self, 
        h5_path: Union[str, Path],
        device: str = get_device()
    ):
        self.h5_file = h5py.File(Path(h5_path), "r")
        self.device = device
        
        self.size = self.h5_file["input_ids"].shape[0]
        logger.info(f"Dataset contains {self.size} sequences")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.h5_file["input_ids"][idx], dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor(self.h5_file["attention_mask"][idx], dtype=torch.long).to(self.device),
            "teacher_embeddings": torch.tensor(self.h5_file["teacher_embeddings"][idx], dtype=torch.float).to(self.device)
        }
    
    def close(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
    
    def __del__(self):
        self.close()
