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
from ..utils import get_device, logger

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
                "input_ids": tokenized_seq["input_ids"].squeeze(0).to(self.device),
                "attention_mask": tokenized_seq["attention_mask"].squeeze(0).to(self.device)
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
        
        logger.info(f"Saving file {file_path}")
        logger.info(f"Input IDs shape: {input_ids_array.shape}")
        logger.info(f"Teacher embeddings shape: {teacher_embeddings_array.shape}")
        
        with h5py.File(file_path, "w") as f:
            f.create_dataset("input_ids", data=input_ids_array.astype(np.int8))
            f.create_dataset("teacher_embeddings", data=teacher_embeddings_array.astype(np.float16))
        
        logger.info(f"Saved {input_ids_array.shape[0]} sequences to {file_path}")

class ProtXDataLoader(IterableDataset):
    def __init__(
        self, 
        h5_path: Union[str, Path, List[Path]],
        device: str,
        seed: Optional[int] = None
    ):
        self.device = device
        self.seed = seed
        
        # single file or multiple files
        if isinstance(h5_path, (str, Path)):
            self.h5_paths = [Path(h5_path)]
        else:
            self.h5_paths = [Path(p) for p in h5_path]
        
        self.current_file = None
        self.current_file_idx = -1
        
        self._calculate_dataset_size()
    
    def _calculate_dataset_size(self):
        """Calculate total dataset size by opening each file temporarily"""
        self.file_sizes = []
        for path in self.h5_paths:
            with h5py.File(path, "r") as f:
                self.file_sizes.append(f["input_ids"].shape[0])
        self.size = sum(self.file_sizes)
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        # If seed is provided, shuffle the file order
        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
            indices = list(range(len(self.h5_paths)))
            rng.shuffle(indices)
            self.h5_paths = [self.h5_paths[i] for i in indices]
            self.file_sizes = [self.file_sizes[i] for i in indices]
        
        self.current_file_idx = -1
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
        
        file_available = self._open_next_file()
        
        while file_available:
            input_ids = torch.tensor(
                self.current_file["input_ids"][:], 
                dtype=torch.long
            )
            teacher_embeddings = torch.tensor(
                self.current_file["teacher_embeddings"][:], 
                dtype=torch.float
            )
            
            attention_masks = (input_ids != 0).long()
            
            for i in range(self.current_file_size):
                yield {
                    "input_ids": input_ids[i].to(self.device),
                    "attention_mask": attention_masks[i].to(self.device),
                    "teacher_embeddings": teacher_embeddings[i].to(self.device)
                }
            
            del input_ids, teacher_embeddings, attention_masks
            gc.collect()
            
            file_available = self._open_next_file()
    
    def _open_next_file(self):
        """Open the next file in the sequence"""
        # Close the current file if it exists
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
        
        # Move to the next file
        self.current_file_idx += 1
        
        # Check if we've gone through all files
        if self.current_file_idx >= len(self.h5_paths):
            return False
        
        # Open the next file
        file_path = self.h5_paths[self.current_file_idx]
        self.current_file = h5py.File(file_path, "r")
        self.current_file_size = self.file_sizes[self.current_file_idx]
        
        return True
    
    def close(self):
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
    
    def __del__(self):
        self.close()
