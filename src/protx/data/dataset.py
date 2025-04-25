import re
import h5py
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from typing import Union
from torch.utils.data import Dataset

from ..models.teacher import TeacherModel
from ..models.student import ProtX
from ..utils.common import get_device
from ..utils.logger import logger

class DataProcessor(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        student_model: ProtX = ProtX(),
        teacher_model: TeacherModel = TeacherModel(),
        max_seq_len: int = 512,
        batch_size: int = 32,
        device: str = get_device()
    ):
        self.data_path = Path(data_path)
        self.student = student_model
        self.teacher = teacher_model
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        logger.info(f"Teacher and student models loaded on {self.device}")

        self.data_gen = (
            (record.id, str(record.seq)) 
            for record in SeqIO.parse(self.data_path, "fasta")
        )

        logger.info(f"{self.data_path} loaded successfully.")

    def __len__(self):
        return sum(1 for _ in self.data_gen)
    
    def process_dataset(self, save_path: Path):
        teacher_tokenizer = self.teacher.tokenizer
        teacher_model = self.teacher.encoder_model

        batch = []
        all_input_ids = []
        all_attention_masks = []
        all_teacher_embeddings = []
        
        logger.info(f"Processing sequences and generating embeddings...")
        with tqdm(self.data_gen) as pbar:
            for _, sequence in pbar:
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
        
        input_ids_array = np.concatenate(all_input_ids, axis=0)
        attention_mask_array = np.concatenate(all_attention_masks, axis=0)
        teacher_embeddings_array = np.concatenate(all_teacher_embeddings, axis=0)

        logger.info(f"Input IDs shape: {input_ids_array.shape}")
        logger.info(f"Attention mask shape: {attention_mask_array.shape}")
        logger.info(f"Teacher embeddings shape: {teacher_embeddings_array.shape}")
        
        with h5py.File(save_path, "w") as f:
            f.create_dataset("input_ids", data=input_ids_array)
            f.create_dataset("attention_mask", data=attention_mask_array)
            f.create_dataset("teacher_embeddings", data=teacher_embeddings_array)
        
        logger.info(f"Dataset saved successfully with {input_ids_array.shape[0]} sequences")
        return save_path
