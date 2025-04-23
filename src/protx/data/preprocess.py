import re
import math
import h5py
import json
import random
from Bio import SeqIO
from tqdm import tqdm
from transformers import T5Tokenizer

from .base import BaseProcessor, logger

class Preprocessor(BaseProcessor):
    """Class for preprocessing the UniRef50 dataset."""

    def filter_sequences(self):
        """Process sequences according to length filters."""
        self.create_dirs(self.processed_data_dir)
        
        logger.info(
            f"Processing UniRef50 sequences with filters: "
            f"min_length={self.min_seq_len}, max_length={self.max_seq_len}, "
            f"max_seqs_number={self.max_seqs_num}"
        )
        
        # First, count sequences for progress bar
        total_seqs = sum(1 for _ in SeqIO.parse(self.uniref50_fasta, 'fasta'))
        
        seq_count = 0
        filtered_count = 0

        n = min(total_seqs, self.max_seqs_num) if self.max_seqs_num else total_seqs
        
        with tqdm(total=n, desc="Processing sequences") as pbar:
            with open(self.filtered_seqs, 'w') as output_handle:
                for record in SeqIO.parse(self.uniref50_fasta, 'fasta'):
                    seq_len = len(record.seq)

                    # Check sequence length
                    if self.min_seq_len <= seq_len <= self.max_seq_len:
                        SeqIO.write([record], output_handle, 'fasta')
                        seq_count += 1
                        pbar.update(1)
                        if seq_count >= self.max_seqs_num:
                            break
                    else:
                        filtered_count += 1
                        # Only update if we're limiting sequences
                        if self.max_seqs_num:
                            pbar.total = min(total_seqs, self.max_seqs_num + filtered_count)
        
        logger.info(f"Processed {seq_count} sequences (filtered out {filtered_count}) to {self.filtered_seqs}")
        self.total_seqs = seq_count


    def split(self, shuffle=True):
        """
        Split the filtered seqs to train and val.
        """
        logger.info(f"Creating splits with val ratio {self.val_ratio} (shuffle={shuffle})")

        index = self._build_index(self.filtered_seqs)
        total_seqs = len(index)
        
        if shuffle:
            random.shuffle(index)
        
        val_size = int(total_seqs * self.val_ratio)
        train_size = total_seqs - val_size
        
        logger.info(f"Sequences: {total_seqs}, Train: {train_size}, Val: {val_size}")
        
        with open(self.filtered_seqs, 'rb') as src:
            with open(self.train_file, 'wb') as train, open(self.val_file, 'wb') as val:
                with tqdm(total=total_seqs, desc="Splitting data") as pbar:
                    for i, (start, end) in enumerate(index):
                        src.seek(start)
                        data = src.read(end - start)
                        (train if i < train_size else val).write(data)
                        pbar.update(1)
        
        logger.info(f"Created files in {self.processed_data_dir}")
        self.train_count = train_size
        self.val_count = val_size
        self.total_seqs = total_seqs
        
        self._create_info_file()
    
    def tokenize_sequences(self):
        """
        Tokenize train and val seqs using ProtT5 tokenizer in batches.
        Save tokenized data in HDF5 format.
        """
        batch_size = self.batch_size
        
        # Define paths for tokenized data directories
        tokenized_dir = self.processed_data_dir / "tokenized"
        self.create_dirs(tokenized_dir)

        # Load ProtT5 tokenizer
        logger.info("Loading ProtT5 tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)

        # Define data types for HDF5 string storage
        string_dt = h5py.string_dtype(encoding='utf-8')

        # Process train and val files
        for input_file, name in [
            (self.train_file, tokenized_dir / "train"),
            (self.val_file, tokenized_dir / "val")
        ]:
            if not input_file.exists():
                logger.warning(f"{name} file not found at {input_file}. Skipping tokenization.")
                continue
                
            # Define output HDF5 file path inside the specific directory
            output_h5_file = f"{name}.h5"

            # Count sequences for progress tracking
            seq_count = sum(1 for _ in SeqIO.parse(input_file, "fasta"))
            num_batches = math.ceil(seq_count / batch_size)
            
            logger.info(f"Tokenizing {seq_count} {name} sequences into {output_h5_file} ({num_batches} batches, batch_size={batch_size})")
            
            # Open HDF5 file for writing
            with h5py.File(output_h5_file, 'w') as hf:
                # Store metadata as attributes
                hf.attrs['num_seqs'] = seq_count
                hf.attrs['batch_size'] = batch_size
                hf.attrs['num_batches'] = num_batches
                hf.attrs['tokenizer'] = "Rostlab/prot_t5_xl_uniref50"

                # Process in batches
                batch_idx = 0
                sequences_processed = 0
                
                with tqdm(total=seq_count, desc=f"Tokenizing {name} sequences to HDF5") as pbar:
                    records = []
                    seq_ids = []
                    batch_seqs = []
                    
                    for record in SeqIO.parse(input_file, "fasta"):
                        seq_ids.append(record.id)
                        records.append(record)
                        
                        sequence = str(record.seq)
                        processed_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                        batch_seqs.append(processed_seq)
                        
                        sequences_processed += 1
                        pbar.update(1)
                        
                        if len(batch_seqs) == batch_size or sequences_processed == seq_count:
                            if not batch_seqs: continue

                            batch_encoding = tokenizer.batch_encode_plus(
                                batch_seqs,
                                add_special_tokens=True,
                                padding="longest",
                                return_tensors="pt"
                            )
                            
                            batch_group = hf.create_group(f"batch_{batch_idx:05d}")
                            batch_group.create_dataset('input_ids', data=batch_encoding['input_ids'].numpy())
                            batch_group.create_dataset('attention_mask', data=batch_encoding['attention_mask'].numpy())
                            batch_group.create_dataset('seq_ids', data=seq_ids, dtype=string_dt)
                            batch_group.create_dataset('sequences', data=batch_seqs, dtype=string_dt)

                            records = []
                            seq_ids = []
                            batch_seqs = []
                            batch_idx += 1
            
            with h5py.File(output_h5_file, 'a') as hf:
                 hf.attrs['num_batches'] = batch_idx 

            logger.info(f"Tokenized {name} data saved to {output_h5_file}")

        file_structure = {
            "description": "#### This file is just for DEMONSTRATION of the .h5 file structure ####",
            "attributes": {
                "num_seqs": "integer - Total number of sequences in the dataset",
                "batch_size": "integer - Size of each batch",
                "num_batches": "integer - Total number of batches",
                "tokenizer": "string - Name of the tokenizer used"
            },
            "groups": {
                "batch_00000": {
                    "datasets": {
                        "input_ids": "numpy array - Shape: [batch_size, longest_sequence_length] - Tokenized input ids",
                        "attention_mask": "numpy array - Shape: [batch_size, longest_sequence_length] - Attention mask for padded sequences",
                        "seq_ids": "string array - Sequence identifiers",
                        "sequences": "string array - Processed protein sequences with spaces between amino acids"
                    }
                },
                "batch_00001": "... and so on for each batch"
            }
        }
        
        with open(tokenized_dir / "file_structure.json", 'w') as f:
            json.dump(file_structure, f, indent=2)
        
        logger.info(f"File structure documentation saved to {tokenized_dir / 'file_structure.json'}")

    def _create_info_file(self):
        """Create a file with information about the processed dataset."""
        with open(self.info_file, 'w') as f:
            f.write(f"Dataset: UniRef50\n")
            f.write(f"Source: {self.uniref50_url}\n")
            f.write(f"Processing parameters:\n")
            f.write(f"  - Min sequence length: {self.min_seq_len}\n")
            f.write(f"  - Max sequence length: {self.max_seq_len}\n")
            f.write(f"  - Max sequences: {self.max_seqs_num if self.max_seqs_num else 'No limit'}\n")
            f.write(f"  - Validation ratio: {self.val_ratio}\n")
            f.write(f"Total processed sequences: {self.total_seqs}\n")
            f.write(f"Training sequences: {self.train_count}\n")
            f.write(f"Validation sequences: {self.val_count}\n")
        
        logger.info(f"Created dataset info file at {self.info_file}")

    @staticmethod
    def _count_records(file_path):
        with open(file_path, 'rb') as f:
            return sum(1 for line in f if line.startswith(b'>'))
    
    @staticmethod
    def _build_index(file_path):
        index = []
        with tqdm(desc="Building sequence index", unit="record") as pbar:
            with open(file_path, 'rb') as f:
                start_pos = 0
                while True:
                    line = f.readline()
                    if not line: break
                    if line.startswith(b'>'):
                        if index:
                            index[-1] = (index[-1][0], start_pos)
                        index.append((start_pos, None))
                        pbar.update(1)
                    start_pos = f.tell()
            if index:
                index[-1] = (index[-1][0], start_pos)
        return index
