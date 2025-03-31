from Bio import SeqIO
from tqdm import tqdm
from .base import BaseProcessor, logger
from transformers import T5Tokenizer
import math
import torch
import re
import random

class Preprocessor(BaseProcessor):
    """Class for preprocessing the UniRef50 dataset."""

    def filter_sequences(self):
        """Process sequences according to length filters."""
        self.create_dirs(self.processed_data_dir)
        
        logger.info(f"Processing UniRef50 sequences with filters: "
                   f"min_length={self.min_sequence_length}, max_length={self.max_sequence_length}, "
                   f"max_seqs_number={self.max_seqs_number}")
        
        # First, count sequences for progress bar
        total_seqs = sum(1 for _ in SeqIO.parse(self.uniref50_fasta, 'fasta'))
        
        sequence_count = 0
        filtered_count = 0
        
        with tqdm(total=min(total_seqs, self.max_seqs_number or float('inf')), 
                 desc="Processing sequences") as pbar:
            with open(self.processed_sequences, 'w') as output_handle:
                for record in SeqIO.parse(self.uniref50_fasta, 'fasta'):
                    seq_len = len(record.seq)
                    if self.min_sequence_length <= seq_len <= self.max_sequence_length:
                        SeqIO.write([record], output_handle, 'fasta')
                        sequence_count += 1
                        pbar.update(1)
                        if self.max_seqs_number and sequence_count >= self.max_seqs_number:
                            break
                    else:
                        filtered_count += 1
                        if self.max_seqs_number:  # Only update if we're limiting sequences
                            pbar.total = min(total_seqs, self.max_seqs_number + filtered_count)
        
        logger.info(f"Processed {sequence_count} sequences (filtered out {filtered_count}) to {self.processed_sequences}")
        self.total_sequences = sequence_count
    
    def split(self, shuffle=True):
        """Split processed data into training and validation sets."""
        
        logger.info(f"Creating splits with val ratio {self.val_ratio} (shuffle={shuffle})")
        
        # Fast record counting using line scanning
        def count_records(file_path):
            with open(file_path, 'rb') as f:
                return sum(1 for line in f if line.startswith(b'>'))
        
        # Build memory-efficient index of record positions
        def build_index(file_path):
            index = []
            with tqdm(desc="Building sequence index", unit="record") as pbar:
                with open(file_path, 'rb') as f:
                    start_pos = 0
                    while True:
                        line = f.readline()
                        if not line: break
                        if line.startswith(b'>'):
                            if index:  # Update previous record's end position
                                index[-1] = (index[-1][0], start_pos)
                            index.append((start_pos, None))  # (start, end)
                            pbar.update(1)
                        start_pos = f.tell()
                if index:  # Set final record's end position
                    index[-1] = (index[-1][0], start_pos)
            return index

        # Core splitting logic
        if shuffle:
            # Shuffled split using memory-mapped index
            index = build_index(self.processed_sequences)
            total_sequences = len(index)
            logger.info(f"Shuffling {total_sequences} sequences...")
            random.shuffle(index)
        else:
            # Non-shuffled split using fast line counting
            total_sequences = count_records(self.processed_sequences)
        
        val_size = int(total_sequences * self.val_ratio)
        train_size = total_sequences - val_size
        
        logger.info(f"Sequences: {total_sequences}, Train: {train_size}, Val: {val_size}")
        
        # Ultra-fast file splitting
        with open(self.processed_sequences, 'rb') as src:
            if shuffle:
                # Random access write using index
                with open(self.train_file, 'wb') as train, open(self.val_file, 'wb') as val:
                    with tqdm(total=total_sequences, desc="Splitting data") as pbar:
                        for i, (start, end) in enumerate(index):
                            src.seek(start)
                            data = src.read(end - start)
                            (train if i < train_size else val).write(data)
                            pbar.update(1)
            else:
                # Streaming split using single pass
                with open(self.train_file, 'wb') as train, open(self.val_file, 'wb') as val:
                    current_count = 0
                    buffer = []
                    target = train
                    
                    with tqdm(total=total_sequences, desc="Splitting data") as pbar:
                        for line in src:
                            if line.startswith(b'>'):
                                current_count += 1
                                pbar.update(1)
                                if current_count > train_size:
                                    # Flush buffer and switch target
                                    target.write(b''.join(buffer))
                                    buffer = []
                                    target = val
                            buffer.append(line)
                        
                        # Write remaining buffer
                        if buffer:
                            target.write(b''.join(buffer))
        
        logger.info(f"Created files in {self.processed_data_dir}")
        self.train_count = train_size
        self.val_count = val_size
        self.total_sequences = total_sequences
        
        # Create the dataset info file after splitting
        self.create_info_file()
    
    def create_info_file(self):
        """Create a file with information about the processed dataset."""
        with open(self.info_file, 'w') as f:
            f.write(f"Dataset: UniRef50\n")
            f.write(f"Source: {self.uniref50_url}\n")
            f.write(f"Processing parameters:\n")
            f.write(f"  - Min sequence length: {self.min_sequence_length}\n")
            f.write(f"  - Max sequence length: {self.max_sequence_length}\n")
            f.write(f"  - Max sequences: {self.max_seqs_number if self.max_seqs_number else 'No limit'}\n")
            f.write(f"  - Validation ratio: {self.val_ratio}\n")
            f.write(f"Total processed sequences: {self.total_sequences}\n")
            f.write(f"Training sequences: {self.train_count}\n")
            f.write(f"Validation sequences: {self.val_count}\n")
        
        logger.info(f"Created dataset info file at {self.info_file}")
    
    def tokenize_sequences(self):
        """
        Tokenize train and validation sequences using ProtT5 tokenizer in batches.
        Process data according to batch_size parameter.
        Save tokenized data in PyTorch-compatible format.
        """
        # Load batch size from parameters
        batch_size = self.batch_size
        
        # Define paths for tokenized data directories
        train_tokenized_dir = self.processed_data_dir / "train_tokenized"
        val_tokenized_dir = self.processed_data_dir / "val_tokenized"
        
        # Create directories for tokenized data
        for directory in [train_tokenized_dir, val_tokenized_dir]:
            self.create_dirs(directory)
        
        # Load ProtT5 tokenizer
        logger.info("Loading ProtT5 tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        
        # Process train and val files
        for input_file, output_dir, name in [
            (self.train_file, train_tokenized_dir, "train"),
            (self.val_file, val_tokenized_dir, "validation")
        ]:
            if not input_file.exists():
                logger.warning(f"{name} file not found at {input_file}. Skipping tokenization.")
                continue
                
            # Count sequences for progress tracking
            seq_count = sum(1 for _ in SeqIO.parse(input_file, "fasta"))
            num_batches = math.ceil(seq_count / batch_size)
            
            logger.info(f"Tokenizing {seq_count} {name} sequences in {num_batches} batches (batch_size={batch_size})")
            
            # Process in batches
            batch_idx = 0
            sequences_processed = 0
            
            with tqdm(total=seq_count, desc=f"Tokenizing {name} sequences") as pbar:
                records = []
                sequence_ids = []
                batch_sequences = []
                
                for record in SeqIO.parse(input_file, "fasta"):
                    # Store the record ID for reference
                    sequence_ids.append(record.id)
                    records.append(record)
                    
                    # Process sequence: replace rare/ambiguous amino acids with X and add spaces
                    sequence = str(record.seq)
                    processed_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                    batch_sequences.append(processed_seq)
                    
                    sequences_processed += 1
                    pbar.update(1)
                    
                    # Process batch when it reaches the specified size or at the end
                    if len(batch_sequences) == batch_size or sequences_processed == seq_count:
                        # Tokenize the batch
                        batch_encoding = tokenizer.batch_encode_plus(
                            batch_sequences,
                            add_special_tokens=True,
                            padding="longest",
                            return_tensors="pt"
                        )
                        
                        # Create batch data dictionary
                        batch_data = {
                            'input_ids': batch_encoding['input_ids'],
                            'attention_mask': batch_encoding['attention_mask'],
                            'sequence_ids': sequence_ids,
                            'sequences': batch_sequences
                        }
                        
                        # Save batch data as PyTorch file
                        batch_file = output_dir / f"batch_{batch_idx:05d}.pt"
                        torch.save(batch_data, batch_file)
                        
                        # Clear batch data
                        records = []
                        sequence_ids = []
                        batch_sequences = []
                        batch_idx += 1
            
            # Create metadata file
            metadata = {
                'num_sequences': seq_count,
                'num_batches': batch_idx,
                'batch_size': batch_size,
                'tokenizer': "Rostlab/prot_t5_xl_uniref50"
            }
            
            metadata_file = output_dir / "metadata.pt"
            torch.save(metadata, metadata_file)
            
            logger.info(f"Data saved to {output_dir}")
