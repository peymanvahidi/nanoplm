import random
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from io import StringIO

from ..utils import create_dirs, logger

class FilterSplitor():
    """Class for preprocessing the UniRef50 dataset."""

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        min_seq_len: int,
        max_seq_len: int,
        max_seqs_num: int,
        val_ratio: float,
        info_file: Path,
        shuffle: bool = True
    ):
        self.input_file = input_file
        self.output_dir = output_dir
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.max_seqs_num = max_seqs_num
        self.val_ratio = val_ratio
        self.info_file = info_file
        self.shuffle = shuffle

    def filter(
        self,
        output_file: Path
    ):
        create_dirs(output_file)
        
        logger.info(
            f"Processing UniRef50 sequences with filters: "
            f"min_length={self.min_seq_len}, max_length={self.max_seq_len}, "
            f"max_seqs_number={self.max_seqs_num}"
        )
        
        index = self._build_index(self.input_file)
        total_seqs = len(index)
        
        if self.shuffle:
            random.shuffle(index)
        
        seq_count = 0
        passed_count = 0
        
        logger.info(f"Randomly selecting and filtering sequences...")
        
        with tqdm(total=total_seqs if self.max_seqs_num is not None else total_seqs, desc="Processing sequences") as pbar:
            with open(output_file, 'w') as output_handle:
                with open(self.input_file, 'rb') as input_handle:
                    for i, (start, end) in enumerate(index):
                        input_handle.seek(start)
                        seq_data = input_handle.read(end - start).decode('utf-8')
                        
                        record = next(SeqIO.parse(StringIO(seq_data), 'fasta'))
                        seq_count += 1
                        
                        seq_len = len(record.seq)
                        if self.min_seq_len <= seq_len <= self.max_seq_len:
                            SeqIO.write([record], output_handle, 'fasta')
                            passed_count += 1
                            pbar.update(1)
                        
                        if self.max_seqs_num is not None and passed_count >= self.max_seqs_num:
                            break
        
        logger.info(f"Processed {seq_count} sequences out of {total_seqs}: {passed_count} sequences retrieved with length in [{self.min_seq_len}, {self.max_seq_len}].")
        logger.info(f"Output file: {output_file}")

        self.processed_seqs_num = seq_count
        self.num_filtered_seqs = passed_count
        self.filtered_seqs_path = output_file

    def split(
        self,
        train_file: Path,
        val_file: Path
    ):
        """
        Split the filtered seqs to train and val.
        """
        logger.info(f"Creating splits with val ratio {self.val_ratio}")
        
        sequences = list(SeqIO.parse(self.filtered_seqs_path, 'fasta'))
        num_filtered_seqs = len(sequences)
        
        val_size = int(num_filtered_seqs * self.val_ratio)
        train_size = num_filtered_seqs - val_size
        
        logger.info(f"Sequences: {num_filtered_seqs}, Train: {train_size}, Val: {val_size}")
        
        with tqdm(total=num_filtered_seqs, desc="Splitting data") as pbar:
            SeqIO.write(sequences[:train_size], train_file, 'fasta')
            SeqIO.write(sequences[train_size:], val_file, 'fasta')
            
            pbar.update(num_filtered_seqs)
        
        logger.info(f"Created files in {self.output_dir}")

        self.train_count = train_size
        self.val_count = val_size
        self.num_filtered_seqs = num_filtered_seqs

        self.train_file = train_file
        self.val_file = val_file
        
        self._create_info_file()
    
    def _create_info_file(self):
        """Create a file with information about the processed dataset."""
        with open(self.info_file, 'w') as f:
            f.write(f"Dataset: UniRef50\n")
            f.write(f"Source: {self.input_file}\n")
            f.write(f"Processing parameters:\n")
            f.write(f"  - Min sequence length: {self.min_seq_len}\n")
            f.write(f"  - Max sequence length: {self.max_seq_len}\n")
            f.write(f"  - Max sequences: {self.max_seqs_num if self.max_seqs_num else 'No limit'}\n")
            f.write(f"  - Validation ratio: {self.val_ratio}\n")
            f.write(f"Total processed sequences: {self.processed_seqs_num}\n")
            f.write(f"Training sequences: {self.train_count}\n")
            f.write(f"Validation sequences: {self.val_count}\n")
        
        logger.info(f"Created dataset info file at {self.info_file}")

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
