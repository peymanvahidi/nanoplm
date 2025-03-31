from Bio import SeqIO
from tqdm import tqdm
from .base import BaseProcessor, logger

class Preprocessor(BaseProcessor):
    """Class for preprocessing the UniRef50 dataset."""

    def preprocess(self):
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

if __name__ == "__main__":
    # For direct execution
    preprocessor = Preprocessor()
    preprocessor.preprocess() 