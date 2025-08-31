from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from typing import Union

from myplm.utils import create_dirs, logger

class FilterSplitor():
    """Class for preprocessing the UniRef50 dataset."""

    def __init__(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        min_seq_len: int,
        max_seq_len: int,
        max_seqs_num: int,
        val_ratio: float,
        info_file: Union[str, Path],
        skip_n: int = 0
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.max_seqs_num = max_seqs_num
        self.val_ratio = val_ratio
        self.info_file = Path(info_file)
        self.skip_n = skip_n

    def filter(self):
        create_dirs(self.output_file.parent)
        
        logger.info(
            f"Processing UniRef50 sequences with filters: "
            f"min_length={self.min_seq_len}, max_length={self.max_seq_len}, "
            f"max_seqs_number={self.max_seqs_num}, skip_n={self.skip_n}"
        )
        
        seq_count = 0
        passed_count = 0
        skipped_count = 0
        
        logger.info(f"Processing sequences sequentially from {self.input_file}")
        
        with open(self.output_file, 'w') as output_handle:
            for record in tqdm(SeqIO.parse(self.input_file, 'fasta'), desc="Processing sequences"):
                if skipped_count < self.skip_n:
                    skipped_count += 1
                    continue
                
                seq_count += 1
                
                if self.max_seqs_num != -1 and passed_count >= self.max_seqs_num:
                    break
                
                seq_len = len(record.seq)
                if self.min_seq_len <= seq_len <= self.max_seq_len:
                    SeqIO.write([record], output_handle, 'fasta')
                    passed_count += 1
        
        logger.info(f"Processed {seq_count} sequences from the considered set (after skipping {self.skip_n}): {passed_count} sequences retrieved with length in [{self.min_seq_len}, {self.max_seq_len}].")
        logger.info(f"Output file: {self.output_file}")

        self.processed_seqs_num = seq_count
        self.num_filtered_seqs = passed_count
        self.filtered_seqs_path = self.output_file

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
            f.write(f"  - Max sequences: {self.max_seqs_num if self.max_seqs_num != -1 else 'No limit'}\n")
            f.write(f"  - Validation ratio: {self.val_ratio}\n")
            f.write(f"  - Skipped N sequences from input: {self.skip_n}\n")
            f.write(f"Total processed sequences (after skip): {self.processed_seqs_num}\n")
            f.write(f"Filtered sequences (written to output): {self.num_filtered_seqs}\n")
            f.write(f"Training sequences: {self.train_count}\n")
            f.write(f"Validation sequences: {self.val_count}\n")
        
        logger.info(f"Created dataset info file at {self.info_file}")

    # @staticmethod
    # def _build_index_biopython_chunks(file_path):
    #     """Hybrid approach: BioPython parsing with chunk-based reading for optimal performance."""
    #     # Commented out - no longer needed since we're not shuffling
    #     index = []
    #     chunk_size = 1024  # 1KB chunks for better performance with BioPython
        
    #     logger.info("Building sequence index using BioPython with chunk optimization...")
        
    #     with tqdm(desc="Building sequence index", unit="MB") as pbar:
    #         with open(file_path, 'rb') as f:
    #             buffer = b''
    #             current_pos = 0
    #             sequence_start = None
                
    #             while True:
    #                 chunk = f.read(chunk_size)
    #                 if not chunk:
    #                     break
                    
    #                 buffer += chunk
    #                 pbar.update(len(chunk) / (1024 * 1024))
                    
    #                 # Process buffer to find complete sequences
    #                 while True:
    #                     # Find next header line
    #                     header_pos = buffer.find(b'\n>')
    #                     if header_pos == -1:
    #                         # No complete sequence found, need more data
    #                         break
                        
    #                     # If we have a sequence start, record it
    #                     if sequence_start is not None:
    #                         seq_end_pos = current_pos + header_pos + 1
    #                         index.append((sequence_start, seq_end_pos))
                        
    #                     # Find the start of the next sequence
    #                     next_header_start = current_pos + header_pos + 1
    #                     sequence_start = next_header_start
                        
    #                     # Remove processed part from buffer
    #                     buffer = buffer[header_pos + 1:]
    #                     current_pos += header_pos + 1
                    
    #                 # Handle first sequence if buffer starts with header
    #                 if sequence_start is None and buffer.startswith(b'>'):
    #                     sequence_start = current_pos
                    
    #                 # Update position for remaining buffer
    #                 if buffer and not buffer.endswith(b'\n'):
    #                     # Keep incomplete line in buffer
    #                     last_newline = buffer.rfind(b'\n')
    #                     if last_newline != -1:
    #                         current_pos += last_newline + 1
    #                         buffer = buffer[last_newline + 1:]
    #                     else:
    #                         current_pos += len(buffer)
    #                         buffer = b''
    #                 else:
    #                     current_pos += len(buffer)
    #                     buffer = b''
                
    #             # Handle last sequence
    #             if sequence_start is not None:
    #                 index.append((sequence_start, current_pos))
        
    #     logger.info(f"Built index for {len(index)} sequences using hybrid BioPython+chunks method")
    #     return index

    # @staticmethod
    # def _build_index_cached(file_path):
    #     """Build index with caching - saves index to disk for reuse."""
    #     # Commented out - no longer needed since we're not shuffling
    #     file_path = Path(file_path)
        
    #     # Create cache file path based on source file hash
    #     cache_dir = file_path.parent / '.protx_cache'
    #     cache_dir.mkdir(exist_ok=True)
        
    #     # Generate cache file name based on file path and modification time
    #     file_stat = file_path.stat()
    #     file_hash = hashlib.md5(f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}".encode()).hexdigest()
    #     cache_file = cache_dir / f"index_{file_hash}.pkl"
        
    #     # Try to load cached index
    #     if cache_file.exists():
    #         logger.info(f"Loading cached index from {cache_file}")
    #         try:
    #             with open(cache_file, 'rb') as f:
    #                 return pickle.load(f)
    #         except Exception as e:
    #             logger.warning(f"Failed to load cached index: {e}. Rebuilding...")
        
    #     # Build new index
    #     logger.info("Building new sequence index...")
    #     index = FilterSplitor._build_index_biopython_chunks(file_path)
        
    #     # Save to cache
    #     try:
    #         with open(cache_file, 'wb') as f:
    #             pickle.dump(index, f)
    #         logger.info(f"Saved index to cache: {cache_file}")
    #     except Exception as e:
    #         logger.warning(f"Failed to save index to cache: {e}")
        
    #     return index
