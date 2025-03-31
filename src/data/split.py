import random
from tqdm import tqdm
from .base import BaseProcessor, logger

class Splitter(BaseProcessor):
    """Class for splitting the processed data into train and validation sets."""
    
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

if __name__ == "__main__":
    # For direct execution
    splitter = Splitter()
    splitter.split()
