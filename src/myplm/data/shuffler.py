import random
from pathlib import Path
from typing import Union, Optional
from Bio import SeqIO
from tqdm import tqdm

from myplm.utils import logger, create_dirs

class FastaShuffler:
    """Memory-efficient FASTA shuffler using BioPython's indexing capabilities."""

    def __init__(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        seed: Optional[int] = None,
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.seed = seed
        
        if self.seed is not None:
            random.seed(self.seed)

    def shuffle(self):
        """Shuffles sequences using BioPython's memory-efficient indexing."""
        create_dirs(self.output_file.parent)

        # Check if input file exists
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        logger.info(f"Creating BioPython index for {self.input_file}...")
        try:
            # Create memory-efficient index - doesn't load sequences into memory
            record_dict = SeqIO.index(str(self.input_file), "fasta")
            sequence_ids = list(record_dict.keys())
            logger.info(f"Indexed {len(sequence_ids)} sequences")
            
        except Exception as e:
            logger.error(f"Error creating BioPython index: {e}")
            raise

        if not sequence_ids:
            logger.warning("No sequences found in input file")
            return

        # Shuffle sequence IDs (lightweight operation)
        logger.info(f"Shuffling {len(sequence_ids)} sequence IDs...")
        random.shuffle(sequence_ids)

        # Write sequences in shuffled order
        logger.info(f"Writing shuffled sequences to {self.output_file}...")
        try:
            with open(self.output_file, "w") as output_handle:
                with tqdm(total=len(sequence_ids), desc="Writing sequences") as pbar:
                    for seq_id in sequence_ids:
                        record = record_dict[seq_id]  # Fast random access
                        SeqIO.write(record, output_handle, "fasta")
                        pbar.update(1)
                        
        except Exception as e:
            logger.error(f"Error writing shuffled FASTA file {self.output_file}: {e}")
            raise
        finally:
            # Clean up the index
            record_dict.close()
            
        logger.info(f"Successfully shuffled {len(sequence_ids)} sequences to {self.output_file}")

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Create a dummy fasta file
    dummy_input_path = Path("dummy_input.fasta")
    dummy_output_path = Path("dummy_shuffled_output.fasta")
    
    with open(dummy_input_path, "w") as f:
        f.write(">seq1\nACGT\n")
        f.write(">seq2\nGTCA\n")
        f.write(">seq3\nTTTT\n")
        f.write(">seq4\nAAAA\n")
        f.write(">seq5\nCCCC\n")

    shuffler = FastaShuffler(dummy_input_path, dummy_output_path, seed=42)
    shuffler.shuffle()

    # Clean up dummy files
    dummy_input_path.unlink()
    dummy_output_path.unlink()
    logger.info("Shuffler test complete and dummy files cleaned up.") 