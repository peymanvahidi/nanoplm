import random
from pathlib import Path
from typing import Union, Optional
from Bio import SeqIO
from tqdm import tqdm

from ..utils import logger, create_dirs

class FastaShuffler:
    """Shuffles sequences in a FASTA file."""

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
        """Reads, shuffles, and writes FASTA sequences."""
        create_dirs(self.output_file.parent)

        logger.info(f"Reading sequences from {self.input_file}...")
        try:
            sequences = list(SeqIO.parse(self.input_file, "fasta"))
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_file}")
            raise
        except Exception as e:
            logger.error(f"Error reading FASTA file {self.input_file}: {e}")
            raise
        
        logger.info(f"Shuffling {len(sequences)} sequences...")
        random.shuffle(sequences)
        
        logger.info(f"Writing {len(sequences)} shuffled sequences to {self.output_file}...")
        try:
            with open(self.output_file, "w") as out_handle:
                SeqIO.write(sequences, out_handle, "fasta")
        except Exception as e:
            logger.error(f"Error writing shuffled FASTA file {self.output_file}: {e}")
            raise
            
        logger.info(f"Successfully shuffled sequences saved to {self.output_file}")

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