from pathlib import Path
from typing import Union, Dict

from Bio import SeqIO
from tqdm import tqdm

from nanoplm.utils import logger, create_dirs


class SplitError(Exception):
    """Raised when a split operation fails."""

    pass


class Splitor:
    """
    Split a filtered FASTA file into train/val according to a ratio.
    """

    def __init__(
        self,
        input_file: Union[str, Path],
        train_file: Union[str, Path],
        val_file: Union[str, Path],
        val_ratio: float,
    ):
        self.input_file = Path(input_file)
        self.train_file = Path(train_file)
        self.val_file = Path(val_file)
        self.val_ratio = float(val_ratio)

    def split(self):
        """Split filtered sequences into train and val files."""

        if not self.input_file.exists():
            raise SplitError(f"Filtered FASTA not found: {self.input_file}")

        logger.info(f"Creating splits with val ratio {self.val_ratio}")

        sequences = list(SeqIO.parse(self.input_file, "fasta"))
        num_filtered_seqs = len(sequences)

        val_size = int(num_filtered_seqs * self.val_ratio)
        train_size = num_filtered_seqs - val_size

        logger.info(f"Sequences: {num_filtered_seqs}, Train: {train_size}, Val: {val_size}")

        with tqdm(total=num_filtered_seqs, desc="Splitting data") as pbar:
            SeqIO.write(sequences[:train_size], self.train_file, "fasta")
            SeqIO.write(sequences[train_size:], self.val_file, "fasta")
            pbar.update(num_filtered_seqs)
        
        return (train_size, val_size)
