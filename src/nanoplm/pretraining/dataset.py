from torch.utils.data import Dataset
from Bio import SeqIO
from typing import List, Dict
from transformers import PreTrainedTokenizer
from pathlib import Path
import os

from nanoplm.utils.logger import logger


class FastaMLMDataset(Dataset):
    """FASTA dataset that tokenizes sequences for MLM pretraining.

    Uses an on-disk index for random access and defers padding to the collator.

    If initialized with lazy=False, all sequences are tokenized eagerly during
    construction and kept in memory for faster iteration at the cost of RAM.
    """

    def __init__(
        self,
        fasta_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        lazy: bool = False,
    ) -> None:
        self.fasta_path = str(fasta_path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.lazy = bool(lazy)

        # Validate that the FASTA file exists and is readable
        fasta_path_obj = Path(self.fasta_path)
        if not fasta_path_obj.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_path}")

        if not fasta_path_obj.is_file():
            raise ValueError(f"Path is not a file: {self.fasta_path}")

        if not os.access(self.fasta_path, os.R_OK):
            raise PermissionError(f"FASTA file is not readable: {self.fasta_path}")

        # Check if the file has any content
        if fasta_path_obj.stat().st_size == 0:
            raise ValueError(f"FASTA file is empty: {self.fasta_path}")

        # Create or open a persistent SQLite-backed index for random access.
        # This avoids storing all sequences in RAM.
        self._db_path = f"{self.fasta_path}.idx"
        self._index = SeqIO.index_db(self._db_path, [self.fasta_path], "fasta")
        self._keys: List[str] = list(self._index.keys())

        if len(self._keys) == 0:
            raise ValueError(f"No sequences found in FASTA: {self.fasta_path}")

        logger.info(
            f"Loaded FASTA: {self.fasta_path} with {len(self._keys):,} sequences (max_length={self.max_length}, lazy={self.lazy})."
        )

        # If not lazy, tokenize all sequences now and store them in memory
        self._encodings = None
        if not self.lazy:
            encodings: List[Dict[str, any]] = []
            for key in self._keys:
                record = self._index[key]
                sequence = str(record.seq)
                sequence = self.tokenizer.preprocess(sequence)

                encoding = self.tokenizer(
                    sequence,
                    add_special_tokens=True,
                    padding=False,  # defer padding to the collator for dynamic padding
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                encodings.append(
                    {
                        "input_ids": encoding["input_ids"].squeeze(0),
                        "attention_mask": encoding["attention_mask"].squeeze(0),
                    }
                )

            self._encodings = encodings
            logger.info(
                f"Eagerly tokenized and cached {len(self._encodings):,} sequences in memory."
            )

    def __len__(self) -> int:
        if self.lazy:
            return len(self._keys)
        return len(self._encodings) if self._encodings is not None else 0

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        if not self.lazy:
            return self._encodings[idx]

        key = self._keys[idx]
        record = self._index[key]
        sequence = str(record.seq)
        sequence = self.tokenizer.preprocess(sequence)

        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            padding=False,  # defer padding to the collator for dynamic padding
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": encoding["attention_mask"].squeeze(0),  # Remove batch dimension
        }
