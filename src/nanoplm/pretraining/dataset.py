import os
import torch
import bisect
import numpy as np
from math import ceil
from Bio import SeqIO
from typing import List, Dict
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from concurrent.futures import ProcessPoolExecutor

from nanoplm.utils import logger, create_dirs


class SaveShardedFastaMLMDataset:
    """Utility class to tokenize FASTA sequences and save them as flat binary shards.

    This class handles the preprocessing step: reading FASTA, tokenizing sequences,
    and saving them to binary files (.bin + .idx) for fast loading during training.

    This is NOT a Dataset - it's a preprocessing utility that creates shards.
    """

    def __init__(
        self,
        fasta_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        output_dir: str,
        samples_per_shard: int = 10000,
        max_workers: int = -1,
        force: bool = False,
    ) -> None:
        """
        Args:
            fasta_path: Path to input FASTA file
            tokenizer: Tokenizer to use for encoding sequences
            max_length: Maximum sequence length
            output_dir: Directory to save binary shards
            samples_per_shard: Number of sequences per shard file
            max_workers: Number of parallel workers (-1 = all CPUs)
            force: If True, overwrite existing shards
        """
        self.fasta_path = str(fasta_path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.output_dir = Path(output_dir)
        self.samples_per_shard = int(samples_per_shard)
        self.max_workers = os.cpu_count() if max_workers == -1 else max_workers
        self.force = bool(force)

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

        # Create or open a persistent SQLite-backed index for random access
        self._db_path = f"{self.fasta_path}.idx"
        self._index = SeqIO.index_db(self._db_path, [self.fasta_path], "fasta")
        self._keys: List[str] = list(self._index.keys())

        if len(self._keys) == 0:
            raise ValueError(f"No sequences found in FASTA: {self.fasta_path}")

        logger.info(
            f"Loaded FASTA: {self.fasta_path} with {len(self._keys):,} sequences (max_length={self.max_length})."
        )

    def create_shards(self) -> List[Path]:
        """Create binary shards from FASTA sequences.

        Returns:
            List of paths to created shard .bin files
        """
        # Check if shards already exist
        shards_exist = (
            self.output_dir.exists() and len(list(self.output_dir.glob("*.bin"))) > 0
        )

        if shards_exist and not self.force:
            raise FileExistsError(
                f"Binary shards already exist in {self.output_dir}. "
                f"Set force=True to overwrite them."
            )

        if shards_exist and self.force:
            logger.warning(f"Overwriting existing shards in {self.output_dir}")
            for shard in self.output_dir.glob("*.bin"):
                shard.unlink()
            for shard in self.output_dir.glob("*.idx.npy"):
                shard.unlink()

        # Create output directory
        create_dirs(self.output_dir)

        total_seqs = len(self._keys)
        num_shards = ceil(total_seqs / self.samples_per_shard)

        logger.info(
            f"Splitting {total_seqs:,} sequences into {num_shards} shards "
            f"({self.samples_per_shard:,} sequences per shard)"
        )

        # Split keys into shards
        sharded_keys = [
            self._keys[i * self.samples_per_shard : (i + 1) * self.samples_per_shard]
            for i in range(num_shards)
        ]

        # Prepare arguments for parallel processing
        args = [
            (
                i,
                self.fasta_path,
                self._db_path,
                keys,
                self.tokenizer,
                self.max_length,
                self.output_dir,
            )
            for i, keys in enumerate(sharded_keys)
        ]

        # Process shards in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            shard_paths = list(executor.map(process_shard, args))

        logger.info(
            f"Successfully tokenized {total_seqs:,} sequences and saved to {len(shard_paths)} binary shards in {self.output_dir}"
        )

        return [Path(p) for p in shard_paths]


class LoadShardedFastaMLMDataset(Dataset):
    """Dataset for loading pre-tokenized sequences from flat binary shards via memmap.

    Each shard consists of:
    - shard_NNNN.bin: concatenated uint8 tokens (all sequences back-to-back)
    - shard_NNNN.idx: numpy .npy file containing int32 array of sequence lengths

    Uses np.memmap for zero-copy reads; forked DataLoader workers share physical
    pages via OS page cache automatically.
    """

    def __init__(self, data_dir: str, load_all_in_memory: bool = False) -> None:
        """
        Args:
            data_dir: Directory containing binary shard files (*.bin + *.idx)
            load_all_in_memory: Accepted for API compatibility but ignored
                (memmap is already the optimal strategy)
        """
        self.data_dir = Path(data_dir)

        # Validate directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if not self.data_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.data_dir}")

        # Find all shard .bin files
        bin_paths = sorted(self.data_dir.glob("*.bin"))

        if len(bin_paths) == 0:
            raise FileNotFoundError(
                f"No binary shard files (*.bin) found in {self.data_dir}"
            )

        logger.info(f"Found {len(bin_paths)} binary shards in {self.data_dir}")

        # Load index files and create memmaps
        self._mmaps: List[np.memmap] = []
        self._offsets: List[np.ndarray] = []  # per-shard cumulative offsets
        self._sizes: List[np.ndarray] = []  # per-shard sequence lengths
        self.lengths: List[int] = []

        for bin_path in bin_paths:
            idx_path = bin_path.with_name(bin_path.stem + ".idx.npy")
            if not idx_path.exists():
                raise FileNotFoundError(
                    f"Index file not found for shard: {idx_path}"
                )

            sizes = np.load(idx_path)  # int32 array of sequence lengths
            offsets = np.concatenate([[0], np.cumsum(sizes)])

            mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")

            self._mmaps.append(mmap)
            self._offsets.append(offsets)
            self._sizes.append(sizes)
            self.lengths.append(len(sizes))

        self.cum_lengths = np.cumsum(self.lengths)

        logger.info(
            f"Loaded {int(self.cum_lengths[-1]):,} pre-tokenized sequences from {len(bin_paths)} shards"
        )

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        shard_idx, local_idx = self._get_shard(idx)

        offsets = self._offsets[shard_idx]
        start = int(offsets[local_idx])
        end = int(offsets[local_idx + 1])

        raw = self._mmaps[shard_idx][start:end]
        input_ids = torch.tensor(np.array(raw), dtype=torch.long)

        # Generate attention_mask on-the-fly: 1 for non-padding tokens, 0 for padding (pad_token_id=0)
        attention_mask = (input_ids != 0).to(torch.uint8)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _get_shard(self, idx: int) -> tuple[int, int]:
        """Return (shard_index, local_index_in_shard)"""
        shard_idx = bisect.bisect_right(self.cum_lengths, idx)
        if shard_idx > 0:
            idx -= self.cum_lengths[shard_idx - 1]
        return shard_idx, idx

    def cleanup(self):
        """Release memmap objects (public API, kept for compatibility)."""
        for mmap in self._mmaps:
            del mmap
        self._mmaps = []


class LazyFastaMLMDataset(Dataset):
    """FASTA dataset that tokenizes sequences lazily for MLM pretraining.

    Uses an on-disk index for random access and defers padding to the collator.
    """

    def __init__(
        self,
        fasta_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> None:
        self.fasta_path = str(fasta_path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

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

        temp_index = SeqIO.index_db(self._db_path, [self.fasta_path], "fasta")
        self._keys: List[str] = list(temp_index.keys())
        temp_index.close()

        if len(self._keys) == 0:
            raise ValueError(f"No sequences found in FASTA: {self.fasta_path}")

        logger.info(
            f"Loaded FASTA: {self.fasta_path} with {len(self._keys):,} sequences (max_length={self.max_length})."
        )

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        if idx < 0 or idx >= len(self._keys):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        key = self._keys[idx]

        # Create index on-demand to avoid multiprocessing pickle issues
        index = SeqIO.index_db(self._db_path, [self.fasta_path], "fasta")
        try:
            record = index[key]
            sequence = str(record.seq)
        finally:
            index.close()

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
            "attention_mask": encoding["attention_mask"].squeeze(
                0
            ),  # Remove batch dimension
        }


def process_shard(args):
    shard_idx, fasta_path, db_path, shard_keys, tokenizer, max_length, output_dir = args

    # Each process must open its own index
    index = SeqIO.index_db(db_path, [fasta_path], "fasta")

    bin_path = Path(output_dir) / f"shard_{shard_idx:04d}.bin"
    idx_path = Path(output_dir) / f"shard_{shard_idx:04d}.idx.npy"

    token_arrays = []

    for key in tqdm(shard_keys, desc=f"Tokenizing shard {shard_idx}", leave=False):
        record = index[key]
        seq = str(record.seq)

        encoding = tokenizer(
            seq,
            add_special_tokens=True,
            padding=False,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        token_arrays.append(encoding["input_ids"].squeeze(0).numpy().astype(np.uint8))

    # Build sizes array (int32)
    sizes = np.array([len(a) for a in token_arrays], dtype=np.int32)

    # Concatenate all tokens and write .bin
    all_tokens = np.concatenate(token_arrays)
    all_tokens.tofile(str(bin_path))

    # Save sizes as .npy (the .idx file)
    np.save(str(idx_path), sizes)

    index.close()
    return str(bin_path)
