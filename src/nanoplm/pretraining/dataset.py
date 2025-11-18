import os
import torch
import h5py
import bisect
import numpy as np
from math import ceil
from Bio import SeqIO
from typing import List, Dict
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed

from nanoplm.utils import logger, create_dirs


class SaveShardedFastaMLMDataset:
    """Utility class to tokenize FASTA sequences and save them as HDF5 shards.

    This class handles the preprocessing step: reading FASTA, tokenizing sequences,
    and saving them to HDF5 files for fast loading during training.

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
            output_dir: Directory to save HDF5 shards
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
        """Create HDF5 shards from FASTA sequences.

        Returns:
            List of paths to created shard files
        """
        # Check if shards already exist
        shards_exist = (
            self.output_dir.exists() and len(list(self.output_dir.glob("*.h5"))) > 0
        )

        if shards_exist and not self.force:
            raise FileExistsError(
                f"HDF5 shards already exist in {self.output_dir}. "
                f"Set force=True to overwrite them."
            )

        if shards_exist and self.force:
            logger.warning(f"Overwriting existing shards in {self.output_dir}")
            for shard in self.output_dir.glob("*.h5"):
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
            f"Successfully tokenized {total_seqs:,} sequences and saved to {len(shard_paths)} HDF5 shards in {self.output_dir}"
        )

        return [Path(p) for p in shard_paths]


class LoadShardedFastaMLMDataset(Dataset):
    """Dataset for loading pre-tokenized sequences from HDF5 shards.

    Supports two modes:
    - streaming (default)
    - load_all_in_memory: read all shards into memory (dict) at init
    """

    def __init__(self, hdf5_dir: str, load_all_in_memory: bool = False) -> None:
        """
        Args:
            hdf5_dir: Directory containing HDF5 shard files (*.h5)
            load_all_in_memory: Whether to load all shards into memory (default: False)
        """
        self.hdf5_dir = Path(hdf5_dir)
        self._in_memory = bool(load_all_in_memory)

        # Validate directory exists
        if not self.hdf5_dir.exists():
            raise FileNotFoundError(f"HDF5 directory not found: {self.hdf5_dir}")

        if not self.hdf5_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.hdf5_dir}")

        # Find all shard files
        self.shard_paths = sorted(self.hdf5_dir.glob("*.h5"))

        if len(self.shard_paths) == 0:
            raise FileNotFoundError(
                f"No HDF5 shard files (*.h5) found in {self.hdf5_dir}"
            )

        logger.info(f"Found {len(self.shard_paths)} HDF5 shards in {self.hdf5_dir}")

        # Read shard lengths without keeping files open
        self.lengths = []
        for path in self.shard_paths:
            with h5py.File(path, "r") as f:
                n = len(f["input_ids"])
            self.lengths.append(n)

        self.cum_lengths = np.cumsum(self.lengths)

        logger.info(
            f"Loaded {int(self.cum_lengths[-1]):,} pre-tokenized sequences from {len(self.shard_paths)} shards"
        )

        # In-memory option: load all shard data into Python lists
        if self._in_memory:
            logger.info("Loading all shards into memory (parallel)...")
            num_shards = len(self.shard_paths)
            # Decide number of workers: use up to half of CPUs but not more than shards
            try:
                max_workers = max(1, min((os.cpu_count() or 1) // 2, num_shards))
            except Exception:
                max_workers = 1

            # Prepare result container
            self._in_memory_shards = [None] * num_shards

            # Use ProcessPoolExecutor to let each worker open its own HDF5 file
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    futures = {exe.submit(_read_shard_for_worker, str(p)): idx for idx, p in enumerate(self.shard_paths)}
                    for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading shards", leave=False):
                        idx = futures[fut]
                        try:
                            _, data = fut.result()
                            self._in_memory_shards[idx] = data
                        except Exception:
                            # If any worker fails, fall back to sequential loading
                            logger.warning("Parallel shard loading failed; falling back to sequential load.")
                            raise

                # Verify none are None
                for i, entry in enumerate(self._in_memory_shards):
                    if entry is None:
                        raise RuntimeError(f"Shard {i} not loaded correctly")

                # Build flat index for O(1) global access
                self._build_flat_index()
            except Exception:
                # Fallback: sequential load with safe bulk reads
                logger.info("Falling back to sequential shard loading...")
                self._in_memory_shards = []
                for i, path in enumerate(self.shard_paths, start=1):
                    if i % 10 == 0 or i == num_shards:
                        logger.info(f"Loading shard {i}/{num_shards}...")
                    with h5py.File(path, "r") as f:
                        n = len(f["input_ids"])
                        inputs = [None] * n
                        masks = [None] * n
                        for j in range(n):
                            inputs[j] = np.array(f["input_ids"][j], dtype=np.int32)
                            masks[j] = np.array(f["attention_mask"][j], dtype=np.int32)
                        self._in_memory_shards.append((inputs, masks))

                # Build flat index for O(1) global access after sequential fallback
                self._build_flat_index()
        else:
            logger.info("Using streaming mode.")

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        shard_idx, local_idx = self._get_shard(idx)

        # In-memory fast path
        if self._in_memory:
            # If flat indexing exists, use it for true O(1) access
            if hasattr(self, "_flat_inputs") and hasattr(self, "_flat_masks"):
                input_ids = torch.tensor(self._flat_inputs[idx], dtype=torch.long)
                attention_mask = torch.tensor(self._flat_masks[idx], dtype=torch.long)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            inputs, masks = self._in_memory_shards[shard_idx]
            input_ids = torch.tensor(inputs[local_idx], dtype=torch.long)
            attention_mask = torch.tensor(masks[local_idx], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        # Streaming path: open file on-demand
        with h5py.File(self.shard_paths[shard_idx], "r") as f:
            input_ids = torch.tensor(f["input_ids"][local_idx], dtype=torch.uint8)

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

    def _build_flat_index(self) -> None:
        """Build flat input/mask lists from self._in_memory_shards for O(1) global indexing."""
        self._flat_inputs = []
        self._flat_masks = []
        for inputs, masks in self._in_memory_shards:
            self._flat_inputs.extend(inputs)
            self._flat_masks.extend(masks)
        logger.info(f"Flat indexing enabled: {len(self._flat_inputs):,} samples")


def process_shard(args):
    shard_idx, fasta_path, db_path, shard_keys, tokenizer, max_length, output_dir = args

    # Each process must open its own index
    index = SeqIO.index_db(db_path, [fasta_path], "fasta")

    shard_path = Path(output_dir) / f"shard_{shard_idx:04d}.h5"
    input_ids_list = []

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

        input_ids_list.append(encoding["input_ids"].squeeze(0))

    # Write results
    with h5py.File(shard_path, "w") as h5f:
        total = len(input_ids_list)
        h5f.create_dataset(
            "input_ids", (total,), dtype=h5py.special_dtype(vlen=np.uint8)
        )

        for i in tqdm(range(total), desc=f"Writing Shard {shard_idx}", leave=False):
            h5f["input_ids"][i] = np.array(input_ids_list[i], dtype=np.uint8)

    index.close()
    return str(shard_path)


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
        self._index = SeqIO.index_db(self._db_path, [self.fasta_path], "fasta")
        self._keys: List[str] = list(self._index.keys())

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
            "attention_mask": encoding["attention_mask"].squeeze(
                0
            ),  # Remove batch dimension
        }


def _read_shard_for_worker(path_str: str):
    """Helper for parallel shard loading in separate processes.

    Returns a tuple (indexable_path_str, (inputs_list, masks_list)).
    """

    path = Path(path_str)
    inputs = []
    masks = []
    with h5py.File(path, "r") as f:
        n = len(f["input_ids"])
        # Bulk read each variable-length element into a numpy array
        for i in range(n):
            inputs.append(np.array(f["input_ids"][i], dtype=np.int32))
            masks.append(np.array(f["attention_mask"][i], dtype=np.int32))

    return path_str, (inputs, masks)
