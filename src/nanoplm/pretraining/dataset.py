from torch.utils.data import Dataset
from Bio import SeqIO
from typing import List, Dict
from transformers import PreTrainedTokenizer
from tqdm import tqdm
from pathlib import Path
import torch
import os
import h5py
import numpy as np
from math import ceil
from concurrent.futures import ProcessPoolExecutor
from nanoplm.utils.logger import logger
import bisect

def process_shard(args):
    shard_idx, fasta_path, db_path, shard_keys, tokenizer, max_length, output_dir = args

    # Each process must open its own index
    index = SeqIO.index_db(db_path, [fasta_path], "fasta")

    shard_path = Path(output_dir) / f"train_{shard_idx:04d}.h5"
    input_ids_list = []
    attention_masks = []

    for key in tqdm(shard_keys, desc=f"Tokenizing shard {shard_idx}", leave=False):
        record = index[key]
        seq = str(record.seq)
        seq_len = len(seq)

        # easier to just split the input FASTA

        # # for seqs above max_length
        # if seq_len > max_length:

        #     for start in range(0, len(seq), max_length):
        #         chunk = seq[start:start + max_length]

        #         encoding = tokenizer(
        #             chunk,
        #             add_special_tokens=True,
        #             padding=False,
        #             max_length=max_length,
        #             truncation=True,
        #             return_tensors="pt",
        #         )

        #         input_ids_list.append(encoding["input_ids"].squeeze(0))
        #         attention_masks.append(encoding["attention_mask"].squeeze(0))

        # else:

        encoding = tokenizer(
            seq,
            add_special_tokens=True,
            padding=False,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids_list.append(encoding["input_ids"].squeeze(0))
        attention_masks.append(encoding["attention_mask"].squeeze(0))

    # Write results
    with h5py.File(shard_path, "w") as h5f:
        total = len(input_ids_list)
        h5f.create_dataset('input_ids', (total,), dtype=h5py.special_dtype(vlen=np.int32))
        h5f.create_dataset('attention_mask', (total,), dtype=h5py.special_dtype(vlen=np.int32))

        for i in tqdm(range(total), desc=f"Writing Shard {shard_idx}", leave=False):
            h5f['input_ids'][i] = np.array(input_ids_list[i], dtype=np.int32)
            h5f['attention_mask'][i] = np.array(attention_masks[i], dtype=np.int32)

    index.close()
    return str(shard_path)

class FastaMLMDataset(Dataset):
    """FASTA dataset that tokenizes sequences for MLM pretraining.

    Uses an on-disk index for random access and defers padding to the collator.

    If initialized with lazy=False, all sequences are tokenized eagerly during
    construction and kept on disc in a large hdf5 file for fast lookup during training
    """

    def __init__(
        self,
        fasta_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        lazy: bool = False,
        hdf5_dir: str = None,
        samples_per_shard: int = 1000000,
        max_workers: int = 1,
        load_shards: bool = False
    ) -> None:
        self.fasta_path = str(fasta_path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.lazy = bool(lazy)
        self.hdf5_dir = Path(hdf5_dir)
        self.samples_per_shard = int(samples_per_shard)
        self.max_workers = int(max_workers)
        self.load_shards = bool(load_shards)


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

            if not self.load_shards:


                
                # bool for actually creating hdf5
                create_hdf5 = True

                # Validate if the FASTA file exists and is readable - if not skip
                hdf5_path_obj = Path(self.hdf5_dir)


                if create_hdf5:
                    logger.info(f"Creating {hdf5_path_obj}")

                    # create directory
                    hdf5_path_obj.mkdir(parents=True, exist_ok=True)

                    total_seqs = len(self._keys)
                    num_shards = ceil(total_seqs / self.samples_per_shard)

                    logger.info(f"Splitting {total_seqs:,} sequences into {num_shards} shards "
                        f"({self.samples_per_shard:,} per file)")

                    sharded_keys = [
                        self._keys[i * samples_per_shard : (i + 1) * samples_per_shard]
                        for i in range(num_shards)
                    ]

                    
                    args = [
                        (i, self.fasta_path, self._db_path, keys, self.tokenizer, self.max_length, hdf5_path_obj)
                        for i, keys in enumerate(sharded_keys)
                    ]

                    with ProcessPoolExecutor(max_workers=self.max_workers or os.cpu_count()) as executor:
                        results = list(executor.map(process_shard, args))

                    # with h5py.File(hdf5_path_obj, "w") as h5f:
                    #     index = 0 
                    #     total_groups = len(self._keys)
                    #     h5f.create_dataset('input_ids', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))
                    #     h5f.create_dataset('attention_mask', (total_groups,), dtype=h5py.special_dtype(vlen=np.int32))

                    #     i = 0
                    #     for key in tqdm(self._keys, desc="Tokenizing sequences", total=len(self._keys)):
                    #         record = self._index[key]
                    #         sequence = str(record.seq)
                    #         sequence = self.tokenizer.preprocess(sequence)

                    #         encoding = self.tokenizer(
                    #             sequence,
                    #             add_special_tokens=True,
                    #             padding=False,  # defer padding to the collator for dynamic padding
                    #             max_length=self.max_length,
                    #             truncation=True,
                    #             return_tensors="pt",
                    #         )

                    #         h5f["input_ids"][index] = encoding["input_ids"].squeeze(0).cpu().numpy()
                    #         h5f["attention_mask"][index] =  encoding["attention_mask"].squeeze(0).cpu().numpy()
                    #         index += 1

                    logger.info(
                        f"Eagerly tokenized {len(self._keys):,} sequences and saved to hdf5."
                    )
                else:
                    logger.info(f"✅ {hdf5_path_obj} has correct format.")
        


            # Discover all shard files in directory (sorted)
            self.shard_paths = sorted(self.hdf5_dir.glob("*.h5"))
            if not self.shard_paths:
                raise FileNotFoundError(f"No HDF5 shards found in {self.hdf5_dir}")

            # Open shards or just record lengths
            self.shards = []
            self.lengths = []

            for path in self.shard_paths:
                with h5py.File(path, "r") as f:
                    n = len(f["input_ids"])
                self.lengths.append(n)
                self.shards.append(h5py.File(path, "r"))

            self.cum_lengths = np.cumsum(self.lengths)

            self.shards = [h5py.File(path, "r") for path in self.shard_paths]




    def __len__(self) -> int:
        if self.lazy:
            return len(self._keys)
        return int(self.cum_lengths[-1])

    def _get_shard(self, idx):
        """Return (shard_index, local_index_in_shard)"""
        shard_idx = bisect.bisect_right(self.cum_lengths, idx)
        if shard_idx > 0:
            idx -= self.cum_lengths[shard_idx - 1]
        return shard_idx, idx


    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        if not self.lazy:
            # # read from the hdf5 file
            # input_ids = torch.tensor(self.h5f['input_ids'][idx], dtype=torch.long)
            # attention_mask = torch.tensor(self.h5f['attention_mask'][idx], dtype=torch.long)

            # return {
            #     "input_ids": input_ids.squeeze(0),
            #     "attention_mask": attention_mask.squeeze(0),
            # }
            # determine which shard and local index

            shard_idx = bisect.bisect_right(self.cum_lengths, idx)
            local_idx = idx if shard_idx == 0 else idx - self.cum_lengths[shard_idx - 1]

            # Open shard and read
            with h5py.File(self.shard_paths[shard_idx], "r") as f:
                input_ids = torch.tensor(f["input_ids"][local_idx], dtype=torch.long)
                attention_mask = torch.tensor(f["attention_mask"][local_idx], dtype=torch.long)


            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
    }

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
