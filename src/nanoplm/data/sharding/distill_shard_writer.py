from __future__ import annotations

import math
import h5py
import torch
import numpy as np

from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from typing import TYPE_CHECKING, Union, List, Optional
from torch.utils.data import Dataset

from nanoplm.utils import logger, get_device, create_dirs

if TYPE_CHECKING:
    from nanoplm.distillation.models.teacher import BaseTeacher


class SaveKDDataset(Dataset):
    def __init__(
        self,
        input_fasta: Union[str, Path],
        output_path: Union[str, Path],
        teacher: BaseTeacher,
        mode: str,
        max_seq_len: int,
        batch_size: int,
        device: str,
        skip_n: int = 0,
        samples_per_shard: int = 10000,
        force: bool = False,
    ):
        """
        Save knowledge distillation dataset with teacher embeddings.

        Args:
            input_fasta: Path to input FASTA file
            output_path: Path to output H5 file (or prefix for sharded files)
            teacher: Teacher model for generating embeddings
            mode: Processing mode (currently only "get_embeddings")
            max_seq_len: Maximum sequence length
            batch_size: Batch size for teacher embedding calculation
            device: Device to use ("auto", "cuda", "mps", "cpu")
            skip_n: Number of sequences to skip from the beginning
            samples_per_shard: Number of samples per shard file (-1 for single file)
            force: Force overwrite existing files
        """
        self.input_fasta = Path(input_fasta)
        self.output_path = Path(output_path)
        self.teacher = teacher
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device if device != "auto" else get_device()
        self.skip_n = skip_n

        # Validate samples_per_shard
        if samples_per_shard < -1 or samples_per_shard == 0:
            raise ValueError(
                f"samples_per_shard must be -1 (no sharding) or a positive number, got {samples_per_shard}"
            )

        self.samples_per_shard = samples_per_shard
        self.force = force
        self.data_gen = None
        self._cached_len = None

    def __len__(self):
        if self._cached_len is None:
            self._cached_len = max(
                0, sum(1 for _ in SeqIO.parse(self.input_fasta, "fasta")) - self.skip_n
            )
        return self._cached_len

    def _load(self):
        if not self.data_gen:
            raw_generator = SeqIO.parse(self.input_fasta, "fasta")

            if self.skip_n > 0:
                logger.info(
                    f"Skipping first {self.skip_n} sequences from {self.input_fasta}."
                )
                for _ in range(self.skip_n):
                    try:
                        next(raw_generator)
                    except StopIteration:
                        logger.warning(
                            f"Tried to skip {self.skip_n} sequences, but FASTA file has fewer."
                        )
                        break

            self.data_gen = ((record.id, str(record.seq)) for record in raw_generator)
            logger.info(
                f"{self.input_fasta} initialized (with skip_n={self.skip_n}). Now ready for processing."
            )

    def process_dataset(self) -> Union[Path, List[Path]]:
        self._load()

        if self.mode == "get_embeddings":
            self.teacher_model = self.teacher.encoder_model
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.teacher_tokenizer = self.teacher.tokenizer
        create_dirs(self.output_path)

        total_sequences_in_fasta = self.__len__() + self.skip_n
        total_to_process = self.__len__()

        # Log dataset summary
        skip_info = f" (skipping {self.skip_n})" if self.skip_n > 0 else ""

        logger.info(
            f"Dataset: {total_sequences_in_fasta:,} sequences → {total_to_process:,} to process{skip_info}"
        )

        # Determine if sharding is needed based on samples_per_shard
        if self.samples_per_shard == -1 or total_to_process <= self.samples_per_shard:
            # Single file mode
            logger.info(f"Single file will be created at {self.output_path}")
            return self._process_dataset_single()
        else:
            # Calculate number of shards needed
            self.n_files = math.ceil(total_to_process / self.samples_per_shard)
            logger.info(
                f"{self.n_files} sharded files will be created "
                f"(~{self.samples_per_shard:,} samples per shard) at {self.output_path.parent}"
            )
            return self._process_dataset_sharded()

    def _process_dataset_single(self) -> Path:
        """Process dataset into a single H5 file (original behavior)"""

        batch = []

        if self.output_path.exists():
            if self.force:
                logger.info(
                    f"Found existing HDF5 file at {self.output_path}. Will overwrite existing file."
                )
                self.output_path.unlink()
            else:
                raise FileExistsError(
                    f"Found existing HDF5 file at {self.output_path}. Use --force to overwrite existing file."
                )
        else:
            logger.info(
                f"No existing HDF5 file at {self.output_path}. Creating new file."
            )

        processed_sequences = 0
        with h5py.File(self.output_path, "w") as h5f:

            with tqdm(
                total=self.__len__(), desc="Generating embeddings", unit="seq"
            ) as pbar:
                for _, sequence in self.data_gen:
                    teacher_seq = self.teacher.preprocess(sequence)
                    batch.append(teacher_seq)

                    # Process the batch if it's full
                    if len(batch) == self.batch_size:
                        self._process_and_save_batch(
                            h5f=h5f, batch=batch, start_index=processed_sequences
                        )
                        processed_sequences += len(batch)
                        pbar.update(len(batch))
                        batch = []

                # Process any remaining sequences in the last batch
                if batch:
                    self._process_and_save_batch(
                        h5f=h5f, batch=batch, start_index=processed_sequences
                    )
                    processed_sequences += len(batch)
                    pbar.update(len(batch))

        logger.info(f"Processed and saved {processed_sequences} new sequences.")
        logger.info(f"Dataset: {self.output_path}.")
        return self.output_path

    def _process_dataset_sharded(self) -> List[Path]:
        """Process dataset into multiple sharded H5 files based on samples_per_shard"""

        # Generate shard file names
        base_name = self.output_path.stem
        output_dir = self.output_path.parent
        shard_paths = [
            output_dir / f"{base_name}_shard_{i}.h5" for i in range(self.n_files)
        ]

        # Check for existing files and handle them
        for shard_path in shard_paths:
            if shard_path.exists():
                if self.force:
                    logger.info(
                        f"Found existing sharded file at {shard_path}. Will overwrite existing file."
                    )
                    shard_path.unlink()
                else:
                    raise FileExistsError(
                        f"Found existing sharded file at {shard_path}. Use --force to overwrite existing file."
                    )
            else:
                logger.info(f"Creating new shard file: {shard_path.name}")

        # Process sequences and write to one shard file at a time
        batch = []
        current_shard_idx = 0
        current_shard_count = 0
        processed_sequences = 0
        current_h5_file = None

        try:
            with tqdm(
                total=self.__len__(), desc="Generating sharded embeddings", unit="seq"
            ) as pbar:
                for _, sequence in self.data_gen:
                    # Check if we need to open the first file
                    if current_h5_file is None:
                        current_h5_file = h5py.File(shard_paths[current_shard_idx], "w")
                        logger.info(
                            f"Opened shard file: {shard_paths[current_shard_idx].name}"
                        )

                    teacher_seq = self.teacher.preprocess(sequence)
                    batch.append(teacher_seq)

                    if len(batch) == self.batch_size:
                        # Check if current shard is full and we're not on the last shard
                        if (
                            current_shard_count + len(batch) > self.samples_per_shard
                            and current_shard_idx < self.n_files - 1
                        ):
                            # Close current file and move to next
                            current_h5_file.close()
                            logger.info(
                                f"Closed shard file: {shard_paths[current_shard_idx].name} with {current_shard_count} sequences"
                            )

                            current_shard_idx += 1
                            current_shard_count = 0

                            # Open next shard file
                            current_h5_file = h5py.File(
                                shard_paths[current_shard_idx], "w"
                            )
                            logger.info(
                                f"Opened shard file: {shard_paths[current_shard_idx].name}"
                            )

                        # Process the batch
                        self._process_and_save_batch(
                            h5f=current_h5_file,
                            batch=batch,
                            start_index=current_shard_count,
                        )
                        current_shard_count += len(batch)
                        processed_sequences += len(batch)
                        pbar.update(len(batch))
                        batch = []

                # Process any remaining sequences in the last batch
                if batch:
                    self._process_and_save_batch(
                        h5f=current_h5_file,
                        batch=batch,
                        start_index=current_shard_count,
                    )
                    processed_sequences += len(batch)
                    pbar.update(len(batch))

        finally:
            # Close the currently open file
            if current_h5_file is not None:
                current_h5_file.close()
                logger.info(
                    f"Closed final shard file: {shard_paths[current_shard_idx].name}"
                )

        # Log shard information with progress summary
        logger.info("Sharded processing completed! Summary:")
        total_sequences_after = 0
        created_shard_paths = []

        for i, shard_path in enumerate(shard_paths):
            if shard_path.exists():
                with h5py.File(shard_path, "r") as shard_file:
                    shard_size = len(shard_file.keys())
                    file_size_gb = shard_path.stat().st_size / (1024**3)
                    total_sequences_after += shard_size
                    logger.info(
                        f"  Shard {i:2d}: {shard_size:8,} sequences, {file_size_gb:6.1f} GB"
                    )
                    created_shard_paths.append(shard_path)
            else:
                logger.info(f"  Shard {i:2d}: 0 sequences (not created)")

        logger.info(f"Processed and saved {processed_sequences} new sequences.")
        logger.info(f"Total sequences across all shards: {total_sequences_after:,}")
        logger.info(
            f"Successfully created {len(created_shard_paths)} out of {len(shard_paths)} shard files"
        )

        return created_shard_paths

    def _process_and_save_batch(
        self,
        h5f: h5py.File,
        batch: List[str],
        start_index: int,
    ):
        """
        Process a batch of sequences and save to H5 file.

        IMPORTANT: Stores STUDENT-tokenized input_ids (for student model),
        but computes teacher embeddings using teacher tokenization.
        """
        from nanoplm.distillation.models.student.tokenizer import ProtXTokenizer

        # Teacher tokenization for computing embeddings
        teacher_encoding = self.teacher_tokenizer.batch_encode_plus(
            batch,  # Preprocessed sequences like "M K T"
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # Student tokenization for storage (what student model will receive)
        # Convert preprocessed ("M K T") back to raw ("MKT")
        raw_sequences = [seq.replace(" ", "") for seq in batch]
        student_tokenizer = ProtXTokenizer()
        student_encoding = student_tokenizer.batch_encode_plus(
            raw_sequences,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # Compute teacher embeddings using teacher tokenization
        with torch.no_grad():
            teacher_input_ids = teacher_encoding["input_ids"].to(self.device)
            teacher_attention_mask = teacher_encoding["attention_mask"].to(self.device)

            teacher_embeddings = self.teacher_model(
                input_ids=teacher_input_ids, attention_mask=teacher_attention_mask
            ).last_hidden_state

        # Save STUDENT-tokenized input_ids and attention_mask
        # Save TEACHER embeddings (target for student to match)
        for i, (seq_input_ids, seq_attention_mask, seq_teacher_embeddings) in enumerate(
            zip(
                student_encoding["input_ids"].cpu().numpy(),
                student_encoding["attention_mask"].cpu().numpy(),
                teacher_embeddings.cpu().numpy(),
            )
        ):
            grp = h5f.create_group(str(start_index + i))
            grp.create_dataset(
                "input_ids",
                data=seq_input_ids.astype(np.int8),
            )
            grp.create_dataset(
                "attention_mask",
                data=seq_attention_mask.astype(np.int8),
            )
            grp.create_dataset(
                "teacher_embeddings",
                data=seq_teacher_embeddings.astype(np.float16),
            )


def shard_h5_file(
    input_h5_path: Union[str, Path],
    n_sharded_files: int,
    output_dir: Optional[Union[str, Path]] = None,
    total_sequences: Optional[int] = None,
) -> List[Path]:
    """
    Split a large H5 file into smaller sharded files.

    Args:
        input_h5_path: Path to the input H5 file to shard
        n_sharded_files: Number of shard files to create
        output_dir: Directory to save sharded files (defaults to same as input file)
        total_sequences: Total number of sequences (if known, skips counting)

    Returns:
        List of paths to the created shard files
    """
    input_path = Path(input_h5_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate shard file names
    base_name = input_path.stem  # e.g., "train" from "train.h5"
    shard_paths = [
        output_dir / f"{base_name}_shard_{i}.h5" for i in range(n_sharded_files)
    ]

    # Get input file size for progress display
    input_file_size_gb = input_path.stat().st_size / (1024**3)

    logger.info(
        f"Starting to shard {input_path} ({input_file_size_gb:.1f} GB) into {n_sharded_files} files..."
    )

    # Open input file and get total size
    with h5py.File(input_path, "r") as input_h5:
        if total_sequences is not None:
            logger.info(f"Using provided sequence count: {total_sequences:,}")
            sequences_count = total_sequences
        else:
            logger.info(
                "Counting sequences in H5 file (this may take a while for large files)..."
            )
            sequences_count = len(input_h5.keys())
            logger.info(f"Total sequences: {sequences_count:,}")

        sequences_per_shard = sequences_count // n_sharded_files
        logger.info(f"Sequences per shard: {sequences_per_shard:,}")

        # Create shard files
        shard_files = [h5py.File(path, "w") for path in shard_paths]

        try:
            current_shard_idx = 0
            current_shard_count = 0

            # Enhanced progress bar with more information
            with tqdm(
                total=sequences_count,
                desc=f"Sharding {base_name}.h5",
                unit="seq",
                unit_scale=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                dynamic_ncols=True,
            ) as pbar:

                for seq_idx in range(sequences_count):
                    # Move to next shard if current one is full (except for last shard)
                    if (
                        current_shard_count >= sequences_per_shard
                        and current_shard_idx < n_sharded_files - 1
                    ):
                        current_shard_idx += 1
                        current_shard_count = 0

                    # Copy sequence data to current shard
                    source_group = input_h5[str(seq_idx)]
                    target_group = shard_files[current_shard_idx].create_group(
                        str(current_shard_count)
                    )

                    # Copy all datasets from source to target
                    for dataset_name in source_group.keys():
                        target_group.create_dataset(
                            dataset_name, data=source_group[dataset_name][:]
                        )

                    current_shard_count += 1

                    # Update progress bar with additional info
                    percentage = (seq_idx + 1) / sequences_count * 100
                    pbar.set_postfix(
                        {
                            "Shard": f"{current_shard_idx}/{n_sharded_files-1}",
                            "Progress": f"{percentage:.1f}%",
                        }
                    )
                    pbar.update(1)

        finally:
            # Close all shard files
            for shard_file in shard_files:
                shard_file.close()

    # Log shard information with progress summary
    logger.info("Sharding completed! Summary:")
    total_output_size_gb = 0
    for i, shard_path in enumerate(shard_paths):
        with h5py.File(shard_path, "r") as shard_file:
            shard_size = len(shard_file.keys())
            file_size_gb = shard_path.stat().st_size / (1024**3)
            total_output_size_gb += file_size_gb
            logger.info(
                f"  Shard {i:2d}: {shard_size:8,} sequences, {file_size_gb:6.1f} GB"
            )

    logger.info(f"Total output size: {total_output_size_gb:.1f} GB")
    logger.info(f"Successfully created {len(shard_paths)} shard files")
    return shard_paths
