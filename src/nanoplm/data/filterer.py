from pathlib import Path
from typing import Union, Dict

from Bio import SeqIO
from tqdm import tqdm

from nanoplm.utils import create_dirs, logger


class FilterError(Exception):
    """Raised when a filtering operation fails."""

    pass


class Filterer:
    """
    Filter sequences in a FASTA file by length and optional maximum count.

    Follows the component style used in downloader/extractor/shuffler.
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        min_seq_len: int,
        max_seq_len: int,
        seqs_num: int,
        skip_n: int = 0,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.min_seq_len = int(min_seq_len)
        self.max_seq_len = int(max_seq_len)
        self.seqs_num = int(seqs_num)
        self.skip_n = int(skip_n)

        # Stats populated after running filter()
        self.processed_seqs_num: int | None = None
        self.num_filtered_seqs: int | None = None

    def filter(self):
        """Apply filters and write passing sequences to output FASTA."""
        create_dirs(self.output_path.parent)

        logger.info(
            "Filtering sequences with parameters: "
            f"min_length={self.min_seq_len}, max_length={self.max_seq_len}, "
            f"seqs_number={self.seqs_num}, skip_n={self.skip_n}"
        )

        seq_count = 0
        passed_count = 0
        skipped_count = 0

        if not self.input_path.exists():
            raise FilterError(f"Input FASTA not found: {self.input_path}")

        logger.info(f"Processing sequences sequentially from {self.input_path}")

        with open(self.output_path, "w") as output_handle:
            for record in tqdm(SeqIO.parse(self.input_path, "fasta"), desc="Processing sequences"):
                if skipped_count < self.skip_n:
                    skipped_count += 1
                    continue

                seq_count += 1

                if self.seqs_num != -1 and passed_count >= self.seqs_num:
                    break

                seq_len = len(record.seq)
                if self.min_seq_len <= seq_len <= self.max_seq_len:
                    SeqIO.write([record], output_handle, "fasta")
                    passed_count += 1

        logger.info(
            f"Processed {seq_count} sequences from the considered set (after skipping {self.skip_n}): "
            f"{passed_count} sequences retrieved with length in [{self.min_seq_len}, {self.max_seq_len}]."
        )
        logger.info(f"Filtered output saved to: {self.output_path}")

        self.processed_seqs_num = seq_count
        self.num_filtered_seqs = passed_count
