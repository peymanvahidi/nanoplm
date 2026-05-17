from pathlib import Path
from typing import Union

from nanoplm.utils import create_dirs, logger
from nanoplm.utils.common import run_with_heartbeat
from nanoplm.data.preparation._native_ops import (
    filter_fasta_native,
)


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
        *,
        exclude_ids: "Path | None" = None,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.min_seq_len = int(min_seq_len)
        self.max_seq_len = int(max_seq_len)
        self.seqs_num = int(seqs_num)
        self.exclude_ids = exclude_ids

        # Stats populated after running filter()
        self.processed_seqs_num: int | None = None
        self.num_filtered_seqs: int | None = None

    def filter(self):
        """Apply filters and write passing sequences to output FASTA."""
        if self.exclude_ids is not None:
            raise NotImplementedError("exclude_ids requires C support — landed in Phase 10")

        create_dirs(self.output_path.parent)

        logger.info(
            "Filtering sequences with parameters: "
            f"min_length={self.min_seq_len}, max_length={self.max_seq_len}, "
            f"seqs_number={self.seqs_num}"
        )

        if not self.input_path.exists():
            raise FilterError(f"Input FASTA not found: {self.input_path}")

        last_percent = -1

        def progress_cb(_phase: int, progress: float, completed: int, total: int) -> None:
            nonlocal last_percent
            percent = int(progress * 100.0)
            if percent < 100 and percent % 5 != 0:
                return
            if percent == last_percent:
                return
            last_percent = percent
            logger.info(
                "Filter progress: %d%% (%d/%d bytes)",
                percent,
                completed,
                total,
            )

        result = run_with_heartbeat(
            "Filtering FASTA records (native backend)",
            lambda: filter_fasta_native(
                input_path=self.input_path,
                output_path=self.output_path,
                min_seq_len=self.min_seq_len,
                max_seq_len=self.max_seq_len,
                seqs_num=self.seqs_num,
                progress_cb=progress_cb,
            ),
        )
        seq_count = result.processed_seqs
        passed_count = result.passed_seqs

        logger.info(
            f"Processed {seq_count} sequences: "
            f"{passed_count} sequences retrieved with length in [{self.min_seq_len}, {self.max_seq_len}]."
        )
        logger.info(f"Filtered output saved to: {self.output_path}")

        self.processed_seqs_num = seq_count
        self.num_filtered_seqs = passed_count
