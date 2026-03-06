from pathlib import Path
from typing import Union

from nanoplm.utils import create_dirs, logger
from nanoplm.utils.common import run_with_heartbeat
from nanoplm.data.native_fasta_ops import (
    NativeFastaOpsError,
    filter_fasta_native,
    is_native_fasta_ops_available,
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
        if not self.input_path.exists():
            raise FilterError(f"Input FASTA not found: {self.input_path}")

        native_available, native_error = is_native_fasta_ops_available()
        if native_available:
            try:
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
                        skip_n=self.skip_n,
                        progress_cb=progress_cb,
                    ),
                )
                seq_count = result.processed_seqs
                passed_count = result.passed_seqs
                logger.info("Filtering used native FASTA backend.")
            except NativeFastaOpsError as exc:
                logger.warning(
                    "Native FASTA filter failed (%s). Falling back to Python streaming filter.",
                    exc,
                )
                seq_count, passed_count = run_with_heartbeat(
                    "Filtering FASTA records (python backend)",
                    self._filter_python_streaming,
                )
        else:
            logger.warning(
                "Native FASTA filter unavailable (%s). Using Python streaming filter.",
                native_error,
            )
            seq_count, passed_count = run_with_heartbeat(
                "Filtering FASTA records (python backend)",
                self._filter_python_streaming,
            )

        logger.info(
            f"Processed {seq_count} sequences from the considered set (after skipping {self.skip_n}): "
            f"{passed_count} sequences retrieved with length in [{self.min_seq_len}, {self.max_seq_len}]."
        )
        logger.info(f"Filtered output saved to: {self.output_path}")

        self.processed_seqs_num = seq_count
        self.num_filtered_seqs = passed_count

    def _filter_python_streaming(self) -> tuple[int, int]:
        logger.info(f"Processing sequences sequentially from {self.input_path}")

        processed = 0
        passed = 0
        skipped = 0
        stop_now = False
        input_size = self.input_path.stat().st_size
        bytes_read = 0
        last_percent = -1

        header: str | None = None
        seq_parts: list[str] = []

        def flush_record() -> bool:
            nonlocal processed, passed, skipped
            if header is None:
                return False

            if skipped < self.skip_n:
                skipped += 1
                return False

            processed += 1
            if self.seqs_num != -1 and passed >= self.seqs_num:
                return True

            seq = "".join(seq_parts)
            seq_len = len(seq)
            if self.min_seq_len <= seq_len <= self.max_seq_len:
                output_handle.write(">")
                output_handle.write(header)
                output_handle.write("\n")
                output_handle.write(seq)
                output_handle.write("\n")
                passed += 1
            return False

        with open(self.input_path, "r", encoding="utf-8") as input_handle, open(
            self.output_path, "w", encoding="utf-8"
        ) as output_handle:
            for line in input_handle:
                bytes_read += len(line)
                percent = int((bytes_read * 100) / input_size) if input_size > 0 else 100
                if percent >= 100 or (percent % 5 == 0 and percent != last_percent):
                    last_percent = percent
                    logger.info(
                        "Filter progress (python backend): %d%% processed=%d passed=%d",
                        percent,
                        processed,
                        passed,
                    )
                if line.startswith(">"):
                    stop_now = flush_record()
                    if stop_now:
                        break
                    header = line[1:].strip()
                    seq_parts = []
                elif header is not None:
                    seq_line = "".join(line.split())
                    if seq_line:
                        seq_parts.append(seq_line)

            if not stop_now and header is not None:
                flush_record()

        return processed, passed
