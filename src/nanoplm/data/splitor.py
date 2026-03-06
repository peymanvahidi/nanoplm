from pathlib import Path
from typing import Union

from nanoplm.utils import logger, create_dirs
from nanoplm.utils.common import run_with_heartbeat
from nanoplm.data.native_fasta_ops import (
    NativeFastaOpsError,
    is_native_fasta_ops_available,
    split_fasta_native,
)


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

        if self.val_ratio < 0.0 or self.val_ratio > 1.0:
            raise SplitError("val_ratio must be in [0.0, 1.0]")

        create_dirs(self.train_file.parent)
        create_dirs(self.val_file.parent)

        logger.info(f"Creating splits with val ratio {self.val_ratio}")

        native_available, native_error = is_native_fasta_ops_available()
        if native_available:
            try:
                phase_names = {
                    1: "counting records",
                    2: "writing splits",
                }
                last_progress: dict[int, int] = {}

                def progress_cb(phase: int, progress: float, completed: int, total: int) -> None:
                    percent = int(progress * 100.0)
                    if percent < 100 and percent % 5 != 0:
                        return
                    if last_progress.get(phase) == percent:
                        return
                    last_progress[phase] = percent
                    logger.info(
                        "Split progress: %s %d%% (%d/%d bytes)",
                        phase_names.get(phase, f"phase {phase}"),
                        percent,
                        completed,
                        total,
                    )

                result = run_with_heartbeat(
                    "Splitting FASTA into train/val (native backend)",
                    lambda: split_fasta_native(
                        input_path=self.input_file,
                        train_path=self.train_file,
                        val_path=self.val_file,
                        val_ratio=self.val_ratio,
                        progress_cb=progress_cb,
                    ),
                )
                train_size = result.train_size
                val_size = result.val_size
                logger.info("Split used native FASTA backend.")
                logger.info(
                    f"Sequences: {train_size + val_size}, Train: {train_size}, Val: {val_size}"
                )
                return (train_size, val_size)
            except NativeFastaOpsError as exc:
                logger.warning(
                    "Native FASTA split failed (%s). Falling back to Python streaming split.",
                    exc,
                )
        else:
            logger.warning(
                "Native FASTA split unavailable (%s). Using Python streaming split.",
                native_error,
            )

        return run_with_heartbeat(
            "Splitting FASTA into train/val (python backend)",
            self._split_python_streaming,
        )

    def _split_python_streaming(self) -> tuple[int, int]:
        total_sequences = 0
        input_size = self.input_file.stat().st_size
        bytes_read = 0
        last_percent_count = -1
        with open(self.input_file, "r", encoding="utf-8") as input_handle:
            for line in input_handle:
                bytes_read += len(line)
                if line.startswith(">"):
                    total_sequences += 1
                percent = int((bytes_read * 100) / input_size) if input_size > 0 else 100
                if percent >= 100 or (percent % 5 == 0 and percent != last_percent_count):
                    last_percent_count = percent
                    logger.info(
                        "Split progress (python backend): counting %d%%",
                        percent,
                    )

        val_size = int(total_sequences * self.val_ratio)
        train_size = total_sequences - val_size

        logger.info(
            f"Sequences: {total_sequences}, Train: {train_size}, Val: {val_size}"
        )

        current_idx = -1
        bytes_read = 0
        last_percent_write = -1
        with open(self.input_file, "r", encoding="utf-8") as input_handle, open(
            self.train_file, "w", encoding="utf-8"
        ) as train_handle, open(self.val_file, "w", encoding="utf-8") as val_handle:
            out_handle = None
            for line in input_handle:
                bytes_read += len(line)
                if line.startswith(">"):
                    current_idx += 1
                    out_handle = train_handle if current_idx < train_size else val_handle
                if out_handle is not None:
                    out_handle.write(line)
                percent = int((bytes_read * 100) / input_size) if input_size > 0 else 100
                if percent >= 100 or (percent % 5 == 0 and percent != last_percent_write):
                    last_percent_write = percent
                    logger.info(
                        "Split progress (python backend): writing %d%%",
                        percent,
                    )

        return train_size, val_size
