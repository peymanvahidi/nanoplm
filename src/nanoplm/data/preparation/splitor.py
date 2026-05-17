from pathlib import Path
from typing import Union

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.utils import logger
from nanoplm.utils.common import run_with_heartbeat
from nanoplm.data.preparation._native_ops import (
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

        logger.info(f"Creating splits with val ratio {self.val_ratio}")

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

        # Train and val renames are independent: if the val rename succeeds
        # and the train rename then fails, val is left as an orphaned-but-
        # correct file. Acceptable; cross-file atomicity would need staging.
        with atomic_write_path(self.train_file) as tmp_train, \
             atomic_write_path(self.val_file) as tmp_val:
            result = run_with_heartbeat(
                "Splitting FASTA into train/val (native backend)",
                lambda: split_fasta_native(
                    input_path=self.input_file,
                    train_path=tmp_train,
                    val_path=tmp_val,
                    val_ratio=self.val_ratio,
                    progress_cb=progress_cb,
                ),
            )
        train_size = result.train_size
        val_size = result.val_size
        logger.info(
            f"Sequences: {train_size + val_size}, Train: {train_size}, Val: {val_size}"
        )
        return (train_size, val_size)
