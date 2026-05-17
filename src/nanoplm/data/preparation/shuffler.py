import os
import random
from pathlib import Path
from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.data.preparation._native_ops import shuffle_fasta_native
from nanoplm.utils import logger

_DEFAULT_THREADS = min(os.cpu_count() or 1, 16)


class ShufflingError(RuntimeError):
    pass


class FastaShuffler:
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        seed: int | None = None,
        *,
        do_shuffle: bool = True,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.seed = seed
        self.do_shuffle = do_shuffle

    def shuffle(self) -> None:
        try:
            if not self.do_shuffle:
                self._symlink_through()
                return
            effective_seed = self.seed if self.seed is not None else random.randint(1, 2**31 - 1)
            with atomic_write_path(self.output_path) as tmp:
                shuffle_fasta_native(
                    input_path=self.input_path,
                    output_path=tmp,
                    seed=effective_seed,
                    num_threads=_DEFAULT_THREADS,
                )
        except ShufflingError:
            raise
        except Exception as e:
            raise ShufflingError(str(e)) from e

    def _symlink_through(self) -> None:
        if self.output_path.exists() or self.output_path.is_symlink():
            self.output_path.unlink()
        try:
            with atomic_write_path(self.output_path) as tmp:
                tmp.symlink_to(self.input_path.resolve())
            logger.info(
                f"shuffle: do_shuffle=False — symlinked "
                f"{self.input_path} -> {self.output_path}"
            )
        except OSError:
            import shutil
            with atomic_write_path(self.output_path) as tmp:
                shutil.copy2(self.input_path, tmp)
            logger.warning(
                f"shuffle: do_shuffle=False — symlink unsupported, fell back to copy. "
                f"Content changes to {self.input_path} will not auto-trigger DVC re-run."
            )
