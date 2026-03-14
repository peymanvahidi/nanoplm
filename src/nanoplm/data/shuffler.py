import os
import mmap
import shutil
import subprocess
import random
import time
from array import array
from pathlib import Path
from typing import Union, Optional, Literal

import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from nanoplm.utils import logger, get_caller_dir
from nanoplm.utils.common import run_with_heartbeat
from nanoplm.data.native_fasta_ops import (
    NativeFastaOpsError,
    is_native_fasta_ops_available,
    shuffle_fasta_native,
)

Backend = Literal["auto", "seqkit", "fast", "biopython"]
DEFAULT_SHUFFLE_THREADS = min(os.cpu_count() or 1, 16)


class ShufflingError(RuntimeError):
    """Raised when a requested backend cannot be used (e.g., missing dependency)."""

    pass


class FastaShuffler:
    """
    FASTA shuffler with multiple backends:
      - 'auto' (default): use seqkit if available, else fast Python backend
      - 'biopython': portable, no external deps
      - 'seqkit' (faster): shell out to 'seqkit shuffle' for fast, parallel, external-memory shuffle
      - 'fast': offset-index + mmap write path (faster than BioPython fallback)
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        seed: Optional[int] = None,
        backend: Backend = "auto",
        threads: int = DEFAULT_SHUFFLE_THREADS,
        two_pass: bool = True,
        keep_temp: bool = False,
        use_native_fast: bool = True,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.seed = random.randint(1, 1000) if not seed else seed
        self.backend = backend
        self.threads = max(1, int(threads))
        self.two_pass = bool(two_pass)
        self.keep_temp = bool(keep_temp)
        self.use_native_fast = bool(use_native_fast)
        self.caller_dir = get_caller_dir()

    def shuffle(self):
        resolved_input = self._resolve_path(self.input_path)
        if not resolved_input.exists():
            raise ShufflingError(f"Input file not found: {resolved_input}")

        backend = self._choose_backend()
        logger.info(f"Using backend: {backend}")

        if backend == "seqkit":
            return run_with_heartbeat(
                "Shuffling FASTA records (seqkit backend)",
                self._shuffle_with_seqkit,
            )
        if backend == "fast":
            return run_with_heartbeat(
                "Shuffling FASTA records (fast backend)",
                self._shuffle_with_fast_offsets,
            )
        else:
            return run_with_heartbeat(
                "Shuffling FASTA records (biopython backend)",
                self._shuffle_with_biopython,
            )

    def _choose_backend(self) -> Backend:
        if self.backend == "auto":
            if shutil.which("seqkit"):
                return "seqkit"
            return "fast"

        if self.backend == "seqkit":
            if shutil.which("seqkit"):
                return "seqkit"
            else:
                raise ShufflingError(
                    "`seqkit` is not available. Install it first, or use `fast`/`biopython` backend."
                )
        elif self.backend == "fast":
            return "fast"
        elif self.backend == "biopython":
            return "biopython"
        else:
            raise ShufflingError(f"Invalid backend: {self.backend}")

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.caller_dir / path

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        value = float(num_bytes)
        for unit in units:
            if value < 1024.0 or unit == units[-1]:
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024.0
        return f"{num_bytes} B"

    def _shuffle_with_seqkit(self):
        cmd = [
            "seqkit",
            "shuffle",
            "--threads",
            str(self.threads),
        ]
        if self.two_pass:
            cmd.append("--two-pass")
        if self.keep_temp:
            cmd.append("--keep-temp")
        if self.seed:
            cmd += ["--rand-seed", str(self.seed)]

        input_path = self._resolve_path(self.input_path)
        output_path = self._resolve_path(self.output_path)
        cmd += [str(input_path), "-o", str(output_path)]

        logger.info(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            raise ShufflingError(f"seqkit shuffle failed: {e}")

    def _shuffle_with_fast_offsets(self):
        if not self.use_native_fast:
            logger.info(
                "Fast shuffle configured for legacy pipeline. Using Python fast backend."
            )
            return self._shuffle_with_fast_offsets_python()

        native_available, native_error = is_native_fasta_ops_available()
        if native_available:
            try:
                return self._shuffle_with_fast_native()
            except NativeFastaOpsError as exc:
                logger.warning(
                    "Native fast shuffle failed (%s). Falling back to Python fast shuffle.",
                    exc,
                )
        else:
            logger.warning(
                "Native fast shuffle unavailable (%s). Falling back to Python fast shuffle.",
                native_error,
            )

        return self._shuffle_with_fast_offsets_python()

    def _shuffle_with_fast_native(self):
        input_path = self._resolve_path(self.input_path)
        output_path = self._resolve_path(self.output_path)
        phase_names = {
            1: "indexing FASTA records",
            3: "copying shuffled records",
        }
        last_phase_progress: dict[int, int] = {}

        def progress_cb(
            phase: int,
            progress: float,
            completed: int,
            total: int,
            aux: int,
        ) -> None:
            percent = int(progress * 100.0)
            if percent < 100 and percent % 5 != 0:
                return
            if last_phase_progress.get(phase) == percent:
                return
            last_phase_progress[phase] = percent
            phase_name = phase_names.get(phase, f"phase {phase}")
            if phase == 1:
                logger.info(
                    "Fast shuffle progress: %s %d%% (%s / %s scanned, %s records found)",
                    phase_name,
                    percent,
                    self._format_bytes(completed),
                    self._format_bytes(total),
                    f"{aux:,}",
                )
            else:
                logger.info(
                    "Fast shuffle progress: %s %d%% (%s / %s records)",
                    phase_name,
                    percent,
                    f"{completed:,}",
                    f"{total:,}",
                )

        logger.info(
            "Fast shuffle using native backend with %d thread(s).",
            self.threads,
        )
        result = shuffle_fasta_native(
            input_path=input_path,
            output_path=output_path,
            seed=self.seed,
            num_threads=self.threads,
            progress_cb=progress_cb,
        )
        logger.info(
            f"Successfully shuffled {result.record_count} sequences\nFasta file saved to: {output_path}"
        )

    def _shuffle_with_fast_offsets_python(self):
        input_path = self._resolve_path(self.input_path)
        output_path = self._resolve_path(self.output_path)
        input_size = input_path.stat().st_size

        logger.info("Fast shuffle phase 1/2: indexing FASTA record offsets...")
        starts = array("Q")
        next_log_at = time.perf_counter() + 10.0
        with open(input_path, "rb") as f:
            while True:
                line_start = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    starts.append(line_start)
                now = time.perf_counter()
                if now >= next_log_at:
                    scanned = f.tell()
                    pct = (100.0 * scanned / input_size) if input_size > 0 else 0.0
                    logger.info(
                        "Fast shuffle indexing progress: %.1f%% scanned (%s / %s, %s records found)",
                        pct,
                        self._format_bytes(scanned),
                        self._format_bytes(input_size),
                        f"{len(starts):,}",
                    )
                    next_log_at = now + 10.0
            file_size = f.tell()

        num_records = len(starts)
        if num_records == 0:
            raise ShufflingError("No sequences found in input file")

        starts.append(file_size)
        logger.info("Fast shuffle indexing done: %d records", num_records)

        rng = np.random.default_rng(self.seed)
        if num_records <= np.iinfo(np.uint32).max:
            order = np.arange(num_records, dtype=np.uint32)
        else:
            order = np.arange(num_records, dtype=np.int64)
        rng.shuffle(order)

        logger.info("Fast shuffle phase 2/2: writing shuffled FASTA...")
        with open(input_path, "rb") as in_f, open(output_path, "wb") as out_f:
            mm = mmap.mmap(in_f.fileno(), length=0, access=mmap.ACCESS_READ)
            try:
                with tqdm(total=num_records, desc="Writing shuffled sequences") as pbar:
                    written = 0
                    last_percent = -1
                    for idx in order:
                        i = int(idx)
                        out_f.write(mm[starts[i] : starts[i + 1]])
                        pbar.update(1)
                        written += 1
                        percent = int((written * 100) / num_records) if num_records > 0 else 100
                        if percent >= 100 or (percent % 5 == 0 and percent != last_percent):
                            last_percent = percent
                            logger.info(
                                "Fast shuffle write progress: %d%% (%d/%d records)",
                                percent,
                                written,
                                num_records,
                            )
            finally:
                mm.close()

        logger.info(
            f"Successfully shuffled {num_records} sequences\nFasta file saved to: {output_path}"
        )

    def _shuffle_with_biopython(self):
        input_path = self._resolve_path(self.input_path)
        output_path = self._resolve_path(self.output_path)

        try:
            record_dict = SeqIO.index(str(input_path), "fasta")
            sequence_ids = list(record_dict.keys())
            logger.debug(f"Indexed {len(sequence_ids)} sequences")
        except Exception as e:
            raise ShufflingError(f"Error creating BioPython index: {e}")

        if not sequence_ids:
            raise ShufflingError("No sequences found in input file")

        logger.debug(f"Shuffling {len(sequence_ids)} sequence IDs...")
        random.shuffle(sequence_ids)

        logger.debug(f"Writing shuffled sequences to {output_path}...")
        try:
            with open(output_path, "w") as output_handle:
                with tqdm(total=len(sequence_ids), desc="Writing sequences") as pbar:
                    for seq_id in sequence_ids:
                        record = record_dict[seq_id]
                        SeqIO.write(record, output_handle, "fasta")
                        pbar.update(1)
        finally:
            record_dict.close()

        logger.info(
            f"Successfully shuffled {len(sequence_ids)} sequences\nFasta file saved to: {output_path}"
        )
