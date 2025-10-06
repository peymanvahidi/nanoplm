import os
import shutil
import subprocess
import random
from pathlib import Path
from typing import Union, Optional, Literal

from Bio import SeqIO
from tqdm import tqdm
from nanoplm.utils import logger, get_caller_dir

Backend = Literal["seqkit", "biopython"]


class ShufflingError(RuntimeError):
    """Raised when a requested backend cannot be used (e.g., missing dependency)."""

    pass


class FastaShuffler:
    """
    FASTA shuffler with two backends:
      - 'biopython': portable, no external deps
      - 'seqkit' (faster): shell out to 'seqkit shuffle' for fast, parallel, external-memory shuffle
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        seed: Optional[int] = None,
        backend: Backend = "biopython",
        threads: int = os.cpu_count() or 1,
        two_pass: bool = True,
        keep_temp: bool = False,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.seed = random.randint(1, 1000) if not seed else seed
        self.backend = backend
        self.threads = max(1, int(threads))
        self.two_pass = bool(two_pass)
        self.keep_temp = bool(keep_temp)
        self.caller_dir = get_caller_dir()

    def shuffle(self):
        if not self.input_path.exists():
            raise ShufflingError(f"Input file not found: {self.input_path}")

        backend = self._choose_backend()
        logger.info(f"Using backend: {backend}")

        if backend == "seqkit":
            return self._shuffle_with_seqkit()
        else:
            return self._shuffle_with_biopython()

    def _choose_backend(self) -> Backend:
        if self.backend == "seqkit":
            if shutil.which("seqkit"):
                return "seqkit"
            else:
                raise ShufflingError(
                    "`seqkit` is not available. Install it first, or use `biopython` backend."
                )
        elif self.backend == "biopython":
            return "biopython"
        else:
            raise ShufflingError(f"Invalid backend: {self.backend}")

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

        # make them absolute based on caller directory
        self.input_path = self.caller_dir / self.input_path
        self.output_path = self.caller_dir / self.output_path
        cmd += [str(self.input_path), "-o", str(self.output_path)]

        logger.info(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            raise ShufflingError(f"seqkit shuffle failed: {e}")

    def _shuffle_with_biopython(self):
        try:
            record_dict = SeqIO.index(str(self.input_path), "fasta")
            sequence_ids = list(record_dict.keys())
            logger.debug(f"Indexed {len(sequence_ids)} sequences")
        except Exception as e:
            raise ShufflingError(f"Error creating BioPython index: {e}")

        if not sequence_ids:
            raise ShufflingError("No sequences found in input file")

        logger.debug(f"Shuffling {len(sequence_ids)} sequence IDs...")
        random.shuffle(sequence_ids)

        logger.debug(f"Writing shuffled sequences to {self.output_path}...")
        try:
            with open(self.output_path, "w") as output_handle:
                with tqdm(total=len(sequence_ids), desc="Writing sequences") as pbar:
                    for seq_id in sequence_ids:
                        record = record_dict[seq_id]
                        SeqIO.write(record, output_handle, "fasta")
                        pbar.update(1)
        finally:
            record_dict.close()

        logger.info(
            f"Successfully shuffled {len(sequence_ids)} sequences\nFasta file saved to: {self.output_path}"
        )
