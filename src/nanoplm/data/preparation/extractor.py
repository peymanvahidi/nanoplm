import shutil
import subprocess
from pathlib import Path

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.utils import logger
from nanoplm.utils.common import run_with_heartbeat


class ExtractionError(Exception):
    pass


class _CountingReader:
    """Binary file wrapper that counts bytes read through it.

    isal's IGzipFile only reveals decompressed size after the fact, and the
    compressed-to-decompressed ratio isn't known ahead of time, so progress
    is reported against compressed bytes consumed (known upfront from the
    input file's size) rather than decompression completion.
    """

    def __init__(self, fileobj):
        self._fileobj = fileobj
        self.bytes_read = 0

    def read(self, size=-1):
        chunk = self._fileobj.read(size)
        self.bytes_read += len(chunk)
        return chunk

    def readinto(self, b):
        n = self._fileobj.readinto(b)
        if n:
            self.bytes_read += n
        return n


class Extractor:
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def extract(self) -> None:
        if not str(self.input_path).endswith(".gz"):
            raise ExtractionError(
                f"Input must be .gz; got {self.input_path}"
            )
        if shutil.which("pigz"):
            self._extract_pigz()
        else:
            self._extract_isal()

    def _extract_pigz(self) -> None:
        try:
            with atomic_write_path(self.output_path) as tmp:
                with open(tmp, "wb") as out:
                    run_with_heartbeat(
                        "Extracting FASTA (pigz)",
                        lambda: subprocess.check_call(
                            ["pigz", "-dc", str(self.input_path)],
                            stdout=out,
                        ),
                    )
        except subprocess.CalledProcessError as e:
            raise ExtractionError(f"pigz extraction failed: {e}") from e

    def _extract_isal(self) -> None:
        import isal.igzip as igzip
        from isal.isal_zlib import IsalError

        compressed_size = self.input_path.stat().st_size
        last_percent = -1

        def _copy() -> None:
            nonlocal last_percent
            with atomic_write_path(self.output_path) as tmp:
                with open(self.input_path, "rb") as raw_in:
                    counting_in = _CountingReader(raw_in)
                    with igzip.IGzipFile(fileobj=counting_in, mode="rb") as f_in, \
                         open(tmp, "wb") as f_out:
                        while True:
                            chunk = f_in.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            if compressed_size <= 0:
                                continue
                            percent = min(int(counting_in.bytes_read * 100 / compressed_size), 100)
                            if percent < 100 and percent % 5 != 0:
                                continue
                            if percent == last_percent:
                                continue
                            last_percent = percent
                            logger.info(
                                "Extract progress: %d%% of compressed input read (%d/%d bytes)",
                                percent,
                                counting_in.bytes_read,
                                compressed_size,
                            )

        try:
            run_with_heartbeat("Extracting FASTA (isal)", _copy)
        except (IsalError, EOFError, OSError) as e:
            raise ExtractionError(f"isal extraction failed: {e}") from e
