import shutil
import subprocess
from pathlib import Path

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.utils import logger
from nanoplm.utils.common import run_with_heartbeat

_CHUNK_BYTES = 8 * 1024 * 1024
# zstd --long=31 files need a 2 GB window; decoders refuse more than 128 MB by default
ZSTD_MAX_WINDOW = 1 << 31
ZSTD_MISSING = ("needs the zstandard Python package or the zstd tool, and neither is "
                "installed: run `pip install zstandard`, or put zstd on PATH "
                "(e.g. `apt install zstd`)")


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
    """Decompress a .gz (pigz, else isal) or a .zst (zstd, else the zstandard package)."""

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def extract(self) -> None:
        name = self.input_path.name
        if name.endswith(".gz"):
            if shutil.which("pigz"):
                self._extract_cli("pigz", ["pigz", "-dc"])
            else:
                self._extract_isal()
        elif name.endswith(".zst"):
            if shutil.which("zstd"):
                self._extract_cli("zstd", ["zstd", "-dc", "-q", "--memory=2048MB"])
            else:
                self._extract_zstandard()
        else:
            raise ExtractionError(
                f"Input must be .gz or .zst; got {self.input_path}"
            )

    def _extract_cli(self, tool: str, cmd: list) -> None:
        try:
            with atomic_write_path(self.output_path) as tmp:
                with open(tmp, "wb") as out:
                    run_with_heartbeat(
                        f"Extracting FASTA ({tool})",
                        lambda: subprocess.check_call(
                            [*cmd, str(self.input_path)],
                            stdout=out,
                        ),
                    )
        except subprocess.CalledProcessError as e:
            raise ExtractionError(f"{tool} extraction failed: {e}") from e

    def _extract_isal(self) -> None:
        import isal.igzip as igzip
        from isal.isal_zlib import IsalError

        def chunks(raw):
            with igzip.IGzipFile(fileobj=raw, mode="rb") as f_in:
                while chunk := f_in.read(_CHUNK_BYTES):
                    yield chunk

        self._copy("isal", chunks, (IsalError, EOFError, OSError))

    def _extract_zstandard(self) -> None:
        try:
            import zstandard
        except ImportError:
            raise ExtractionError(f"Extracting {self.input_path} {ZSTD_MISSING}") from None

        def chunks(raw):
            # Frame by frame with decompressobj: stream_reader returns a truncated file
            # without an error, decompressobj leaves eof unset.
            dctx = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW)
            dobj, in_frame = dctx.decompressobj(), False
            while data := raw.read(_CHUNK_BYTES):
                while data:
                    in_frame = True
                    yield dobj.decompress(data)
                    if not dobj.eof:
                        break
                    data = dobj.unused_data
                    dobj, in_frame = dctx.decompressobj(), False
            if in_frame:
                raise EOFError("the file ends in the middle of a zstd frame (truncated?)")

        self._copy("zstandard", chunks, (zstandard.ZstdError, EOFError, OSError))

    def _copy(self, tool: str, chunks, errors: tuple) -> None:
        """Write what ``chunks(reader)`` decompresses, logging progress against the
        compressed bytes read."""
        compressed_size = self.input_path.stat().st_size
        if compressed_size == 0:
            # pigz and zstd refuse an empty file; isal and zstandard would write an empty FASTA
            raise ExtractionError(f"{tool} extraction failed: {self.input_path} is empty "
                                  "(0 bytes)")
        last_percent = -1

        def _copy() -> None:
            nonlocal last_percent
            with atomic_write_path(self.output_path) as tmp:
                with open(self.input_path, "rb") as raw_in, open(tmp, "wb") as f_out:
                    counting_in = _CountingReader(raw_in)
                    for chunk in chunks(counting_in):
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
            run_with_heartbeat(f"Extracting FASTA ({tool})", _copy)
        except errors as e:
            raise ExtractionError(f"{tool} extraction failed: {e}") from e
