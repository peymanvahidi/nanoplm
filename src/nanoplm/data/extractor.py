import gzip
from tqdm import tqdm
from pathlib import Path
from typing import Union

from nanoplm.utils import logger


class ExtractionError(Exception):
    """Raised when an extraction operation fails."""
    pass


class Extractor:
    """Class for extracting dataset files."""

    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def extract(self):
        file_size = self.input_path.stat().st_size

        try:
            with gzip.open(self.input_path, "rb") as f_in:
                f_in.read(10)
            with tqdm(
                total=file_size, unit="B", unit_scale=True, desc="Extracting FASTA"
            ) as pbar:
                with gzip.open(self.input_path, "rb") as f_in:
                    with open(self.output_path, "wb") as f_out:
                        while True:
                            chunk = f_in.read(4096)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            pbar.update(len(chunk))
        except gzip.BadGzipFile:
            raise ExtractionError("File is not gzipped")
        except EOFError:
            raise ExtractionError("Corrupted or incomplete gzip file - file ended before the end-of-stream marker was reached")
