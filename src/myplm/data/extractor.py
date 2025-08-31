import gzip
from tqdm import tqdm
from pathlib import Path
from typing import Union

from myplm.utils import logger


class Extractor:
    """Class for extracting dataset files."""

    def __init__(self, input_file: Union[str, Path], output_file: Union[str, Path]):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)

    def extract(self):
        file_size = self.input_file.stat().st_size

        try:
            with gzip.open(self.input_file, "rb") as f_in:
                f_in.read(10)
            with tqdm(
                total=file_size, unit="B", unit_scale=True, desc="Extracting FASTA"
            ) as pbar:
                with gzip.open(self.input_file, "rb") as f_in:
                    with open(self.output_file, "wb") as f_out:
                        while True:
                            chunk = f_in.read(4096)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            pbar.update(len(chunk))
        except gzip.BadGzipFile:
            raise ValueError("File is not gzipped")

        logger.info(f"Extracted dataset to: {self.output_file}")
