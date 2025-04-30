import gzip
import shutil
from tqdm import tqdm
from pathlib import Path

from ..utils import logger

class Extractor():
    """Class for extracting the UniRef50 dataset."""

    def __init__(
        self,
        input_file: Path,
        output_file: Path
    ):
        self.input_file = input_file
        self.output_file = output_file
    
    def extract(self):
        if self.output_file.exists():
            logger.info(f"UniRef50 FASTA already extracted at {self.output_file}")
            return
        
        # Get file size for progress bar
        file_size = self.input_file.stat().st_size
        
        logger.info(f"Extracting UniRef50 FASTA from {self.input_file}")

        try:
            with gzip.open(self.input_file, 'rb') as f_in:
                # Try to read a small chunk to see if it's a valid gzip file
                f_in.read(10)
            # If we get here, it's a valid gzip file, extract it
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Extracting FASTA") as pbar:
                with gzip.open(self.input_file, 'rb') as f_in:
                    with open(self.output_file, 'wb') as f_out:
                        while True:
                            chunk = f_in.read(4096)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            pbar.update(len(chunk))
        except gzip.BadGzipFile:
            # Not a gzip file, just copy it
            logger.info(f"File is not gzipped, copying as-is...")
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Copying FASTA") as pbar:
                with open(self.input_file, 'rb') as f_in:
                    with open(self.output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                pbar.update(file_size)
        
        logger.info(f"Extracted UniRef50 to {self.output_file}")
