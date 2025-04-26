import urllib.request
from pathlib import Path
from tqdm import tqdm

from .base import BaseProcessor, logger

class Downloader(BaseProcessor):
    """Class for downloading the UniRef50 dataset."""

    def __init__(
        self,
        url: str,
        output_file: str,
        no_extra_files: bool = False
    ):
        self.no_extra_files = no_extra_files
        self.url = url
        self.output_file = output_file

    def download(self):
        """Download UniRef50 dataset if it doesn't exist."""
        self.create_dirs(self.output_file)
        
        if self.output_file.exists():
            logger.info(f"{self.output_file} already exists")
            return
        
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=f"Downloading {self.output_file}") as pbar:
            def update_progress(block_count, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                    pbar.update(block_size)
            
            urllib.request.urlretrieve(self.url, self.output_file, reporthook=update_progress)
        
        logger.info(f"Downloaded {self.output_file}")
