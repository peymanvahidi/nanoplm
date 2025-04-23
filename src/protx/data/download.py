import urllib.request
from pathlib import Path
from tqdm import tqdm
from .base import BaseProcessor, logger

class Downloader(BaseProcessor):
    """Class for downloading the UniRef50 dataset."""

    def download(self):
        """Download UniRef50 dataset if it doesn't exist."""
        self.create_dirs(self.raw_data_dir)
        
        if self.uniref50_fasta_gz.exists():
            logger.info(f"UniRef50 already downloaded at {self.uniref50_fasta_gz}")
            return
        
        # Setup for progress bar
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading UniRef50") as pbar:
            def update_progress(block_count, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                    pbar.update(block_size)
            
            urllib.request.urlretrieve(self.uniref50_url, self.uniref50_fasta_gz, reporthook=update_progress)
        
        logger.info(f"Downloaded UniRef50 to {self.uniref50_fasta_gz}")
