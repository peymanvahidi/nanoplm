import urllib.request
from tqdm import tqdm
from .base import BaseProcessor, logger

class Downloader(BaseProcessor):
    """Class for downloading the UniRef50 dataset."""
    
    def download(self):
        """Download UniRef50 dataset if it doesn't exist."""
        self.create_dirs()
        
        logger.info(f"Checking if {self.uniref50_fasta_gz} exists")
        if self.uniref50_fasta_gz.exists():
            logger.info(f"UniRef50 already downloaded at {self.uniref50_fasta_gz}")
            return
        
        logger.info(f"File not found, downloading UniRef50...")
        
        # Setup for progress bar
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading UniRef50") as pbar:
            def update_progress(block_count, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                    pbar.update(block_size)
            
            urllib.request.urlretrieve(self.uniref50_url, self.uniref50_fasta_gz, reporthook=update_progress)
        
        logger.info(f"Downloaded UniRef50 to {self.uniref50_fasta_gz}")

if __name__ == "__main__":
    # For direct execution
    downloader = Downloader()
    downloader.download() 