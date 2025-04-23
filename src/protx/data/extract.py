import gzip
import shutil
from tqdm import tqdm
from .base import BaseProcessor, logger

class Extractor(BaseProcessor):
    """Class for extracting the UniRef50 dataset."""
    
    def extract(self):
        if self.uniref50_fasta.exists():
            logger.info(f"UniRef50 FASTA already extracted at {self.uniref50_fasta}")
            return
        
        # Get file size for progress bar
        file_size = self.uniref50_fasta_gz.stat().st_size
        
        try:
            with gzip.open(self.uniref50_fasta_gz, 'rb') as f_in:
                # Try to read a small chunk to see if it's a valid gzip file
                f_in.read(10)
            # If we get here, it's a valid gzip file, extract it
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Extracting FASTA") as pbar:
                with gzip.open(self.uniref50_fasta_gz, 'rb') as f_in:
                    with open(self.uniref50_fasta, 'wb') as f_out:
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
                with open(self.uniref50_fasta_gz, 'rb') as f_in:
                    with open(self.uniref50_fasta, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                pbar.update(file_size)
        
        logger.info(f"Extracted UniRef50 to {self.uniref50_fasta}")
