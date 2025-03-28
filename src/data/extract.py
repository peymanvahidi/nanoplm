import gzip
from tqdm import tqdm
from .base import BaseProcessor, logger

class Extractor(BaseProcessor):
    """Class for extracting the UniRef50 dataset."""
    
    def extract(self):
        """Extract the gzipped FASTA file if needed."""
        self.create_dirs()
        
        if self.uniref50_fasta.exists():
            logger.info(f"UniRef50 FASTA already extracted at {self.uniref50_fasta}")
            return
        
        logger.info(f"Extracting {self.uniref50_fasta_gz}")
        
        # Get file size for progress bar
        file_size = self.uniref50_fasta_gz.stat().st_size
        
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Extracting FASTA") as pbar:
            with gzip.open(self.uniref50_fasta_gz, 'rb') as f_in:
                with open(self.uniref50_fasta, 'wb') as f_out:
                    while True:
                        chunk = f_in.read(4096)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Extracted UniRef50 to {self.uniref50_fasta}")

if __name__ == "__main__":
    # For direct execution
    extractor = Extractor()
    extractor.extract()
