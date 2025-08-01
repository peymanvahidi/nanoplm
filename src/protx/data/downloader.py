import urllib.request
from pathlib import Path
from tqdm import tqdm

from ..utils import create_dirs, logger


class Downloader:
    """Class for downloading the UniRef50 dataset."""

    def __init__(
        self,
        url: str,
        output_file: str
    ):
        self.url = url
        self.output_file = Path(output_file)

    def download(self):
        """Download UniRef50 dataset if it doesn't exist."""
        create_dirs(self.output_file)

        if self.output_file.exists():
            logger.info(f"{self.output_file} already exists")
            return

        with tqdm(
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {self.output_file}"
        ) as pbar:
            def update_progress(block_count, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                    pbar.update(block_size)

            urllib.request.urlretrieve(self.url, self.output_file, reporthook=update_progress)
        
        file_size = Path(self.output_file).stat().st_size
        logger.info(f"Downloaded {self.output_file} ({file_size / (1024*1024):.2f} MB)")
