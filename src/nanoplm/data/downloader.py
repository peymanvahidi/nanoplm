import ssl
import certifi
import urllib.request
from pathlib import Path
from tqdm import tqdm

from nanoplm.utils import create_dirs, logger


class DownloadError(Exception):
    """Raised when a download operation fails."""

    pass


class Downloader:
    """Class for downloading the UniRef50 dataset."""

    def __init__(self, url: str, output_path: str | Path):
        self.url = url
        self.output_path = Path(output_path)

    def download(self):
        """Download UniRef50 dataset if it doesn't exist."""
        create_dirs(self.output_path)

        ssl_context = self._build_ssl_context()

        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context)
        )
        request = urllib.request.Request(self.url)

        with opener.open(request) as response:
            total_size_header = response.headers.get("Content-Length")
            total_size = int(total_size_header) if total_size_header else 0

            try:
                with tqdm(
                    total=total_size if total_size > 0 else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {Path(self.url).name}",
                ) as pbar:
                    with open(self.output_path, "wb") as out_file:
                        while True:
                            chunk = response.read(1024 * 64)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            pbar.update(len(chunk))
            except Exception as e:
                raise DownloadError(f"Error downloading dataset: {e}")

    def _build_ssl_context(self) -> ssl.SSLContext:
        """Create a secure SSL context, preferring the certifi CA bundle when available."""
        context = ssl.create_default_context()

        # Prefer certifi bundle to mitigate SSL issues on some systems
        try:
            context.load_verify_locations(certifi.where())
            logger.debug("Using certifi CA bundle for SSL verification")
        except Exception:
            # Fall back to system trust store
            logger.debug("Using system CA store for SSL verification")

        return context
