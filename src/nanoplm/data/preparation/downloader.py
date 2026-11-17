import os
import ssl
import certifi
import urllib.error
import urllib.request
from pathlib import Path
from tqdm import tqdm

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.utils import logger


class DownloadError(Exception):
    """Raised when a download operation fails."""

    pass


class Downloader:
    """Class for downloading the UniRef50 dataset."""

    def __init__(self, url: str, output_path: str | Path):
        self.url = url
        self.output_path = Path(output_path)

    @classmethod
    def link_local(cls, src: Path, dst: Path) -> None:
        """
        Make `dst` refer to `src`. Prefers symlink, falls back to hardlink,
        then copy. Logs a warning on the copy path.
        """
        src = Path(src).resolve()
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
            return
        except OSError:
            pass
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
        import shutil
        shutil.copy2(src, dst)
        logger.warning(
            f"get-file: symlink/hardlink both unsupported; copied {src} -> {dst}. "
            f"Content changes to {src} will not auto-trigger DVC re-run."
        )

    def download(self):
        """Download UniRef50 dataset if it doesn't exist."""
        ssl_context = self._build_ssl_context()

        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context)
        )
        request = urllib.request.Request(self.url)

        try:
            response_cm = opener.open(request)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            raise DownloadError(f"{self.url}: {e}") from e

        with response_cm as response:
            total_size_header = response.headers.get("Content-Length")
            total_size = int(total_size_header) if total_size_header else 0

            try:
                with atomic_write_path(self.output_path) as tmp:
                    written = 0
                    with tqdm(
                        total=total_size if total_size > 0 else None,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {Path(self.url).name}",
                    ) as pbar:
                        with open(tmp, "wb") as out_file:
                            while True:
                                chunk = response.read(1024 * 64)
                                if not chunk:
                                    break
                                out_file.write(chunk)
                                written += len(chunk)
                                pbar.update(len(chunk))
                    # Raised inside atomic_write_path so the temp file is removed.
                    if total_size_header and written != total_size:
                        raise DownloadError(
                            f"{self.url}: got {written} of {total_size} bytes; "
                            "the download stopped early, run again"
                        )
            except DownloadError:
                raise
            except Exception as e:
                raise DownloadError(f"{self.url}: error downloading: {e}") from e

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
