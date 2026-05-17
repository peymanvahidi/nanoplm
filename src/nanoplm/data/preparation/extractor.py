import shutil
import subprocess
from pathlib import Path

from nanoplm.data.infra.atomic import atomic_write_path


class ExtractionError(Exception):
    pass


class Extractor:
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def extract(self) -> None:
        if not str(self.input_path).endswith(".gz"):
            raise ExtractionError(
                f"Input must be .gz; got {self.input_path}"
            )
        if shutil.which("pigz"):
            self._extract_pigz()
        else:
            self._extract_isal()

    def _extract_pigz(self) -> None:
        try:
            with atomic_write_path(self.output_path) as tmp:
                with open(tmp, "wb") as out:
                    subprocess.check_call(
                        ["pigz", "-dc", str(self.input_path)],
                        stdout=out,
                    )
        except subprocess.CalledProcessError as e:
            raise ExtractionError(f"pigz extraction failed: {e}") from e

    def _extract_isal(self) -> None:
        import isal.igzip as igzip
        from isal.isal_zlib import IsalError
        try:
            with atomic_write_path(self.output_path) as tmp:
                with igzip.open(self.input_path, "rb") as f_in, \
                     open(tmp, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out, length=8 * 1024 * 1024)
        except (IsalError, EOFError, OSError) as e:
            raise ExtractionError(f"isal extraction failed: {e}") from e
