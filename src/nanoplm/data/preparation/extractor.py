import shutil
import subprocess
from pathlib import Path


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
            with open(self.output_path, "wb") as out:
                subprocess.check_call(
                    ["pigz", "-dc", str(self.input_path)],
                    stdout=out,
                )
        except subprocess.CalledProcessError as e:
            self.output_path.unlink(missing_ok=True)
            raise ExtractionError(f"pigz extraction failed: {e}") from e

    def _extract_isal(self) -> None:
        import isal.igzip as igzip
        from isal.isal_zlib import IsalError
        try:
            with igzip.open(self.input_path, "rb") as f_in, \
                 open(self.output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out, length=8 * 1024 * 1024)
        except (IsalError, EOFError, OSError) as e:
            self.output_path.unlink(missing_ok=True)
            raise ExtractionError(f"isal extraction failed: {e}") from e
