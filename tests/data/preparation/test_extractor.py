import gzip
import shutil
from pathlib import Path
import pytest
from nanoplm.data.preparation.extractor import Extractor, ExtractionError


def _make_gz(tmp_path, payload=b">seq1\nMAGV\n>seq2\nVVAGM\n"):
    src = tmp_path / "in.fasta.gz"
    with gzip.open(src, "wb") as f:
        f.write(payload)
    return src, payload


def test_extracts_with_isal_when_pigz_absent(tmp_path, monkeypatch):
    """When pigz isn't on PATH, falls back to isal and produces identical output."""
    monkeypatch.setenv("PATH", str(tmp_path))  # empty PATH → no pigz
    src, payload = _make_gz(tmp_path)
    dst = tmp_path / "out.fasta"
    Extractor(input_path=src, output_path=dst).extract()
    assert dst.read_bytes() == payload


def test_extracts_with_pigz_when_present(tmp_path, monkeypatch):
    """When pigz is on PATH, uses pigz (verify by smoke — both produce identical output)."""
    if not shutil.which("pigz"):
        pytest.skip("pigz not installed in test environment")
    src, payload = _make_gz(tmp_path)
    dst = tmp_path / "out.fasta"
    Extractor(input_path=src, output_path=dst).extract()
    assert dst.read_bytes() == payload


def test_raises_on_corrupt_gz(tmp_path):
    src = tmp_path / "corrupt.fasta.gz"
    src.write_bytes(b"not a real gzip file")
    dst = tmp_path / "out.fasta"
    with pytest.raises(ExtractionError):
        Extractor(input_path=src, output_path=dst).extract()
