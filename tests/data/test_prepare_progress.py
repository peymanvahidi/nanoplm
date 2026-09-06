# tests/data/test_prepare_progress.py
from pathlib import Path
from nanoplm.data.preparation.shuffler import FastaShuffler


def _write_fasta(path: Path, num_records: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_records):
            f.write(f">seq{i}\n")
            f.write("ACDEFGHIKLMNPQRSTVWY" * 3 + "\n")


def test_shuffle_logs_progress(tmp_path, monkeypatch):
    src = tmp_path / "in.fasta"
    dst = tmp_path / "out.fasta"
    _write_fasta(src, 60)

    # Spy on the nanoplm logger (caplog doesn't work because propagate=False)
    seen = []
    from nanoplm.utils.logger import logger as _nplm_logger
    monkeypatch.setattr(
        _nplm_logger,
        "info",
        lambda msg, *a, **k: seen.append(msg % a if a else msg),
    )

    FastaShuffler(input_path=src, output_path=dst, seed=42).shuffle()

    assert any("Shuffle progress" in msg for msg in seen)


def test_shuffle_preserves_all_sequences(tmp_path):
    src = tmp_path / "in.fasta"
    dst = tmp_path / "out.fasta"
    _write_fasta(src, 60)

    FastaShuffler(input_path=src, output_path=dst, seed=42).shuffle()

    assert dst.exists()
    assert set(dst.read_text().splitlines()) == set(src.read_text().splitlines())


def test_extract_isal_logs_progress(tmp_path, monkeypatch):
    import gzip
    from nanoplm.data.preparation.extractor import Extractor

    monkeypatch.setenv("PATH", str(tmp_path))  # empty PATH → no pigz

    payload = bytearray()
    for i in range(20000):
        payload += f">seq{i}\n".encode("utf-8")
        payload += b"ACDEFGHIKLMNPQRSTVWY\n"
    payload = bytes(payload)

    src = tmp_path / "in.fasta.gz"
    with gzip.open(src, "wb") as f:
        f.write(payload)
    dst = tmp_path / "out.fasta"

    seen = []
    from nanoplm.utils.logger import logger as _nplm_logger
    monkeypatch.setattr(
        _nplm_logger,
        "info",
        lambda msg, *a, **k: seen.append(msg % a if a else msg),
    )

    Extractor(input_path=src, output_path=dst).extract()

    assert dst.read_bytes() == payload
    assert any("Extract progress" in msg for msg in seen)


def test_extract_pigz_logs_heartbeat(tmp_path, monkeypatch):
    import gzip
    import shutil
    import pytest
    from nanoplm.data.preparation.extractor import Extractor

    if not shutil.which("pigz"):
        pytest.skip("pigz not installed in test environment")

    payload = b">seq1\nMAGV\n>seq2\nVVAGM\n"
    src = tmp_path / "in.fasta.gz"
    with gzip.open(src, "wb") as f:
        f.write(payload)
    dst = tmp_path / "out.fasta"

    seen = []
    from nanoplm.utils.logger import logger as _nplm_logger
    monkeypatch.setattr(
        _nplm_logger,
        "info",
        lambda msg, *a, **k: seen.append(msg % a if a else msg),
    )

    Extractor(input_path=src, output_path=dst).extract()

    assert dst.read_bytes() == payload
    assert any("Extracting FASTA (pigz) started" in msg for msg in seen)
    assert any("completed in" in msg for msg in seen)


def test_from_yaml_dry_run_does_not_init_dvc(tmp_path, monkeypatch):
    import re
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    from nanoplm.cli.templates import load_template

    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">a\nMAGVAGM\n")

    template = load_template("data_pretrain")
    template = template.replace("local_fasta: null", f"local_fasta: {fasta}")
    template = re.sub(r"^\s*url:.*$", "  url: null", template, count=1, flags=re.MULTILINE)

    cfg = tmp_path / "data.yaml"
    cfg.write_text(template)

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg), "--dry-run"])

    assert not (tmp_path / ".dvc").exists()
    assert result.exit_code != 0
    assert "dvc init" in result.output.lower() or "not initialized" in result.output.lower()


def test_from_yaml_regenerated_message_appears_once(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    from nanoplm.cli.templates import load_template

    cfg = tmp_path / "data.yaml"
    cfg.write_text(load_template("data_pretrain"))

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})(),
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg), "--dry-run"])

    assert result.output.count("Regenerated") == 1
    assert result.exit_code == 0, result.output
