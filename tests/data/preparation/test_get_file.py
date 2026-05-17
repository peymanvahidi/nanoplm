# tests/data/preparation/test_get_file.py
from pathlib import Path
from click.testing import CliRunner
from nanoplm.cli.data import data


def test_get_file_with_local_gz_creates_symlink(tmp_path):
    src = tmp_path / "local.fasta.gz"
    src.write_bytes(b"fake gzip bytes")
    out = tmp_path / "out.fasta.gz"
    runner = CliRunner()
    result = runner.invoke(
        data,
        ["get-file", "--local-fasta", str(src), "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.is_symlink() or out.read_bytes() == src.read_bytes()


def test_get_file_with_local_fasta_creates_link(tmp_path):
    src = tmp_path / "local.fasta"
    src.write_text(">a\nMAGV\n")
    out = tmp_path / "out.fasta"
    runner = CliRunner()
    result = runner.invoke(
        data,
        ["get-file", "--local-fasta", str(src), "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_get_file_errors_when_both_url_and_local_fasta(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        data,
        ["get-file", "--url", "https://x.com/a.gz",
         "--local-fasta", str(tmp_path / "x.fasta.gz"),
         "-o", str(tmp_path / "out.gz")],
    )
    assert result.exit_code != 0


def test_get_file_errors_when_neither_set(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        data,
        ["get-file", "-o", str(tmp_path / "out.gz")],
    )
    assert result.exit_code != 0


def test_get_file_errors_when_local_fasta_missing(tmp_path):
    """A typo'd --local-fasta path should be caught with a clear error, not create a dangling symlink."""
    runner = CliRunner()
    out = tmp_path / "out.fasta"
    result = runner.invoke(
        data,
        ["get-file", "--local-fasta", str(tmp_path / "does-not-exist.fasta"), "-o", str(out)],
    )
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower()
    assert not out.exists()
