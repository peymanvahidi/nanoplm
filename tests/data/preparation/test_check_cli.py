# tests/data/preparation/test_check_cli.py
from pathlib import Path
from click.testing import CliRunner
from nanoplm.cli.data import data


def test_check_clean_fasta_exits_zero(tmp_path):
    src = tmp_path / "clean.fasta"
    src.write_text(">a\nMAGV\n")
    out_dir = tmp_path / "validation"
    runner = CliRunner()
    result = runner.invoke(
        data,
        ["check", "-i", str(src), "-o", str(out_dir), "--drop-invalid", "false"],
    )
    assert result.exit_code == 0


def test_check_invalid_fasta_strict_exits_nonzero(tmp_path):
    src = tmp_path / "dirty.fasta"
    src.write_text(">bad\nMAG!\n")
    out_dir = tmp_path / "validation"
    runner = CliRunner()
    result = runner.invoke(
        data,
        ["check", "-i", str(src), "-o", str(out_dir), "--drop-invalid", "false"],
    )
    assert result.exit_code != 0
    assert "invalid" in result.output.lower() or "fix" in result.output.lower()


def test_check_invalid_fasta_drop_exits_zero(tmp_path):
    src = tmp_path / "dirty.fasta"
    src.write_text(">bad\nMAG!\n>good\nMAGV\n")
    out_dir = tmp_path / "validation"
    runner = CliRunner()
    result = runner.invoke(
        data,
        ["check", "-i", str(src), "-o", str(out_dir), "--drop-invalid", "true"],
    )
    assert result.exit_code == 0
    assert (out_dir / "invalid_ids.txt").read_text().strip() == "bad"


def test_a_mid_sequence_stop_codon_stops_or_is_dropped(tmp_path):
    """A '*' inside a sequence is invalid like any other bad residue; a trailing one is not."""
    src = tmp_path / "prodigal.fa"
    src.write_text(">ok\nMAGV*\n>bad\nMA*GV*\n")
    runner = CliRunner()
    strict = runner.invoke(
        data, ["check", "-i", str(src), "-o", str(tmp_path / "v1"), "--drop-invalid", "false"])
    assert strict.exit_code != 0
    drop = runner.invoke(
        data, ["check", "-i", str(src), "-o", str(tmp_path / "v2"), "--drop-invalid", "true"])
    assert drop.exit_code == 0
    assert (tmp_path / "v2" / "invalid_ids.txt").read_text().strip() == "bad"
