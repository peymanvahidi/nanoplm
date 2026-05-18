# tests/data/preparation/test_checker.py
from pathlib import Path
import json
import pytest
from nanoplm.data.preparation.checker import Checker, CheckResult


def test_clean_fasta_passes(tmp_path):
    src = tmp_path / "clean.fasta"
    src.write_text(">a\nMAGV\n>b\nVVAGM\n")
    out_dir = tmp_path / "validation"
    result = Checker(input_path=src, output_dir=out_dir).check()
    assert isinstance(result, CheckResult)
    assert result.total_sequences == 2
    assert result.invalid_count == 0
    assert (out_dir / "validation.json").exists()
    assert (out_dir / "invalid.fasta").exists()
    assert (out_dir / "invalid_ids.txt").exists()
    assert (out_dir / "invalid_ids.txt").read_text() == ""


def test_invalid_fasta_reports_problem_sequences(tmp_path):
    src = tmp_path / "dirty.fasta"
    src.write_text(">good\nMAGV\n>bad\nMAG!\n>empty\n\n")
    out_dir = tmp_path / "validation"
    result = Checker(input_path=src, output_dir=out_dir).check()
    assert result.invalid_count == 2
    invalid_ids = (out_dir / "invalid_ids.txt").read_text().strip().splitlines()
    assert set(invalid_ids) == {"bad", "empty"}


def test_drop_invalid_false_leaves_invalid_ids_empty(tmp_path):
    """When drop_invalid=False, invalid_ids.txt is empty (filter excludes nothing)."""
    src = tmp_path / "dirty.fasta"
    src.write_text(">good\nMAGV\n>bad\nMAG!\n")
    out_dir = tmp_path / "validation"
    Checker(input_path=src, output_dir=out_dir, drop_invalid=False).check()
    assert (out_dir / "invalid_ids.txt").read_text() == ""
