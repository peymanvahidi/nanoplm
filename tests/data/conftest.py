# tests/data/conftest.py
import pytest
from pathlib import Path


@pytest.fixture
def tiny_fasta(tmp_path):
    p = tmp_path / "tiny.fasta"
    p.write_text(">a\nMAGVAGM\n>b\nVVAGM\n>c\nMMAA\n")
    return p


@pytest.fixture
def malformed_fasta(tmp_path):
    p = tmp_path / "bad.fasta"
    p.write_text(">good\nMAGV\n>empty\n\n>illegal\nMAG!\n>good2\nVVA\n")
    return p
