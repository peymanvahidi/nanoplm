# tests/data/preparation/test_shuffle_noop.py
from pathlib import Path
from nanoplm.data.preparation.shuffler import FastaShuffler


def test_shuffle_false_creates_symlink(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">a\nMAGV\n>b\nVVAGM\n")
    dst = tmp_path / "out.fasta"
    FastaShuffler(input_path=src, output_path=dst, do_shuffle=False).shuffle()
    assert dst.exists()
    assert dst.is_symlink() or dst.read_text() == src.read_text()


def test_shuffle_true_calls_native(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">a\nMAGV\n>b\nVVAGM\n>c\nMM\n>d\nAA\n")
    dst = tmp_path / "out.fasta"
    FastaShuffler(input_path=src, output_path=dst, do_shuffle=True, seed=24).shuffle()
    assert dst.exists()
    # File should contain same sequences (possibly reordered)
    assert set(dst.read_text().splitlines()) == set(src.read_text().splitlines())
