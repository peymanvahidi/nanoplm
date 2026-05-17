# tests/data/preparation/test_filter.py
from pathlib import Path
from nanoplm.data.preparation.filterer import Filterer


def test_filter_excludes_ids_from_file(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">a\nMAGV\n>b\nVVAGM\n>c\nMMAA\n")
    excl = tmp_path / "excl.txt"
    excl.write_text("b\n")
    out = tmp_path / "out.fasta"
    Filterer(
        input_path=src,
        output_path=out,
        min_seq_len=1,
        max_seq_len=100,
        seqs_num=-1,
        exclude_ids=excl,
    ).filter()
    content = out.read_text()
    assert ">a" in content
    assert ">b" not in content
    assert ">c" in content


def test_filter_with_empty_exclude_file_is_noop(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">a\nMAGV\n>b\nVVAGM\n")
    excl = tmp_path / "excl.txt"
    excl.write_text("")
    out = tmp_path / "out.fasta"
    Filterer(
        input_path=src,
        output_path=out,
        min_seq_len=1,
        max_seq_len=100,
        seqs_num=-1,
        exclude_ids=excl,
    ).filter()
    content = out.read_text()
    assert ">a" in content
    assert ">b" in content


def test_filter_without_exclude_ids_works(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">a\nMAGV\n")
    out = tmp_path / "out.fasta"
    Filterer(
        input_path=src, output_path=out,
        min_seq_len=1, max_seq_len=100, seqs_num=-1,
    ).filter()
    assert (out).read_text().startswith(">a")


def test_filter_excluded_ids_count_toward_processed(tmp_path):
    """Sequences excluded by ID should still be counted as 'processed'."""
    src = tmp_path / "in.fasta"
    src.write_text(">a\nMAGV\n>b\nVVAGM\n>c\nMMAA\n")
    excl = tmp_path / "excl.txt"
    excl.write_text("b\n")
    out = tmp_path / "out.fasta"
    f = Filterer(
        input_path=src,
        output_path=out,
        min_seq_len=1,
        max_seq_len=100,
        seqs_num=-1,
        exclude_ids=excl,
    )
    f.filter()
    # All 3 sequences were processed; 2 passed (b was excluded).
    assert f.processed_seqs_num == 3
    assert f.num_filtered_seqs == 2
