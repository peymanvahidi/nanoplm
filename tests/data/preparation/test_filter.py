# tests/data/preparation/test_filter.py
from pathlib import Path

import pytest
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


def test_filter_missing_exclude_file_errors_at_cli(tmp_path):
    """CLI should reject a missing --exclude-ids file at parse time, not runtime."""
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    src = tmp_path / "in.fasta"
    src.write_text(">a\nMAGV\n")
    out = tmp_path / "out.fasta"
    runner = CliRunner()
    result = runner.invoke(
        data,
        [
            "filter",
            "-i", str(src),
            "-o", str(out),
            "--min-seq-len", "1",
            "--max-seq-len", "100",
            "--exclude-ids", str(tmp_path / "does-not-exist.txt"),
        ],
    )
    assert result.exit_code != 0
    # Click's exists=True produces a message containing "does not exist" or similar
    assert "does not exist" in result.output.lower() or "no such" in result.output.lower()


def _filter(tmp_path, text, min_len=1, max_len=100):
    src = tmp_path / "in.fa"
    src.write_text(text)
    out = tmp_path / "out.fasta"
    f = Filterer(input_path=src, output_path=out, min_seq_len=min_len, max_seq_len=max_len,
                 seqs_num=-1)
    f.filter()
    return out.read_text(), f


def test_filter_strips_one_trailing_stop_codon(tmp_path):
    got, _ = _filter(tmp_path, ">a # 1 # 15 # 1 # ID=1_1\nMAGV*\n>b\nVVA\nGM*\n>c\nMAGV\n")
    assert got == ">a # 1 # 15 # 1 # ID=1_1\nMAGV\n>b\nVVAGM\n>c\nMAGV\n"


def test_filter_counts_length_after_the_stop_codon_is_stripped(tmp_path):
    # MAGV* is 4 residues: kept by max 4, dropped by min 5
    got, f = _filter(tmp_path, ">a\nMAGV*\n>b\nMAGVK*\n", min_len=1, max_len=4)
    assert got == ">a\nMAGV\n" and f.num_filtered_seqs == 1
    got, _ = _filter(tmp_path, ">a\nMAGV*\n>b\nMAGVK*\n", min_len=5, max_len=100)
    assert got == ">b\nMAGVK\n"


def test_filter_leaves_uniprot_style_input_unchanged(tmp_path):
    text = (">sp|P0|X_HUMAN Protein OS=Homo sapiens OX=9606 GN=X PE=1 SV=1\nMAGVKLLW\n"
            ">tr|Q1|Y_HUMAN Other\nMKV\n")
    got, _ = _filter(tmp_path, text)
    assert got == text


@pytest.mark.parametrize("header", [b" b", b"\tb x", b" b desc ", b"b\x0bdesc", b"b\x0c"])
def test_check_and_filter_agree_on_the_id(tmp_path, header):
    """With drop_invalid the filter drops the ids check lists. "> b" used to be id "" in
    check and "b" in the filter, so its invalid sequence reached filtered.fasta. Both now
    take the id cluster_table / cluster_index take: the header's first word."""
    from nanoplm.data.preparation.checker import Checker
    src = tmp_path / "in.fa"
    src.write_bytes(b">a\nMAGV\n>" + header + b"\nMA*G\n")
    Checker(input_path=src, output_dir=tmp_path / "val", drop_invalid=True).check()
    ids = (tmp_path / "val/invalid_ids.txt").read_bytes().splitlines()
    assert ids == [header.split(None, 1)[0]] == [b"b"]
    out = tmp_path / "out.fasta"
    Filterer(input_path=src, output_path=out, min_seq_len=1, max_seq_len=100, seqs_num=-1,
             exclude_ids=tmp_path / "val/invalid_ids.txt").filter()
    assert out.read_bytes() == b">a\nMAGV\n"


def test_check_writes_uniprot_style_invalid_records_unchanged(tmp_path):
    from nanoplm.data.preparation.checker import Checker
    bad = b">sp|P0|X_HUMAN Protein OS=Homo sapiens OX=9606 GN=X PE=1 SV=1\nMAG!V\n"
    src = tmp_path / "in.fasta"
    src.write_bytes(b">tr|Q1|Y_HUMAN Other\nMKV\n" + bad)
    Checker(input_path=src, output_dir=tmp_path / "val", drop_invalid=True).check()
    assert (tmp_path / "val/invalid.fasta").read_bytes() == bad
    assert (tmp_path / "val/invalid_ids.txt").read_bytes() == b"sp|P0|X_HUMAN\n"
