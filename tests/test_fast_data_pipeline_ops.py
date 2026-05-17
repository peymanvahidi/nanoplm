import time
import os
from pathlib import Path

import pytest
from Bio import SeqIO

from nanoplm.data.preparation.filterer import Filterer
from nanoplm.data.preparation.shuffler import FastaShuffler
from nanoplm.data.preparation.splitor import Splitor


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    return [(record.id, str(record.seq)) for record in SeqIO.parse(path, "fasta")]


def _count_headers(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                count += 1
    return count


def _first_header(path: Path) -> bytes:
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                return line.rstrip()
    return b""


def test_fast_shuffle_backend_preserves_records_and_seed(tmp_path):
    input_path = tmp_path / "input.fasta"
    out1 = tmp_path / "out1.fasta"
    out2 = tmp_path / "out2.fasta"

    with open(input_path, "w", encoding="utf-8") as f:
        for i in range(40):
            f.write(f">s{i}\n")
            f.write(f"ACDEFGHIKLMNPQRSTVWY{i % 10}\n")

    original = _parse_fasta(input_path)
    original_ids = [x[0] for x in original]

    sh1 = FastaShuffler(
        input_path=input_path,
        output_path=out1,
        seed=777,
    )
    sh1.shuffle()

    sh2 = FastaShuffler(
        input_path=input_path,
        output_path=out2,
        seed=777,
    )
    sh2.shuffle()

    out1_records = _parse_fasta(out1)
    out2_records = _parse_fasta(out2)

    assert out1_records == out2_records
    assert sorted(x[0] for x in out1_records) == sorted(original_ids)
    assert [x[0] for x in out1_records] != original_ids


def test_shuffle_noop_symlink(tmp_path):
    """do_shuffle=False should create a symlink (or copy) rather than shuffling."""
    input_path = tmp_path / "input.fasta"
    output_path = tmp_path / "out.fasta"
    input_path.write_text(">a\nACDE\n>b\nFGHI\n", encoding="utf-8")

    shuffler = FastaShuffler(
        input_path=input_path,
        output_path=output_path,
        do_shuffle=False,
    )
    shuffler.shuffle()

    assert output_path.exists()
    # Content must be identical to input
    assert _parse_fasta(output_path) == _parse_fasta(input_path)


def test_filterer_native_filters_by_length(tmp_path):
    """Native filter path: only sequences in [min_seq_len, max_seq_len] are kept."""
    fasta_text = (
        ">s0\nAAAA\n"
        ">s1\nA\n"
        ">s2\nACDEFG\n"
        ">s3\nACDEFGH\n"
        ">s4\nCCCCC\n"
        ">s5\nTTTT\n"
    )
    input_path = tmp_path / "input.fasta"
    output_path = tmp_path / "filtered.fasta"
    input_path.write_text(fasta_text, encoding="utf-8")

    filterer = Filterer(
        input_path=input_path,
        output_path=output_path,
        min_seq_len=4,
        max_seq_len=6,
        seqs_num=-1,
    )
    filterer.filter()

    result = _parse_fasta(output_path)
    result_ids = [r[0] for r in result]
    # s1 (len=1) and s3 (len=7) should be filtered out
    assert "s1" not in result_ids
    assert "s3" not in result_ids
    # s0 (len=4), s2 (len=6), s4 (len=5), s5 (len=4) should pass
    assert "s0" in result_ids
    assert "s2" in result_ids
    assert "s4" in result_ids
    assert "s5" in result_ids
    assert filterer.num_filtered_seqs == len(result)


def test_filterer_exclude_ids_raises_not_implemented(tmp_path):
    """exclude_ids raises NotImplementedError until Phase 10."""
    input_path = tmp_path / "input.fasta"
    output_path = tmp_path / "filtered.fasta"
    input_path.write_text(">a\nACDE\n", encoding="utf-8")
    exclude_ids = tmp_path / "exclude.txt"
    exclude_ids.write_text("a\n")

    filterer = Filterer(
        input_path=input_path,
        output_path=output_path,
        min_seq_len=1,
        max_seq_len=100,
        seqs_num=-1,
        exclude_ids=exclude_ids,
    )
    with pytest.raises(NotImplementedError, match="exclude_ids"):
        filterer.filter()


def test_splitor_native_preserves_order(tmp_path):
    """Native split path: train + val concatenation equals original in order."""
    fasta_text = (
        ">s0\nAAAA\n"
        ">s1\nCCCC\n"
        ">s2\nDDDD\n"
        ">s3\nEEEE\n"
        ">s4\nFFFF\n"
    )
    input_path = tmp_path / "input.fasta"
    input_path.write_text(fasta_text, encoding="utf-8")
    original = _parse_fasta(input_path)

    train_path = tmp_path / "train.fasta"
    val_path = tmp_path / "val.fasta"

    splitor = Splitor(
        input_file=input_path,
        train_file=train_path,
        val_file=val_path,
        val_ratio=0.4,
    )
    train_size, val_size = splitor.split()

    expected_val = int(len(original) * 0.4)
    expected_train = len(original) - expected_val
    assert (train_size, val_size) == (expected_train, expected_val)

    concat = _parse_fasta(train_path) + _parse_fasta(val_path)
    assert concat == original


@pytest.mark.slow
def test_fast_shuffle_benchmark_5m_records(tmp_path, capsys):
    records = int(os.environ.get("NANOPLM_SHUFFLE_BENCH_RECORDS", "5000000"))
    min_records_per_sec = float(os.environ.get("NANOPLM_SHUFFLE_BENCH_MIN_RECORDS_PER_SEC", "0"))

    input_path = tmp_path / "input.fasta"
    output_path = tmp_path / "output.fasta"
    seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN"

    with open(input_path, "w", encoding="utf-8") as f:
        for i in range(records):
            f.write(f">seq{i}\n{seq}\n")

    shuffler = FastaShuffler(
        input_path=input_path,
        output_path=output_path,
        seed=777,
    )

    start = time.perf_counter()
    shuffler.shuffle()
    elapsed = time.perf_counter() - start
    records_per_sec = records / max(elapsed, 1e-9)

    assert output_path.exists()
    assert output_path.stat().st_size == input_path.stat().st_size
    assert _count_headers(output_path) == records
    assert _first_header(output_path) != _first_header(input_path)

    if min_records_per_sec > 0:
        assert records_per_sec >= min_records_per_sec

    captured = capsys.readouterr()
    benchmark_line = (
        f"fast shuffle benchmark: records={records:,} "
        f"elapsed={elapsed:.3f}s rate={records_per_sec:,.0f} rec/s "
        f"input_size={input_path.stat().st_size / (1024 ** 2):.1f} MiB"
    )
    print(benchmark_line)
    if captured.out:
        print(captured.out)
