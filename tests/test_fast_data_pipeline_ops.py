import shutil
import tempfile
from pathlib import Path

from Bio import SeqIO

from nanoplm.data.filterer import Filterer
from nanoplm.data.native_fasta_ops import _DISABLE_NATIVE_FASTA_OPS_ENV
from nanoplm.data.shuffler import FastaShuffler
from nanoplm.data.splitor import Splitor


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    return [(record.id, str(record.seq)) for record in SeqIO.parse(path, "fasta")]


def _reference_filter(
    records: list[tuple[str, str]],
    min_seq_len: int,
    max_seq_len: int,
    seqs_num: int,
    skip_n: int,
) -> tuple[list[tuple[str, str]], int, int]:
    out: list[tuple[str, str]] = []
    skipped = 0
    processed = 0
    passed = 0

    for rec in records:
        if skipped < skip_n:
            skipped += 1
            continue

        processed += 1
        if seqs_num != -1 and passed >= seqs_num:
            break

        seq_len = len(rec[1])
        if min_seq_len <= seq_len <= max_seq_len:
            out.append(rec)
            passed += 1

    return out, processed, passed


def test_filterer_native_and_python_fallback_match_reference(monkeypatch):
    fasta_text = (
        ">s0\nAAAA\n"
        ">s1\nA\n"
        ">s2\nACDEFG\n"
        ">s3\nACDEFGH\n"
        ">s4\nCCCCC\n"
        ">s5\nTTTT\n"
    )
    params = dict(min_seq_len=3, max_seq_len=6, seqs_num=2, skip_n=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.fasta"
        native_output = tmpdir / "filtered_native.fasta"
        py_output = tmpdir / "filtered_py.fasta"
        input_path.write_text(fasta_text, encoding="utf-8")

        records = _parse_fasta(input_path)
        expected_records, expected_processed, expected_passed = _reference_filter(
            records, **params
        )

        monkeypatch.delenv(_DISABLE_NATIVE_FASTA_OPS_ENV, raising=False)
        native_filterer = Filterer(
            input_path=input_path,
            output_path=native_output,
            **params,
        )
        native_filterer.filter()
        assert _parse_fasta(native_output) == expected_records
        assert native_filterer.processed_seqs_num == expected_processed
        assert native_filterer.num_filtered_seqs == expected_passed

        monkeypatch.setenv(_DISABLE_NATIVE_FASTA_OPS_ENV, "1")
        py_filterer = Filterer(
            input_path=input_path,
            output_path=py_output,
            **params,
        )
        py_filterer.filter()
        assert _parse_fasta(py_output) == expected_records
        assert py_filterer.processed_seqs_num == expected_processed
        assert py_filterer.num_filtered_seqs == expected_passed


def test_splitter_native_and_python_fallback_preserve_order(monkeypatch):
    fasta_text = (
        ">s0\nAAAA\n"
        ">s1\nCCCC\n"
        ">s2\nDDDD\n"
        ">s3\nEEEE\n"
        ">s4\nFFFF\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.fasta"
        input_path.write_text(fasta_text, encoding="utf-8")
        original = _parse_fasta(input_path)

        native_train = tmpdir / "train_native.fasta"
        native_val = tmpdir / "val_native.fasta"
        py_train = tmpdir / "train_py.fasta"
        py_val = tmpdir / "val_py.fasta"

        monkeypatch.delenv(_DISABLE_NATIVE_FASTA_OPS_ENV, raising=False)
        native_splitter = Splitor(
            input_file=input_path,
            train_file=native_train,
            val_file=native_val,
            val_ratio=0.4,
        )
        native_train_size, native_val_size = native_splitter.split()

        monkeypatch.setenv(_DISABLE_NATIVE_FASTA_OPS_ENV, "1")
        py_splitter = Splitor(
            input_file=input_path,
            train_file=py_train,
            val_file=py_val,
            val_ratio=0.4,
        )
        py_train_size, py_val_size = py_splitter.split()

        expected_val = int(len(original) * 0.4)
        expected_train = len(original) - expected_val
        assert (native_train_size, native_val_size) == (expected_train, expected_val)
        assert (py_train_size, py_val_size) == (expected_train, expected_val)

        native_concat = _parse_fasta(native_train) + _parse_fasta(native_val)
        py_concat = _parse_fasta(py_train) + _parse_fasta(py_val)
        assert native_concat == original
        assert py_concat == original


def test_fast_shuffle_backend_preserves_records_and_seed():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.fasta"
        out1 = tmpdir / "out1.fasta"
        out2 = tmpdir / "out2.fasta"

        with open(input_path, "w", encoding="utf-8") as f:
            for i in range(40):
                f.write(f">s{i}\n")
                f.write(f"ACDEFGHIKLMNPQRSTVWY{i % 10}\n")

        original = _parse_fasta(input_path)
        original_ids = [x[0] for x in original]

        sh1 = FastaShuffler(
            input_path=input_path,
            output_path=out1,
            backend="fast",
            seed=777,
        )
        sh1.shuffle()

        sh2 = FastaShuffler(
            input_path=input_path,
            output_path=out2,
            backend="fast",
            seed=777,
        )
        sh2.shuffle()

        out1_records = _parse_fasta(out1)
        out2_records = _parse_fasta(out2)

        assert out1_records == out2_records
        assert sorted(x[0] for x in out1_records) == sorted(original_ids)
        assert [x[0] for x in out1_records] != original_ids


def test_auto_shuffle_backend_selection():
    shuffler = FastaShuffler("in.fasta", "out.fasta", backend="auto")
    selected = shuffler._choose_backend()
    if shutil.which("seqkit"):
        assert selected == "seqkit"
    else:
        assert selected == "fast"
