import tempfile
from pathlib import Path

import numpy as np
import pytest

from nanoplm.pretraining.dataset import ShardWriter, ShardedDataset
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.native_shard_writer import (
    _DISABLE_NATIVE_ENV,
    create_shards_native,
    is_native_shard_writer_available,
)


def _expected_tokens(tokenizer, sequences, max_length):
    expected = []
    for seq in sequences:
        encoding = tokenizer(
            seq,
            add_special_tokens=True,
            padding=False,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        expected.append(encoding["input_ids"].squeeze(0).numpy().astype(np.uint8))
    return expected


def test_create_shards_native_matches_tokenizer_outputs():
    native_available, native_error = is_native_shard_writer_available()
    if not native_available:
        pytest.skip(f"Native shard writer unavailable: {native_error}")

    sequences = [
        "ALGVSREDTIPKFQNYMHWCX",
        "uzobx",  # tokenizer normalizes U/Z/O/B -> X
        "mKa*?!",  # lowercase + unknown characters
        "",  # empty sequence should still emit EOS
        "A" * 1000,  # truncation case
    ]
    max_length = 32
    tokenizer = ProtModernBertTokenizer()

    fasta_text = (
        ">s0\nALGVSREDTIPKFQNYMHWCX\n"
        ">s1\nuzobx\n"
        ">s2\nmKa*?!\n"
        ">s3\n"
        ">s4\n" + ("A" * 1000) + "\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "sample.fasta"
        output_dir = Path(tmpdir) / "shards"
        output_dir.mkdir(parents=True, exist_ok=True)
        fasta_path.write_text(fasta_text, encoding="utf-8")

        result = create_shards_native(
            fasta_path=fasta_path,
            output_dir=output_dir,
            max_length=max_length,
            samples_per_shard=2,
            num_threads=4,
            use_bos_token=False,
        )

        assert result.sequence_count == len(sequences)
        assert len(result.shard_paths) == 3

        ds = ShardedDataset(str(output_dir))
        assert len(ds) == len(sequences)

        expected = _expected_tokens(tokenizer, sequences, max_length=max_length)
        for idx in range(len(ds)):
            got = ds[idx]["input_ids"].numpy().astype(np.uint8)
            np.testing.assert_array_equal(got, expected[idx])


def test_shard_writer_python_fallback_still_produces_valid_shards(monkeypatch):
    sequences = ["AAAA", "CccC", "uzob", "WWWW", "M" * 40]
    tokenizer = ProtModernBertTokenizer()
    max_length = 16
    expected = _expected_tokens(tokenizer, sequences, max_length=max_length)

    fasta_text = (
        ">a\nAAAA\n"
        ">b\nCccC\n"
        ">c\nuzob\n"
        ">d\nWWWW\n"
        ">e\n" + ("M" * 40) + "\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "sample.fasta"
        output_dir = Path(tmpdir) / "shards"
        fasta_path.write_text(fasta_text, encoding="utf-8")

        monkeypatch.setenv(_DISABLE_NATIVE_ENV, "1")
        saver = ShardWriter(
            fasta_path=str(fasta_path),
            tokenizer=tokenizer,
            max_length=max_length,
            output_dir=str(output_dir),
            samples_per_shard=2,
            max_workers=1,
        )
        shard_paths = saver.create_shards()

        assert saver.sequence_count == len(sequences)
        assert len(shard_paths) == 3

        ds = ShardedDataset(str(output_dir))
        assert len(ds) == len(sequences)
        for idx in range(len(ds)):
            got = ds[idx]["input_ids"].numpy().astype(np.uint8)
            np.testing.assert_array_equal(got, expected[idx])
