import tempfile
from pathlib import Path

import numpy as np
import pytest

from nanoplm.data.datasets.pretrain import ShardedDataset
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.data.sharding.pretrain_shard_writer import (
    ShardWriter,
    NativeShardWriterError,
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


def test_shard_writer_raises_when_native_disabled(tmp_path, monkeypatch):
    """ShardWriter.create_shards must raise NativeShardWriterError when the native
    backend is disabled by env var (no more Python fallback)."""
    fasta = tmp_path / "seq.fasta"
    fasta.write_text(">s1\nMAGV\n>s2\nKVAR\n")
    out_dir = tmp_path / "shards"

    monkeypatch.setenv(_DISABLE_NATIVE_ENV, "1")

    saver = ShardWriter(
        fasta_path=str(fasta),
        tokenizer=ProtModernBertTokenizer(),
        max_length=8,
        output_dir=str(out_dir),
        samples_per_shard=10,
        max_workers=1,
        force=True,
    )

    with pytest.raises(NativeShardWriterError):
        saver.create_shards()
