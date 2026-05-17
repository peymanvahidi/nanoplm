import pytest
from pathlib import Path
from unittest.mock import MagicMock
from nanoplm.data.sharding.pretrain_shard_writer import ShardWriter


def test_rejects_non_native_tokenizer(tmp_path):
    """Custom tokenizers must raise NotImplementedError with TODO reference."""
    fake_tokenizer = MagicMock()
    fake_tokenizer.__class__.__name__ = "SomeBPETokenizer"
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">seq1\nMAGV\n")
    writer = ShardWriter(
        fasta_path=str(fasta),
        tokenizer=fake_tokenizer,
        max_length=64,
        output_dir=str(tmp_path / "out"),
    )
    with pytest.raises(NotImplementedError) as exc:
        writer.create_shards()
    assert "TODO(custom-tokenizers)" in str(exc.value)
