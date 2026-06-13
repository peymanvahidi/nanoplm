from pathlib import Path
import json
import yaml
import numpy as np
from click.testing import CliRunner
from nanoplm.cli.data import data


def _make_pretrain_shard_dir(d: Path, sequence_counts: list[int]) -> int:
    """Create N shards under d, each with the given sequence count.

    Each shard's .idx.npy is a length-N int32 array of per-sequence encoded
    lengths, matching the C writer's actual on-disk format. Sizes are filled
    with deterministic non-zero values so callers can verify token sums.

    Returns the total sum of all per-sequence sizes written.
    """
    d.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, n in enumerate(sequence_counts):
        (d / f"shard_{i:04d}.bin").write_bytes(b"\x00" * 10)
        sizes = np.arange(1, n + 1, dtype=np.int32)
        total += int(sizes.sum())
        np.save(d / f"shard_{i:04d}.idx.npy", sizes)
    return total


def test_write_manifest_pretrain(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    expected_train_tokens = _make_pretrain_shard_dir(train_dir, [100, 100])  # 2 shards, 100 each → 200
    _make_pretrain_shard_dir(val_dir, [25])           # 1 shard, 25 seqs

    runner = CliRunner()
    result = runner.invoke(data, [
        "write-manifest", "--kind", "pretrain", "-o", str(tmp_path),
        "--min-seq-len", "20", "--max-seq-len", "512",
        "--val-ratio", "0.1", "--samples-per-shard", "100",
        "--register-tokens", "0",
    ])
    assert result.exit_code == 0, result.output
    manifest_path = tmp_path / ".data_manifest"
    assert manifest_path.exists()

    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["train_sequences"] == 200
    assert manifest["val_sequences"] == 25
    assert manifest["seqs_num"] == 225
    assert manifest["total_train_tokens"] == expected_train_tokens


def test_write_manifest_errors_when_train_dir_missing(tmp_path):
    val_dir = tmp_path / "val"
    _make_pretrain_shard_dir(val_dir, [25])
    # train/ deliberately omitted

    runner = CliRunner()
    result = runner.invoke(data, [
        "write-manifest", "--kind", "pretrain", "-o", str(tmp_path),
        "--min-seq-len", "20", "--max-seq-len", "512",
        "--val-ratio", "0.1", "--samples-per-shard", "100",
        "--register-tokens", "0",
    ])
    assert result.exit_code != 0
    assert "train" in result.output.lower()


def test_write_manifest_pretrain_records_register_tokens(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _make_pretrain_shard_dir(train_dir, [10])
    _make_pretrain_shard_dir(val_dir, [5])

    runner = CliRunner()
    result = runner.invoke(data, [
        "write-manifest", "--kind", "pretrain", "-o", str(tmp_path),
        "--min-seq-len", "20", "--max-seq-len", "512",
        "--val-ratio", "0.1", "--samples-per-shard", "10",
        "--register-tokens", "2",
    ])
    assert result.exit_code == 0, result.output

    manifest = yaml.safe_load((tmp_path / ".data_manifest").read_text())
    assert manifest["num_register_tokens"] == 2
