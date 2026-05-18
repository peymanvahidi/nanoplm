from pathlib import Path
import json
import numpy as np
from click.testing import CliRunner
from nanoplm.cli.data import data


def _make_pretrain_shard_dir(d: Path, sequence_counts: list[int]) -> None:
    """Create N shards under d, each with the given sequence count.

    Convention: `shard_0000.idx.npy` is a numpy int64 array of N+1 offsets
    for N sequences. `_count_pretrain_shard_sequences` returns `len(arr)-1`.
    """
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(sequence_counts):
        (d / f"shard_{i:04d}.bin").write_bytes(b"\x00" * 10)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.save(d / f"shard_{i:04d}.idx.npy", offsets)


def test_write_manifest_pretrain(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _make_pretrain_shard_dir(train_dir, [100, 100])  # 2 shards, 100 each → 200
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
