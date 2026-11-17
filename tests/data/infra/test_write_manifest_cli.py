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


# PretrainManifest's fields on master (before cluster subsampling). Master reads a
# manifest with PretrainManifest(**data), so any other key makes it fail; old
# worktrees and checkpoints' code must keep reading a plain dataset built today.
MASTER_PRETRAIN_FIELDS = {
    "pipeline_mode", "seqs_num", "min_seq_len", "max_seq_len", "val_ratio", "train_dir",
    "val_dir", "train_sequences", "val_sequences", "num_register_tokens", "created_at",
    "sharded", "samples_per_shard", "total_train_tokens",
}


def test_plain_manifest_has_only_the_fields_master_reads(tmp_path):
    _make_pretrain_shard_dir(tmp_path / "train", [100])
    _make_pretrain_shard_dir(tmp_path / "val", [10])
    result = CliRunner().invoke(data, [
        "write-manifest", "--kind", "pretrain", "-o", str(tmp_path),
        "--min-seq-len", "20", "--max-seq-len", "512",
        "--val-ratio", "0.1", "--samples-per-shard", "100",
    ])
    assert result.exit_code == 0, result.output
    keys = set(yaml.safe_load((tmp_path / ".data_manifest").read_text()))
    assert keys <= MASTER_PRETRAIN_FIELDS, keys - MASTER_PRETRAIN_FIELDS


def _write(tmp_path, *extra):
    _make_pretrain_shard_dir(tmp_path / "train", [10])
    _make_pretrain_shard_dir(tmp_path / "val", [5])
    return CliRunner().invoke(data, [
        "write-manifest", "--kind", "pretrain", "-o", str(tmp_path),
        "--min-seq-len", "20", "--max-seq-len", "512",
        "--val-ratio", "0.1", "--samples-per-shard", "10", *extra,
    ])


def test_write_manifest_records_shuffle_only_when_false(tmp_path):
    from nanoplm.data.infra.manifest import read_manifest
    on, off = tmp_path / "on", tmp_path / "off"
    assert _write(on).exit_code == 0
    assert "shuffle" not in yaml.safe_load((on / ".data_manifest").read_text())
    assert read_manifest(on).shuffle is None
    r = _write(off, "--no-shuffle")
    assert r.exit_code == 0, r.output
    assert yaml.safe_load((off / ".data_manifest").read_text())["shuffle"] is False
    assert read_manifest(off).shuffle is False


def test_write_manifest_no_shuffle_for_a_cluster_dataset(tmp_path, monkeypatch):
    from nanoplm.data.infra.manifest import PretrainManifest, write_manifest
    from nanoplm.data.preparation import cluster_pipeline

    def fake(output_dir, *args):
        m = PretrainManifest(pipeline_mode="pretrain", seqs_num=3, min_seq_len=1,
                             max_seq_len=9, val_ratio=0.1, train_dir="train", val_dir="val",
                             train_sequences=2, val_sequences=1, samples_per_shard=2,
                             cluster_index_dir="cluster_index")
        write_manifest(output_dir, m)
        return m

    monkeypatch.setattr(cluster_pipeline, "write_cluster_manifest", fake)
    r = CliRunner().invoke(data, [
        "write-manifest", "--kind", "pretrain", "--cluster-index", "-o", str(tmp_path),
        "--min-seq-len", "1", "--max-seq-len", "9", "--val-ratio", "0.1",
        "--samples-per-shard", "2", "--no-shuffle"])
    assert r.exit_code == 0, r.output
    got = yaml.safe_load((tmp_path / ".data_manifest").read_text())
    assert got["shuffle"] is False and got["cluster_index_dir"] == "cluster_index"
