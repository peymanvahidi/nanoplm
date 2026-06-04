import numpy as np
from nanoplm.cli.data import _count_pretrain_shard_sequences, _sum_pretrain_shard_tokens
from nanoplm.data.infra.manifest import PretrainManifest


def _write_shard(dirpath, name, sizes):
    dirpath.mkdir(parents=True, exist_ok=True)
    np.save(dirpath / f"{name}.idx.npy", np.asarray(sizes, dtype=np.int32))


def test_sum_tokens_matches_idx_sum(tmp_path):
    d = tmp_path / "train"
    _write_shard(d, "shard_0", [10, 20, 30])
    _write_shard(d, "shard_1", [5, 5])
    assert _count_pretrain_shard_sequences(d) == 5
    assert _sum_pretrain_shard_tokens(d) == 70


def test_sum_tokens_empty_dir_is_zero(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    assert _sum_pretrain_shard_tokens(d) == 0


def test_manifest_round_trips_total_train_tokens():
    m = PretrainManifest(
        pipeline_mode="pretrain", seqs_num=5, min_seq_len=5, max_seq_len=30,
        val_ratio=0.1, train_dir="train", val_dir="val",
        train_sequences=5, val_sequences=0, samples_per_shard=2,
        total_train_tokens=70,
    )
    assert m.total_train_tokens == 70
    m2 = PretrainManifest(
        pipeline_mode="pretrain", seqs_num=5, min_seq_len=5, max_seq_len=30,
        val_ratio=0.1, train_dir="train", val_dir="val",
        train_sequences=5, val_sequences=0, samples_per_shard=2,
    )
    assert m2.total_train_tokens == 0
