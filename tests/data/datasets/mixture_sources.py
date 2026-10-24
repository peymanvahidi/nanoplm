"""Tiny prepared sources for the mixture tests, written in the on-disk shard format."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nanoplm.data.infra.manifest import PretrainManifest, write_manifest

from tests.data.datasets.cluster_pool import build_pool


def write_shards(folder: Path, lengths, seed: int, tag: int) -> None:
    """One shard whose rows start with ``tag`` (the source) and then the row number in
    base 20, so every row differs. Every token is a real amino-acid id (4..24)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i, n in enumerate(lengths):
        row = rng.integers(4, 25, int(n), dtype=np.uint8)
        row[0], row[1], row[2], row[3] = tag, i // 400 % 20 + 4, i // 20 % 20 + 4, i % 20 + 4
        row[-1] = 1
        rows.append(row)
    (folder / "shard_0000.bin").write_bytes(np.concatenate(rows).tobytes())
    with open(folder / "shard_0000.idx.npy", "wb") as f:
        np.save(f, np.asarray(lengths, dtype=np.int32))


def plain_source(root: Path, n_train: int, tag: int, seed: int = 0, n_val: int = 9,
                 max_len: int = 64, register_tokens: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    train = rng.integers(8, max_len, n_train)
    val = rng.integers(8, max_len, n_val)
    write_shards(root / "train", train, seed, tag)
    write_shards(root / "val", val, seed + 1, tag)
    write_manifest(root, PretrainManifest(
        pipeline_mode="pretrain", seqs_num=n_train + n_val, min_seq_len=8, max_seq_len=max_len,
        val_ratio=0.1, train_dir="train", val_dir="val", train_sequences=n_train,
        val_sequences=n_val, num_register_tokens=register_tokens, samples_per_shard=1000,
        total_train_tokens=int(train.sum())))
    return root


def cluster_source(root: Path, seed: int = 0, n_small: int = 300):
    """A cluster-sampled source (build_pool's pool and index) with a manifest; returns
    (folder, pool)."""
    pool = build_pool(root, seed=seed, n_small=n_small, n_shards=3)
    write_shards(root / "val", [20] * 7, seed + 5, 3)
    n_train = len(pool.train_clusters)
    write_manifest(root, PretrainManifest(
        pipeline_mode="pretrain", seqs_num=n_train + 7, min_seq_len=20, max_seq_len=128,
        val_ratio=0.01, train_dir="train", val_dir="val", train_sequences=n_train,
        val_sequences=7, samples_per_shard=1000, total_train_tokens=n_train * 70,
        cluster_index_dir="cluster_index", cluster_level="uniref90",
        pool_sequences=len(pool.lengths), pool_tokens=int(pool.lengths.sum())))
    return root, pool
