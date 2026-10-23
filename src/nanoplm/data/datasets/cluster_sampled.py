"""Cluster-sampled pretraining dataset: one item per UniRef50 train cluster per epoch.

The train directory holds a *pool*: every UniRef90 (or UniRef100) member sequence that
passed the length filter, pre-tokenized in FASTA order. The cluster index next to it
(``cluster_index/``, see ``nanoplm.data.preparation.cluster_index``) groups pool rows by
UniRef50 cluster. Item ``i`` is a visit to train cluster ``c = train_clusters[i]``, and
the member it returns is drawn uniformly from the cluster's ``n_c`` surviving members
(representative included):

    slot = splitmix64(splitmix64(member_seed_key(seed) ^ c) ^ epoch) % n_c
    row  = cluster_members[cluster_offsets[global_to_universe[c]] + slot]

The draw is a pure function of (seed, cluster, epoch). It holds no RNG state, so the
member a cluster gives in an epoch does not depend on num_workers, fetch order, world
size or resume. The order of the visits does: the sampler permutes cluster positions
each epoch as it would for any other dataset, and with packing the DataLoader workers
each pack their share and interleave the batches. ``set_epoch`` moves every cluster to
a fresh member.

The arithmetic is bit-identical to the experiment shim it was ported from
(``runtime_sampler_patch.py``, frozen under ``tests/fixtures/``); the test suite
checks this.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from nanoplm.data.datasets.pretrain import ShardedDataset
from nanoplm.data.datasets.splitmix import MASK64, splitmix64, splitmix64_vec
from nanoplm.utils import logger

# b"RTMEMB": separates the member stream from anything else keyed on the same seed
MEMBER_TAG = 0x52_54_4D_45_4D_42
# rows_for_epoch works in chunks so its temporaries stay small at 33M clusters
VEC_CHUNK = 1 << 22

INDEX_MANIFEST = "pool_manifest.json"
# Identity of the member draw. Part of the fingerprint, so a change to the draw is a
# dataset change for resume purposes. Kept equal to the shim's tag on purpose.
_FINGERPRINT_TAG = b"runtime-cluster-sampler-v1"


# ------------------------------------------------------------------ the stateless draw


def member_seed_key(seed: int) -> int:
    """The 64-bit key every member draw starts from; the training seed goes in as is."""
    return splitmix64((int(seed) ^ MEMBER_TAG) & MASK64)


def member_key(seed: int, cluster: int, epoch: int) -> int:
    return splitmix64(splitmix64(member_seed_key(seed) ^ int(cluster)) ^ int(epoch))


def member_slot(seed: int, cluster: int, epoch: int, n: int) -> int:
    return member_key(seed, cluster, epoch) % int(n)


def member_slots_vec(seed: int, clusters: np.ndarray, epoch, n: np.ndarray) -> np.ndarray:
    """Vectorized ``member_slot``; returns the same values. ``epoch`` is one int or an
    array broadcast against ``clusters`` (a mixture draws each item at its own pass)."""
    k0 = np.uint64(member_seed_key(seed))
    e = np.asarray(epoch)
    e = np.uint64(int(e) & MASK64) if e.ndim == 0 else e.astype(np.uint64)
    with np.errstate(over="ignore"):
        z = splitmix64_vec(np.asarray(clusters).astype(np.uint64) ^ k0)
        z = splitmix64_vec(z ^ e)
    return (z % np.asarray(n).astype(np.uint64)).astype(np.int64)


# ------------------------------------------------------------------------ the dataset


class ClusterSampledDataset(ShardedDataset):
    """ShardedDataset over the pool shards, indexed by train cluster, member drawn per epoch.

    Every index array is memory-mapped, so a DataLoader worker costs no extra RAM for
    the index whether it is forked or spawned. Only the main process materialises
    per-epoch row and length arrays (``rows_for_epoch``, ``get_all_sequence_lengths``).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        index_dir: Union[str, Path],
        seed: int,
        pad_token_id: int = 0,
        epoch: int = 0,
    ) -> None:
        self._cs_init(str(data_dir), str(index_dir), int(seed), int(pad_token_id),
                      int(epoch), check=True, log=True)

    # ---- init -------------------------------------------------------------------
    def _cs_init(self, data_dir: str, index_dir: str, seed: int, pad_token_id: int,
                 epoch: int, check: bool, log: bool) -> None:
        # stock init over the pool shards: _bin_paths, per-shard _offsets, cum_lengths
        self._init_dataset(data_dir, pad_token_id=pad_token_id, log=log)
        self.index_dir = Path(index_dir)
        self.seed = seed
        self._epoch = epoch
        self._lengths_cache: Optional[tuple] = None
        self._pool_lengths: Optional[np.ndarray] = None
        ix = self.index_dir
        man_path = ix / INDEX_MANIFEST
        if not man_path.exists():
            raise FileNotFoundError(f"cluster index manifest not found: {man_path}")
        man = json.loads(man_path.read_text())
        if man.get("verify", {}).get("ok") is not True:
            raise RuntimeError(f"{man_path}: verify.ok is not true; verify the index first")
        self.index_manifest = man

        pool_rows = int(self.cum_lengths[-1])
        sro = np.load(ix / "shard_row_offsets.npy")
        if len(self._bin_paths) != len(sro) - 1 or not np.array_equal(
                np.asarray(self.cum_lengths, dtype=np.int64), sro[1:]):
            raise RuntimeError(
                f"{data_dir}: {len(self._bin_paths)} shards / {pool_rows} rows do not match "
                f"{ix}/shard_row_offsets.npy ({len(sro) - 1} shards, {int(sro[-1])} rows); "
                "the train directory must be the pool the index was built on")

        self._tc = np.load(ix / "train_clusters.npy", mmap_mode="r")
        self._g2u = np.load(ix / "global_to_universe.npy", mmap_mode="r")
        self._off = np.load(ix / "cluster_offsets.npy", mmap_mode="r")
        self._members = np.load(ix / "cluster_members.npy", mmap_mode="r")
        self._n = len(self._tc)
        if self._n == 0:
            raise RuntimeError(f"{ix}/train_clusters.npy is empty")
        if check:
            self._check_index(pool_rows)
        if log:
            logger.info(
                f"Cluster-sampled train set: {self._n:,} train clusters over "
                f"{pool_rows:,} pool rows, {len(self._members):,} members, "
                f"seed={seed}, index={ix}")

    def _check_index(self, pool_rows: int) -> None:
        """One pass over the index in the main process; workers skip it."""
        n_members = len(self._members)
        for a in range(0, self._n, VEC_CHUNK):
            start, size = self._segments(a, min(a + VEC_CHUNK, self._n))
            if size.min() < 1 or int((start + size).max()) > n_members:
                raise RuntimeError("cluster CSR is inconsistent with cluster_members.npy")
        for a in range(0, n_members, VEC_CHUNK):
            chunk = self._members[a:a + VEC_CHUNK]
            if int(chunk.min()) < 0 or int(chunk.max()) >= pool_rows:
                raise RuntimeError("cluster_members.npy points outside the pool")

    def _segments(self, a: int, b: int):
        """CSR (start, size) of train positions [a, b)."""
        uloc = np.asarray(self._g2u[np.asarray(self._tc[a:b])], dtype=np.int64)
        if uloc.min() < 0:
            raise RuntimeError("train_clusters.npy holds a cluster outside the universe")
        start = np.asarray(self._off[uloc], dtype=np.int64)
        size = np.asarray(self._off[uloc + 1], dtype=np.int64) - start
        return start, size

    # ---- pickling (spawn/forkserver workers; fork does not use it) ---------------
    def __getstate__(self) -> dict:
        return {"data_dir": str(self.data_dir), "pad_token_id": self.pad_token_id,
                "index_dir": str(self.index_dir), "seed": self.seed, "epoch": self._epoch}

    def __setstate__(self, state) -> None:
        self._cs_init(state["data_dir"], state["index_dir"], state["seed"],
                      state["pad_token_id"], state["epoch"], check=False, log=False)

    # ---- the contract -----------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def cluster_size(self, idx: int) -> int:
        c = int(self._tc[idx])
        u = int(self._g2u[c])
        return int(self._off[u + 1]) - int(self._off[u])

    def pool_row(self, idx: int) -> int:
        """Pool row visited at position idx in the current epoch."""
        c = int(self._tc[idx])
        u = int(self._g2u[c])
        start = int(self._off[u])
        slot = member_slot(self.seed, c, self._epoch, int(self._off[u + 1]) - start)
        return int(self._members[start + slot])

    def rows_for_epoch(self, epoch: Optional[int] = None) -> np.ndarray:
        """Pool row of every train position at ``epoch`` (default: the current one)."""
        e = self._epoch if epoch is None else int(epoch)
        out = np.empty(self._n, dtype=np.int64)
        for a in range(0, self._n, VEC_CHUNK):
            b = min(a + VEC_CHUNK, self._n)
            start, size = self._segments(a, b)
            slot = member_slots_vec(self.seed, np.asarray(self._tc[a:b]), e, size)
            out[a:b] = self._members[start + slot]
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of bounds for {self._n} train clusters")
        return self.read_pool_row(self.pool_row(idx))

    def item_at(self, idx: int, epoch: int) -> Dict[str, torch.Tensor]:
        """Position ``idx`` with the member of pass ``epoch``, whatever the current epoch
        is. A mixture keeps one pass counter per source and reads through this."""
        c = int(self._tc[idx])
        u = int(self._g2u[c])
        start = int(self._off[u])
        slot = member_slot(self.seed, c, int(epoch), int(self._off[u + 1]) - start)
        return self.read_pool_row(int(self._members[start + slot]))

    def lengths_at(self, idx: np.ndarray, epochs: np.ndarray) -> np.ndarray:
        """Token count of the member position ``idx[i]`` gives in pass ``epochs[i]``."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty(len(idx), dtype=np.int32)
        lengths = self.pool_lengths()
        for a in range(0, len(idx), VEC_CHUNK):
            b = min(a + VEC_CHUNK, len(idx))
            tc = np.asarray(self._tc[idx[a:b]])
            uloc = np.asarray(self._g2u[tc], dtype=np.int64)
            start = np.asarray(self._off[uloc], dtype=np.int64)
            size = np.asarray(self._off[uloc + 1], dtype=np.int64) - start
            slot = member_slots_vec(self.seed, tc, np.asarray(epochs)[a:b], size)
            out[a:b] = lengths[self._members[start + slot]]
        return out

    def read_pool_row(self, row: int) -> Dict[str, torch.Tensor]:
        """Pool row ``row``, read the way ShardedDataset.__getitem__ reads its rows.

        ShardedDataset.__getitem__ itself cannot be reused: it bounds-checks against
        len(self), which here is the train cluster count, not the pool row count.
        """
        pid = os.getpid()
        if getattr(self, "_mmaps_pid", None) != pid:
            self._mmaps = [np.memmap(p, dtype=np.uint8, mode="r") for p in self._bin_paths]
            self._mmaps_pid = pid
        shard_idx, local_idx = self._get_shard(row)
        offsets = self._offsets[shard_idx]
        raw = self._mmaps[shard_idx][int(offsets[local_idx]):int(offsets[local_idx + 1])].copy()
        input_ids = torch.from_numpy(raw).long()
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def pool_lengths(self) -> np.ndarray:
        """Token count of every pool row (cached; main process only)."""
        if self._pool_lengths is None:
            self._pool_lengths = np.concatenate(
                [np.diff(o) for o in self._offsets]).astype(np.int32)
        return self._pool_lengths

    def get_all_sequence_lengths(self) -> np.ndarray:
        """Length of the member each position returns in the current epoch."""
        key = (self._epoch, self.seed)
        if self._lengths_cache is None or self._lengths_cache[0] != key:
            self._lengths_cache = (key, self.pool_lengths()[self.rows_for_epoch()])
        return self._lengths_cache[1]

    @property
    def pool_size(self) -> int:
        return int(self.cum_lengths[-1])

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(_FINGERPRINT_TAG)
        h.update(super().fingerprint().encode())
        h.update((int(self.seed) & MASK64).to_bytes(8, "little"))
        for a in range(0, self._n, VEC_CHUNK):
            h.update(np.asarray(self._tc[a:a + VEC_CHUNK], dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    @property
    def total_tokens(self) -> int:
        """Tokens in the current epoch's stream."""
        return int(self.get_all_sequence_lengths().astype(np.int64).sum())


def open_train_dataset(dataset_dir: Union[str, Path], manifest, seed: int) -> ShardedDataset:
    """The train set a prepared dataset directory describes.

    A manifest with ``cluster_index_dir`` gives a ClusterSampledDataset keyed on the
    training ``seed``, a mixture manifest a MixtureDataset over its sources, any other
    manifest a plain ShardedDataset. Validation sets are
    always plain ShardedDatasets (cluster representatives, never resampled).
    """
    from nanoplm.data.preparation.cluster_config import check_not_experiment_dir

    dataset_dir = Path(dataset_dir)
    if getattr(manifest, "is_mixture", False):
        from nanoplm.data.datasets.mixture import open_mixture
        return open_mixture(dataset_dir, manifest, seed=int(seed))
    check_not_experiment_dir(dataset_dir, manifest)
    train_dir = dataset_dir / manifest.train_dir
    index_dir = getattr(manifest, "cluster_index_dir", None)
    if index_dir:
        return ClusterSampledDataset(train_dir, dataset_dir / index_dir, seed=int(seed))
    return ShardedDataset(data_dir=str(train_dir))
