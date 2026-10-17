"""Runtime cluster-aware sampler (loaderbench approach B), injected without touching src/.

    RUNTIME_SAMPLER=1 RUNTIME_SAMPLER_TRAIN_DIR=<dataset_dir>/train \
    RUNTIME_SAMPLER_INDEX=output/pool/index RUNTIME_SAMPLER_MEMBER_SEED=1 \
    python -c "import ds_sampler_patch; from nanoplm.cli import cli; \
               import loaderbench_probe, runtime_sampler_patch; cli()" pretrain from-yaml ...

WHAT. One dataset item is one visit to a TRAIN UniRef50 cluster: len == #train clusters
(32,795,962) and the stock DistributedSampler permutes cluster positions each epoch. For
position i, cluster c = train_clusters[i] and the member is

    slot = member_key(member_seed, c, epoch) % n_c      (splitmix64 chain, stateless)
    row  = cluster_members[cluster_offsets[g2u[c]] + slot]

and the item is pool row `row`, read with the same memmap slice ShardedDataset uses, so
the bytes are the pool row's bytes. The key is a pure function of (seed, cluster, epoch):
no RNG state, so it does not depend on num_workers, fetch order, world size or resume.

HOW IT IS INJECTED. pure_pipeline.run_pure_pretraining builds the train set with the module
global `ShardedDataset(data_dir=<dataset_dir>/train)` (pure_pipeline.py:2368 at 87fc66f) and
the val set the same way on the next line. This module rebinds that ONE global to a factory
that returns RuntimeClusterDataset for the configured train dir and a stock ShardedDataset
for anything else (val stays stock). The stock class itself is not modified.

THE CONTRACTS THE PIPELINE RELIES ON:
  * dataset_dir holds train/ (-> the pool shards), val/ (-> val_rep), and a .data_manifest
    written by `nanoplm data write-manifest`, so validate_pretrain_dataset and read_manifest
    run unchanged over real files. The manifest's train_sequences counts pool rows (what is
    in train/); the dataset exposes only train clusters.
  * get_all_sequence_lengths() -> the length of the member __getitem__ returns at the
    current epoch, per position. TokenPackingDataset.__len__ / fast_forward use it.
  * set_epoch(e): TokenPackingDataset.set_epoch forwards to dataset.set_epoch before
    iter(train_loader); packing runs non-persistent fork workers, so every worker is forked
    after set_epoch and inherits the epoch. Each worker prints its epoch in its
    RUNTIME_SAMPLER line, so the log proves it.
  * fingerprint() hashes the stock pool fingerprint plus the sampler identity.

Every process prints: one `RUNTIME_SAMPLER pid=... role=init` line when the dataset is
built (each rank's main process) and one `RUNTIME_SAMPLER pid=... role=fetch` line on its
first __getitem__ (each DataLoader worker). No such lines = not patched.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from nanoplm.data.datasets.pretrain import ShardedDataset

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB
# b"RTMEMB": separates this key stream from anything else keyed on the same seed
MEMBER_TAG = 0x52_54_4D_45_4D_42
VEC_CHUNK = 1 << 22

MARKER = "RUNTIME_SAMPLER_REQUIRED"


# ------------------------------------------------------------------ the stateless draw


def _sm64(z: int) -> int:
    """splitmix64 output function applied to z + gamma (one full avalanche)."""
    z = (z + GAMMA) & MASK64
    z = ((z ^ (z >> 30)) * C1) & MASK64
    z = ((z ^ (z >> 27)) * C2) & MASK64
    return z ^ (z >> 31)


def seed_key(seed: int) -> int:
    return _sm64((int(seed) ^ MEMBER_TAG) & MASK64)


def member_key(seed: int, cluster: int, epoch: int) -> int:
    return _sm64(_sm64(seed_key(seed) ^ int(cluster)) ^ int(epoch))


def member_slot(seed: int, cluster: int, epoch: int, n: int) -> int:
    return member_key(seed, cluster, epoch) % int(n)


def _sm64_vec(z: np.ndarray) -> np.ndarray:
    z = z + np.uint64(GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(C1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(C2)
    return z ^ (z >> np.uint64(31))


def member_slots_vec(seed: int, clusters: np.ndarray, epoch: int, n: np.ndarray) -> np.ndarray:
    """Vectorized member_slot; identical values (tools/test_runtime_sampler_patch.py)."""
    k0 = np.uint64(seed_key(seed))
    with np.errstate(over="ignore"):
        z = _sm64_vec(np.asarray(clusters).astype(np.uint64) ^ k0)
        z = _sm64_vec(z ^ np.uint64(int(epoch)))
    return (z % np.asarray(n).astype(np.uint64)).astype(np.int64)


# ------------------------------------------------------------------------ the dataset


class RuntimeClusterDataset(ShardedDataset):
    """ShardedDataset over the pool shards, indexed by TRAIN cluster, member drawn at fetch."""

    def __init__(self, data_dir, pad_token_id: int = 0, *, index_dir, member_seed: int,
                 epoch: int = 0):
        self._rt_init(str(data_dir), int(pad_token_id), str(index_dir), int(member_seed),
                      int(epoch), log=True)

    # ---- init -------------------------------------------------------------------
    def _rt_init(self, data_dir, pad_token_id, index_dir, member_seed, epoch, log):
        t0 = time.perf_counter()
        # stock init over the pool shards: _bin_paths, per-shard _offsets, cum_lengths
        self._init_dataset(data_dir, pad_token_id=pad_token_id, log=log)
        self.index_dir = Path(index_dir)
        self.member_seed = member_seed
        self._epoch = epoch
        ix = self.index_dir
        man = json.loads((ix / "pool_manifest.json").read_text())
        if man.get("verify", {}).get("ok") is not True:
            raise RuntimeError(f"{ix}/pool_manifest.json: verify.ok is not True")
        # global row id == pool row id: same shard list, same per-shard row counts
        sro = np.load(ix / "shard_row_offsets.npy")
        if len(self._bin_paths) != len(sro) - 1 or not np.array_equal(
                np.asarray(self.cum_lengths, dtype=np.int64), sro[1:]):
            raise RuntimeError(
                f"{data_dir}: {len(self._bin_paths)} shards / {int(self.cum_lengths[-1])} rows "
                f"do not match {ix}/shard_row_offsets.npy ({len(sro) - 1} shards, "
                f"{int(sro[-1])} rows); train/ must be the pool the index was built on")
        tc = np.load(ix / "train_clusters.npy")
        g2u = np.load(ix / "global_to_universe.npy", mmap_mode="r")
        off = np.load(ix / "cluster_offsets.npy", mmap_mode="r")
        uloc = np.asarray(g2u[tc], dtype=np.int64)
        if uloc.min() < 0:
            raise RuntimeError("train_clusters.npy holds a cluster outside the universe")
        start = np.asarray(off[uloc], dtype=np.int64)
        size = np.asarray(off[uloc + 1], dtype=np.int64) - start
        del uloc
        members = np.load(ix / "cluster_members.npy")        # int32 pool rows, in RAM
        if size.min() < 1 or int((start + size).max()) > len(members):
            raise RuntimeError("cluster CSR is inconsistent with cluster_members.npy")
        if int(members.max()) >= int(self.cum_lengths[-1]):
            raise RuntimeError("cluster_members.npy points past the pool")
        self._tc = tc.astype(np.int64)      # int64 so int(...) needs no dtype games
        self._start = start
        self._size = size.astype(np.int32)
        self._members = members
        self._n = len(tc)
        self._rt_logged_pid = None
        if log:
            rank = os.environ.get("RANK", "?")
            print(f"RUNTIME_SAMPLER pid={os.getpid()} role=init rank={rank} "
                  f"n_train_clusters={self._n:,} pool_rows={int(self.cum_lengths[-1]):,} "
                  f"members={len(members):,} seed={member_seed} epoch={epoch} "
                  f"index={ix} build_s={time.perf_counter() - t0:.1f}",
                  file=sys.stderr, flush=True)

    # ---- pickling (spawn/forkserver workers; fork does not use it) ---------------
    def __getstate__(self):
        return {"data_dir": str(self.data_dir), "pad_token_id": self.pad_token_id,
                "index_dir": str(self.index_dir), "member_seed": self.member_seed,
                "epoch": self._epoch}

    def __setstate__(self, state):
        self._rt_init(state["data_dir"], state["pad_token_id"], state["index_dir"],
                      state["member_seed"], state["epoch"], log=False)

    # ---- the contract -----------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def pool_row(self, idx: int) -> int:
        """Pool row visited at position idx for the current epoch."""
        c = int(self._tc[idx])
        slot = member_slot(self.member_seed, c, self._epoch, int(self._size[idx]))
        return int(self._members[int(self._start[idx]) + slot])

    def rows_for_epoch(self, epoch: int | None = None) -> np.ndarray:
        e = self._epoch if epoch is None else int(epoch)
        out = np.empty(self._n, dtype=np.int64)
        for a in range(0, self._n, VEC_CHUNK):
            b = min(a + VEC_CHUNK, self._n)
            slot = member_slots_vec(self.member_seed, self._tc[a:b], e, self._size[a:b])
            out[a:b] = self._members[self._start[a:b] + slot]
        return out

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of bounds for {self._n} train clusters")
        row = self.pool_row(idx)
        pid = os.getpid()
        if self._rt_logged_pid != pid:
            self._rt_logged_pid = pid
            wi = torch.utils.data.get_worker_info()
            print(f"RUNTIME_SAMPLER pid={pid} role=fetch rank={os.environ.get('RANK', '?')} "
                  f"worker={wi.id if wi is not None else '-'} epoch={self._epoch} "
                  f"seed={self.member_seed} first_idx={idx} cluster={int(self._tc[idx])} "
                  f"n_c={int(self._size[idx])} row={row}", file=sys.stderr, flush=True)
        return self.read_pool_row(row)

    def read_pool_row(self, row: int):
        """Stock ShardedDataset.__getitem__ body, addressed by POOL row."""
        pid = os.getpid()
        if getattr(self, "_mmaps_pid", None) != pid:
            self._mmaps = [np.memmap(p, dtype=np.uint8, mode="r") for p in self._bin_paths]
            self._mmaps_pid = pid
        shard_idx, local_idx = self._get_shard(row)
        offsets = self._offsets[shard_idx]
        start = int(offsets[local_idx])
        end = int(offsets[local_idx + 1])
        raw = self._mmaps[shard_idx][start:end].copy()
        input_ids = torch.from_numpy(raw).long()
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_all_sequence_lengths(self) -> np.ndarray:
        pool_len = np.concatenate([np.diff(o) for o in self._offsets]).astype(np.int32)
        return pool_len[self.rows_for_epoch()]

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(b"runtime-cluster-sampler-v1")
        h.update(super().fingerprint().encode())
        h.update(int(self.member_seed).to_bytes(8, "little"))
        h.update(self._tc.tobytes())
        return h.hexdigest()[:16]

    @property
    def total_tokens(self) -> int:
        return int(self.get_all_sequence_lengths().astype(np.int64).sum())



# ---------------------------------------------------------------------- installation


def install(train_dir: str, index_dir: str, member_seed: int) -> None:
    import nanoplm.pretraining.pure_pipeline as pp
    if pp.ShardedDataset is not ShardedDataset:
        raise RuntimeError(f"pure_pipeline.ShardedDataset is {pp.ShardedDataset!r}, not the "
                           "stock class; refusing to stack patches")
    target = Path(train_dir).resolve()

    def runtime_or_stock(data_dir, pad_token_id: int = 0):
        p = Path(data_dir)
        if p.resolve() == target:
            return RuntimeClusterDataset(data_dir, pad_token_id, index_dir=index_dir,
                                         member_seed=member_seed)
        if p.name == "train" and (p.parent / MARKER).exists():
            raise RuntimeError(f"{p.parent} is a runtime-sampler dataset dir but {p} did not "
                               f"match RUNTIME_SAMPLER_TRAIN_DIR={train_dir}")
        return ShardedDataset(data_dir=data_dir, pad_token_id=pad_token_id)

    pp.ShardedDataset = runtime_or_stock
    if os.environ.get("RANK", "0") == "0":
        print(f"RUNTIME_SAMPLER_PATCH installed: pure_pipeline.ShardedDataset -> factory "
              f"(train_dir={train_dir} index={index_dir} member_seed={member_seed})",
              file=sys.stderr, flush=True)


if os.environ.get("RUNTIME_SAMPLER") == "1":
    install(os.environ["RUNTIME_SAMPLER_TRAIN_DIR"], os.environ["RUNTIME_SAMPLER_INDEX"],
            int(os.environ.get("RUNTIME_SAMPLER_MEMBER_SEED", "1")))
