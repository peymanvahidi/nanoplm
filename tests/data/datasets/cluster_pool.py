"""A small synthetic pool + cluster index, written directly in the on-disk format.

Built without the data-prep code, so the dataset tests do not depend on it. The
layout matches what ``nanoplm.data.preparation.cluster_index`` writes and what the
frozen shim reads.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SHIM_PATH = REPO / "tests/fixtures/cluster_sampling/runtime_sampler_patch.py"
SHIM_SHA256 = "7c992ed2e962829e7bd4cf2df85768a8c8b60eb5c53f307c32492992377599a1"
RECOVERED_SHIM = (REPO / "agent_space/data-sampler-recovered/scratchpad-recovered/shim"
                  / "runtime_sampler_patch.py")

# cluster sizes the uniformity test needs, plus a tail of small clusters
SPECIAL_SIZES = (2, 3, 7, 16, 50, 300)


def load_shim():
    """Import the frozen shim by path. RUNTIME_SAMPLER is unset, so it installs nothing."""
    assert hashlib.sha256(SHIM_PATH.read_bytes()).hexdigest() == SHIM_SHA256
    spec = importlib.util.spec_from_file_location("frozen_runtime_sampler_patch", SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Pool:
    pool_dir: Path
    index_dir: Path
    pool_cluster: np.ndarray      # pool row -> global cluster
    sizes: np.ndarray             # global cluster -> n_c (0 outside the universe)
    train_clusters: np.ndarray
    val_clusters: np.ndarray
    lengths: np.ndarray           # pool row -> tokens


def _save(path: Path, arr) -> None:
    with open(path, "wb") as f:
        np.save(f, arr)


def build_pool(root: Path, seed: int = 0, n_small: int = 1500, n_shards: int = 6) -> Pool:
    """Write pool shards and a verified index under ``root``.

    Clusters: ``n_small`` with 1..5 members (about half singletons), one of each size
    in SPECIAL_SIZES, and a few clusters that are NOT in the universe (their rep did
    not survive). Rows are scattered over the pool so no cluster is contiguous.
    """
    rng = np.random.default_rng(seed)
    small = np.where(rng.random(n_small) < 0.5, 1, rng.integers(2, 6, n_small))
    sizes_u = np.concatenate([small, SPECIAL_SIZES]).astype(np.int64)
    rng.shuffle(sizes_u)
    n_outside = 40
    n_global = len(sizes_u) + n_outside
    # global cluster ids: universe clusters are a random subset of 0..n_global-1
    universe = np.sort(rng.choice(n_global, size=len(sizes_u), replace=False)).astype(np.int32)
    outside = np.setdiff1d(np.arange(n_global, dtype=np.int32), universe)
    outside_sizes = rng.integers(1, 4, n_outside)
    labels = np.concatenate([np.repeat(universe, sizes_u), np.repeat(outside, outside_sizes)])
    labels = labels[rng.permutation(len(labels))].astype(np.int32)
    n_rows = len(labels)

    # uneven shard sizes, so no row -> shard arithmetic can assume full shards
    cuts = np.sort(rng.choice(np.arange(1, n_rows), size=n_shards - 1, replace=False))
    rows_per_shard = np.diff(np.concatenate([[0], cuts, [n_rows]])).tolist()
    lengths = rng.integers(21, 120, n_rows).astype(np.int32)

    pool_dir = root / "train"
    index_dir = root / "cluster_index"
    pool_dir.mkdir(parents=True)
    index_dir.mkdir()
    r0 = 0
    for s, k in enumerate(rows_per_shard):
        ls = lengths[r0:r0 + k]
        toks = rng.integers(4, 25, int(ls.sum()), dtype=np.uint8)
        # last token of every row is EOS (1), so rows differ in more than length
        ends = np.cumsum(ls) - 1
        toks[ends] = 1
        (pool_dir / f"shard_{s:04d}.bin").write_bytes(toks.tobytes())
        _save(pool_dir / f"shard_{s:04d}.idx.npy", ls)
        r0 += k

    sizes = np.bincount(labels, minlength=n_global).astype(np.int32)
    in_u = np.zeros(n_global, dtype=bool)
    in_u[universe] = True
    g2u = np.full(n_global, -1, dtype=np.int32)
    g2u[universe] = np.arange(len(universe), dtype=np.int32)
    sel = np.flatnonzero(in_u[labels])
    order = np.argsort(g2u[labels[sel]], kind="stable")
    members = sel[order].astype(np.int32)
    n_u = sizes[universe].astype(np.int64)
    offsets = np.zeros(len(universe) + 1, dtype=np.int64)
    np.cumsum(n_u, out=offsets[1:])
    # hold out 60 small clusters; the SPECIAL_SIZES ones stay in train for the tests
    small_local = np.flatnonzero(n_u <= 5)
    val_local = np.sort(rng.choice(small_local, size=60, replace=False))
    in_val = np.zeros(len(universe), dtype=bool)
    in_val[val_local] = True
    train = universe[~in_val]
    val = universe[in_val]
    sro = np.zeros(len(rows_per_shard) + 1, dtype=np.int64)
    np.cumsum(rows_per_shard, out=sro[1:])

    for name, arr in [("train_clusters.npy", train), ("val_clusters.npy", val),
                      ("universe_clusters.npy", universe), ("global_to_universe.npy", g2u),
                      ("cluster_offsets.npy", offsets), ("cluster_members.npy", members),
                      ("cluster_sizes_filtered.npy", sizes), ("pool_cluster.npy", labels),
                      ("shard_row_offsets.npy", sro)]:
        _save(index_dir / name, arr)
    (index_dir / "pool_manifest.json").write_text(json.dumps({"verify": {"ok": True}}))
    sizes_out = sizes.copy()
    sizes_out[~in_u] = 0
    return Pool(pool_dir, index_dir, labels, sizes_out, train, val, lengths)
