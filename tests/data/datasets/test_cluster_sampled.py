"""ClusterSampledDataset: real pool rows of the visited train cluster, drawn statelessly.

Ported from the experiment's tools/test_runtime_sampler_patch.py onto a synthetic pool
(tests/data/datasets/cluster_pool.py). Parity is checked against a frozen,
byte-identical copy of the shim (tests/fixtures/cluster_sampling/), imported by path:
agent_space/ is not in git, so the copy keeps the parity test alive after the
experiment directory is gone.
"""
from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np
import pytest
import torch
from scipy.stats import chisquare
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nanoplm.data.datasets import cluster_sampled as C
from nanoplm.data.datasets.cluster_sampled import ClusterSampledDataset
from nanoplm.data.datasets.pretrain import ShardedDataset, TokenPackingDataset

from tests.data.datasets.cluster_pool import (
    RECOVERED_SHIM, SHIM_PATH, SHIM_SHA256, SPECIAL_SIZES, build_pool, load_shim,
)

SEED = 1


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    return build_pool(tmp_path_factory.mktemp("cluster_pool"))


@pytest.fixture(scope="module")
def shim():
    return load_shim()


@pytest.fixture
def ds(pool):
    return ClusterSampledDataset(pool.pool_dir, pool.index_dir, seed=SEED)


@pytest.fixture(scope="module")
def stock_pool(pool):
    return ShardedDataset(data_dir=str(pool.pool_dir))


# ------------------------------------------------------------------ the hash itself


def test_frozen_shim_is_the_recovered_file():
    import hashlib as _h
    assert _h.sha256(SHIM_PATH.read_bytes()).hexdigest() == SHIM_SHA256
    if RECOVERED_SHIM.exists():
        assert RECOVERED_SHIM.read_bytes() == SHIM_PATH.read_bytes()


def test_scalar_hash_matches_vectorized_and_the_shim(shim):
    rng = np.random.default_rng(0)
    cl = rng.integers(0, 2**31 - 1, 3000)
    n = rng.integers(1, 5000, 3000)
    for seed in (0, 1, 7, 42, 2**40 + 3):
        assert C.member_seed_key(seed) == shim.seed_key(seed)
        for epoch in (0, 1, 19, 10**6):
            v = C.member_slots_vec(seed, cl, epoch, n)
            s = np.array([C.member_slot(seed, int(c), epoch, int(k)) for c, k in zip(cl, n)])
            assert np.array_equal(v, s)
            assert np.array_equal(v, shim.member_slots_vec(seed, cl, epoch, n))
            assert (v >= 0).all() and (v < n).all()


def test_seed_key_is_the_training_seed_through_the_shims_derivation(shim):
    # RUNTIME_SAMPLER_MEMBER_SEED=s went through seed_key(s); the pretrain seed goes
    # through the same function, so pretrain seed s reproduces member seed s.
    for s in (0, 1, 42, 12345):
        assert C.member_key(s, 17, 3) == shim.member_key(s, 17, 3)


# ------------------------------------------------------------------ dataset contract


def test_len_is_train_clusters(ds, pool):
    assert len(ds) == len(pool.train_clusters)
    sizes = np.array([ds.cluster_size(i) for i in range(len(ds))])
    assert np.array_equal(sizes, pool.sizes[pool.train_clusters])


def test_bit_identical_to_the_shim(ds, pool, shim):
    old = shim.RuntimeClusterDataset(pool.pool_dir, 0, index_dir=pool.index_dir,
                                     member_seed=SEED)
    for e in (0, 1, 5, 17):
        assert np.array_equal(ds.rows_for_epoch(e), old.rows_for_epoch(e)), e
    for e in (0, 3):
        ds.set_epoch(e)
        old.set_epoch(e)
        assert np.array_equal(ds.get_all_sequence_lengths(), old.get_all_sequence_lengths())
        for i in range(0, len(ds), 97):
            assert ds.pool_row(i) == old.pool_row(i)
            assert torch.equal(ds[i]["input_ids"], old[i]["input_ids"])
    assert ds.fingerprint() == old.fingerprint()
    other = ClusterSampledDataset(pool.pool_dir, pool.index_dir, seed=42)
    old42 = shim.RuntimeClusterDataset(pool.pool_dir, 0, index_dir=pool.index_dir,
                                       member_seed=42)
    assert np.array_equal(other.rows_for_epoch(2), old42.rows_for_epoch(2))


def test_tokens_byte_identical_to_the_claimed_pool_row(ds, pool, stock_pool):
    for e in (0, 4):
        ds.set_epoch(e)
        for i in range(len(ds)):
            row = ds.pool_row(i)
            got, ref = ds[i], stock_pool[row]
            assert torch.equal(got["input_ids"], ref["input_ids"])
            assert torch.equal(got["attention_mask"], ref["attention_mask"])
    # a reference that does not use ShardedDataset: raw bytes at the row's offset
    sro = np.load(pool.index_dir / "shard_row_offsets.npy")
    for row in (0, 1, len(pool.lengths) - 1, 1234):
        s = int(np.searchsorted(sro, row, side="right") - 1)
        off = int(pool.lengths[sro[s]:row].astype(np.int64).sum())
        raw = np.fromfile(pool.pool_dir / f"shard_{s:04d}.bin", dtype=np.uint8,
                          count=int(pool.lengths[row]), offset=off)
        got = ds.read_pool_row(row)["input_ids"].numpy().astype(np.uint8)
        assert np.array_equal(got, raw)


def test_every_visit_lands_in_its_own_train_cluster(ds, pool):
    assert not np.isin(pool.val_clusters, pool.train_clusters).any()
    for e in (0, 1, 17):
        rows = ds.rows_for_epoch(e)
        assert np.array_equal(pool.pool_cluster[rows], pool.train_clusters), e


def test_singletons_are_visited_and_return_their_only_member(ds, pool):
    singles = np.flatnonzero(pool.sizes[pool.train_clusters] == 1)
    assert singles.size > 100
    r0, r9 = ds.rows_for_epoch(0), ds.rows_for_epoch(9)
    assert np.array_equal(r0[singles], r9[singles])
    assert np.array_equal(pool.pool_cluster[r0[singles]], pool.train_clusters[singles])


def test_scalar_path_matches_vector_rows(ds):
    for e in (0, 3):
        rows = ds.rows_for_epoch(e)
        ds.set_epoch(e)
        assert all(ds.pool_row(i) == rows[i] for i in range(len(ds)))


def test_lengths_follow_the_epoch(ds):
    seen = []
    for e in (0, 2):
        ds.set_epoch(e)
        L = ds.get_all_sequence_lengths()
        assert L.shape == (len(ds),) and L.dtype == np.int32
        assert all(int(L[i]) == len(ds[i]["input_ids"]) for i in range(0, len(ds), 7))
        assert ds.total_tokens == int(L.astype(np.int64).sum())
        seen.append(L.copy())
    assert not np.array_equal(seen[0], seen[1])


def test_every_epoch_draws_new_members(ds, pool):
    multi = pool.sizes[pool.train_clusters] > 1
    r = [ds.rows_for_epoch(e)[multi] for e in range(4)]
    for a in range(4):
        for b in range(a + 1, 4):
            assert (r[a] != r[b]).mean() > 0.3


class _Probe(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ids = self.ds[i]["input_ids"].numpy().tobytes()
        return i, self.ds.pool_row(i), hashlib.sha1(ids).hexdigest(), os.getpid()


def _fetch(ds, order, workers, ctx=None):
    ds.cleanup()
    dl = DataLoader(_Probe(ds), batch_size=None, sampler=list(order), num_workers=workers,
                    multiprocessing_context=ctx if workers else None)
    return {int(i): (int(r), h) for i, r, h, _ in dl}


@pytest.mark.parametrize("epoch", [0, 2])
def test_same_member_regardless_of_workers_and_order(ds, epoch):
    pos = np.arange(len(ds))[::3]
    ds.set_epoch(epoch)      # set BEFORE the workers fork: they must inherit it
    truth = ds.rows_for_epoch(epoch)
    rng = np.random.default_rng(epoch)
    runs = [_fetch(ds, pos, 0), _fetch(ds, pos[::-1], 1), _fetch(ds, rng.permutation(pos), 3),
            _fetch(ds, rng.permutation(pos), 4)]
    for got in runs:
        assert got == runs[0]
        assert all(got[int(i)][0] == truth[i] for i in pos)


def test_spawn_workers_rebuild_the_dataset_from_its_pickle(ds):
    pos = np.arange(len(ds))[::11]
    ds.set_epoch(3)
    truth = ds.rows_for_epoch(3)
    got = _fetch(ds, pos, 2, ctx="spawn")
    assert all(got[int(i)][0] == truth[i] for i in pos)


def test_world_size_does_not_enter(ds):
    """DistributedSampler shards positions differently at 2 vs 8 replicas; the member
    for a position must not care which replica fetched it."""
    seen = {}
    ds.set_epoch(1)
    for reps in (2, 8):
        for rank in range(reps):
            s = DistributedSampler(ds, num_replicas=reps, rank=rank, shuffle=True, seed=42)
            s.set_epoch(1)
            for i in s:
                assert seen.setdefault(i, ds.pool_row(i)) == ds.pool_row(i)
    assert len(seen) == len(ds)


def test_different_seed_and_epoch_change_members(ds, pool):
    size = pool.sizes[pool.train_clusters]
    multi = size > 1
    inv = (1.0 / size[multi]).astype(np.float64)
    exp, sd = inv.mean(), np.sqrt((inv * (1 - inv)).sum()) / inv.size
    r0 = ds.rows_for_epoch(0)[multi]
    other = ClusterSampledDataset(pool.pool_dir, pool.index_dir, seed=2)
    for name, r1 in (("seed", other.rows_for_epoch(0)[multi]),
                     ("epoch", ds.rows_for_epoch(1)[multi])):
        agree = float((r0 == r1).mean())
        # P(same member) is 1/n_c per cluster if the keys are independent
        assert abs(agree - exp) < 6 * sd, (name, agree, exp, sd)


@pytest.mark.parametrize("target_n", list(SPECIAL_SIZES) + ["max"])
def test_uniform_within_cluster(ds, pool, target_n):
    sizes = pool.sizes[pool.train_clusters]
    if target_n == "max":
        i = int(np.argmax(sizes))
    else:
        hits = np.flatnonzero(sizes == target_n)
        assert hits.size, target_n
        i = int(hits[0])
    n = int(sizes[i])
    c = int(pool.train_clusters[i])
    draws = min(max(200 * n, 20_000), 100_000)
    ep = C.member_slots_vec(SEED, np.full(draws, c), 0, np.full(draws, n))  # one value
    assert len(set(ep.tolist())) == 1
    over_epochs = np.array([C.member_slot(SEED, c, e, n) for e in range(draws)])
    over_seeds = np.array([C.member_slot(s, c, 0, n) for s in range(draws)])
    for name, x in (("epochs", over_epochs), ("seeds", over_seeds)):
        counts = np.bincount(x, minlength=n)
        assert counts.size == n
        assert chisquare(counts).pvalue > 1e-4, (name, n)


def test_pickle_roundtrip_keeps_runtime_behaviour(ds):
    ds.set_epoch(5)
    d2 = pickle.loads(pickle.dumps(ds))
    assert type(d2) is type(ds) and len(d2) == len(ds) and d2.epoch == 5 and d2.seed == SEED
    assert all(d2.pool_row(i) == ds.pool_row(i) for i in range(len(ds)))
    assert d2.fingerprint() == ds.fingerprint()
    assert len(pickle.dumps(ds)) < 2000   # paths and ints only, no arrays


def test_fingerprint_tracks_seed_and_differs_from_the_stock_pool(ds, pool, stock_pool):
    assert ds.fingerprint() != stock_pool.fingerprint()
    other = ClusterSampledDataset(pool.pool_dir, pool.index_dir, seed=SEED + 1)
    assert other.fingerprint() != ds.fingerprint()
    ds.set_epoch(4)   # the epoch is not part of the dataset identity
    assert ds.fingerprint() == ClusterSampledDataset(
        pool.pool_dir, pool.index_dir, seed=SEED).fingerprint()


def test_index_arrays_are_mmapped(ds):
    for arr in (ds._tc, ds._g2u, ds._off, ds._members):
        assert isinstance(arr, np.memmap)


def test_unverified_or_mismatched_index_is_refused(pool, tmp_path):
    import json
    import shutil
    bad = tmp_path / "ix"
    shutil.copytree(pool.index_dir, bad)
    (bad / "pool_manifest.json").write_text(json.dumps({"verify": {"ok": False}}))
    with pytest.raises(RuntimeError, match="verify.ok"):
        ClusterSampledDataset(pool.pool_dir, bad, seed=SEED)
    (bad / "pool_manifest.json").write_text(json.dumps({"verify": {"ok": True}}))
    sro = np.load(bad / "shard_row_offsets.npy")
    sro[1] += 1
    np.save(bad / "shard_row_offsets.npy", sro)
    with pytest.raises(RuntimeError, match="shard_row_offsets"):
        ClusterSampledDataset(pool.pool_dir, bad, seed=SEED)


# ---------------------------------------------------------- packing, epochs, resume


def _packed(pool, rank=0, world=2, seed=SEED):
    d = ClusterSampledDataset(pool.pool_dir, pool.index_dir, seed=seed)
    sampler = DistributedSampler(d, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    return TokenPackingDataset(d, max_tokens_per_batch=1024, drop_last=True, sampler=sampler)


def _batches(packed, workers):
    dl = DataLoader(packed, batch_size=None, num_workers=workers,
                    collate_fn=lambda b: tuple(hashlib.sha1(s["input_ids"].numpy().tobytes())
                                               .hexdigest() for s in b))
    return list(dl)


def test_packed_length_is_recomputed_for_each_epoch(pool):
    packed = _packed(pool)
    lens = {}
    for e in range(6):
        packed.set_epoch(e)
        lens[e] = len(packed)
        assert lens[e] == len(_batches(packed, 0)), e
    assert len(set(lens.values())) > 1, lens


@pytest.mark.parametrize("workers", [0, 2])
def test_mid_epoch_resume_reproduces_the_rest_of_the_epoch(pool, workers):
    """The pure pipeline resumes by calling set_epoch(resume_epoch) at the top of the
    pass and then discarding the first resume_data_step batches. A fresh dataset
    (a restarted job) must then yield exactly the uninterrupted run's tail."""
    epoch, skip = 3, 5
    full = _packed(pool)
    full.set_epoch(epoch)
    want = _batches(full, workers)
    assert len(want) > skip + 3

    resumed = _packed(pool)            # restarted process: dataset starts at epoch 0
    resumed.set_epoch(epoch)           # set_epoch BEFORE the loader iterates / skips
    got = _batches(resumed, workers)
    assert got[skip:] == want[skip:]

    # the other order (skip on an epoch-0 loader) gives a different stream
    stale = _packed(pool)
    assert _batches(stale, workers)[skip:] != want[skip:]


def test_fast_forward_moves_to_the_requested_epoch(pool):
    packed = _packed(pool)
    consumed = packed.fast_forward(4, epoch=2)
    assert packed.dataset.epoch == 2 and packed.sampler.epoch == 2
    assert consumed > 0


# ------------------------------------------------------------ manifest and selection


def _manifest(**over):
    from nanoplm.data.infra.manifest import PretrainManifest
    kw = dict(pipeline_mode="pretrain", seqs_num=10, min_seq_len=20, max_seq_len=512,
              val_ratio=0.01, train_dir="train", val_dir="val", train_sequences=10,
              val_sequences=1, samples_per_shard=100)
    kw.update(over)
    return PretrainManifest(**kw)


def test_manifest_with_a_cluster_index_selects_the_cluster_dataset(pool, tmp_path):
    from nanoplm.data.datasets import open_train_dataset
    from nanoplm.data.infra.manifest import read_manifest, write_manifest
    root = pool.pool_dir.parent
    m = _manifest(cluster_index_dir="cluster_index", cluster_level="uniref90",
                  train_sequences=len(pool.train_clusters), pool_sequences=len(pool.lengths))
    write_manifest(tmp_path, m)
    back = read_manifest(tmp_path)
    assert back.is_cluster_sampled and back.cluster_index_dir == "cluster_index"
    train = open_train_dataset(root, back, seed=7)
    assert isinstance(train, ClusterSampledDataset) and train.seed == 7
    assert len(train) == back.train_sequences
    plain = open_train_dataset(root, _manifest(), seed=7)
    assert type(plain) is ShardedDataset and len(plain) == len(pool.lengths)


def test_manifests_without_the_cluster_fields_still_load(tmp_path):
    from nanoplm.data.infra.manifest import read_manifest, write_manifest
    write_manifest(tmp_path, _manifest())
    text = (tmp_path / ".data_manifest").read_text()
    assert "cluster_index_dir" not in text   # None fields are not written
    m = read_manifest(tmp_path)
    assert not m.is_cluster_sampled and m.pool_sequences is None


def test_shuffle_off_marks_the_fingerprint_and_survives_pickle(ds, pool, stock_pool):
    off = ClusterSampledDataset(pool.pool_dir, pool.index_dir, seed=SEED, shuffle=False)
    assert off.fingerprint() != ds.fingerprint()
    assert off._bin_paths == ds._bin_paths        # the pool keeps the index's shard order
    assert pickle.loads(pickle.dumps(off)).fingerprint() == off.fingerprint()
    plain_off = ShardedDataset(data_dir=str(pool.pool_dir), shuffle=False)
    assert plain_off.fingerprint() != stock_pool.fingerprint()
    assert pickle.loads(pickle.dumps(plain_off)).fingerprint() == plain_off.fingerprint()


def test_manifest_shuffle_false_reaches_the_train_dataset(pool):
    from nanoplm.data.datasets import open_train_dataset
    root = pool.pool_dir.parent
    on = open_train_dataset(root, _manifest(), seed=7)
    off = open_train_dataset(root, _manifest(shuffle=False), seed=7)
    assert on.shuffle and not off.shuffle
    assert on.fingerprint() != off.fingerprint()
    cl = dict(cluster_index_dir="cluster_index", cluster_level="uniref90",
              train_sequences=len(pool.train_clusters), pool_sequences=len(pool.lengths))
    c_on = open_train_dataset(root, _manifest(**cl), seed=7)
    c_off = open_train_dataset(root, _manifest(shuffle=False, **cl), seed=7)
    assert not c_off.shuffle and c_on.fingerprint() != c_off.fingerprint()
    assert c_on.pool_row(3) == c_off.pool_row(3)  # the member draw does not change


def test_shuffle_off_orders_shards_by_number(tmp_path):
    for k in (2, 10000, 9999, 1):
        np.full(k % 7 + 1, k % 251, dtype=np.uint8).tofile(tmp_path / f"shard_{k:04d}.bin")
        np.save(tmp_path / f"shard_{k:04d}.idx.npy", np.array([k % 7 + 1], dtype=np.int32))
    names = lambda d: [os.path.basename(p) for p in d._bin_paths]
    assert names(ShardedDataset(str(tmp_path), shuffle=False)) == [
        "shard_0001.bin", "shard_0002.bin", "shard_9999.bin", "shard_10000.bin"]
    # shuffled training keeps the name order, so running jobs resume unchanged
    assert names(ShardedDataset(str(tmp_path))) == [
        "shard_0001.bin", "shard_0002.bin", "shard_10000.bin", "shard_9999.bin"]
