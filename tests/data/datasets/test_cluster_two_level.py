"""Two-level member draw: UniRef50 cluster -> sub-cluster (uniform) -> member (uniform).

The index gets its sub-cluster arrays from ``cluster_pool.write_subclusters``, built by
brute force from a pool row -> sub-cluster map; an index without them keeps the
uniform draw bit for bit.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil

import numpy as np
import pytest
import torch
from scipy.stats import chi2_contingency, chisquare
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nanoplm.data.datasets import cluster_sampled as C
from nanoplm.data.datasets.cluster_sampled import ClusterSampledDataset
from nanoplm.data.datasets.pretrain import ShardedDataset, TokenPackingDataset

from tests.data.datasets.cluster_pool import build_pool, random_subclusters, write_subclusters

SEED = 1


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    return build_pool(tmp_path_factory.mktemp("two_level_pool"))


def _copy_index(pool, dest):
    shutil.copytree(pool.index_dir, dest)
    return dest


def _cluster_of_size(pool, n):
    """(train position, global cluster, its rows ascending) of the first size-n cluster."""
    i = int(np.flatnonzero(pool.sizes[pool.train_clusters] == n)[0])
    c = int(pool.train_clusters[i])
    return i, c, np.flatnonzero(pool.pool_cluster == c)


@pytest.fixture(scope="module")
def skewed(pool, tmp_path_factory):
    """Two-level index where the 50-member cluster is split 49 + 1 and the 300-member one
    200 + 99 + 1; every other member is its own sub-cluster."""
    ix = _copy_index(pool, tmp_path_factory.mktemp("skewed") / "ix")
    sub = np.arange(len(pool.pool_cluster), dtype=np.int64)  # all singletons
    _, _, r50 = _cluster_of_size(pool, 50)
    _, _, r300 = _cluster_of_size(pool, 300)
    sub[r50[:49]] = -1
    sub[r300[:200]] = -2
    sub[r300[200:299]] = -3
    write_subclusters(ix, sub)
    return ix


@pytest.fixture(scope="module")
def random_ix(pool, tmp_path_factory):
    ix = _copy_index(pool, tmp_path_factory.mktemp("random") / "ix")
    write_subclusters(ix, random_subclusters(pool, seed=4))
    return ix


@pytest.fixture
def ds2(pool, random_ix):
    return ClusterSampledDataset(pool.pool_dir, random_ix, seed=SEED)


# ------------------------------------------------------------------ the distribution


def test_uniform_over_sub_clusters_then_inside_one(pool, skewed):
    ds = ClusterSampledDataset(pool.pool_dir, skewed, seed=SEED)
    assert ds.cluster_draw == "two_level"
    for n, parts in ((50, (49, 1)), (300, (200, 99, 1))):
        i, c, rows = _cluster_of_size(pool, n)
        E = 30_000
        by_pass = ds.rows_at(np.full(E, i), np.arange(E))
        by_seed = []
        for s in range(6000):
            ds.seed = s
            by_seed.append(ds._draw_row(c, 0))
        ds.seed = SEED
        bounds = np.cumsum((0,) + parts)
        for name, r in (("passes", by_pass), ("seeds", np.asarray(by_seed))):
            assert np.isin(r, rows).all()
            slot = np.searchsorted(rows, r)
            sub = np.searchsorted(bounds, slot, side="right") - 1
            counts = np.bincount(sub, minlength=len(parts))
            # each sub-cluster equally likely, the single member included (~1/k, not 1/n)
            assert chisquare(counts).pvalue > 1e-4, (n, name, counts)
            big = slot[sub == 0]
            assert chisquare(np.bincount(big, minlength=parts[0])).pvalue > 1e-4, (n, name)
        assert abs((by_pass == rows[-1]).mean() - 1 / len(parts)) < 0.02


def test_all_singleton_sub_clusters_give_todays_rows(pool, tmp_path):
    ix = _copy_index(pool, tmp_path / "ix")
    write_subclusters(ix, np.arange(len(pool.pool_cluster)))
    uni = ClusterSampledDataset(pool.pool_dir, pool.index_dir, seed=SEED)
    two = ClusterSampledDataset(pool.pool_dir, ix, seed=SEED)
    assert two.cluster_draw == "two_level" and uni.cluster_draw == "uniform"
    assert np.array_equal(np.load(ix / "sub_members.npy"), np.load(ix / "cluster_members.npy"))
    for seed in (SEED, 42, 2**40 + 3):
        uni.seed = two.seed = seed
        for e in (0, 1, 7, 10**6):
            assert np.array_equal(two.rows_for_epoch(e), uni.rows_for_epoch(e)), (seed, e)
        assert all(two._draw_row(c, 3) == uni._draw_row(c, 3) for c in pool.train_clusters[:300])


def test_every_visit_lands_in_its_own_cluster_and_sub_cluster(pool, ds2):
    sub = random_subclusters(pool, seed=4)
    multi = pool.sizes[pool.train_clusters] > 1
    seen = [ds2.rows_for_epoch(e) for e in range(4)]
    for rows in seen:
        assert np.array_equal(pool.pool_cluster[rows], pool.train_clusters)
    assert (seen[0][multi] != seen[1][multi]).mean() > 0.2
    assert len(np.unique(sub[np.concatenate(seen)])) > len(pool.train_clusters)


# Square clusters: equal-size sub-clusters whose size shares a factor with their count.
# There a second draw that reuses z (slot = z % size) is tied to j = z % n_sub (with
# n_sub == size, slot == j always) and only n_sub of the n_sub * size members are drawn.
SQUARES = ((4, 4), (6, 6))


@pytest.fixture(scope="module")
def square(tmp_path_factory):
    """A pool whose 16- and 36-member clusters are split 4 x 4 and 6 x 6; every other
    member is its own sub-cluster. Returns (pool, index dir)."""
    root = tmp_path_factory.mktemp("square")
    pool = build_pool(root, seed=2, n_small=200, n_shards=3,
                      special_sizes=tuple(k * m for k, m in SQUARES))
    sub = np.arange(len(pool.pool_cluster), dtype=np.int64)
    for t, (k, m) in enumerate(SQUARES):
        _, _, rows = _cluster_of_size(pool, k * m)
        sub[rows] = -1000 * (t + 1) - np.arange(k * m) // m
    write_subclusters(pool.index_dir, sub)
    return pool, pool.index_dir


def _cells(ds, c):
    """pool row -> (sub-cluster index j, slot inside it) for global cluster c."""
    u = int(ds._g2u[c])
    s0, s1 = int(ds._cso[u]), int(ds._cso[u + 1])
    out = {}
    for j in range(s1 - s0):
        a, b = int(ds._so[s0 + j]), int(ds._so[s0 + j + 1])
        for slot in range(b - a):
            out[int(ds._sub_members[a + slot])] = (j, slot)
    return out


def test_square_sub_clusters_draw_every_member_uniformly(square):
    pool, ix = square
    ds = ClusterSampledDataset(pool.pool_dir, ix, seed=SEED)
    E = 20_000
    for k, m in SQUARES:
        i, c, rows = _cluster_of_size(pool, k * m)
        cells = _cells(ds, c)
        assert sorted(cells) == rows.tolist() and {jm for jm in cells.values()} == {
            (j, s) for j in range(k) for s in range(m)}
        table = np.zeros((k, m), dtype=np.int64)
        for seed in (SEED, 7, 2**40 + 3):
            ds.seed = seed
            drawn = ds.rows_at(np.full(E, i), np.arange(E))
            # every member is reachable under every seed, not just n_sub of them
            assert len(np.unique(drawn)) == k * m, (k, m, seed)
            for r in drawn.tolist():
                table[cells[r]] += 1
        ds.seed = SEED
        # uniform over all k*m cells (expected 3750 / 1667 per cell)
        assert chisquare(table.ravel()).pvalue > 1e-6, (k, m, table)
        mean = table.mean()
        assert (np.abs(table - mean) < 0.12 * mean).all(), (k, m, table)
        # level 1 (j) independent of level 2 (slot): the (j, slot) table has no structure
        assert chi2_contingency(table).pvalue > 1e-6, (k, m, table)
        # in particular slot == j is no likelier than any other diagonal
        assert abs(np.trace(table) / table.sum() - 1 / m) < 0.02, (k, m)


# ------------------------------------------------------------------ an independent oracle

_M64 = (1 << 64) - 1


def _sm64(z):
    z = (z + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _oracle_row(ix, seed, cluster, pass_idx):
    """The two-level draw written out from the spec, on the index's own arrays:
    z = sm(sm(key(seed) ^ c) ^ pass); j = z % n_sub; slot = sm(z ^ SUB_TAG) % size_j."""
    g2u = np.load(ix / "global_to_universe.npy")
    cso = np.load(ix / "cluster_sub_offsets.npy").astype(np.int64)
    so = np.load(ix / "sub_offsets.npy").astype(np.int64)
    sm = np.load(ix / "sub_members.npy")
    key = _sm64((int(seed) ^ 0x52544D454D42) & _M64)            # b"RTMEMB"
    z = _sm64(_sm64(key ^ int(cluster)) ^ int(pass_idx))
    u = int(g2u[cluster])
    j = int(cso[u]) + z % int(cso[u + 1] - cso[u])
    slot = _sm64(z ^ 0x525453554243) % int(so[j + 1] - so[j])   # b"RTSUBC"
    return int(sm[so[j] + slot])


ORACLE_PASSES = (0, 1, 7, 2**40)
ORACLE_SEEDS = (SEED, 42, 2**40 + 3)


@pytest.mark.parametrize("which", ["square", "random"])
def test_draw_matches_an_independent_oracle(request, pool, which):
    if which == "square":
        p, ix = request.getfixturevalue("square")
        clusters = [_cluster_of_size(p, k * m)[1] for k, m in SQUARES]
    else:
        p, ix = pool, request.getfixturevalue("random_ix")
        cso = np.diff(np.load(ix / "cluster_sub_offsets.npy").astype(np.int64))
        g2u = np.load(ix / "global_to_universe.npy")
        n_sub = cso[g2u[p.train_clusters]]
        sizes = p.sizes[p.train_clusters]
        pick = np.flatnonzero((n_sub >= 2) & (sizes > n_sub))[:8]
        assert len(pick) == 8
        clusters = [int(c) for c in p.train_clusters[pick]]
        clusters.append(_cluster_of_size(p, 300)[1])
        clusters.append(int(p.train_clusters[np.flatnonzero(sizes == 1)[0]]))
    pos = np.array([int(np.flatnonzero(p.train_clusters == c)[0]) for c in clusters])
    tc = np.asarray(clusters)
    ds = ClusterSampledDataset(p.pool_dir, ix, seed=SEED)
    assert ds.cluster_draw == "two_level"
    stock = ShardedDataset(data_dir=str(p.pool_dir))
    for seed in ORACLE_SEEDS:
        ds.seed = seed
        for e in ORACLE_PASSES:
            want = np.array([_oracle_row(ix, seed, c, e) for c in clusters])
            assert [ds._draw_row(c, e) for c in clusters] == want.tolist(), (seed, e)
            assert np.array_equal(ds._draw_rows(tc, e), want), (seed, e)
            assert np.array_equal(ds._draw_rows(tc, np.full(len(tc), e)), want), (seed, e)
            assert np.array_equal(ds.rows_at(pos, e), want), (seed, e)
            for q, r in zip(pos.tolist(), want.tolist()):
                assert torch.equal(ds.item_at(q, e)["input_ids"], stock[r]["input_ids"])
        # mixed passes in one call
        ep = np.resize(np.asarray(ORACLE_PASSES, dtype=np.int64), len(tc))
        want = [_oracle_row(ix, seed, c, int(e)) for c, e in zip(clusters, ep)]
        assert ds._draw_rows(tc, ep).tolist() == want
        assert ds.rows_at(pos, ep).tolist() == want


# ------------------------------------------------------------------ the paths agree


def test_every_path_agrees_on_a_two_level_index(pool, ds2):
    stock = ShardedDataset(data_dir=str(pool.pool_dir))
    lengths = ds2.pool_lengths()
    idx = np.arange(len(ds2))
    for e in (0, 3):
        rows = ds2.rows_for_epoch(e)
        assert np.array_equal(ds2.rows_at(idx, e), rows)
        assert np.array_equal(ds2.rows_at(idx, np.full(len(idx), e)), rows)
        assert np.array_equal(ds2.lengths_at(idx, np.full(len(idx), e)), lengths[rows])
        ds2.set_epoch(e)
        assert np.array_equal(ds2.get_all_sequence_lengths(), lengths[rows])
        for i in range(0, len(ds2), 13):
            assert ds2.pool_row(i) == rows[i] == ds2._draw_row(pool.train_clusters[i], e)
            got = ds2[i]["input_ids"]
            assert torch.equal(got, ds2.item_at(i, e)["input_ids"])
            assert torch.equal(got, stock[int(rows[i])]["input_ids"])
    # mixed passes in one call (the mixture's use)
    ep = np.arange(len(idx)) % 5
    want = np.array([ds2._draw_row(pool.train_clusters[i], int(ep[i])) for i in idx])
    assert np.array_equal(ds2.rows_at(idx, ep), want)
    assert np.array_equal(ds2.lengths_at(idx, ep), lengths[want])


def test_sub_arrays_are_mmapped(ds2):
    for arr in (ds2._sub_members, ds2._so, ds2._cso):
        assert isinstance(arr, np.memmap)


# ------------------------------------------------------------------ the fingerprint


def _todays_fingerprint(ds):
    h = hashlib.sha256()
    h.update(b"runtime-cluster-sampler-v1")
    h.update(ShardedDataset.fingerprint(ds).encode())
    h.update(int(ds.seed).to_bytes(8, "little"))
    h.update(np.asarray(ds._tc, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def test_fingerprints_separate_the_index_kinds_and_the_sub_cluster_maps(pool, tmp_path, ds2):
    uni = ClusterSampledDataset(pool.pool_dir, pool.index_dir, seed=SEED)
    assert uni.cluster_draw == "uniform"
    assert uni.fingerprint() == _todays_fingerprint(uni)
    single = _copy_index(pool, tmp_path / "single")
    write_subclusters(single, np.arange(len(pool.pool_cluster)))
    other = _copy_index(pool, tmp_path / "other")
    write_subclusters(other, random_subclusters(pool, seed=9))
    fps = {uni.fingerprint(), ds2.fingerprint(),
           ClusterSampledDataset(pool.pool_dir, single, seed=SEED).fingerprint(),
           ClusterSampledDataset(pool.pool_dir, other, seed=SEED).fingerprint()}
    assert len(fps) == 4
    ds2.set_epoch(3)
    assert ds2.fingerprint() == ClusterSampledDataset(
        pool.pool_dir, ds2.index_dir, seed=SEED).fingerprint()


# ------------------------------------------------------------------ the warning


@pytest.mark.parametrize("level,source,sub,warns", [
    ("uniref100", "uniref50_xml", False, True),
    ("uniref90", "uniref50_xml", False, False),
    ("uniref100", "tsv", False, False),
    (None, None, False, False),
    ("uniref100", "uniref50_xml", True, False),
])
def test_warning_only_when_two_levels_were_possible(pool, tmp_path, monkeypatch,
                                                    level, source, sub, warns):
    ix = _copy_index(pool, tmp_path / "ds" / "ix")
    (tmp_path / "ds" / "data.yaml").write_text("x: 1\n")
    man = json.loads((ix / "pool_manifest.json").read_text())
    man.update(level=level, clusters={"source": source})
    (ix / "pool_manifest.json").write_text(json.dumps(man))
    if sub:
        write_subclusters(ix, random_subclusters(pool))
    said = []
    monkeypatch.setattr(C.logger, "warning", lambda msg, *a: said.append(msg))
    monkeypatch.delenv("RANK", raising=False)
    ds = ClusterSampledDataset(pool.pool_dir, ix, seed=SEED)
    if warns:
        assert len(said) == 1
        assert "no UniRef90 sub-clusters" in said[0] and "uniformly" in said[0]
        assert f"from-yaml data.yaml in {(tmp_path / 'ds').resolve()}" in said[0]
    else:
        assert said == []
    pickle.loads(pickle.dumps(ds))               # a worker re-opening it stays quiet
    monkeypatch.setenv("RANK", "1")
    ClusterSampledDataset(pool.pool_dir, ix, seed=SEED)
    assert len(said) == int(warns)


# ------------------------------------------------------------------ _check_index


def _tamper(pool, tmp_path, how):
    ix = _copy_index(pool, tmp_path / "ix")
    write_subclusters(ix, random_subclusters(pool, seed=4))
    so = np.load(ix / "sub_offsets.npy")
    cso = np.load(ix / "cluster_sub_offsets.npy")
    sm = np.load(ix / "sub_members.npy")
    man = json.loads((ix / "pool_manifest.json").read_text())
    if how == "file_missing":
        (ix / "sub_offsets.npy").unlink()
    elif how == "no_block":
        man.pop("subclusters")
    elif how == "block_without_files":
        for n in C.SUB_ARRAYS:
            (ix / f"{n}.npy").unlink()
    elif how == "no_digest":
        man["subclusters"] = {}
    elif how == "short_members":
        np.save(ix / "sub_members.npy", sm[:-1])
    elif how == "cso_length":
        np.save(ix / "cluster_sub_offsets.npy", np.append(cso, cso[-1]))
    elif how == "so_end":
        so[-1] -= 1
        np.save(ix / "sub_offsets.npy", so)
    elif how == "so_start":
        so[0] = 1
        np.save(ix / "sub_offsets.npy", so)
    elif how == "cso_end":
        cso[-1] -= 1
        np.save(ix / "cluster_sub_offsets.npy", cso)
    elif how == "empty_sub":
        k = int(np.flatnonzero(np.diff(so.astype(np.int64)) > 1)[0])
        so[k + 1] = so[k]
        np.save(ix / "sub_offsets.npy", so)
    elif how == "member_outside":
        sm[3] = len(pool.lengths)
        np.save(ix / "sub_members.npy", sm)
    elif how == "train_cluster_without_sub":
        u = int(np.load(ix / "global_to_universe.npy")[pool.train_clusters[0]])
        cso[u + 1] = cso[u]                     # its sub-clusters handed to the next one
        np.save(ix / "cluster_sub_offsets.npy", cso)
    elif how == "not_nested":
        g2u = np.load(ix / "global_to_universe.npy")
        n_sub = np.diff(cso.astype(np.int64))
        u = next(int(g2u[c]) for c in pool.train_clusters if n_sub[g2u[c] + 1] >= 2)
        cso[u + 1] += 1                         # steals the next cluster's first sub-cluster
        np.save(ix / "cluster_sub_offsets.npy", cso)
    elif how == "float_offsets":
        np.save(ix / "sub_offsets.npy", so.astype(np.float64))
    (ix / "pool_manifest.json").write_text(json.dumps(man))
    return ix


@pytest.mark.parametrize("how,match", [
    ("file_missing", "sub_offsets.npy is missing"),
    ("no_block", "no 'subclusters' block"),
    ("block_without_files", "is missing"),
    ("no_digest", "no digest"),
    ("short_members", "sub_members has"),
    ("cso_length", "cluster_sub_offsets has"),
    ("so_end", "sub_offsets must run"),
    ("so_start", "sub_offsets must run"),
    ("cso_end", "cluster_sub_offsets must run"),
    ("empty_sub", "empty sub-cluster"),
    ("member_outside", "outside the pool"),
    ("train_cluster_without_sub", "has no sub-cluster"),
    ("not_nested", "do not span"),
    ("float_offsets", "integer"),
])
def test_check_index_rejects_inconsistent_sub_arrays(pool, tmp_path, how, match):
    ix = _tamper(pool, tmp_path, how)
    with pytest.raises(RuntimeError, match=match):
        ClusterSampledDataset(pool.pool_dir, ix, seed=SEED)


# ------------------------------------------------------------------ workers


class _Probe(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return i, self.ds.pool_row(i), self.ds.cluster_draw


@pytest.mark.parametrize("ctx", ["fork", "spawn"])
def test_workers_draw_the_same_rows(ds2, ctx):
    ds2.set_epoch(3)
    truth = ds2.rows_for_epoch(3)
    pos = np.arange(len(ds2))[::7]
    ds2.cleanup()
    dl = DataLoader(_Probe(ds2), batch_size=None, sampler=list(pos), num_workers=2,
                    multiprocessing_context=ctx)
    got = list(dl)
    assert len(got) == len(pos)
    assert all(r == truth[i] and draw == "two_level" for i, r, draw in got)


def test_pickle_roundtrip_reopens_the_sub_arrays(ds2):
    ds2.set_epoch(2)
    d = pickle.loads(pickle.dumps(ds2))
    assert d.cluster_draw == "two_level" and isinstance(d._sub_members, np.memmap)
    assert np.array_equal(d.rows_for_epoch(), ds2.rows_for_epoch())
    assert d.fingerprint() == ds2.fingerprint()
    assert len(pickle.dumps(ds2)) < 2000


# ------------------------------------------------------------------ packing and mixtures


def test_packed_length_matches_the_batches(pool, random_ix):
    d = ClusterSampledDataset(pool.pool_dir, random_ix, seed=SEED)
    sampler = DistributedSampler(d, num_replicas=2, rank=1, shuffle=True, seed=SEED)
    packed = TokenPackingDataset(d, max_tokens_per_batch=1024, drop_last=True, sampler=sampler)
    for e in range(3):
        packed.set_epoch(e)
        assert len(packed) == len(list(DataLoader(packed, batch_size=None,
                                                  collate_fn=lambda b: len(b)))), e


def test_mixture_over_a_two_level_source(tmp_path):
    from nanoplm.data.datasets import open_train_dataset
    from nanoplm.data.infra import mixture as mx
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.datasets.mixture_sources import cluster_source, plain_source

    a = plain_source(tmp_path / "a", 300, tag=5, seed=1)
    c, _ = cluster_source(tmp_path / "c", seed=3, two_level=True)
    mx.write_mixture(tmp_path, [("a", a, 0.6, "a.fasta"), ("c", c, 0.4, "c.fasta")])
    mix = open_train_dataset(tmp_path, read_manifest(tmp_path), seed=5)
    assert mix.datasets[1].cluster_draw == "two_level"
    for e in (0, 3):
        mix.set_epoch(e)
        idx = np.arange(len(mix))
        got = mix.lengths_at_positions(idx)
        assert np.array_equal(got, [len(mix[i]["input_ids"]) for i in idx])
        sampler = DistributedSampler(mix, num_replicas=2, rank=0, shuffle=True, seed=5)
        packed = TokenPackingDataset(mix, max_tokens_per_batch=512, drop_last=True,
                                     sampler=sampler, num_workers=2)
        packed.set_epoch(e)
        n = len(packed)
        assert n == len(list(DataLoader(packed, batch_size=None, num_workers=2,
                                         collate_fn=lambda b: len(b)))) > 0
