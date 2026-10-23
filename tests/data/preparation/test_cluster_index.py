"""cluster_index: the pool <-> hierarchy join, the universe, CSR, holdout and verify
(ported from the experiment's test_build_index.py onto a tiny synthetic release)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from nanoplm.data.datasets.cluster_sampled import ClusterSampledDataset
from nanoplm.data.preparation import cluster_index as ci
from nanoplm.data.preparation import uniref_hierarchy as uh
from nanoplm.data.preparation.filterer import Filterer
from nanoplm.data.sharding.pretrain_shard_writer import ShardWriter
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from tests.data.preparation.uniref_fixture import MAX_LEN, MIN_LEN, make_release

VAL_RATIO = 0.1
SPS = 97


def _pool(root, fasta, level="uniref90", sps=SPS):
    filtered = root / f"{level}_filtered.fasta"
    Filterer(fasta, filtered, MIN_LEN, MAX_LEN, -1).filter()
    ShardWriter(str(filtered), ProtModernBertTokenizer(), MAX_LEN, str(root / "train"),
                samples_per_shard=sps, max_workers=1).create_shards()
    return filtered


def _build(root, release, level="uniref90", **kw):
    fasta = release.fasta90 if level == "uniref90" else release.fasta100
    filtered = _pool(root, fasta, level)
    uh.xml_to_members_tsv(release.xml, root / "members.tsv", level=level)
    kw.setdefault("val_ratio", VAL_RATIO)
    man = ci.build_cluster_index(filtered, root / "train", root / "members.tsv",
                                 root / "cluster_index", MAX_LEN, level=level,
                                 tokenizer=ProtModernBertTokenizer(), **kw)
    return filtered, man


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return make_release(tmp_path_factory.mktemp("release"))


@pytest.fixture(scope="module")
def built(tmp_path_factory, release):
    root = tmp_path_factory.mktemp("built")
    filtered, man = _build(root, release)
    return root, filtered, man


def _ix(root, name):
    return np.load(root / "cluster_index" / f"{name}.npy")


def _pool_accessions(filtered):
    return [line[len(">UniRef90_"):].split()[0] for line in filtered.read_text().splitlines()
            if line.startswith(">")]


def _survives(seq):
    return MIN_LEN <= len(seq) <= MAX_LEN


def test_pool_cluster_matches_ground_truth(built, release):
    root, filtered, _ = built
    accs = _pool_accessions(filtered)
    assert all(_survives(release.u90[a][2]) for a in accs)
    assert len(accs) == sum(_survives(v[2]) for v in release.u90.values())   # filter only
    assert np.array_equal(_ix(root, "pool_cluster"), [release.u90[a][0] for a in accs])
    assert np.array_equal(_ix(root, "pool_is_rep"), [release.u90[a][1] for a in accs])


def test_universe_is_rep_survival_and_sizes_count_survivors(built, release):
    root, _, man = built
    rep_ok = {c for c, is_rep, seq in release.u90.values() if is_rep and _survives(seq)}
    any_ok = {c for c, _, seq in release.u90.values() if _survives(seq)}
    assert set(_ix(root, "universe_clusters").tolist()) == rep_ok
    assert any_ok - rep_ok, "fixture should hold a cluster whose rep did not survive"
    surv = np.bincount([c for c, _, seq in release.u90.values() if _survives(seq)],
                       minlength=len(release.cluster_ids))
    assert np.array_equal(_ix(root, "cluster_sizes_filtered"), surv)
    assert man["counts"]["clusters_in_universe"] == len(rep_ok)


def test_csr_round_trips(built):
    root, _, _ = built
    pc, mem, off = _ix(root, "pool_cluster"), _ix(root, "cluster_members"), _ix(root, "cluster_offsets")
    for u, c in enumerate(_ix(root, "universe_clusters")):
        seg = mem[off[u]:off[u + 1]]
        assert np.array_equal(seg, np.flatnonzero(pc == c))


def test_holdout_is_seeded_whole_cluster_and_not_a_prefix(built):
    root, _, man = built
    universe, val, train = (_ix(root, "universe_clusters"), _ix(root, "val_clusters"),
                            _ix(root, "train_clusters"))
    assert len(val) == round(len(universe) * VAL_RATIO)
    assert np.array_equal(val, universe[ci.draw_val_clusters(len(universe), VAL_RATIO,
                                                             ci.HOLDOUT_SEED)])
    assert np.array_equal(train, np.setdiff1d(universe, val))
    assert not np.isin(val, train).any()
    assert val.max() > universe[len(val) * 2]          # spread, not a prefix
    assert np.array_equal(_ix(root, "val_rep_rows"), _ix(root, "cluster_rep_row")[val])
    assert man["holdout"]["val_ratio"] == VAL_RATIO


def test_manifest_counts_and_verify(built):
    root, _, man = built
    assert man["verify"]["ok"], man["verify"]["failures"]
    assert man["level"] == "uniref90"
    lengths = _ix(root, "pool_lengths")
    assert man["counts"]["pool_rows"] == len(lengths)
    assert man["counts"]["pool_tokens"] == int(lengths.sum())
    assert man["filter"]["token_length_rule"] == f"tokens = min(residues + 1, {MAX_LEN})"
    # expected tokens of one pass: per-cluster mean member length, summed over train
    pc, train = _ix(root, "pool_cluster"), _ix(root, "train_clusters")
    want = sum(lengths[pc == c].mean() for c in train)
    assert abs(man["counts"]["expected_tokens_per_pass"] - want) <= 1


def test_dataset_opens_the_built_index(built):
    root, _, _ = built
    ds = ClusterSampledDataset(root / "train", root / "cluster_index", seed=5)
    pc = _ix(root, "pool_cluster")
    for e in (0, 1):
        assert np.array_equal(pc[ds.rows_for_epoch(e)], _ix(root, "train_clusters"))


def _copy(built, tmp_path):
    root = built[0]
    dst = tmp_path / "copy"
    shutil.copytree(root / "cluster_index", dst / "cluster_index")
    (dst / "train").symlink_to(root / "train", target_is_directory=True)
    return dst


@pytest.mark.parametrize("tamper,check", [
    ("csr_swap", "csr_member_cluster_agrees_with_pool_cluster"),
    ("pool_lengths", "pool_lengths_is_the_concatenated_shard_index"),
    ("train_holds_val", "train_is_universe_minus_val"),
])
def test_verify_rejects_a_tampered_index(built, tmp_path, tamper, check):
    d = _copy(built, tmp_path)
    ix = d / "cluster_index"
    if tamper == "csr_swap":
        mem, off = np.load(ix / "cluster_members.npy"), np.load(ix / "cluster_offsets.npy")
        a, b = int(off[0]), int(off[-2])
        mem[[a, b]] = mem[[b, a]]
        np.save(ix / "cluster_members.npy", mem)
    elif tamper == "pool_lengths":
        L = np.load(ix / "pool_lengths.npy")
        L[3] += 1
        np.save(ix / "pool_lengths.npy", L)
    else:
        np.save(ix / "train_clusters.npy", np.sort(np.concatenate(
            [np.load(ix / "train_clusters.npy"), np.load(ix / "val_clusters.npy")[:1]])))
    res = ci.verify_cluster_index(ix, d / "train")
    assert not res["ok"] and check in res["failures"]


def test_join_refuses_an_accession_outside_the_hierarchy(release, tmp_path):
    fasta = tmp_path / "extra.fasta"
    fasta.write_text(release.fasta90.read_text() + ">UniRef90_ZZZ9999999 stray\n" + "A" * 40 + "\n")
    filtered = _pool(tmp_path, fasta)
    uh.xml_to_members_tsv(release.xml, tmp_path / "members.tsv")
    with pytest.raises(ci.ClusterIndexError, match="pool sequences are not in"):
        ci.build_cluster_index(filtered, tmp_path / "train", tmp_path / "members.tsv",
                               tmp_path / "cluster_index", MAX_LEN, VAL_RATIO)
    # through the stage, the error names the persisted XML and how to refresh it
    from nanoplm.data.preparation.cluster_pipeline import build_index_and_val
    from nanoplm.data.preparation.cluster_table import build_members_table
    build_members_table("uniref_xml", tmp_path / "members.tsv", fasta=release.fasta90,
                        level="uniref90", xml=release.xml)
    with pytest.raises(ci.ClusterIndexError,
                       match=f"{release.xml} is kept between runs .* delete it to re-download"):
        build_index_and_val(filtered, tmp_path / "train", tmp_path / "members.tsv",
                            tmp_path / "cluster_index", tmp_path / "val", MAX_LEN, VAL_RATIO, 50,
                            cluster_xml=release.xml)


def test_a_fasta_that_is_not_the_pools_order_is_fatal(built, release, tmp_path):
    root, filtered, _ = built
    recs = filtered.read_text().split(">")[1:]
    # rotating the records keeps the set but breaks the row order
    other = tmp_path / "rotated.fasta"
    other.write_text("".join(">" + r for r in recs[1:] + recs[:1]))
    with pytest.raises(ci.ClusterIndexError, match="not the same rows"):
        ci.build_cluster_index(other, root / "train", root / "members.tsv",
                               tmp_path / "ix", MAX_LEN, VAL_RATIO)


def test_decode_check_catches_an_equal_length_swap(built, tmp_path):
    root, filtered, _ = built
    recs = filtered.read_text().split(">")[1:]
    seqs = [r.split("\n", 1)[1].replace("\n", "") for r in recs]
    lens = np.array([len(s) for s in seqs])
    # swap the sequences of two equal-length records: the length rule cannot see it
    pairs = [(i, j) for i in range(len(seqs)) for j in range(i + 1, len(seqs))
             if lens[i] == lens[j] and seqs[i] != seqs[j]]
    i, j = pairs[0]
    seqs[i], seqs[j] = seqs[j], seqs[i]
    swapped = tmp_path / "swapped.fasta"
    swapped.write_text("".join(f">{r.split(chr(10), 1)[0]}\n{s}\n" for r, s in zip(recs, seqs)))
    old = ci.DECODE_SPOT_ROWS
    try:
        ci.DECODE_SPOT_ROWS = 10 ** 6       # decode every row of the tiny pool
        with pytest.raises(ci.ClusterIndexError, match="not the tokenization"):
            ci.build_cluster_index(swapped, root / "train", root / "members.tsv",
                                   tmp_path / "ix", MAX_LEN, VAL_RATIO,
                                   tokenizer=ProtModernBertTokenizer())
    finally:
        ci.DECODE_SPOT_ROWS = old


def test_max_clusters_caps_the_universe_with_a_seeded_subset(release, tmp_path):
    full_root, cap_root = tmp_path / "full", tmp_path / "cap"
    _build(full_root, release)
    _, man = _build(cap_root, release, max_clusters=100)
    full = np.load(full_root / "cluster_index" / "universe_clusters.npy")
    capped = np.load(cap_root / "cluster_index" / "universe_clusters.npy")
    assert len(capped) == 100 and np.isin(capped, full).all()
    assert capped.max() > full[150]                       # not a prefix
    assert man["cap"]["applied"] and man["cap"]["clusters_before_cap"] == len(full)
    assert len(np.load(cap_root / "cluster_index" / "val_clusters.npy")) == 10


def test_uniref100_pool_shares_the_uniref90_universe_and_split(built, release, tmp_path):
    _, man = _build(tmp_path, release, level="uniref100")
    assert man["verify"]["ok"] and man["level"] == "uniref100"
    rep_ok = {c for c, is_rep, seq in release.u100.values() if is_rep and _survives(seq)}
    assert set(np.load(tmp_path / "cluster_index" / "universe_clusters.npy").tolist()) == rep_ok
    root90 = built[0]
    for name in ("universe_clusters", "train_clusters", "val_clusters"):
        assert np.array_equal(np.load(tmp_path / "cluster_index" / f"{name}.npy"),
                              _ix(root90, name)), name


def test_row_gather_raises_a_too_small_samples_per_shard(built, tmp_path, monkeypatch):
    from nanoplm.data.datasets.pretrain import ShardedDataset
    root = built[0]
    rows = _ix(root, "val_rep_rows")
    cliff = len(list((root / "train").glob("*.bin"))) + 1    # the pool itself still fits
    assert len(rows) >= cliff
    monkeypatch.setattr(ci, "SHARD_SORT_CLIFF", cliff)
    ci.write_row_gather(root / "train", rows, tmp_path / "val", samples_per_shard=1)
    assert len(list((tmp_path / "val").glob("*.bin"))) < cliff
    assert len(ShardedDataset(str(tmp_path / "val"))) == len(rows)


def test_shard_count_over_the_sort_cliff_is_refused(built, monkeypatch):
    root = built[0]
    monkeypatch.setattr(ci, "SHARD_SORT_CLIFF", 3)
    with pytest.raises(ci.ClusterIndexError, match="Raise samples_per_shard"):
        ci.read_pool_lengths(root / "train")


def test_row_gather_keeps_the_given_order(built, tmp_path):
    from nanoplm.data.datasets.pretrain import ShardedDataset
    import torch
    root = built[0]
    rows = _ix(root, "val_rep_rows")[::-1]
    ci.write_row_gather(root / "train", rows, tmp_path / "val", samples_per_shard=7)
    val = ShardedDataset(str(tmp_path / "val"))
    pool = ShardedDataset(str(root / "train"))
    assert len(val) == len(rows)
    for i, r in enumerate(rows):
        assert torch.equal(val[i]["input_ids"], pool[int(r)]["input_ids"])


def test_level_detection(release, tmp_path):
    assert ci.detect_uniref_level(release.fasta90) == "uniref90"
    assert ci.detect_uniref_level(release.fasta100) == "uniref100"
    u50 = tmp_path / "u50.fasta"
    u50.write_text(">UniRef50_A0A1 x\nMKV\n")
    assert ci.detect_uniref_level(u50) == "uniref50"
    other = tmp_path / "o.fasta"
    other.write_text(">sp|P12345|X\nMKV\n")
    with pytest.raises(ValueError, match="not a UniRef header"):
        ci.detect_uniref_level(other)
