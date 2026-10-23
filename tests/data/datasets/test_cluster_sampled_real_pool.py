"""Read-only check against the experiment's real UniRef90 pool and index.

Skipped unless the recovered build exists at agent_space/data-sampler-recovered/output
(97.8M pool rows in 3,261 shards, 32.8M train clusters). Slow: --run-slow.
Peak RAM is a few GB (the shim keeps cluster_members in memory).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanoplm.data.datasets.cluster_sampled import ClusterSampledDataset
from nanoplm.data.preparation import cluster_index as ci
from nanoplm.data.preparation import uniref_hierarchy as uh
from tests.data.datasets.cluster_pool import REPO, load_shim

OUT = REPO / "agent_space/data-sampler-recovered/output"
POOL = OUT / "pool/shards"
INDEX = OUT / "pool/index"
HIERARCHY = OUT / "hierarchy"
N_TRAIN = 32_795_962
N_ITEMS = 3000
MEMBER_SEED = 1     # what the experiment's runtime runs used (RUNTIME_SAMPLER_MEMBER_SEED)

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (POOL.is_dir() and INDEX.is_dir()),
                                 reason=f"recovered pool not found at {OUT}")]


@pytest.fixture(scope="module")
def pair():
    shim = load_shim()
    new = ClusterSampledDataset(POOL, INDEX, seed=MEMBER_SEED)
    old = shim.RuntimeClusterDataset(POOL, 0, index_dir=INDEX, member_seed=MEMBER_SEED)
    yield new, old
    new.cleanup()
    old.cleanup()


def test_opens_the_real_pool(pair):
    new, old = pair
    assert len(new) == len(old) == N_TRAIN
    assert new.pool_size == 97_803_490


@pytest.mark.parametrize("epoch", [0, 1])
def test_same_stream_as_the_shim(pair, epoch):
    new, old = pair
    assert np.array_equal(new.rows_for_epoch(epoch), old.rows_for_epoch(epoch))
    new.set_epoch(epoch)
    old.set_epoch(epoch)
    for i in range(N_ITEMS):
        assert new.pool_row(i) == old.pool_row(i)
    got = [new[i]["input_ids"] for i in range(N_ITEMS)]
    new.cleanup()            # 3,261 memmaps per dataset against an 8,192 fd limit
    for i in range(N_ITEMS):
        assert torch.equal(got[i], old[i]["input_ids"]), i
    old.cleanup()


def test_lengths_and_fingerprint_match_the_shim(pair):
    new, old = pair
    for e in (0, 1):
        new.set_epoch(e)
        old.set_epoch(e)
        assert np.array_equal(new.get_all_sequence_lengths(), old.get_all_sequence_lengths())
    assert new.fingerprint() == old.fingerprint()


def test_holdout_draw_reproduces_the_experiments_val_clusters():
    man = json.loads((INDEX / "pool_manifest.json").read_text())
    universe = np.load(INDEX / "universe_clusters.npy", mmap_mode="r")
    val = np.load(INDEX / "val_clusters.npy")
    ho = man["holdout"]
    assert ho["seed"] == ci.HOLDOUT_SEED
    local = ci.draw_val_clusters(len(universe), ho["val_ratio"], ho["seed"])
    assert np.array_equal(np.asarray(universe[local]), val)


def test_the_hash_matches_the_real_hierarchy():
    stats = json.loads((HIERARCHY / "hierarchy_stats.json").read_text())
    for acc, want in stats["hash"]["test_vectors"].items():
        assert f"0x{uh.hash_one(acc.encode()):016x}" == want
    keys = np.load(HIERARCHY / "u90_keys.npy", mmap_mode="r")
    # the representative of UniRef50 cluster 0 is in the key set under our hash
    first_acc = (HIERARCHY / "cluster_ids.txt").open("rb").readline().strip()
    k = uh.hash_one(first_acc)
    pos = int(np.searchsorted(keys, k))
    assert pos < len(keys) and int(keys[pos]) == k


XML = Path("/e/project1/.snapshots/weekly-20260927/profound/peyman/nanoplm/agent_space/"
           "wt-data-sampler/experiments/data-sampler/output/raw/uniref50.xml.gz")
FASTA = OUT / "pool/uniref90_filtered.fasta"


@pytest.mark.skipif(not (XML.exists() and FASTA.exists()), reason="uniref50.xml.gz not found")
def test_xml_members_table_rebuilds_the_experiments_index(tmp_path):
    """uniref50.xml -> members.tsv -> index must equal the experiment's index byte for byte
    (about 15 min, ~5 GB of members.tsv under tmp_path)."""
    from nanoplm.data.preparation import cluster_index as ci
    tsv = tmp_path / "members.tsv"
    uh.xml_to_members_tsv(XML, tsv, "uniref90")
    ci.build_cluster_index(FASTA, POOL, tsv, tmp_path / "index", 512, 0.0044,
                           level="uniref90")
    for name in ("pool_cluster", "pool_is_rep", "cluster_sizes_filtered", "universe_clusters",
                 "global_to_universe", "cluster_offsets", "cluster_members", "cluster_rep_row",
                 "universe_mask", "val_clusters", "val_rep_rows", "train_clusters",
                 "shard_row_offsets"):
        a = np.load(INDEX / f"{name}.npy", mmap_mode="r")
        b = np.load(tmp_path / "index" / f"{name}.npy", mmap_mode="r")
        assert a.dtype == b.dtype and np.array_equal(a, b), name
