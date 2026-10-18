"""cluster_pipeline: end-to-end prep of a cluster-sampled dataset from a tiny release."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nanoplm.data.datasets import ClusterSampledDataset, ShardedDataset, open_train_dataset
from nanoplm.data.infra.manifest import read_manifest
from nanoplm.data.infra.validation import validate_pretrain_dataset
from nanoplm.data.preparation import cluster_pipeline as cp
from tests.data.preparation.uniref_fixture import MAX_LEN, MIN_LEN, make_release


def _cfg(root, release, level="uniref90", **kw):
    fasta = release.fasta90 if level == "uniref90" else release.fasta100
    base = dict(members_fasta=fasta, cluster_xml=release.xml, output_dir=root / "out",
                validation_dir=root / "validation", filtered_fasta=root / "filtered.fasta",
                min_seq_len=MIN_LEN, max_seq_len=MAX_LEN, val_ratio=0.1,
                samples_per_shard=101, max_workers=1, cluster_source="uniref_xml",
                level=level)
    base.update(kw)
    return cp.ClusterDataConfig(**base)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return make_release(tmp_path_factory.mktemp("release"))


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, release):
    root = tmp_path_factory.mktemp("prep")
    cfg = _cfg(root, release)
    return cfg, cp.prepare_cluster_dataset(cfg)


def test_layout_and_manifest(prepared):
    cfg, m = prepared
    out = cfg.output_dir
    assert (out / "train").is_dir() and (out / "val").is_dir()
    assert (out / "cluster_index" / "index" / "pool_manifest.json").exists()
    assert (out / "cluster_index" / "members.tsv").exists()
    back = read_manifest(out)
    assert back == m
    assert back.cluster_index_dir == "cluster_index/index" and back.cluster_level == "uniref90"
    train = np.load(out / "cluster_index" / "index" / "train_clusters.npy")
    val = np.load(out / "cluster_index" / "index" / "val_clusters.npy")
    assert back.train_sequences == len(train) and back.val_sequences == len(val)
    assert back.seqs_num == len(train) + len(val)
    assert back.pool_sequences == len(ShardedDataset(str(out / "train")))
    assert back.total_train_tokens < back.pool_tokens     # one member per cluster
    validate_pretrain_dataset(out)


def test_training_picks_the_cluster_dataset_and_val_holds_the_reps(prepared):
    cfg, m = prepared
    out = cfg.output_dir
    train = open_train_dataset(out, m, seed=3)
    assert isinstance(train, ClusterSampledDataset) and len(train) == m.train_sequences
    val = ShardedDataset(str(out / "val"))
    pool = ShardedDataset(str(out / "train"))
    reps = np.load(out / "cluster_index" / "index" / "val_rep_rows.npy")
    assert len(val) == m.val_sequences == len(reps)
    for i in range(len(val)):
        assert torch.equal(val[i]["input_ids"], pool[int(reps[i])]["input_ids"])
    # no train visit can land in a val cluster
    pc = np.load(out / "cluster_index" / "index" / "pool_cluster.npy")
    vc = np.load(out / "cluster_index" / "index" / "val_clusters.npy")
    for e in range(3):
        assert not np.isin(pc[train.rows_for_epoch(e)], vc).any()
    nonrep = ShardedDataset(str(out / "cluster_index" / "index" / "val_nonrep"))
    assert len(nonrep) == len(np.load(out / "cluster_index" / "index" / "val_nonrep_rows.npy"))


def test_pool_keeps_fasta_order_and_the_length_window(prepared, release):
    cfg, _ = prepared
    accs = [ln[len(">UniRef90_"):].split()[0] for ln in cfg.filtered_fasta.read_text().splitlines()
            if ln.startswith(">")]
    original = [ln[len(">UniRef90_"):].split()[0] for ln in release.fasta90.read_text().splitlines()
                if ln.startswith(">")]
    kept = [a for a in original if MIN_LEN <= len(release.u90[a][2]) <= MAX_LEN]
    assert accs == kept                                # filtered, never shuffled


def test_uniref100_input(release, tmp_path):
    m = cp.prepare_cluster_dataset(_cfg(tmp_path, release, level="uniref100"))
    assert m.cluster_level == "uniref100"


def test_seqs_num_caps_clusters(release, tmp_path):
    m = cp.prepare_cluster_dataset(_cfg(tmp_path, release, seqs_num=120))
    assert m.seqs_num == 120 and m.val_sequences == 12 and m.train_sequences == 108


def test_too_small_samples_per_shard_is_raised(release, tmp_path, monkeypatch):
    monkeypatch.setattr(cp.ci, "SHARD_SORT_CLIFF", 5)
    m = cp.prepare_cluster_dataset(_cfg(tmp_path, release, samples_per_shard=10))
    rows = m.pool_sequences
    assert -(-rows // 10) >= 5                           # 10 per shard would be refused
    assert m.samples_per_shard == cp.ci.shard_size_for(rows, 10) > 10
    shards = sorted((tmp_path / "out" / "train").glob("*.bin"))
    assert len(shards) == -(-rows // m.samples_per_shard) < 5


@pytest.mark.parametrize("rows, asked, used", [
    (97_800_000, 2000, 10_000), (97_800_000, 30_000, 30_000),
    (169_000_000, 2000, 20_000), (9_999, 1, 1), (10_000, 1, 2), (500, 2000, 2000)])
def test_shard_size_for(rows, asked, used):
    assert cp.ci.shard_size_for(rows, asked) == used
    assert -(-rows // used) < cp.ci.SHARD_SORT_CLIFF


def test_invalid_sequences_are_dropped_from_the_pool(release, tmp_path):
    fasta = tmp_path / "with_invalid.fasta"
    lines = release.fasta90.read_text().split("\n")
    # the first record that would otherwise survive the length filter
    k, acc = next((k, ln[len(">UniRef90_"):].split()[0]) for k, ln in enumerate(lines)
                  if ln.startswith(">") and
                  MIN_LEN <= len(release.u90[ln[len(">UniRef90_"):].split()[0]][2]) <= MAX_LEN)
    lines[k + 1] = "MK!V" + lines[k + 1][4:]            # an illegal character
    fasta.write_text("\n".join(lines))
    cfg = _cfg(tmp_path, release, members_fasta=fasta, drop_invalid_sequences=True)
    cp.prepare_cluster_dataset(cfg)
    assert f">UniRef90_{acc} " not in cfg.filtered_fasta.read_text()




def test_programmatic_drop_invalid_default_matches_the_template():
    import yaml
    from nanoplm.cli.templates import load_template
    tpl = yaml.safe_load(load_template("data_pretrain"))
    want = tpl["data_params"]["drop_invalid_sequences"]
    assert cp.ClusterDataConfig.__dataclass_fields__["drop_invalid_sequences"].default == want
    tpl["data_params"].pop("drop_invalid_sequences")
    tpl["cluster_subsampling"].update(enabled=True, members="uniref90")
    tpl["data_dirs"]["url"] = cp.cc.UNIPROT_FASTA["uniref50"]
    cfg, _ = cp.config_from_yaml(tpl)
    assert cfg.drop_invalid_sequences == want
