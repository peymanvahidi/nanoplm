"""Mixtures: the mixture manifest, its hash check, MixtureDataset and the run plan."""
from __future__ import annotations

import hashlib
import pickle
from collections import Counter

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nanoplm.data.datasets import open_train_dataset
from nanoplm.data.datasets.cluster_sampled import ClusterSampledDataset
from nanoplm.data.datasets.mixture import MixtureDataset, allocate, pass_permutation
from nanoplm.data.datasets.pretrain import ShardedDataset, TokenPackingDataset
from nanoplm.data.infra import mixture as mx
from nanoplm.data.infra.manifest import MixtureManifest, read_manifest
from nanoplm.data.infra.validation import validate_pretrain_dataset

from tests.data.datasets.mixture_sources import cluster_source, plain_source

SEED = 5


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A mixture folder with two plain sources and a cluster source under it."""
    root = tmp_path_factory.mktemp("mix")
    a = plain_source(root / "a", 300, tag=5, seed=1)
    b = plain_source(root / "b", 40, tag=6, seed=2)
    c, pool = cluster_source(root / "c", seed=3)
    return root, {"a": a, "b": b, "c": c}, pool


def _write(root, folders, weights):
    mx.write_mixture(root, [(n, folders[n], w, f"{n}.fasta") for n, w in weights.items()])
    return read_manifest(root)


def _key(sample) -> str:
    return hashlib.sha1(sample["input_ids"].numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------ the manifest


def test_mixture_manifest_records_every_source(built):
    root, folders, _ = built
    m = _write(root, folders, {"a": 0.5, "b": 0.2, "c": 0.3})
    assert isinstance(m, MixtureManifest) and m.is_mixture
    rows = {r["name"]: r for r in m.sources}
    assert list(rows) == ["a", "b", "c"]
    assert rows["a"]["dir"] == "a" and rows["a"]["kind"] == "plain"
    assert rows["c"]["kind"] == "cluster" and rows["c"]["pool_sequences"] > 0
    assert rows["b"]["weight"] == 0.2 and rows["b"]["fasta"] == "b.fasta"
    assert rows["a"]["manifest_sha256"] == mx.manifest_hash(folders["a"])
    for k in ("train_sequences", "val_sequences", "total_train_tokens", "max_seq_len",
              "min_seq_len", "register_tokens"):
        assert k in rows["a"]
    assert m.max_seq_len == 128            # the largest over the sources
    assert m.train_sequences == sum(r["train_sequences"] for r in m.sources)
    assert m.num_register_tokens == 0


def test_a_single_dataset_manifest_loads_unchanged(built):
    _, folders, _ = built
    m = read_manifest(folders["a"])
    assert type(m).__name__ == "PretrainManifest" and not m.is_mixture
    assert type(open_train_dataset(folders["a"], m, seed=1)) is ShardedDataset


def test_weights_that_do_not_sum_to_1_name_each_weight_and_the_sum(built, tmp_path):
    _, folders, _ = built
    with pytest.raises(mx.MixtureError,
                       match=r"must sum to 1 .*: 0\.2 \(a\) \+ 0\.7 \(b\) = 0\.9"):
        mx.write_mixture(tmp_path, [("a", folders["a"], 0.2, ""), ("b", folders["b"], 0.7, "")])


def test_weights_within_the_tolerance_pass():
    assert mx.weights_error({"a": 0.1, "b": 0.2, "c": 0.7}) is None
    assert "above 0" in mx.weights_error({"a": 0.0, "b": 1.0})


def test_register_tokens_must_agree(built, tmp_path):
    _, folders, _ = built
    other = plain_source(tmp_path / "r", 20, tag=7, register_tokens=1)
    with pytest.raises(mx.MixtureError, match="same register_tokens; got a 0, r 1"):
        mx.write_mixture(tmp_path / "m", [("a", folders["a"], 0.5, ""), ("r", other, 0.5, "")])


def test_a_mixture_is_not_a_source(built, tmp_path):
    root, folders, _ = built
    _write(root, folders, {"a": 0.5, "b": 0.5})
    with pytest.raises(mx.MixtureError, match="is itself a mixture"):
        mx.write_mixture(tmp_path, [("m", root, 0.5, ""), ("a", folders["a"], 0.5, "")])


def test_training_refuses_a_source_rebuilt_after_the_mix(tmp_path):
    a = plain_source(tmp_path / "a", 30, tag=5)
    b = plain_source(tmp_path / "b", 30, tag=6)
    m = _write(tmp_path, {"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    validate_pretrain_dataset(tmp_path)               # fine as written
    plain_source(tmp_path / "b", 31, tag=6)          # b rebuilt: other counts
    for call in (lambda: open_train_dataset(tmp_path, m, seed=1),
                 lambda: validate_pretrain_dataset(tmp_path)):
        with pytest.raises(mx.MixtureError,
                           match=r"mixture source b: .*changed after the mixture was written"
                                 r".*Rerun `nanoplm data from-yaml`"):
            call()


def test_prepared_source_outside_the_mix_is_stored_absolute(built, tmp_path):
    _, folders, _ = built
    m = _write(tmp_path, folders, {"a": 0.5, "b": 0.5})
    assert m.sources[0]["dir"] == str(folders["a"].resolve())
    assert len(open_train_dataset(tmp_path, m, seed=1)) == 340


def test_eval_sources():
    m = MixtureManifest(pipeline_mode="pretrain", seqs_num=1, min_seq_len=1, max_seq_len=8,
                        val_ratio=0.1, train_dir="", val_dir="", train_sequences=1,
                        val_sequences=1, samples_per_shard=1,
                        sources=[{"name": "x"}, {"name": "y"}])
    assert mx.resolve_eval_sources("all", m) == ["x", "y"]
    assert mx.resolve_eval_sources(["y", "x"], m) == ["x", "y"]
    with pytest.raises(mx.MixtureError, match=r"unknown source z \(the mixture has x, y\)"):
        mx.resolve_eval_sources(["x", "z"], m)
    with pytest.raises(mx.MixtureError, match="empty"):
        mx.resolve_eval_sources([], m)
    from nanoplm.data.infra.manifest import PretrainManifest
    plain = PretrainManifest(pipeline_mode="pretrain", seqs_num=1, min_seq_len=1, max_seq_len=8,
                             val_ratio=0.1, train_dir="t", val_dir="v", train_sequences=1,
                             val_sequences=1, samples_per_shard=1)
    assert mx.resolve_eval_sources("all", plain) == []
    with pytest.raises(mx.MixtureError, match="not a mixture"):
        mx.resolve_eval_sources(["x"], plain)


# ------------------------------------------------------------------ MixtureDataset


@pytest.fixture
def mix(built):
    root, folders, _ = built
    return open_train_dataset(root, _write(root, folders, {"a": 0.5, "b": 0.2, "c": 0.3}),
                              seed=SEED)


def test_allocate_is_exact_and_sums():
    assert allocate([0.1, 0.2, 0.7], 10) == [1, 2, 7]
    assert sum(allocate([1 / 3] * 3, 100)) == 100
    assert allocate([0.999, 0.001], 10) == [9, 1]     # every source gets a position


def test_pass_permutation_is_a_permutation():
    for n in (1, 2, 7, 64, 97, 1000):
        for p in range(3):
            a, b = pass_permutation(1, "x", p, n)
            assert sorted((a * u + b) % n for u in range(n)) == list(range(n))
    assert pass_permutation(1, "x", 0, 1000) != pass_permutation(1, "x", 1, 1000)
    assert pass_permutation(1, "x", 0, 1000) != pass_permutation(1, "y", 0, 1000)


def test_exact_shares_in_every_epoch_and_over_a_long_window(mix):
    n = len(mix)
    assert n == sum(mix.sizes)
    want = [round(w * n) for w in (0.5, 0.2, 0.3)]
    assert mix.counts == want
    total = Counter()
    sampler = DistributedSampler(mix, num_replicas=1, rank=0, shuffle=True, seed=SEED)
    for e in range(7):
        mix.set_epoch(e)
        sampler.set_epoch(e)
        got = Counter(mix.locate(i)[0] for i in sampler)
        assert [got[s] for s in range(3)] == want
        total.update(got)
    assert [total[s] / (7 * n) for s in range(3)] == pytest.approx([0.5, 0.2, 0.3], abs=1e-3)


def test_each_source_pass_reads_every_item_once(mix):
    """b (40 items) repeats ~2.9x per mixture epoch, a (300) takes half a pass: either
    way pass p of a source is a permutation of its items."""
    seen = {0: {}, 1: {}, 2: {}}
    for e in range(6):
        for i in range(len(mix)):
            s, j, p = mix.locate(i, e)
            seen[s].setdefault(p, []).append(j)
    for s, passes in seen.items():
        last = max(passes)
        for p, items in passes.items():
            if p < last:
                assert sorted(items) == list(range(mix.sizes[s])), (s, p)
    assert max(seen[1]) == 6 * mix.counts[1] // 40 and max(seen[0]) == 6 * mix.counts[0] // 300


def test_items_are_the_sources_items(mix, built):
    _, folders, _ = built
    a = ShardedDataset(str(folders["a"] / "train"))
    for i in (0, 17, 149):
        s, j, _ = mix.locate(i)
        assert s == 0 and torch.equal(mix[i]["input_ids"], a[j]["input_ids"])


def test_lengths_match_items(mix):
    for e in (0, 3):
        mix.set_epoch(e)
        lengths = mix.get_all_sequence_lengths()
        assert len(lengths) == len(mix)
        assert all(lengths[i] == len(mix[i]["input_ids"]) for i in range(len(mix)))


def test_a_repeated_cluster_source_draws_new_members_each_pass(built, tmp_path):
    _, folders, pool = built
    m = _write(tmp_path, folders, {"a": 0.1, "c": 0.9})
    mix = open_train_dataset(tmp_path, m, seed=SEED)
    c = mix.datasets[1]
    assert isinstance(c, ClusterSampledDataset)
    assert mix.counts[1] > 1.5 * mix.sizes[1]        # c repeats within one epoch
    rows = {}
    for i, e in ((i, e) for e in (0, 1) for i in range(len(mix))):
        mix.set_epoch(e)
        s, j, p = mix.locate(i)
        if s == 1:
            rows.setdefault(p, {})[j] = mix[i]["input_ids"]
            # pass p of the source draws the member of ClusterSampledDataset epoch p
            assert torch.equal(rows[p][j], c.read_pool_row(int(c.rows_for_epoch(p)[j]))["input_ids"])
    assert len(rows[0]) == len(rows[1]) == len(c)    # passes 0 and 1 are whole
    assert len(rows) >= 3
    multi = [j for j in range(len(c)) if c.cluster_size(j) > 1]
    changed = sum(not torch.equal(rows[0][j], rows[1][j]) for j in multi)
    assert changed > 0.4 * len(multi)


def test_same_items_in_workers_and_for_every_world_size(mix):
    mix.set_epoch(2)
    direct = [_key(mix[i]) for i in range(len(mix))]
    clone = pickle.loads(pickle.dumps(mix))          # a spawned worker's copy
    assert [_key(clone[i]) for i in range(len(mix))] == direct
    dl = DataLoader(mix, batch_size=None, num_workers=2, collate_fn=_key)
    assert list(dl) == direct
    for world in (1, 2, 3):
        for rank in range(world):
            sampler = DistributedSampler(mix, num_replicas=world, rank=rank, seed=SEED)
            sampler.set_epoch(2)
            assert all(_key(mix[i]) == direct[i] for i in sampler)


def _packed(dataset_dir, manifest, world=2, rank=0):
    d = open_train_dataset(dataset_dir, manifest, seed=SEED)
    s = DistributedSampler(d, num_replicas=world, rank=rank, shuffle=True, seed=SEED)
    return TokenPackingDataset(d, max_tokens_per_batch=512, drop_last=True, sampler=s)


def _batches(packed, workers):
    dl = DataLoader(packed, batch_size=None, num_workers=workers,
                    collate_fn=lambda b: tuple(_key(x) for x in b))
    return list(dl)


@pytest.mark.parametrize("workers", [0, 2])
def test_mid_epoch_resume_reproduces_the_rest_of_the_epoch(built, workers):
    root, folders, _ = built
    m = _write(root, folders, {"a": 0.5, "b": 0.2, "c": 0.3})
    epoch, skip = 3, 4
    full = _packed(root, m)
    full.set_epoch(epoch)
    want = _batches(full, workers)
    assert len(want) > skip + 3
    if workers == 0:
        assert len(want) == len(full)
    # the pure pipeline resumes with set_epoch(epoch), then drops the first batches
    resumed = _packed(root, m)
    resumed.set_epoch(epoch)
    assert _batches(resumed, workers)[skip:] == want[skip:]


def test_packed_length_is_recomputed_for_each_epoch(built):
    root, folders, _ = built
    packed = _packed(root, _write(root, folders, {"a": 0.5, "b": 0.2, "c": 0.3}))
    lens = set()
    for e in range(4):
        packed.set_epoch(e)
        lens.add(len(packed))
        assert len(packed) == len(_batches(packed, 0))
    assert len(lens) > 1


def test_fingerprint_follows_the_weights(built, tmp_path):
    _, folders, _ = built
    fp = []
    for w in (0.5, 0.6):
        m = _write(tmp_path, folders, {"a": w, "b": round(1 - w, 6)})
        fp.append(open_train_dataset(tmp_path, m, seed=SEED).fingerprint())
    assert fp[0] != fp[1]


# ------------------------------------------------------------------ the run plan


def test_epochs_in_the_table_match_what_the_dataset_reads(tmp_path):
    """Constant lengths per source make the token budget an exact sample count: the
    table's epochs are then what E mixture epochs read from each source."""
    a = plain_source(tmp_path / "a", 50, tag=5)
    b = plain_source(tmp_path / "b", 7, tag=6)
    for folder, n in ((a, 50), (b, 7)):            # rewrite with one length
        from tests.data.datasets.mixture_sources import write_shards
        write_shards(folder / "train", [30 if n == 50 else 10] * n, 0, 1)
        man = yaml.safe_load((folder / ".data_manifest").read_text())
        man["total_train_tokens"] = n * (30 if n == 50 else 10)
        from nanoplm.data.infra.manifest import PretrainManifest, write_manifest
        write_manifest(folder, PretrainManifest(**man))
    m = _write(tmp_path, {"a": a, "b": b}, {"a": 0.4, "b": 0.6})
    mix = open_train_dataset(tmp_path, m, seed=SEED)
    epochs = 9
    budget = epochs * sum(k * (30 if s == 0 else 10) for s, k in enumerate(mix.counts))
    plan = mx.run_plan(m, budget)
    reads = Counter()
    for e in range(epochs):
        reads.update(mix.locate(i, e)[0] for i in range(len(mix)))
    for s, p in enumerate(plan):
        assert p["epochs"] == pytest.approx(reads[s] / mix.sizes[s])
        assert p["share"] == mix.counts[s] / len(mix)
    assert plan[1]["epochs"] > 10
    lines = mx.format_run_plan(plan, budget)
    assert lines[1].split() == ["source", "share", "size", "tokens", "used", "epochs"]
    assert lines[-1].startswith("Warning: sources.b: read ")
    assert "lower sources.b.weight or add data" in lines[-1]
    assert not any("sources.a:" in ln for ln in lines)
