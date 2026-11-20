"""TokenPackingDataset.skip_batches(k): a new DataLoader yields exactly the batches a
replay yields after discarding the first k, for every k, through a real DataLoader
(its worker walk and in-order results are torch-internal; these tests pin them)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nanoplm.data.datasets.pretrain import TokenPackingDataset, split_skip
from nanoplm.pretraining import pure_pipeline

MAXTOK = 100


class Synth(Dataset):
    def __init__(self, n, mode, seed):
        rng = np.random.default_rng(seed)
        if mode == "uniform":
            self.lens = rng.integers(3, 60, size=n).astype(np.int32)
        elif mode == "skew":  # no shuffle: worker w reads idx = w mod (W*nw), very unequal workers
            self.lens = (3 + (np.arange(n) % 8) * 9 + rng.integers(0, 5, size=n)).astype(np.int32)
        elif mode == "exact":  # many exact fills (the == branch of the packer)
            self.lens = rng.choice([10, 20, 25, 50], size=n).astype(np.int32)

    def __len__(self):
        return len(self.lens)

    def get_all_sequence_lengths(self):
        return self.lens

    def __getitem__(self, i):
        return {"input_ids": torch.full((int(self.lens[i]),), i % 251, dtype=torch.uint8),
                "idx": int(i)}


def _ids(samples):
    return tuple(s["idx"] for s in samples)


def _loader(packed, nw, collate=_ids):
    kw = dict(num_workers=nw, prefetch_factor=2) if nw > 0 else {}
    return DataLoader(packed, batch_size=None, collate_fn=collate, **kw)


CASES = [  # n, mode, world, rank, shuffle, epoch
    (150, "uniform", 1, 0, True, 0),
    (300, "exact", 2, 1, True, 3),
    (160, "skew", 1, 0, False, 0),
]


@pytest.mark.parametrize("nw", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n,mode,world,rank,shuffle,epoch", CASES)
def test_skip_batches_yields_the_rest_of_the_replay_for_every_k(n, mode, world, rank,
                                                               shuffle, epoch, nw):
    ds = Synth(n, mode, seed=n + world)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, seed=7)
    packed = TokenPackingDataset(ds, max_tokens_per_batch=MAXTOK, drop_last=True,
                                 split_samples=False, sampler=sampler, num_workers=nw)
    packed.set_epoch(epoch)
    total = len(packed)
    full = list(_loader(packed, nw))
    assert len(full) == total > 10
    exhausted_early = False
    for k in range(total + 1):
        plan = packed.skip_batches(k)
        if plan is not None and k < total:
            exhausted_early |= any(d == c for d, c in zip(plan["done"], plan["counts"]))
        assert list(_loader(packed, nw)) == full[k:], (mode, nw, k, total)
        assert len(packed) == total  # len() stays the full pass
    if mode == "skew" and nw == 4:
        # very unequal workers: one runs out well before the pass ends
        counts = packed.skip_batches(1)["counts"]
        assert max(counts) >= 2 * min(counts)
        assert exhausted_early
    packed.skip_batches(0)
    assert list(_loader(packed, nw)) == full


def test_plan_is_dropped_when_the_pass_changes():
    ds = Synth(150, "uniform", seed=1)
    sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=7)
    packed = TokenPackingDataset(ds, max_tokens_per_batch=MAXTOK, sampler=sampler, num_workers=2)
    packed.set_epoch(1)
    want = list(_loader(packed, 2))
    packed.set_epoch(0)
    packed.skip_batches(5)
    packed.set_epoch(0)  # same pass: the plan stays
    assert packed._skip_plan is not None
    packed.set_epoch(1)
    assert packed._skip_plan is None
    assert list(_loader(packed, 2)) == want


@pytest.mark.parametrize("nw", [0, 3])
def test_plan_for_another_worker_count_raises(nw):
    packed = _small_packed()  # built for 2 workers
    packed.skip_batches(4)
    with pytest.raises(RuntimeError, match="exact-skip plan"):
        list(_loader(packed, nw))


def test_split_skip_walks_the_live_workers():
    assert split_skip([2, 1, 3], 0) == ([0, 0, 0], 0)
    assert split_skip([2, 1, 3], 4) == ([2, 1, 1], 1)  # 0,1,2,0 then the walk is at 1
    assert split_skip([2, 1, 3], 6) == ([2, 1, 3], 0)
    with pytest.raises(ValueError):
        split_skip([2, 1, 3], 7)


@pytest.mark.parametrize("bad", [dict(split_samples=True), dict(drop_last=False), dict(sampler=None)])
def test_uncovered_settings_raise_and_leave_no_plan(bad):
    ds = Synth(150, "uniform", seed=1)
    kw = dict(max_tokens_per_batch=MAXTOK, drop_last=True, split_samples=False,
              sampler=DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=7))
    kw.update(bad)
    packed = TokenPackingDataset(ds, **kw)
    with pytest.raises(ValueError):
        packed.skip_batches(3)
    assert packed._skip_plan is None


# ------------------------------------------------------------- mixture with a cluster source


def test_skip_on_a_mixture_with_a_cluster_source(tmp_path):
    from nanoplm.data.datasets import open_train_dataset
    from nanoplm.data.infra import mixture as mx
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.datasets.mixture_sources import cluster_source, plain_source

    a = plain_source(tmp_path / "a", 300, tag=5, seed=1)
    c, _ = cluster_source(tmp_path / "c", seed=3)
    mx.write_mixture(tmp_path, [("a", a, 0.6, "a.fasta"), ("c", c, 0.4, "c.fasta")])
    d = open_train_dataset(tmp_path, read_manifest(tmp_path), seed=5)
    sampler = DistributedSampler(d, num_replicas=2, rank=1, shuffle=True, seed=5)
    packed = TokenPackingDataset(d, max_tokens_per_batch=512, drop_last=True,
                                 sampler=sampler, num_workers=2)
    packed.set_epoch(2)

    def rows(batch):  # the unmasked token rows
        return tuple(hashlib.sha1(s["input_ids"].numpy().tobytes()).hexdigest() for s in batch)

    total = len(packed)
    full = list(_loader(packed, 2, rows))
    assert len(full) == total > 5
    for k in range(total + 1):
        packed.skip_batches(k)
        assert list(_loader(packed, 2, rows)) == full[k:], k


# ------------------------------------------------------------- the pipeline's rank agreement


def _small_packed():
    ds = Synth(150, "uniform", seed=1)
    sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=7)
    return TokenPackingDataset(ds, max_tokens_per_batch=MAXTOK, sampler=sampler, num_workers=2)


def test_plan_resume_skip_skips_when_every_rank_can():
    packed = _small_packed()
    assert pure_pipeline._plan_resume_skip(packed, 4, False, torch.device("cpu")) is True
    assert packed._skip_plan is not None


def test_plan_resume_skip_falls_back_when_skip_batches_raises(monkeypatch):
    packed = _small_packed()
    full = list(_loader(packed, 2))

    real = packed.skip_batches

    def boom(k):  # this rank cannot skip; clearing (k=0) still works
        if k > 0:
            raise ValueError("no skip here")
        return real(k)

    monkeypatch.setattr(packed, "skip_batches", boom)
    assert pure_pipeline._plan_resume_skip(packed, 4, False, torch.device("cpu")) is False
    assert packed._skip_plan is None
    assert list(_loader(packed, 2)) == full  # replay reads the pass from its start


def test_plan_resume_skip_clears_the_plan_when_another_rank_failed(monkeypatch):
    packed = _small_packed()
    full = list(_loader(packed, 2))
    calls = []

    def all_reduce(t, op=None):  # another rank reports failure
        calls.append(op)
        t.zero_()

    monkeypatch.setattr(pure_pipeline.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pure_pipeline.dist, "all_reduce", all_reduce)
    assert pure_pipeline._plan_resume_skip(packed, 4, True, torch.device("cpu")) is False
    assert calls == [pure_pipeline.dist.ReduceOp.MIN]
    assert packed._skip_plan is None
    monkeypatch.undo()
    assert list(_loader(packed, 2)) == full
