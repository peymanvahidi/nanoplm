"""TokenPackingDataset.__len__ equals what a multi-worker DataLoader yields.

With more than one DataLoader worker each worker packs its own
DistributedSampler shard and drops its own partial last batch, so the length
must be summed per worker, not packed over the rank's order.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nanoplm.data.datasets.pretrain import TokenPackingDataset

N = 1237  # not divisible by any W * nw under test
MAX_TOKENS = 96


class LengthsDataset(Dataset):
    """Map-style dataset whose samples carry their index; exposes lengths."""

    def __init__(self, lengths):
        self.lengths = np.asarray(lengths, dtype=np.int32)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return {"input_ids": torch.full((int(self.lengths[idx]),), int(idx), dtype=torch.long)}

    def get_all_sequence_lengths(self):
        return self.lengths


def _lengths(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(5, 60, size=N)


def _packed(ds, world, rank, nw, *, drop_last_sampler=False, drop_last=True, epoch=0):
    sampler = DistributedSampler(
        ds, num_replicas=world, rank=rank, shuffle=True, seed=7, drop_last=drop_last_sampler
    )
    packed = TokenPackingDataset(
        ds, max_tokens_per_batch=MAX_TOKENS, drop_last=drop_last, sampler=sampler, num_workers=nw
    )
    packed.set_epoch(epoch)
    return packed


def _count_yield(packed, nw):
    loader = DataLoader(packed, batch_size=None, num_workers=nw, collate_fn=lambda b: b)
    return sum(1 for _ in loader)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("nw", [0, 1, 2, 4])
def test_len_equals_dataloader_yield(world, nw):
    ds = LengthsDataset(_lengths())
    for rank in range(world):
        packed = _packed(ds, world, rank, nw, epoch=3)
        assert len(packed) == _count_yield(packed, nw), (world, nw, rank)


@pytest.mark.parametrize("drop_last_sampler", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
def test_len_equals_yield_drop_last_variants(drop_last_sampler, drop_last):
    ds = LengthsDataset(_lengths(1))
    world, nw = 4, 4
    for rank in range(world):
        packed = _packed(
            ds, world, rank, nw, drop_last_sampler=drop_last_sampler, drop_last=drop_last
        )
        assert len(packed) == _count_yield(packed, nw)


def test_len_tracks_epoch_and_worker_count():
    ds = LengthsDataset(_lengths(2))
    packed = _packed(ds, 4, 1, 4, epoch=0)
    len(packed)
    for epoch in (1, 2):
        packed.set_epoch(epoch)
        assert len(packed) == _count_yield(packed, 4)
    packed.set_num_workers(2)
    assert len(packed) == _count_yield(packed, 2)


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("n", [N, 5])  # n=5 < W*nw forces multi-cycle padding
def test_worker_indices_match_per_worker_distributed_sampler(shuffle, drop_last, n):
    ds = LengthsDataset(_lengths()[:n])
    for world in (1, 2, 4, 8):
        for nw in (1, 2, 4):
            for rank in range(world):
                sampler = DistributedSampler(
                    ds, num_replicas=world, rank=rank, shuffle=shuffle, seed=11, drop_last=drop_last
                )
                packed = TokenPackingDataset(ds, MAX_TOKENS, sampler=sampler, num_workers=nw)
                fast = packed._distributed_worker_indices(nw, epoch=5)
                for w in range(nw):
                    ref = DistributedSampler(
                        ds,
                        num_replicas=world * nw,
                        rank=rank * nw + w,
                        shuffle=shuffle,
                        seed=11,
                        drop_last=drop_last,
                    )
                    ref.set_epoch(5)
                    assert fast[w].tolist() == list(ref), (world, nw, rank, w)


def test_len_draws_one_permutation(monkeypatch):
    ds = LengthsDataset(_lengths())
    packed = _packed(ds, 4, 2, 4, epoch=1)
    calls = []
    real = torch.randperm

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "randperm", counting)
    len(packed)
    assert len(calls) == 1
    len(packed)  # cached
    assert len(calls) == 1
    packed.set_epoch(1)  # same epoch: cache kept
    len(packed)
    assert len(calls) == 1
    packed.set_epoch(2)
    len(packed)
    assert len(calls) == 2
