"""Batch-warmup splits of a packed batch must keep the collator's max_seqlen buckets.

Each distinct int ``max_seqlen`` is a new Dynamo guard, so an unbucketed
split value recompiles the layer frame until it hits the recompile limit.
"""

import pytest
import torch

from nanoplm.pretraining.collator import build_power_of_two_buckets
from nanoplm.pretraining.pure_pipeline import _split_batch

_TOKENS = 8192
_BUCKETS = build_power_of_two_buckets(512, min_power_of_two=32)


def _packed_batch(lengths):
    valid = sum(lengths)
    pad = _TOKENS - valid
    seg = list(lengths) + ([pad] if pad else [])
    cu = torch.zeros(len(seg) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(seg).cumsum(0)
    input_ids = torch.zeros(_TOKENS, dtype=torch.long)
    input_ids[:valid] = torch.arange(1, valid + 1)
    labels = torch.full((_TOKENS,), -100, dtype=torch.long)
    labels[:valid] = input_ids[:valid]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "cu_seqlens": cu,
        "max_seqlen": 512,
        "position_ids": torch.cat([torch.arange(n) for n in seg]),
        "num_valid_tokens": torch.tensor(valid, dtype=torch.long),
    }


@pytest.mark.parametrize("fill", [1.0, 0.6], ids=["full", "part"])
@pytest.mark.parametrize("divisor", [2, 4, 8])
def test_split_max_seqlen_stays_in_buckets(divisor, fill):
    gen = torch.Generator().manual_seed(divisor)
    lengths = []
    while True:
        n = int(torch.randint(20, 513, (1,), generator=gen))
        if sum(lengths) + n > _TOKENS * fill:
            break
        lengths.append(n)
    batch = _packed_batch(lengths)
    per_split = _TOKENS // divisor
    subs = _split_batch(
        batch, n_splits=divisor, effective_tokens_per_split=per_split,
        max_seqlen_buckets=_BUCKETS,
    )

    pads = []
    for sub in subs:
        # flash varlen requires int32 cu_seqlens; every tensor keeps its input dtype.
        for key, value in sub.items():
            if torch.is_tensor(value) and key in batch:
                assert value.dtype == batch[key].dtype, key
                assert value.device == batch[key].device, key
        cu = sub["cu_seqlens"]
        seg = (cu[1:] - cu[:-1]).tolist()
        valid = int(sub["num_valid_tokens"])
        assert sub["max_seqlen"] in _BUCKETS
        assert sub["max_seqlen"] >= max(seg)
        assert int(cu[0]) == 0 and int(cu[-1]) == per_split
        assert sub["input_ids"].numel() == per_split
        assert (sub["labels"][valid:] == -100).all()
        starts = cu[:-1].long()
        assert (sub["position_ids"][starts] == 0).all()
        assert torch.equal(
            sub["position_ids"],
            torch.cat([torch.arange(n) for n in seg]),
        )
        pads.append(per_split - valid)

    # Splits keep whole sequences in order; a full batch may lose its tail.
    kept = torch.cat([s["input_ids"][: int(s["num_valid_tokens"])] for s in subs])
    assert torch.equal(kept, batch["input_ids"][: kept.numel()])
    if fill < 1.0:
        assert kept.numel() == sum(lengths)
        assert max(pads) > _BUCKETS[-1], "test precondition: a pad longer than the top bucket"


def test_split_without_buckets_keeps_one_pad_segment():
    batch = _packed_batch([300, 400, 100])
    subs = _split_batch(batch, n_splits=2, effective_tokens_per_split=_TOKENS // 2)
    for sub in subs:
        cu = sub["cu_seqlens"]
        valid = int(sub["num_valid_tokens"])
        assert cu.dtype == torch.int32
        assert int(cu[-1] - cu[-2]) == _TOKENS // 2 - valid
        assert sub["max_seqlen"] == int((cu[1:] - cu[:-1]).max())
