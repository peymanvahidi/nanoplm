"""Per-rank DataLoader worker seeding: each rank draws its own MLM masks."""

import functools
import pickle

import torch
from torch.utils.data import DataLoader

from nanoplm.pretraining.collator import ProtDataCollatorForLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.pure_pipeline import _seed_worker_for_rank

SEQS = [
    "MKALCLLLLPVLGLLTGSSGSAEDKRWYHNQ",
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN",
    "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVG",
    "GSHMKVLAAGIVGLPNVGKSTLFNALTKAGIE",
] * 4


def _batches(rank=None, seed=1234):
    tokenizer = ProtModernBertTokenizer()
    ds = [
        {k: v for k, v in tokenizer(s, padding=False, truncation=True, max_length=64).items()
         if k in ("input_ids", "attention_mask")}
        for s in SEQS
    ]
    collator = ProtDataCollatorForLM(tokenizer=tokenizer, mlm_probability=0.3)
    init = None if rank is None else functools.partial(_seed_worker_for_rank, rank)
    torch.manual_seed(seed)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collator,
                        num_workers=2, multiprocessing_context="fork", worker_init_fn=init)
    return [(b["input_ids"].clone(), b["labels"].clone()) for b in loader]


def _unmasked(ids, labels):
    return torch.where(labels != -100, labels, ids)


def test_rank_zero_unchanged_other_ranks_differ():
    plain = _batches(None)
    r0, r1, r2 = _batches(0), _batches(1), _batches(2)

    # (1) rank 0 keeps PyTorch's seeding: bit-identical to no worker_init_fn
    for (a_ids, a_lab), (b_ids, b_lab) in zip(plain, r0):
        assert torch.equal(a_ids, b_ids) and torch.equal(a_lab, b_lab)

    # (2) first batch of each worker: masks differ across ranks on the same rows
    for w in range(2):
        m0, m1, m2 = (r[w][1] != -100 for r in (r0, r1, r2))
        assert not torch.equal(m0, m1)
        assert not torch.equal(m0, m2)
        assert not torch.equal(m1, m2)

    # (3) deterministic per rank
    for (a_ids, a_lab), (b_ids, b_lab) in zip(r1, _batches(1)):
        assert torch.equal(a_ids, b_ids) and torch.equal(a_lab, b_lab)

    # (4) the unmasked data stream is the same on every rank
    for b0, b1, b2 in zip(r0, r1, r2):
        u0 = _unmasked(*b0)
        assert torch.equal(u0, _unmasked(*b1))
        assert torch.equal(u0, _unmasked(*b2))


def test_worker_init_fn_picklable():
    fn = pickle.loads(pickle.dumps(functools.partial(_seed_worker_for_rank, 3)))
    assert fn.func is _seed_worker_for_rank and fn.args == (3,)
