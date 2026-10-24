"""The short-pass check against grad_accum, and the once-per-run rank-mismatch warning."""
import logging

import pytest
import torch

import nanoplm.pretraining.pure_pipeline as pp


def test_pass_shorter_than_grad_accum_fails_at_startup_naming_the_fix():
    with pytest.raises(RuntimeError, match="per pass.*Lower global_batch_tokens"):
        pp._check_pass_len(7, 8, 65536)


def test_pass_within_ten_percent_of_grad_accum_warns_at_startup():
    msg = pp._check_pass_len(21, 20, 65536)
    assert msg and "within 10% of grad_accum=20" in msg and "Lower global_batch_tokens" in msg
    assert pp._check_pass_len(23, 20, 65536) is None


def test_short_later_pass_error_is_actionable():
    with pytest.raises(RuntimeError) as e:
        pp._check_pass_len(19, 20, 65536, pass_idx=3)
    text = str(e.value)
    assert "in pass 4" in text and "Pass lengths vary" in text
    assert "Lower global_batch_tokens or use a larger dataset" in text
    assert pp._check_pass_len(21, 20, 65536, pass_idx=3) is None   # no margin warning later


def test_rank_length_mismatch_warns_once_per_run(monkeypatch, caplog):
    lens = iter([(10, 12), (9, 12), (11, 13)])

    def all_reduce(t, op):
        lo, hi = current
        t.fill_(lo if op == pp.dist.ReduceOp.MIN else hi)

    monkeypatch.setattr(pp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pp.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(pp.dist, "all_reduce", all_reduce)
    sync = pp._TrainLenSync(distributed=True, device=torch.device("cpu"))
    got = []
    with caplog.at_level(logging.WARNING):
        for current in lens:
            got.append(sync(current[0]))
    assert got == [10, 9, 11]
    warned = [r for r in caplog.records if "Mismatched train loader lengths" in r.getMessage()]
    assert len(warned) == 1
