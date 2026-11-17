"""Startup checks on the batch setup: rounding warning and num_workers agreement."""

import pytest

from nanoplm.pretraining.config import PretrainingConfig
from nanoplm.pretraining.utils import (
    batch_rounding_warning,
    check_num_workers_agree,
    compute_batch_setup,
)


def test_no_warning_when_exact():
    # 16 x 2048 x 32 ranks x grad_accum 4 = 4,194,304
    assert batch_rounding_warning(4_194_304, 4_194_304, 32, 16, 2048) is None


def test_warning_names_target_achieved_world_and_hint():
    # 48 ranks: grad_accum = ceil(128/48) = 3 -> 3 x 48 x 16 x 2048
    achieved = 3 * 48 * 16 * 2048
    msg = batch_rounding_warning(4_194_304, achieved, 48, 16, 2048)
    assert msg is not None
    assert "4,194,304" in msg and f"{achieved:,}" in msg
    assert "world_size=48" in msg
    assert "divides 128" in msg and "micro_batch_seqs" in msg


def test_compute_batch_setup_logs_warning(monkeypatch):
    import nanoplm.pretraining.utils as u

    seen = []
    monkeypatch.setattr(u.logger, "warning", lambda m, *a: seen.append(m))
    cfg = PretrainingConfig(dataset_dir="/tmp", global_batch_tokens=4_194_304,
                            micro_batch_seqs=16)
    setup = compute_batch_setup(cfg, 2048, 48)
    assert setup.achieved_global_batch_tokens == 3 * 48 * 16 * 2048
    assert len(seen) == 1 and "world_size=48" in seen[0]

    seen.clear()
    compute_batch_setup(cfg, 2048, 32)
    assert seen == []


def test_num_workers_agree_ok():
    check_num_workers_agree([4, 4, 4, 4])
    check_num_workers_agree([0])


def test_num_workers_disagree_raises():
    with pytest.raises(RuntimeError, match=r"num_workers differs between ranks.*3 on rank\(s\) 2"):
        check_num_workers_agree([4, 4, 3, 4])
