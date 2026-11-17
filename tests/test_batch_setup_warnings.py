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
    assert "divides 128" in msg
    assert "no GPU count" not in msg and "One micro-step" not in msg


def _achieved(target, mbs, seq_len, world):
    step = mbs * seq_len * world
    return -(-target // step) * step


def test_warning_target_not_multiple_of_micro_step():
    # default config (256,000, 64 x 256) on 8 GPUs: grad_accum 2 -> 262,144
    msg = batch_rounding_warning(256_000, _achieved(256_000, 64, 256, 8), 8, 64, 256)
    assert msg == (
        "Global batch is 262,144 tokens, not the target global_batch_tokens=256,000 "
        "(+6,144 tokens, +2.4%): grad_accum rounds up at world_size=8 with "
        "micro_batch_seqs=64 x max_seq_len=256 = 16,384 tokens per GPU per micro-step. "
        "The target is not a multiple of 16,384, so no GPU count hits it with this "
        "micro_batch_seqs. Set global_batch_tokens=262,144 to keep this batch, "
        "or change micro_batch_seqs."
    )


def test_warning_no_micro_batch_seqs_advice_when_none_hits():
    # 256,000 / (256 x 3) is not whole: no micro_batch_seqs hits the target on 3 GPUs
    msg = batch_rounding_warning(256_000, _achieved(256_000, 64, 256, 3), 3, 64, 256)
    assert msg.endswith("Set global_batch_tokens=294,912 to keep this batch.")


@pytest.mark.parametrize("target,mbs,seq_len,world,step", [
    (256_000, 64, 256, 128, 2_097_152),      # default config, not a multiple
    (4_194_304, 16, 2048, 256, 8_388_608),   # run config, a multiple of 32,768
    (4_194_304, 16, 2048, 384, 12_582_912),  # mbs=1 still overshoots on 384 GPUs
])
def test_warning_one_micro_step_exceeds_target(target, mbs, seq_len, world, step):
    msg = batch_rounding_warning(target, step, world, mbs, seq_len)
    assert f"One micro-step on {world} GPUs is already {step:,} tokens" in msg
    assert f"Raise global_batch_tokens to {step:,}" in msg
    if target % (seq_len * world):
        assert "lower micro_batch_seqs to shrink the overshoot" in msg
    else:
        assert "lower micro_batch_seqs," in msg and "shrink" not in msg
    if target % (mbs * seq_len):
        assert "no GPU count hits it" in msg and "divides" not in msg
    else:
        assert "divides 128" in msg


def test_warning_single_gpu_wording():
    # target is exactly one micro-step: 1 GPU is the only GPU count that hits it
    msg = batch_rounding_warning(32_768, 65_536, 2, 16, 2048)
    assert msg.endswith("or use 1 GPU.")
    msg = batch_rounding_warning(30_000, 32_768, 1, 16, 2048)
    assert "on 1 GPU is" in msg and "1 GPUs" not in msg


def test_warning_small_overshoot_shows_token_difference():
    msg = batch_rounding_warning(4_190_000, 4_194_304, 128, 16, 2048)
    assert "+4,304 tokens, +0.1%" in msg
    assert "x)" not in msg and "1.00x" not in msg


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
