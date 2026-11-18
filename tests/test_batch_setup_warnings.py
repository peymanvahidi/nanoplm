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


def _achieved(target, mbs, seq_len, world):
    step = mbs * seq_len * world
    return -(-target // step) * step


def _msg(target, mbs, seq_len, world):
    return batch_rounding_warning(target, _achieved(target, mbs, seq_len, world),
                                  world, mbs, seq_len)


def test_warning_gpu_count_fix():
    # 48 ranks: grad_accum = ceil(128/48) = 3 -> 3 x 48 x 16 x 2048
    msg = _msg(4_194_304, 16, 2048, 48)
    assert msg == (
        "Global batch is 4,718,592 tokens, not global_batch_tokens=4,194,304 (+12.5%). "
        "Why: one micro-step on 48 GPUs is 1,572,864 tokens (48 x 16 x 2048) and an "
        "optimizer step uses whole micro-steps, here 3. Fix: use a GPU count that "
        "divides 128, or set global_batch_tokens=4,718,592."
    )


def test_warning_target_not_multiple_of_micro_step():
    # default config (256,000, 64 x 256) on 8 GPUs: grad_accum 2 -> 262,144
    msg = _msg(256_000, 64, 256, 8)
    assert msg.startswith("Global batch is 262,144 tokens, not global_batch_tokens=256,000 (+2.4%).")
    assert "Fix: set global_batch_tokens=262,144, or change micro_batch_seqs." in msg
    assert "No GPU count hits 256,000" in msg and "divides" not in msg


def test_warning_no_micro_batch_seqs_advice_when_none_hits():
    # 256,000 / (256 x 3) is not whole: no micro_batch_seqs hits the target on 3 GPUs
    msg = _msg(256_000, 64, 256, 3)
    assert "Fix: set global_batch_tokens=294,912." in msg
    assert "micro_batch_seqs," not in msg and "change micro_batch_seqs" not in msg


@pytest.mark.parametrize("target,mbs,seq_len,world,step", [
    (256_000, 64, 256, 128, 2_097_152),      # default config, not a multiple
    (4_194_304, 16, 2048, 256, 8_388_608),   # run config, a multiple of 32,768
    (4_194_304, 16, 2048, 384, 12_582_912),  # mbs=1 still overshoots on 384 GPUs
])
def test_warning_one_micro_step_exceeds_target(target, mbs, seq_len, world, step):
    msg = batch_rounding_warning(target, step, world, mbs, seq_len)
    assert f"Why: one micro-step on {world} GPUs is already {step:,} tokens" in msg
    assert f"Fix: set global_batch_tokens={step:,}" in msg
    if target % (seq_len * world):
        assert "lower micro_batch_seqs to shrink the overshoot" in msg
    else:
        assert "lower micro_batch_seqs," in msg and "shrink" not in msg
    if target % (mbs * seq_len):
        assert "No GPU count hits" in msg and "divides" not in msg
    else:
        assert "divides 128" in msg


def test_warning_single_gpu_wording():
    # target is exactly one micro-step: 1 GPU is the only GPU count that hits it
    msg = batch_rounding_warning(32_768, 65_536, 2, 16, 2048)
    assert msg.endswith("or use 1 GPU.")
    msg = batch_rounding_warning(30_000, 32_768, 1, 16, 2048)
    assert "on 1 GPU is" in msg and "1 GPUs" not in msg


def test_warning_small_overshoot_never_prints_zero_percent():
    assert "(+0.1%)" in batch_rounding_warning(4_190_000, 4_194_304, 128, 16, 2048)
    assert "(+0.04%)" in batch_rounding_warning(4_192_600, 4_194_304, 128, 16, 2048)
    assert "(+1 token)" in batch_rounding_warning(4_194_303, 4_194_304, 128, 16, 2048)


@pytest.mark.parametrize("target,mbs,seq_len,world", [
    (256_000, 64, 256, 8), (256_000, 64, 256, 128), (256_000, 64, 256, 3),
    (4_194_304, 16, 2048, 48), (4_194_304, 16, 2048, 256), (4_194_304, 16, 2048, 384),
    (4_194_303, 16, 2048, 1000), (12_345_678, 7, 1000, 999),
])
def test_warning_is_short(target, mbs, seq_len, world):
    assert len(_msg(target, mbs, seq_len, world).split()) <= 48


def test_compute_batch_setup_logs_warning(monkeypatch):
    import nanoplm.pretraining.utils as u

    seen = []
    monkeypatch.setattr(u.logger, "warning", lambda m, *a: seen.append(m))
    cfg = PretrainingConfig(dataset_dir="/tmp", global_batch_tokens=4_194_304,
                            micro_batch_seqs=16)
    setup = compute_batch_setup(cfg, 2048, 48)
    assert setup.achieved_global_batch_tokens == 3 * 48 * 16 * 2048
    assert len(seen) == 1 and "48 GPUs" in seen[0]

    seen.clear()
    compute_batch_setup(cfg, 2048, 32)
    assert seen == []


def test_num_workers_agree_ok():
    check_num_workers_agree([4, 4, 4, 4])
    check_num_workers_agree([0])


def test_num_workers_disagree_raises():
    with pytest.raises(RuntimeError, match=r"num_workers differs between ranks.*3 on rank\(s\) 2"):
        check_num_workers_agree([4, 4, 3, 4])
