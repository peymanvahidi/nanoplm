import pytest
from nanoplm.pretraining.config import PretrainingConfig
from nanoplm.pretraining.utils import resolve_total_steps


def _cfg(**kw):
    base = dict(dataset_dir="/tmp")
    base.update(kw)
    return PretrainingConfig(**base)


class TestResolveTotalSteps:
    def test_wsd_tokens(self):
        r = resolve_total_steps(_cfg(lr_schedule="wsd", stable_tokens=9_000_000,
                                     decay_tokens=1_000_000),
                                achieved_global_batch_tokens=250_000)
        assert r.stable_steps == 36 and r.decay_steps == 4 and r.total_steps == 40

    def test_wsd_step_override(self):
        r = resolve_total_steps(_cfg(lr_schedule="wsd", stable_steps=30, decay_steps=5),
                                achieved_global_batch_tokens=250_000)
        assert r.stable_steps == 30 and r.decay_steps == 5 and r.total_steps == 35

    def test_single_phase_tokens(self):
        r = resolve_total_steps(_cfg(lr_schedule="cosine", train_tokens=10_000_000),
                                achieved_global_batch_tokens=250_000)
        assert r.stable_steps is None and r.decay_steps is None and r.total_steps == 40

    def test_single_phase_max_steps(self):
        r = resolve_total_steps(_cfg(lr_schedule="warmup_stable", max_steps=123),
                                achieved_global_batch_tokens=250_000)
        assert r.total_steps == 123

    def test_single_phase_token_floor_is_one(self):
        # tiny budget floors to >=1 step
        r = resolve_total_steps(_cfg(lr_schedule="cosine", train_tokens=1),
                                achieved_global_batch_tokens=250_000)
        assert r.total_steps == 1

    def test_lr_schedule_case_insensitive(self):
        r = resolve_total_steps(_cfg(lr_schedule="WSD", stable_tokens=9_000_000,
                                     decay_tokens=1_000_000),
                                achieved_global_batch_tokens=250_000)
        assert r.stable_steps == 36 and r.decay_steps == 4 and r.total_steps == 40


def test_total_steps_is_a_step_boundary_multiple_of_known_inputs():
    r = resolve_total_steps(_cfg(lr_schedule="wsd", stable_tokens=1, decay_tokens=1),
                            achieved_global_batch_tokens=250_000)
    assert r.stable_steps == 1 and r.decay_steps == 1 and r.total_steps == 2
