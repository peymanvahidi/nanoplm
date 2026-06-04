import pytest
from nanoplm.pretraining.config import PretrainingConfig
from nanoplm.cli.pretrain import _validate_duration_config


def _cfg(**kw):
    base = dict(dataset_dir="/tmp", lr_schedule="wsd")
    base.update(kw)
    return PretrainingConfig(**base)


class TestDurationValidation:
    def test_wsd_with_token_budgets_ok(self):
        _validate_duration_config(_cfg(stable_tokens=9_000_000, decay_tokens=1_000_000))

    def test_wsd_with_step_overrides_ok(self):
        _validate_duration_config(_cfg(stable_steps=100, decay_steps=20))

    def test_wsd_missing_decay_budget_raises(self):
        with pytest.raises(ValueError, match="decay"):
            _validate_duration_config(_cfg(stable_tokens=9_000_000))

    def test_wsd_tokens_and_steps_both_set_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            _validate_duration_config(_cfg(stable_tokens=9_000_000, stable_steps=100,
                                           decay_tokens=1_000_000))

    def test_single_phase_uses_train_tokens(self):
        _validate_duration_config(_cfg(lr_schedule="cosine", train_tokens=10_000_000))

    def test_single_phase_with_wsd_budget_raises(self):
        with pytest.raises(ValueError, match="train_tokens|max_steps"):
            _validate_duration_config(_cfg(lr_schedule="warmup_stable",
                                           stable_tokens=9_000_000, decay_tokens=1_000_000))

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="lr_schedule"):
            _validate_duration_config(_cfg(lr_schedule="bogus", train_tokens=1))
