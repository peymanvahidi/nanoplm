import pytest
from nanoplm.pretraining.config import ResumeConfig
from nanoplm.cli.pretrain import _validate_resume_config_modes


class TestResumeModeValidation:
    def test_not_resuming_skips(self):
        _validate_resume_config_modes(ResumeConfig(is_resume=False, checkpoint_dir=""))

    def test_continue_needs_no_decay_fields(self):
        _validate_resume_config_modes(
            ResumeConfig(is_resume=True, checkpoint_dir="/tmp/ckpt", mode="continue"))

    def test_continue_with_decay_fields_raises(self):
        with pytest.raises(ValueError, match="continue"):
            _validate_resume_config_modes(
                ResumeConfig(is_resume=True, checkpoint_dir="/tmp/ckpt",
                             mode="continue", decay_tokens=1_000_000))

    def test_decay_with_token_budget_ok(self):
        _validate_resume_config_modes(
            ResumeConfig(is_resume=True, checkpoint_dir="/tmp/ckpt", mode="decay",
                         decay_tokens=1_000_000, decay_shape="1-sqrt", lr_decay_to_fraction=0.1))

    def test_decay_without_budget_raises(self):
        with pytest.raises(ValueError, match="decay_tokens|decay_steps"):
            _validate_resume_config_modes(
                ResumeConfig(is_resume=True, checkpoint_dir="/tmp/ckpt", mode="decay"))

    def test_decay_tokens_and_steps_both_set_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            _validate_resume_config_modes(
                ResumeConfig(is_resume=True, checkpoint_dir="/tmp/ckpt", mode="decay",
                             decay_tokens=1_000_000, decay_steps=100))

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            _validate_resume_config_modes(
                ResumeConfig(is_resume=True, checkpoint_dir="/tmp/ckpt", mode="bogus"))


class TestResumeLoadStrictKeys:
    def test_rejects_unknown_key(self):
        from nanoplm.cli.pretrain import _load_resume_config
        with pytest.raises(ValueError, match="resume_epochs"):
            _load_resume_config({"is_resume": True, "checkpoint_dir": ".", "resume_epochs": 2})

    def test_accepts_decay_keys(self):
        from nanoplm.cli.pretrain import _load_resume_config
        rc = _load_resume_config({
            "is_resume": True, "checkpoint_dir": ".", "mode": "decay",
            "decay_tokens": 1_000_000, "decay_shape": "1-sqrt", "lr_decay_to_fraction": 0.1,
        })
        assert rc.mode == "decay" and rc.decay_tokens == 1_000_000

    def test_decay_shape_invalid_rejected(self):
        from nanoplm.cli.pretrain import _load_resume_config
        with pytest.raises(ValueError, match="decay_shape"):
            _load_resume_config({"is_resume": True, "checkpoint_dir": ".", "mode": "decay",
                                 "decay_tokens": 1000, "decay_shape": "bogus"})

    def test_mode_case_normalized(self):
        from nanoplm.cli.pretrain import _load_resume_config
        rc = _load_resume_config({"is_resume": True, "checkpoint_dir": ".", "mode": "Decay",
                                  "decay_tokens": 1000})
        assert rc.mode == "decay"

    def test_decay_tokens_string_coerced_to_int(self):
        from nanoplm.cli.pretrain import _load_resume_config
        rc = _load_resume_config({"is_resume": True, "checkpoint_dir": ".", "mode": "decay",
                                  "decay_tokens": "1000"})
        assert rc.decay_tokens == 1000 and isinstance(rc.decay_tokens, int)
