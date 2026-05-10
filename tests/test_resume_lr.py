"""Unit tests for the resume LR EXTEND/RESET contract.

Spawned out of test_integration.py so they can be collected even while
the integration file has unrelated pre-existing syntax issues.
"""

import logging

import pytest
import yaml
import torch

from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig


class TestResumeConfigValidation:
    """Unit tests for the EXTEND vs RESET resume contract.

    These tests cover the YAML/CLI-load-time validator
    (_validate_resume_config_modes) and the strict-key behavior of
    _load_resume_config.  They do not run training; they verify the
    contract that the new schedule logic depends on.
    """

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            is_resume=True,
            checkpoint_dir="/tmp/ckpt",
            resume_epochs=2,
            warmup_steps=None,
            learning_rate=None,
            muon_learning_rate=None,
            lr_schedule=None,
            lr_decay_to_fraction=None,
            reset_scheduler=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_extend_with_all_defaults_passes(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs())
        # Should not raise.
        _validate_resume_config_modes(cfg, optimizer="adamw")

    def test_extend_with_explicit_decay_only_passes(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(
            lr_schedule="linear", lr_decay_to_fraction=0.001,
        ))
        _validate_resume_config_modes(cfg, optimizer="adamw")

    def test_extend_with_learning_rate_set_raises(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(learning_rate=5e-5))
        with pytest.raises(ValueError, match="reset_scheduler=true"):
            _validate_resume_config_modes(cfg, optimizer="adamw")

    def test_extend_with_muon_learning_rate_set_raises(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(muon_learning_rate=5e-4))
        with pytest.raises(ValueError, match="reset_scheduler=true"):
            _validate_resume_config_modes(cfg, optimizer="muon")

    def test_reset_with_all_required_passes_adamw(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(
            reset_scheduler=True,
            lr_schedule="linear", lr_decay_to_fraction=0.01,
            learning_rate=5e-5,
        ))
        _validate_resume_config_modes(cfg, optimizer="adamw")

    def test_reset_missing_lr_schedule_raises(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(
            reset_scheduler=True,
            lr_decay_to_fraction=0.01, learning_rate=5e-5,
        ))
        with pytest.raises(ValueError, match="lr_schedule"):
            _validate_resume_config_modes(cfg, optimizer="adamw")

    def test_reset_missing_decay_fraction_raises(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(
            reset_scheduler=True,
            lr_schedule="linear", learning_rate=5e-5,
        ))
        with pytest.raises(ValueError, match="lr_decay_to_fraction"):
            _validate_resume_config_modes(cfg, optimizer="adamw")

    def test_reset_missing_learning_rate_raises(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(
            reset_scheduler=True,
            lr_schedule="linear", lr_decay_to_fraction=0.01,
        ))
        with pytest.raises(ValueError, match="learning_rate"):
            _validate_resume_config_modes(cfg, optimizer="adamw")

    def test_reset_muon_requires_both_peaks(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(
            reset_scheduler=True,
            lr_schedule="linear", lr_decay_to_fraction=0.01,
            learning_rate=5e-5,  # missing muon_learning_rate
        ))
        with pytest.raises(ValueError, match="muon_learning_rate"):
            _validate_resume_config_modes(cfg, optimizer="muon")

    def test_reset_muon_with_both_peaks_passes(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(
            reset_scheduler=True,
            lr_schedule="linear", lr_decay_to_fraction=0.01,
            learning_rate=5e-5, muon_learning_rate=5e-4,
        ))
        _validate_resume_config_modes(cfg, optimizer="normuon")

    def test_non_muon_with_muon_lr_raises(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        cfg = ResumeConfig(**self._base_kwargs(
            reset_scheduler=True,
            lr_schedule="linear", lr_decay_to_fraction=0.01,
            learning_rate=5e-5, muon_learning_rate=5e-4,
        ))
        with pytest.raises(ValueError, match="muon_learning_rate"):
            _validate_resume_config_modes(cfg, optimizer="adamw")

    def test_not_resuming_skips_validation(self):
        from nanoplm.cli.pretrain import _validate_resume_config_modes
        # A non-resume config with stray fields should not raise.
        cfg = ResumeConfig(
            is_resume=False, checkpoint_dir="",
            learning_rate=1e-3, reset_scheduler=False,
        )
        _validate_resume_config_modes(cfg, optimizer="adamw")

    def test_yaml_load_rejects_skip_warmup_field(self):
        from nanoplm.cli.pretrain import _load_resume_config
        with pytest.raises(ValueError, match="skip_warmup"):
            _load_resume_config({
                "is_resume": True,
                "checkpoint_dir": ".",
                "skip_warmup": False,
            })


class TestResumeScheduleMath:
    """Unit tests for the EXTEND/RESET schedule curve produced by
    _rebuild_scheduler_for_resume.

    Uses a tiny in-memory optimizer wrapping a single dummy parameter
    so we can exercise the lambda math without spinning up a full
    pipeline.  Verifies the user's repro case: continue from current LR,
    decay linearly to peak * fraction over the resumed window.
    """

    @pytest.fixture
    def saved_pretrain_yaml(self, tmp_path):
        """Write a checkpoint pretrain_config.yaml that the rebuilder reads
        for inheritance defaults.  Returns the checkpoint dir path."""
        ckp = tmp_path / "checkpoint-60"
        ckp.mkdir()
        (ckp / "pretrain_config.yaml").write_text(yaml.safe_dump({
            "adam_learning_rate": 1e-4,
            "muon_learning_rate": 1e-3,
            "lr_decay_to_fraction": 1.0,
            "lr_schedule": "cosine",
            "warmup_steps": 20,
        }))
        return str(ckp)

    def _make_optim(self, lr):
        """Plain torch.optim.SGD over one dummy param at the given LR."""
        param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([param], lr=lr)

    def _step_to(self, scheduler, optimizer, target_step, start_step):
        """Step the scheduler until last_epoch == target_step.

        After scheduler init with last_epoch=start_step-1, the scheduler
        has advanced once (to start_step).  We need (target_step -
        start_step) further .step() calls.
        """
        for _ in range(target_step - start_step):
            scheduler.step()
        return optimizer.param_groups[0]["lr"]

    def test_extend_continues_from_current_lr_and_decays(self, saved_pretrain_yaml):
        """User's repro: peak=current LR, linear decay to peak*0.001 over 12 steps."""
        from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
        from nanoplm.pretraining.pure_pipeline import _rebuild_scheduler_for_resume

        pcfg = PretrainingConfig(dataset_dir="/tmp")  # only LR fields used as fallback
        rcfg = ResumeConfig(
            is_resume=True, checkpoint_dir=saved_pretrain_yaml,
            resume_epochs=2,
            lr_schedule="linear", lr_decay_to_fraction=0.001,
        )
        opt = self._make_optim(1e-4)  # checkpoint's current LR
        start_step = 60
        eff_total = 72

        sched = _rebuild_scheduler_for_resume(
            optimizer=opt, resume_config=rcfg, pretrain_config=pcfg,
            checkpoint_dir=saved_pretrain_yaml,
            start_step=start_step, total_steps=eff_total,
            is_main=False,
        )

        # After init, last_epoch=start_step → lr at start_step = peak.
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4, rel=1e-6)

        # At last_epoch = eff_total - 1 (i.e. last training step's lambda):
        # progress = (12-1)/12; linear factor = max(0.001, 1 - 0.999 * 11/12).
        last_lr = self._step_to(sched, opt, target_step=eff_total - 1,
                                start_step=start_step)
        expected = 1e-4 * max(0.001, 1.0 - 0.999 * 11.0 / 12.0)
        assert last_lr == pytest.approx(expected, rel=1e-6)
        # And it should be monotonically below the peak.
        assert last_lr < 1e-4

    def test_extend_default_decay_targets_original_final_lr(self, tmp_path):
        """Q4(b): when lr_decay_to_fraction is null, end at saved_base * saved_decay."""
        from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
        from nanoplm.pretraining.pure_pipeline import _rebuild_scheduler_for_resume

        # Saved schedule decays cosine 1e-4 -> 1e-5 (fraction 0.1) over 1000 steps.
        ckp = tmp_path / "checkpoint-500"
        ckp.mkdir()
        (ckp / "pretrain_config.yaml").write_text(yaml.safe_dump({
            "adam_learning_rate": 1e-4,
            "muon_learning_rate": 1e-3,
            "lr_decay_to_fraction": 0.1,
            "lr_schedule": "cosine",
            "warmup_steps": 100,
        }))

        pcfg = PretrainingConfig(dataset_dir="/tmp")
        rcfg = ResumeConfig(
            is_resume=True, checkpoint_dir=str(ckp),
            resume_epochs=0,  # not used for total computation here
            # lr_schedule and lr_decay_to_fraction left null -> inherit
        )
        # Pretend we resume mid-decay at LR=5e-5, halfway through.
        opt = self._make_optim(5e-5)
        start_step, eff_total = 500, 1000

        sched = _rebuild_scheduler_for_resume(
            optimizer=opt, resume_config=rcfg, pretrain_config=pcfg,
            checkpoint_dir=str(ckp),
            start_step=start_step, total_steps=eff_total,
            is_main=False,
        )

        # At resume seam (last_epoch=start_step), LR equals current LR (peak).
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-5, rel=1e-6)

        # Auto-fraction = (1e-4 * 0.1) / 5e-5 = 0.2.
        # At last training step's lambda (last_epoch=eff_total-1):
        # cosine progress = 499/500 ≈ 0.998. cosine factor =
        # 0.2 + 0.5*0.8*(1 + cos(pi*0.998)) ≈ 0.2 + 0.4*(1 + cos(pi*0.998))
        # which is just above 0.2.  Final LR should be just above 1e-5.
        last_lr = self._step_to(sched, opt, target_step=eff_total - 1,
                                start_step=start_step)
        # Looser tolerance: just-above-0.2 * 5e-5 ≈ 1e-5.
        assert 1e-5 <= last_lr < 1.5e-5, f"final LR {last_lr:.3e} not near original end 1e-5"

    def test_reset_uses_explicit_peak_and_fraction(self, saved_pretrain_yaml):
        """RESET path: peak comes from resume.learning_rate; fraction explicit."""
        from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
        from nanoplm.pretraining.pure_pipeline import _rebuild_scheduler_for_resume

        pcfg = PretrainingConfig(dataset_dir="/tmp")
        rcfg = ResumeConfig(
            is_resume=True, checkpoint_dir=saved_pretrain_yaml,
            resume_epochs=2,
            reset_scheduler=True,
            learning_rate=2e-5, lr_schedule="linear", lr_decay_to_fraction=0.5,
        )
        # Optimizer's loaded LR (1e-4) should be IGNORED in favor of resume.learning_rate.
        opt = self._make_optim(1e-4)
        start_step, eff_total = 60, 72

        sched = _rebuild_scheduler_for_resume(
            optimizer=opt, resume_config=rcfg, pretrain_config=pcfg,
            checkpoint_dir=saved_pretrain_yaml,
            start_step=start_step, total_steps=eff_total,
            is_main=False,
        )

        # Peak after rebuild = 2e-5 (the new RESET peak), not 1e-4.
        assert opt.param_groups[0]["lr"] == pytest.approx(2e-5, rel=1e-6)

        last_lr = self._step_to(sched, opt, target_step=eff_total - 1,
                                start_step=start_step)
        expected = 2e-5 * max(0.5, 1.0 - 0.5 * 11.0 / 12.0)
        assert last_lr == pytest.approx(expected, rel=1e-6)

    def test_eff_total_at_start_step_raises(self, saved_pretrain_yaml):
        """Resume window with no remaining steps must error."""
        from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
        from nanoplm.pretraining.pure_pipeline import _rebuild_scheduler_for_resume

        pcfg = PretrainingConfig(dataset_dir="/tmp")
        rcfg = ResumeConfig(is_resume=True, checkpoint_dir=saved_pretrain_yaml)
        opt = self._make_optim(1e-4)

        with pytest.raises(ValueError, match="eff_total"):
            _rebuild_scheduler_for_resume(
                optimizer=opt, resume_config=rcfg, pretrain_config=pcfg,
                checkpoint_dir=saved_pretrain_yaml,
                start_step=60, total_steps=60,
                is_main=False,
            )


class TestHFResumeWarning:
    """Verify the HF pipeline warns when ResumeConfig fields it can't honor
    are set, since the HF Trainer path bit-exactly resumes its own scheduler
    and ignores the EXTEND/RESET semantics.

    Uses mock.patch on the module-level logger because the project logger
    sets ``propagate=False`` and its handlers are attached before pytest's
    capture machinery hooks in.
    """

    def test_warns_when_lr_decay_set_on_hf(self):
        from unittest.mock import patch
        from nanoplm.pretraining.pipeline import _warn_unsupported_resume_fields_on_hf
        rcfg = ResumeConfig(
            is_resume=True, checkpoint_dir="/tmp/ckpt",
            lr_decay_to_fraction=0.001, lr_schedule="linear",
        )
        with patch("nanoplm.pretraining.pipeline.logger") as mock_logger:
            _warn_unsupported_resume_fields_on_hf(rcfg)
        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        assert "HF Trainer pipeline ignores" in msg
        assert "lr_decay_to_fraction" in msg
        assert "lr_schedule" in msg

    def test_warns_when_reset_scheduler_set_on_hf(self):
        from unittest.mock import patch
        from nanoplm.pretraining.pipeline import _warn_unsupported_resume_fields_on_hf
        rcfg = ResumeConfig(
            is_resume=True, checkpoint_dir="/tmp/ckpt",
            reset_scheduler=True, learning_rate=5e-5,
            lr_schedule="linear", lr_decay_to_fraction=0.01,
        )
        with patch("nanoplm.pretraining.pipeline.logger") as mock_logger:
            _warn_unsupported_resume_fields_on_hf(rcfg)
        msg = mock_logger.warning.call_args[0][0]
        assert "reset_scheduler" in msg
        assert "learning_rate" in msg

    def test_silent_when_only_resume_epochs_set(self):
        from unittest.mock import patch
        from nanoplm.pretraining.pipeline import _warn_unsupported_resume_fields_on_hf
        rcfg = ResumeConfig(
            is_resume=True, checkpoint_dir="/tmp/ckpt",
            resume_epochs=2,
        )
        with patch("nanoplm.pretraining.pipeline.logger") as mock_logger:
            _warn_unsupported_resume_fields_on_hf(rcfg)
        # resume_epochs IS honored by HF Trainer (it just continues), so no warning.
        mock_logger.warning.assert_not_called()

    def test_silent_when_not_resuming(self):
        from unittest.mock import patch
        from nanoplm.pretraining.pipeline import _warn_unsupported_resume_fields_on_hf
        with patch("nanoplm.pretraining.pipeline.logger") as mock_logger:
            _warn_unsupported_resume_fields_on_hf(None)
        mock_logger.warning.assert_not_called()
