"""
Tests for LR scheduler creation and resume-with-override logic in the pure pipeline.

Covers:
- _create_scheduler: warmup, linear/cosine decay, last_epoch positioning
- _rebuild_scheduler_for_resume: reset_scheduler, skip_warmup, lr_schedule
  override, lr_decay_to_fraction override, learning_rate override,
  checkpoint config loading and fallback
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml
from torch.optim import AdamW

from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
from nanoplm.pretraining.pure_pipeline import (
    _create_scheduler,
    _rebuild_scheduler_for_resume,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_optimizer(lr: float = 1e-3) -> AdamW:
    """Create a minimal AdamW optimizer for scheduler tests."""
    params = [torch.nn.Parameter(torch.randn(4, 4))]
    return AdamW(params, lr=lr)


def _step_scheduler(scheduler, n: int) -> list[float]:
    """Step the scheduler *n* times and return the LR after each step."""
    lrs = []
    for _ in range(n):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    return lrs


def _make_pretrain_config(**overrides) -> PretrainingConfig:
    """Create a PretrainingConfig with a temp dataset_dir and optional overrides."""
    defaults = dict(
        dataset_dir="/tmp/fake_dataset",
        warmup_steps=10,
        adam_learning_rate=1e-3,
        muon_learning_rate=1e-2,
        lr_decay_to_fraction=0.1,
        lr_schedule="cosine",
    )
    defaults.update(overrides)
    return PretrainingConfig(**defaults)


def _make_resume_config(**overrides) -> ResumeConfig:
    """Create a ResumeConfig with sensible defaults."""
    defaults = dict(is_resume=True, checkpoint_dir="/tmp/fake_ckpt")
    defaults.update(overrides)
    return ResumeConfig(**defaults)


def _write_checkpoint_metadata(
    ckpt_dir: Path,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    pretrain_config: PretrainingConfig,
) -> None:
    """Write training_state.json and pretrain_config.yaml into a checkpoint dir."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "global_step": global_step,
        "epoch": 0,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    (ckpt_dir / "training_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )

    pc_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in pretrain_config.__dict__.items()
    }
    with open(ckpt_dir / "pretrain_config.yaml", "w") as f:
        yaml.dump(pc_dict, f, default_flow_style=False, sort_keys=False)


# =========================================================================
# _create_scheduler tests
# =========================================================================


class TestCreateScheduler:
    """Unit tests for _create_scheduler."""

    def test_warmup_ramp(self):
        """LR should ramp linearly from 0 to base_lr during warmup."""
        opt = _make_optimizer(lr=1e-3)
        warmup = 10
        total = 100
        sched = _create_scheduler(opt, warmup, total, 1e-3, 0.1, "linear")

        lrs = _step_scheduler(sched, warmup)
        # Step 1 should be ~1/10 of base, step 10 should be ~1.0 * base
        assert lrs[0] == pytest.approx(1e-3 * (1 / 10), rel=1e-5)
        assert lrs[-1] == pytest.approx(1e-3, rel=1e-5)

    def test_linear_decay_end(self):
        """After all steps, linear schedule should reach lr_decay_to_fraction."""
        opt = _make_optimizer(lr=1e-3)
        warmup = 10
        total = 100
        decay_frac = 0.1
        sched = _create_scheduler(opt, warmup, total, 1e-3, decay_frac, "linear")

        lrs = _step_scheduler(sched, total)
        assert lrs[-1] == pytest.approx(1e-3 * decay_frac, rel=1e-4)

    def test_cosine_decay_end(self):
        """After all steps, cosine schedule should reach lr_decay_to_fraction."""
        opt = _make_optimizer(lr=1e-3)
        warmup = 10
        total = 100
        decay_frac = 0.1
        sched = _create_scheduler(opt, warmup, total, 1e-3, decay_frac, "cosine")

        lrs = _step_scheduler(sched, total)
        assert lrs[-1] == pytest.approx(1e-3 * decay_frac, rel=1e-4)

    def test_cosine_higher_than_linear_early(self):
        """In early decay phase, cosine should be higher than linear."""
        opt_c = _make_optimizer(lr=1e-3)
        opt_l = _make_optimizer(lr=1e-3)
        warmup = 0
        total = 100
        decay_frac = 0.0

        sched_c = _create_scheduler(opt_c, warmup, total, 1e-3, decay_frac, "cosine")
        sched_l = _create_scheduler(opt_l, warmup, total, 1e-3, decay_frac, "linear")

        # At step 25 (25% progress), cosine is still high, linear has dropped more
        lrs_c = _step_scheduler(sched_c, 25)
        lrs_l = _step_scheduler(sched_l, 25)
        assert lrs_c[-1] > lrs_l[-1]

    def test_last_epoch_positions_scheduler(self):
        """With last_epoch, scheduler should start at that step's LR."""
        opt = _make_optimizer(lr=1e-3)
        warmup = 10
        total = 100
        decay_frac = 0.1

        # Create scheduler positioned at step 50
        sched = _create_scheduler(
            opt, warmup, total, 1e-3, decay_frac, "linear", last_epoch=49
        )
        lr_at_50 = sched.get_last_lr()[0]

        # Compare with a fresh scheduler stepped to 50
        opt2 = _make_optimizer(lr=1e-3)
        sched2 = _create_scheduler(opt2, warmup, total, 1e-3, decay_frac, "linear")
        lrs = _step_scheduler(sched2, 50)

        assert lr_at_50 == pytest.approx(lrs[-1], rel=1e-5)

    def test_constant_schedule_after_warmup(self):
        """Constant schedule should hold peak LR after warmup with no decay."""
        opt = _make_optimizer(lr=1e-3)
        warmup = 10
        total = 100
        sched = _create_scheduler(opt, warmup, total, 1e-3, 0.1, "constant")

        # Step through warmup
        lrs = _step_scheduler(sched, warmup)
        assert lrs[-1] == pytest.approx(1e-3, rel=1e-5)

        # Step well past warmup — LR should stay at peak
        lrs_post = _step_scheduler(sched, total - warmup)
        assert lrs_post[0] == pytest.approx(1e-3, rel=1e-5)
        assert lrs_post[len(lrs_post) // 2] == pytest.approx(1e-3, rel=1e-5)
        assert lrs_post[-1] == pytest.approx(1e-3, rel=1e-5)

    def test_constant_ignores_decay_fraction(self):
        """Constant schedule should return 1.0 regardless of lr_decay_to_fraction."""
        opt = _make_optimizer(lr=1e-3)
        sched = _create_scheduler(opt, 0, 100, 1e-3, 0.01, "constant")

        lrs = _step_scheduler(sched, 100)
        # Every step should be at peak, not decaying to 0.01
        assert all(lr == pytest.approx(1e-3, rel=1e-5) for lr in lrs)

    def test_zero_warmup(self):
        """With warmup=0, LR should start near peak and decay immediately."""
        opt = _make_optimizer(lr=1e-3)
        sched = _create_scheduler(opt, 0, 100, 1e-3, 0.1, "linear")

        # Step 1: progress = 1/100 = 0.01, lr_mult = max(0.1, 1 - 0.9*0.01) = 0.991
        lrs = _step_scheduler(sched, 1)
        assert lrs[0] == pytest.approx(1e-3 * 0.991, rel=1e-4)
        # Should never go through a warmup ramp — even step 1 is near peak
        assert lrs[0] > 1e-3 * 0.98


# =========================================================================
# _rebuild_scheduler_for_resume tests
# =========================================================================


class TestRebuildSchedulerForResume:
    """Unit tests for _rebuild_scheduler_for_resume."""

    def test_reset_scheduler_creates_fresh_curve(self):
        """reset_scheduler=True should create a schedule from step 0 over remaining steps."""
        opt = _make_optimizer(lr=1e-3)
        pretrain_cfg = _make_pretrain_config()
        resume_cfg = _make_resume_config(reset_scheduler=True)

        start_step = 50
        total_steps = 100
        warmup_steps = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir)
            _write_checkpoint_metadata(ckpt, start_step, total_steps, warmup_steps, pretrain_cfg)
            resume_cfg.checkpoint_dir = str(ckpt)

            sched = _rebuild_scheduler_for_resume(
                opt, resume_cfg, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )

        # Remaining steps = 100 - 50 = 50. The new schedule should cover 50 steps.
        # After stepping through all 50 remaining steps, LR should reach decay floor.
        lrs = _step_scheduler(sched, 50)
        expected_floor = 1e-3 * pretrain_cfg.lr_decay_to_fraction
        assert lrs[-1] == pytest.approx(expected_floor, rel=1e-3)

    def test_skip_warmup_starts_at_peak(self):
        """skip_warmup=True should set warmup to 0, starting at peak LR."""
        opt = _make_optimizer(lr=1e-3)
        pretrain_cfg = _make_pretrain_config(warmup_steps=20)
        resume_cfg = _make_resume_config(reset_scheduler=True, skip_warmup=True)

        start_step = 50
        total_steps = 100
        warmup_steps = 20

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir)
            _write_checkpoint_metadata(ckpt, start_step, total_steps, warmup_steps, pretrain_cfg)
            resume_cfg.checkpoint_dir = str(ckpt)

            sched = _rebuild_scheduler_for_resume(
                opt, resume_cfg, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )

        # At step 0 (after init), LR should be at peak (no warmup)
        assert sched.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-5)
        # After one training step, should still be very close to peak
        lrs = _step_scheduler(sched, 1)
        assert lrs[0] > 1e-3 * 0.99

    def test_lr_schedule_override(self):
        """Overriding lr_schedule from cosine to linear should change the curve."""
        pretrain_cfg = _make_pretrain_config(lr_schedule="cosine")

        start_step = 50
        total_steps = 100
        warmup_steps = 10

        # Build with cosine (default from checkpoint)
        opt_c = _make_optimizer(lr=1e-3)
        resume_cosine = _make_resume_config(reset_scheduler=True, skip_warmup=True)

        # Build with linear override
        opt_l = _make_optimizer(lr=1e-3)
        resume_linear = _make_resume_config(
            reset_scheduler=True, skip_warmup=True, lr_schedule="linear"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir)
            _write_checkpoint_metadata(ckpt, start_step, total_steps, warmup_steps, pretrain_cfg)
            resume_cosine.checkpoint_dir = str(ckpt)
            resume_linear.checkpoint_dir = str(ckpt)

            sched_c = _rebuild_scheduler_for_resume(
                opt_c, resume_cosine, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )
            sched_l = _rebuild_scheduler_for_resume(
                opt_l, resume_linear, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )

        # At ~1/3 progress (step 16 of 50 remaining), cosine decays slower
        lrs_c = _step_scheduler(sched_c, 16)
        lrs_l = _step_scheduler(sched_l, 16)
        assert lrs_c[-1] > lrs_l[-1]

    def test_lr_schedule_override_to_constant(self):
        """Overriding lr_schedule to constant should hold LR at peak after warmup."""
        pretrain_cfg = _make_pretrain_config(lr_schedule="cosine")

        start_step = 50
        total_steps = 100
        warmup_steps = 10

        opt = _make_optimizer(lr=1e-3)
        resume_cfg = _make_resume_config(
            reset_scheduler=True, skip_warmup=True, lr_schedule="constant"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir)
            _write_checkpoint_metadata(ckpt, start_step, total_steps, warmup_steps, pretrain_cfg)
            resume_cfg.checkpoint_dir = str(ckpt)

            sched = _rebuild_scheduler_for_resume(
                opt, resume_cfg, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )

        # Every step should stay at peak LR
        remaining = total_steps - start_step
        lrs = _step_scheduler(sched, remaining)
        assert all(lr == pytest.approx(1e-3, rel=1e-5) for lr in lrs)

    def test_lr_decay_to_fraction_override(self):
        """Overriding lr_decay_to_fraction should change the floor LR."""
        opt = _make_optimizer(lr=1e-3)
        pretrain_cfg = _make_pretrain_config(lr_decay_to_fraction=0.1)
        resume_cfg = _make_resume_config(
            reset_scheduler=True, skip_warmup=True, lr_decay_to_fraction=0.5
        )

        start_step = 0
        total_steps = 100
        warmup_steps = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir)
            _write_checkpoint_metadata(ckpt, start_step, total_steps, warmup_steps, pretrain_cfg)
            resume_cfg.checkpoint_dir = str(ckpt)

            sched = _rebuild_scheduler_for_resume(
                opt, resume_cfg, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )

        lrs = _step_scheduler(sched, 100)
        # Floor should be 0.5 * base_lr, not 0.1 * base_lr
        assert lrs[-1] == pytest.approx(1e-3 * 0.5, rel=1e-3)

    def test_learning_rate_override(self):
        """Overriding learning_rate should change the peak LR."""
        opt = _make_optimizer(lr=1e-3)
        new_lr = 5e-4
        pretrain_cfg = _make_pretrain_config(adam_learning_rate=1e-3)
        resume_cfg = _make_resume_config(
            reset_scheduler=True, skip_warmup=True, learning_rate=new_lr
        )

        start_step = 50
        total_steps = 100
        warmup_steps = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir)
            _write_checkpoint_metadata(ckpt, start_step, total_steps, warmup_steps, pretrain_cfg)
            resume_cfg.checkpoint_dir = str(ckpt)

            sched = _rebuild_scheduler_for_resume(
                opt, resume_cfg, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )

        # At step 0, scheduler should be at peak = new_lr
        assert sched.get_last_lr()[0] == pytest.approx(new_lr, rel=1e-5)
        # Optimizer base LR was overridden
        assert opt.param_groups[0]["initial_lr"] == pytest.approx(new_lr, rel=1e-6)

    def test_fallback_to_current_config_without_checkpoint_yaml(self):
        """Without pretrain_config.yaml in checkpoint, should use current config."""
        opt = _make_optimizer(lr=1e-3)
        pretrain_cfg = _make_pretrain_config(
            lr_schedule="linear", lr_decay_to_fraction=0.2
        )
        resume_cfg = _make_resume_config(reset_scheduler=True, skip_warmup=True)

        start_step = 50
        total_steps = 100
        warmup_steps = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir)
            # Only write training_state.json, NOT pretrain_config.yaml
            state = {
                "global_step": start_step,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            }
            (ckpt / "training_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            resume_cfg.checkpoint_dir = str(ckpt)

            sched = _rebuild_scheduler_for_resume(
                opt, resume_cfg, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )

        remaining = total_steps - start_step  # 50
        lrs = _step_scheduler(sched, remaining)
        # Should use pretrain_cfg's lr_decay_to_fraction=0.2
        assert lrs[-1] == pytest.approx(1e-3 * 0.2, rel=1e-3)

    def test_no_reset_reconstructs_original_curve(self):
        """Without reset_scheduler, schedule should continue from where it left off."""
        opt = _make_optimizer(lr=1e-3)
        pretrain_cfg = _make_pretrain_config(
            lr_schedule="linear", lr_decay_to_fraction=0.1
        )
        resume_cfg = _make_resume_config(reset_scheduler=False)

        start_step = 50
        total_steps = 100
        warmup_steps = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir)
            _write_checkpoint_metadata(ckpt, start_step, total_steps, warmup_steps, pretrain_cfg)
            resume_cfg.checkpoint_dir = str(ckpt)

            sched = _rebuild_scheduler_for_resume(
                opt, resume_cfg, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )

        # The scheduler should be positioned at step 50.
        # Stepping to 100 should reach the decay floor.
        remaining = total_steps - start_step
        lrs = _step_scheduler(sched, remaining)
        assert lrs[-1] == pytest.approx(1e-3 * 0.1, rel=1e-3)

    def test_warmup_steps_override(self):
        """Explicitly overriding warmup_steps should change warmup duration."""
        opt = _make_optimizer(lr=1e-3)
        pretrain_cfg = _make_pretrain_config(warmup_steps=10)
        resume_cfg = _make_resume_config(
            reset_scheduler=True, warmup_steps=5
        )

        start_step = 50
        total_steps = 100
        warmup_steps = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir)
            _write_checkpoint_metadata(ckpt, start_step, total_steps, warmup_steps, pretrain_cfg)
            resume_cfg.checkpoint_dir = str(ckpt)

            sched = _rebuild_scheduler_for_resume(
                opt, resume_cfg, pretrain_cfg, str(ckpt),
                start_step, total_steps, warmup_steps, is_main=False,
            )

        # With warmup=5 over remaining=50, step 5 should be at peak
        lrs = _step_scheduler(sched, 5)
        assert lrs[-1] == pytest.approx(1e-3, rel=1e-4)
        # Step 3 should still be ramping
        assert lrs[2] < lrs[-1]


# =========================================================================
# Integration: save checkpoint -> resume with overrides
# =========================================================================


class TestSaveAndResumeIntegration:
    """Integration test: create a real checkpoint, then resume with scheduler override."""

    @staticmethod
    def _make_tiny_model():
        from nanoplm.pretraining.models.modern_bert.model import (
            ProtModernBertMLM,
            ProtModernBertMLMConfig,
        )
        cfg = ProtModernBertMLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=32,
        )
        return ProtModernBertMLM(cfg)

    def test_save_resume_with_reset_scheduler(self):
        """Full cycle: train -> save -> resume with reset_scheduler + linear decay."""
        from nanoplm.pretraining.pure_pipeline import _save_checkpoint, _load_checkpoint

        model = self._make_tiny_model().to("cpu")
        optimizer = AdamW(model.parameters(), lr=1e-3)
        pretrain_cfg = _make_pretrain_config(
            lr_schedule="cosine",
            lr_decay_to_fraction=0.1,
            warmup_steps=5,
        )
        total_steps = 50
        warmup_steps = 5

        # Create initial scheduler and step it 20 times (simulating training)
        sched = _create_scheduler(
            optimizer, warmup_steps, total_steps, 1e-3, 0.1, "cosine"
        )
        for _ in range(20):
            sched.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint at step 20
            _save_checkpoint(
                model, optimizer, sched,
                global_step=20, epoch=0,
                output_dir=tmpdir,
                logging_steps=1, eval_steps=10, save_steps=10,
                distributed=False, is_main=True,
                pretrain_config=pretrain_cfg,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )

            ckpt_dir = Path(tmpdir) / "checkpoint-20"
            assert ckpt_dir.exists()
            assert (ckpt_dir / "pretrain_config.yaml").exists()
            assert (ckpt_dir / "training_state.json").exists()

            # Now resume with reset_scheduler + linear + different decay
            resume_cfg = _make_resume_config(
                checkpoint_dir=str(ckpt_dir),
                reset_scheduler=True,
                lr_schedule="linear",
                lr_decay_to_fraction=0.01,
                skip_warmup=True,
            )

            # Create fresh model + optimizer (simulating a new process)
            model2 = self._make_tiny_model().to("cpu")
            opt2 = AdamW(model2.parameters(), lr=1e-3)
            dummy_sched = _create_scheduler(opt2, warmup_steps, total_steps, 1e-3, 0.1, "cosine")

            # Load checkpoint (skip scheduler loading)
            step, epoch = _load_checkpoint(
                model2, opt2, dummy_sched, str(ckpt_dir), "cpu",
                distributed=False, load_scheduler=False,
            )
            assert step == 20

            # Rebuild scheduler with overrides
            new_sched = _rebuild_scheduler_for_resume(
                opt2, resume_cfg, pretrain_cfg, str(ckpt_dir),
                start_step=20, total_steps=total_steps,
                warmup_steps=warmup_steps, is_main=False,
            )

            # Verify: at step 0, LR should be at peak (skip_warmup=True)
            assert new_sched.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-4)

            # Verify: stepping to the end should reach new decay floor (0.01)
            remaining = total_steps - 20
            lrs_rest = _step_scheduler(new_sched, remaining)
            assert lrs_rest[-1] == pytest.approx(1e-3 * 0.01, rel=1e-2)

            # Verify: model weights were loaded (not random)
            orig_params = dict(model.named_parameters())
            loaded_params = dict(model2.named_parameters())
            for name in orig_params:
                assert torch.allclose(
                    orig_params[name].detach().cpu(),
                    loaded_params[name].detach().cpu(),
                ), f"Parameter {name} not restored correctly"
