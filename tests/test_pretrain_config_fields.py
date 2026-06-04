from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig


def test_num_epochs_is_gone():
    cfg = PretrainingConfig(dataset_dir="/tmp")
    assert not hasattr(cfg, "num_epochs")


def test_new_duration_fields_exist_with_defaults():
    cfg = PretrainingConfig(dataset_dir="/tmp")
    assert cfg.lr_schedule == "wsd"
    assert cfg.warmup_steps == 1000
    assert cfg.stable_tokens is None
    assert cfg.decay_tokens is None
    assert cfg.train_tokens is None
    assert cfg.decay_shape == "1-sqrt"
    assert cfg.stable_steps is None and cfg.decay_steps is None and cfg.max_steps is None


def test_resume_config_has_modes_not_legacy_fields():
    rc = ResumeConfig(is_resume=False, checkpoint_dir="")
    assert rc.mode == "continue"
    assert rc.decay_tokens is None and rc.decay_steps is None
    assert rc.decay_shape is None and rc.lr_decay_to_fraction is None
    for legacy in ("resume_epochs", "reset_scheduler", "learning_rate",
                   "muon_learning_rate", "lr_schedule"):
        assert not hasattr(rc, legacy), f"{legacy} should be removed"
