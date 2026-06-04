import json, pytest, torch
from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
from nanoplm.pretraining.pure_pipeline import _rebuild_scheduler_for_resume


def _trunk_ckpt(tmp_path, lr=1e-4):
    ckp = tmp_path / "checkpoint-stable-end"
    ckp.mkdir()
    (ckp / "training_state.json").write_text(json.dumps({
        "global_step": 36, "phase": "stable-end", "lr_schedule": "wsd",
        "achieved_global_batch_tokens": 250_000, "stable_steps": 36, "decay_steps": 4,
    }))
    return str(ckp)


def _opt(lr):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


class TestResumeDecayMath:
    def test_decay_starts_at_checkpoint_lr_and_reaches_floor(self, tmp_path):
        ckp = _trunk_ckpt(tmp_path, lr=1e-4)
        rcfg = ResumeConfig(is_resume=True, checkpoint_dir=ckp, mode="decay",
                            decay_steps=4, decay_shape="linear", lr_decay_to_fraction=0.1)
        opt = _opt(1e-4)  # the trunk's stable LR
        sched = _rebuild_scheduler_for_resume(
            optimizer=opt, resume_config=rcfg, pretrain_config=PretrainingConfig(dataset_dir="/tmp"),
            checkpoint_dir=ckp, decay_steps=4)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4, rel=1e-6)  # peak = checkpoint LR
        for _ in range(3):
            sched.step()
        # last decay step (step 3 of 4): 0.1 + 0.9*(1-3/4)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4 * (0.1 + 0.9 * 0.25), rel=1e-6)
