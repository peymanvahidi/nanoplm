"""Resume on a different rank count: the saved per-rank data position is
rescaled to the new count, with a warning; old checkpoints are left alone."""

import json
from pathlib import Path

import torch

from nanoplm.pretraining.pure_pipeline import _rescale_resume_data_step, _save_checkpoint


def test_same_world_size_unchanged():
    assert _rescale_resume_data_step(100, 16, 16, 4, 4) == (100, [])


def test_old_checkpoint_without_keys_unchanged():
    assert _rescale_resume_data_step(100, None, 32, None, 4) == (100, [])


def test_scale_up_halves_position():
    step, warns = _rescale_resume_data_step(101, 16, 32)
    assert step == round(101 * 16 / 32)
    assert len(warns) == 1
    assert "16 ranks" in warns[0] and "32 ranks" in warns[0]
    assert "estimate" in warns[0]


def test_scale_down_doubles_position():
    step, warns = _rescale_resume_data_step(100, 32, 16)
    assert step == 200 and len(warns) == 1


def test_num_workers_change_warns():
    step, warns = _rescale_resume_data_step(100, 8, 8, 4, 2)
    assert step == 100
    assert len(warns) == 1 and "num_workers=2" in warns[0] and "num_workers=4" in warns[0]


def test_training_state_saves_world_size_and_num_workers(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    _save_checkpoint(model, opt, sched, global_step=5, epoch=0,
                     output_dir=str(tmp_path), logging_steps=1, eval_steps=1, save_steps=1,
                     epoch_data_step=7, world_size=16, num_workers=3)
    state = json.loads((Path(tmp_path) / "checkpoint-5" / "training_state.json").read_text())
    assert state["world_size"] == 16 and state["num_workers"] == 3
    assert state["epoch_data_step"] == 7
