"""Resume on a different layout: the saved per-rank data position is rescaled
by pass fraction (or by rank ratio when the pass length was not saved), with a
warning; old checkpoints keep their position and get one caveat."""

import json
from pathlib import Path

import torch

from nanoplm.pretraining.pure_pipeline import _rescale_resume_data_step, _save_checkpoint


def test_same_world_size_unchanged():
    assert _rescale_resume_data_step(100, 16, 16, 4, 4) == (100, False, [])
    assert _rescale_resume_data_step(100, 16, 16, 4, 4, 826, 826) == (100, False, [])


def test_old_checkpoint_without_keys_unchanged():
    step, changed, warns = _rescale_resume_data_step(100, None, 32, None, 4)
    assert step == 100 and changed is False
    assert len(warns) == 1
    assert "does not record its GPU count" in warns[0] and "micro-step 100" in warns[0]


def test_scale_up_halves_position():
    step, changed, warns = _rescale_resume_data_step(101, 16, 32)
    assert step == round(101 * 16 / 32) and changed
    assert len(warns) == 1
    assert "16 -> 32 GPUs" in warns[0]
    assert "may repeat or be skipped" in warns[0]


def test_scale_down_doubles_position():
    step, changed, warns = _rescale_resume_data_step(100, 32, 16)
    assert step == 200 and changed and len(warns) == 1


def test_num_workers_change_warns():
    step, changed, warns = _rescale_resume_data_step(100, 8, 8, 4, 2)
    assert step == 100 and changed
    assert len(warns) == 1 and "num_workers 4 -> 2" in warns[0]


def test_pass_fraction_and_pass_end():
    # Numbers from investigate/b/resume_sim.out (W=16, nw=4, 826 micro-steps).
    step, changed, warns = _rescale_resume_data_step(413, 16, 8, 4, 4, 826, 1654)
    assert (step, changed) == (827, True) and len(warns) == 1
    assert "16 -> 8 GPUs" in warns[0] and "pass length 826 -> 1654" in warns[0]
    # A finished pass maps to the end of the new pass: next pass, no message.
    assert _rescale_resume_data_step(826, 16, 8, 4, 4, 826, 1654) == (1654, True, [])
    step, _, _ = _rescale_resume_data_step(825, 16, 32, 4, 4, 826, 409)
    assert step == 409
    # Exactly at the end of an unchanged pass: next pass, unchanged layout.
    assert _rescale_resume_data_step(826, 16, 16, 4, 4, 826, 826) == (826, False, [])
    # saved_len == 0 (degenerate): any step counts as a finished pass, no crash.
    assert _rescale_resume_data_step(5, 16, 16, 4, 4, 0, 830) == (830, True, [])
    # micro_batch_seqs changed on the same GPUs: only the pass length differs.
    step, changed, warns = _rescale_resume_data_step(413, 16, 16, 4, 4, 826, 1685)
    assert step == round(413 * 1685 / 826) and changed
    assert "pass length 826 -> 1685" in warns[0] and "GPUs" not in warns[0]


def test_quiet_at_step_zero_and_old_checkpoint_at_zero():
    assert _rescale_resume_data_step(0, 16, 32, 4, 4) == (0, True, [])
    assert _rescale_resume_data_step(0, 16, 32, 4, 4, 826, 409) == (0, True, [])
    assert _rescale_resume_data_step(0, None, 32, None, 4) == (0, False, [])


def test_num_workers_only_change_sets_layout_changed():
    step, changed, _ = _rescale_resume_data_step(5, 16, 16, 4, 8, 826, 826)
    assert step == 5 and changed is True


def test_unchanged_step_says_kept():
    step, changed, warns = _rescale_resume_data_step(100, 16, 16, 4, 8, 826, 826)
    assert (step, changed) == (100, True) and len(warns) == 1
    assert "data position kept at micro-step 100" in warns[0]
    assert "rescaled" not in warns[0] and "may repeat or be skipped" in warns[0]


def test_training_state_saves_world_size_and_num_workers(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    _save_checkpoint(model, opt, sched, global_step=5, epoch=0,
                     output_dir=str(tmp_path), logging_steps=1, eval_steps=1, save_steps=1,
                     epoch_data_step=7, world_size=16, num_workers=3, epoch_data_len=826)
    state = json.loads((Path(tmp_path) / "checkpoint-5" / "training_state.json").read_text())
    assert state["world_size"] == 16 and state["num_workers"] == 3
    assert state["epoch_data_step"] == 7 and state["epoch_data_len"] == 826
