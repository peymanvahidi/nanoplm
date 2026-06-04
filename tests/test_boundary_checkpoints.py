from nanoplm.pretraining.pure_pipeline import _checkpoint_dir_name


class TestCheckpointDirName:
    def test_periodic(self):
        assert _checkpoint_dir_name(global_step=500, phase="periodic") == "checkpoint-500"

    def test_stable_end_role(self):
        assert _checkpoint_dir_name(global_step=36, phase="stable-end") == "checkpoint-stable-end"

    def test_final_role(self):
        assert _checkpoint_dir_name(global_step=40, phase="final") == "checkpoint-final"


import json, torch
from pathlib import Path
from nanoplm.pretraining.pure_pipeline import _save_checkpoint
from nanoplm.pretraining.config import PretrainingConfig


def test_training_state_carries_schedule_descriptor(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    cfg = PretrainingConfig(dataset_dir="/tmp", lr_schedule="wsd", decay_shape="1-sqrt",
                            lr_decay_to_fraction=0.1)
    _save_checkpoint(model, opt, sched, global_step=36, epoch=0,
                     output_dir=str(tmp_path), logging_steps=1, eval_steps=1, save_steps=1,
                     distributed=False, is_main=True, pretrain_config=cfg,
                     phase="stable-end", tokens_seen=9_000_000,
                     achieved_global_batch_tokens=250_000, stable_steps=36, decay_steps=4)
    state = json.loads((Path(tmp_path) / "checkpoint-stable-end" / "training_state.json").read_text())
    assert state["phase"] == "stable-end"
    assert state["lr_schedule"] == "wsd"
    assert state["tokens_seen"] == 9_000_000
    assert state["stable_steps"] == 36 and state["decay_steps"] == 4
