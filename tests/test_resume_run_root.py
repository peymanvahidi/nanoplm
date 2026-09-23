"""Resuming into a fresh ckp_dir must create the run directory it writes into.

`prepare_run_and_steps` derives `run_root` from the resumed checkpoint's parent
NAME rebased onto the configured `ckp_dir`, not from where the checkpoint
actually lives. Pointing a resume at a new `ckp_dir`, which is how you keep the
source run's checkpoints read-only, therefore hit a FileNotFoundError writing
`.resume_counter` into a directory nobody had created.
"""

import json
from pathlib import Path

from nanoplm.pretraining.config import PretrainingConfig, ResumeConfig
from nanoplm.pretraining.utils import prepare_run_and_steps


def _source_checkpoint(root: Path, step: int = 500) -> Path:
    ckpt = root / "src-run" / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    (ckpt / "training_state.json").write_text(
        json.dumps({"global_step": step, "epoch": 0}), encoding="utf-8"
    )
    return ckpt


def test_resume_into_fresh_ckp_dir_creates_run_root(tmp_path):
    ckpt = _source_checkpoint(tmp_path / "source")
    fresh = tmp_path / "fresh"

    run = prepare_run_and_steps(
        PretrainingConfig(dataset_dir=str(tmp_path / "data"), ckp_dir=str(fresh)),
        ResumeConfig(is_resume=True, checkpoint_dir=str(ckpt)),
        effective_world_size=1,
    )

    run_root = fresh / "src-run"
    assert run_root.is_dir(), "run_root was not created"
    assert (run_root / ".resume_counter").read_text().strip() == "1"
    assert run.run_name == "src-run"
    assert run.wandb_run_name == "src-run-re1"
    assert run.run_root == run_root
    assert run.is_resume and run.resume_step == 500


def test_second_resume_increments_the_counter(tmp_path):
    ckpt = _source_checkpoint(tmp_path / "source")
    fresh = tmp_path / "fresh"
    args = (
        PretrainingConfig(dataset_dir=str(tmp_path / "data"), ckp_dir=str(fresh)),
        ResumeConfig(is_resume=True, checkpoint_dir=str(ckpt)),
    )
    prepare_run_and_steps(*args, effective_world_size=1)
    run = prepare_run_and_steps(*args, effective_world_size=1)
    assert run.wandb_run_name == "src-run-re2"
    assert (fresh / "src-run" / ".resume_counter").read_text().strip() == "2"
