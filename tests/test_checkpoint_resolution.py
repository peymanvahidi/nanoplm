"""Checkpoint resolution & archival must rank role-named dirs by their true step.

Role-named checkpoint directories (`checkpoint-final`, `checkpoint-stable-end`)
have no numeric suffix to parse. Their true optimizer step lives in
`training_state.json["global_step"]`. Both the eval-side resolver
(`resolve_checkpoint_dir` / `_checkpoint_step`) and the pretraining-side
archiver (`_archive_future_checkpoints`) must read that real step so that
`checkpoint-final` (the terminal model) wins resolution and role dirs are
archived on resume.
"""

import json
from pathlib import Path

from nanoplm.eval.checkpoint import _checkpoint_step, resolve_checkpoint_dir
from nanoplm.pretraining.utils import _archive_future_checkpoints


def _make_ckpt(run_dir: Path, name: str, global_step: int) -> Path:
    ckpt = run_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "training_state.json").write_text(
        json.dumps({"global_step": global_step}), encoding="utf-8"
    )
    return ckpt


def _build_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run-abc"
    run_dir.mkdir()
    _make_ckpt(run_dir, "checkpoint-100", 100)
    _make_ckpt(run_dir, "checkpoint-200", 200)
    _make_ckpt(run_dir, "checkpoint-stable-end", 200)
    _make_ckpt(run_dir, "checkpoint-final", 240)
    return run_dir


def test_checkpoint_step_reads_role_dirs_true_step(tmp_path):
    run_dir = _build_run(tmp_path)
    # Role dirs resolve to their global_step, NOT -1.
    assert _checkpoint_step(run_dir / "checkpoint-final") == 240
    assert _checkpoint_step(run_dir / "checkpoint-stable-end") == 200
    # Numbered dirs still parse from the name.
    assert _checkpoint_step(run_dir / "checkpoint-200") == 200
    assert _checkpoint_step(run_dir / "checkpoint-100") == 100
    # Non-checkpoint names stay -1.
    assert _checkpoint_step(run_dir) == -1


def test_resolve_picks_terminal_final(tmp_path):
    run_dir = _build_run(tmp_path)
    resolved = resolve_checkpoint_dir(run_dir)
    assert resolved.name == "checkpoint-final"


def test_resolve_direct_role_dir_returns_as_is(tmp_path):
    run_dir = _build_run(tmp_path)
    role_dir = run_dir / "checkpoint-stable-end"
    resolved = resolve_checkpoint_dir(role_dir)
    assert resolved == role_dir.resolve()


def test_archive_includes_role_dirs(tmp_path):
    run_dir = _build_run(tmp_path)
    _archive_future_checkpoints(run_dir, resume_step=150)

    # Below resume_step: left in place.
    assert (run_dir / "checkpoint-100").is_dir()

    # Above resume_step (incl. role dirs by true step): moved out of run root.
    assert not (run_dir / "checkpoint-200").exists()
    assert not (run_dir / "checkpoint-stable-end").exists()
    assert not (run_dir / "checkpoint-final").exists()

    archive_dirs = [p for p in run_dir.iterdir() if p.name.startswith("archived_")]
    assert len(archive_dirs) == 1
    archive_dir = archive_dirs[0]
    moved = {p.name for p in archive_dir.iterdir()}
    assert moved == {"checkpoint-200", "checkpoint-stable-end", "checkpoint-final"}


def test_unresolvable_role_dir_is_not_archived(tmp_path):
    """A role dir whose step can't be resolved (no training_state.json) must
    yield step -1 and be LEFT in place — never mistaken for a future step.
    A numbered dir at exactly resume_step must also be kept (> not >=)."""
    run_dir = tmp_path / "run-xyz"
    run_dir.mkdir()
    (run_dir / "checkpoint-stable-end").mkdir()  # no training_state.json -> step -1
    _make_ckpt(run_dir, "checkpoint-150", 150)   # exactly at resume_step
    _make_ckpt(run_dir, "checkpoint-300", 300)   # above -> archived

    assert _checkpoint_step(run_dir / "checkpoint-stable-end") == -1

    _archive_future_checkpoints(run_dir, resume_step=150)

    assert (run_dir / "checkpoint-stable-end").is_dir()  # -1 step: kept
    assert (run_dir / "checkpoint-150").is_dir()          # == resume_step: kept
    assert not (run_dir / "checkpoint-300").exists()      # > resume_step: archived
