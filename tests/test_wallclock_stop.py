"""Wall-clock stop policy and the SIGTERM exit reporter."""

import os
import signal
import subprocess
import sys

import pytest

from nanoplm.pretraining.pure_pipeline import (
    _SLURM_DEADLINE_MARGIN_SEC,
    _describe_stop_policy,
    _resolve_stop_policy,
    _slurm_deadline_ts,
)

NOW = 1757100000.0
MARGIN = _SLURM_DEADLINE_MARGIN_SEC
JOB_3H = NOW + 2.911 * 3600.0


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "not-a-timestamp",
        str(NOW - 3600.0),
        str(NOW + MARGIN - 1.0),
        str(NOW + MARGIN),
    ],
)
def test_slurm_deadline_ts_unusable(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_END_TIME", raw)
    assert _slurm_deadline_ts(MARGIN, NOW) is None


def test_slurm_deadline_ts_future(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(JOB_3H))
    assert _slurm_deadline_ts(MARGIN, NOW) == JOB_3H - MARGIN


def test_resolve_unset(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    p = _resolve_stop_policy(None, MARGIN, NOW)
    assert p.source == "unset"
    assert p.enabled is False
    assert p.limit_sec is None
    assert p.deadline_ts is None
    assert p.slurm_deadline_ts is None


def test_resolve_config_without_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    p = _resolve_stop_policy(3.0, MARGIN, NOW)
    assert p.source == "config"
    assert p.limit_sec == 10800.0
    assert p.deadline_ts is None
    assert p.slurm_deadline_ts is None
    assert p.enabled is True


def test_resolve_slurm_only(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(JOB_3H))
    p = _resolve_stop_policy(None, MARGIN, NOW)
    assert p.source == "slurm"
    assert p.deadline_ts == JOB_3H - MARGIN
    assert p.limit_sec is None
    assert p.slurm_deadline_ts == JOB_3H - MARGIN
    assert p.enabled is True


def test_resolve_config_beats_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(JOB_3H))
    p = _resolve_stop_policy(3.0, MARGIN, NOW)
    assert p.source == "config"
    assert p.limit_sec == 10800.0
    assert p.deadline_ts is None
    assert p.slurm_deadline_ts == JOB_3H - MARGIN


@pytest.mark.parametrize("hours", [0, 0.0, None])
def test_resolve_falsy_hours_fall_through_to_slurm(monkeypatch, hours):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(JOB_3H))
    p = _resolve_stop_policy(hours, MARGIN, NOW)
    assert p.source == "slurm"
    assert p.deadline_ts == JOB_3H - MARGIN


def test_describe_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(JOB_3H))
    text = _describe_stop_policy(_resolve_stop_policy(None, MARGIN, NOW), NOW)
    assert "SLURM_JOB_END_TIME" in text
    assert "2.83h from now" in text
    assert "margin 5m" in text


def test_describe_config(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    text = _describe_stop_policy(_resolve_stop_policy(3.0, MARGIN, NOW), NOW)
    assert "source=config" in text
    assert "max_wallclock_hours=3h" in text
    assert "PAST the SLURM deadline" not in text


def test_describe_unset(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    text = _describe_stop_policy(_resolve_stop_policy(None, MARGIN, NOW), NOW)
    assert "job's time limit will kill this run" in text


def test_describe_config_overrunning_the_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(JOB_3H))
    text = _describe_stop_policy(_resolve_stop_policy(3.0, MARGIN, NOW), NOW)
    assert "PAST the SLURM deadline, this run will be killed" in text
    assert "0.17h PAST" in text


def test_describe_config_inside_a_long_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(NOW + 12 * 3600.0))
    text = _describe_stop_policy(_resolve_stop_policy(3.0, MARGIN, NOW), NOW)
    assert "PAST the SLURM deadline" not in text


_SIGTERM_SCRIPT = """
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
import os, signal
from nanoplm.pretraining.pure_pipeline import _RunProgress, _install_sigterm_reporter

progress = _RunProgress(
    total_steps=190735,
    run_root="/tmp/x",
    wandb_run_id="abc",
    step=53307,
    tokens=55898238976,
    last_checkpoint="checkpoint-53300",
)
_install_sigterm_reporter(progress, lambda: 10479.6)
os.kill(os.getpid(), signal.SIGTERM)
print("UNREACHABLE")
"""


def test_sigterm_reporter_logs_then_dies_of_sigterm(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", _SIGTERM_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "NANOPLM_LOG_DIR": str(tmp_path)},
    )
    # nanoplm's logger has propagate=False and its own stdout handler, so
    # basicConfig's stderr root handler never sees the record.
    out = proc.stdout
    assert proc.returncode in (-signal.SIGTERM, 143), proc.returncode
    assert "reason=sigterm" in out
    assert "step=53,307/190,735 (27.9%)" in out
    assert "last=checkpoint-53300" in out
    assert "UNREACHABLE" not in proc.stdout + proc.stderr
