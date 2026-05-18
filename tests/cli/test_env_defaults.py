# tests/cli/test_env_defaults.py
import os
from click.testing import CliRunner
from nanoplm.cli.data import data


def test_pretokenize_max_workers_default_from_env(monkeypatch):
    monkeypatch.setenv("NANOPLM_MAX_WORKERS", "7")
    runner = CliRunner()
    result = runner.invoke(data, ["pre-tokenize", "--help"])
    assert result.exit_code == 0
    assert "--max-workers" in result.output


def test_pretokenize_max_workers_picks_up_env_in_default(monkeypatch):
    monkeypatch.setenv("NANOPLM_MAX_WORKERS", "7")
    # Re-resolve the default by inspecting the Click option
    cmd = data.commands["pre-tokenize"]
    for p in cmd.params:
        if p.name == "max_workers":
            default_val = p.default() if callable(p.default) else p.default
            assert default_val == 7
            return
    raise AssertionError("--max-workers option not found")


def test_kdembed_device_picks_up_env_in_default(monkeypatch):
    monkeypatch.setenv("NANOPLM_DEVICE", "cpu")
    cmd = data.commands["kd-embed"]
    for p in cmd.params:
        if p.name == "device":
            default_val = p.default() if callable(p.default) else p.default
            assert default_val == "cpu"
            return
    raise AssertionError("--device option not found")


def test_pretokenize_max_workers_fallback_default(monkeypatch):
    monkeypatch.delenv("NANOPLM_MAX_WORKERS", raising=False)
    cmd = data.commands["pre-tokenize"]
    for p in cmd.params:
        if p.name == "max_workers":
            default_val = p.default() if callable(p.default) else p.default
            assert default_val == 2
            return
    raise AssertionError("--max-workers option not found")


def test_kdembed_device_fallback_default(monkeypatch):
    monkeypatch.delenv("NANOPLM_DEVICE", raising=False)
    cmd = data.commands["kd-embed"]
    for p in cmd.params:
        if p.name == "device":
            default_val = p.default() if callable(p.default) else p.default
            assert default_val == "auto"
            return
    raise AssertionError("--device option not found")
