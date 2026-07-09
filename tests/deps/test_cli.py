"""CLI wiring/dispatch tests for `nanoplm deps` — the real installer is mocked."""
import sys
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from nanoplm.cli import cli
from nanoplm.cli.deps import deps, _script_path


def _ok(*_a, **_k):
    return SimpleNamespace(returncode=0)


def test_deps_group_attached_to_cli():
    res = CliRunner().invoke(cli, ["deps", "--help"])
    assert res.exit_code == 0, res.output
    assert "install" in res.output and "verify" in res.output


def test_install_surface_is_minimal():
    res = CliRunner().invoke(deps, ["install", "--help"])
    assert res.exit_code == 0, res.output
    assert "--arch" in res.output and "--force" in res.output
    assert "--only" not in res.output and "--profile" not in res.output


def test_packaged_script_resolves():
    p = _script_path()
    assert p.name == "install.sh"
    assert p.is_file()


def test_install_runs_script_then_fresh_verify(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    with patch("nanoplm.cli.deps.subprocess.run", side_effect=_ok) as run:
        res = CliRunner().invoke(deps, ["install"])
    assert res.exit_code == 0, res.output
    assert run.call_count == 2
    script_argv = run.call_args_list[0].args[0]
    assert script_argv[0] == "bash"
    assert script_argv[1].endswith("install.sh")
    assert script_argv[2:] == ["--arch", "90"]
    assert run.call_args_list[0].kwargs["env"]["VIRTUAL_ENV"] == sys.prefix
    verify_argv = run.call_args_list[1].args[0]
    assert verify_argv == [sys.executable, "-m", "nanoplm.deps.verify"]


def test_install_passes_arch_and_force(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    with patch("nanoplm.cli.deps.subprocess.run", side_effect=_ok) as run:
        res = CliRunner().invoke(deps, ["install", "--arch", "120", "--force"])
    assert res.exit_code == 0, res.output
    assert run.call_args_list[0].args[0][2:] == ["--arch", "120", "--force"]


def test_install_script_failure_skips_verify(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    with patch("nanoplm.cli.deps.subprocess.run",
               side_effect=[SimpleNamespace(returncode=3)]) as run:
        res = CliRunner().invoke(deps, ["install"])
    assert res.exit_code == 3
    assert run.call_count == 1  # verify must NOT run after a failed install


def test_install_refuses_non_venv(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    with patch("nanoplm.cli.deps.subprocess.run") as run:
        res = CliRunner().invoke(deps, ["install"])
    assert res.exit_code != 0
    run.assert_not_called()


def test_verify_subcommand_dispatches_fresh_interpreter():
    with patch("nanoplm.cli.deps.subprocess.run", side_effect=_ok) as run:
        res = CliRunner().invoke(deps, ["verify"])
    assert res.exit_code == 0, res.output
    assert run.call_args.args[0] == [sys.executable, "-m", "nanoplm.deps.verify"]
