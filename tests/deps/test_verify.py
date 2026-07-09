"""Unit tests for nanoplm.deps.verify — no GPU, no hard deps needed."""
import sys
import types

import pytest

from nanoplm.deps import verify


@pytest.fixture()
def fake_torch(monkeypatch):
    torch = types.SimpleNamespace(
        __version__=verify.PINNED_TORCH,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_check_torch_ok(fake_torch):
    ok, detail = verify.check_torch()
    assert ok, detail


def test_check_torch_wrong_build(fake_torch):
    fake_torch.__version__ = "2.12.1"  # PyPI build, not +cu130
    ok, detail = verify.check_torch()
    assert not ok
    assert "2.12.1+cu130" in detail


def test_check_import_missing_module():
    ok, detail = verify.check_import("nanoplm_no_such_module_xyz")
    assert not ok


def test_check_import_degrades_on_missing_driver(tmp_path, monkeypatch):
    # A module that IS installed but whose import needs the GPU driver
    # (docker build has no libcuda): verify must degrade to presence-check.
    mod = tmp_path / "fake_driver_dep.py"
    mod.write_text("raise OSError('libcuda.so.1: cannot open shared object file')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    ok, detail = verify.check_import("fake_driver_dep")
    assert ok
    assert "driver" in detail.lower()


def test_check_import_real_breakage_still_fails(tmp_path, monkeypatch):
    mod = tmp_path / "fake_broken_dep.py"
    mod.write_text("raise ImportError('boom, unrelated to drivers')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    ok, detail = verify.check_import("fake_broken_dep")
    assert not ok


def test_check_gpu_skips_without_gpu(fake_torch):
    ok, detail = verify.check_gpu()
    assert ok
    assert "GPU node" in detail


def test_main_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(verify, "run_checks", lambda: [("a", True, "fine")])
    assert verify.main() == 0
    monkeypatch.setattr(verify, "run_checks", lambda: [("a", True, "fine"), ("b", False, "nope")])
    assert verify.main() == 1
    out = capsys.readouterr().out
    assert "✗" in out and "nope" in out
