"""Static packaging assertions (stdlib tomllib; no build, no network)."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def _load():
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def test_torch_pinned_exactly():
    deps = _load()["project"]["dependencies"]
    assert "torch==2.12.1" in deps, "torch must be pinned ==2.12.1 (deps install repairs the build flavor)"
    assert not any(d.startswith("torch>=") for d in deps)


def test_dion_not_a_declared_dependency():
    # dion is git-only; a PyPI/clone install must resolve without it —
    # `nanoplm deps install` owns it.
    deps = " ".join(_load()["project"]["dependencies"])
    assert "dion" not in deps


def test_installer_script_ships_as_package_data():
    pkg_data = _load()["tool"]["setuptools"]["package-data"]
    assert "*.sh" in pkg_data["nanoplm.deps"]


def test_eval_extra_unchanged():
    assert _load()["project"]["optional-dependencies"]["eval"] == ["biotrainer==1.4.0"]
