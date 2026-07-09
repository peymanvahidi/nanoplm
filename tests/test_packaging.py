"""Static packaging assertions (stdlib tomllib; no build, no network)."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
OLD_TE_SCRIPT = ROOT / "install_te_jureca.sh"


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


def test_readme_documents_both_paths_and_the_platform():
    text = README.read_text()
    assert "nanoplm deps install" in text
    assert "module load CUDA/13 NCCL cuDNN Ninja" in text
    assert "ghcr.io/peymanvahidi/nanoplm" in text
    assert "apptainer pull" in text
    assert "Hopper" in text


def test_old_te_script_is_gone_everywhere():
    assert not OLD_TE_SCRIPT.exists(), "install_te_jureca.sh must be deleted (absorbed into nanoplm.deps)"
    assert "install_te_jureca" not in README.read_text()


def test_readme_drops_manual_per_dep_installs():
    text = README.read_text()
    assert "pip install git+https://github.com/microsoft/dion.git" not in text
    assert "alint77/sonic-moe" not in text
