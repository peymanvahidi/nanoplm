"""Static packaging assertions (stdlib tomllib; no build, no network)."""
import fnmatch
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
OLD_TE_SCRIPT = ROOT / "install_te_jureca.sh"
DOCKER_ARTIFACTS = (
    ROOT / "docker",
    ROOT / ".dockerignore",
    ROOT / ".github" / "workflows" / "docker-build.yml",
)


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
    # biotrainer-core is declared explicitly: nanoplm.eval imports from it directly
    # (ZeroShotMethod, STANDARD_AAS, AutoEvalReport), not only through biotrainer.
    assert _load()["project"]["optional-dependencies"]["eval"] == [
        "biotrainer==2.0.0",
        "biotrainer-core>=2.0.0",
    ]


def test_readme_documents_the_install_path_and_the_platform():
    text = README.read_text()
    assert "nanoplm deps install" in text
    assert "module load CUDA/13 NCCL cuDNN Ninja" in text
    assert "Hopper" in text


def test_old_te_script_is_gone_everywhere():
    assert not OLD_TE_SCRIPT.exists(), "install_te_jureca.sh must be deleted (absorbed into nanoplm.deps)"
    assert "install_te_jureca" not in README.read_text()


def test_container_path_is_gone_everywhere():
    # `nanoplm deps install` is the only supported install path.
    for path in DOCKER_ARTIFACTS:
        assert not path.exists(), f"{path.name} must stay deleted"
    text = README.read_text().lower()
    for term in ["docker", "apptainer", "ghcr.io", "enroot"]:
        assert term not in text, f"README still mentions {term}"


def test_readme_drops_manual_per_dep_installs():
    text = README.read_text()
    assert "pip install git+https://github.com/microsoft/dion.git" not in text
    assert "alint77/sonic-moe" not in text


def test_config_yaml_ships_as_package_data():
    # recycling_compatibility.yaml is opened at ModernBertConfig.__post_init__
    # time, so a wheel without it breaks every recycling model.
    pkg_data = _load()["tool"]["setuptools"]["package-data"]
    assert "nanoplm.config" in pkg_data, "nanoplm.config needs a package-data entry"
    patterns = pkg_data["nanoplm.config"]
    assert "*.yaml" in patterns
    assert "*.json" in patterns


def test_config_data_files_exist_on_disk():
    cfg_dir = ROOT / "src" / "nanoplm" / "config"
    assert (cfg_dir / "recycling_compatibility.yaml").exists()
    assert (cfg_dir / "recycling_wandb_dashboard.json").exists()


# --- Generic gate: every shipped non-.py file must be package-data-covered --
#
# Hardcoding one assertion per data file (as the tests above do) only catches
# a defect after it's already been hit once — that's how the stale
# "nanoplm.pretraining"/"nanoplm.data" keys survived: both named directories
# containing no csrc/ at all, while the real C sources
# (data/preparation/csrc/fast_fasta_ops.c, data/sharding/csrc/
# fast_fasta_shard_writer.c) went completely unmatched and were silently
# dropped from the wheel. Walk the source tree instead and require every
# shipped non-.py file to be matched by *some* package-data pattern for its
# nearest enclosing package — this kills the whole defect class rather than
# one instance at a time, and would have caught both that bug and this one.


def _iter_shipped_non_py_files():
    src_root = ROOT / "src" / "nanoplm"
    for path in sorted(src_root.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix == ".py":
            continue
        if path.name == ".gitkeep":
            continue
        # Agent / editor instruction files are documentation for humans and
        # tools working IN the tree, never runtime assets, so they are not
        # expected to ship in a wheel. Excluded by exact name rather than by
        # suffix: a package that genuinely ships a .md asset should still be
        # caught by this gate.
        if path.name in {"CLAUDE.md", "AGENTS.md", "GEMINI.md", "README.md"}:
            continue
        yield path


def _enclosing_package(path: Path) -> tuple[str, Path]:
    """Walk up from `path`'s parent to the nearest directory that is a
    Python package (has __init__.py) under src/. Returns (dotted package
    name, that directory)."""
    src_dir = ROOT / "src"
    d = path.parent
    while True:
        if (d / "__init__.py").exists():
            return ".".join(d.relative_to(src_dir).parts), d
        if d == src_dir or d == d.parent:
            raise AssertionError(
                f"{path.relative_to(ROOT)}: no ancestor package (__init__.py) found"
            )
        d = d.parent


def test_every_shipped_non_py_file_is_covered_by_package_data():
    pkg_data = _load()["tool"]["setuptools"]["package-data"]
    uncovered = []
    for path in _iter_shipped_non_py_files():
        pkg_name, pkg_dir = _enclosing_package(path)
        rel = path.relative_to(pkg_dir).as_posix()
        patterns = pkg_data.get(pkg_name, [])
        if not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            uncovered.append(
                f"{path.relative_to(ROOT)} (package {pkg_name!r} has no pattern "
                f"matching {rel!r}; patterns={patterns})"
            )
    assert uncovered == [], (
        "shipped non-.py files not matched by any package-data pattern — "
        f"these would be silently dropped from a wheel: {uncovered}"
    )
