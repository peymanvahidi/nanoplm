"""Static assertions on the container definition — no docker needed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker" / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


def test_dockerfile_exists():
    assert DOCKERFILE.is_file()


def test_single_stage_on_devel_base():
    text = DOCKERFILE.read_text()
    assert text.count("FROM ") == 1, "single-stage on purpose (runtime nvcc for the MoE JIT)"
    assert "nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04" in text
    assert "runtime-ubuntu" not in text


def test_image_runs_path_two_verbatim():
    text = DOCKERFILE.read_text()
    assert "nanoplm deps install --arch" in text
    assert "python -m nanoplm.deps.verify" in text
    assert "--extra-index-url https://download.pytorch.org/whl/cu130" in text


def test_arch_build_arg_defaults_to_hopper():
    assert "ARG NVTE_CUDA_ARCHS=90" in DOCKERFILE.read_text()


def test_venv_is_active_via_env():
    text = DOCKERFILE.read_text()
    assert "VIRTUAL_ENV=/opt/venv" in text
    assert "/opt/venv/bin" in text


def test_dockerignore_keeps_context_tiny():
    assert DOCKERIGNORE.is_file()
    text = DOCKERIGNORE.read_text()
    for pat in ["*", "!pyproject.toml", "!README.md", "!src", "!docker"]:
        assert pat in text, f"{pat} missing from .dockerignore allowlist"
