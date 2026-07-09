"""Tests for src/nanoplm/deps/install.sh that DON'T require CUDA or network.

Real installs never run here: preflight failures are provoked with fake tools
on PATH, and the happy path runs under NANOPLM_DEPS_DRY_RUN=1, which prints
`DRY> <command>` lines instead of executing them.

The test venv's torch runtime-reports the full pinned build (`2.12.1+cu130`)
and can also have some/all of dion/TE/FA3/quack/sonic-moe already installed,
so the dry-run sequence test (`test_dry_run_prints_full_pinned_sequence`)
passes `--force` to make every step print deterministically regardless of
what's already satisfied in this venv. Skip behavior itself is exercised
separately via the steps' runtime/presence checks (torch's runtime
`__version__` compare, quack's `[cu13]` payload-presence guard, etc.).
"""
import os
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "src" / "nanoplm" / "deps" / "install.sh"

SYS_PATH = "/usr/bin:/bin"  # keep bash/git/coreutils reachable next to the fake bins


def _fake_bin(tmp_path, cuda_release="13.0", compute_cap=None, smi_fails=False):
    """Fake nvcc/ninja (+ optional nvidia-smi) ahead of the system PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    nvcc = bindir / "nvcc"
    nvcc.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "--version" ]; then\n'
        f'  echo "Cuda compilation tools, release {cuda_release}, V{cuda_release}.0"\n'
        "fi\n"
    )
    nvcc.chmod(0o755)
    ninja = bindir / "ninja"
    ninja.write_text("#!/bin/bash\necho 1.13.0\n")
    ninja.chmod(0o755)
    # Always created: it must shadow any real /usr/bin/nvidia-smi (some login
    # nodes have non-Hopper visualization GPUs).
    smi = bindir / "nvidia-smi"
    if smi_fails or compute_cap is None:  # login-node behavior: binary exists, no devices
        smi.write_text("#!/bin/bash\necho 'No devices were found' >&2\nexit 6\n")
    else:
        smi.write_text(f"#!/bin/bash\necho '{compute_cap}'\n")
    smi.chmod(0o755)
    return bindir


def _run(script_args, path, extra_env=None, unset=()):
    env = dict(os.environ)
    env["PATH"] = path
    env["VIRTUAL_ENV"] = sys.prefix
    env.pop("CUDA_HOME", None)
    env.pop("EBROOTCUDA", None)
    env.pop("EBROOTCUDNN", None)
    for k in unset:
        env.pop(k, None)
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(SCRIPT), *script_args], capture_output=True, text=True, env=env
    )


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file(), f"{SCRIPT} missing"
    assert SCRIPT.stat().st_mode & stat.S_IXUSR, "script is not executable (chmod +x)"


def test_bash_syntax_is_valid():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_source_safe():
    # Sourcing must define functions without running main.
    r = subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}" && type preflight >/dev/null && echo SOURCED_OK'],
        capture_output=True, text=True,
        env={**os.environ, "VIRTUAL_ENV": sys.prefix},
    )
    assert "SOURCED_OK" in r.stdout, r.stderr
    assert "preflight OK" not in r.stdout  # main must not have run


def test_all_pins_present():
    text = SCRIPT.read_text()
    assert "0333cc1136cd456b3fe3a56407d4ff104cfef604" in text          # dion
    assert "4923037d858aebfe5738ebb17d02f88d7871c333" in text          # sonic-moe
    assert "Dao-AILab/sonic-moe" in text
    assert "quack-kernels[cu13]==0.4.1" in text
    assert "v2.16" in text                                             # TE tag
    assert "cu130torch2120cxx11abitrue.8a8b2f" in text                 # FA3 wheel
    assert "deps-assets" in text                                       # FA3 release tag
    assert "nvidia-cutlass-dsl-libs-cu13" in text                      # quack [cu13] payload guard


def test_no_venv_refused(tmp_path):
    bindir = _fake_bin(tmp_path)
    r = _run([], path=f"{bindir}:{SYS_PATH}", unset=("VIRTUAL_ENV",),
             extra_env={"VIRTUAL_ENV": ""})
    assert r.returncode == 1
    assert "virtualenv" in (r.stderr + r.stdout).lower()


def test_missing_nvcc_prints_jureca_module_hint(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    r = _run([], path=f"{empty}:{SYS_PATH}")
    assert r.returncode == 1
    assert "module load CUDA/13 NCCL cuDNN Ninja" in r.stderr


def test_wrong_cuda_major_refused(tmp_path):
    bindir = _fake_bin(tmp_path, cuda_release="12.4")
    r = _run([], path=f"{bindir}:{SYS_PATH}")
    assert r.returncode == 1
    assert "13" in r.stderr


def test_cuda_home_defaults_to_nvcc_prefix(tmp_path):
    bindir = _fake_bin(tmp_path)
    r = _run([], path=f"{bindir}:{SYS_PATH}", extra_env={"NANOPLM_DEPS_DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    assert f"CUDA_HOME={tmp_path}" in r.stdout  # dirname(dirname(fake nvcc)) == tmp_path


def test_cuda_home_prefers_ebrootcuda(tmp_path):
    bindir = _fake_bin(tmp_path)
    r = _run([], path=f"{bindir}:{SYS_PATH}",
             extra_env={"NANOPLM_DEPS_DRY_RUN": "1", "EBROOTCUDA": "/opt/ebcuda"})
    assert r.returncode == 0, r.stderr
    assert "CUDA_HOME=/opt/ebcuda" in r.stdout


def test_non_hopper_gpu_refused(tmp_path):
    bindir = _fake_bin(tmp_path, compute_cap="8.9")
    r = _run([], path=f"{bindir}:{SYS_PATH}", extra_env={"NANOPLM_DEPS_DRY_RUN": "1"})
    assert r.returncode == 1
    assert "Hopper" in r.stderr


def test_hopper_gpu_passes_gate(tmp_path):
    bindir = _fake_bin(tmp_path, compute_cap="9.0")
    r = _run([], path=f"{bindir}:{SYS_PATH}", extra_env={"NANOPLM_DEPS_DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr


def test_gpuless_nvidia_smi_skips_gate(tmp_path):
    # Login nodes have /usr/bin/nvidia-smi but no devices (exit 6).
    bindir = _fake_bin(tmp_path, smi_fails=True)
    r = _run([], path=f"{bindir}:{SYS_PATH}", extra_env={"NANOPLM_DEPS_DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr


def test_unknown_flag_exits_2(tmp_path):
    bindir = _fake_bin(tmp_path)
    r = _run(["--bogus"], path=f"{bindir}:{SYS_PATH}")
    assert r.returncode == 2


def test_dry_run_prints_full_pinned_sequence(tmp_path):
    # --force makes every step print unconditionally, regardless of what's
    # already satisfied in this venv, so the sequence is deterministic.
    bindir = _fake_bin(tmp_path)
    r = _run(["--force"], path=f"{bindir}:{SYS_PATH}", extra_env={"NANOPLM_DEPS_DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # 1. torch (forced, regardless of what's already satisfied)
    assert "pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130" in out
    # 2. dion @ SHA
    assert "git+https://github.com/microsoft/dion.git@0333cc1136cd456b3fe3a56407d4ff104cfef604" in out
    # 3. TE: pre-install runtime deps, clone tag, no-isolation no-deps build
    assert "nvdlfw-inspect" in out
    assert "--branch v2.16" in out
    assert "--no-build-isolation --no-deps" in out
    # 4. FA3 from the release mirror
    assert "releases/download/deps-assets" in out
    # 5. quack + sonic-moe
    assert "quack-kernels[cu13]==0.4.1" in out
    assert "git+https://github.com/Dao-AILab/sonic-moe.git@4923037d858aebfe5738ebb17d02f88d7871c333" in out
    # steps run in pin order
    assert out.index("torch==2.12.1") < out.index("dion") < out.index("v2.16") \
        < out.index("deps-assets") < out.index("quack-kernels")


def test_arch_flag_exported(tmp_path):
    bindir = _fake_bin(tmp_path)
    r = _run(["--arch", "120"], path=f"{bindir}:{SYS_PATH}",
             extra_env={"NANOPLM_DEPS_DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    assert "archs=120" in r.stdout


def test_fa3_sha256_is_pinned():
    import re
    text = SCRIPT.read_text()
    m = re.search(r'FA3_SHA256="\$\{FA3_SHA256:-([0-9a-f]{64})\}"', text)
    assert m, "FA3_SHA256 must default to a pinned 64-hex digest (not UNSET)"


def test_fa3_sha_unset_refused_in_real_mode(tmp_path):
    # Real mode (no DRY_RUN) must refuse to reach the download with an unpinned
    # sha256. Call the step function directly via sourcing — no install runs.
    # FORCE=1 bypasses the satisfied short-circuit ("present; skipping") so the
    # refusal branch is reached deterministically even in a venv where FA3 is
    # really installed — the guard must trigger regardless of venv state.
    cmd = (
        f'source "{SCRIPT}"; FA3_SHA256=UNSET; DRY_RUN=""; FORCE=1; step_fa3'
    )
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       env={**os.environ, "VIRTUAL_ENV": sys.prefix})
    assert r.returncode == 1
    assert "FA3_SHA256" in r.stderr
