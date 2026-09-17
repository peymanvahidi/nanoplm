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

`_run` also pins NANOPLM_CUDA_SEARCH_ROOTS="" so preflight's off-PATH toolkit
probe cannot reach a REAL /usr/local/cuda-13* on the test host — otherwise the
"no nvcc" tests would pass preflight on any CUDA 13 machine. The probe itself
is tested by pointing that variable at fake toolkit trees under tmp_path.
"""
import os
import stat
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "src" / "nanoplm" / "deps" / "install.sh"

SYS_PATH = "/usr/bin:/bin"  # keep bash/git/coreutils reachable next to the fake bins


def _write_nvcc(path, release="13.0"):
    """A fake nvcc whose --version mimics the real release banner."""
    path.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "--version" ]; then\n'
        f'  echo "Cuda compilation tools, release {release}, V{release}.0"\n'
        "fi\n"
    )
    path.chmod(0o755)
    return path


def _fake_toolkit(root, release="13.3"):
    """A fake toolkit tree (<root>/bin/nvcc) meant to stay OFF PATH, for the probe."""
    (root / "bin").mkdir(parents=True, exist_ok=True)
    _write_nvcc(root / "bin" / "nvcc", release)
    return root


def _fake_bin(tmp_path, cuda_release="13.0", compute_cap=None, smi_fails=False,
              with_nvcc=True, with_ninja=True):
    """Fake nvcc/ninja (+ optional nvidia-smi) ahead of the system PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    if with_nvcc:
        _write_nvcc(bindir / "nvcc", cuda_release)
    if with_ninja:
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
    env["NANOPLM_CUDA_SEARCH_ROOTS"] = ""  # hermetic: never probe the host's real toolkit
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
    assert "dd6c02f5173dc43cc7a0974535d7011a74bda494" in text          # dion
    assert "4923037d858aebfe5738ebb17d02f88d7871c333" in text          # sonic-moe
    assert "Dao-AILab/sonic-moe" in text
    assert "quack-kernels[cu13]==0.5.0" in text                        # 0.5.0 = compile-under-flock fix
    assert "nvidia-cutlass-dsl[cu13]==4.5.3" in text                   # pinned: 0.5.0 only floors it
    assert "v2.16" in text                                             # TE tag
    assert "cu130torch2120cxx11abitrue.8a8b2f" in text                 # FA3 wheel
    assert "nvidia-cutlass-dsl-libs-cu13" in text                      # quack [cu13] payload guard
    assert "windreamer/flash-attention3-wheels" in text                # FA3 single source (windreamer mirror)
    assert "2026.05.20-072445f" in text                                # FA3 pinned release tag
    assert 'FA3_WHEEL_URLENC="${FA3_WHEEL//+/%2B}"' in text            # GitHub 504s on literal '+' in the path


def test_no_venv_refused(tmp_path):
    bindir = _fake_bin(tmp_path)
    r = _run([], path=f"{bindir}:{SYS_PATH}", unset=("VIRTUAL_ENV",),
             extra_env={"VIRTUAL_ENV": ""})
    assert r.returncode == 1
    assert "virtualenv" in (r.stderr + r.stdout).lower()


def test_missing_nvcc_prints_jureca_module_hint(tmp_path):
    # Probe disabled by _run, so this is the "no toolkit anywhere" refusal.
    empty = tmp_path / "empty"
    empty.mkdir()
    r = _run([], path=f"{empty}:{SYS_PATH}")
    assert r.returncode == 1
    assert "module load CUDA/13 NCCL cuDNN Ninja" in r.stderr
    # The driver-vs-toolkit note: nvidia-smi's "CUDA Version" row misleads here.
    assert "DRIVER" in r.stderr


def test_off_path_toolkit_is_found_and_exported(tmp_path):
    # The plain-toolkit host: CUDA 13 installed under a conventional root, its
    # bin/ not on PATH, no module system. Preflight must recover, not refuse.
    _fake_toolkit(tmp_path / "cuda-13.3")
    bindir = _fake_bin(tmp_path, with_nvcc=False)  # ninja present, nvcc absent
    r = _run([], path=f"{bindir}:{SYS_PATH}",
             extra_env={"NANOPLM_DEPS_DRY_RUN": "1",
                        "NANOPLM_CUDA_SEARCH_ROOTS": f"{tmp_path}/cuda-13*"})
    assert r.returncode == 0, r.stderr
    assert "nvcc was not on PATH" in r.stdout
    assert f"CUDA_HOME={tmp_path}/cuda-13.3" in r.stdout            # probe drives CUDA_HOME
    assert f"nvcc={tmp_path}/cuda-13.3/bin/nvcc" in r.stdout        # ...and PATH
    assert "preflight OK" in r.stdout


def test_off_path_toolkit_probe_skips_wrong_major(tmp_path):
    # A CUDA 12 toolkit under a probed root must not satisfy the gate.
    _fake_toolkit(tmp_path / "cuda-12.6", release="12.6")
    bindir = _fake_bin(tmp_path, with_nvcc=False)
    r = _run([], path=f"{bindir}:{SYS_PATH}",
             extra_env={"NANOPLM_CUDA_SEARCH_ROOTS": f"{tmp_path}/cuda-1*"})
    assert r.returncode == 1
    assert "no CUDA 13 toolkit under" in r.stderr


def test_off_path_probe_prefers_cuda_home(tmp_path):
    # An explicitly pointed-at toolkit wins over the probed roots.
    _fake_toolkit(tmp_path / "explicit")
    _fake_toolkit(tmp_path / "cuda-13.3")
    bindir = _fake_bin(tmp_path, with_nvcc=False)
    r = _run([], path=f"{bindir}:{SYS_PATH}",
             extra_env={"NANOPLM_DEPS_DRY_RUN": "1",
                        "CUDA_HOME": str(tmp_path / "explicit"),
                        "NANOPLM_CUDA_SEARCH_ROOTS": f"{tmp_path}/cuda-13*"})
    assert r.returncode == 0, r.stderr
    assert f"CUDA_HOME={tmp_path}/explicit" in r.stdout


def test_cuda_home_without_cuda13_is_overridden_with_a_warning(tmp_path):
    # A stale CUDA_HOME (CUDA 12, or plain wrong) must not be carried into the
    # build once the probe has found a real 13 toolkit — and must be announced.
    _fake_toolkit(tmp_path / "stale", release="12.6")
    _fake_toolkit(tmp_path / "cuda-13.3")
    bindir = _fake_bin(tmp_path, with_nvcc=False)
    r = _run([], path=f"{bindir}:{SYS_PATH}",
             extra_env={"NANOPLM_DEPS_DRY_RUN": "1",
                        "CUDA_HOME": str(tmp_path / "stale"),
                        "NANOPLM_CUDA_SEARCH_ROOTS": f"{tmp_path}/cuda-13*"})
    assert r.returncode == 0, r.stderr
    assert "overriding" in r.stderr
    assert f"CUDA_HOME={tmp_path}/cuda-13.3" in r.stdout


def test_on_path_nvcc_is_left_alone(tmp_path):
    # No probe noise when the toolchain is already exported (the cluster path).
    _fake_toolkit(tmp_path / "cuda-13.3")
    bindir = _fake_bin(tmp_path)
    r = _run([], path=f"{bindir}:{SYS_PATH}",
             extra_env={"NANOPLM_DEPS_DRY_RUN": "1",
                        "NANOPLM_CUDA_SEARCH_ROOTS": f"{tmp_path}/cuda-13*"})
    assert r.returncode == 0, r.stderr
    assert "nvcc was not on PATH" not in r.stdout
    assert f"CUDA_HOME={tmp_path}" in r.stdout  # dirname(dirname(fake nvcc)) == tmp_path


def test_missing_ninja_is_bootstrapped_from_pypi(tmp_path):
    # ninja is a wheel, so pip can supply it in-venv instead of refusing.
    bindir = _fake_bin(tmp_path, with_ninja=False)
    r = _run([], path=f"{bindir}:{SYS_PATH}", extra_env={"NANOPLM_DEPS_DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    assert "ninja not on PATH" in r.stdout
    assert "pip install ninja" in r.stdout
    assert "preflight OK" in r.stdout


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


def test_non_hopper_gpu_warns_and_continues(tmp_path):
    bindir = _fake_bin(tmp_path, compute_cap="8.9")
    r = _run([], path=f"{bindir}:{SYS_PATH}", extra_env={"NANOPLM_DEPS_DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    assert "WARNING" in r.stderr
    assert "Hopper" in r.stderr
    assert "preflight OK" in r.stdout


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
    assert "git+https://github.com/microsoft/dion.git@dd6c02f5173dc43cc7a0974535d7011a74bda494" in out
    # 3. TE: pre-install runtime deps, clone tag, no-isolation no-deps build
    assert "nvdlfw-inspect" in out
    assert "--branch v2.16" in out
    assert "--no-build-isolation --no-deps" in out
    # 4. FA3 from the windreamer mirror
    assert "windreamer/flash-attention3-wheels/releases/download/2026.05.20-072445f" in out
    # 5. quack + DSL co-resolved in one pip call, then sonic-moe
    assert "quack-kernels[cu13]==0.5.0" in out
    assert "nvidia-cutlass-dsl[cu13]==4.5.3" in out
    assert "git+https://github.com/Dao-AILab/sonic-moe.git@4923037d858aebfe5738ebb17d02f88d7871c333" in out
    # steps run in pin order
    assert out.index("torch==2.12.1") < out.index("dion") < out.index("v2.16") \
        < out.index("flash-attention3-wheels") < out.index("quack-kernels")


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


def test_fa3_download_uses_windreamer(tmp_path):
    # Single source now: fa3_download curls the public windreamer wheel URL and
    # writes it to the -o destination. Stub curl ahead of PATH to succeed (write
    # a marker byte) ONLY for a windreamer URL and fail anything else, so a stray
    # fetch to some other host would be caught as a failure.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    curl = bindir / "curl"
    curl.write_text(
        "#!/bin/bash\n"
        'dest=""\n'
        'prev=""\n'
        'url="${@: -1}"\n'
        'for arg in "$@"; do\n'
        '  if [ "$prev" = "-o" ]; then dest="$arg"; fi\n'
        '  prev="$arg"\n'
        "done\n"
        'case "$url" in\n'
        '  *windreamer*) printf \'X\' > "$dest"; exit 0 ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    curl.chmod(0o755)
    out = tmp_path / "out.whl"
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = sys.prefix
    env["PATH"] = f"{bindir}:{SYS_PATH}"
    cmd = f'source "{SCRIPT}"; fa3_download "{out}"'
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    assert "windreamer" in (r.stdout + r.stderr)


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
