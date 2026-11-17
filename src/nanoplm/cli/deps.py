"""`nanoplm deps` — install & verify the hard GPU dependency stack.

One supported environment: Linux + Hopper (sm90) + CUDA >= 13.0, full feature
set always. `install` runs the packaged src/nanoplm/deps/install.sh (all pins
live there) and then verifies in a fresh interpreter; `verify` re-runs just the
verification.
"""
import atexit
import os
import subprocess
import sys
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path

import click

# Keep the packaged script materialized for the process lifetime: as_file() may
# extract to a temp path (zip installs) and delete it when its context closes,
# which must not happen before bash finishes.
_RESOURCE_STACK = ExitStack()
atexit.register(_RESOURCE_STACK.close)


def _script_path() -> Path:
    p = _RESOURCE_STACK.enter_context(
        as_file(files("nanoplm.deps").joinpath("install.sh"))
    )
    return Path(p)


def _fresh_verify(env=None) -> int:
    """Run verification in a fresh interpreter (stale in-process torch otherwise)."""
    return subprocess.run([sys.executable, "-m", "nanoplm.deps.verify"], env=env).returncode


@click.group(name="deps")
def deps():
    """Install / verify nanoPLM's GPU dependency stack (Hopper + CUDA 13)."""


@deps.command("install")
@click.option("--arch", default="90", show_default=True,
              help="CUDA arch for source builds (NVTE_CUDA_ARCHS). 90 = Hopper/H100.")
@click.option("--force", is_flag=True, help="Reinstall every component even if already satisfied.")
@click.option("--with-mmseqs", is_flag=True,
              help="Also install the MMseqs2 binary into the venv (only needed when "
                   "cluster_subsampling clusters with MMseqs2).")
@click.option("--mmseqs-only", is_flag=True,
              help="Install only the MMseqs2 binary; no GPU stack, no verification.")
def install(arch: str, force: bool, with_mmseqs: bool, mmseqs_only: bool):
    """Install the full pinned stack into the active virtualenv, then verify.

    On clusters, load the CUDA toolchain first (JURECA:
    `module load CUDA/13 NCCL cuDNN Ninja`). Elsewhere, a CUDA 13 toolkit that
    is installed but not exported (/usr/local/cuda*) is found automatically, and
    a missing ninja is installed into the venv. Run from a login node — the
    TransformerEngine source build needs internet but no GPU (~10 min).
    """
    in_venv = sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))
    if not in_venv:
        raise click.ClickException(
            "nanoPLM is running from a non-virtual (system/base) Python. Create and "
            "activate a venv first: python3 -m venv .venv && source .venv/bin/activate"
        )
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = sys.prefix
    cmd = ["bash", str(_script_path()), "--arch", arch] + (["--force"] if force else [])
    if mmseqs_only:
        cmd.append("--mmseqs-only")
    elif with_mmseqs:
        cmd.append("--with-mmseqs")
    try:
        rc = subprocess.run(cmd, env=env).returncode
    except FileNotFoundError:
        raise click.ClickException("bash is required to run the dependency installer.")
    if rc != 0:
        raise SystemExit(rc)
    if mmseqs_only:
        return
    click.echo(">> verifying the environment (fresh interpreter)")
    rc = _fresh_verify(env=env)
    if rc != 0:
        raise SystemExit(rc)


@deps.command("verify")
def verify():
    """Check torch build, hard-dep imports, and the GPU (when one is visible)."""
    rc = _fresh_verify()
    if rc != 0:
        raise SystemExit(rc)
