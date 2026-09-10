#!/usr/bin/env bash
# nanoPLM hard-dependency installer — supported environments:
#   Linux x86_64 or aarch64 · NVIDIA Hopper (sm90: H100 / GH200) · CUDA >= 13.0 · torch 2.12.1+cu130
#
# Installs, idempotently and always all of them:
#   1. torch 2.12.1+cu130                    2. dion (pinned commit)
#   3. TransformerEngine v2.16 (from source) 4. flash-attn-3 (pinned wheel, sha256)
#   5. quack-kernels[cu13]==0.5.0 + nvidia-cutlass-dsl[cu13]==4.5.3
#      + sonic-moe (pinned commit)
#
# Not user-facing: run `nanoplm deps install`.
#
# Everything installs with `$VIRTUAL_ENV/bin/python -m pip` — never uv — because
# quack-kernels forbids uv on CUDA 13.x and one tool means one behavior.
set -euo pipefail

# ---- pins: THE single source of truth (the spec table mirrors this block) ----
TORCH_PIN="2.12.1+cu130"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
DION_SHA="dd6c02f5173dc43cc7a0974535d7011a74bda494"           # microsoft/dion main (upstream)
TE_TAG="${TE_TAG:-v2.16}"
SONICMOE_SHA="4923037d858aebfe5738ebb17d02f88d7871c333"       # Dao-AILab/sonic-moe PR #64 head
SONICMOE_URL="git+https://github.com/Dao-AILab/sonic-moe.git@${SONICMOE_SHA}"
# quack 0.5.0 is the earliest release that runs cute.compile() UNDER the per-key
# exclusive flock (quack/cache/jit.py:224-234). 0.4.1 compiled in the unlocked
# window between the shared-lock read and the exclusive-lock write, so on a cold
# cache every rank -- and every autotuner _compile_worker subprocess -- compiled
# the same kernel concurrently and one died with a bare exit(1). 0.5.1-0.5.3 pin
# the PRERELEASE nvidia-cutlass-dsl==4.6.0.dev0; 0.6.x forces ==4.6.0. 0.5.0 only
# FLOORS the DSL (>=4.5.2), which pip would float to 4.6.0 -- a version published
# after 0.5.0 and untested with it -- so pin the release-era DSL explicitly.
QUACK_SPEC="quack-kernels[cu13]==0.5.0"
DSL_SPEC="nvidia-cutlass-dsl[cu13]==4.5.3"
# flash-attn-3 is the one hard-dep shipped as a PREBUILT wheel (no source build),
# so the exact artifact differs per CPU arch: x86_64 (H100-class boxes) vs aarch64
# (GH200 superchips / JUPITER). Each arch pins its own wheel filename, installed-
# version needle, and sha256; all are fetched from the shared windreamer mirror
# (FA3_URL) below. NOTE: the two builds are NOT the same commit/date (x86_64 =
# 20260519/8a8b2f, aarch64 = 20260520/417891) — they are upstream's independent
# per-arch builds, pinned as-is. FA3_SHA256 stays env-overridable; a real install
# refuses to run while it is UNSET (see step_fa3).
FA3_ARCH="$(uname -m)"
case "$FA3_ARCH" in
  x86_64)
    FA3_WHEEL="flash_attn_3-3.0.0+20260519.cu130torch2120cxx11abitrue.8a8b2f-cp39-abi3-linux_x86_64.whl"
    FA3_VERSION_NEEDLE="3.0.0+20260519"
    FA3_SHA256="${FA3_SHA256:-177359694998fee12e293d15d3398482fc0193f5f40d4aca3ceb257105d5bbf2}"
    ;;
  aarch64|arm64)
    FA3_WHEEL="flash_attn_3-3.0.0+20260520.cu130torch2120cxx11abitrue.417891-cp39-abi3-linux_aarch64.whl"
    FA3_VERSION_NEEDLE="3.0.0+20260520"
    FA3_SHA256="${FA3_SHA256:-8a520336c5126cf638db1b39eee74b74da57fd5eaa9aac275c5ae949b291bb80}"
    ;;
  *)
    echo "ERROR: unsupported CPU architecture for flash-attn-3: $FA3_ARCH (supported: x86_64, aarch64)" >&2
    exit 1
    ;;
esac
FA3_WHEEL_URLENC="${FA3_WHEEL//+/%2B}"   # GitHub's release-download endpoint 504s on a literal '+' in the path
# Single source: the public windreamer wheel mirror (no auth), pinned to a fixed
# release tag. Downloaded bytes are sha256-gated in step_fa3 regardless.
FA3_URL="https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.05.20-072445f/${FA3_WHEEL_URLENC}"

# ---- knobs -------------------------------------------------------------------
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90}"
export MAX_JOBS="${MAX_JOBS:-32}"
FORCE=0
DRY_RUN="${NANOPLM_DEPS_DRY_RUN:-}"
JURECA_HINT="module load CUDA/13 NCCL cuDNN Ninja"

log() { echo ">> $*"; }
err() { echo "ERROR: $*" >&2; }
warn() { echo "WARNING: $*" >&2; }
# All mutating commands go through run_cmd so NANOPLM_DEPS_DRY_RUN=1 prints the
# exact command instead of executing (this is how the unit tests cover the steps).
run_cmd() { if [ -n "$DRY_RUN" ]; then echo "DRY> $*"; else "$@"; fi; }

# Temp dirs are cleaned on EXIT (not RETURN): a RETURN trap does NOT fire when a
# function aborts under `set -e` — exactly the paths where cleanup matters.
# The if-form keeps errexit here from rewriting the script's 0/1/2 exit contract.
TMP_DIRS=()
cleanup() { for d in "${TMP_DIRS[@]:-}"; do if [ -n "$d" ]; then rm -rf "$d"; fi; done; }
trap cleanup EXIT

PY="${VIRTUAL_ENV:-/nonexistent}/bin/python"

# dist_ok <dist-name> <needle>: 0 iff <dist> is installed and <needle> occurs in
# its version or its direct_url.json (git installs record the commit there).
dist_ok() {
  "$PY" - "$1" "$2" <<'PY' 2>/dev/null
import sys
from importlib.metadata import distribution, PackageNotFoundError
name, needle = sys.argv[1], sys.argv[2]
try:
    d = distribution(name)
except PackageNotFoundError:
    sys.exit(1)
hay = (d.version or "") + " " + (d.read_text("direct_url.json") or "")
sys.exit(0 if needle in hay else 1)
PY
}

satisfied() {  # satisfied <dist> <needle> — honors --force
  [ "$FORCE" -eq 0 ] && dist_ok "$1" "$2"
}

# torch_runtime_ok: 0 iff "$PY" can import torch and torch.__version__ equals
# $TORCH_PIN exactly. torch's dist metadata is flavor-blind — 2.12.x wheels
# all report the plain "2.12.1" in their version/direct_url.json regardless of
# whether they're the +cu130 build or a CPU/other-CUDA build, and index
# installs write no direct_url.json at all — so dist_ok can never distinguish
# flavors here. Only the runtime __version__ does; this mirrors the check
# verify.py uses as the authority for "is the right torch installed".
torch_runtime_ok() {
  "$PY" - "$TORCH_PIN" <<'PY' 2>/dev/null
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if torch.__version__ == sys.argv[1] else 1)
PY
}

satisfied_torch() { [ "$FORCE" -eq 0 ] && torch_runtime_ok; }

# quack_ok: dion dep-pulls plain `quack-kernels==0.4.1` on its own (metadata
# only — dion never imports quack at runtime), so step_moe has to UPGRADE over
# whatever dion left behind. Pin-check BOTH dists: seeing quack-kernels alone
# does not mean our `[cu13]` extra ran, and a stale 4.4.2 payload left by dion's
# pull must not falsely satisfy us.
quack_ok() {
  dist_ok quack-kernels "0.5.0" && dist_ok nvidia-cutlass-dsl-libs-cu13 "4.5.3"
}

satisfied_quack() { [ "$FORCE" -eq 0 ] && quack_ok; }

# ---- preflight: fail loudly, no auto-detect ----------------------------------
resolve_cuda_home() {
  if [ -z "${CUDA_HOME:-}" ]; then
    if [ -n "${EBROOTCUDA:-}" ]; then
      export CUDA_HOME="$EBROOTCUDA"                     # EasyBuild clusters (JURECA)
    elif command -v nvcc >/dev/null 2>&1; then
      CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
      export CUDA_HOME                                   # desktops / plain toolkit installs
    fi
  fi
  # TE's build reads CUDNN_HOME/CUDNN_PATH; on non-EasyBuild systems cuDNN sits
  # on the default include/lib paths.
  if [ -n "${EBROOTCUDNN:-}" ]; then
    export CUDNN_HOME="$EBROOTCUDNN" CUDNN_PATH="$EBROOTCUDNN"
  fi
}

# Install-time declares the build target (NVTE_CUDA_ARCHS); run-time verifies
# the actual GPU (`nanoplm deps verify`) — so a local non-Hopper GPU here is
# only ever a warning, never a refusal.
hopper_gate() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0      # no GPU tool: build box / login node
  local caps
  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)" || return 0
  [ -z "$caps" ] && return 0                             # nvidia-smi present but no devices (login node)
  if [ "$NVTE_CUDA_ARCHS" = "90" ] && ! echo "$caps" | grep -qx "9.0"; then
    warn "installing for Hopper (sm90); this machine's GPU (compute capability: $caps) is not a supported run target — nanoPLM will not run on THIS GPU."
    warn "Fine on login/build nodes. Tip: export CUDA_VISIBLE_DEVICES= to hide local GPUs (also skips the GPU row in the final verification)."
  fi
}

preflight() {
  if [ -z "${VIRTUAL_ENV:-}" ] || [ ! -x "$PY" ]; then
    err "no active virtualenv (\$VIRTUAL_ENV unset, or it has no bin/python)."
    err "Create and activate one first:  python3 -m venv .venv && source .venv/bin/activate"
    exit 1
  fi
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    log "pip missing in this venv (created with 'uv venv'?) — bootstrapping via ensurepip"
    run_cmd "$PY" -m ensurepip --upgrade
  fi
  if ! command -v nvcc >/dev/null 2>&1 || ! nvcc --version 2>/dev/null | grep -q "release 13"; then
    err "nvcc (CUDA 13.x) not found on PATH."
    err "On JURECA/EasyBuild clusters run:"
    err "    $JURECA_HINT        # do NOT module purge"
    err "then re-run \`nanoplm deps install\`."
    err "Elsewhere: install the CUDA 13.0 toolkit and put its bin/ on PATH."
    exit 1
  fi
  if ! command -v ninja >/dev/null 2>&1; then
    err "ninja not found on PATH (needed to build TransformerEngine)."
    err "JURECA: $JURECA_HINT    |    Debian/Ubuntu: apt install ninja-build"
    exit 1
  fi
  command -v git >/dev/null 2>&1 || { err "git not found on PATH."; exit 1; }
  resolve_cuda_home
  [ -n "${CUDA_HOME:-}" ] || { err "CUDA_HOME could not be determined."; exit 1; }
  hopper_gate
  log "preflight OK: python=$PY  nvcc=$(command -v nvcc)  CUDA_HOME=$CUDA_HOME  archs=$NVTE_CUDA_ARCHS"
}

# ---- steps (each: skip when satisfied, unless --force) ------------------------
step_torch() {
  if satisfied_torch; then log "torch $TORCH_PIN present; skipping."; return 0; fi
  log "installing torch $TORCH_PIN from $TORCH_INDEX"
  run_cmd "$PY" -m pip install "torch==2.12.1" --index-url "$TORCH_INDEX"
}

step_dion() {
  if satisfied dion "$DION_SHA"; then log "dion @ ${DION_SHA:0:12} present; skipping."; return 0; fi
  log "installing dion @ ${DION_SHA:0:12}"
  run_cmd "$PY" -m pip install "git+https://github.com/microsoft/dion.git@${DION_SHA}"
}

step_te() {
  if satisfied transformer_engine "2.16"; then log "TransformerEngine 2.16 present; skipping."; return 0; fi
  # --no-build-isolation reads the live env and --no-deps installs ONLY
  # transformer_engine (re-resolving torch's CUDA wheel stack against PyPI can
  # pull wheels that clash with the toolchain we build against). So TE's build
  # requirements AND pure-Python runtime deps must already be in the venv:
  log "pre-installing TransformerEngine build + runtime deps"
  run_cmd "$PY" -m pip install \
    "setuptools>=61.0" "cmake>=3.21" wheel "pybind11[global]" ninja \
    pydantic einops onnx onnxscript packaging "importlib-metadata>=1.0" nvdlfw-inspect
  local build_dir
  build_dir="$(mktemp -d "${TMPDIR:-/tmp}/te-build.XXXXXX")"
  TMP_DIRS+=("$build_dir")
  log "cloning TransformerEngine $TE_TAG (archs=$NVTE_CUDA_ARCHS, MAX_JOBS=$MAX_JOBS; ~7-10 min build)"
  run_cmd git clone --depth 1 --branch "$TE_TAG" --recurse-submodules --shallow-submodules \
    https://github.com/NVIDIA/TransformerEngine.git "$build_dir/TransformerEngine"
  run_cmd "$PY" -m pip install -v --no-build-isolation --no-deps "$build_dir/TransformerEngine"
}

fa3_download() {  # fa3_download <dest-file>
  local dest="$1"
  if [ -n "${NANOPLM_FA3_WHEEL:-}" ]; then
    log "using pre-downloaded FA3 wheel: $NANOPLM_FA3_WHEEL"
    cp "$NANOPLM_FA3_WHEEL" "$dest"; return 0
  fi
  log "downloading flash-attn-3 wheel from the public windreamer mirror"
  if curl -fSL --proto '=https' --tlsv1.2 --retry 3 --retry-connrefused \
       -o "$dest" "$FA3_URL" 2>/dev/null; then
    return 0
  fi
  err "could not download the flash-attn-3 wheel from $FA3_URL"
  err "If this host has no outbound internet, download the wheel elsewhere and"
  err "point NANOPLM_FA3_WHEEL at the file, then re-run."
  return 1
}

step_fa3() {
  if satisfied flash_attn_3 "$FA3_VERSION_NEEDLE"; then log "flash-attn-3 present; skipping."; return 0; fi
  if [ -n "$DRY_RUN" ]; then
    echo "DRY> download $FA3_URL (sha256-verified)"
    echo "DRY> $PY -m pip install <downloaded>/$FA3_WHEEL"
    return 0
  fi
  if [ "$FA3_SHA256" = "UNSET" ]; then
    err "FA3_SHA256 is unpinned (UNSET) — the release-mirror task must set it before real installs."
    exit 1
  fi
  local tmp
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/fa3.XXXXXX")"
  TMP_DIRS+=("$tmp")
  log "downloading flash-attn-3 wheel"
  fa3_download "$tmp/$FA3_WHEEL" || exit 1
  echo "${FA3_SHA256}  $tmp/$FA3_WHEEL" | sha256sum -c - \
    || { err "flash-attn-3 wheel sha256 mismatch — refusing to install."; exit 1; }
  run_cmd "$PY" -m pip install "$tmp/$FA3_WHEEL"
}

step_moe() {
  if satisfied_quack; then log "quack-kernels 0.5.0 present; skipping."; else
    log "installing $QUACK_SPEC + $DSL_SPEC"
    # Installed in one command so pip co-resolves them (0.5.0 floors the DSL at
    # >=4.5.2; we pin ==4.5.3). This upgrades dion's transitive quack 0.4.1 /
    # DSL 4.4.2, so pip prints a benign "dion 0.1.0 requires quack-kernels==0.4.1"
    # warning: dion never imports quack, and verify.py checks imports rather than
    # `pip check`, so the install still verifies clean.
    run_cmd "$PY" -m pip install "$QUACK_SPEC" "$DSL_SPEC" --extra-index-url "$TORCH_INDEX"
  fi
  if satisfied sonic-moe "$SONICMOE_SHA"; then log "sonic-moe @ ${SONICMOE_SHA:0:12} present; skipping."; else
    log "installing sonic-moe @ ${SONICMOE_SHA:0:12}"
    run_cmd "$PY" -m pip install "$SONICMOE_URL"
  fi
}

main() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --arch)   NVTE_CUDA_ARCHS="$2"; export NVTE_CUDA_ARCHS; shift 2 ;;
      --arch=*) NVTE_CUDA_ARCHS="${1#*=}"; export NVTE_CUDA_ARCHS; shift ;;
      --force)  FORCE=1; shift ;;
      *) err "unknown argument: $1 (supported: --arch <cc>, --force)"; exit 2 ;;
    esac
  done
  preflight
  step_torch
  step_dion
  step_te
  step_fa3
  step_moe
  log "=== hard-dependency install finished (verification runs next) ==="
}

# Source-safe: run main only when executed, not when sourced (tests source this
# file to call preflight/step functions directly).
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
