#!/bin/bash
# Build & install TransformerEngine 2.16 for JURECA (JSC) into the project .venv.
#
# Why this script exists: every PREBUILT TE wheel (2.14/2.15/2.16) fails to import
# on JURECA with `undefined symbol: cublasLtGroupedMatrixLayoutInit_internal` —
# the wheels link a newer cublasLt than the cluster's `module load CUDA/13`
# provides. Building from source against the cluster's own cublasLt fixes it
# (CMake guards out the grouped-GEMM path that needs the missing symbol).
#
# Run ONCE from a login node (needs internet to clone TE). No GPU needed.
#   bash install_te_jureca.sh
#
set -euo pipefail

# Repo root = this script's dir; .venv must already exist there.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_TAG="${TE_TAG:-v2.16}"

[ -d "$ROOT/.venv" ] || { echo "ERROR: $ROOT/.venv not found — create it and install nanoplm first." >&2; exit 1; }

# Clone TE into a temporary build dir that is removed on exit — the built wheel
# lands in .venv, so the source tree isn't needed afterwards.
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/te-build.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT

# Cluster toolchain (load on top of StdEnv; do NOT module purge — it strips the
# compiler that NCCL/Ninja need). TE setup reads the EasyBuild *ROOT* vars.
module load CUDA/13 NCCL cuDNN Ninja
export CUDA_HOME="$EBROOTCUDA" CUDNN_HOME="$EBROOTCUDNN" CUDNN_PATH="$EBROOTCUDNN"

# Build knobs: PyTorch framework; sm90 (H100/H200) + sm120 (Blackwell consumer,
# e.g. RTX 5090). Override NVTE_CUDA_ARCHS to retarget.
export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90;120}" MAX_JOBS="${MAX_JOBS:-32}"

echo ">> cloning TransformerEngine $TE_TAG into $BUILD_DIR ..."
git clone --recursive --depth 1 --branch "$TE_TAG" \
  https://github.com/NVIDIA/TransformerEngine.git "$BUILD_DIR/TransformerEngine"

source "$ROOT/.venv/bin/activate"
echo ">> nvcc: $(nvcc --version | grep release) | CUDA_HOME=$CUDA_HOME | archs=$NVTE_CUDA_ARCHS | MAX_JOBS=$MAX_JOBS"

# Build + install from source into .venv (no build isolation so it links the
# venv's torch). Takes ~7-10 min on a login node.
VIRTUAL_ENV="$ROOT/.venv" uv pip install -v --no-build-isolation "$BUILD_DIR/TransformerEngine"

# Verify it imports against the module's CUDA (the cublasLt symbol check).
python -c "import transformer_engine.pytorch as te; print('OK: transformer_engine', te.__version__ if hasattr(te,'__version__') else 'source')" \
  && echo "=== TransformerEngine installed and imports cleanly ===" \
  || { echo "ERROR: TE built but failed to import." >&2; exit 1; }
