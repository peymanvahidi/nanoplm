# syntax=docker/dockerfile:1.10
# nanoPLM — ONE supported environment: Hopper (sm90) + CUDA 13.0.
#
# Single stage on the -devel base ON PURPOSE: sonic-moe/quack JIT-compile
# CUTLASS kernels at RUNTIME and need nvcc + CUDA headers inside the running
# container. A slim -runtime final stage passes import checks at build time and
# then fails at the first MoE step. Bigger image, correct behavior.
#
# This image IS install path 2 executed in the base image (`pip install .[eval]`
# + `nanoplm deps install`) — one implementation, zero drift between paths.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ARG NVTE_CUDA_ARCHS=90
ARG MAX_JOBS=8
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=${MAX_JOBS} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv git curl ca-certificates ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv && pip install --upgrade pip

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src

# Core install. The extra index resolves torch 2.12.1+cu130 immediately, so the
# deps-install torch step below no-ops instead of re-downloading multi-GB wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[eval]" --extra-index-url https://download.pytorch.org/whl/cu130

# The hard stack — the same command venv users run. No GPU exists during
# `docker build` (fine: nvcc cross-compiles sm90; the installer's GPU gate only
# fires when a device is visible). While the repo is private the flash-attn-3
# release asset needs auth:
#   docker build --secret id=github_token,env=GITHUB_TOKEN ...
# The secret is mounted for this step only and never lands in a layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=secret,id=github_token,env=GITHUB_TOKEN \
    nanoplm deps install --arch ${NVTE_CUDA_ARCHS}

# Baked-in gate: torch pin + presence/import checks (auto-degrades exactly where
# imports need a GPU driver, which docker build doesn't have). The full check —
# `nanoplm deps verify` — runs at `docker run --gpus` time.
RUN python -m nanoplm.deps.verify

CMD ["bash"]
