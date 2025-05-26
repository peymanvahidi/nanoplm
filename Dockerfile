# Multi-stage Dockerfile for ProtX - Protein Language Model Distillation
# Stage 1: Base image with CUDA and Python
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    build-essential \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

RUN pip install uv --break-system-packages

RUN uv venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Verify virtual environment is active
RUN which python && which pip && python --version

RUN uv pip install --upgrade pip

# Stage 2: Development image with all dependencies
FROM base AS development

COPY pyproject.toml ./
COPY uv.lock ./

RUN uv pip install -r pyproject.toml

COPY . .

RUN uv pip install -e .

RUN chmod +x scripts/*.sh 2>/dev/null || true

# Stage 3: Inference image (minimal for serving)
# FROM base AS inference

# RUN uv pip install torch transformers sentencepiece onnx

# COPY --from=development /app/src/protx/models /app/src/protx/models
# COPY --from=development /app/src/protx/utils /app/src/protx/utils
# COPY --from=development /app/src/protx/__init__.py /app/src/protx/
# COPY inference_server.py ./

# Default stage is development
FROM development AS final

# # These are equivalent:
# docker build -t protx .
# docker build --target final -t protx .
# docker build --target development -t protx .

# # To build other stages, you must specify:
# docker build --target inference -t protx:inference .

# Set default command to bash
CMD ["/bin/bash"] 