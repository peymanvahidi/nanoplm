<a name="readme-top"></a>

<div align="center">

<img src="https://github.com/user-attachments/assets/dd520214-1f12-44c6-a6da-716934e4e981" alt="logo" width="600"/>

**F**rom **F**ASTA to **F**oundation model - **F**ast.

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/peymanvahidi/nanoplm/publish-to-pypi.yml?style=plastic&logo=github-actions&label=CI)](https://github.com/peymanvahidi/nanoplm/actions/workflows/publish-to-pypi.yml)
[![License](https://img.shields.io/github/license/peymanvahidi/nanoplm?style=plastic&color=orange&logo=github&label=License)](./LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/nanoplm?style=plastic&color=4b8bbe&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/nanoplm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)

<p>🚀 Ship a protein language model easier and faster. nanoPLM gives you a batteries‑included CLI, reproducible data workflows, and a simple YAML files to control everything.</p>

</div>

## 🧬 What makes nanoPLM different?

- **Control everything with simple YAML files**: Prepare your data and Pretrain your model, with YAML files.
- **Data you can trust**: Using Data Version Control (DVC) under the hood.
- **Scale sensibly**: Multi‑GPU ready.

---

## 🛠️ Install

> **Supported platform:** Linux x86_64 · NVIDIA **Hopper** GPU (H100/H200, sm90) ·
> CUDA ≥ 13.0 driver. Every install carries the full feature set, torch 2.12.1+cu130,
> TransformerEngine 2.16 (FP8), FlashAttention-3, SonicMoE/QuACK kernels, and Dion.
> Other GPUs or OSes may partially work but are unsupported.

```bash
uv venv && source .venv/bin/activate
uv pip install "nanoplm @ git+https://github.com/peymanvahidi/nanoplm" \
    --extra-index-url https://download.pytorch.org/whl/cu130
nanoplm deps install
```

Contributors: clone and use `uv pip install -e .` (add `--extra-index-url https://download.pytorch.org/whl/cu130` to skip a duplicate torch download), then `nanoplm deps install`. Once nanoPLM is public on PyPI this becomes `uv pip install nanoplm`.

**On clusters (e.g. JURECA):** load the toolchain first, then install from a **login node** (needs internet; no GPU; the TransformerEngine source build takes ~10 min):

```bash
module load CUDA/13 NCCL cuDNN Ninja    # do NOT module purge
nanoplm deps install
```

### Benchmarking extra

`nanoplm eval` runs on [biotrainer](https://github.com/sacdallago/biotrainer)'s autoeval pipeline, so the `eval` extra pulls in **biotrainer 1.4.0**:

```bash
uv pip install "nanoplm[eval] @ git+https://github.com/peymanvahidi/nanoplm"   # or: uv pip install -e ".[eval]"
```

To take biotrainer straight from source instead, use the command below:

```bash
uv pip install biotrainer
```

---

## 🤖 Zero‑to‑model in 4 commands

### 1. Get data YAML file

```bash
nanoplm data get-yaml
```

>You'll get a `data.yaml` and a `dvc.yaml`. Just edit the `data.yaml` if you want.

> We're using DVC under the hood, so you can track your data version.

### 2. Prepare your data

Use the command below to prepare your data for pLM pretraining (you'll get train and val FASTAs)

```bash
nanoplm data from-yaml
```

> By default, this picks up `data.yaml` in your current directory. You can optionally specify a different path argument (relative or absolute) if needed.
Like: `nanoplm data from-yaml <path/to/data.yaml>`


📊 Now your data is ready! Let's start the training.

### 3. Get a pretrain YAML file

```bash
nanoplm pretrain get-yaml
```

> This writes a `pretrain.yaml` to your current directory.

### 4. Start your pretraining

```bash
nanoplm pretrain from-yaml
```

---

## Configuration

Every stage is driven by a YAML file: generate it with `get-yaml`, edit it, then hand it back with `from-yaml`. Each generated file carries an inline comment on every field, so the file you get *is* the reference.

| File | Generate with | Configures |
| --- | --- | --- |
| `data.yaml` | `nanoplm data get-yaml` | source FASTA, filtering, shuffling, train/val split, pretraining shard generation |
| `pretrain.yaml` | `nanoplm pretrain get-yaml` | model architecture, MLM objective, optimizer and LR schedule, precision, packing, multi-GPU, resume |
| `eval.yaml` | `nanoplm eval get-yaml` | downstream benchmarks for a trained checkpoint (needs the [benchmarking extra](#benchmarking-extra)) |

`nanoplm data get-yaml` also writes a `dvc.yaml` alongside the config, so the data
stages stay version-tracked.

Tip: Paths are resolved relative to where you run the command (not where the YAML lives).

---

## Requirements

- Python 3.11+, which backs `nanoplm eval` (it also caps below 3.14, so 3.11–3.13). Developed and tested on 3.12.
- Linux x86_64 with an NVIDIA Hopper GPU; see the supported-platform note under [Install](#️-install).

---

## Contributing

PRs welcome. If you’re unsure where to start, open an issue with your use‑case.

---

## Like it? Star it.

If nanoPLM saved you time, a star helps others find it and keeps development going.

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top
    </a>
</p>
