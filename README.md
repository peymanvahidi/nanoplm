<a name="readme-top"></a>

<div align="center">

<img src="https://github.com/user-attachments/assets/dd520214-1f12-44c6-a6da-716934e4e981" alt="logo" width="600"/>

**F**rom **F**ASTA to **F**oundation model — **F**ast.

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/peymanvahidi/nanoplm/publish-to-pypi.yml?style=plastic&logo=github-actions&label=CI)](https://github.com/peymanvahidi/nanoplm/actions/workflows/publish-to-pypi.yml)
[![License](https://img.shields.io/github/license/peymanvahidi/nanoplm?style=plastic&color=orange&logo=github&label=License)](./LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/nanoplm?style=plastic&color=4b8bbe&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/nanoplm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)
[![Docs](https://img.shields.io/badge/docs-nanoPLM-14b8a6?style=plastic&logo=readthedocs&logoColor=white)](https://peymanvahidi.github.io/nanoplm/)

### 📖 [Read the documentation →](https://peymanvahidi.github.io/nanoplm/)

<p> nanoPLM gives you a batteries‑included CLI, reproducible data workflows, and a simple YAML files to control everything. THE FOCUS IS ON HAVING A SIMPLIFIED ENCODER ONLY TRAINING PIPELINE SIMILAR TO NANOCHAT, by no means is this project is producing SOTA models, this is mostly an educational playground to experiment on various aspects of encoder model training e2e pipeline</p>

</div>

## What makes nanoPLM different?

- **Control everything with simple YAML files**: Prepare your data and Pretrain your model, with YAML files.
- **Data you can trust**: Using Data Version Control (DVC) under the hood.
- **Scale sensibly**: Multi‑GPU ready.

---

## 🛠️ Install

> **Supported platform:** Linux x86_64 or aarch64 · NVIDIA **Hopper** GPU
> (H100/H200/GH200, sm90) · CUDA ≥ 13.0. Every full install carries the same
> stack: torch 2.12.1+cu130, TransformerEngine 2.16 (FP8), FlashAttention-3,
> SonicMoE/QuACK, and Dion. Other GPUs or OSes may partially work but are
> unsupported. Step-by-step docs:
> [Install](https://peymanvahidi.github.io/nanoplm/start/install/).

```bash
uv venv && source .venv/bin/activate
uv pip install "nanoplm @ git+https://github.com/peymanvahidi/nanoplm" \
    --extra-index-url https://download.pytorch.org/whl/cu130
nanoplm deps install
```

Contributors: clone and use `uv pip install -e .` (add
`--extra-index-url https://download.pytorch.org/whl/cu130` so torch resolves to
the cu130 wheel), then `nanoplm deps install`. Once nanoPLM is public on PyPI
this becomes `uv pip install nanoplm`.

**On clusters (e.g. JURECA):** load the toolchain first, then install from a
**login node** (needs internet; no GPU; the TransformerEngine source build takes
~10 min):

```bash
module load CUDA/13 NCCL cuDNN Ninja    # do NOT module purge
nanoplm deps install
```

Verify on a GPU node with `nanoplm deps verify`.

### Benchmarking extra

`nanoplm eval` runs on [biotrainer](https://github.com/sacdallago/biotrainer)'s
autoeval pipeline, so the `eval` extra pulls in **biotrainer 1.4.0**:

```bash
uv pip install "nanoplm[eval] @ git+https://github.com/peymanvahidi/nanoplm"
# or, for an editable checkout:
uv pip install -e ".[eval]"
```

---

## 🤖 Zero‑to‑model in 4 commands

### 1. Get data YAML file

```bash
nanoplm data get-yaml
```

>You'll get [data.yaml](https://peymanvahidi.github.io/nanoplm/reference/data/) and a dvc.yaml files. Just edit the params.yaml if you want.

> We're using DVC under the hood, so you can track your data version.

### 2. Prepare your data

Use the command below to prepare your data for pLM pretraining (you'll get train and val FASTAs)

```bash
nanoplm data from-yaml
```

> By default, this uses `params.yaml` in your current directory. You can optionally specify a different path argument (relative or absolute) if needed.
Like: `nanoplm data from-yaml <path/to/params.yaml>`

> For pretraining shard generation, nanoPLM auto-uses a native C tokenizer/writer backend (OpenMP-enabled if available) when using the default protein tokenizer.  
> Disable it with `NANOPLM_DISABLE_NATIVE_SHARD_WRITER=1`.
>
> `shuffle`, `filter`, and `split` also use native C FASTA backends when available.  
> Disable them with `NANOPLM_DISABLE_NATIVE_FASTA_OPS=1`.


📊 Now your data is ready! Let's start the training.

### 3. Get a pretrain or distillation YAML file

```bash
nanoplm pretrain get-yaml
```

> This writes [pretrain.yaml](https://peymanvahidi.github.io/nanoplm/reference/pretrain/) to your current directory.

```bash
nanoplm distill get-yaml
```

> This writes `distill.yaml` to your current directory. The distillation pipeline is
> [unmaintained](https://peymanvahidi.github.io/nanoplm/distillation/) — use pretraining instead.

### 4. Start your pretraining or distillation

```bash
nanoplm pretrain from-yaml --pure-torch
```

A backend flag is required — see [Choose a backend](https://peymanvahidi.github.io/nanoplm/guides/backends/).

---

## Configuration

Everything is controlled by three YAML files. Scaffold one, edit it, run it:

| Stage | Scaffold | Run | Reference |
|---|---|---|---|
| Data | `nanoplm data get-yaml` | `nanoplm data from-yaml` | [data.yaml](https://peymanvahidi.github.io/nanoplm/reference/data/) |
| Pretraining | `nanoplm pretrain get-yaml` | `nanoplm pretrain from-yaml --pure-torch` | [pretrain.yaml](https://peymanvahidi.github.io/nanoplm/reference/pretrain/) |
| Evaluation | `nanoplm eval get-yaml` | `nanoplm eval from-yaml` | [eval.yaml](https://peymanvahidi.github.io/nanoplm/reference/eval/) |

The reference tables are generated directly from the `get-yaml` templates and the config
dataclasses, and CI fails if they drift — so they cannot go stale the way a hand-written
README section can.

**Start here:** [How the pipeline fits together](https://peymanvahidi.github.io/nanoplm/pipeline/)
· [Your first PLM](https://peymanvahidi.github.io/nanoplm/start/first-plm/)
· [Troubleshooting](https://peymanvahidi.github.io/nanoplm/troubleshooting/)

Paths inside a config are resolved relative to the config file's directory.

---

## Requirements

- Python 3.12+
- macOS or Linux
- GPU recommended (CPU is fine for tiny tests)

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
