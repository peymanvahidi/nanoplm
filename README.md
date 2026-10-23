<a name="readme-top"></a>

<div align="center">

<img src="https://github.com/user-attachments/assets/dd520214-1f12-44c6-a6da-716934e4e981" alt="logo" width="600"/>

**F**rom **F**ASTA to **F**oundation model - **F**ast.

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/peymanvahidi/nanoplm/publish-to-pypi.yml?style=plastic&logo=github-actions&label=CI)](https://github.com/peymanvahidi/nanoplm/actions/workflows/publish-to-pypi.yml)
[![License](https://img.shields.io/github/license/peymanvahidi/nanoplm?style=plastic&color=orange&logo=github&label=License)](./LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/nanoplm?style=plastic&color=4b8bbe&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/nanoplm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)
[![Docs](https://img.shields.io/badge/docs-nanoPLM-14b8a6?style=plastic&logo=readthedocs&logoColor=white)](https://peymanvahidi.github.io/nanoplm/)

### 📖 [Read the documentation →](https://peymanvahidi.github.io/nanoplm/)

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

`nanoplm eval` runs on [biotrainer](https://github.com/sacdallago/biotrainer)'s autoeval pipeline, so the `eval` extra pulls in **biotrainer 2.1.1** (plus `biotrainer-core`):

```bash
uv pip install "nanoplm[eval] @ git+https://github.com/peymanvahidi/nanoplm"   # or: uv pip install -e ".[eval]"
```

If `~/.cache/biotrainer/autoeval` holds datasets from an older biotrainer, `nanoplm eval` refuses to start until you set `dataset_dir` to a fresh directory or delete that cache.

To install only biotrainer at the pinned version, without the extra:

```bash
uv pip install "biotrainer==2.1.1" "biotrainer-core==2.1.1"
```

---

## 🤖 Zero‑to‑model in 4 commands

### 1. Get data YAML file

```bash
nanoplm data get-yaml
```

> You'll get [data.yaml](https://peymanvahidi.github.io/nanoplm/reference/data/) and `dvc.yaml`. The template builds 20,000 SwissProt sequences. To use your own sequences, set `fasta:` under `sources` to a local `.fasta` or `.fasta.gz` file. Don't edit `dvc.yaml`. `from-yaml` rewrites it from `data.yaml` on every run, and [DVC](https://dvc.org/) uses it to skip the steps whose inputs have not changed.

### 2. Prepare your data

This tokenizes the sequences into train and val shards, and writes a `.data_manifest` that describes them, all under `output_dir`:

```bash
nanoplm data from-yaml
```

> This reads `data.yaml` in your current directory. To use another file, pass its path: `nanoplm data from-yaml <path/to/data.yaml>`.

> [Prepare data](https://peymanvahidi.github.io/nanoplm/pipeline/data/) covers the pre-run table, invalid sequences and moving an old `data_dirs` config. [Cluster subsampling](https://peymanvahidi.github.io/nanoplm/guides/cluster-subsampling/) trains on one random member of every cluster each epoch, for example a UniRef90 sequence from each UniRef50 cluster.

Now your data is ready. Let's start the training.

### 3. Get a pretrain YAML file

```bash
nanoplm pretrain get-yaml
```

> This writes [pretrain.yaml](https://peymanvahidi.github.io/nanoplm/reference/pretrain/) to your current directory.

```bash
nanoplm distill get-yaml
```

> This writes `distill.yaml` to your current directory. The distillation pipeline is
> [unmaintained](https://peymanvahidi.github.io/nanoplm/distillation/) — use pretraining instead.

### 4. Start your pretraining

```bash
nanoplm pretrain from-yaml
```

The stock `pretrain.yaml` sets `pure_torch: true`, so no backend flag is needed.

---

## Configuration

Everything is controlled by three YAML files. Scaffold one, edit it, run it:

| Stage | Scaffold | Run | Reference |
|---|---|---|---|
| Data | `nanoplm data get-yaml` | `nanoplm data from-yaml` | [data.yaml](https://peymanvahidi.github.io/nanoplm/reference/data/) |
| Pretraining | `nanoplm pretrain get-yaml` | `nanoplm pretrain from-yaml` | [pretrain.yaml](https://peymanvahidi.github.io/nanoplm/reference/pretrain/) |
| Evaluation | `nanoplm eval get-yaml` | `nanoplm eval from-yaml` | [eval.yaml](https://peymanvahidi.github.io/nanoplm/reference/eval/) |

The reference tables are generated directly from the `get-yaml` templates and the config
dataclasses, and CI fails if they drift — so they cannot go stale the way a hand-written
README section can.

**Start here:** [How the pipeline fits together](https://peymanvahidi.github.io/nanoplm/pipeline/)
· [Your first PLM](https://peymanvahidi.github.io/nanoplm/start/first-plm/)
· [Troubleshooting](https://peymanvahidi.github.io/nanoplm/troubleshooting/)

`nanoplm data from-yaml` resolves a relative path in its config against the directory that
holds the config file. `nanoplm pretrain from-yaml` and `nanoplm eval from-yaml` resolve a
relative path against the working directory you run them from, not against the config file.
Run those two from the directory their relative paths belong to. In an sbatch script, set
`#SBATCH --chdir` to that directory. An absolute path in a config is used as written.

---

## Requirements

- Python 3.12+, which backs `nanoplm eval` (biotrainer 2.1.1 caps below 3.14, so 3.12–3.13). Developed and tested on 3.12.
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
