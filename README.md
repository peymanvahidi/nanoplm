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

Install the package from PyPI:

```bash
pip install nanoplm
```

For a local editable install:

```bash
uv pip install -e .
```
For benchmarking you also need Biotrainer.
```bash
pip install git+https://github.com/sacdallago/biotrainer@main
```

For CUDA training, install Dion manually from the Microsoft repository:

```bash
pip install git+https://github.com/microsoft/dion.git
```

MoE training now uses an in-tree CUTLASS grouped GEMM extension instead of the
external `grouped_gemm` package. The extension is JIT-built on first use, so
you only need a CUDA toolkit that matches `torch.version.cuda`.

1. Check your active PyTorch CUDA version:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

2. Point `CUDA_HOME` at the matching toolkit and set the architectures you want
to build for:

```bash
export CUDA_HOME=/path/to/cuda-matching-torch
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"  # H100; adjust for your GPU
```

3. Run nanoPLM normally. The first CUDA MoE call will compile the vendored
CUTLASS backend and cache it under `.cache/torch_extensions`:

```bash
nanoplm pretrain from-yaml --pure-torch
```

If you need different architectures, override `TORCH_CUDA_ARCH_LIST` before
running your first MoE job. If CUTLASS is not checked out under
`third_party/cutlass`, set `NANOPLM_CUTLASS_ROOT` or `CUTLASS_ROOT` to a CUTLASS
checkout.

### TransformerEngine on JURECA (JSC)

The prebuilt `transformer_engine` wheels (2.14–2.16) fail to import on JURECA
(`undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`) because they link
a newer cublasLt than `module load CUDA/13` provides. Build it from source once,
from a **login node** (needs internet; no GPU required):

```bash
bash install_te_jureca.sh
```

This clones TE 2.16, builds it for sm90 + sm120 against the cluster's CUDA,
installs it into `.venv`, and verifies the import. Override `NVTE_CUDA_ARCHS`
(e.g. `NVTE_CUDA_ARCHS=90`) to retarget. Only needed for the FP8 / `--pure-te`
pipelines; the default bf16 / `--pure-torch` path does not require TE.

### Optional: SonicMoE expert backend

Set `moe_kernel_backend: sonicmoe` in the pretraining YAML to run the routed
experts on the [SonicMoE](https://github.com/Dao-AILab/sonic-moe) /
[QuACK](https://github.com/Dao-AILab/quack) fused kernels (sync-free forward and
backward, dispatch fused into the GEMM). It requires our SonicMoE fork, which
adds squared-ReLU (`srelu`) support on top of upstream (PR
[Dao-AILab/sonic-moe#64](https://github.com/Dao-AILab/sonic-moe/pull/64)):

```bash
pip install quack-kernels            # use quack-kernels[cu13] on CUDA 13
pip install git+https://github.com/alint77/sonic-moe.git@feat/srelu-relu_sq-support
```

Requires a Hopper (SM90) or newer GPU. The default `moe_kernel_backend: cutlass`
needs none of this.

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
