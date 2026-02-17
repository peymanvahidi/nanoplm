<a name="readme-top"></a>

<div align="center">

<img src="https://github.com/user-attachments/assets/dd520214-1f12-44c6-a6da-716934e4e981" alt="logo" width="600"/>

**F**rom **F**ASTA to **F**oundation model — **F**ast.

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/peymanvahidi/nanoplm/publish-to-pypi.yml?style=plastic&logo=github-actions&label=CI)](https://github.com/peymanvahidi/nanoplm/actions/workflows/publish-to-pypi.yml)
[![License](https://img.shields.io/github/license/peymanvahidi/nanoplm?style=plastic&color=orange&logo=github&label=License)](./LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/nanoplm?style=plastic&color=4b8bbe&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/nanoplm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic)](https://github.com/psf/black)

<p>🚀 Ship a protein language model without writing a training loop. nanoPLM gives you a batteries‑included CLI, reproducible data workflows, and a simple YAML files to control everything.</p>

</div>

## 🧬 What makes nanoPLM different?

- **Control everything with simple YAML files**: Prepare your data and Pretrain your model, with YAML files.
- **Data you can trust**: Using Data Version Control (DVC) under the hood.
- **Scale sensibly**: Multi‑GPU ready.

---

## 🛠️ Install

Install the package from PyPi

```bash
pip install nanoplm
```
Remember for CUDA, you should install some other dependencies as well.
```bash
pip install "nanoplm[cuda]"

---

## 🤖 Zero‑to‑model in 4 commands

### 1. Get data YAML file

```bash
nanoplm data get-yaml
```

>You'll get [params.yaml](#data-preparation-yaml) and a dvc.yaml files. Just edit the params.yaml if you want.

> We're using DVC under the hood, so you can track your data version.

### 2. Prepare your data

Use the command below to prepare your data for pLM pretraining (you'll get train and val FASTAs)

```bash
nanoplm data from-yaml
```

> By default, this uses `params.yaml` in your current directory. You can optionally specify a different path argument (relative or absolute) if needed.
Like: `nanoplm data from-yaml <path/to/params.yaml>`


📊 Now your data is ready! Let's start the training.

### 3. Get a pretrain or distillation YAML file

```bash
nanoplm pretrain get-yaml
```

> This writes [pretraining YAML file](#pretraining-yaml) to your current directory.

```bash
nanoplm distill get-yaml
```

> This writes [distillation YAML file](#distill-yaml) to your current directory.

### 4. Start your pretraining or distillation

```bash
nanoplm pretrain from-yaml
```
or
```bash
nanoplm distill from-yaml
```

---

## Data Preparation YAML

```yaml
data_params:
  # Pipeline mode: 'pretrain', 'distillation', or 'none'
  # - 'pretrain': Generate HDF5 shards for MLM pretraining
  # - 'distillation': Generate teacher embeddings for knowledge distillation
  # - 'none': Only run data preparation (download, filter, split)
  pipeline_mode: "pretrain"

  seqs_num: 20000
  min_seq_len: 20
  max_seq_len: 512
  val_ratio: 0.1
  device: "auto"

  shuffle_backend: "biopython"  # or "seqkit" (faster, requires installation)
  shuffle: true
  shuffle_seed: 24
  filter_skip_n: 0

# Pretrain config (used when pipeline_mode: 'pretrain')
# A .data_manifest file will be created in output_dir for use by pretrain pipeline
pretrain_config:
  output_dir: "output/data/pretrain_data"  # Will contain train/ and val/ subdirs
  samples_per_shard: 2000
  max_workers: 2  # -1 to use all available CPUs
  force: false

# Distillation config (used when pipeline_mode: 'distillation')
# A .data_manifest file will be created in output_dir for use by distill pipeline
distillation_config:
  output_dir: "output/data/distillation_data"  # Will contain train/ and val/ subdirs
  on_the_fly: false  # If true, skip embedding generation (embeddings computed during training)
  samples_per_shard: 2000  # -1 for single file (no sharding)
  teacher_model: "prott5"
  embed_calc_batch_size: 4

# Data directories
data_dirs:
  url: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
  # swissprot: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz"
  # trembl: "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz"
  # uniref50: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
  # uniref90: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"
  # uniref100: "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz"
  compressed_fasta: "output/data/raw/uniref50.fasta.gz"
  extracted_fasta: "output/data/raw/uniref50.fasta"
  shuffled_fasta: "output/data/raw/uniref50_shuffled.fasta"
  filtered_fasta: "output/data/filter/uniref50_filtered.fasta"
  splitted_fasta_dir: "output/data/split"
```

## Pretraining YAML

```yaml
# Pretraining configuration for nanoPLM
#
# IMPORTANT: Before running pretraining, ensure you have prepared your data with:
#   1. Set pipeline_mode: 'pretrain' in params.yaml
#   2. Run: nanoplm data from-yaml
# This will generate binary shards and a .data_manifest file.

model:
  hidden_size: 1024
  intermediate_size: 2048
  num_hidden_layers: 16
  num_attention_heads: 16
  vocab_size: 32
  mlp_activation: "swiglu"
  mlp_dropout: 0.0
  mlp_bias: false
  attention_bias: false
  attention_dropout: 0.0
  classifier_activation: "gelu"
  max_position_embeddings: 1024 # needs to be at least as long as max seq length

pretraining:
  # Dataset directory (contains .data_manifest from nanoplm data from-yaml)
  # Note: paths are RELATIVE to where you RUN the command, NOT the YAML file.
  dataset_dir: "output/data/pretrain_data"

  # Output model path
  ckp_dir: "output/pretraining_checkpoints"

  # Hyperparameters
  #   micro_batch_size: samples per GPU per forward pass (limited by GPU memory)
  #   global_batch_size: total tokens per optimizer step across all GPUs
  #   gradient_accumulation_steps is inferred automatically:
  #     grad_accum = ceil(global_batch_size / (micro_batch_size * max_seq_len * num_gpus))
  micro_batch_size: 32
  global_batch_size: 1048576  # 2^20 ≈ 1M tokens/step (based on PLM best practices)
  num_epochs: 10
  warmup_ratio: 0.05

  optimizer: "adamw"  # adamw, stable_adamw, muon, normuon (muon and normouon only supported with CUDA)
  # AdamW hyperparameters (also used for AdamW side [1D and embedding/unembed params] when optimizer=muon or normuon)
  adam_learning_rate: 1e-3
  adam_weight_decay: 0.0
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8
  # Muon/NorMuon hyperparameters (used only when optimizer: muon or normuon)
  muon_learning_rate: 1e-3
  muon_weight_decay: 0.01
  muon_cautious_weight_decay: true
  muon_use_polar_express: false
  muon_momentum: 0.95
  muon_nesterov: true
  muon_eps: 1e-7

  mlm_probability: 0.3
  mask_replace_prob: 0.8
  random_token_prob: 0.1
  keep_probability: 0.1
  logging_steps: 1
  eval_steps: 250
  save_steps: 5000
  seed: 42
  num_workers: "auto"
  prefetch_factor: 2

  # Mixed precision training (recommended: keep enabled for 1.5-3x speedup)
  # When bf16 is true, automatically selects the best precision for your hardware:
  #   - CUDA Ampere+ (A100, RTX 3090+): bf16 + TF32
  #   - CUDA Volta/Turing (V100, RTX 2080): fp16 fallback
  #   - Apple Silicon (M1/M2/M3): fp16 (hardware accelerated)
  #   - CPU: fp32 (no mixed precision)
  bf16: true
  tf32: true  # TF32 mode on Ampere+ CUDA GPUs only (automatically not used on MPS/CPU)
             # Provides 3x faster fp32 matmuls with negligible precision loss

  multi_gpu: false
  world_size: 1  # Use "auto" if you want to use all available GPUs
  project_name: "nanoplm-pretraining"

# Pure-torch training loop settings (alternative to HF Trainer).
pure_torch:
  enabled: false
  # torch.compile: compile the model for faster training. Disable for debugging,
  # unsupported hardware (e.g. Apple Silicon), or to avoid warmup overhead.
  use_compile: true
  # Sequence packing: concatenates shorter sequences into fewer rows to eliminate
  # padding waste and increase GPU utilization. Requires flash attention.
  use_packing: false
  # Fixed row count for static-shape compilation when use_packing is true (enables torch.compile dynamic=False).
  # Set to ceil(micro_batch_size * avg_len / max_seq_len) + margin. Leave null for dynamic=True.
  target_packed_rows: null

resume:
  # Set is_resume: true to resume training from a checkpoint
  # When resuming, the model, tokenizer, and training state will be loaded from checkpoint_dir
  # extra_epochs: adds to 'pretraining.num_epochs' to define total epochs.
  is_resume: false
  checkpoint_dir: "output/pretraining_checkpoints/run-1/checkpoint-1"
  extra_epochs: 0
```

## Distill YAML

```yaml
# Distillation configuration for nanoPLM
#
# IMPORTANT: Before running distillation, ensure you have prepared your data with:
#   1. Set pipeline_mode: 'distillation' in params.yaml
#   2. Set distillation_config.on_the_fly in params.yaml:
#      - false (default): Pre-compute teacher embeddings during data preparation
#      - true: Generate teacher embeddings on-the-fly during training
#   3. Run: nanoplm data from-yaml
# This will generate a .data_manifest file with the appropriate configuration.

model:
  hidden_size: 1024
  intermediate_size: 2048
  num_hidden_layers: 16
  num_attention_heads: 16
  mlp_activation: "swiglu"
  mlp_dropout: 0.0
  mlp_bias: false
  attention_bias: false
  attention_dropout: 0.0
  classifier_activation: "gelu"
  projection_layer: true  # Set to false if student hidden_size matches teacher (1024)

distillation:

  # Dataset directory (contains .data_manifest from nanoplm data from-yaml)
  # The manifest automatically provides:
  #   - max_seq_len, max_seqs_num, val_ratio
  #   - on_the_fly mode and dataset paths (FASTA or H5)
  # Note: paths are RELATIVE to where you RUN the command, NOT the YAML file.
  dataset_dir: "output/data/distillation_data"

  # Output checkpoint path
  ckp_dir: "output/distillation_checkpoints"

  # Training hyperparameters
  num_epochs: 10
  batch_size: 32
  learning_rate: 1e-3
  gradient_accumulation_steps: 1
  warmup_ratio: 0.05

  # LR scheduler
  lr_scheduler: "cosine"  # cosine, linear, polynomial, constant
  lr_scheduler_kwargs: {}

  # Data loader optimization
  max_open_files: 5
  chunk_size: 32
  prefetch_batches: 2
  use_threading: true
  num_workers: 4

  # Checkpointing
  project_name: "nanoplm-distillation"
  logging_steps: 10
  eval_steps: 50
  save_steps: 100

  # Mixed precision training (recommended: keep enabled for 1.5-3x speedup)
  # When bf16 is true, automatically selects the best precision for your hardware:
  #   - CUDA Ampere+ (A100, RTX 3090+): bf16 + TF32
  #   - CUDA Volta/Turing (V100, RTX 2080): fp16 fallback
  #   - Apple Silicon (M1/M2/M3): fp16 (hardware accelerated)
  #   - CPU: fp32 (no mixed precision)
  bf16: true
  tf32: true  # TF32 mode on Ampere+ CUDA GPUs only (automatically not used on MPS/CPU)
             # Provides 3x faster fp32 matmuls with negligible precision loss

  # Distributed training
  multi_gpu: false
  world_size: 1
  seed: 42

resume:
  # Set is_resume: true to resume training from a checkpoint
  # When resuming, the model, tokenizer, and training state will be loaded from checkpoint_dir
  # extra_epochs: adds to 'distillation.num_epochs' to define total epochs.
  is_resume: false
  checkpoint_dir: "output/distillation/run-1/checkpoint-1"
  extra_epochs: 0
```

Tip: Paths are resolved relative to where you run the command (not where the YAML lives).

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
