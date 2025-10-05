<div align="center">

<img src="https://github.com/user-attachments/assets/dd520214-1f12-44c6-a6da-716934e4e981" alt="logo" width="600"/>

**F**rom **F**ASTA to **F**oundation model — **F**ast.

</div>

<p>🚀 Ship a protein language model without writing a training loop. nanoPLM gives you a batteries‑included CLI, reproducible data workflows, and a simple YAML files to control everything.</p>

---

## 🧬 What makes nanoPLM different?

- **Control everything with simple YAML files**: Prepare your data and Pretrain your model, with YAML files.
- **Data you can trust**: Using Data Version Control (DVC) under the hood.
- **Scale sensibly**: Multi‑GPU ready.

---

## 🛠️ Install

Clone the repo and then

```bash
pip install .
```
PyPi package comming soon!

---

## 🤖 Zero‑to‑model in 4 commands

### 1. Get data YAML file

```bash
nanoplm data get-yaml
```

You'll get a params.yaml and dvc.yaml files. Just edit the params.yaml if you want.
> We're using DVC under the hood, so you can track your data version.

[This](#data-preparation-yaml) is the YAML file you get for data preparation.

### 2. Prepare your data

Use the command below to prepare your data for pLM pretraining (you'll get train and val FASTAs)

```bash
nanoplm data from-yaml <path/to/params.yaml>
```

Or if you want to prepare your data for Knowledge distillation also use the `--distillation` flag.
This way two extra stages for calculating teacher embeddings for train and val files would also happen.

```bash
nanoplm data from-yaml <path/to/params.yaml> --distillation
```

📊 Now your data is ready! Let's start the training.

### 3. Get a pretrain YAML file

```bash
nanoplm pretrain get-yaml
```

This writes the pretraining YAML to your current directory. Prefer a different folder?

```bash
nanoplm pretrain get-yaml <output/dir>
```

### 4. Start your pretraining

```bash
nanoplm pretrain from-yaml <path/to/pretrain.yaml>
```

---

## Data Preparation YAML

```yaml
data_params:
  seqs_num: 20000
  min_seq_len: 20
  max_seq_len: 512
  val_ratio: 0.1

  teacher_model: "prott5"
  embed_calc_batch_size: 4

  device: "auto"
  
  shuffle: true
  shuffle_seed: 24
  # If you want to skip some sequences
  filter_skip_n: 0
  # For sharding the train and val datasets for KNOWLEDGE DISTILLATION
  train_shards: 5
  val_shards: 2

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

  kd_train_dir: "output/data/kd_dataset/train"
  kd_val_dir: "output/data/kd_dataset/val"
```

## Pretraining YAML

```yaml
# Pretraining configuration for nanoPLM

model:
  hidden_size: 1024
  intermediate_size: 2048
  num_hidden_layers: 16
  num_attention_heads: 16
  vocab_size: 29
  mlp_activation: "swiglu"
  mlp_dropout: 0.0
  mlp_bias: False
  attention_bias: False
  attention_dropout: 0.0
  classifier_activation: "gelu"

pretraining:
  # Dataset
  # Note: these paths are RELATIVE to where you RUN the command NOT the YAML file.
  train_fasta: "output/data/split/train.fasta"
  val_fasta: "output/data/split/val.fasta"

  # Output model path
  ckp_dir: "output/pretraining_checkpoints"

  # Hyperparameters
  max_length: 1024
  batch_size: 32
  num_epochs: 10
  # Info for lazy dataset loading
  # True: Low memory usage, tokenize on-demand (slower iteration, faster startup)
  # False: High memory usage, tokenize all sequences at once (faster iteration, slower startup)
  lazy_dataset: False
  warmup_ratio: 0.05
  optimizer: "adamw" # adamw, stable_adamw
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8
  learning_rate: 3e-6
  weight_decay: 0.0
  gradient_accumulation_steps: 1
  mlm_probability: 0.3
  mask_replace_prob: 0.8
  random_token_prob: 0.1
  keep_probability: 0.1
  logging_steps_percentage: 0.01 # 100 logging in total 
  eval_steps_percentage: 0.025 # 40 evaluations in total 
  save_steps_percentage: 0.1 # 10 saves in total 
  seed: 42
  num_workers: 0
  multi_gpu: False
  world_size: 1 # Use "auto" if you want to use all available GPUs
  run_name: "nanoplm-pretraining"
```

Tip: Paths are resolved relative to where you run the command (not where the YAML lives).

---

## Requirements

- Python 3.10+
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