<div align="center">

# nanoPLM

**From FASTA to foundation model — fast.**

</div>

<p>Ship a protein language model without writing a training loop. nanoPLM gives you a batteries‑included CLI, reproducible data workflows, and a single YAML to control everything.</p>

---

## 🧬 What makes nanoPLM different?

- **Zero boilerplate**: Generate a ready‑to‑edit `pretrain.yaml` and hit run.
- **Data you can trust**: Deterministic shuffle/split pipelines using `dvc`.
- **Scale sensibly**: Multi‑GPU ready; CPU works for tiny runs.

---

## Install

Clone the repo and then

```bash
pip install .
```
PyPi package comming soon!

---

## 🤖 Zero‑to‑model in 3 commands

1) Prepare data

```bash
dvc repro split # Prepare and split dataset into train/val
```

2) Create a starter YAML

```bash
nanoplm pretrain get-yaml
```

This writes a YAML to your current directory. Prefer a different folder?

```bash
nanoplm pretrain get-yaml <output_dir>
```

3) Train

```bash
nanoplm pretrain from-yaml <path/to/pretrain.yaml>
```

---

## The YAML you’ll edit

```yaml
# Pretraining configuration for nanoPLM

model:
  hidden_size: 1024
  intermediate_size: 2048
  num_hidden_layers: 16
  num_attention_heads: 16
  vocab_size: 29
  mlp_activation: swiglu
  mlp_dropout: 0.0
  mlp_bias: false
  attention_bias: false
  attention_dropout: 0.0
  classifier_activation: gelu

pretraining:
  # Dataset
  # Note: these paths are RELATIVE to where you RUN the command NOT the YAML file.
  train_fasta: output/data/split/train.fasta
  val_fasta: output/data/split/val.fasta

  # Output model path
  ckp_dir: output/pretraining_checkpoints

  # Hyperparameters
  max_length: 1024
  batch_size: 32
  num_epochs: 10
  warmup_ratio: 0.05
  optimizer: adamw # adamw, stable_adamw
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
  run_name: nanoplm-pretraining
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