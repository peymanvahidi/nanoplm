---
title: CLI
description: "Every nanoplm command, argument, and option, grouped by data, pretrain, eval, deps, and the unmaintained distill."
---

`nanoplm` has five command groups: `data`, `pretrain`, `eval`, `deps`, and `distill`. The pipeline groups share the same shape:

```
nanoplm <group> get-yaml | from-yaml | run
```

- **`get-yaml`** scaffolds a config file (`data.yaml`, `pretrain.yaml`, `eval.yaml`).
- **`from-yaml`** runs that config. This is the supported path.
- **`run`** takes the same knobs as command-line flags instead of a file.

For `pretrain` and `eval`, **`from-yaml` is the path to use.** `pretrain run` works, but it exposes a strictly smaller surface than the YAML: it has no flags for HSDP, MoE, NOBLE, recycling, the profiler, RoPE thetas, or FP8 CPU offload, and it silently defaults every one of them. Reach for it only for a quick one-off.

`deps` is different: it installs and verifies the pinned GPU stack. See [Install](/nanoplm/start/install/).

The root group takes `--version`. Every command takes `-h` / `--help`.

---

## deps

`nanoplm deps` installs and verifies the hard GPU dependency stack (Hopper + CUDA 13). It is required after installing the package itself; the package alone does not pull Dion, TransformerEngine, FlashAttention-3, or SonicMoE/QuACK.

### deps install

```
nanoplm deps install [OPTIONS]
```

Runs the packaged installer into the **active virtualenv**, then verifies in a fresh interpreter. Must be run with a venv activated. On clusters, load the CUDA 13 toolchain first and run from a login node (internet, no GPU required; TE source build ~10 min).

| Flag | Meaning | Default |
| --- | --- | --- |
| `--arch TEXT` | CUDA arch for source builds (`NVTE_CUDA_ARCHS`). `90` = Hopper / H100 | `90` |
| `--force` | Reinstall every component even if already satisfied | off |

```bash
module load CUDA/13 NCCL cuDNN Ninja    # clusters only
nanoplm deps install
```

### deps verify

```
nanoplm deps verify
```

Checks torch build (`2.12.1+cu130`), hard-dep imports (`dion`, `transformer_engine.pytorch`, `flash_attn_interface`, `quack`, `sonicmoe`), and GPU compute capability when a device is visible. Re-run on a GPU node after installing on a login node.

---

## data

`nanoplm data` builds the binary shards and the `.data_manifest` that pretraining consumes. Its `--help` is grouped, and that grouping is the right mental model:

- **Main commands**: `get-yaml`, `from-yaml`. Most users only ever need these two.
- **Individual pipeline stages**: `get-file`, `extract`, `check`, `shuffle`, `filter`, `split`, plus the pretraining-specific `pre-tokenize`. `from-yaml` orchestrates all of them through DVC. You can also run any of them standalone to debug one step.

Field-by-field descriptions live in the [data.yaml reference](/nanoplm/reference/data/).

### data get-yaml

```
nanoplm data get-yaml [OUTPUT] [OPTIONS]
```

Writes `data.yaml` **and** a matching `dvc.yaml` (the stage DAG). `OUTPUT` is a target directory and defaults to the current directory.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--distillation` | Write `distillation_data.yaml` instead of `data.yaml` | off |
| `--force` | Overwrite existing files | off |

```bash
nanoplm data get-yaml
```

```
Wrote data.yaml and dvc.yaml in <cwd>.
Edit data.yaml and run `nanoplm data from-yaml` to execute.
```

### data from-yaml

```
nanoplm data from-yaml [CONFIG] [OPTIONS]
```

Regenerates `dvc.yaml` from the config, runs `dvc init` if needed, then executes the pipeline with `dvc repro`. `CONFIG` defaults to `data.yaml` in the current directory (falling back to `distillation_data.yaml`).

| Flag | Meaning | Default |
| --- | --- | --- |
| `--target TEXT` | Run one DVC stage, as named in `dvc.yaml` (for example `split`, or `cluster_index` with cluster subsampling), instead of the whole pipeline | full pipeline |
| `--no-auto-init` | Fail instead of running `dvc init` when DVC is not initialized | off |
| `--force` | Pass `-f` to `dvc repro`, re-running stages DVC considers unchanged | off |
| `--verbose` | Pass `-v` to `dvc repro` | off |
| `--fix` | Set `drop_invalid_sequences: true` in the config (every occurrence, so a per-source override too), then run | off |
| `--dry-run` | Print the DVC plan without executing it | off |
| `--log-level [debug\|info\|warning\|error]` | nanoplm log level | `info` |

```bash
nanoplm data from-yaml
```

```
data.yaml: 1 source -> <cwd>/output/data/pretrain_data
source     fasta                         max seqs  len     val  clustering
swissprot  uniprot_sprot.fasta.gz (url)  20,000    20-512  0.1  off
Regenerated <cwd>/dvc.yaml
Rebuild: all (first run)
Running dvc repro (full pipeline)...
# ... dvc stage output ...
```

[Prepare data](/nanoplm/pipeline/data/#read-the-table-before-the-run) explains the table and the line under it. `--dry-run` needs a folder where `from-yaml` has already run once, and stops at the first download that has not happened yet. `--fix` is for when the `check` stage rejects invalid sequences: see [Prepare data](/nanoplm/pipeline/data/#invalid-sequences-will-stop-you-once).

### data get-file

```
nanoplm data get-file [DATASET] [OPTIONS]
```

Acquire a FASTA source. Give **exactly one** of a `DATASET` key (`swissprot`, `trembl`, `uniref50`, `uniref90`, `uniref100`), `--url`, or `--local-fasta`.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--url TEXT` | URL to download | `""` |
| `--local-fasta TEXT` | Local `.fasta` / `.fasta.gz` to link instead of downloading | `""` |
| `--output`, `-o PATH` | **Required.** Output file or directory | - |
| `--force`, `-f` | Overwrite existing output | off |

### data extract

```
nanoplm data extract [OPTIONS]
```

Decompress a `.gz` source. This stage is skipped entirely when the source is already a plain `.fasta`.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Compressed input | - |
| `--output`, `-o PATH` | **Required.** Output file or directory | - |
| `--force`, `-f` | Overwrite existing output | off |

### data check

```
nanoplm data check [OPTIONS]
```

Validate the FASTA. Writes `validation.json`, `invalid.fasta`, and `invalid_ids.txt` into the output directory. With `--drop-invalid false` it **exits non-zero** if any sequence is invalid, halting the pipeline.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Input FASTA | - |
| `--output`, `-o PATH` | **Required.** Output directory for the validation sidecars | - |
| `--drop-invalid [true\|false]` | Drop invalid sequences and exit 0 | `false` |

### data shuffle

```
nanoplm data shuffle [OPTIONS]
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Input FASTA | - |
| `--output`, `-o PATH` | **Required.** Output FASTA | - |
| `--backend [auto\|seqkit\|fast\|biopython]` | Shuffle backend; `auto` picks seqkit when installed, else the native `fast` backend | `auto` |
| `--seed INTEGER` | Random seed | none |
| `--shuffle [true\|false]` | `false` links the input to the output unchanged | `true` |

### data filter

```
nanoplm data filter [OPTIONS]
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Input FASTA | - |
| `--output`, `-o PATH` | **Required.** Output FASTA | - |
| `--min-seq-len INTEGER` | Drop sequences shorter than this | `20` |
| `--max-seq-len INTEGER` | Drop sequences longer than this | `1024` |
| `--seqs-num INTEGER` | Keep at most N sequences; `-1` keeps all | `-1` |
| `--skip-n INTEGER` | Skip N sequences from the start of the input | `0` |
| `--exclude-ids PATH` | Text file of sequence IDs (one per line) to exclude | none |

### data split

```
nanoplm data split [OPTIONS]
```

Split into `train.fasta` and `val.fasta`.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Filtered FASTA | - |
| `--output`, `-o PATH` | **Required.** Output directory | - |
| `--val-ratio FLOAT` | Fraction held out for validation | `0.1` |

### data pre-tokenize

```
nanoplm data pre-tokenize [OPTIONS]
```

Pretraining-specific. Tokenizes a FASTA into the binary shards (`shard_NNNN.bin` + `shard_NNNN.idx.npy`) that pretraining memory-maps.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Input FASTA | - |
| `--output`, `-o PATH` | **Required.** Output directory for the shards | - |
| `--max-seq-len INTEGER` | Pad/truncate length. Recorded in the manifest, and the value pretraining reads back for its token math | `1024` |
| `--max-workers INTEGER` | Parallel workers | `$NANOPLM_MAX_WORKERS`, else `2` |
| `--samples-per-shard INTEGER` | Sequences per shard | `10000` |
| `--register-tokens INTEGER` | Learnable scratch tokens (`0`, `1`, or `2`) appended after EOS | `0` |
| `--force`, `-f` | Overwrite existing output | off |

`write-manifest`, and the cluster subsampling stages `get-cluster-xml`, `cluster-members`, `cluster-pool` and `cluster-index`, also exist but are hidden: the generated `dvc.yaml` calls them. You do not run them yourself.

---

## pretrain

`nanoplm pretrain` trains the masked-language model. Fields are documented in the [pretrain.yaml reference](/nanoplm/reference/pretrain/).

### pretrain get-yaml

```
nanoplm pretrain get-yaml [OUTPUT] [OPTIONS]
```

Writes `./pretrain.yaml`. If `OUTPUT` is a directory, the file lands inside it.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--force` | Overwrite if the file exists | off |

```bash
nanoplm pretrain get-yaml
```

```
Template written to: <cwd>/pretrain.yaml
```

### pretrain from-yaml

```
nanoplm pretrain from-yaml [CONFIG] [--pure-torch]
```

Runs pretraining from a config with `model:`, `pretraining:`, and `resume:` blocks. `CONFIG` defaults to `pretrain.yaml`.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--pure-torch` | Force the pure-torch model and training loop | off (YAML `pure_torch: true` is the template default) |

The stock template sets top-level `pure_torch: true`, so a normal `from-yaml` run needs no flag. Pure-torch is the only pretraining backend; a config with neither the YAML key nor the flag stops with `Pretraining requires pure_torch: true.`

```bash
nanoplm pretrain from-yaml
```

```
# ...
Effective world size: 1 | grad_accum: 8 | achieved global batch tokens: 262144
step 1 | loss 3.0412 | lr 1.00e-07
step 2 | loss 3.0388 | lr 2.00e-07
# ...
```

Multi-GPU runs go through `torchrun`, handing it the console-script path:

```bash
torchrun --nproc-per-node=4 $(which nanoplm) pretrain from-yaml
```

### pretrain run

```
nanoplm pretrain run --dataset-dir DIR [OPTIONS] --pure-torch
```

Flag-driven pretraining without a YAML file. `--dataset-dir` is the only required option; `--pure-torch` **is** mandatory here because there is no YAML default to fall back on. Every flag maps to a `pretrain.yaml` key of the same name, so the [pretrain.yaml reference](/nanoplm/reference/pretrain/) is the authoritative source for meanings and defaults.

- **Dataset and output**: `--dataset-dir`, `--ckp-dir`.
- **Batch and duration**: `--micro-batch-seqs`, `--global-batch-tokens`, `--global-batch-warmup-steps`, `--train-tokens`, `--max-steps`, `--stable-tokens`, `--stable-steps`, `--decay-tokens`, `--decay-steps`.
- **LR and optimizer**: `--lr-schedule`, `--warmup-steps`, `--decay-shape`, `--lr-decay-to-fraction`, `--optimizer`, the `--adam-*` and `--muon-*` families, `--muonh-normalization`, `--gradient-clipping/--no-gradient-clipping`.
- **MLM**: `--mlm-probability`, `--mlm-masking-strategy`, `--mlm-span-mean-length`, `--mlm-span-max-length`, `--mask-replace-prob`, `--random-token-prob`, `--keep-probability`, `--mlm-standard-amino-acids-only/--mlm-all-non-special-tokens`.
- **Precision**: `--bf16`, `--tf32`, `--fp8`, `--fp8-target`, `--fp8-recipe`, `--fp8-format`, `--fp8-amax-history-len`, `--fp8-reduce-amax`, `--fp8-skip-last-n-layers`.
- **Distributed**: `--multi-gpu`, `--distributed-mode [fsdp|ddp]`, `--fsdp-shard-granularity`, `--fsdp-reshard-after-forward`, `--world-size`, `--ddp-bucket-cap-mb`.
- **Model**: `--hidden-size`, `--intermediate-size`, `--num-hidden-layers`, `--num-attention-heads`, `--num-kv-heads`, `--norm-type`, `--mlp-activation`, `--classifier-activation`, `--wo-init`, `--fused-qkv`, `--fused-up-gate`, `--attn-layer-pattern`, `--activation-checkpointing`, and the pure-torch feature toggles (`--use-canon-layers`, `--use-repo`, `--use-mhc-lite`, `--use-diff-attn-v2`, `--use-paired-head-attention`, `--use-qk-norm`, `--use-resid-lambdas`, `--use-x0-lambdas`, `--use-prores`).
- **Resume**: `--resume/--no-resume`, `--resume-checkpoint-dir`, `--resume-mode`, `--resume-decay-tokens`, `--resume-decay-steps`, `--resume-decay-shape`, `--resume-lr-decay-to-fraction`.
- **Logging and debug**: `--logging-steps`, `--eval-steps`, `--save-steps`, `--seed`, `--project-name`, `--run-name`, `--debug-non-finite-params`, `--debug-layerwise-metrics`, `--debug-layerwise-log-every`.

**What `run` cannot reach.** There are no flags for `hsdp_shard_size` (and `--distributed-mode` offers only `fsdp` and `ddp`), any `moe_*` key, any `noble_*` key, any `recycling_*` key, any `profiler_*` key, `global_rope_theta` / `local_rope_theta` / `rope_dim_fraction`, `muonh_scope`, `mhc_triton_fused`, or the `fp8_cpu_activation_offload*` keys. Those are YAML-only, and `run` fills them with defaults without warning. Use `from-yaml` if you need any of them.

---

## eval

`nanoplm eval` scores a checkpoint on zero-shot and supervised probe benchmarks through biotrainer 2.1.1's autoeval. It needs the `eval` extra (`uv pip install -e ".[eval]"` or the GitHub equivalent; see [Install](/nanoplm/start/install/)), and it loads **pure-torch checkpoints only**. Fields are documented in the [eval.yaml reference](/nanoplm/reference/eval/). The old probe suites (`pbc`, `flip`) were dropped in the biotrainer 2.0.0 migration; passing them fails with a message naming the supported frameworks.

### eval get-yaml

```
nanoplm eval get-yaml [OUTPUT] [OPTIONS]
```

Writes `./eval.yaml`.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--force` | Overwrite if the file exists | off |

### eval from-yaml

```
nanoplm eval from-yaml [CONFIG]
```

Runs evaluation from a config with a single `eval:` block. `CONFIG` defaults to `eval.yaml`. No other options.

```bash
nanoplm eval from-yaml
```

```
# ...
Resolved checkpoint: output/pretraining_checkpoints/run-14071932/checkpoint-5000
# ...
Results written to autoeval_output/autoeval_report_checkpoint-5000.json
```

### eval run

```
nanoplm eval run --checkpoint-dir DIR [OPTIONS]
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--checkpoint-dir TEXT` | **Required.** A `checkpoint-*` directory, or a `run-*` directory (the newest checkpoint inside is used) | - |
| `--framework [pbc_supervised\|pbc_supervised_contact\|pbc_zeroshot_contact\|pgym]` | Benchmark suite; repeat the flag to run several | `pgym` |
| `--pbc-supervised-dataset TEXT` | Dataset(s) for `pbc_supervised`; repeat the flag for several, or pass `all` | `secondary_structure`, `scl` |
| `--zero-shot-method [masked_marginals\|wt_marginals\|pseudoperplexity]` | Scoring method, used only when `--framework` includes `pgym` | `masked_marginals` |
| `--output-dir TEXT` | Where results are written | `autoeval_output` |
| `--zero-shot-batch-size INTEGER` | Sequences per forward pass; batches masked positions and Jacobian rows | `60` |
| `--device TEXT` | Device override | auto-detected |
| `--embedder-name TEXT` | Name used in the report and output directory | checkpoint directory name |
| `--development-mode / --full-eval` | Subsampled task sets, or the full benchmark | `--development-mode` |
| `--dataset-dir TEXT` | Pre-downloaded benchmark datasets, for offline machines. Mutually exclusive with `--force-download` | none |
| `--force-download` | Re-download benchmark datasets | off |
| `--loop-count INTEGER` | Refinement depth to run a looped (loopie or huginn) checkpoint at, overriding the depth it was trained with | the trained depth |
| `--torch-compile / --no-torch-compile` | Run the model through `torch.compile` with dynamic shapes (roughly 2x faster than eager) | `--torch-compile` |

---

## Environment variables

| Variable | Effect |
| --- | --- |
| `NANOPLM_MAX_WORKERS` | Default for `data pre-tokenize --max-workers` (otherwise `2`). `data from-yaml` exports it from `data_params.max_workers`. |
| `NANOPLM_DEVICE` | Default device for the distillation `kd-embed` stage (otherwise `auto`). `data from-yaml` exports it from `data_params.device`. |
| `NANOPLM_MMSEQS` | Path to the `mmseqs` binary for [cluster subsampling](/nanoplm/guides/cluster-subsampling/#when-mmseqs2-runs). Otherwise nanoPLM looks on `PATH`, then in the venv's `bin/`. |
| `NANOPLM_DISABLE_NATIVE_SHARD_WRITER` | Set to `1` to fall back to the Python shard writer instead of the native C one. |
| `NANOPLM_DISABLE_NATIVE_FASTA_OPS` | Set to `1` to fall back to the Python FASTA shuffle/filter/split/validate path. |
| `SLURM_JOB_ID` | When set, it replaces the `DDMMHHMM` timestamp in the run directory name, making the run directory `<run_name>-<SLURM_JOB_ID>`. |

---

## Unmaintained: distill, kd-embed, kd-reshard

:::caution[Not maintained]
These commands belong to the knowledge-distillation pipeline, which is **unmaintained** and has already drifted out of step with the rest of the tool. Do not use them for new work. They are listed here only so the `--help` output is not a mystery. See [Distillation (unmaintained)](/nanoplm/distillation/).
:::

- `nanoplm data get-yaml --distillation`: writes `distillation_data.yaml` instead of `data.yaml`.
- `nanoplm data kd-embed`: computes ProtT5 teacher embeddings into HDF5 shards. Options: `-i`, `-o`, `--teacher-model` (`prott5`), `--max-seq-len` (`1024`), `--samples-per-shard` (`-1`, meaning one file), `--batch-size` (`4`), `--device` (`$NANOPLM_DEVICE`, else `auto`), `--skip-n` (`0`), `--force`.
- `nanoplm data kd-reshard`: splits one large `.h5` into several. Options: `--input-file`/`-i`, `--n-shards`/`-n` (both required), `--output-dir`/`-o`, `--total-sequences`/`-t`.
- `nanoplm distill get-yaml | from-yaml | run`: trains a student against the teacher embeddings.

To train a model, use [`nanoplm pretrain`](/nanoplm/pipeline/pretrain/) instead.
