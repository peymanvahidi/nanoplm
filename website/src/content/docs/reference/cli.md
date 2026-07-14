---
title: CLI
description: Every nanoplm command, argument, and option, grouped by the four command families — data, pretrain, eval, and the unmaintained distill.
---

`nanoplm` has four command groups: `data`, `pretrain`, `eval`, and `distill`. Every group has the same shape:

```
nanoplm <group> get-yaml | from-yaml | run
```

- **`get-yaml`** scaffolds a config file (`data.yaml`, `pretrain.yaml`, `eval.yaml`).
- **`from-yaml`** runs that config. This is the supported path.
- **`run`** takes the same knobs as command-line flags instead of a file.

For `pretrain` and `eval`, **`from-yaml` is the path to use.** `pretrain run` works, but it exposes a strictly smaller surface than the YAML: it has no flags for HSDP, MoE, NOBLE, recycling, the profiler, RoPE thetas, or FP8 CPU offload, and it silently defaults every one of them. Reach for it only for a quick one-off.

The root group takes `--version`. Every command takes `-h` / `--help`.

---

## data

`nanoplm data` builds the binary shards and the `.data_manifest` that pretraining consumes. Its `--help` is grouped, and that grouping is the right mental model:

- **Main commands** — `get-yaml`, `from-yaml`. Most users only ever need these two.
- **Individual pipeline stages** — `get-file`, `extract`, `check`, `shuffle`, `filter`, `split`, plus the pretraining-specific `pre-tokenize`. `from-yaml` orchestrates all of them through DVC. You can also run any of them standalone to debug one step.

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
Wrote data.yaml and dvc.yaml in /work/my-plm.
Edit data.yaml and run `nanoplm data from-yaml` to execute.
```

### data from-yaml

```
nanoplm data from-yaml [CONFIG] [OPTIONS]
```

Regenerates `dvc.yaml` from the config, runs `dvc init` if needed, then executes the pipeline with `dvc repro`. `CONFIG` defaults to `data.yaml` in the current directory (falling back to `distillation_data.yaml`).

| Flag | Meaning | Default |
| --- | --- | --- |
| `--target TEXT` | Run one DVC stage (for example `split`) instead of the whole pipeline | full pipeline |
| `--no-auto-init` | Fail instead of running `dvc init` when DVC is not initialized | off |
| `--force` | Pass `-f` to `dvc repro`, re-running stages DVC considers unchanged | off |
| `--verbose` | Pass `-v` to `dvc repro` | off |
| `--fix` | Rewrite `data_params.drop_invalid_sequences: true` in the config, then run | off |
| `--dry-run` | Print the DVC plan without executing it | off |
| `--log-level [debug\|info\|warning\|error]` | nanoplm log level | `info` |

```bash
nanoplm data from-yaml
```

```
Running dvc repro (full pipeline)...
# ... dvc stage output ...

Pretraining data pipeline complete.

  Source:          https://ftp.uniprot.org/pub/.../uniprot_sprot.fasta.gz
  Total sequences: 20000
    Train:         18000
    Val:           2000
  Manifest:        output/data/pretrain_data/.data_manifest
```

`--fix` is the escape hatch when the `check` stage rejects invalid sequences — see [Troubleshooting](/nanoplm/troubleshooting/).

### data get-file

```
nanoplm data get-file [DATASET] [OPTIONS]
```

Acquire a FASTA source. Give **exactly one** of a `DATASET` key (`swissprot`, `trembl`, `uniref50`, `uniref90`, `uniref100`), `--url`, or `--local-fasta`.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--url TEXT` | URL to download | `""` |
| `--local-fasta TEXT` | Local `.fasta` / `.fasta.gz` to link instead of downloading | `""` |
| `--output`, `-o PATH` | **Required.** Output file or directory | — |
| `--force`, `-f` | Overwrite existing output | off |

### data extract

```
nanoplm data extract [OPTIONS]
```

Decompress a `.gz` source. This stage is skipped entirely when the source is already a plain `.fasta`.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Compressed input | — |
| `--output`, `-o PATH` | **Required.** Output file or directory | — |
| `--force`, `-f` | Overwrite existing output | off |

### data check

```
nanoplm data check [OPTIONS]
```

Validate the FASTA. Writes `validation.json`, `invalid.fasta`, and `invalid_ids.txt` into the output directory. With `--drop-invalid false` it **exits non-zero** if any sequence is invalid, halting the pipeline.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Input FASTA | — |
| `--output`, `-o PATH` | **Required.** Output directory for the validation sidecars | — |
| `--drop-invalid [true\|false]` | Drop invalid sequences and exit 0 | `false` |

### data shuffle

```
nanoplm data shuffle [OPTIONS]
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Input FASTA | — |
| `--output`, `-o PATH` | **Required.** Output FASTA | — |
| `--backend [auto\|seqkit\|fast\|biopython]` | Shuffle backend; `auto` picks seqkit when installed, else the native `fast` backend | `auto` |
| `--seed INTEGER` | Random seed | none |
| `--shuffle [true\|false]` | `false` links the input to the output unchanged | `true` |

### data filter

```
nanoplm data filter [OPTIONS]
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Input FASTA | — |
| `--output`, `-o PATH` | **Required.** Output FASTA | — |
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
| `--input`, `-i PATH` | **Required.** Filtered FASTA | — |
| `--output`, `-o PATH` | **Required.** Output directory | — |
| `--val-ratio FLOAT` | Fraction held out for validation | `0.1` |

### data pre-tokenize

```
nanoplm data pre-tokenize [OPTIONS]
```

Pretraining-specific. Tokenizes a FASTA into the binary shards (`shard_NNNN.bin` + `shard_NNNN.idx.npy`) that pretraining memory-maps.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--input`, `-i PATH` | **Required.** Input FASTA | — |
| `--output`, `-o PATH` | **Required.** Output directory for the shards | — |
| `--max-seq-len INTEGER` | Pad/truncate length. Recorded in the manifest, and the value pretraining reads back for its token math | `1024` |
| `--max-workers INTEGER` | Parallel workers | `$NANOPLM_MAX_WORKERS`, else `2` |
| `--samples-per-shard INTEGER` | Sequences per shard | `10000` |
| `--register-tokens INTEGER` | Learnable scratch tokens (`0`, `1`, or `2`) appended after EOS | `0` |
| `--force`, `-f` | Overwrite existing output | off |

`nanoplm data write-manifest` also exists but is hidden: the generated `dvc.yaml` calls it to write the `.data_manifest`. You do not invoke it yourself.

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
Template written to: /work/my-plm/pretrain.yaml
```

### pretrain from-yaml

```
nanoplm pretrain from-yaml [CONFIG] --pure-torch | --pure-te
```

Runs pretraining from a config with `model:`, `pretraining:`, and `resume:` blocks. `CONFIG` defaults to `pretrain.yaml`.

| Flag | Meaning | Default |
| --- | --- | --- |
| `--pure-torch` | Use the pure-torch model and training loop | off |
| `--pure-te` | Use the Transformer Engine model and training loop | off |

:::caution[A backend flag is required]
Pretraining fails immediately with `Pretraining requires pure_torch: true or pure_te: true.` if you pass neither. There is no third backend. Use `--pure-torch` unless you specifically need Transformer Engine — see [Choose a backend](/nanoplm/guides/backends/).
:::

You may instead set the top-level `pure_torch: true` (or `pure_te: true`) key in the YAML; the CLI flag wins when both are present. The two are mutually exclusive.

```bash
nanoplm pretrain from-yaml --pure-torch
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
torchrun --nproc-per-node=4 $(which nanoplm) pretrain from-yaml --pure-torch
```

### pretrain run

```
nanoplm pretrain run --dataset-dir DIR [OPTIONS] --pure-torch | --pure-te
```

Flag-driven pretraining without a YAML file. `--dataset-dir` is the only required option; a backend flag is still mandatory. Every flag maps to a `pretrain.yaml` key of the same name, so the [pretrain.yaml reference](/nanoplm/reference/pretrain/) is the authoritative source for meanings and defaults.

- **Dataset and output** — `--dataset-dir`, `--ckp-dir`.
- **Batch and duration** — `--micro-batch-seqs`, `--global-batch-tokens`, `--global-batch-warmup-steps`, `--train-tokens`, `--max-steps`, `--stable-tokens`, `--stable-steps`, `--decay-tokens`, `--decay-steps`.
- **LR and optimizer** — `--lr-schedule`, `--warmup-steps`, `--decay-shape`, `--lr-decay-to-fraction`, `--optimizer`, the `--adam-*` and `--muon-*` families, `--muonh-normalization`, `--gradient-clipping/--no-gradient-clipping`.
- **MLM** — `--mlm-probability`, `--mlm-masking-strategy`, `--mlm-span-mean-length`, `--mlm-span-max-length`, `--mask-replace-prob`, `--random-token-prob`, `--keep-probability`, `--mlm-standard-amino-acids-only/--mlm-all-non-special-tokens`.
- **Precision** — `--bf16`, `--tf32`, `--fp8`, `--fp8-target`, `--fp8-recipe`, `--fp8-format`, `--fp8-amax-history-len`, `--fp8-reduce-amax`, `--fp8-skip-last-n-layers`.
- **Distributed** — `--multi-gpu`, `--distributed-mode [fsdp|ddp]`, `--fsdp-shard-granularity`, `--fsdp-reshard-after-forward`, `--world-size`, `--ddp-bucket-cap-mb`.
- **Model** — `--hidden-size`, `--intermediate-size`, `--num-hidden-layers`, `--num-attention-heads`, `--num-kv-heads`, `--norm-type`, `--mlp-activation`, `--classifier-activation`, `--wo-init`, `--fused-qkv`, `--fused-up-gate`, `--attn-layer-pattern`, `--activation-checkpointing`, and the pure-torch feature toggles (`--use-canon-layers`, `--use-repo`, `--use-mhc-lite`, `--use-diff-attn-v2`, `--use-paired-head-attention`, `--use-qk-norm`, `--use-resid-lambdas`, `--use-x0-lambdas`, `--use-prores`).
- **Resume** — `--resume/--no-resume`, `--resume-checkpoint-dir`, `--resume-mode`, `--resume-decay-tokens`, `--resume-decay-steps`, `--resume-decay-shape`, `--resume-lr-decay-to-fraction`.
- **Logging and debug** — `--logging-steps`, `--eval-steps`, `--save-steps`, `--seed`, `--project-name`, `--run-name`, `--debug-non-finite-params`, `--debug-layerwise-metrics`, `--debug-layerwise-log-every`.

**What `run` cannot reach.** There are no flags for `hsdp_shard_size` (and `--distributed-mode` offers only `fsdp` and `ddp`), any `moe_*` key, any `noble_*` key, any `recycling_*` key, any `profiler_*` key, `global_rope_theta` / `local_rope_theta` / `rope_dim_fraction`, `muonh_scope`, `mhc_triton_fused`, or the `fp8_cpu_activation_offload*` keys. Those are YAML-only, and `run` fills them with defaults without warning. Use `from-yaml` if you need any of them.

---

## eval

`nanoplm eval` scores a checkpoint on downstream benchmarks through biotrainer's autoeval. It needs `pip install nanoplm[eval]`, and it loads **pure-torch checkpoints only**. Fields are documented in the [eval.yaml reference](/nanoplm/reference/eval/).

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
Results written to autoeval_output/checkpoint-5000/
```

### eval run

```
nanoplm eval run --checkpoint-dir DIR [OPTIONS]
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--checkpoint-dir TEXT` | **Required.** A `checkpoint-*` directory, or a `run-*` directory (the newest checkpoint inside is used) | — |
| `--framework [pbc\|flip\|pgym]` | Benchmark suite; repeat the flag to run several | `pbc` |
| `--zero-shot-method [masked_marginals\|wt_marginals\|pseudoperplexity]` | Scoring method, used only when `--framework` includes `pgym` | `masked_marginals` |
| `--output-dir TEXT` | Where results are written | `autoeval_output` |
| `--batch-size INTEGER` | Embedding batch size | `16` |
| `--min-seq-length INTEGER` | Benchmark sequence-length filter | `0` |
| `--max-seq-length INTEGER` | Benchmark sequence-length filter | `2000` |
| `--device TEXT` | Device override | auto-detected |
| `--embedder-name TEXT` | Name used in the report and output directory | checkpoint directory name |
| `--parallel-tasks INTEGER` | Parallel probe workers; `0` runs sequentially | `0` |
| `--tasks TEXT` | Whitelist of task names; repeat the flag. Honored in parallel mode only | none |
| `--force-download` | Re-download benchmark datasets | off |

---

## Environment variables

| Variable | Effect |
| --- | --- |
| `NANOPLM_MAX_WORKERS` | Default for `data pre-tokenize --max-workers` (otherwise `2`). `data from-yaml` exports it from `data_params.max_workers`. |
| `NANOPLM_DEVICE` | Default device for the distillation `kd-embed` stage (otherwise `auto`). `data from-yaml` exports it from `data_params.device`. |
| `NANOPLM_DISABLE_NATIVE_SHARD_WRITER` | Set to `1` to fall back to the Python shard writer instead of the native C one. |
| `NANOPLM_DISABLE_NATIVE_FASTA_OPS` | Set to `1` to fall back to the Python FASTA shuffle/filter/split/validate path. |
| `SLURM_JOB_ID` | When set, it replaces the `DDMMHHMM` timestamp in the run directory name, making the run directory `<run_name>-<SLURM_JOB_ID>`. |

---

## Unmaintained: distill, kd-embed, kd-reshard

:::caution[Not maintained]
These commands belong to the knowledge-distillation pipeline, which is **unmaintained** and has already drifted out of step with the rest of the tool. Do not use them for new work. They are listed here only so the `--help` output is not a mystery. See [Distillation (unmaintained)](/nanoplm/distillation/).
:::

- `nanoplm data get-yaml --distillation` — writes `distillation_data.yaml` instead of `data.yaml`.
- `nanoplm data kd-embed` — computes ProtT5 teacher embeddings into HDF5 shards. Options: `-i`, `-o`, `--teacher-model` (`prott5`), `--max-seq-len` (`1024`), `--samples-per-shard` (`-1`, meaning one file), `--batch-size` (`4`), `--device` (`$NANOPLM_DEVICE`, else `auto`), `--skip-n` (`0`), `--force`.
- `nanoplm data kd-reshard` — splits one large `.h5` into several. Options: `--input-file`/`-i`, `--n-shards`/`-n` (both required), `--output-dir`/`-o`, `--total-sequences`/`-t`.
- `nanoplm distill get-yaml | from-yaml | run` — trains a student against the teacher embeddings.

To train a model, use [`nanoplm pretrain`](/nanoplm/pipeline/pretrain/) instead.
