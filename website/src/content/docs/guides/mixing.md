---
title: Mixing datasets
description: Train on several FASTA sources at once, each with a weight that is its share of the training sequences. One data.yaml, one dataset_dir.
---

Put two or more entries under `sources` in `data.yaml` and give each a `weight`. The weight is the source's **share of training sequences** (not tokens). The weights must sum to 1.

```yaml title="data.yaml"
output_dir: "output/data/pretrain_data"
sources:
  swissprot:
    fasta: swissprot
    weight: 0.2
  uniref50:
    fasta: uniref50
    weight: 0.8
    cluster_subsampling:
      members: uniref90
```

`nanoplm data from-yaml` builds each source into `output_dir/<source>/` and then writes the mixture manifest to `output_dir/.data_manifest`. Point `pretrain.yaml`'s `dataset_dir` at `output_dir` as usual.

## Rules

- Weights are above 0 and sum to 1 (within 1e-6). The error lists each weight and the sum.
- All sources use the same `register_tokens`. `max_workers` is set once, in `data_params`.
- A source can be `prepared: <folder>`, a dataset another `data.yaml` built. It is not rebuilt.
- Pretraining needs `use_packing: true` (the template default).
- The model's position check uses the largest `max_seq_len` of the sources.

## What reruns

There is one `dvc.yaml`. Each source has its own stages (`swissprot-filter`, `uniref50-cluster_index`, ...). A URL that several sources use is downloaded once, and a `.gz` that several sources read is extracted and checked once, into `output_dir/downloads/` (stages `download-<file>`, `extract-<file>`, `check-<file>`). A source that sets its own `drop_invalid_sequences` keeps its own check. The last stage, `mixture`, is the only one that reads the weights, so changing only the weights reruns only that stage, in seconds.

Going from one source to several (or back) moves where the dataset lives: one source writes `train/` and `val/` into `output_dir`, a mix writes them into `output_dir/<source>/`. The old outputs are not deleted. `from-yaml` prints a warning listing them, and you can delete them.

The mixture manifest records a hash of each source's manifest. If a source is rebuilt or edited after the mix was written, training stops and asks you to rerun `nanoplm data from-yaml`.

## How training draws

Each source keeps its own epoch counter. A small source with a large weight repeats; a large source with a small weight is read partly. A cluster-subsampled source draws new members every time it repeats.

One mixture epoch is `N` samples, the sum of the source sizes (sequences, or clusters for a cluster source). Each source gets `weight × N` of them. The shares are exact over each mixture epoch.

At the start, training prints one row per source: share, size, tokens used and epochs over the run's token budget. It warns when a source would be read more than 10 times. The same table goes into the W&B config as `data_mixture`.

## Eval

Eval keeps today's batch count and splits it evenly over the sources. It logs `eval/loss/<source>` for each source and `eval/loss`, the share-weighted mean of those. With a recycling eval sweep, each loop count adds `eval/loss/<source>/r<k>`, and the unsuffixed keys are those of the target loop count; when the sweep leaves that out, they are not logged (as for a single dataset).

Each source has its own val loader; `num_workers` is split between them, so an eval starts about as many worker processes as with one dataset.

To evaluate only some sources, set `eval_sources` in `pretrain.yaml`; they then get the whole eval budget:

```yaml title="pretrain.yaml"
pretraining:
  eval_sources: [swissprot]   # default: all
```
