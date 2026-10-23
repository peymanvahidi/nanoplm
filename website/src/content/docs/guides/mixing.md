---
title: Mixing datasets
description: Train on several FASTA sources at once. Each source has a weight, its share of the training sequences. One data.yaml, one dataset_dir.
---

A mix is a `data.yaml` with two or more entries under `sources`. Each entry has a `weight`: the share of training sequences drawn from that source. With `weight: 0.2`, one training sequence in five comes from that source, however long its sequences are. Weights are above 0 and sum to 1.

The outputs on this page come from a tiny mix: `small.fasta` with 300 sequences and `big.fasta` with 3,000.

## Write a mix

```yaml title="data.yaml"
output_dir: "output/data/pretrain_data"
sources:
  small:
    fasta: "small.fasta"
    weight: 0.2
  big:
    fasta: "big.fasta"
    weight: 0.8
data_params:            # defaults for every source; a source may override any of them
  max_seqs: -1
  min_seq_len: 20
  max_seq_len: 512
  val_ratio: 0.05
```

For a third source, add an entry and lower the other weights so they still sum to 1:

```yaml title="data.yaml"
sources:
  small:
    fasta: "small.fasta"
    weight: 0.2
  big:
    fasta: "big.fasta"
    weight: 0.7
  extra:
    fasta: "extra.fasta"
    weight: 0.1
```

Weights that do not sum to 1 stop the run before anything is built:

```text
Error: data.yaml: the weights must sum to 1 (they are shares of training sequences): 0.2 (small) + 0.7 (big) = 0.9
```

`nanoplm data from-yaml data.yaml` builds each source into its own folder, `output_dir/<source>/`, then writes the mixture manifest to `output_dir/.data_manifest`:

```text title="Output (trimmed)"
data.yaml: 2 sources -> <cwd>/output/data/pretrain_data
weight = share of training sequences
source  weight  fasta                         max seqs  len     val   clustering
small   0.2     small.fasta (local, 37.0 kB)  all       20-512  0.05  off
big     0.8     big.fasta (local, 662.8 kB)   all       20-512  0.05  off
...
Pretraining data mixture complete (2 sources; train and val count clusters for a cluster source).
  source  weight  kind   train  val
  small   0.2     plain  285    15
  big     0.8     plain  2,850  150
  Manifest:  <cwd>/output/data/pretrain_data/.data_manifest
  Set pretrain.yaml's dataset_dir to <cwd>/output/data/pretrain_data.
```

Point `pretrain.yaml`'s `dataset_dir` at `output_dir`, as for one source. A few rules apply to every mix:

- All sources use the same `register_tokens`, and `max_workers` is set once, in `data_params`.
- Pretraining needs `use_packing: true`, the template default. Without it the run stops with `... is a mixture, which needs use_packing=true.`
- The model's position limit is checked against the largest `max_seq_len` of the sources.

### A real mix

SwissProt plus UniRef50 with [cluster subsampling](/nanoplm/guides/cluster-subsampling/):

```yaml title="data.yaml"
output_dir: "output/data/pretrain_data"
sources:
  swissprot:
    fasta: swissprot
    weight: 0.2
  uniref50:
    fasta: uniref50
    weight: 0.8
    max_seqs: -1         # counts clusters here: all of them
    val_ratio: 0.0044    # share of clusters held out, as in the UniRef90 experiment
    cluster_subsampling:
      members: uniref90
```

SwissProt sequences are also UniRef90 members, so these two sources overlap. A sequence in one source's val set can be a training sequence of the other, and nothing removes it. Keep that in mind when you compare `eval/loss/swissprot` and `eval/loss/uniref50`.

## Epochs per source

Training runs for a token budget, not a number of epochs, so each source is read as many times as its weight and size work out to. At the start, training prints one row per source:

```text title="Training start"
Data mixture over a budget of 3,014,656 tokens (share = share of training sequences):
  source  share        size  tokens used  epochs
  small   0.200    285 seqs      361,851   10.85
  big     0.800  2,850 seqs    2,652,805    4.34
Warning: sources.small: read 10.8 times over the run (more than 10x); lower sources.small.weight or add data
```

The numbers come from three steps. The budget divided by the mean sequence length gives the number of sequences the run draws. Each source supplies its `share` of them. That count divided by the source's size is its `epochs`. Here `small` holds 9% of the train sequences but gets 20% of the draws, so it is read 10.85 times, and the run warns because that is more than 10. The counts ignore packing waste, so read them as estimates. A cluster source counts clusters, not sequences, in `size`.

The `share` column is the weight rounded to whole sequences. The same table is stored in the W&B config as `data_mixture`.

A source with `epochs` below 1 trains on a part of itself, spread over the whole source. Each time a source is read again, it is read in a new order, and a cluster source draws new members. A `decay` resume starts again from the mix's first epoch, as described in [Cooldown and resuming](/nanoplm/guides/cooldown-and-resume/).

## Validation

Each source keeps its own val set, cut by that source's `val_ratio`. Eval uses the number of batches one val set holding all of them would take, and splits it evenly over the sources. In the tiny mix:

- 15 + 150 = 165 val sequences at `micro_batch_seqs: 4` make 42 batches, so each source gets 21 batches, 84 sequences.
- `small` has only 15, so it evaluates all 15. Its unused part does not go to `big`.
- `big` evaluates 84 of its 150, picked evenly over its whole val set.

Training logs the split at the start:

```text
Mixture eval: small 15 of 15 val seqs, big 84 of 150 val seqs
```

Each eval logs these keys:

| Key | Value |
| --- | --- |
| `eval/loss/<source>` | The loss on that source's val sequences |
| `eval/loss` | The mean of the `eval/loss/<source>` values, weighted by `weight`. Here 0.2 × small + 0.8 × big |

With a [recycling](/nanoplm/guides/recycling/) eval sweep, each loop count `k` adds `eval/loss/<source>/r<k>`.

To evaluate only some sources, list them in `pretrain.yaml`:

```yaml title="pretrain.yaml"
pretraining:
  eval_sources: [small]   # default: all
```

The listed sources share the same batch count, so one listed source gets all of it. `eval/loss` then averages over the listed sources only. With one source, it equals that source's key. A source with no val sequences is skipped under `all` and is an error when you list it. An unknown name, or a list on a dataset that is not a mix, is an error too.

## Sweep ratios cheaply

The last stage of the pipeline, `mixture`, is the only one that reads the weights. Change only the weights and rerun `from-yaml`, and no source is rebuilt:

```text
Rebuild: small: up to date; big: up to date; mixture: may rerun. (DVC skips a stage its run cache already has.)
```

That rewrites the one mix in `output_dir`. To keep several ratios side by side, give each ratio its own folder with its own `data.yaml`, and point every source at a dataset you built already with `prepared:`. Say `base/` holds the `data.yaml` above and its built output, and `sweep_50_50/` sits next to it:

```yaml title="sweep_50_50/data.yaml"
output_dir: "output/mix"
sources:
  small:
    prepared: "../base/output/data/pretrain_data/small"
    weight: 0.5
  big:
    prepared: "../base/output/data/pretrain_data/big"
    weight: 0.5
```

`nanoplm data from-yaml sweep_50_50/data.yaml` writes its `dvc.yaml` next to that `data.yaml`, so it does not touch `base/`. It runs only the `mixture` stage. `sweep_50_50/output/mix/` then holds only the mixture manifest, about 1 kB, and `dataset_dir` points at it. A few things to know:

- `prepared:` names a built dataset: a mix's `<output_dir>/<source>`, or the `output_dir` of a single-source `data.yaml`. A mix's own `output_dir` is refused, with `is a mixture; list its sources here instead`.
- A prepared source is used as it was built. You cannot set `data_params` on it.
- If a source is rebuilt in the original folder, every mix that uses it is stale. Training stops and says the source's manifest `changed after the mixture was written`. Rerun `from-yaml` in each sweep folder.

## When sources change

Adding a source builds only that source and the `mixture` stage:

```text
Rebuild: small: up to date; big: up to date; extra: 7 of 7 stages may rerun; mixture: may rerun. (DVC skips a stage its run cache already has.)
```

Removing or renaming a source leaves its files on disk. `from-yaml` lists them and deletes nothing:

```text
Warning: an earlier build left these outputs, which the new dvc.yaml no longer uses; nothing deletes them:
  <cwd>/output/data/pretrain_data/extra/validation
  <cwd>/output/data/pretrain_data/extra/shuffled.fasta
  ...
  <cwd>/output/data/pretrain_data/extra/.data_manifest
and their lines in <cwd>/output/data/pretrain_data/extra/.gitignore
To free the disk, delete them and run `dvc gc --workspace` (it removes every cached file the current dvc.lock does not use).
```

The warning repeats on every run until the files are gone. The same warning lists the old files when you go from one source to a mix or back, since one source writes `train/` and `val/` into `output_dir` and a mix writes them into `output_dir/<source>/`. Deleting a cluster source's `cluster_subsampling` block leaves its `cluster_index/` the same way.

DVC's cache keeps a linked copy of every output, so deleting the files frees the disk only after a cache cleanup:

```sh
rm -rf output/data/pretrain_data/extra
dvc gc --workspace      # asks y/n first; -f skips the question
```

`dvc gc --workspace` removes every cached file the current `dvc.lock` does not use, older mixture manifests included. Before you delete a source's folder, check that no sweep folder still uses it as `prepared:`.
