---
title: Mixing datasets
description: Train on several FASTA sources at once, for example 20% SwissProt and 80% UniRef50. One data.yaml, one dataset_dir.
---

A mix trains on two or more FASTA files at once. You list each file under `sources` in `data.yaml` and give it a `weight`, its share of the training sequences. With `weight: 0.2`, one training sequence in five comes from that source, whatever the sequence lengths.

## Write a mix

1. List the sources in `data.yaml`. The weights must be above 0 and sum to 1.

   ```yaml title="data.yaml"
   output_dir: "output/data/pretrain_data"
   sources:
     swissprot:
       fasta: "swissprot_300.fasta"
       weight: 0.2
     uniref50:
       fasta: "uniref50_3000.fasta"
       weight: 0.8
   data_params:            # defaults for every source; a source may override them, but not val_ratio or shuffle
     max_seqs: -1
     min_seq_len: 20
     max_seq_len: 512
     val_ratio: 0.05
   ```

   The outputs on this page come from this tiny mix of two local files, 300 and 3,000 sequences. For the full datasets, write `fasta: swissprot` and `fasta: uniref50`.

2. Build it:

   ```sh
   nanoplm data from-yaml data.yaml
   ```

   ```text title="Output (trimmed)"
   data.yaml: 2 sources -> <cwd>/output/data/pretrain_data
   weight = share of training sequences
   source     weight  fasta                                  max seqs  len     val        cluster subsampling
   swissprot  0.2     swissprot_300.fasta (local, 37.0 kB)   all       20-512  by weight  off
   uniref50   0.8     uniref50_3000.fasta (local, 662.8 kB)  all       20-512  by weight  off
   val = data_params.val_ratio: 0.05 of the pool, shared out by weight
   ...
   Pretraining data mixture complete (2 sources; train and val count clusters for a cluster source).
     source     weight  kind   train  val
     swissprot  0.2     plain  267    33
     uniref50   0.8     plain  2,868  132
     Manifest:  <cwd>/output/data/pretrain_data/.data_manifest
     Set pretrain.yaml's dataset_dir to <cwd>/output/data/pretrain_data.
   ```

   Each source gets its own folder, `output_dir/<source>/`. The file `output_dir/.data_manifest` lists the sources and their weights.

3. Set `dataset_dir` in `pretrain.yaml` to `output_dir`, as for one source, and train.

If the weights do not sum to 1, `from-yaml` stops before it builds anything:

```text
Error: data.yaml: the weights must sum to 1 (they are shares of training sequences): 0.2 (swissprot) + 0.7 (uniref50) = 0.9
```

A mix also has these rules:

- Pretraining needs `use_packing: true`, the template default. Without it the run stops with `... is a mixture, which needs use_packing=true.`
- All sources use the same `register_tokens`. Set `max_workers` once, in `data_params`.
- The model's position limit must fit the largest `max_seq_len` of the sources.

### With cluster subsampling

This mix trains on one UniRef90 member of each UniRef50 cluster per epoch. See [cluster subsampling](/nanoplm/guides/cluster-subsampling/).

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
    cluster_subsampling:
      members: uniref90
      clusters: uniprot
data_params:
  val_ratio: 0.0044      # 147k val: 29k swissprot sequences + 118k uniref50 clusters
```

SwissProt sequences are also UniRef90 members, so the two sources overlap. A SwissProt validation sequence can be a UniRef50 training sequence, and nothing removes it. Keep that in mind when you compare `eval/loss/swissprot` and `eval/loss/uniref50`.

## How often each source is read

Training runs for a token budget, not a number of epochs. An epoch of a source is one full read of its training sequences. How many epochs each source gets follows from its weight and size. Training prints them at the start:

```text title="Training start"
Data mixture over a budget of 3,014,656 tokens (share = share of training sequences):
  source     share        size  tokens used  epochs
  swissprot  0.200    285 seqs      361,851   10.85
  uniref50   0.800  2,850 seqs    2,652,805    4.34
Warning: sources.swissprot: read 10.8 times over the run (more than 10x); lower sources.swissprot.weight or add data
```

SwissProt holds 9% of the training sequences but gets 20% of the draws, so the run reads it 10.85 times. Above 10 reads, training warns.

The numbers are estimates that ignore packing waste. For a cluster source, `size` counts clusters. W&B stores the table in the run config as `data_mixture`.

A source with `epochs` below 1 trains on a part of itself, spread over the whole source. Each new epoch of a source reads it in a new order, and a cluster source draws new members. A `decay` resume starts the mix again from its first epoch, as described in [Cooldown and resuming](/nanoplm/guides/cooldown-and-resume/).

## Validation

`data_params` sets the validation split of the whole mix once, with one of:

- `val_ratio`, a share of the pool. The pool adds up every source: its sequences for a plain source, its clusters for a [cluster source](/nanoplm/guides/cluster-subsampling/), after `max_seqs`.
- `val_size`, a number of validation sequences (clusters, for a cluster source).

The validation sequences are then shared out by `weight`. In the example above the pool is 300 + 3,000 sequences, so `val_ratio: 0.05` holds out 165: 33 from SwissProt (0.2) and 132 from UniRef50 (0.8). With the full SwissProt (570k sequences) and the UniRef50 clusters (32.9M), `val_ratio: 0.0044` holds out 147k: 29k SwissProt sequences and 118k UniRef50 clusters.

Setting `val_ratio` or `val_size` on a source is an error. So is setting both. Every source needs at least one validation sequence and keeps at least one training sequence. If a source is too small for its share, `from-yaml` stops:

```text
Error: sources.swissprot would need 669,400 val sequences but has 570,000; lower data_params.val_ratio or use data_params.val_size
```

Each source's share depends on the size of every source, so `from-yaml` counts each source after its filter (a `count` stage), then a `val_split` stage works out the shares. A source's split reruns only when its share changes. Changing a weight changes the shares, so it reruns `val_split`, the mixture and each source whose share moved. Nothing is downloaded or filtered again. A plain source splits and tokenizes again; a cluster source rebuilds only `cluster_index`, not its tokenized members. The `Rebuild:` line before the run cannot see this yet: it says the sources are up to date.

Each eval logs:

| Key | Value |
| --- | --- |
| `eval/loss/<source>` | The loss on that source's validation sequences |
| `eval/loss` | The mean of the source losses, weighted by `weight`: here 0.2 × swissprot + 0.8 × uniref50 |

With a [recycling](/nanoplm/guides/recycling/) eval sweep, each loop count `k` adds `eval/loss/<source>/r<k>`.

An eval runs as many batches as one combined validation set would, split evenly over the sources. Here 165 sequences at `micro_batch_seqs: 4` make 42 batches, so each source gets 21 batches, up to 84 sequences. SwissProt has only 33 and uses all of them. Its unused part does not go to UniRef50, which uses 84 of its 132, spread over the whole set. Training logs the split at the start:

```text
Mixture eval: swissprot 33 of 33 val seqs, uniref50 84 of 132 val seqs
```

To evaluate only some sources, list them in `pretrain.yaml`:

```yaml title="pretrain.yaml"
pretraining:
  eval_sources: [swissprot]   # default: all
```

The listed sources get all the eval batches, and `eval/loss` averages over them only. It is an error to list an unknown name, a source with no validation sequences, or any name when `dataset_dir` is not a mix. Under `all`, a source with no validation sequences is skipped.

## Compare several ratios

If you change only the weights and rerun `from-yaml`, it redraws the validation split (see [Validation](#validation)). A plain source then splits and tokenizes again; a cluster source keeps its tokenized members. The line before the run only shows the first stages:

```text
Rebuild: swissprot: up to date; uniref50: up to date; val_split: may rerun; mixture: may rerun. (DVC skips a stage its run cache already has.)
```

That replaces the mix in `output_dir`. To keep several ratios side by side, give each ratio its own folder with its own `data.yaml`; each one builds its sources again. To skip the downloads, set each `fasta` to the file the first build downloaded, for example `../base/output/data/pretrain_data/downloads/uniprot_sprot.fasta.gz`. A local file is read where it is. Each ratio shares the validation split out by its own weights, so the ratios hold out different sequences.

## Add, remove or rename a source

Adding a source builds only that source and the `mixture` stage:

```text
Rebuild: swissprot: up to date; uniref50: up to date; pdb: 7 of 7 stages may rerun; mixture: may rerun. (DVC skips a stage its run cache already has.)
```

Removing or renaming a source deletes nothing. `from-yaml` lists the files left behind on every run until they are gone:

```text
Warning: an earlier build left these outputs, which the new dvc.yaml no longer uses; nothing deletes them:
  <cwd>/output/data/pretrain_data/pdb/validation
  <cwd>/output/data/pretrain_data/pdb/shuffled.fasta
  ...
  <cwd>/output/data/pretrain_data/pdb/.data_manifest
and their lines in <cwd>/output/data/pretrain_data/pdb/.gitignore
To free the disk, delete them and run `dvc gc --workspace` (it removes every cached file the current dvc.lock does not use).
```

The same warning appears when you switch between one source and a mix, because one source writes `train/` and `val/` into `output_dir` and a mix writes them into `output_dir/<source>/`. Deleting a `cluster_subsampling` block leaves that source's `cluster_index/` behind too.

DVC keeps a linked copy of every output in its cache, so the disk is freed only after both steps:

```sh
rm -rf output/data/pretrain_data/pdb
dvc gc --workspace      # asks y/n first; -f skips the question
```

`dvc gc` also removes older mixture manifests.
