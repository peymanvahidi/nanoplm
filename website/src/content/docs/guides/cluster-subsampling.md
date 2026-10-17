---
title: Cluster subsampling
description: Train on one random member of every cluster per epoch instead of the cluster representatives. Turn it on in data.yaml with the cluster_subsampling block; UniProt or MMseqs2 clusters, or your own table.
---

A normal data run trains on a fixed set of sequences, for example the UniRef50 representatives. Cluster subsampling keeps the clusters but swaps the sequence: **each epoch, every cluster contributes one random member**. With UniRef90 members under UniRef50 clusters that is 32.8M samples per epoch (one per train cluster), drawn from a pool of 97.8M sequences, and a member seen in epoch 1 is usually not the one seen in epoch 2.

What changes against a normal run:

- **Samples per epoch = train clusters**, not sequences. The manifest's `train_sequences` counts clusters and `total_train_tokens` is the expected token count of one epoch.
- **The member draw is keyed on `pretrain.yaml`'s `seed`** and the epoch. It does not depend on the number of GPUs, dataloader workers or on resuming.
- **Validation is held out by whole cluster** and scores each held-out cluster's representative, the same sequences every epoch. Sibling sequences of a validation cluster never reach training.
- Pretraining needs `use_packing: true` (the template default).

## Turn it on

The block sits at the end of `data.yaml`:

```yaml title="data.yaml"
cluster_subsampling:     # each epoch, train on one random member of every cluster
  enabled: false
  members: null          # uniref90, uniref100, a FASTA URL/path, or null (= url itself)
  clusters: auto         # auto, or a path to your own cluster_id<TAB>sequence_id table
  identity: 0.5          # MMseqs2 clustering only: min sequence identity within a cluster
  coverage: 0.8          # MMseqs2 clustering only: min alignment coverage
```

- [`members`](/nanoplm/reference/data/#data-cluster_subsampling-members): the sequences training draws from. `uniref90` and `uniref100` are shortcuts for UniProt's `uniref90.fasta.gz` / `uniref100.fasta.gz`. A value starting with `http(s)://` is downloaded, anything else is a local path. `null` uses `data_dirs.url` (or `local_fasta`) itself.
- [`clusters`](/nanoplm/reference/data/#data-cluster_subsampling-clusters): `auto` picks the clusters as described below. A path is a tab-separated `cluster_id<TAB>sequence_id` table, one line per sequence; sequence ids are the first word of the FASTA header.
- [`identity`](/nanoplm/reference/data/#data-cluster_subsampling-identity) and [`coverage`](/nanoplm/reference/data/#data-cluster_subsampling-coverage) are only read when MMseqs2 does the clustering.

Then run `nanoplm data from-yaml` as usual and point `pretrain.yaml`'s `dataset_dir` at `output_dir`. Training picks the cluster-subsampled dataset from the manifest; nothing else changes in `pretrain.yaml`.

### Settings for a UniRef90 run

The template's `data_params` are sized for a quick SwissProt run. For UniRef-scale clusters change three of them:

| Key | Template | UniRef90 run | Why |
| --- | --- | --- | --- |
| `seqs_num` | `20000` | `-1` | In this mode it caps the number of **clusters**, not sequences. |
| `samples_per_shard` | `2000` | `30000` | 97.8M pool sequences at 2000 per shard would be 49k shard files: past the 10,000 shard files the pipeline supports, and past the open-file limit of most nodes. The pool stage refuses it. |
| `val_ratio` | `0.1` | `0.0044` | The fraction of **clusters** held out. 0.0044 holds out 145k clusters. |

`drop_invalid_sequences` works as in a normal run, on the member sequences. A cluster whose representative is dropped (invalid or outside `min_seq_len`..`max_seq_len`) leaves training with all its members.

## Scenarios

| # | Settings | What happens |
| --- | --- | --- |
| 1 | `enabled: false` | The normal pipeline, unchanged. |
| 2 | `url`: UniRef50, `members: uniref90` | UniRef90 members under UniProt's UniRef50 clusters (`uniref50.xml.gz`). 32.8M samples per epoch. The UniRef50 FASTA itself is never downloaded. |
| 3 | `url`: UniRef50, `members: uniref100` | UniRef100 members under UniProt's UniRef50 clusters. Same clusters, train and validation split as scenario 2. |
| 4 | your own FASTA as `url` / `local_fasta`, `enabled: true` | MMseqs2 clusters your sequences. |
| 5 | your own FASTA, `clusters: <table>` | Your table defines the clusters. No MMseqs2. |
| 6 | `url`: UniRef90 (or a UniRef90 `local_fasta`), `members: null` | The members are the url itself, under UniProt's UniRef50 clusters. No MMseqs2. |

## How `clusters: auto` decides

A FASTA counts as UniRef90 or UniRef100 by its headers (`>UniRef90_...`) when it is a local file, and by its file name (`uniref90.fasta.gz`) when it is a URL. The stages check the headers again once the file is on disk.

- **The members are a UniRef90 or UniRef100 set, and `url` is UniRef50 or that same set**: UniProt's UniRef50 clusters. `uniref50.xml.gz` is downloaded into the raw directory next to `compressed_fasta`; a copy already there is reused. Each UniRef50 entry is one cluster. For UniRef90 the representative is the entry's representative member. For UniRef100 it is the UniRef100 record with the same accession as the UniRef90 representative, which is the same sequence, so UniRef90 and UniRef100 runs share one set of clusters and one train / validation split.
- **Anything else** (your own FASTA, TrEMBL, UniRef50 itself): MMseqs2 `easy-linclust` clusters the filtered member sequences with `--min-seq-id identity -c coverage --cov-mode 0`. Coverage mode 0 asks the alignment to cover that fraction of **both** sequences, so a fragment does not join a cluster on a short local match. The representative is the one MMseqs2 picks. MMseqs2 runs in a temporary directory under `output_dir` that is deleted afterwards.
- **`clusters: <path>`**: the table is used as given. A cluster's representative is the member whose id equals the cluster id, else its first listed member. Sequences missing from the table become clusters of their own.

Whatever the source, it ends up as one table, `output_dir/cluster_index/members.tsv` (`cluster_id<TAB>sequence_id`, grouped by cluster, representative first), and the rest of the pipeline is built from that table.

## Errors and warnings

| Message | Cause |
| --- | --- |
| `cluster_subsampling.members is uniref90, so data_dirs.url must be UniRef50 or uniref90 itself` | UniRef members with a `url` (for example TrEMBL) whose clusters cannot be UniRef50. |
| `cluster_subsampling.members (...) is not a UniRef set and differs from data_dirs.url. Put that FASTA in data_dirs.url (or local_fasta) and set cluster_subsampling.members: null.` | Your own member FASTA goes in `url`, so there is only one source. |
| `... clustered with MMseqs2, but mmseqs was not found` | MMseqs2 is needed (scenario 4). Install it with `nanoplm deps install --mmseqs-only`, put a release binary on `PATH`, or set `NANOPLM_MMSEQS=/path/to/mmseqs`. |
| `... was built by the data-sampler experiment scripts ... Rebuild it with nanoplm data from-yaml` | A dataset directory from the old experiment (it has a `RUNTIME_SAMPLER_REQUIRED` marker, or a pool with the experiment's index next to it). Its manifest describes the whole pool, so it is refused. |
| Warning: `identity / coverage only apply to MMseqs2 clustering; they are ignored ...` | `identity` or `coverage` differs from the template default (0.5 / 0.8) while UniProt clusters or your own table are used. |
| Warning: `cluster_subsampling ignores data_params.shuffle / shuffle_seed` | The pool keeps FASTA order; the training sampler shuffles the clusters each epoch. |

## Stages and caching

`nanoplm data from-yaml` runs these as DVC stages:

```
get_members_fasta → extract_members → check → filter → cluster_pool
get_cluster_xml → cluster_members → cluster_index → write_manifest
```

`get_members_fasta` / `extract_members` only appear when `members` differs from `url`; otherwise the normal `get_file` / `extract` stages fetch the url. `get_cluster_xml` only appears for UniProt clusters. The `shuffle` and `split` stages are not used.

Each stage reruns only when its own inputs change. Changing `identity` or `coverage` reruns `cluster_members` and the stages after it, never the downloads or the tokenized pool. Changing `val_ratio` or `seqs_num` reruns `cluster_index` and the manifest. The XML is not copied into the DVC cache.

## Costs

Measured on UniProt release 2026_03, on one login node:

| Item | Size or time |
| --- | --- |
| Downloads | `uniref90.fasta.gz` 32 GB, `uniref100.fasta.gz` 63 GB, `uniref50.xml.gz` 20.6 GB |
| UniRef90 pool (`train/`) | about 25 GB, 97.8M sequences in 3,261 shards (plus the 34 GB filtered FASTA) |
| `members.tsv` (UniRef90) | 4.8 GB, 121.5M lines; reading the XML takes about 3.5 min |
| Cluster index (UniRef90) | about 2.3 GB (`cluster_index/index/`); building it takes about 10 min |
| MMseqs2 | roughly linear in the number of sequences: minutes for a few million, hours for about 100M |
