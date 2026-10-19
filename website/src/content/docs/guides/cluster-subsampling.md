---
title: Cluster subsampling
description: Train on one random member of every cluster per epoch instead of the cluster representatives. Turn it on in data.yaml with the cluster_subsampling block; UniProt or MMseqs2 clusters.
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
  clustering: auto       # auto: UniProt clusters for UniRef data, MMseqs2 otherwise; or uniprot, mmseqs2
  identity: 0.5          # mmseqs2 only: min sequence identity within a cluster
  coverage: 0.8          # mmseqs2 only: min alignment coverage
```

- [`members`](/nanoplm/reference/data/#data-cluster_subsampling-members): the sequences training draws from. `uniref90` and `uniref100` are shortcuts for UniProt's `uniref90.fasta.gz` / `uniref100.fasta.gz`. A value starting with `http(s)://` is downloaded, anything else is a local path. `null` uses `data_dirs.url` (or `local_fasta`) itself.
- [`clustering`](/nanoplm/reference/data/#data-cluster_subsampling-clustering): where the clusters come from. `auto` uses UniProt's UniRef50 clusters for UniRef data and MMseqs2 for anything else (see [Where the clusters come from](#where-the-clusters-come-from)). `uniprot` requires UniProt clusters and fails at config time when the data is not UniRef. `mmseqs2` always clusters the members with MMseqs2, also for UniRef data.
- [`identity`](/nanoplm/reference/data/#data-cluster_subsampling-identity) and [`coverage`](/nanoplm/reference/data/#data-cluster_subsampling-coverage) are only read when MMseqs2 does the clustering.

Before any stage runs, `nanoplm data from-yaml` prints the cluster source it picked, and the summary at the end repeats it:

```text title="Output"
Cluster source: UniProt uniref50.xml.gz (members: UniRef90)
```

```text title="Output"
Cluster source: MMseqs2 linclust, identity 0.5, coverage 0.8 (members: /data/my_proteins.fasta)
```

Then run `nanoplm data from-yaml` as usual and point `pretrain.yaml`'s `dataset_dir` at `output_dir`. Training picks the cluster-subsampled dataset from the manifest; nothing else changes in `pretrain.yaml`.

### Settings for a UniRef90 or UniRef100 run

The template's `data_params` are sized for a quick SwissProt run. They still work in this mode, but for UniRef-scale clusters set these three:

| Key | Template | UniRef90 / UniRef100 run | Why |
| --- | --- | --- | --- |
| `seqs_num` | `20000` | `-1` | In this mode it caps the number of **clusters**, not sequences. With the template's value, training sees at most 20,000 clusters. |
| `val_ratio` | `0.1` | `0.0044` | The fraction of **clusters** held out. 0.0044 holds out 145k of the 32.9M UniRef50 clusters; 0.1 would hold out 3.3M. |
| `samples_per_shard` | `2000` | `30000` | The pool has 97.8M (UniRef90) or 169.0M (UniRef100) sequences. At 2000 per shard that would be 49k or 85k shard files, past the 9,999 the shard names sort correctly for. A value that is too small is raised automatically and the log names the value used (10000 for UniRef90, 20000 for UniRef100); 30000 gives 3,261 and 5,635 shards. |

`nanoplm data from-yaml` warns before any stage runs when `seqs_num` is positive or `val_ratio` is above 0.01.

`drop_invalid_sequences` works as in a normal run, on the member sequences. With UniProt clusters, a cluster whose representative is dropped (invalid or outside `min_seq_len`..`max_seq_len`) leaves training with all its members; this keeps the index identical to the experiment's. MMseqs2 only clusters sequences that passed the filter, so its representatives always survive.

## Scenarios

`clustering` is `auto` unless the row says otherwise.

| # | Settings | Cluster source | What happens |
| --- | --- | --- | --- |
| 1 | `enabled: false` | none | The normal pipeline, unchanged. |
| 2 | `url`: UniRef50, `members: uniref90` | UniProt (`uniref50.xml.gz`) | UniRef90 members under UniProt's UniRef50 clusters. 32.8M samples per epoch. The UniRef50 FASTA itself is never downloaded. |
| 3 | `url`: UniRef50, `members: uniref100` | UniProt (`uniref50.xml.gz`) | UniRef100 members under UniProt's UniRef50 clusters. Same clusters, train and validation split as scenario 2. |
| 4 | your own FASTA as `url` / `local_fasta`, `enabled: true` | MMseqs2 | MMseqs2 clusters your sequences at `identity` / `coverage`. |
| 5 | `url`: UniRef90 (or a UniRef90 `local_fasta`), `members: null` | UniProt (`uniref50.xml.gz`) | The members are the url itself, under UniProt's UniRef50 clusters. No MMseqs2. |
| 6 | scenario 2, 3 or 5 with `clustering: mmseqs2`, e.g. `identity: 0.3` | MMseqs2 | MMseqs2 clusters the UniRef90 / UniRef100 members at your `identity` / `coverage`, for example 30% identity clusters instead of UniProt's 50%. `uniref50.xml.gz` is not downloaded, and with `url` UniRef50 neither is its FASTA: only the members are fetched. |

MMseqs2 runs in scenarios 4 and 6, and whenever `url` is UniRef50 with `members: null` (it then clusters UniRef50 itself; see the warning below). `clustering: uniprot` accepts scenarios 2, 3 and 5 and refuses the rest.

## Where the clusters come from

A FASTA counts as UniRef90 or UniRef100 by its headers (`>UniRef90_...`) when it is a local file, and by its file name (`uniref90.fasta.gz`) when it is a URL. The stages check the headers again once the file is on disk.

With `clustering: auto`:

- **The members are a UniRef90 or UniRef100 set, and `url` is UniRef50 or that same set**: UniProt's UniRef50 clusters. `uniref50.xml.gz` is downloaded into the raw directory next to `compressed_fasta`; a copy already there is reused. Each UniRef50 entry is one cluster. For UniRef90 the representative is the entry's representative member. For UniRef100 it is the UniRef100 record with the same accession as the UniRef90 representative, which is the same sequence, so UniRef90 and UniRef100 runs share one set of clusters and one train / validation split.
- **Anything else** (your own FASTA, TrEMBL, UniRef50 itself): MMseqs2 `easy-linclust` clusters the filtered member sequences with `--min-seq-id identity -c coverage --cov-mode 0`. Coverage mode 0 asks the alignment to cover that fraction of **both** sequences, so a fragment does not join a cluster on a short local match. The representative is the one MMseqs2 picks. MMseqs2 runs in a temporary directory under `output_dir` that is deleted afterwards.

`clustering: uniprot` takes the first branch and refuses anything else; `clustering: mmseqs2` always takes the second, on the member sequences.

Either way, the clusters end up as one table, `output_dir/cluster_index/members.tsv` (`cluster_id<TAB>sequence_id`, grouped by cluster, representative first), and the rest of the pipeline is built from that table.

## Errors and warnings

| Message | Cause |
| --- | --- |
| `cluster_subsampling.members is uniref90, so data_dirs.url must be UniRef50 or uniref90 itself` | UniRef members with a `url` (for example TrEMBL) whose clusters cannot be UniRef50. |
| `cluster_subsampling.clustering: uniprot: UniProt clusters need UniRef data (url UniRef50 + members uniref90/uniref100, or a UniRef90/100 url)` | `clustering: uniprot` with data UniProt has no clusters for: your own FASTA, TrEMBL, or UniRef50 with `members: null`. Use one of the named combinations, or `clustering: auto` / `mmseqs2`. |
| `cluster_subsampling.clustering must be one of auto, uniprot, mmseqs2` | Any other value, including `null`. |
| `cluster_subsampling.members: uniref50 cannot be the members` | UniRef50 entries are the clusters. Use `members: uniref90` or `uniref100`. |
| `cluster_subsampling.members (...) is not a UniRef set and differs from data_dirs.url. Put that FASTA in data_dirs.url (or local_fasta) and set cluster_subsampling.members: null.` | Your own member FASTA goes in `url`, so there is only one source. |
| `... clustered with MMseqs2, but mmseqs was not found` | MMseqs2 is needed (scenarios 4 and 6). Install it with `nanoplm deps install --mmseqs-only`, put a release binary on `PATH`, or set `NANOPLM_MMSEQS=/path/to/mmseqs`. |
| `... was built by the data-sampler experiment scripts ... Rebuild it with nanoplm data from-yaml` | A dataset directory from the old experiment (it has a `RUNTIME_SAMPLER_REQUIRED` marker, or a pool with the experiment's index next to it). Its manifest describes the whole pool, so it is refused. |
| Warning: `identity / coverage only apply to MMseqs2 clustering; they are ignored with UniProt clusters. Set cluster_subsampling.clustering: mmseqs2 to use them` | `identity` or `coverage` differs from the template default (0.5 / 0.8) while UniProt clusters are used. |
| Warning: `MMseqs2 will cluster all ~60M UniRef50 sequences` | `url` is UniRef50, `members` is `null` and `clustering` is `auto`, so MMseqs2 clusters UniRef50 itself. To train on the members of UniProt's UniRef50 clusters, set `members: uniref90` (with `clustering: auto` or `uniprot`). An explicit `clustering: mmseqs2` does not warn. |
| `clustering: mmseqs2 clusters the members (...); the data_dirs.url FASTA (...) is not needed and is not downloaded` | Not a warning: scenario 6 with `url` UniRef50 and UniRef members. |
| Warning: `cluster_subsampling ignores data_params.shuffle / shuffle_seed` | `shuffle` or `shuffle_seed` differs from the template (`true` / `24`). The pool keeps FASTA order; the training sampler shuffles the clusters each epoch. |

## Stages and caching

`nanoplm data from-yaml` runs these as DVC stages:

```
get_members_fasta → extract_members → check → filter → cluster_pool
get_cluster_xml → cluster_members → cluster_index → write_manifest
```

`get_members_fasta` / `extract_members` only appear when `members` differs from `url`; otherwise the normal `get_file` / `extract` stages fetch the url. `get_cluster_xml` only appears for UniProt clusters, so `clustering: mmseqs2` never fetches the XML. The `shuffle` and `split` stages are not used.

Each stage reruns only when its own inputs change. Changing `clustering`, `identity` or `coverage` reruns `cluster_members` and the stages after it, never the tokenized pool. Switching `clustering` between UniProt and MMseqs2 also adds or drops `get_cluster_xml`. Changing `val_ratio` or `seqs_num` reruns `cluster_index` and the manifest. Changing `max_workers` reruns `cluster_pool` (and `cluster_members` with MMseqs2). The XML is not copied into the DVC cache.

Switching `enabled` between `true` and `false` changes how the `get_file` stage is written, so DVC downloads `url` again even when the file is already there.

### Disk space

DVC keeps a copy of every stage output in its cache (`.dvc/cache`), so by default a run takes about twice the space of its outputs: for UniRef90 the cached downloads, FASTAs, pool and index come to about 140 GB more. To avoid the copy, let the cache link files instead. In the directory of `data.yaml`, before the first run:

```bash
dvc init            # add --no-scm outside a git repository, --subdir inside one
dvc config cache.type reflink,hardlink,symlink
```

`nanoplm data from-yaml` then uses that repository. On a project that has already run, set `cache.type` the same way and run `dvc checkout --relink`.

After changing settings, `dvc gc --workspace` deletes cached versions that the current outputs no longer use.

## Costs

Measured on UniProt release 2026_03, on one login node:

| Item | Size or time |
| --- | --- |
| Downloads | `uniref90.fasta.gz` 32 GB, `uniref100.fasta.gz` 63 GB, `uniref50.xml.gz` 20.6 GB |
| UniRef90 pool (`train/`) | about 25 GB, 97.8M sequences in 3,261 shards (plus the 34 GB filtered FASTA) |
| `members.tsv` (UniRef90) | 4.8 GB, 121.5M lines; reading the XML takes about 3.5 min |
| Cluster index (UniRef90) | about 2.3 GB (`cluster_index/index/`); building it takes about 10 min |
| MMseqs2 | roughly linear in the number of sequences: minutes for a few million, hours for about 100M |
