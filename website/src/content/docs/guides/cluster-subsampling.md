---
title: Cluster subsampling
description: Train on one random member of every cluster per epoch instead of a fixed set of sequences. Turn it on with a cluster_subsampling block under a data.yaml source; UniProt or MMseqs2 clusters.
---

A normal data run trains on a fixed set of sequences, for example the UniRef50 representatives. Cluster subsampling keeps the clusters but changes the sequence: **each epoch, every cluster contributes one random member**. With UniRef90 members under UniRef50 clusters, an epoch is 32.8M samples (one per train cluster) drawn from 97.8M sequences, and the member a cluster gives in epoch 2 is usually not the one it gave in epoch 1.

What changes against a normal run:

- **An epoch has one sample per train cluster.** The manifest's `train_sequences` counts clusters, and `total_train_tokens` is the expected token count of one epoch.
- **The member draw depends only on `pretrain.yaml`'s `seed` and the epoch**, not on the number of GPUs, the dataloader workers or a resume.
- **Validation holds out whole clusters** and scores each held-out cluster's representative, the same sequences every epoch. No member of a validation cluster reaches training.
- Pretraining needs `use_packing: true`, the template default.

## Turn it on

Add a `cluster_subsampling` block under the source in `data.yaml`. The block being there turns it on; delete or comment it out to turn it off. The template has this example, commented out:

```yaml title="data.yaml"
sources:
  uniref50:
    fasta: uniref50
    max_seqs: -1         # counts clusters here: all of them
    val_ratio: 0.0044    # share of clusters held out, as in the UniRef90 experiment
    cluster_subsampling:
      members: uniref90  # uniref90, uniref100, a UniRef90/100 FASTA URL/path, or null (= this fasta)
      clustering: auto   # auto: UniProt clusters for UniRef data, MMseqs2 otherwise; or uniprot, mmseqs2
      identity: 0.5      # mmseqs2 only: min sequence identity within a cluster
      coverage: 0.8      # mmseqs2 only: min alignment coverage
```

This trains on UniRef90 sequences under UniProt's UniRef50 clusters. The UniRef50 FASTA itself is never downloaded: the stages fetch `uniref90.fasta.gz` and `uniref50.xml.gz`, which lists each cluster's members.

- [`members`](/nanoplm/reference/data/#data-cluster_subsampling-members) is the set of sequences training draws from, and it is required. `uniref90` and `uniref100` are shortcuts for UniProt's files. A URL or path must be a UniRef90 or UniRef100 FASTA. `null` uses the source's own `fasta`: that is how you cluster your own sequences.
- [`clustering`](/nanoplm/reference/data/#data-cluster_subsampling-clustering) picks where the clusters come from. See [When MMseqs2 runs](#when-mmseqs2-runs).
- [`identity`](/nanoplm/reference/data/#data-cluster_subsampling-identity) and [`coverage`](/nanoplm/reference/data/#data-cluster_subsampling-coverage) only apply when MMseqs2 makes the clusters.

Set `max_seqs: -1` and a small `val_ratio` on the source, as above. Both count **clusters** here. With the template's `data_params` (`max_seqs: 20000`, `val_ratio: 0.1`), training would see at most 20,000 clusters and hold out 10% of them, so `from-yaml` warns before any stage runs:

```text
Warning: sources.uniref50: max_seqs=20000 trains on at most 20,000 clusters; set sources.uniref50.max_seqs: -1 to use all of them
Warning: sources.uniref50: val_ratio=0.1 holds out 10.0% of the clusters for validation; the UniRef90 experiment used sources.uniref50.val_ratio: 0.0044
```

`val_ratio: 0.0044` holds out 145k of the 32.9M UniRef50 clusters. You do not need to change `samples_per_shard`: when the pool would need more than 9,999 shards, the pool stage raises it and logs the value it used.

Then point `pretrain.yaml`'s `dataset_dir` at `output_dir`. Training reads the cluster index from the manifest; nothing else in `pretrain.yaml` changes.

## Read the table

The pre-run table ([explained here](/nanoplm/pipeline/data/#read-the-table-before-the-run)) lists the files the stages will read and the cluster source. For the example above:

```text
data.yaml: 1 source -> /work/output/data/pretrain_data
source    fasta                                            max seqs       len     val      clustering
uniref50  uniref90.fasta.gz (url) + uniref50.xml.gz (url)  all clusters*  20-512  0.0044*  UniProt uniref50.xml.gz (members: UniRef90)
* = set on this source
```

For your own FASTA with `members: null`, the `clustering` column names MMseqs2, and the summary at the end of the run gives the cluster counts:

```text
mine    my_proteins.fasta (local, 71.9 kB)  all clusters  20-512  0.05  MMseqs2 linclust, identity 0.5, coverage 0.8
...
Pretraining data pipeline complete (cluster subsampling).

  Members:         my_proteins.fasta
  Cluster source:  MMseqs2 linclust, identity 0.5, coverage 0.8
  Pool sequences:  300
  MMseqs2 clusters:  68
    Train:         65
    Val:           3
```

## When MMseqs2 runs

UniProt publishes the UniRef50 clusters, so for UniRef90 or UniRef100 members nanoPLM reads them from `uniref50.xml.gz` and MMseqs2 does not run. For any other members (your own FASTA, TrEMBL, UniRef50 itself), MMseqs2 makes the clusters.

| `fasta` | `members` | `clustering: auto` uses |
| --- | --- | --- |
| `uniref50` | `uniref90` or `uniref100` | UniProt clusters |
| `uniref90` or `uniref100` (or a local copy) | `null` | UniProt clusters |
| your own FASTA, or `trembl` | `null` | MMseqs2 |
| `uniref50` | `null` | MMseqs2 over all ~60M UniRef50 sequences (hours; `from-yaml` warns) |

`clustering: mmseqs2` uses MMseqs2 in every row, for example to cluster UniRef90 at `identity: 0.3` instead of UniProt's 50%. Then `uniref50.xml.gz` is not downloaded, and with `fasta: uniref50` the UniRef50 FASTA is not either: only the members are. `clustering: uniprot` accepts only the first two rows and fails at config time on the others.

UniRef90 and UniRef100 members share one set of clusters and one train / validation split: the UniRef100 representative is the same sequence as the UniRef90 one.

MMseqs2 runs `easy-linclust --min-seq-id <identity> -c <coverage> --cov-mode 0` over the filtered members. Coverage mode 0 means the alignment must cover that share of **both** sequences, so a fragment does not join a cluster on a short local match. MMseqs2 picks each cluster's representative. Install it with `nanoplm deps install --mmseqs-only` (see [Install](/nanoplm/start/install/#flags)). `from-yaml` checks for the binary before any stage runs.

What MMseqs2 costs, with `max_workers: 8` threads:

| Members | Time | RAM | Temporary disk |
| --- | --- | --- | --- |
| 2M sequences | 5 min | 2.6 GB | 4 GB |
| 10M sequences | 25-30 min | 10-13 GB | 15-20 GB |
| ~100M sequences | hours | | |

Time grows about linearly with the number of sequences. The temporary disk is about 3 times the filtered FASTA, under `output_dir/cluster_index/.mmseqs_tmp`, and is deleted afterwards. Run MMseqs2 on a compute node, not a shared login node. `max_workers` sets the thread count (`-1` uses every core).

Either way, the clusters end up as one table, `output_dir/cluster_index/members.tsv` (`cluster_id<TAB>sequence_id`, representative first), and the rest of the pipeline reads that table.

## Disk and time for UniRef

Measured on UniProt release 2026_03 on a login node (decimal GB). Index build times include verification and the validation shards.

| Item | UniRef90 members | UniRef100 members |
| --- | --- | --- |
| Download (`downloads/`) | `uniref90.fasta.gz` 32.1 GB, about 27 min | `uniref100.fasta.gz` 63.2 GB, about 53 min |
| `uniref50.xml.gz` (both) | 20.6 GB, about 20 min | same file |
| Extracted FASTA (`<source>/`) | 61.0 GB, about 4 min | 121.1 GB, about 8 min |
| Filtered FASTA | 36.2 GB, about 2.5 min | 66.7 GB, about 5 min |
| Pool (`train/`) | 24.9 GB, 97.8M sequences, about 3 min | 46.6 GB, 169.0M sequences, about 5 min |
| `members.tsv` | 4.8 GB, about 3.5 min | 8.9 GB, about 5 min |
| Cluster index + `val/` | about 2.3 GB, 10-14 min, 10-12 GB RAM | about 3.2 GB, 15-20 min, 18-20 GB RAM |
| **Total disk** | **about 182 GB** | **about 330 GB** |
| Needed for training | about 27 GB (`train/`, `cluster_index/`, `val/`) | about 50 GB |

The total assumes DVC's linked cache, the default for a new folder (see [Stages and caching](#stages-and-caching)). With a copy cache it is about 343 GB and 640 GB. On a busy GPFS file system, DVC hashing the outputs adds 15-50 min to a first run.

`uniref50.xml.gz` is downloaded once into `output_dir/downloads/` and kept, even after a new UniProt release. If the cluster index then reports pool sequences missing from `members.tsv`, the error names the XML: delete it to download the current one.

## Errors and warnings

`from-yaml` checks the block before any stage runs. `<name>` is your source name.

| Message | Fix |
| --- | --- |
| `sources.<name>.cluster_subsampling is empty. It needs members (null = this fasta)` | Set `members`, or delete the block to turn it off. |
| `sources.<name>.cluster_subsampling: remove enabled; the block being present turns it on` | Delete the `enabled` key. |
| `sources.<name>.cluster_subsampling.members is uniref90, so sources.<name>.fasta must be UniRef50 or uniref90 itself` | UniRef members need `fasta: uniref50` (or the same UniRef set). |
| `...members: uniref50 cannot be the members: UniRef50 entries are the clusters themselves` | Use `members: uniref90` or `uniref100`. |
| `...members (...) is not a UniRef set and differs from sources.<name>.fasta. Put that FASTA in fasta and set members: null.` | Your own FASTA goes in `fasta`, with `members: null`. |
| `...clustering: uniprot: UniProt clusters need UniRef data` | Use one of the UniProt rows in [When MMseqs2 runs](#when-mmseqs2-runs), or `clustering: auto`. |
| `sources.<name>: these members are clustered with MMseqs2, but mmseqs was not found` | `nanoplm deps install --mmseqs-only`, a release binary on `PATH`, or `NANOPLM_MMSEQS=/path/to/mmseqs`. |
| `...val_ratio=... of at most ... clusters (max_seqs) holds out 0; cluster subsampling needs at least one train and one val cluster` | Raise `val_ratio` to the value the message names, or set `max_seqs: -1`. |
| `... was built by the data-sampler experiment scripts ...` (at pretraining start) | `dataset_dir` is a folder from the old experiment scripts. Rebuild it with `nanoplm data from-yaml`. |
| Warning: `max_seqs=... trains on at most ... clusters` / `val_ratio=... holds out ...% of the clusters` | See [Turn it on](#turn-it-on). |
| Warning: `fasta is UniRef50 and members is null, so MMseqs2 will cluster all ~60M UniRef50 sequences` | To use UniProt's clusters, set `members: uniref90`. To cluster UniRef50 on purpose, set `clustering: mmseqs2` and the warning goes away. |
| Warning: `identity / coverage only apply to MMseqs2 clustering` | They are ignored with UniProt clusters. Set `clustering: mmseqs2` to use them. |
| Warning: `cluster_subsampling ignores shuffle / shuffle_seed` | The pool keeps FASTA order; training shuffles the clusters each epoch. |

`drop_invalid_sequences` works as in a normal run, on the members. With UniProt clusters, a cluster whose representative is dropped (invalid, or outside `min_seq_len`..`max_seq_len`) is left out of training with all its members. MMseqs2 only clusters sequences that passed the filter, so it never loses a representative.

## Stages and caching

With cluster subsampling, `from-yaml` runs these stages instead of `shuffle` and `split`:

```
get_members_fasta → extract_members → check → filter → cluster_pool
get_cluster_xml → cluster_members → cluster_index → write_manifest
```

`get_members_fasta` and `extract_members` only appear when `members` differs from `fasta`; otherwise the normal `get_file` and `extract` stages fetch the source fasta. `get_cluster_xml` only appears for UniProt clusters. A local file has no download stage.

Each stage reruns only when its own inputs change:

| You change | What reruns |
| --- | --- |
| `clustering`, `identity`, `coverage` | `cluster_members` and later. Never the tokenized pool. |
| `val_ratio`, `max_seqs` | `cluster_index` and the manifest. |
| `max_workers` | Nothing. |
| Remove the block from `fasta: uniref50` + `members: uniref90` | Downloads `uniref50.fasta.gz`, which the cluster run never fetched. |

When `from-yaml` creates the DVC repository, it sets `cache.type hardlink,symlink`, so DVC's cache (`.dvc/cache`) links to the outputs instead of holding a second copy. Linked outputs are read-only; run `dvc unprotect <path>` before editing one. If `output_dir` and `.dvc/cache` are on different file systems, DVC falls back to symlinks and the data lives in `.dvc/cache`: deleting the cache then breaks the dataset. After changing settings, `dvc gc --workspace` deletes cached versions that the current outputs no longer use, and keeps the current ones. A repository that already existed keeps its own `cache.type`; to switch it, run `dvc config cache.type hardlink,symlink` and `dvc checkout --relink` in the folder of `data.yaml`.
