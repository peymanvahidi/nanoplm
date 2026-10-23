---
title: Cluster subsampling
description: Train on one random member of every cluster per epoch instead of the cluster representatives. Turn it on with a cluster_subsampling block under a data.yaml source; UniProt or MMseqs2 clusters.
---

A normal data run trains on a fixed set of sequences, for example the UniRef50 representatives. Cluster subsampling keeps the clusters but swaps the sequence: **each epoch, every cluster contributes one random member**. With UniRef90 members under UniRef50 clusters that is 32.8M samples per epoch (one per train cluster), drawn from a pool of 97.8M sequences, and a member seen in epoch 1 is usually not the one seen in epoch 2.

What changes against a normal run:

- **Samples per epoch = train clusters**, not sequences. The manifest's `train_sequences` counts clusters and `total_train_tokens` is the expected token count of one epoch.
- **The member draw is keyed on `pretrain.yaml`'s `seed`** and the epoch. It does not depend on the number of GPUs, dataloader workers or on resuming.
- **Validation is held out by whole cluster** and scores each held-out cluster's representative, the same sequences every epoch. Sibling sequences of a validation cluster never reach training.
- Pretraining needs `use_packing: true` (the template default).

## Turn it on

Add a `cluster_subsampling` block under the source in `data.yaml`. The block being there turns it on; delete or comment it out to turn it off. There is no `enabled` key. The template ships this example commented out:

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

- [`members`](/nanoplm/reference/data/#data-cluster_subsampling-members): the sequences training draws from. It is required, so an empty block is an error rather than a silent default. `uniref90` and `uniref100` are shortcuts for UniProt's `uniref90.fasta.gz` / `uniref100.fasta.gz`. Any other value is a UniRef90 or UniRef100 FASTA: one starting with `http(s)://` or `ftp://` is downloaded, anything else is a local path. `null` uses the source's `fasta` itself; that is how to cluster your own sequences.
- [`clustering`](/nanoplm/reference/data/#data-cluster_subsampling-clustering): where the clusters come from. `auto` uses UniProt's UniRef50 clusters for UniRef data and MMseqs2 for anything else (see [Where the clusters come from](#where-the-clusters-come-from)). `uniprot` requires UniProt clusters and fails at config time when the data is not UniRef. `mmseqs2` always clusters the members with MMseqs2, also for UniRef data.
- [`identity`](/nanoplm/reference/data/#data-cluster_subsampling-identity) and [`coverage`](/nanoplm/reference/data/#data-cluster_subsampling-coverage) are only read when MMseqs2 does the clustering.

Before any stage runs, `nanoplm data from-yaml` prints one row per source. The `fasta` column lists the files the stages read: here the members FASTA and the XML, since `uniref50.fasta.gz` is never fetched. `(url)` means it will be downloaded, `(reuse, <size>, <date>)` that a copy written on that date is already there. The last column names the cluster source it picked; `*` marks a value set on the source:

```text title="Output"
data.yaml: 1 source -> /work/output/data/pretrain_data
source    fasta                                            max seqs       len     val      clustering
uniref50  uniref90.fasta.gz (url) + uniref50.xml.gz (url)  all clusters*  20-512  0.0044*  UniProt uniref50.xml.gz (members: UniRef90)
* = set on this source
```

With your own FASTA the last column reads `MMseqs2 linclust, identity 0.5, coverage 0.8`. The summary at the end repeats it.

Then point `pretrain.yaml`'s `dataset_dir` at `output_dir`. Training picks the cluster-subsampled dataset from the manifest; nothing else changes in `pretrain.yaml`.

### Settings for a UniRef90 or UniRef100 run

The template's `data_params` are sized for a quick SwissProt run. They still work in this mode, but for UniRef-scale clusters set these three, on the source or in `data_params`:

| Key | Template | UniRef90 / UniRef100 run | Why |
| --- | --- | --- | --- |
| `max_seqs` | `20000` | `-1` | In this mode it caps the number of **clusters**, not sequences. With the template's value, training sees at most 20,000 clusters. |
| `val_ratio` | `0.1` | `0.0044` | The fraction of **clusters** held out. 0.0044 holds out 145k of the 32.9M UniRef50 clusters; 0.1 would hold out 3.3M. |
| `samples_per_shard` | `2000` | `30000` | The pool has 97.8M (UniRef90) or 169.0M (UniRef100) sequences. At 2000 per shard that would be 49k or 85k shard files, past the 9,999 the shard names sort correctly for. A value that is too small is raised automatically and the log names the value used (10000 for UniRef90, 20000 for UniRef100); 30000 gives 3,261 and 5,635 shards. |

`nanoplm data from-yaml` warns before any stage runs when `max_seqs` is positive or `val_ratio` is above 0.01.

`drop_invalid_sequences` works as in a normal run, on the member sequences. With UniProt clusters, a cluster whose representative is dropped (invalid or outside `min_seq_len`..`max_seq_len`) leaves training with all its members; this keeps the index identical to the experiment's. MMseqs2 only clusters sequences that passed the filter, so its representatives always survive.

## Scenarios

`clustering` is `auto` unless the row says otherwise.

| # | Settings | Cluster source | What happens |
| --- | --- | --- | --- |
| 1 | no `cluster_subsampling` block | none | The normal pipeline, unchanged. |
| 2 | `fasta: uniref50`, `members: uniref90` | UniProt (`uniref50.xml.gz`) | UniRef90 members under UniProt's UniRef50 clusters. 32.8M samples per epoch. The UniRef50 FASTA itself is never downloaded. |
| 3 | `fasta: uniref50`, `members: uniref100` | UniProt (`uniref50.xml.gz`) | UniRef100 members under UniProt's UniRef50 clusters. Same clusters, train and validation split as scenario 2. |
| 4 | your own FASTA as `fasta`, `members: null` | MMseqs2 | MMseqs2 clusters your sequences at `identity` / `coverage`. |
| 5 | `fasta: uniref90` (or a local UniRef90 file), `members: null` | UniProt (`uniref50.xml.gz`) | The members are the source fasta itself, under UniProt's UniRef50 clusters. No MMseqs2. |
| 6 | scenario 2, 3 or 5 with `clustering: mmseqs2`, e.g. `identity: 0.3` | MMseqs2 | MMseqs2 clusters the UniRef90 / UniRef100 members at your `identity` / `coverage`, for example 30% identity clusters instead of UniProt's 50%. `uniref50.xml.gz` is not downloaded, and with `fasta: uniref50` neither is its FASTA: only the members are fetched. |

MMseqs2 runs in scenarios 4 and 6, and whenever `fasta` is UniRef50 with `members: null` (it then clusters UniRef50 itself; see the warning below). `clustering: uniprot` accepts scenarios 2, 3 and 5 and refuses the rest.

## Where the clusters come from

A FASTA counts as UniRef90 or UniRef100 by its headers (`>UniRef90_...`) when it is a local file, and by its file name (`uniref90.fasta.gz`) when it is a URL. The stages check the headers again once the file is on disk.

With `clustering: auto`:

- **The members are a UniRef90 or UniRef100 set, and `fasta` is UniRef50 or that same set**: UniProt's UniRef50 clusters. `uniref50.xml.gz` is downloaded into `output_dir/downloads/`; a copy already there is reused, even after a new UniProt release. If the cluster index then reports pool sequences missing from `members.tsv`, the error names the XML: delete it to re-download. Each UniRef50 entry is one cluster. For UniRef90 the representative is the entry's representative member. For UniRef100 it is the UniRef100 record with the same accession as the UniRef90 representative, which is the same sequence, so UniRef90 and UniRef100 runs share one set of clusters and one train / validation split.
- **Anything else** (your own FASTA, TrEMBL, UniRef50 itself): MMseqs2 `easy-linclust` clusters the filtered member sequences with `--min-seq-id identity -c coverage --cov-mode 0`. Coverage mode 0 asks the alignment to cover that fraction of **both** sequences, so a fragment does not join a cluster on a short local match. The representative is the one MMseqs2 picks. MMseqs2 runs in a temporary directory under `output_dir` that is deleted afterwards.

`clustering: uniprot` takes the first branch and refuses anything else; `clustering: mmseqs2` always takes the second, on the member sequences.

Either way, the clusters end up as one table, `output_dir/cluster_index/members.tsv` (`cluster_id<TAB>sequence_id`, grouped by cluster, representative first), and the rest of the pipeline is built from that table.

## Errors and warnings

| Message | Cause |
| --- | --- |
| `sources.<name>.cluster_subsampling.members is uniref90, so sources.<name>.fasta must be UniRef50 or uniref90 itself` | UniRef members with a `fasta` (for example TrEMBL) whose clusters cannot be UniRef50. |
| `cluster_subsampling.clustering: uniprot: UniProt clusters need UniRef data (fasta UniRef50 + members uniref90/uniref100, or a UniRef90/100 fasta)` | `clustering: uniprot` with data UniProt has no clusters for: your own FASTA, TrEMBL, or UniRef50 with `members: null`. Use one of the named combinations, or `clustering: auto` / `mmseqs2`. |
| `cluster_subsampling.clustering must be one of auto, uniprot, mmseqs2` | Any other value, including `null`. |
| `cluster_subsampling.members: uniref50 cannot be the members` | UniRef50 entries are the clusters. Use `members: uniref90` or `uniref100`. |
| `cluster_subsampling.members (...) is not a UniRef set and differs from sources.<name>.fasta. Put that FASTA in fasta and set members: null.` | Your own member FASTA goes in `fasta`. |
| `cluster_subsampling is empty. It needs members (null = this fasta)` / `needs members` | A bare `cluster_subsampling:`, or one with every field commented out. Set `members`, or delete the block to turn it off. |
| `remove enabled; the block being present turns it on` | An `enabled` key left over from an earlier layout. |
| `cluster_subsampling now sits under its source` | A top-level `cluster_subsampling` block. Move it under the source. |
| `sources.<name>: these members are clustered with MMseqs2, but mmseqs was not found` | MMseqs2 is needed (scenarios 4 and 6). Install it with `nanoplm deps install --mmseqs-only`, put a release binary on `PATH`, or set `NANOPLM_MMSEQS=/path/to/mmseqs`. |
| `... was built by the data-sampler experiment scripts ... Rebuild it with nanoplm data from-yaml` | A dataset directory from the old experiment (it has a `RUNTIME_SAMPLER_REQUIRED` marker, or a pool with the experiment's index next to it). Its manifest describes the whole pool, so it is refused. |
| Warning: `identity / coverage only apply to MMseqs2 clustering; they are ignored with UniProt clusters. Set sources.<name>.cluster_subsampling.clustering: mmseqs2 to use them` | `identity` or `coverage` differs from the template default (0.5 / 0.8) while UniProt clusters are used. |
| Warning: `MMseqs2 will cluster all ~60M UniRef50 sequences` | `fasta` is UniRef50, `members` is `null` and `clustering` is `auto`, so MMseqs2 clusters UniRef50 itself. To train on the members of UniProt's UniRef50 clusters, set `members: uniref90` (with `clustering: auto` or `uniprot`). An explicit `clustering: mmseqs2` does not warn. |
| `clustering: mmseqs2 clusters the members (...); the source fasta (...) is not needed and is not downloaded` | Not a warning: scenario 6 with `fasta: uniref50` and UniRef members. |
| Warning: `cluster_subsampling ignores shuffle / shuffle_seed` | `shuffle` or `shuffle_seed` differs from the template (`true` / `24`). The pool keeps FASTA order; the training sampler shuffles the clusters each epoch. |

## Stages and caching

`nanoplm data from-yaml` runs these as DVC stages:

```
get_members_fasta → extract_members → check → filter → cluster_pool
get_cluster_xml → cluster_members → cluster_index → write_manifest
```

`get_members_fasta` / `extract_members` only appear when `members` differs from `fasta`; otherwise the normal `get_file` / `extract` stages fetch the source fasta. A local file is read in place, with no `get_file` stage. `get_cluster_xml` only appears for UniProt clusters, so `clustering: mmseqs2` never fetches the XML. The `shuffle` and `split` stages are not used.

Each stage reruns only when its own inputs change. Changing `clustering`, `identity` or `coverage` reruns `cluster_members` and the stages after it, never the tokenized pool. Switching `clustering` between UniProt and MMseqs2 also adds or drops `get_cluster_xml`. Changing `val_ratio` or `max_seqs` reruns `cluster_index` and the manifest. Changing `max_workers` reruns nothing: it only sets how many cores the stages use. The XML is not copied into the DVC cache.

A download stage depends only on its URL. Adding or removing the `cluster_subsampling` block does not download the source fasta again, and when the members are the source fasta it does not extract or check it again either. Removing the block from a `fasta: uniref50` + `members: uniref90` source does download `uniref50.fasta.gz`, because the cluster run never fetched it.

### Disk space

A run's files all sit under `output_dir`: `downloads/` (one file per URL), `<source>/` (extracted, filtered and split FASTAs), and the dataset itself (`train/`, `val/`, `cluster_index/`, `.data_manifest`).

DVC keeps every stage output in its cache (`.dvc/cache`). When `nanoplm data from-yaml` creates the DVC repository for a `sources:` data.yaml, it sets `cache.type hardlink,symlink`, so the cache links to the outputs instead of holding a second copy. Linked outputs are read-only; to edit one, run `dvc unprotect <path>` first. If `output_dir` and `.dvc/cache` are on different file systems, DVC cannot hardlink and falls back to symlinks into the cache. The dataset then lives in `.dvc/cache`: deleting the cache, or a `dvc gc` that drops the current versions, breaks it (`dvc gc --workspace` keeps them). Hardlinked files survive either. A repository that already exists keeps its own `cache.type`. To switch one, run `dvc config cache.type hardlink,symlink` and `dvc checkout --relink` in the directory of `data.yaml`. Old `data_dirs` configs keep DVC's default copy cache, because their `get_file` stage links a `local_fasta` into the outputs and a linked cache would make your own file read-only.

The extracted and filtered FASTAs stay in the cache. With a linked cache that costs no extra disk, and keeping them lets DVC restore them instead of extracting again. With a copy cache they would take about 97 GB extra for UniRef90. After changing settings, `dvc gc --workspace` deletes cached versions that the current outputs no longer use.

## Costs

Measured on UniProt release 2026_03 on a login node (decimal GB). Index build times include verification and the validation shards.

| Item | UniRef90 | UniRef100 |
| --- | --- | --- |
| Download (`downloads/`) | `uniref90.fasta.gz` 32.1 GB, about 27 min | `uniref100.fasta.gz` 63.2 GB, about 53 min |
| `uniref50.xml.gz` (both) | 20.6 GB, about 20 min; not in the DVC cache | same file |
| Extracted FASTA (`<source>/`) | 61.0 GB, about 4 min | 121.1 GB, about 8 min |
| Filtered FASTA | 36.2 GB, about 2.5 min | 66.7 GB, about 5 min |
| Pool (`train/`) | 24.9 GB, 97.8M sequences in 3,261 shards, about 3 min | 46.6 GB, 169.0M sequences in 5,635 shards, about 5 min |
| `members.tsv` | 4.8 GB, about 3.5 min | 8.9 GB, about 5 min |
| Cluster index + `val/` | about 2.3 GB, 10-14 min, 10-12 GB RAM | about 3.2 GB, 15-20 min, 18-20 GB RAM |
| **Total disk** | **about 182 GB** (about 343 GB with a copy cache) | **about 330 GB** (about 640 GB with a copy cache) |
| Needed for training | about 27 GB (`train/`, `cluster_index/`, `val/`) | about 50 GB |

On a busy GPFS file system, DVC hashing the outputs adds 15-50 min to a first run.

MMseqs2 (`clustering: mmseqs2`, or your own FASTA) runs time roughly linear in the number of sequences. With 8 threads, 2M sequences took 5 min, 2.6 GB RAM and 4 GB of temporary disk; 10M sequences take about 25-30 min, 10-13 GB RAM and 15-20 GB of temporary disk under `output_dir/cluster_index/.mmseqs_tmp`, about 3 times the filtered FASTA. About 100M sequences take hours. The temporary directory is deleted afterwards. Run MMseqs2 on a compute node, not a shared login node.
