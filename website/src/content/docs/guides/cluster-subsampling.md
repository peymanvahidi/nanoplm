---
title: Cluster subsampling
description: Train on one random member of every cluster per epoch instead of a fixed set of sequences. Turn it on with cluster_subsampling under a data.yaml source; the clusters come from UniProt, CL tags in the FASTA headers, a TSV file, or MMseqs2.
---

A normal data run trains on a fixed set of sequences, for example the UniRef50 representatives. Cluster subsampling trains on clusters instead. Each epoch, every cluster gives one random member, and in the next epoch it usually gives a different one.

Three words come up on this page:

- A **cluster** is a group of similar sequences, such as one UniRef50 entry.
- Its **members** are the sequences in it, such as the UniRef90 sequences under that UniRef50 entry.
- Its **representative** is the one member that stands for the cluster.

For example, with UniRef90 members under UniRef50 clusters, nanoPLM tokenizes the 97.8M UniRef90 sequences that pass the length filter, once. An epoch then draws 32.8M of them, one per training cluster.

How this differs from a normal run:

- An epoch has one sample per training cluster. The manifest's `train_sequences` counts clusters, and `total_train_tokens` is the expected token count of one epoch.
- Which member a cluster gives depends only on `pretrain.yaml`'s `seed` and the epoch. The number of GPUs, the dataloader workers and resuming do not change it. See [Seeds and shuffle](#seeds-and-shuffle).
- Validation holds out whole clusters and scores each one's representative, the same sequences every epoch. No member of a validation cluster is ever trained on.
- Pretraining needs `use_packing: true`, which is the template default.

## Turn it on

Add a `cluster_subsampling` block under a source in `data.yaml` and set `clusters` in it, to say where the clusters come from. `false`, or no block, turns it off. This example, from `nanoplm data get-yaml --subsampling`, trains on UniRef90 sequences under UniProt's UniRef50 clusters:

```yaml title="data.yaml"
sources:
  uniref50:
    fasta: uniref50
    cluster_subsampling:
      members: uniref90  # train on these sequences; leave out to use this fasta
      clusters: uniprot  # uniprot, headers (CL: tags in the FASTA headers), mmseqs2 (slow: hours), or a .tsv of representative<TAB>member
```

nanoPLM downloads `uniref90.fasta.gz` and `uniref50.xml.gz`, the file that lists each cluster's members. It never downloads the UniRef50 FASTA itself.

- [`members`](/nanoplm/reference/data/#data-cluster_subsampling-members) names the sequences that training draws from. Leave it out to use the source's own `fasta`. `uniref90` and `uniref100` are shortcuts for UniProt's files. A URL or path must be a UniRef90 or UniRef100 FASTA.
- [`clusters`](/nanoplm/reference/data/#data-cluster_subsampling-clusters) says where the clusters come from. It is required and has no default. See [Where the clusters come from](#where-the-clusters-come-from). Older configs call it `clustering`; that still works, and `from-yaml` asks you to rename it.
- [`identity`](/nanoplm/reference/data/#data-cluster_subsampling-identity) and [`coverage`](/nanoplm/reference/data/#data-cluster_subsampling-coverage) only apply when MMseqs2 makes the clusters. Setting them to other values with another cluster source is an error.

Here `max_seqs`, `val_ratio` and `val_size` count clusters, not sequences; `max_seqs: -1` uses every cluster. `val_ratio` is set once, in `data_params`; the UniRef90 experiment used `val_ratio: 0.0044` (145k of 32.9M clusters). If you leave the template's values (`max_seqs: 20000`, `val_ratio: 0.1`), training sees at most 20,000 clusters and holds out 10% of them. `from-yaml` warns about both before any stage runs:

```text
Warning: sources.uniref50: max_seqs=20000 (from data_params) trains on at most 20,000 clusters; set sources.uniref50.max_seqs: -1 to use all of them
Warning: sources.uniref50: val_ratio=0.1 (from data_params) holds out about 10.0% of the clusters for validation; the UniRef90 experiment used data_params.val_ratio: 0.0044
```

One `uniref50` source with `val_ratio: 0.0044` and `shuffle_seed: 20260918` holds out the same clusters as the experiment; with another `shuffle_seed` it holds out as many clusters, but a different set. In a [mix](/nanoplm/guides/mixing/#validation), the validation split is shared out by weight, so UniRef50 gets a different number of clusters and they are a different set.

You do not need to change `samples_per_shard`. If the tokenized members would need more than 9,999 shards, nanoPLM raises it and logs the value it used.

Then set `pretrain.yaml`'s `dataset_dir` to `output_dir`. Nothing else in `pretrain.yaml` changes.

## Where the clusters come from

`clusters` takes one of four values. nanoPLM never picks one for you, so MMseqs2 only runs when you ask for it.

| `clusters` | The clusters come from | Use it for |
| --- | --- | --- |
| `uniprot` | UniProt's UniRef50 clusters, read from `uniref50.xml.gz` | UniRef90 or UniRef100 members (`fasta: uniref50` with `members: uniref90` or `uniref100`, or a UniRef90/100 `fasta`) |
| `headers` | a `CL:<representative>` tag in every FASTA header | FASTA files that carry the tags, such as Logan's protein files |
| a `.tsv` file | `representative<TAB>member` lines | clusters you made yourself, for example with MMseqs2 |
| `mmseqs2` | MMseqs2, run on the sequences that pass the filter | sequences without clusters; it takes hours on large data (see [When MMseqs2 runs](#when-mmseqs2-runs)) |

`from-yaml` checks the value before any stage runs: `uniprot` on data that is not UniRef is an error, and so is `headers` when the first headers have no `CL:` tag.

UniRef90 and UniRef100 members share one set of clusters and one train / validation split, because a UniRef100 representative is the same sequence as the UniRef90 one.

### UniProt UniRef clusters

For UniRef90 or UniRef100 members, nanoPLM reads UniProt's UniRef50 clusters from `uniref50.xml.gz`:

```yaml title="data.yaml"
sources:
  uniref50:
    fasta: uniref50
    cluster_subsampling:
      members: uniref90
      clusters: uniprot
```

To cluster UniRef90 another way, for example at `identity: 0.3` instead of UniProt's 50%, set `clusters: mmseqs2`. nanoPLM then downloads only the members, not `uniref50.xml.gz` and not the UniRef50 FASTA.

### CL tags in the FASTA headers

Each header is the record id, then a tab (or space), then `CL:` and the id of the cluster's representative:

```text
>ID<TAB>CL:<representative>
```

Logan's protein files use this format. In these two headers (sequences left out), the first record is a representative, since its id is its own cluster id, and the second is a member of its cluster:

```text
>DRR000534_5840_1	CL:DRR000534_5840_1
>SRR10247178_697980_1	CL:DRR000534_5840_1
```

```yaml title="data.yaml"
sources:
  logan:
    fasta: "data/logan_sample.fasta.zst"
    cluster_subsampling:
      clusters: headers
```

- Every header needs a tag. One record without a tag stops the `cluster_members` stage.
- Every cluster needs its representative in the file: a record whose id is the cluster id.
- The records need not be grouped by cluster or sorted; they can come in any order.

Files like Logan's "all proteins" (about 31.6 billion records, 99% of them duplicates) are too large for this; use the Logan50 representatives file as a plain source instead, without `cluster_subsampling`.

### A cluster file

A `.tsv` file with one `representative<TAB>member` line per sequence, the representative also listed as its own member. This is MMseqs2's `_cluster.tsv` from `easy-cluster` or `easy-linclust`, and the output of `createtsv`:

```text
P1	P1
P1	P7
P2	P2
```

```yaml title="data.yaml"
sources:
  mine:
    fasta: "data/my_proteins.fasta"
    cluster_subsampling:
      clusters: "data/my_clusters.tsv"
```

- The ids are FASTA ids, the first word of each header without `>`. MMseqs2 writes the accession of a UniProt-style header (`P00001` for `sp|P00001|NAME`), so run it on a FASTA with plain ids.
- `.tsv.gz` and `.tsv.zst` are read as they are. A URL is downloaded into `output_dir/downloads/`.
- It must list every record of the FASTA exactly once, and every id in it must be in the FASTA. The `cluster_members` stage checks the whole file and says how many ids break a rule, with examples.

### MMseqs2, when you have no clusters

```yaml title="data.yaml"
sources:
  mine:
    fasta: "data/my_proteins.fasta"
    cluster_subsampling:
      clusters: mmseqs2
      identity: 0.5    # min sequence identity within a cluster
      coverage: 0.8    # min alignment coverage
```

`identity` and `coverage` default to these values. See [When MMseqs2 runs](#when-mmseqs2-runs) for what it costs.

Whichever source the clusters come from, nanoPLM writes them to one table, `output_dir/cluster_index/members.tsv`, with one `cluster_id<TAB>sequence_id` line per member and the representative first.

## Read the table

`from-yaml` prints a table before any stage runs ([columns explained here](/nanoplm/pipeline/data/#read-the-table-before-the-run)). With cluster subsampling, it also lists the cluster file, where the clusters come from and what the source trains on. For the UniRef example above:

```text
data.yaml: 1 source -> /work/output/data/pretrain_data
source    fasta                                            max seqs       len     val      cluster subsampling
uniref50  uniref90.fasta.gz (url) + uniref50.xml.gz (url)  all clusters*  20-512  0.0044*  clusters: UniProt uniref50.xml.gz | trains on: UniRef90 members
* = set on this source
```

The last column reads `clusters: CL: tags in headers`, `clusters: data/my_clusters.tsv` or `clusters: MMseqs2 (50% identity, 80% coverage)` for the other sources. The summary at the end of the run gives the cluster counts. "Pool sequences" is the number of member sequences that passed the filter and were tokenized:

```text
mine    my_proteins.fasta (local, 71.9 kB)  all clusters  20-512  0.05  clusters: MMseqs2 (50% identity, 80% coverage) | trains on: this fasta
...
Pretraining data pipeline complete (cluster subsampling).

  Members:         my_proteins.fasta
  Cluster source:  MMseqs2 (50% identity, 80% coverage)
  Pool sequences:  300
  MMseqs2 clusters:  68
    Train:         65
    Val:           3
```

## When MMseqs2 runs

MMseqs2 runs only when you set `clusters: mmseqs2`.

Install MMseqs2 with `nanoplm deps install --mmseqs-only` (see [Install](/nanoplm/start/install/#flags)). `from-yaml` checks for the binary before any stage runs. It runs `easy-linclust --min-seq-id <identity> -c <coverage> --cov-mode 0` over the members that passed the filter, and MMseqs2 picks each cluster's representative. With `--cov-mode 0` the alignment must cover that share of both sequences, so a fragment does not join a cluster on a short local match.

What MMseqs2 costs, with `max_workers: 8` threads:

| Members | Time | RAM | Temporary disk |
| --- | --- | --- | --- |
| 2M sequences | 5 min | 2.6 GB | 4 GB |
| 10M sequences | 25-30 min | 10-13 GB | 15-20 GB |
| ~100M sequences | hours | | |

Time grows about linearly with the number of sequences. The temporary files take about 3 times the size of the filtered FASTA, in `output_dir/cluster_index/.mmseqs_tmp`, and are deleted afterwards. Run MMseqs2 on a compute node, not a shared login node. `max_workers` sets the thread count, and `-1` uses every core.

## Disk and time for UniRef

Measured on UniProt release 2026_03 on a login node, in decimal GB.

| Item | UniRef90 members | UniRef100 members |
| --- | --- | --- |
| Download (`downloads/`) | `uniref90.fasta.gz` 32.1 GB, about 27 min | `uniref100.fasta.gz` 63.2 GB, about 53 min |
| `uniref50.xml.gz` (both) | 20.6 GB, about 20 min | same file |
| Extracted FASTA (`<source>/`) | 61.0 GB, about 4 min | 121.1 GB, about 8 min |
| Filtered FASTA | 36.2 GB, about 2.5 min | 66.7 GB, about 5 min |
| Tokenized members (`train/`) | 24.9 GB, 97.8M sequences, about 3 min | 46.6 GB, 169.0M sequences, about 5 min |
| `members.tsv` | 4.8 GB, about 3.5 min | 8.9 GB, about 5 min |
| `cluster_index/` + `val/` | about 2.3 GB, 10-14 min, 10-12 GB RAM | about 3.2 GB, 15-20 min, 18-20 GB RAM |
| **Total disk** | **about 182 GB** | **about 330 GB** |
| Needed for training | about 27 GB (`train/`, `cluster_index/`, `val/`) | about 50 GB |

The total assumes DVC's linked cache, the default in a new folder (see [Disk use and the DVC cache](/nanoplm/pipeline/data/#disk-use-and-the-dvc-cache)). With a copy cache it is about 343 GB and 640 GB. On a busy GPFS file system, a first run takes 15-50 min longer while DVC hashes the outputs.

nanoPLM downloads `uniref50.xml.gz` once into `output_dir/downloads/` and keeps it, even after a new UniProt release. If the members FASTA and the XML come from different UniProt releases, the `cluster_index` stage fails with an error that names the XML. Delete the XML so the next run downloads the current one.

## Errors and warnings

`from-yaml` checks the block before any stage runs. `<name>` is your source name.

| Message | Fix |
| --- | --- |
| `sources.<name>.cluster_subsampling is empty. Set clusters in it to ...` | Set `clusters`, or delete the block to turn it off. |
| `sources.<name>.cluster_subsampling: true no longer turns it on` | Write a block that sets `clusters`. See [Where the clusters come from](#where-the-clusters-come-from). |
| `sources.<name>.cluster_subsampling.clusters is required` | Set `clusters` to `uniprot`, `headers`, `mmseqs2` or a `.tsv` file. |
| `...clusters: auto was removed` | Choose where the clusters come from. See [Where the clusters come from](#where-the-clusters-come-from). |
| `sources.<name>.cluster_subsampling: remove enabled` | Delete the `enabled` key. The block turns it on, `false` (or no block) turns it off. |
| `sources.<name>.cluster_subsampling.members is uniref90, so sources.<name>.fasta must be UniRef50 or uniref90 itself` | UniRef members need `fasta: uniref50` (or the same UniRef set). |
| `...members: uniref50 cannot be the members: UniRef50 entries are the clusters themselves` | Use `members: uniref90` or `uniref100`. |
| `...members (...) is not a UniRef set and differs from sources.<name>.fasta. Put that FASTA in fasta and leave members out.` | Your own FASTA goes in `fasta`, without `members`. |
| `...clusters must be uniprot, headers, mmseqs2, or a cluster file ending in .tsv` | Use one of these values. |
| `...clusters: uniprot: UniProt clusters need UniRef data` | Use `fasta: uniref50` with `members: uniref90` or `uniref100`, or set `clusters` to `headers`, `mmseqs2` or a `.tsv` file. |
| `...clusters: headers needs a CL:<cluster id> tag after the id in every FASTA header` | The FASTA has no `CL:` tags; use another `clusters` value. |
| `...identity / coverage only apply when MMseqs2 makes the clusters` | Remove them, or set `clusters: mmseqs2`. |
| `sources.<name>: these members are clustered with MMseqs2, but mmseqs was not found` | `nanoplm deps install --mmseqs-only`, a release binary on `PATH`, or `NANOPLM_MMSEQS=/path/to/mmseqs`. |
| `...val_ratio=... of at most ... clusters (max_seqs) holds out 0; cluster subsampling needs at least one train and one val cluster` | Raise `data_params.val_ratio` to the value the message names, or set `max_seqs: -1`. |
| `... is a cluster-sampled dataset, which needs use_packing=true.` (at pretraining start) | Set `use_packing: true` in `pretrain.yaml`. |
| `... was built by the data-sampler experiment scripts ...` (at pretraining start) | `dataset_dir` is a folder from the old experiment scripts. Rebuild it with `nanoplm data from-yaml`. |
| Warning: `...clustering is now called clusters; the old name still works` | Rename the key. If it was `uniprot` or `mmseqs2`, the rename reruns `cluster_members` once. |
| Warning: `max_seqs=... trains on at most ... clusters` / `val_ratio=... holds out ...% of the clusters` | See [Turn it on](#turn-it-on). |

The `cluster_members` stage checks a cluster file or the `CL:` tags against every record. It stops when a record has no tag or is missing from the file, when an id is in the file but not in the FASTA or is listed twice, or when a cluster has no representative. The message gives the counts and a few ids.

`drop_invalid_sequences` works as in a normal run, on the members. With UniProt clusters, `CL:` tags or a cluster file, the clusters are made before the length filter. If the filter drops a cluster's representative (it is invalid, or shorter than `min_seq_len` or longer than `max_seq_len`), the whole cluster is left out of training. The summary at the end of `from-yaml` says how many clusters were left out this way, and how many of them had members that passed the filter. MMseqs2 only clusters sequences that passed the filter, so it never loses a representative.

## Seeds and shuffle

Two seeds decide what a cluster run sees:

| Seed | Sets |
| --- | --- |
| `data.yaml`'s `shuffle_seed` | Which clusters are held out for validation, and which are kept when `max_seqs` is below the number of clusters. |
| `pretrain.yaml`'s `seed` | The order training visits the clusters in, and which member each cluster gives in each epoch. It also sets model init and masking. |

`shuffle: false` makes training visit the clusters in the order of the cluster index, the same order every epoch. The member each cluster gives is still a random draw. The validation clusters and the `max_seqs` cap stay a random draw with `shuffle_seed` too: a block of clusters in file order would make a biased validation set.

Builds made before `shuffle_seed` reached the cluster split used a fixed seed, 20260918. Set `shuffle_seed: 20260918` to keep their validation clusters. With the default 24, `cluster_index` reruns once and holds out a different set. Changing `shuffle_seed` reruns `cluster_index` and the stages after it: see [Stages and caching](#stages-and-caching).

## Stages and caching

With cluster subsampling, `from-yaml` runs these stages instead of `shuffle` and `split`:

```
get_members_fasta → extract_members → check → filter → cluster_pool
get_cluster_xml / get_cluster_tsv → cluster_members → cluster_index → write_manifest
```

`get_members_fasta` and `extract_members` only appear when `members` differs from `fasta`. Otherwise the normal `get_file` and `extract` stages fetch the source `fasta`. `get_cluster_xml` only appears for UniProt clusters, and `get_cluster_tsv` only for a cluster file given as a URL. A local file has no download stage. You can run a single stage with `from-yaml --target <stage>`.

Each stage reruns only when its own inputs change:

| You change | What reruns |
| --- | --- |
| `clusters`, `identity`, `coverage`, the cluster file | `cluster_members` and later. The tokenized members are kept. |
| `val_ratio`, `val_size`, `max_seqs`, a weight in a mix | `cluster_index` and the manifest (in a mix, only when the source's share of the validation split changes). |
| `shuffle_seed` | `cluster_index` and the manifest. In a mix, every plain source also reruns from `shuffle` on, so its split and tokenization are redone. |
| `max_workers` | Nothing. |
| Remove the block from a `fasta: uniref50`, `members: uniref90` source | The normal pipeline runs, starting with downloading `uniref50.fasta.gz`, which the cluster run never fetched. |

For DVC's cache and disk use, see [Disk use and the DVC cache](/nanoplm/pipeline/data/#disk-use-and-the-dvc-cache).
