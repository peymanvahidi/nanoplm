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
- Which member a cluster gives depends only on `pretrain.yaml`'s `seed` and the epoch. The number of GPUs, the dataloader workers and resuming do not change it.
- Validation holds out whole clusters and scores each one's representative, the same sequences every epoch. No member of a validation cluster is ever trained on.
- Pretraining needs `use_packing: true`, which is the template default.

## Turn it on

Set `cluster_subsampling: true` under a source in `data.yaml`. nanoPLM then finds where the clusters come from (see [How auto decides](#how-auto-decides)). `false`, or no key, turns it off.

```yaml title="data.yaml"
sources:
  mine:
    fasta: "data/my_proteins.fasta"
    cluster_subsampling: true
```

Write a mapping instead of `true` only when you need to set something. This example trains on UniRef90 sequences under UniProt's UniRef50 clusters:

```yaml title="data.yaml"
sources:
  uniref50:
    fasta: uniref50
    cluster_subsampling:
      members: uniref90  # train on these sequences; leave out to use this fasta
      # clusters: auto   # auto (found from the data), mmseqs2, or a .tsv of representative<TAB>member
```

nanoPLM downloads `uniref90.fasta.gz` and `uniref50.xml.gz`, the file that lists each cluster's members. It never downloads the UniRef50 FASTA itself.

- [`members`](/nanoplm/reference/data/#data-cluster_subsampling-members) names the sequences that training draws from. Leave it out to use the source's own `fasta`. `uniref90` and `uniref100` are shortcuts for UniProt's files. A URL or path must be a UniRef90 or UniRef100 FASTA.
- [`clusters`](/nanoplm/reference/data/#data-cluster_subsampling-clusters) says where the clusters come from. The default, `auto`, finds out from the data. See [Where the clusters come from](#where-the-clusters-come-from). Older configs call it `clustering`; that still works, and `from-yaml` asks you to rename it.
- [`identity`](/nanoplm/reference/data/#data-cluster_subsampling-identity) and [`coverage`](/nanoplm/reference/data/#data-cluster_subsampling-coverage) only apply when MMseqs2 makes the clusters. Setting them to other values with another cluster source is an error.

Here `max_seqs`, `val_ratio` and `val_size` count clusters, not sequences; `max_seqs: -1` uses every cluster. `val_ratio` is set once, in `data_params`; the UniRef90 experiment used `val_ratio: 0.0044` (145k of 32.9M clusters). If you leave the template's values (`max_seqs: 20000`, `val_ratio: 0.1`), training sees at most 20,000 clusters and holds out 10% of them. `from-yaml` warns about both before any stage runs:

```text
Warning: sources.uniref50: max_seqs=20000 (from data_params) trains on at most 20,000 clusters; set sources.uniref50.max_seqs: -1 to use all of them
Warning: sources.uniref50: val_ratio=0.1 (from data_params) holds out about 10.0% of the clusters for validation; the UniRef90 experiment used data_params.val_ratio: 0.0044
```

One `uniref50` source with `val_ratio: 0.0044` holds out the same clusters as the experiment. In a [mix](/nanoplm/guides/mixing/#validation), the validation split is shared out by weight, so UniRef50 gets a different number of clusters and they are a different set.

You do not need to change `samples_per_shard`. If the tokenized members would need more than 9,999 shards, nanoPLM raises it and logs the value it used.

Then set `pretrain.yaml`'s `dataset_dir` to `output_dir`. Nothing else in `pretrain.yaml` changes.

## Where the clusters come from

nanoPLM reads clusters in three formats. If you have none, MMseqs2 makes them.

### UniProt UniRef clusters

For UniRef90 or UniRef100 members, nanoPLM reads UniProt's UniRef50 clusters from `uniref50.xml.gz`:

```yaml title="data.yaml"
sources:
  uniref50:
    fasta: uniref50
    cluster_subsampling:
      members: uniref90
```

### CL tags in the FASTA headers

Each header carries `CL:<cluster id>` as a word after the id, as in Logan's protein files:

```text
>DRR000534_5840_1	CL:DRR000534_5840_1
>SRR10247178_697980_1	CL:DRR000534_5840_1
```

```yaml title="data.yaml"
sources:
  logan:
    fasta: "data/logan_sample.fasta.zst"
    cluster_subsampling: true
```

The representative is the record whose id is its own cluster id. The records can come in any order. Every record needs a tag, and every cluster needs its representative in the file. Files like Logan's "all proteins" (about 31.6 billion records, 99% of them duplicates) are too large for this; use the Logan50 representatives file as a plain source instead, without `cluster_subsampling`.

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
      identity: 0.5    # min sequence identity within a cluster
      coverage: 0.8    # min alignment coverage
```

`cluster_subsampling: true` does the same with these defaults. See [When MMseqs2 runs](#when-mmseqs2-runs) for what it costs.

Whichever source the clusters come from, nanoPLM writes them to one table, `output_dir/cluster_index/members.tsv`, with one `cluster_id<TAB>sequence_id` line per member and the representative first.

## How auto decides

`clusters: auto` takes the first of these that applies:

1. The members are UniRef90 or UniRef100, and `fasta` is UniRef50 or that same set: UniProt's clusters.
2. The first headers of the members carry `CL:` tags: the tags.
3. Otherwise: MMseqs2.

| `fasta` | `members` | `clusters: auto` uses |
| --- | --- | --- |
| `uniref50` | `uniref90` or `uniref100` | UniProt clusters |
| `uniref90` or `uniref100` (or a local copy) | left out | UniProt clusters |
| a FASTA with `CL:` tags | left out | the `CL:` tags |
| your own FASTA, or `swissprot` | left out | MMseqs2 |
| `uniref50` | left out | MMseqs2 over all ~60M UniRef50 sequences (hours; `from-yaml` warns) |
| `trembl`, or any FASTA over 20 GB | left out | `from-yaml` stops: set `clusters: mmseqs2`, or give a cluster file |

`auto` decides before any stage runs. It reads the start of a local file, or of a file already in `output_dir/downloads/`. For a URL it asks the server for the first 1 MB. If that fails, `from-yaml` stops and asks you to set `clusters`. A `.gz` or `.zst` counts as twice its size for the 20 GB limit. On UniProt's own files (`swissprot`, `trembl`, `uniref50`, ...) it sends no request.

To override `auto`:

- `clusters: mmseqs2` uses MMseqs2 in every row, for example to cluster UniRef90 at `identity: 0.3` instead of UniProt's 50%. nanoPLM then downloads only the members, not `uniref50.xml.gz` and not the UniRef50 FASTA.
- `clusters: uniprot` works only in the first two rows. On the others, `from-yaml` stops before any stage runs.
- `clusters: headers` uses the `CL:` tags; `from-yaml` stops if the first headers have none.
- `clusters: <file>.tsv` uses that file, in every row.

UniRef90 and UniRef100 members share one set of clusters and one train / validation split, because a UniRef100 representative is the same sequence as the UniRef90 one.

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

MMseqs2 runs only when you set `clusters: mmseqs2`, or when `auto` finds no clusters in the data (see [How auto decides](#how-auto-decides)).

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
| `sources.<name>.cluster_subsampling is empty. Write cluster_subsampling: true to turn it on` | Write `true`, set a field, or delete the key to turn it off. |
| `sources.<name>.cluster_subsampling: remove enabled` | Delete the `enabled` key. `true` turns it on, `false` turns it off. |
| `sources.<name>.cluster_subsampling.members is uniref90, so sources.<name>.fasta must be UniRef50 or uniref90 itself` | UniRef members need `fasta: uniref50` (or the same UniRef set). |
| `...members: uniref50 cannot be the members: UniRef50 entries are the clusters themselves` | Use `members: uniref90` or `uniref100`. |
| `...members (...) is not a UniRef set and differs from sources.<name>.fasta. Put that FASTA in fasta and leave members out.` | Your own FASTA goes in `fasta`, without `members`. |
| `...clusters must be auto, mmseqs2, uniprot, headers, or a cluster file ending in .tsv` | Use one of these values. |
| `...clusters: uniprot: UniProt clusters need UniRef data` | Use one of the UniProt rows in [How auto decides](#how-auto-decides), or `clusters: auto`. |
| `...clusters: headers needs a CL:<cluster id> tag after the id in every FASTA header` | The FASTA has no `CL:` tags; use another `clusters` value. |
| `...<file> has no cluster information, so clusters: auto would cluster it with MMseqs2: about ... GB of FASTA` | Set `clusters: mmseqs2` to run it anyway, or give a cluster file. |
| `...could not read the start of <url> to find its clusters` | Set `clusters` yourself: `headers`, `mmseqs2` or a `.tsv` file. |
| `...identity / coverage only apply when MMseqs2 makes the clusters` | Remove them, or set `clusters: mmseqs2`. |
| `sources.<name>: these members are clustered with MMseqs2, but mmseqs was not found` | `nanoplm deps install --mmseqs-only`, a release binary on `PATH`, or `NANOPLM_MMSEQS=/path/to/mmseqs`. |
| `...val_ratio=... of at most ... clusters (max_seqs) holds out 0; cluster subsampling needs at least one train and one val cluster` | Raise `data_params.val_ratio` to the value the message names, or set `max_seqs: -1`. |
| `... is a cluster-sampled dataset, which needs use_packing=true.` (at pretraining start) | Set `use_packing: true` in `pretrain.yaml`. |
| `... was built by the data-sampler experiment scripts ...` (at pretraining start) | `dataset_dir` is a folder from the old experiment scripts. Rebuild it with `nanoplm data from-yaml`. |
| Warning: `...clustering is now called clusters; the old name still works` | Rename the key. If it was `uniprot` or `mmseqs2`, the rename reruns `cluster_members` once. |
| Warning: `max_seqs=... trains on at most ... clusters` / `val_ratio=... holds out ...% of the clusters` | See [Turn it on](#turn-it-on). |
| Warning: `fasta is UniRef50 and members is left out, so MMseqs2 will cluster all ~60M UniRef50 sequences` | To use UniProt's clusters, set `members: uniref90`. To cluster UniRef50 on purpose, set `clusters: mmseqs2` and the warning goes away. |
| Warning: `cluster_subsampling ignores shuffle / shuffle_seed` | The tokenized members keep FASTA order; training shuffles the clusters each epoch. |

The `cluster_members` stage checks a cluster file or the `CL:` tags against every record. It stops when a record has no tag or is missing from the file, when an id is in the file but not in the FASTA or is listed twice, or when a cluster has no representative. The message gives the counts and a few ids.

`drop_invalid_sequences` works as in a normal run, on the members. With UniProt clusters, `CL:` tags or a cluster file, the clusters are made before the length filter. If the filter drops a cluster's representative (it is invalid, or shorter than `min_seq_len` or longer than `max_seq_len`), the whole cluster is left out of training. MMseqs2 only clusters sequences that passed the filter, so it never loses a representative.

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
| `clusters`, `identity`, `coverage`, the cluster file | `cluster_members` and later. The tokenized members are kept. A `clusters` value that `auto` resolves to the same source reruns nothing. |
| `val_ratio`, `val_size`, `max_seqs`, a weight in a mix | `cluster_index` and the manifest (in a mix, only when the source's share of the validation split changes). |
| `max_workers` | Nothing. |
| Remove the block from a `fasta: uniref50`, `members: uniref90` source | The normal pipeline runs, starting with downloading `uniref50.fasta.gz`, which the cluster run never fetched. |

For DVC's cache and disk use, see [Disk use and the DVC cache](/nanoplm/pipeline/data/#disk-use-and-the-dvc-cache).
