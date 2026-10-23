"""Data prep for cluster subsampling: a member pool, a cluster table, and an index over them.

Stages (``nanoplm data from-yaml`` runs them as DVC stages; see _dvc_yaml_gen):

  get_file / extract     the source FASTA, unchanged (skipped when members come from elsewhere)
  get_members_fasta      members FASTA when it differs from the source (+ extract_members)
  check                  Checker on the member FASTA
  filter                 Filterer on the member FASTA, min/max_seq_len, no row cap, FASTA order
  cluster_pool           pre-tokenize the filtered FASTA into output_dir/train (the pool)
  get_cluster_xml        uniref50.xml.gz (UniProt clusters only)
  cluster_members        cluster_index/members.tsv from the XML or MMseqs2
                         (cluster_table.build_members_table)
  cluster_index          cluster_index/index/ arrays and val/ (held-out reps)
  write_manifest         .data_manifest with cluster_index_dir

val_ratio is the fraction of clusters held out, max_seqs caps the number of clusters,
shuffle / shuffle_seed are ignored. Training seeds the member draw with its own seed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from nanoplm.data.datasets.cluster_sampled import INDEX_MANIFEST
from nanoplm.data.infra.manifest import PretrainManifest, write_manifest
from nanoplm.data.preparation import cluster_config as cc
from nanoplm.data.preparation import cluster_index as ci


def _tokenizer(register_tokens: int):
    from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
    return ProtModernBertTokenizer(num_register_tokens=register_tokens)


def count_fasta_records(fasta: Path) -> int:
    n = 0
    with open(fasta, "rb") as f:
        for line in f:
            n += line[:1] == b">"
    return n


def fetch_cluster_xml(dest: Path, url: Optional[str] = None, local: Optional[str] = None) -> Path:
    """Download ``url`` (atomically) or link ``local`` to ``dest``."""
    from nanoplm.data.preparation.downloader import Downloader

    if bool(url) == bool(local):
        raise ValueError("give exactly one of url / local for the cluster XML")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if local:
        Downloader.link_local(Path(local), dest)
    else:
        Downloader(url, dest).download()
    return dest


def build_pool(filtered_fasta: Path, pool_dir: Path, max_seq_len: int, samples_per_shard: int,
               register_tokens: int = 0, max_workers: int = 8,
               n_rows: Optional[int] = None) -> None:
    """Pre-tokenize the filtered FASTA into the pool, FASTA order kept.

    samples_per_shard is raised when the pool would need too many shards
    (cluster_index.shard_size_for); the manifest records the value used.
    """
    from nanoplm.data.sharding.pretrain_shard_writer import ShardWriter
    samples_per_shard = ci.shard_size_for(
        count_fasta_records(filtered_fasta) if n_rows is None else n_rows,
        samples_per_shard, "cluster pool")
    ShardWriter(str(filtered_fasta), _tokenizer(register_tokens), max_seq_len, str(pool_dir),
                samples_per_shard=samples_per_shard, max_workers=max_workers,
                force=True).create_shards()


def build_index_and_val(filtered_fasta: Path, pool_dir: Path, members_tsv: Path,
                        index_dir: Path, val_dir: Path, max_seq_len: int, val_ratio: float,
                        samples_per_shard: int, seqs_num: int = -1, register_tokens: int = 0,
                        holdout_seed: int = ci.HOLDOUT_SEED) -> dict:
    """The cluster index, then val/ (the held-out clusters' representatives)."""
    info_path = Path(members_tsv).with_suffix(".json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    man = ci.build_cluster_index(
        filtered_fasta, pool_dir, members_tsv, index_dir, max_seq_len, val_ratio,
        max_clusters=seqs_num, holdout_seed=holdout_seed,
        tokenizer=_tokenizer(register_tokens), level=info.get("level"),
        clusters_info={k: info[k] for k in ("source", "release_version", "identity",
                                             "coverage", "cov_mode") if k in info},
        cluster_xml=info.get("input") if info.get("source") == "uniref50_xml" else None)
    rep = np.load(Path(index_dir) / "val_rep_rows.npy")
    ci.write_row_gather(pool_dir, rep, val_dir, samples_per_shard)
    return man


def write_cluster_manifest(output_dir: Path, min_seq_len: int, max_seq_len: int,
                           val_ratio: float, samples_per_shard: int,
                           register_tokens: int = 0) -> PretrainManifest:
    output_dir = Path(output_dir)
    index_dir = output_dir / cc.CLUSTER_INDEX_DIR
    man = json.loads((index_dir / INDEX_MANIFEST).read_text())
    if man.get("verify", {}).get("ok") is not True:
        raise RuntimeError(f"{index_dir}: the index is not verified")
    c = man["counts"]
    manifest = PretrainManifest(
        pipeline_mode="pretrain",
        seqs_num=c["clusters_in_universe"],
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        val_ratio=val_ratio,
        train_dir="train",
        val_dir="val",
        train_sequences=c["train_clusters"],
        val_sequences=c["val_clusters"],
        num_register_tokens=register_tokens,
        sharded=True,
        samples_per_shard=ci.shard_size_for(c["pool_rows"], samples_per_shard, "manifest"),
        total_train_tokens=c["expected_tokens_per_pass"],
        cluster_index_dir=cc.CLUSTER_INDEX_DIR,
        cluster_level=man.get("level") or man.get("clusters", {}).get("source"),
        pool_sequences=c["pool_rows"],
        pool_tokens=c["pool_tokens"],
    )
    write_manifest(output_dir, manifest)
    return manifest
