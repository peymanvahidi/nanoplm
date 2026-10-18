"""Data prep for cluster subsampling: a member pool, a cluster table, and an index over them.

Stages (``nanoplm data from-yaml`` runs them as DVC stages; see _dvc_yaml_gen):

  get_file / extract     the url FASTA, unchanged (skipped when members come from elsewhere)
  get_members_fasta      members FASTA when it differs from url (+ extract_members)
  check                  Checker on the member FASTA
  filter                 Filterer on the member FASTA, min/max_seq_len, no row cap, FASTA order
  cluster_pool           pre-tokenize the filtered FASTA into output_dir/train (the pool)
  get_cluster_xml        uniref50.xml.gz next to compressed_fasta (UniProt clusters only)
  cluster_members        cluster_index/members.tsv from the XML, MMseqs2 or the user table
  cluster_index          cluster_index/index/ arrays, val/ (held-out reps) and
                         cluster_index/index/val_nonrep/
  write_manifest         .data_manifest with cluster_index_dir

val_ratio is the fraction of clusters held out, seqs_num caps the number of clusters,
shuffle / shuffle_seed are ignored. Training seeds the member draw with its own seed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from nanoplm.data.datasets.cluster_sampled import INDEX_MANIFEST
from nanoplm.data.infra.manifest import PretrainManifest, write_manifest
from nanoplm.data.preparation import cluster_config as cc
from nanoplm.data.preparation import cluster_index as ci
from nanoplm.data.preparation import cluster_table as ct
from nanoplm.data.preparation import uniref_hierarchy as uh
from nanoplm.utils import logger

VAL_NONREP_DIRNAME = "val_nonrep"
CLUSTER_SOURCES = ("uniref_xml", "mmseqs", "table")


@dataclass
class ClusterDataConfig:
    """Everything cluster-mode prep needs, resolved (see cluster_config.resolve_cluster_plan).

    members_fasta     extracted member FASTA (UniRef90/100 or any FASTA)
    cluster_source    uniref_xml | mmseqs | table
    level             uniref90 / uniref100 for uniref_xml (checked against the headers)
    """

    members_fasta: Path
    output_dir: Path
    validation_dir: Path
    filtered_fasta: Path
    min_seq_len: int
    max_seq_len: int
    val_ratio: float
    samples_per_shard: int
    cluster_source: str = "uniref_xml"
    level: Optional[str] = None
    cluster_xml: Optional[Path] = None
    cluster_table: Optional[Path] = None
    identity: float = cc.DEFAULT_IDENTITY
    coverage: float = cc.DEFAULT_COVERAGE
    register_tokens: int = 0
    seqs_num: int = -1
    drop_invalid_sequences: bool = True
    max_workers: int = 8
    holdout_seed: int = ci.HOLDOUT_SEED
    mmseqs: Optional[str] = None

    def __post_init__(self):
        if self.cluster_source not in CLUSTER_SOURCES:
            raise ValueError(f"cluster_source must be one of {CLUSTER_SOURCES}")
        for f in ("members_fasta", "output_dir", "validation_dir", "filtered_fasta",
                  "cluster_xml", "cluster_table"):
            v = getattr(self, f)
            if v is not None:
                setattr(self, f, Path(v))

    @property
    def pool_dir(self) -> Path:
        return self.output_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.output_dir / "val"

    @property
    def members_tsv(self) -> Path:
        return self.output_dir / cc.INDEX_SUBDIR / cc.MEMBERS_TSV

    @property
    def index_dir(self) -> Path:
        return self.output_dir / cc.CLUSTER_INDEX_DIR

    @property
    def mmseqs_tmp(self) -> Path:
        return self.output_dir / cc.INDEX_SUBDIR / cc.MMSEQS_TMP


def config_from_yaml(config: dict, base_dir=None) -> tuple[ClusterDataConfig, cc.ClusterPlan]:
    """ClusterDataConfig for a parsed data.yaml with cluster_subsampling enabled.

    ``base_dir`` is data.yaml's directory: relative paths in the config resolve against
    it, as they do for the DVC stages. None keeps them relative to the current directory.
    """
    plan = cc.resolve_cluster_plan(config, base_dir)
    if plan is None:
        raise ValueError("cluster_subsampling is not enabled")
    dirs, p = config["data_dirs"], config.get("data_params") or {}

    def at(path):
        return None if path is None else cc.on_disk(path, base_dir)

    cfg = ClusterDataConfig(
        members_fasta=at(plan.members_fasta), output_dir=at(plan.output_dir),
        validation_dir=at(dirs["validation_dir"]), filtered_fasta=at(dirs["filtered_fasta"]),
        min_seq_len=p["min_seq_len"], max_seq_len=p["max_seq_len"], val_ratio=p["val_ratio"],
        samples_per_shard=p["samples_per_shard"], cluster_source=plan.cluster_source,
        level=plan.members_level, cluster_xml=at(plan.xml_path), cluster_table=at(plan.table),
        identity=plan.identity, coverage=plan.coverage,
        register_tokens=p.get("register_tokens", 0), seqs_num=p.get("seqs_num", -1),
        drop_invalid_sequences=p.get("drop_invalid_sequences", True),
        max_workers=p.get("max_workers") or 8,
    )
    return cfg, plan


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


# --------------------------------------------------------------------- stages


def stage_check(cfg: ClusterDataConfig) -> None:
    from nanoplm.data.preparation.checker import Checker
    Checker(cfg.members_fasta, cfg.validation_dir,
            drop_invalid=cfg.drop_invalid_sequences).check()


def stage_filter(cfg: ClusterDataConfig) -> int:
    """Length filter, file order kept; seqs_num caps clusters later, never rows here."""
    from nanoplm.data.preparation.filterer import Filterer
    exclude = cfg.validation_dir / "invalid_ids.txt"
    f = Filterer(cfg.members_fasta, cfg.filtered_fasta, cfg.min_seq_len, cfg.max_seq_len,
                 -1, exclude_ids=exclude if exclude.exists() else None)
    f.filter()
    return int(f.num_filtered_seqs or 0)


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


def stage_pool(cfg: ClusterDataConfig, n_rows: Optional[int] = None) -> None:
    build_pool(cfg.filtered_fasta, cfg.pool_dir, cfg.max_seq_len, cfg.samples_per_shard,
               cfg.register_tokens, cfg.max_workers, n_rows)


def build_members_table(source: str, out_tsv: Path, *, fasta: Path, level: Optional[str] = None,
                        xml: Optional[Path] = None, table: Optional[Path] = None,
                        identity: float = cc.DEFAULT_IDENTITY,
                        coverage: float = cc.DEFAULT_COVERAGE, tmp_dir: Optional[Path] = None,
                        mmseqs: Optional[str] = None, threads: Optional[int] = None) -> dict:
    """members.tsv from one of the three sources. ``fasta`` is the member FASTA
    (filtered for mmseqs / table; any of them for the uniref_xml level check)."""
    out_tsv = Path(out_tsv)
    if source == "uniref_xml":
        got = ci.detect_uniref_level(Path(fasta))
        if got != level:
            raise ValueError(f"{fasta} holds {got} sequences, but the plan expects {level}; "
                             "check cluster_subsampling.members and data_dirs.url")
        stats = uh.xml_to_members_tsv(xml, out_tsv, level)
    elif source == "mmseqs":
        stats = ct.mmseqs_members_tsv(fasta, out_tsv, identity, coverage,
                                      tmp_dir or out_tsv.parent / cc.MMSEQS_TMP,
                                      threads=threads, mmseqs=mmseqs)
    elif source == "table":
        stats = ct.table_members_tsv(table, fasta, out_tsv)
    else:
        raise ValueError(f"unknown cluster source {source!r}")
    with open(out_tsv.with_suffix(".json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def stage_members(cfg: ClusterDataConfig) -> dict:
    fasta = cfg.filtered_fasta if cfg.cluster_source != "uniref_xml" else cfg.members_fasta
    return build_members_table(cfg.cluster_source, cfg.members_tsv, fasta=fasta, level=cfg.level,
                               xml=cfg.cluster_xml, table=cfg.cluster_table,
                               identity=cfg.identity, coverage=cfg.coverage,
                               tmp_dir=cfg.mmseqs_tmp, mmseqs=cfg.mmseqs,
                               threads=cfg.max_workers)


def build_index_and_val(filtered_fasta: Path, pool_dir: Path, members_tsv: Path,
                        index_dir: Path, val_dir: Path, max_seq_len: int, val_ratio: float,
                        samples_per_shard: int, seqs_num: int = -1, register_tokens: int = 0,
                        holdout_seed: int = ci.HOLDOUT_SEED) -> dict:
    """The cluster index, then val/ (held-out reps) and index/val_nonrep/."""
    info_path = Path(members_tsv).with_suffix(".json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    man = ci.build_cluster_index(
        filtered_fasta, pool_dir, members_tsv, index_dir, max_seq_len, val_ratio,
        max_clusters=seqs_num, holdout_seed=holdout_seed,
        tokenizer=_tokenizer(register_tokens), level=info.get("level"),
        clusters_info={k: info[k] for k in ("source", "release_version", "identity",
                                             "coverage", "cov_mode") if k in info})
    rep = np.load(Path(index_dir) / "val_rep_rows.npy")
    nonrep = np.load(Path(index_dir) / "val_nonrep_rows.npy")
    ci.write_row_gather(pool_dir, rep, val_dir, samples_per_shard)
    if len(nonrep):
        ci.write_row_gather(pool_dir, nonrep, Path(index_dir) / VAL_NONREP_DIRNAME,
                            samples_per_shard)
    return man


def stage_index(cfg: ClusterDataConfig) -> dict:
    return build_index_and_val(cfg.filtered_fasta, cfg.pool_dir, cfg.members_tsv, cfg.index_dir,
                               cfg.val_dir, cfg.max_seq_len, cfg.val_ratio,
                               cfg.samples_per_shard, cfg.seqs_num, cfg.register_tokens,
                               cfg.holdout_seed)


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


def stage_manifest(cfg: ClusterDataConfig) -> PretrainManifest:
    return write_cluster_manifest(cfg.output_dir, cfg.min_seq_len, cfg.max_seq_len,
                                  cfg.val_ratio, cfg.samples_per_shard, cfg.register_tokens)


def prepare_cluster_dataset(cfg: ClusterDataConfig) -> PretrainManifest:
    """Every stage after the member FASTA (and XML) are on disk, in order."""
    logger.info(f"cluster subsampling: {cfg.cluster_source} clusters over {cfg.members_fasta}")
    stage_check(cfg)
    n_rows = stage_filter(cfg)
    stage_pool(cfg, n_rows)
    stage_members(cfg)
    stage_index(cfg)
    return stage_manifest(cfg)
