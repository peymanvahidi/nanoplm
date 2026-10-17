"""Data prep for cluster-sampled pretraining: UniRef90/UniRef100 pool + UniRef50 index.

The existing pipeline is get_file -> extract -> check -> shuffle -> filter -> split ->
pre-tokenize -> write_manifest. In cluster mode the source FASTA is uniref90 or
uniref100 (detected from the first header) and the stages become:

  get_file / extract            unchanged (FASTA), plus fetch_cluster_xml for uniref50.xml.gz
  check                         unchanged (Checker)
  filter                        unchanged (Filterer) on the *extracted* FASTA, no row cap
  pre-tokenize -> train/        unchanged (ShardWriter): the pool, in FASTA order, no shuffle
  hierarchy                     uniref_hierarchy.extract_hierarchy -> cluster_index/hierarchy/
  index                         cluster_index.build_cluster_index -> cluster_index/
  val gathers -> val/           cluster representatives of the held-out clusters
                                (cluster_index/val_nonrep/: one other member each)
  write_manifest                PretrainManifest with cluster_index_dir set

The data config keeps its meaning where it can: min/max_seq_len filter the pool,
val_ratio is the fraction of clusters held out, seqs_num caps the number of clusters,
samples_per_shard and register_tokens are as before. shuffle / shuffle_seed do not
apply (the pool must keep FASTA order; visit order comes from the training sampler)
and are ignored with a warning. Training seeds the member draw with its own seed.

Every stage is a function of ClusterDataConfig, so a DVC stage can run one of them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np

from nanoplm.data.datasets.cluster_sampled import CLUSTER_INDEX_DIRNAME
from nanoplm.data.infra.manifest import PretrainManifest, write_manifest
from nanoplm.data.preparation import cluster_index as ci
from nanoplm.data.preparation import uniref_hierarchy as uh
from nanoplm.utils import logger

HIERARCHY_DIRNAME = "hierarchy"
VAL_NONREP_DIRNAME = "val_nonrep"


@dataclass
class ClusterDataConfig:
    """Everything cluster-mode prep needs; mirrors the data.yaml fields it will map onto.

    extracted_fasta   uniref90 / uniref100 FASTA (the existing extract stage's output)
    cluster_xml       local path of uniref50.xml(.gz); see cluster_xml_path()
    output_dir        dataset dir: train/ (pool), val/, cluster_index/, .data_manifest
    validation_dir    check stage output (invalid_ids.txt)
    filtered_fasta    filter stage output; its record order is the pool row order
    seqs_num          cap on the number of clusters in the universe (-1: no cap)
    val_ratio         fraction of universe clusters held out for validation
    """

    extracted_fasta: Path
    cluster_xml: Path
    output_dir: Path
    validation_dir: Path
    filtered_fasta: Path
    min_seq_len: int
    max_seq_len: int
    val_ratio: float
    samples_per_shard: int
    register_tokens: int = 0
    seqs_num: int = -1
    drop_invalid_sequences: bool = True
    shuffle: bool = False
    shuffle_seed: Optional[int] = None
    max_workers: int = 8
    holdout_seed: int = ci.HOLDOUT_SEED

    def __post_init__(self):
        for f in ("extracted_fasta", "cluster_xml", "output_dir", "validation_dir",
                  "filtered_fasta"):
            setattr(self, f, Path(getattr(self, f)))

    @property
    def pool_dir(self) -> Path:
        return self.output_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.output_dir / "val"

    @property
    def index_dir(self) -> Path:
        return self.output_dir / CLUSTER_INDEX_DIRNAME

    @property
    def hierarchy_dir(self) -> Path:
        return self.index_dir / HIERARCHY_DIRNAME


def cluster_xml_path(compressed_fasta: Path, source: str) -> Path:
    """Where the XML lives: next to compressed_fasta, under the source's file name."""
    name = Path(urlparse(source).path if "://" in source else source).name
    if not (name.endswith(".xml") or name.endswith(".xml.gz")):
        raise ValueError(f"the cluster source must be a .xml or .xml.gz file, got {source}")
    return Path(compressed_fasta).parent / name


def fetch_cluster_xml(dest: Path, url: Optional[str] = None, local: Optional[str] = None) -> Path:
    """The get_file stage for the XML: download ``url`` or link ``local`` to ``dest``."""
    from nanoplm.data.preparation.downloader import Downloader

    if bool(url) == bool(local):
        raise ValueError("give exactly one of url / local for the cluster XML")
    dest = Path(dest)
    if local:
        Downloader.link_local(Path(local), dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        Downloader(url, dest).download()
    return dest


def pool_level(cfg: ClusterDataConfig) -> str:
    level = ci.detect_uniref_level(cfg.extracted_fasta)
    if level == "uniref50":
        raise ValueError(
            f"{cfg.extracted_fasta} is UniRef50; cluster sampling needs a UniRef90 or "
            "UniRef100 member FASTA. Use the normal pipeline for UniRef50.")
    return level


def warn_ignored_params(cfg: ClusterDataConfig) -> None:
    if cfg.shuffle or cfg.shuffle_seed is not None:
        msg = ("cluster sampling ignores shuffle / shuffle_seed: the pool keeps FASTA order "
               "and the training sampler orders the visits")
        logger.warning(msg)
        warnings.warn(msg, stacklevel=2)


def _tokenizer(cfg: ClusterDataConfig):
    from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
    return ProtModernBertTokenizer(num_register_tokens=cfg.register_tokens)


# --------------------------------------------------------------------- stages


def stage_check(cfg: ClusterDataConfig) -> None:
    from nanoplm.data.preparation.checker import Checker
    Checker(cfg.extracted_fasta, cfg.validation_dir,
            drop_invalid=cfg.drop_invalid_sequences).check()


def stage_filter(cfg: ClusterDataConfig) -> int:
    """Length filter on the extracted FASTA, in place order. seqs_num caps clusters
    later, never rows here: a row prefix would be a cluster-biased subset."""
    from nanoplm.data.preparation.filterer import Filterer
    exclude = cfg.validation_dir / "invalid_ids.txt"
    f = Filterer(cfg.extracted_fasta, cfg.filtered_fasta, cfg.min_seq_len, cfg.max_seq_len,
                 -1, exclude_ids=exclude if exclude.exists() else None)
    f.filter()
    return int(f.num_filtered_seqs or 0)


def stage_pool(cfg: ClusterDataConfig, n_rows: Optional[int] = None) -> None:
    """Pre-tokenize the filtered FASTA into output_dir/train, FASTA order kept."""
    from nanoplm.data.sharding.pretrain_shard_writer import ShardWriter
    if n_rows is not None and -(-n_rows // cfg.samples_per_shard) >= ci.SHARD_SORT_CLIFF:
        raise ValueError(
            f"{n_rows:,} pool rows at samples_per_shard={cfg.samples_per_shard} make "
            f"{ci.SHARD_SORT_CLIFF:,}+ shards, past what the shard file names sort "
            f"correctly for; use samples_per_shard >= {-(-n_rows // (ci.SHARD_SORT_CLIFF - 1))}")
    ShardWriter(str(cfg.filtered_fasta), _tokenizer(cfg), cfg.max_seq_len, str(cfg.pool_dir),
                samples_per_shard=cfg.samples_per_shard, max_workers=cfg.max_workers,
                force=True).create_shards()


def stage_hierarchy(cfg: ClusterDataConfig, level: str) -> dict:
    return uh.extract_hierarchy(cfg.cluster_xml, cfg.hierarchy_dir, level=level)


def stage_index(cfg: ClusterDataConfig, level: str) -> dict:
    return ci.build_cluster_index(
        cfg.filtered_fasta, cfg.pool_dir, cfg.hierarchy_dir, cfg.index_dir, level,
        cfg.max_seq_len, cfg.val_ratio, max_clusters=cfg.seqs_num,
        holdout_seed=cfg.holdout_seed, tokenizer=_tokenizer(cfg))


def stage_val(cfg: ClusterDataConfig) -> tuple[int, int]:
    """val/ = representatives of the held-out clusters (what validation scores), in
    val_rep_rows order; cluster_index/val_nonrep/ = one other member of each held-out
    multi-member cluster, for a per-cluster uniform-member loss."""
    rep = np.load(cfg.index_dir / "val_rep_rows.npy")
    nonrep = np.load(cfg.index_dir / "val_nonrep_rows.npy")
    ci.write_row_gather(cfg.pool_dir, rep, cfg.val_dir, cfg.samples_per_shard)
    if len(nonrep):
        ci.write_row_gather(cfg.pool_dir, nonrep, cfg.index_dir / VAL_NONREP_DIRNAME,
                            cfg.samples_per_shard)
    return len(rep), len(nonrep)


def stage_manifest(cfg: ClusterDataConfig) -> PretrainManifest:
    import json
    from nanoplm.data.datasets.cluster_sampled import INDEX_MANIFEST

    man = json.loads((cfg.index_dir / INDEX_MANIFEST).read_text())
    if man.get("verify", {}).get("ok") is not True:
        raise RuntimeError(f"{cfg.index_dir}: the index is not verified")
    c = man["counts"]
    manifest = PretrainManifest(
        pipeline_mode="pretrain",
        seqs_num=c["clusters_in_universe"],
        min_seq_len=cfg.min_seq_len,
        max_seq_len=cfg.max_seq_len,
        val_ratio=cfg.val_ratio,
        train_dir="train",
        val_dir="val",
        train_sequences=c["train_clusters"],
        val_sequences=c["val_clusters"],
        num_register_tokens=cfg.register_tokens,
        sharded=True,
        samples_per_shard=cfg.samples_per_shard,
        total_train_tokens=c["expected_tokens_per_pass"],
        cluster_index_dir=CLUSTER_INDEX_DIRNAME,
        cluster_level=man["level"],
        pool_sequences=c["pool_rows"],
        pool_tokens=c["pool_tokens"],
    )
    write_manifest(cfg.output_dir, manifest)
    return manifest


def prepare_cluster_dataset(cfg: ClusterDataConfig) -> PretrainManifest:
    """Run every cluster-mode stage after get_file/extract, in order."""
    level = pool_level(cfg)
    warn_ignored_params(cfg)
    logger.info(f"cluster sampling: {level} pool from {cfg.extracted_fasta}")
    stage_check(cfg)
    n_rows = stage_filter(cfg)
    stage_pool(cfg, n_rows)
    stage_hierarchy(cfg, level)
    stage_index(cfg, level)
    stage_val(cfg)
    return stage_manifest(cfg)
