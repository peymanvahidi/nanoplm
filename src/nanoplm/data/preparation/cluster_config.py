"""Resolve data.yaml's ``cluster_subsampling`` block into one plan.

The one place that decides where the member sequences come from, which cluster source
``clusters: auto`` picks, every derived path, and which settings are errors or are
ignored. Config validation, dvc.yaml generation and the stage commands all use it.

Relative paths are relative to ``base_dir``, the directory of data.yaml: that is
where ``nanoplm data from-yaml`` runs DVC, so the stages see the same files.

Levels: a FASTA is UniRef50 / UniRef90 / UniRef100 by its headers (``>UniRef90_...``)
when the file is on disk, else by its file name (``uniref90.fasta.gz``); anything else
is a custom set. Stage commands re-check the headers when the file exists.

clusters: auto
  UniProt UniRef50 clusters (from uniref50.xml.gz) when the members are a UniRef90 or
  UniRef100 set and url is UniRef50 or that same set. Otherwise MMseqs2 clusters the
  members (easy-linclust, --min-seq-id identity, -c coverage, --cov-mode 0).
clusters: <path>
  a cluster_id<TAB>sequence_id table, used as given.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

UNIREF_BASE = "https://ftp.uniprot.org/pub/databases/uniprot/uniref"
UNIPROT_FASTA = {lvl: f"{UNIREF_BASE}/{lvl}/{lvl}.fasta.gz"
                 for lvl in ("uniref50", "uniref90", "uniref100")}
UNIREF50_XML_URL = f"{UNIREF_BASE}/uniref50/uniref50.xml.gz"
DEFAULT_IDENTITY = 0.5
DEFAULT_COVERAGE = 0.8
INDEX_SUBDIR = "cluster_index"
MEMBERS_TSV = "members.tsv"
ARRAYS_SUBDIR = "index"          # cluster_index/index: the arrays ClusterSampledDataset reads
CLUSTER_INDEX_DIR = f"{INDEX_SUBDIR}/{ARRAYS_SUBDIR}"   # relative to output_dir (manifest)
MMSEQS_TMP = ".mmseqs_tmp"
KEYS = ("enabled", "members", "clusters", "identity", "coverage")
# val_ratio counts clusters here; above this, warn (the template's 0.1 is sized for
# sequences). The experiment held out 0.44% of the UniRef50 clusters.
MAX_QUIET_VAL_RATIO = 0.01
# data.yaml template values: left as they are, shuffle / shuffle_seed draw no warning
TEMPLATE_SHUFFLE = True
TEMPLATE_SHUFFLE_SEED = 24
EXPERIMENT_VAL_RATIO = 0.0044


class ClusterConfigError(ValueError):
    """The cluster_subsampling block cannot be resolved."""


def _name(source: str) -> str:
    return Path(urlparse(source).path if "://" in source else source).name


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _header_level(path: Path) -> Optional[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(path, "rb") as f:
            for line in f:
                if line.startswith(b">"):
                    for lvl in ("uniref100", "uniref90", "uniref50"):
                        if line.startswith(b">UniRef" + lvl[6:].encode() + b"_"):
                            return lvl
                    return None
    except (OSError, EOFError):
        return None
    return None


def on_disk(path, base_dir=None) -> Path:
    """``path`` as the stages see it: relative paths are under ``base_dir`` (data.yaml's
    directory), or the current directory when ``base_dir`` is None."""
    p = Path(path)
    return p if p.is_absolute() or base_dir is None else Path(base_dir) / p


def level_of(source: str, base_dir=None) -> Optional[str]:
    """'uniref50' / 'uniref90' / 'uniref100', or None for any other sequence set."""
    if not is_url(source) and on_disk(source, base_dir).is_file():
        return _header_level(on_disk(source, base_dir))
    m = re.match(r"uniref(50|90|100)\.fasta(\.gz)?$", _name(source).lower())
    return f"uniref{m.group(1)}" if m else None


def _strip_gz(name: str) -> str:
    return name[:-3] if name.endswith(".gz") else name


@dataclass
class ClusterPlan:
    source: str                    # data_dirs url or local_fasta
    source_level: Optional[str]
    members: str                   # URL or path of the member FASTA
    members_level: Optional[str]   # uniref90 / uniref100 / None (custom)
    members_from_source: bool      # the members are the url itself
    members_compressed: Optional[Path]   # where get_members_fasta puts a .gz download
    members_fasta: Path            # the extracted member FASTA the pool is built from
    cluster_source: str            # "uniref_xml" | "mmseqs" | "table"
    xml_path: Optional[Path] = None
    xml_url: Optional[str] = None
    table: Optional[Path] = None
    identity: float = DEFAULT_IDENTITY
    coverage: float = DEFAULT_COVERAGE
    output_dir: Path = Path(".")
    warnings: list = field(default_factory=list)

    @property
    def members_tsv(self) -> Path:
        return self.output_dir / INDEX_SUBDIR / MEMBERS_TSV

    @property
    def index_dir(self) -> Path:
        return self.output_dir / CLUSTER_INDEX_DIR

    @property
    def mmseqs_tmp(self) -> Path:
        return self.output_dir / INDEX_SUBDIR / MMSEQS_TMP

    @property
    def pool_dir(self) -> Path:
        return self.output_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.output_dir / "val"


def _fraction(block: dict, key: str, default: float) -> float:
    v = block.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1:
        raise ClusterConfigError(f"cluster_subsampling.{key} must be a number in (0, 1], "
                                 f"got {v!r}")
    return float(v)


def is_enabled(config: dict) -> bool:
    block = config.get("cluster_subsampling") or {}
    v = block.get("enabled", False)
    if not isinstance(v, bool):
        raise ClusterConfigError(f"cluster_subsampling.enabled must be true or false, got {v!r}")
    return v


def resolve_cluster_plan(config: dict, base_dir=None) -> Optional[ClusterPlan]:
    """The plan for a parsed data.yaml, or None when cluster subsampling is off.

    ``base_dir`` is data.yaml's directory; files are looked up there, but the plan keeps
    every path as written, since the stages run in that directory too.
    """
    block = config.get("cluster_subsampling") or {}
    unknown = sorted(set(block) - set(KEYS))
    if unknown:
        raise ClusterConfigError(f"unknown cluster_subsampling keys: {unknown}")
    if not is_enabled(config):
        return None
    dirs = config.get("data_dirs") or {}
    params = config.get("data_params") or {}
    source = dirs.get("url") or dirs.get("local_fasta")
    if not source:
        raise ClusterConfigError("data_dirs.url or data_dirs.local_fasta must be set")
    compressed = dirs.get("compressed_fasta")
    extracted = dirs.get("extracted_fasta")
    if not extracted or not dirs.get("output_dir"):
        raise ClusterConfigError("data_dirs.extracted_fasta and data_dirs.output_dir are required")
    raw_dir = Path(compressed).parent if compressed else Path(extracted).parent
    identity = _fraction(block, "identity", DEFAULT_IDENTITY)
    coverage = _fraction(block, "coverage", DEFAULT_COVERAGE)
    warnings = []
    seqs_num = params.get("seqs_num", -1)
    if isinstance(seqs_num, int) and seqs_num > 0:
        warnings.append(f"cluster_subsampling: data_params.seqs_num={seqs_num} trains on at "
                        f"most {seqs_num:,} clusters; set it to -1 for all of them")
    val_ratio = params.get("val_ratio")
    if isinstance(val_ratio, (int, float)) and val_ratio > MAX_QUIET_VAL_RATIO:
        warnings.append(f"cluster_subsampling: data_params.val_ratio={val_ratio} holds out "
                        f"{val_ratio:.1%} of the clusters for validation; the UniRef90 "
                        f"experiment used {EXPERIMENT_VAL_RATIO}")
    if (params.get("shuffle", TEMPLATE_SHUFFLE) != TEMPLATE_SHUFFLE
            or params.get("shuffle_seed", TEMPLATE_SHUFFLE_SEED) != TEMPLATE_SHUFFLE_SEED):
        warnings.append("cluster_subsampling ignores data_params.shuffle / shuffle_seed: the "
                        "pool keeps FASTA order and the training sampler orders the visits")

    source_level = level_of(source, base_dir)
    members = block.get("members")
    if members is not None and (not isinstance(members, str) or not members):
        raise ClusterConfigError(f"cluster_subsampling.members must be a string or null, got "
                                 f"{members!r}")
    if members in ("uniref90", "uniref100"):
        members = UNIPROT_FASTA[members]
    if members is None or members == source:
        members, members_level, from_source = source, source_level, True
    else:
        members_level, from_source = level_of(members, base_dir), False
        if members == "uniref50" or members_level == "uniref50":
            raise ClusterConfigError(
                f"cluster_subsampling.members: {members} cannot be the members: UniRef50 "
                "entries are the clusters themselves. Use members: uniref90 or uniref100 to "
                "train on the members of UniProt's UniRef50 clusters.")
        elif members_level is not None and members_level == source_level:
            # e.g. local_fasta is uniref90 and members: uniref90 -> no second download
            members, from_source = source, True
    if source_level == "uniref50" and from_source:
        members_level = None            # UniRef50 itself: no finer UniProt level to sample
    if not from_source:
        if members_level in ("uniref90", "uniref100"):
            if source_level not in ("uniref50", members_level):
                raise ClusterConfigError(
                    f"cluster_subsampling.members is {members_level}, so data_dirs.url must "
                    f"be UniRef50 or {members_level} itself (it is {source}).")
        else:
            raise ClusterConfigError(
                f"cluster_subsampling.members ({members}) is not a UniRef set and differs "
                f"from data_dirs.url. Put that FASTA in data_dirs.url (or local_fasta) and "
                "set cluster_subsampling.members: null.")
    if from_source:
        members_compressed = Path(compressed) if compressed and source.endswith(".gz") else None
        members_fasta = Path(extracted)
    else:
        name = _name(members)
        members_compressed = raw_dir / name if name.endswith(".gz") else None
        members_fasta = raw_dir / _strip_gz(name)

    clusters = block.get("clusters", "auto")
    if clusters in (None, "auto"):
        if members_level in ("uniref90", "uniref100"):
            cluster_source = "uniref_xml"
        else:
            cluster_source = "mmseqs"
    elif isinstance(clusters, str):
        cluster_source = "table"
        if not on_disk(clusters, base_dir).is_file():
            raise ClusterConfigError(f"cluster_subsampling.clusters: {clusters} not found")
    else:
        raise ClusterConfigError(f"cluster_subsampling.clusters must be auto or a path, got "
                                 f"{clusters!r}")
    if cluster_source == "mmseqs" and source_level == "uniref50":
        warnings.append("cluster_subsampling: data_dirs.url is UniRef50 and members is null, "
                        "so MMseqs2 will cluster all ~60M UniRef50 sequences (hours). To "
                        "train on the members of UniProt's UniRef50 clusters instead, set "
                        "cluster_subsampling.members: uniref90")
    if cluster_source != "mmseqs" and (identity != DEFAULT_IDENTITY
                                       or coverage != DEFAULT_COVERAGE):
        what = "UniProt clusters" if cluster_source == "uniref_xml" else "your cluster table"
        warnings.append(f"cluster_subsampling.identity / coverage only apply to MMseqs2 "
                        f"clustering; they are ignored with {what}")
    plan = ClusterPlan(
        source=source, source_level=source_level, members=members,
        members_level=members_level, members_from_source=from_source,
        members_compressed=members_compressed, members_fasta=members_fasta,
        cluster_source=cluster_source, identity=identity, coverage=coverage,
        output_dir=Path(dirs["output_dir"]), warnings=warnings,
        table=Path(clusters) if cluster_source == "table" else None,
    )
    if cluster_source == "uniref_xml":
        plan.xml_url = UNIREF50_XML_URL
        plan.xml_path = raw_dir / _name(UNIREF50_XML_URL)
    return plan


OLD_MARKER = "RUNTIME_SAMPLER_REQUIRED"


def check_not_experiment_dir(dataset_dir: Path, manifest) -> None:
    """Refuse a dataset dir built by the old data-sampler experiment scripts."""
    dataset_dir = Path(dataset_dir)
    msg = (f"{dataset_dir} was built by the data-sampler experiment scripts, whose manifest "
           "describes the whole member pool. Rebuild it with `nanoplm data from-yaml` and "
           "cluster_subsampling.enabled: true.")
    if (dataset_dir / OLD_MARKER).exists():
        raise ClusterConfigError(msg)
    if getattr(manifest, "cluster_index_dir", None):
        return
    train = (dataset_dir / manifest.train_dir).resolve()
    if (train.parent / "index" / "pool_manifest.json").exists():
        raise ClusterConfigError(msg)
