"""Resolve a source's ``cluster_subsampling`` block into one plan.

The one place that decides where the member sequences come from, which cluster source
is used, every derived path, and which settings are errors or are ignored. Config
validation, dvc.yaml generation and the stage commands all use it. The block sits under
a source (data_config.Source); being there turns cluster subsampling on.

Relative paths are relative to ``base_dir``, the directory of data.yaml: that is
where ``nanoplm data from-yaml`` runs DVC, so the stages see the same files.

Levels: a FASTA is UniRef50 / UniRef90 / UniRef100 by its headers (``>UniRef90_...``)
when the file is on disk, else by its file name (``uniref90.fasta.gz``); anything else
is a custom set. Stage commands re-check the headers when the file exists.

Cluster source (``clustering``): ``auto`` uses UniProt UniRef50 clusters (from
uniref50.xml.gz) when the members are a UniRef90 or UniRef100 set and the source fasta is
UniRef50 or that same set, and MMseqs2 otherwise. ``uniprot`` requires that UniRef case;
``mmseqs2`` always clusters the members with MMseqs2 (easy-linclust, --min-seq-id
identity, -c coverage, --cov-mode 0).
"""
from __future__ import annotations

import contextlib
import gzip
import io
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nanoplm.data.preparation.data_config import (fasta_stem, file_name as _name, is_compressed,
                                                  is_fasta_name, is_url, strip_compressed)
from nanoplm.data.preparation.extractor import ZSTD_MAX_WINDOW, ZSTD_MISSING

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
CLUSTERINGS = ("auto", "uniprot", "mmseqs2")
UNIPROT_NEEDS = ("UniProt clusters need UniRef data (fasta UniRef50 + members uniref90/"
                 "uniref100, or a UniRef90/100 fasta)")
# val_ratio counts clusters here; above this, warn (the template's 0.1 is sized for
# sequences). The experiment held out 0.44% of the UniRef50 clusters.
MAX_QUIET_VAL_RATIO = 0.01
# data.yaml template values: left as they are, shuffle / shuffle_seed draw no warning
TEMPLATE_SHUFFLE = True
TEMPLATE_SHUFFLE_SEED = 24
EXPERIMENT_VAL_RATIO = 0.0044


class ClusterConfigError(ValueError):
    """A cluster_subsampling block cannot be resolved."""


@contextlib.contextmanager
def _open_fasta(path: Path):
    """A FASTA for reading in binary, decompressed when it is .gz or .zst. A .zst is read
    with the zstandard package, else with the zstd tool; a bad .zst raises OSError."""
    if not str(path).endswith(".zst"):
        with (gzip.open if str(path).endswith(".gz") else open)(path, "rb") as f:
            yield f
        return
    try:
        import zstandard
    except ImportError:
        zstandard = None
    if zstandard is not None:
        # the extractor's window limit: a --long=31 file needs more than the 128 MiB default
        dctx = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW)
        with io.BufferedReader(zstandard.open(path, "rb", dctx=dctx)) as f:
            try:
                yield f
            except zstandard.ZstdError as e:
                raise OSError(f"{path}: {e}") from e
        return
    zstd = shutil.which("zstd")
    if zstd is None:
        raise ClusterConfigError(f"Reading the headers of {path} {ZSTD_MISSING}")
    proc = subprocess.Popen([zstd, "-dcq", "--memory=2048MB", str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        yield proc.stdout
    finally:
        proc.kill()
        proc.stdout.close()
        proc.wait()


def _header_level(path: Path) -> Optional[str]:
    try:
        with _open_fasta(path) as f:
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
    name = _name(source).lower()
    stem = fasta_stem(name) if is_fasta_name(name) else None
    return stem if stem in ("uniref50", "uniref90", "uniref100") else None


@dataclass
class ClusterPlan:
    source: str                    # the source's fasta (URL or path)
    source_level: Optional[str]
    members: str                   # URL or path of the member FASTA
    members_level: Optional[str]   # uniref90 / uniref100 / None (custom)
    members_from_source: bool      # the members are the source fasta itself
    members_compressed: Optional[Path]   # the members .gz: a download, or a local file
    members_fasta: Path            # the extracted member FASTA the pool is built from
    cluster_source: str            # "uniref_xml" | "mmseqs"
    clustering: str = "auto"       # as written in data.yaml: auto | uniprot | mmseqs2
    xml_path: Optional[Path] = None
    xml_url: Optional[str] = None
    identity: float = DEFAULT_IDENTITY
    coverage: float = DEFAULT_COVERAGE
    output_dir: Path = Path(".")
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)    # informational, printed before the stages

    def source_line(self) -> str:
        """What clusters the members, e.g. 'UniProt uniref50.xml.gz (members: UniRef90)' or
        'MMseqs2 linclust, identity 0.5, coverage 0.8'."""
        if self.cluster_source == "uniref_xml":
            return (f"UniProt {_name(self.xml_url)} "
                    f"(members: UniRef{self.members_level[len('uniref'):]})")
        # the members file is already in the table's fasta cell
        return f"MMseqs2 linclust, identity {self.identity:g}, coverage {self.coverage:g}"

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


def _fraction(block: dict, key: str, default: float, where: str) -> float:
    v = block.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1:
        raise ClusterConfigError(f"{where}.{key} must be a number in (0, 1], got {v!r}")
    return float(v)


def _setting(src, name: str) -> str:
    """``val_ratio=0.1``, plus where it comes from when that is not the source itself."""
    key = src.key(name)
    text = f"{name}={src.params.get(name)}"
    return text if key.startswith("sources.") else f"{text} (from {key.rsplit('.', 1)[0]})"


def _members_paths(members: str, src) -> tuple[Optional[Path], Path]:
    """(compressed, extracted) for a members FASTA that is not the source fasta: a URL
    is downloaded into downloads/, a local file is read in place."""
    name = _name(members)
    # a mix extracts into downloads/ (data_config._source_paths)
    unpacked = Path(src.downloads_dir if src.mixed else src.work_dir)
    if is_url(members):
        got = Path(src.download_path(members))
        return (got, unpacked / strip_compressed(name)) if is_compressed(name) else (None, got)
    if is_compressed(name):
        return Path(members), unpacked / strip_compressed(name)
    return None, Path(members)


def resolve_cluster_plan(src, base_dir=None) -> Optional[ClusterPlan]:
    """The plan for one source (data_config.Source), or None without a cluster block.

    ``base_dir`` is data.yaml's directory; files are looked up there, but the plan keeps
    every path as written, since the stages run in that directory too.
    """
    block = src.cluster
    if block is None:
        return None
    where = f"sources.{src.name}.cluster_subsampling"
    params = src.params
    source = src.fasta
    compressed = src.paths["compressed_fasta"]
    extracted = src.paths["extracted_fasta"]
    clustering = block.get("clustering", "auto")
    if clustering not in CLUSTERINGS:
        raise ClusterConfigError(f"{where}.clustering must be one of "
                                 f"{', '.join(CLUSTERINGS)}, got {clustering!r}")
    identity = _fraction(block, "identity", DEFAULT_IDENTITY, where)
    coverage = _fraction(block, "coverage", DEFAULT_COVERAGE, where)
    warnings = []
    # every warning names the source, and a fix is set on that source alone
    on = f"sources.{src.name}"
    max_seqs = params.get("max_seqs", -1)
    if isinstance(max_seqs, int) and max_seqs > 0:
        warnings.append(f"{on}: {_setting(src, 'max_seqs')} trains on at most {max_seqs:,} "
                        f"clusters; set {on}.max_seqs: -1 to use all of them")
    if (params.get("shuffle", TEMPLATE_SHUFFLE) != TEMPLATE_SHUFFLE
            or params.get("shuffle_seed", TEMPLATE_SHUFFLE_SEED) != TEMPLATE_SHUFFLE_SEED):
        warnings.append(f"{on}: cluster_subsampling ignores shuffle / shuffle_seed: the pool "
                        "keeps FASTA order and the training sampler orders the visits")

    source_level = level_of(source, base_dir)
    members = block.get("members")
    if members is not None and (not isinstance(members, str) or not members):
        raise ClusterConfigError(f"{where}.members must be a string or null, got {members!r}")
    if members in ("uniref90", "uniref100"):
        members = UNIPROT_FASTA[members]
    if members is None or members == source:
        members, members_level, from_source = source, source_level, True
    else:
        if (members != "uniref50" and not is_url(members)
                and not on_disk(members, base_dir).is_file()):
            raise ClusterConfigError(
                f"{where}.members: {members} not found (a relative path is relative to "
                "data.yaml's directory)")
        members_level, from_source = level_of(members, base_dir), False
        if members == "uniref50" or members_level == "uniref50":
            raise ClusterConfigError(
                f"{where}.members: {members} cannot be the members: UniRef50 entries are "
                "the clusters themselves. Use members: uniref90 or uniref100 to train on the "
                "members of UniProt's UniRef50 clusters.")
        elif members_level is not None and members_level == source_level:
            # e.g. fasta is a local uniref90 and members: uniref90 -> no second download
            members, from_source = source, True
    if source_level == "uniref50" and from_source:
        members_level = None            # UniRef50 itself: no finer UniProt level to sample
    if not from_source:
        if members_level in ("uniref90", "uniref100"):
            if source_level not in ("uniref50", members_level):
                raise ClusterConfigError(
                    f"{where}.members is {members_level}, so sources.{src.name}.fasta must "
                    f"be UniRef50 or {members_level} itself (it is {source}).")
        else:
            raise ClusterConfigError(
                f"{where}.members ({members}) is not a UniRef set and differs from "
                f"sources.{src.name}.fasta. Put that FASTA in fasta and set members: null.")
    if from_source:
        members_compressed = Path(compressed) if compressed else None
        members_fasta = Path(extracted)
    else:
        members_compressed, members_fasta = _members_paths(members, src)

    uniref = members_level in ("uniref90", "uniref100")
    if clustering == "uniprot" and not uniref:
        raise ClusterConfigError(
            f"{where}.clustering: uniprot: {UNIPROT_NEEDS}; got fasta {source} and members "
            f"{block.get('members') or 'null'}. Use clustering: auto or mmseqs2 to cluster these "
            "sequences with MMseqs2.")
    cluster_source = "uniref_xml" if uniref and clustering != "mmseqs2" else "mmseqs"
    notes = []
    val_ratio = params.get("val_ratio")
    if isinstance(val_ratio, (int, float)) and val_ratio > MAX_QUIET_VAL_RATIO:
        # the experiment's holdout is a share of UniProt's UniRef50 clusters
        # (in a mix, val_ratio is a share of the whole pool, which counts these clusters)
        fix = (f"the UniRef90 experiment used data_params.val_ratio: {EXPERIMENT_VAL_RATIO}"
               if cluster_source == "uniref_xml"
               else "set data_params.val_ratio lower to train on more of them")
        warnings.append(f"{on}: {_setting(src, 'val_ratio')} holds out about {val_ratio:.1%} of "
                        f"the clusters for validation; {fix}")
    if cluster_source == "mmseqs" and source_level == "uniref50" and clustering == "auto":
        warnings.append(f"{on}: fasta is UniRef50 and members is null, so MMseqs2 will "
                        "cluster all ~60M UniRef50 sequences (hours). To train on the members "
                        "of UniProt's UniRef50 clusters instead, set "
                        f"{on}.cluster_subsampling.members: uniref90 (with clustering: auto or "
                        "uniprot)")
    if cluster_source == "uniref_xml" and (identity != DEFAULT_IDENTITY
                                           or coverage != DEFAULT_COVERAGE):
        warnings.append(f"{on}: cluster_subsampling.identity / coverage only apply to MMseqs2 "
                        "clustering; they are ignored with UniProt clusters. Set "
                        f"{on}.cluster_subsampling.clustering: mmseqs2 to use them")
    if cluster_source == "mmseqs" and not from_source:
        notes.append(f"{on}: clustering: mmseqs2 clusters the members ({members}); the source "
                     f"fasta ({source}) is not needed and is not downloaded")
    plan = ClusterPlan(
        source=source, source_level=source_level, members=members,
        members_level=members_level, members_from_source=from_source,
        members_compressed=members_compressed, members_fasta=members_fasta,
        cluster_source=cluster_source, clustering=clustering, identity=identity,
        coverage=coverage, output_dir=Path(src.data_dir), warnings=warnings, notes=notes,
    )
    if cluster_source == "uniref_xml":
        plan.xml_url = UNIREF50_XML_URL
        plan.xml_path = Path(src.download_path(UNIREF50_XML_URL))
    return plan


OLD_MARKER = "RUNTIME_SAMPLER_REQUIRED"


def check_not_experiment_dir(dataset_dir: Path, manifest) -> None:
    """Refuse a dataset dir built by the old data-sampler experiment scripts."""
    dataset_dir = Path(dataset_dir)
    msg = (f"{dataset_dir} was built by the data-sampler experiment scripts, whose manifest "
           "describes the whole member pool. Rebuild it with `nanoplm data from-yaml` and a "
           "cluster_subsampling block on the source.")
    if (dataset_dir / OLD_MARKER).exists():
        raise ClusterConfigError(msg)
    if getattr(manifest, "cluster_index_dir", None):
        return
    train = (dataset_dir / manifest.train_dir).resolve()
    if (train.parent / "index" / "pool_manifest.json").exists():
        raise ClusterConfigError(msg)
