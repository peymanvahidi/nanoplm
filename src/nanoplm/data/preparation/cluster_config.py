"""Resolve a source's ``cluster_subsampling`` block into one plan.

The one place that decides where the member sequences come from, which cluster source
is used, every derived path, and which settings are errors or are ignored. Config
validation, dvc.yaml generation and the stage commands all use it. The block sits under
a source (data_config.Source) and must set ``clusters``.

Relative paths are relative to ``base_dir``, the directory of data.yaml: that is
where ``nanoplm data from-yaml`` runs DVC, so the stages see the same files.

Levels: a FASTA is UniRef50 / UniRef90 / UniRef100 by its headers (``>UniRef90_...``)
when the file is on disk, else by its file name (``uniref90.fasta.gz``); anything else
is a custom set. Stage commands re-check the headers when the file exists.

Cluster source (``clusters``, once ``clustering``), required:

  uniprot   UniProt UniRef50 clusters (from uniref50.xml.gz); the members must be a
            UniRef90 or UniRef100 set and the source fasta UniRef50 or that same set.
  headers   the CL:<cluster id> tags (cluster_import); the first headers must carry them.
            They are read from the file on disk, or with a small HTTP range request of
            the URL's start.
  mmseqs2   clusters the members with MMseqs2 (easy-linclust, --min-seq-id identity,
            -c coverage, --cov-mode 0).
  a .tsv    a representative<TAB>member file (cluster_import), a path or a URL that is
            downloaded; .tsv.gz and .tsv.zst are read as they are.

The stage that writes members.tsv checks the whole file.
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nanoplm.data.preparation.cluster_import import CL_TAG, DecompressorMissing, open_decompressed
from nanoplm.data.preparation.data_config import (CLUSTERS_CHOICES, FASTA_SHORTCUTS,
                                                  fasta_stem, file_name as _name,
                                                  is_compressed, is_fasta_name, is_url,
                                                  strip_compressed)
from nanoplm.data.preparation.extractor import ZSTD_MAX_WINDOW

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
CLUSTERS = ("uniprot", "headers", "mmseqs2")      # or a cluster file, TSV_SUFFIXES
TSV_SUFFIXES = (".tsv", ".tsv.gz", ".tsv.zst")
UNIPROT_NEEDS = ("UniProt clusters need UniRef data (fasta UniRef50 + members uniref90/"
                 "uniref100, or a UniRef90/100 fasta)")
# UniProt's FASTA shortcuts: their headers carry no CL: tag, so clusters: headers is
# refused on them without a request
UNIPROT_FASTAS = frozenset(FASTA_SHORTCUTS[k] for k in
                           ("swissprot", "trembl", "uniref50", "uniref90", "uniref100"))
PEEK_BYTES = 1 << 20           # decompressed bytes read for the first headers
PEEK_FETCH = 1 << 20           # bytes asked of a server (a range request)
PEEK_TIMEOUT = 30
# val_ratio counts clusters here; above this, warn (the template's 0.1 is sized for
# sequences). The experiment held out 0.44% of the UniRef50 clusters.
MAX_QUIET_VAL_RATIO = 0.01
# data.yaml template values: left as they are, shuffle / shuffle_seed draw no warning
TEMPLATE_SHUFFLE = True
TEMPLATE_SHUFFLE_SEED = 24
EXPERIMENT_VAL_RATIO = 0.0044


class ClusterConfigError(ValueError):
    """A cluster_subsampling block cannot be resolved."""


def _header_level(path: Path) -> Optional[str]:
    try:
        with open_decompressed(path) as f:
            for line in f:
                if line.startswith(b">"):
                    for lvl in ("uniref100", "uniref90", "uniref50"):
                        if line.startswith(b">UniRef" + lvl[6:].encode() + b"_"):
                            return lvl
                    return None
    except DecompressorMissing as e:
        raise ClusterConfigError(str(e)) from None
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


_URL_PEEKS: dict = {}           # url -> its first headers, or the reason they could not be read


def _headers(data: bytes, whole: bool) -> list:
    lines = data.split(b"\n")
    if not whole:
        lines = lines[:-1]                  # the last line may be cut short
    return [line[1:].rstrip(b"\r") for line in lines if line.startswith(b">")]


def _peek_file(path: Path) -> list:
    """The first headers of a FASTA file (without '>')."""
    try:
        with open_decompressed(path) as f:
            data = f.read(PEEK_BYTES)
    except OSError:
        raise
    except Exception as e:              # noqa: BLE001  a damaged .gz / .zst
        raise OSError(f"{path}: {type(e).__name__}: {e}") from None
    return _headers(data, len(data) < PEEK_BYTES)


def _decompress_start(raw: bytes, name: str) -> bytes:
    """The start of a file decompressed from its first bytes (.gz and .zst both decode
    from the start of a file)."""
    if name.endswith(".gz"):
        return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw, PEEK_BYTES)
    if name.endswith(".zst"):
        try:
            import zstandard
        except ImportError:
            zstd = shutil.which("zstd")
            if zstd is None:
                raise OSError("needs the zstandard package or the zstd tool") from None
            return subprocess.run([zstd, "-dcq", "--memory=2048MB"], input=raw,
                                  capture_output=True).stdout[:PEEK_BYTES]
        dctx = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW)
        return dctx.decompressobj().decompress(raw)[:PEEK_BYTES]
    return raw[:PEEK_BYTES]


def _http_start(url: str) -> tuple:
    """(the first PEEK_FETCH bytes of ``url``, its total size or None): one range request;
    a server that sends the whole file is cut off after those bytes."""
    from nanoplm.data.preparation.downloader import Downloader
    context = Downloader(url, "")._build_ssl_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{PEEK_FETCH - 1}"})
    with opener.open(req, timeout=PEEK_TIMEOUT) as r:
        got = r.headers.get("Content-Range") or ""
        total = got.rsplit("/", 1)[-1] if got.startswith("bytes") else (
            r.headers.get("Content-Length") if getattr(r, "status", None) in (None, 200)
            else None)
        raw = r.read(PEEK_FETCH)
    return raw, int(total) if total and str(total).isdigit() else None


def peek_url(url: str) -> list:
    """The first headers of a FASTA URL, read once per process. Raises OSError (with the
    reason) when the URL cannot be read."""
    if url not in _URL_PEEKS:
        try:
            raw, total = _http_start(url)
            data = _decompress_start(raw, _name(url))
            whole = total is not None and len(raw) >= total
            _URL_PEEKS[url] = _headers(data, whole)
        except Exception as e:          # noqa: BLE001  any failure: say why
            _URL_PEEKS[url] = f"{type(e).__name__}: {e}"
    got = _URL_PEEKS[url]
    if isinstance(got, str):
        raise OSError(got)
    return got


def _peek_members(members: str, compressed: Optional[Path], extracted: Path,
                  base_dir) -> list:
    """The first headers of the members FASTA. A copy already on disk is read (the
    extracted FASTA, else the download) and a URL only when there is none, so the check
    is the same before and after the download."""
    got = on_disk(compressed, base_dir) if compressed else None
    for path in (on_disk(extracted, base_dir), got):
        if path is not None and path.is_file():
            return _peek_file(path)
    if is_url(members):
        return peek_url(members)
    raise OSError(f"{members} not found (a relative path is relative to data.yaml's "
                  "directory)")


def cl_tag(header: bytes) -> Optional[bytes]:
    """The cluster id of a header's CL:<cluster id> token (after the id), or None."""
    for token in header.split()[1:]:
        if token.startswith(CL_TAG) and len(token) > len(CL_TAG):
            return token[len(CL_TAG):]
    return None


@dataclass
class ClusterPlan:
    source: str                    # the source's fasta (URL or path)
    source_level: Optional[str]
    members: str                   # URL or path of the member FASTA
    members_level: Optional[str]   # uniref90 / uniref100 / None (custom)
    members_from_source: bool      # the members are the source fasta itself
    members_compressed: Optional[Path]   # the members .gz: a download, or a local file
    members_fasta: Path            # the extracted member FASTA the pool is built from
    cluster_source: str            # "uniref_xml" | "mmseqs" | "headers" | "tsv"
    clusters: str = ""             # as written in data.yaml: uniprot | headers | mmseqs2 |
                                   # a .tsv
    clusters_key: Optional[str] = None   # where data.yaml sets it (clusters or clustering)
    xml_path: Optional[Path] = None
    xml_url: Optional[str] = None
    tsv_path: Optional[Path] = None      # the cluster file: a local path, or the download
    tsv_url: Optional[str] = None
    identity: float = DEFAULT_IDENTITY
    coverage: float = DEFAULT_COVERAGE
    output_dir: Path = Path(".")
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)    # informational, printed before the stages

    def clusters_text(self) -> str:
        """Where the clusters come from: 'UniProt uniref50.xml.gz', 'CL: tags in headers',
        the cluster file, or 'MMseqs2 (50% identity, 80% coverage)'."""
        if self.cluster_source == "uniref_xml":
            return f"UniProt {_name(self.xml_url)}"
        if self.cluster_source == "headers":
            return "CL: tags in headers"
        if self.cluster_source == "tsv":
            return self.clusters if not self.tsv_url else _name(self.tsv_url)
        return (f"MMseqs2 ({self.identity * 100:g}% identity, "
                f"{self.coverage * 100:g}% coverage)")

    def trains_on(self) -> str:
        """'this fasta', or 'UniRef90 members' when the members are another file."""
        if self.members_from_source:
            return "this fasta"
        if self.members_level:
            return f"UniRef{self.members_level[len('uniref'):]} members"
        return _name(self.members)

    def source_line(self) -> str:
        """'clusters: UniProt uniref50.xml.gz | trains on: UniRef90 members'."""
        return f"clusters: {self.clusters_text()} | trains on: {self.trains_on()}"

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


def is_tsv_name(value: str) -> bool:
    """Whether a URL or path names a cluster file: .tsv, .tsv.gz or .tsv.zst."""
    return _name(value).endswith(TSV_SUFFIXES)


def _clusters_setting(value, key: str, base_dir) -> str:
    """The clusters value, checked: one of CLUSTERS or a cluster file."""
    if value is None:
        raise ClusterConfigError(f"{key} is required: {CLUSTERS_CHOICES}")
    if isinstance(value, str) and value in CLUSTERS:
        return value
    if isinstance(value, str) and value and is_tsv_name(value):
        if not is_url(value) and not on_disk(value, base_dir).is_file():
            raise ClusterConfigError(f"{key}: {value} not found (a relative path is relative to "
                                     "data.yaml's directory)")
        return value
    raise ClusterConfigError(
        f"{key} must be uniprot, headers, mmseqs2, or a cluster file ending in .tsv "
        f"(.tsv.gz, .tsv.zst) with representative<TAB>member lines, got {value!r}")


def resolve_cluster_plan(src, base_dir=None) -> Optional[ClusterPlan]:
    """The plan for one source (data_config.Source), or None without a cluster block.

    ``base_dir`` is data.yaml's directory; files are looked up there, but the plan keeps
    every path as written, since the stages run in that directory too.
    """
    block = src.cluster
    if block is None:
        return None
    where = f"sources.{src.name}.cluster_subsampling"
    written = getattr(src, "cluster_keys", None) or {}
    ckey = written.get("clusters", f"{where}.clusters")
    params = src.params
    source = src.fasta
    compressed = src.paths["compressed_fasta"]
    extracted = src.paths["extracted_fasta"]
    clusters = _clusters_setting(block.get("clusters"), ckey, base_dir)
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
                f"sources.{src.name}.fasta. Put that FASTA in fasta and leave members out.")
    if from_source:
        members_compressed = Path(compressed) if compressed else None
        members_fasta = Path(extracted)
    else:
        members_compressed, members_fasta = _members_paths(members, src)

    uniref = members_level in ("uniref90", "uniref100")
    if clusters == "uniprot" and not uniref:
        raise ClusterConfigError(
            f"{ckey}: uniprot: {UNIPROT_NEEDS}; got fasta {source} and members "
            f"{block.get('members') or 'null'}. Use clusters: headers, mmseqs2 or a .tsv "
            "cluster file for these sequences.")

    name = _name(members)
    notes = []
    if clusters not in CLUSTERS:
        cluster_source = "tsv"
    elif clusters == "uniprot":
        cluster_source = "uniref_xml"
    elif clusters == "mmseqs2":
        cluster_source = "mmseqs"
    else:
        cluster_source = "headers"
        if members in UNIPROT_FASTAS:
            raise ClusterConfigError(f"{ckey}: headers needs a CL:<cluster id> tag after the id "
                                     f"in every FASTA header; UniProt's {name} has none")
        try:
            got = _peek_members(members, members_compressed, members_fasta, base_dir)
        except OSError:
            got = None                  # the stage checks every header
        if got and not all(cl_tag(h) for h in got):
            first = next(h for h in got if not cl_tag(h))
            raise ClusterConfigError(
                f"{ckey}: headers needs a CL:<cluster id> tag after the id in every FASTA "
                f"header; {sum(not cl_tag(h) for h in got):,} of the first {len(got):,} "
                f"headers of {name} have none (e.g. >{first[:80].decode(errors='replace')})")

    if cluster_source != "mmseqs" and (identity != DEFAULT_IDENTITY
                                       or coverage != DEFAULT_COVERAGE):
        found = {"uniref_xml": "UniProt's clusters", "headers": f"the CL: tags of {name}",
                 "tsv": clusters}[cluster_source]
        raise ClusterConfigError(
            f"{where}: identity / coverage only apply when MMseqs2 makes the clusters, and "
            f"these come from {found}. Remove identity / coverage, or set "
            f"{where}.clusters: mmseqs2 to cluster with MMseqs2")
    val_ratio = params.get("val_ratio")
    if isinstance(val_ratio, (int, float)) and val_ratio > MAX_QUIET_VAL_RATIO:
        # the experiment's holdout is a share of UniProt's UniRef50 clusters
        # (in a mix, val_ratio is a share of the whole pool, which counts these clusters)
        fix = (f"the UniRef90 experiment used data_params.val_ratio: {EXPERIMENT_VAL_RATIO}"
               if cluster_source == "uniref_xml"
               else "set data_params.val_ratio lower to train on more of them")
        warnings.append(f"{on}: {_setting(src, 'val_ratio')} holds out about {val_ratio:.1%} of "
                        f"the clusters for validation; {fix}")
    if cluster_source != "uniref_xml" and not from_source:
        notes.append(f"{on}: clusters: {clusters} clusters the members ({members}); the source "
                     f"fasta ({source}) is not needed and is not downloaded")
    plan = ClusterPlan(
        source=source, source_level=source_level, members=members,
        members_level=members_level, members_from_source=from_source,
        members_compressed=members_compressed, members_fasta=members_fasta,
        cluster_source=cluster_source, clusters=clusters, clusters_key=written.get("clusters"),
        identity=identity, coverage=coverage, output_dir=Path(src.data_dir),
        warnings=warnings, notes=notes,
    )
    if cluster_source == "uniref_xml":
        plan.xml_url = UNIREF50_XML_URL
        plan.xml_path = Path(src.download_path(UNIREF50_XML_URL))
    elif cluster_source == "tsv":
        if is_url(clusters):
            plan.tsv_url = clusters
            plan.tsv_path = Path(src.download_path(clusters))
        else:
            plan.tsv_path = Path(clusters)
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
