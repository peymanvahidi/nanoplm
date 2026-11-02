"""Read a pretraining data.yaml into sources.

Two layouts are accepted:

  sources layout   ``output_dir`` + ``sources: {name: spec}`` + ``data_params`` defaults.
                   Each source names a ``fasta``: a UniProt shortcut, a URL that is
                   downloaded, or a local file read in place.
                   Any data_params field but val_ratio / val_size may be set on a source
                   to override the default; the val size is set once (val_split).
  data_dirs layout the old one: ``data_dirs.url`` / ``local_fasta`` and nine paths. It is
                   read as one source named "main" and every stage keeps its old shape.

Several sources make a mixture: each has a weight, its share of the training
sequences, and the weights sum to 1. Relative paths are relative to data.yaml's
directory, where ``nanoplm data from-yaml`` runs DVC.

The sources layout puts everything under ``output_dir``:

  downloads/<file>          one file per URL (the source FASTA, members, uniref50.xml.gz),
                            shared by every source that names that URL
  <source>/                 that source's intermediate FASTAs, validation/ and split/
  train/ val/ .data_manifest   the dataset pretrain.yaml's dataset_dir points at
  cluster_index/            members.tsv and the index (cluster subsampling only)

With several sources, train/ val/ cluster_index/ and .data_manifest of each source go
into ``<source>/`` (Source.data_dir) and ``output_dir/.data_manifest`` is the mixture
manifest (nanoplm.data.infra.mixture). A .gz or .zst is extracted into downloads/ and
checked there (``<file>_validation/``), once for all the sources that read it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml

from nanoplm.config.datasets import DATASET_URLS
from nanoplm.data.infra.mixture import WEIGHT_TOLERANCE, weights_error

FASTA_SHORTCUTS = {k: DATASET_URLS[k]
                   for k in ("swissprot", "trembl", "uniref50", "uniref90", "uniref100")}
URL_SCHEMES = ("http://", "https://", "ftp://")
FASTA_SUFFIXES = (".fasta", ".fa")
COMPRESSED_SUFFIXES = (".gz", ".zst")      # the extract stage unpacks these
# data_params fields of a pretraining run, in docs-schema order: the template's, with
# val_size (shipped commented out) where the schema adds it, before "Rarely changed"
PARAM_FIELDS = ("max_seqs", "min_seq_len", "max_seq_len", "val_ratio", "drop_invalid_sequences",
                "val_size", "shuffle", "shuffle_seed", "samples_per_shard", "max_workers",
                "register_tokens")
# the val size: data_params sets exactly one of them, for the whole data.yaml
VAL_FIELDS = ("val_ratio", "val_size")
# the "rarely changed" fields may be left out (master's template values). In the sources
# layout a field at its default goes into dvc.yaml as a value, not a DVC param, so
# writing it out or deleting it reruns nothing.
PARAM_DEFAULTS = {"drop_invalid_sequences": False, "shuffle": True, "shuffle_seed": 24,
                  "samples_per_shard": 2000, "max_workers": 2, "register_tokens": 0}
OLD_PARAM_NAMES = {"seqs_num": "max_seqs"}
SOURCE_KEYS = ("fasta", "weight", "cluster_subsampling")
# source keys that are gone, and what to do instead
REMOVED_SOURCE_KEYS = {"prepared": "To train on that folder as it is, set pretrain.yaml's "
                                    "dataset_dir to it; to use it in a mix, set fasta to the "
                                    "FASTA it was built from."}
CLUSTER_KEYS = ("members", "clusters", "identity", "coverage")
OLD_CLUSTER_KEYS = {"clustering": "clusters"}
# where the clusters come from: cluster_config checks the value
CLUSTERS_CHOICES = ("uniprot (UniProt's UniRef50 clusters, for UniRef90/UniRef100 members), "
                    "headers (CL: tags in the FASTA headers), mmseqs2 (clusters with MMseqs2; "
                    "slow: hours on large files), or a .tsv file of representative<TAB>member "
                    "lines")
TOP_KEYS = ("output_dir", "sources", "data_params")
DATA_DIRS_KEYS = ("url", "local_fasta", "compressed_fasta", "extracted_fasta", "validation_dir",
                  "shuffled_fasta", "filtered_fasta", "splitted_fasta_dir", "output_dir")
# folders under output_dir that a source name would collide with
RESERVED_NAMES = ("downloads", "train", "val", "cluster_index")
LEGACY_NAME = "main"
# data_params keys older data.yaml templates shipped that nothing reads any more
LEGACY_IGNORED = ("pipeline_backend", "device")
# data_params fields that one setting serves every source of a mix: max_workers reaches
# the stages through one environment variable
SHARED_FIELDS = ("max_workers",)
# data_params fields set only there, even with one source: shuffle_seed also seeds the
# cluster val and max_seqs draws
DATA_PARAMS_ONLY = ("shuffle", "shuffle_seed")


class DataConfigError(ValueError):
    """data.yaml cannot be read into sources."""


# ------------------------------------------------------------------ strict YAML


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses a key written twice in one mapping (safe_load keeps the last)."""


def _no_duplicates(loader, node, deep=False):
    seen = {}
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise DataConfigError(f"line {key_node.start_mark.line + 1}: duplicate key {key!r} "
                                  f"(first on line {seen[key] + 1})")
        seen[key] = key_node.start_mark.line
    return loader.construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def load_strict(path) -> dict:
    """Parse data.yaml, refusing duplicate keys. Errors name the file and line."""
    path = Path(path)
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except DataConfigError as e:
        raise DataConfigError(f"{path.name}:{str(e)[len('line '):]}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise DataConfigError(f"{path.name}: expected a mapping at the top level")
    return data


# ------------------------------------------------------------------ the model


def is_url(value: str) -> bool:
    """A FASTA that is downloaded (http, https or ftp), not a local path."""
    return value.startswith(URL_SCHEMES)


def file_name(value: str) -> str:
    """The file name of a URL (query and fragment dropped) or a path."""
    return Path(urlparse(value).path if "://" in value else value).name


def compression_suffix(value: str) -> str:
    """``.gz`` or ``.zst`` for a URL or path of a compressed file, else ``""``."""
    name = file_name(value)
    return next((s for s in COMPRESSED_SUFFIXES if name.endswith(s)), "")


def is_compressed(value: str) -> bool:
    """Whether a URL or path names a .gz or .zst file (by its extension only)."""
    return bool(compression_suffix(value))


def strip_compressed(name: str) -> str:
    """The file name once extracted: ``x.fa.zst`` -> ``x.fa``."""
    for suffix in COMPRESSED_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def fasta_stem(name: str) -> str:
    """The file name without its FASTA and compression suffixes: ``x.fa.zst`` -> ``x``."""
    name = strip_compressed(name)
    for suffix in FASTA_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def is_fasta_name(value: str) -> bool:
    """Whether a URL or path ends with .fasta or .fa, optionally .gz or .zst."""
    return strip_compressed(file_name(value)).endswith(FASTA_SUFFIXES)


def literal(value) -> str:
    """A value as DVC writes a ${...} reference to it into a command."""
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


def is_default(value, default) -> bool:
    """``value == default`` without True == 1 or 0 == False."""
    return type(value) is type(default) and value == default


def _join(*parts: str) -> str:
    return "/".join(p.rstrip("/") for p in parts if p)


@dataclass
class Source:
    """One source, resolved. ``params`` holds every data_params field under its new name;
    ``keys`` maps each field to where data.yaml writes it, for DVC params and ${...}."""

    name: str
    legacy: bool = False
    fasta: Optional[str] = None            # URL or path; a shortcut is expanded
    weight: Optional[float] = None
    params: dict = field(default_factory=dict)
    keys: dict = field(default_factory=dict)
    overrides: tuple = ()                  # data_params fields set on this source
    defaulted: frozenset = frozenset()     # fields at their PARAM_DEFAULTS value
    cluster: Optional[dict] = None         # the cluster_subsampling block, None = off
    cluster_keys: dict = field(default_factory=dict)   # block field -> where data.yaml sets it
    paths: dict = field(default_factory=dict)
    output_dir: str = ""                   # the top-level output_dir (dataset_dir)
    downloads_dir: str = ""
    mixed: bool = False                    # one of several sources
    # its val count comes from the val_split stage (a mix, or val_size) instead of
    # its own val_ratio (see val_split)
    val_from_split: bool = False

    @property
    def data_dir(self) -> str:
        """Where the source's train/ val/ and .data_manifest go: output_dir itself for
        a single source, ``<output_dir>/<name>`` in a mix."""
        return self.work_dir if self.mixed else self.output_dir

    @property
    def is_url(self) -> bool:
        return bool(self.fasta) and is_url(self.fasta)

    @property
    def is_compressed(self) -> bool:
        return bool(self.fasta) and is_compressed(self.fasta)

    @property
    def work_dir(self) -> str:
        return _join(self.output_dir, self.name)

    def key(self, name: str) -> str:
        """Where data.yaml sets a field (``data_params.X`` when it is left out)."""
        return self.keys[name]

    def param_key(self, name: str) -> Optional[str]:
        """The DVC param for a field, None for one at its default."""
        return None if name in self.defaulted else self.keys[name]

    def ref(self, name: str) -> str:
        """``${data_params.max_seq_len}``, or the source's override; a field at its
        default is written as its value."""
        if name in self.defaulted:
            return literal(self.params[name])
        return "${" + self.keys[name] + "}"

    def path(self, name: str) -> str:
        """A path as dvc.yaml writes it: ${data_dirs.X} (old layout) or the path itself."""
        return "${data_dirs." + name + "}" if self.legacy else self.paths[name]

    def path_params(self, *names: str) -> list:
        """The data_dirs keys the old layout lists as params (the new one embeds paths)."""
        return [f"data_dirs.{n}" for n in names] if self.legacy else []

    def download_path(self, url: str) -> str:
        return _join(self.downloads_dir, file_name(url))


@dataclass
class DataConfig:
    output_dir: str
    sources: dict
    legacy: bool = False
    warnings: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)      # data.yaml as parsed
    val: tuple = ()                              # ("val_ratio", 0.1) or ("val_size", n)

    @property
    def val_from_split(self) -> bool:
        """The val counts come from the val_split stage (see val_split)."""
        return any(s.val_from_split for s in self.sources.values())

    @property
    def source(self) -> Source:
        """The one source of a single-source config."""
        if len(self.sources) != 1:
            raise DataConfigError(f"{len(self.sources)} sources; use cfg.sources")
        return next(iter(self.sources.values()))

    @property
    def is_mixture(self) -> bool:
        return len(self.sources) > 1


# ------------------------------------------------------------------ parsing


def _check_keys(where: str, got, allowed) -> None:
    unknown = [k for k in got if k not in allowed]
    if unknown:
        names = ", ".join(str(k) for k in unknown)
        raise DataConfigError(f"{where}: unknown key{'s' if len(unknown) > 1 else ''} {names} "
                              f"(allowed: {', '.join(allowed)})")


def _params(block: dict, where: str, warnings: list) -> dict:
    """data_params fields of ``block`` under their new names, old names accepted."""
    out = {}
    for k, v in block.items():
        new = OLD_PARAM_NAMES.get(k, k)
        if new != k:
            if new in block:
                raise DataConfigError(f"{where}: set {new} or {k}, not both ({k} is the old name)")
            warnings.append(f"{where}.{k} is now called {new}; the old name still works")
        out[new] = (v, f"{where}.{k}")
    return out


def _fasta(value, where: str, base_dir) -> str:
    if not isinstance(value, str) or not value:
        raise DataConfigError(f"{where}.fasta must be a shortcut ({', '.join(FASTA_SHORTCUTS)}), "
                              f"a URL or a file path, got {value!r}")
    fasta = FASTA_SHORTCUTS.get(value, value)
    name = file_name(fasta)
    if not is_url(fasta):
        p = Path(fasta)
        on_disk = p if p.is_absolute() or base_dir is None else Path(base_dir) / p
        if on_disk.is_dir():
            raise DataConfigError(f"{where}.fasta is a directory, not a .fasta/.fa file: "
                                  f"{fasta}")
    if not is_fasta_name(name):
        raise DataConfigError(f"{where}.fasta must be a shortcut ({', '.join(FASTA_SHORTCUTS)}) "
                              "or a URL or path that ends with .fasta or .fa (optionally .gz or "
                              f".zst), got: {fasta}")
    return fasta


def _cluster_block(block, where: str, warnings: list) -> tuple:
    """(the block's fields under their new names, {field: where data.yaml sets it}), or
    (None, {}) when it is off. A block must set clusters."""
    where = f"{where}.cluster_subsampling"
    if block is False:
        return None, {}
    if block is True:
        raise DataConfigError(f"{where}: true no longer turns it on; write a block that sets "
                              f"clusters to {CLUSTERS_CHOICES}")
    if block is None:
        raise DataConfigError(f"{where} is empty. Set clusters in it to {CLUSTERS_CHOICES}; "
                              "or delete the block to turn it off")
    if not isinstance(block, dict):
        raise DataConfigError(f"{where} must be a mapping with clusters (or false), got "
                              f"{block!r}")
    if "enabled" in block:
        raise DataConfigError(f"{where}: remove enabled; the block turns it on, false (or no "
                              "block) turns it off")
    _check_keys(where, [OLD_CLUSTER_KEYS.get(k, k) for k in block], CLUSTER_KEYS)
    out, keys = {}, {}
    for k, v in block.items():
        new = OLD_CLUSTER_KEYS.get(k, k)
        if new != k:
            if new in block:
                raise DataConfigError(f"{where}: set {new} or {k}, not both ({k} is the old "
                                      "name)")
            warnings.append(f"{where}.{k} is now called {new}; the old name still works; "
                            "renaming it reruns the cluster_members stage once")
        out[new] = v
        keys[new] = f"{where}.{k}"
    if out.get("clusters") == "auto":
        raise DataConfigError(f"{keys['clusters']}: auto was removed; choose where the "
                              f"clusters come from: {CLUSTERS_CHOICES}")
    if out.get("clusters") is None:
        raise DataConfigError(f"{keys.get('clusters', f'{where}.clusters')} is required: "
                              f"{CLUSTERS_CHOICES}")
    return out, keys


def _weight(value, where: str, single: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DataConfigError(f"{where}.weight must be a number, got {value!r}")
    if not single:
        return float(value)
    if abs(value - 1) > WEIGHT_TOLERANCE:
        raise DataConfigError(f"{where}.weight is {value:g}; with one source the weight must "
                              "be 1 (or leave it out). Weights are shares of training "
                              "sequences and must sum to 1")
    return float(value)


def _parse_sources(config: dict, base_dir, name_of: str) -> DataConfig:
    warnings: list = []
    _check_keys(name_of, config, TOP_KEYS)
    sources = config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise DataConfigError(f"{name_of}: sources needs one entry, e.g. "
                              "sources: {swissprot: {fasta: swissprot}}")
    # a removed key first: a config that used it may also lack what replaced it
    for name, spec in sources.items():
        gone = [k for k in REMOVED_SOURCE_KEYS if isinstance(spec, dict) and k in spec]
        if gone:
            raise DataConfigError(f"sources.{name}.{gone[0]} was removed. "
                                  f"{REMOVED_SOURCE_KEYS[gone[0]]}")
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise DataConfigError(f"{name_of}: output_dir is required (the folder pretrain.yaml's "
                              "dataset_dir points at)")
    else:
        output_dir = output_dir.rstrip("/") or "/"
    single = len(sources) == 1
    defaults_raw = config.get("data_params") or {}
    if not isinstance(defaults_raw, dict):
        raise DataConfigError(f"{name_of}: data_params must be a mapping")
    _check_keys("data_params", [OLD_PARAM_NAMES.get(k, k) for k in defaults_raw], PARAM_FIELDS)
    defaults = _params(defaults_raw, "data_params", warnings)
    val = _val_setting(defaults, name_of)

    out = {}
    for name, spec in sources.items():
        where = f"sources.{name}"
        if name in (*CLUSTER_KEYS, *OLD_CLUSTER_KEYS, "cluster_subsampling", *SOURCE_KEYS):
            raise DataConfigError(f"{where}: {name} is not a source name; indent it under one")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise DataConfigError(f"{where}: a source name is letters, digits and _ (not "
                                  "starting with a digit)")
        if name in RESERVED_NAMES:
            raise DataConfigError(f"{where}: {name} is a folder name under output_dir; pick "
                                  "another source name")
        if not isinstance(spec, dict):
            raise DataConfigError(f"{where} must be a mapping with fasta")
        allowed = SOURCE_KEYS + PARAM_FIELDS
        _check_keys(where, [OLD_PARAM_NAMES.get(k, k) for k in spec], allowed)
        src = Source(name=name, output_dir=output_dir, mixed=not single,
                     downloads_dir=_join(output_dir, "downloads"))
        over = _params({k: v for k, v in spec.items() if k not in SOURCE_KEYS}, where, warnings)
        if "weight" in spec:
            src.weight = _weight(spec["weight"], where, single)
        elif not single:
            raise DataConfigError(f"{where} needs a weight (its share of training sequences; "
                                  "the weights must sum to 1)")
        per_source_val = [f for f in VAL_FIELDS if f in over]
        if per_source_val:
            raise DataConfigError(
                f"{where}.{per_source_val[0]}: move it to data_params. The val split is set "
                "once for the whole data.yaml and shared out by the source weights")
        only = [f for f in DATA_PARAMS_ONLY if f in over]
        if only:
            raise DataConfigError(f"{where}.{only[0]}: set it in data_params; it applies to "
                                  "every source")
        shared = [f for f in SHARED_FIELDS if f in over]
        if shared and not single:
            raise DataConfigError(f"{where}.{shared[0]}: set it in data_params; it applies to "
                                  "every source of a mix")
        if spec.get("fasta") is None:
            raise DataConfigError(f"{where} needs fasta (a FASTA to build from)")
        src.fasta = _fasta(spec["fasta"], where, base_dir)
        if "cluster_subsampling" in spec:
            src.cluster, src.cluster_keys = _cluster_block(spec["cluster_subsampling"], where,
                                                           warnings)
        merged = {**defaults, **over}
        missing = [f for f in PARAM_FIELDS
                   if f not in merged and f not in PARAM_DEFAULTS and f not in VAL_FIELDS]
        if missing:
            raise DataConfigError(f"{name_of}: data_params.{missing[0]} is required")
        if not val:
            raise DataConfigError(f"{name_of}: data_params.val_ratio is required (or val_size, "
                                  "a number of val sequences)")
        src.params = {f: merged[f][0] if f in merged else PARAM_DEFAULTS.get(f)
                      for f in PARAM_FIELDS}
        src.keys = {f: merged[f][1] if f in merged else f"data_params.{f}"
                    for f in PARAM_FIELDS}
        src.defaulted = frozenset(f for f, d in PARAM_DEFAULTS.items()
                                  if is_default(src.params[f], d))
        src.overrides = tuple(f for f in PARAM_FIELDS if f in over)
        src.paths = _source_paths(src)
        src.val_from_split = not single or val[0] == "val_size"
        if src.cluster is not None and not src.val_from_split:
            _check_cluster_holdout(src)
        out[name] = src
    if not single:
        _check_mix(out, name_of)
    return DataConfig(output_dir=output_dir, sources=out, warnings=warnings, val=val)


def _val_setting(defaults: dict, name_of: str) -> tuple:
    """("val_ratio", r) or ("val_size", n): data_params sets at most one of them (a
    built source needs one: _parse_sources)."""
    got = [f for f in VAL_FIELDS if f in defaults]
    if len(got) == 2:
        raise DataConfigError(f"{name_of}: set data_params.val_ratio or data_params.val_size, "
                              "not both")
    if not got:
        return ()
    value = defaults[got[0]][0]
    if got[0] == "val_size" and (isinstance(value, bool) or not isinstance(value, int)
                                 or value < 1):
        raise DataConfigError(f"{name_of}: data_params.val_size must be a whole number of val "
                              f"sequences, at least 1, got {value!r}")
    return got[0], value


def _check_cluster_holdout(src: Source) -> None:
    """With max_seqs > 0 a cluster source has at most max_seqs clusters, so a val_ratio
    that holds out none of them (or all) is known before any stage runs."""
    n, ratio = src.params.get("max_seqs"), src.params.get("val_ratio")
    if not (isinstance(n, int) and not isinstance(n, bool) and n > 0
            and isinstance(ratio, (int, float)) and 0 < ratio < 1):
        return
    held = round(n * ratio)
    if held == 0 or held == n:
        raise DataConfigError(
            f"sources.{src.name}: {src.key('val_ratio')}={ratio:g} of at most {n:,} clusters "
            f"({src.key('max_seqs')}) holds out {held}; cluster subsampling needs at least one "
            f"train and one val cluster. Set data_params.val_ratio "
            + (f"to at least {1 / n:.2g}" if held == 0 else f"below {1 - 1 / n:.2g}")
            + f", or sources.{src.name}.max_seqs: -1")


def _check_mix(sources: dict, name_of: str) -> None:
    """The weights of several sources sum to 1, and shuffle is on. One register_tokens
    for all of them is checked by `nanoplm data from-yaml` and again by write_mixture."""
    err = weights_error({n: s.weight for n, s in sources.items()})
    if err:
        raise DataConfigError(f"{name_of}: {err}")
    if any(s.params.get("shuffle") is False for s in sources.values()):
        raise DataConfigError(f"{name_of}: data_params.shuffle: false is not supported for a "
                              "mix yet; it comes with the mixes redesign. Remove it, or train "
                              "on one source")


def _source_paths(src: Source) -> dict:
    work = src.work_dir
    # a mix extracts into downloads/, so the sources that read one .gz share the extract
    unpacked = src.downloads_dir if src.mixed else work
    if src.is_url:
        got = src.download_path(src.fasta)
        compressed, extracted = (got, _join(unpacked, strip_compressed(file_name(src.fasta)))) \
            if src.is_compressed else (None, got)
    elif src.is_compressed:                 # a local .gz / .zst is read in place
        compressed, extracted = src.fasta, _join(unpacked, strip_compressed(file_name(src.fasta)))
    else:                                   # a local .fasta is read in place
        compressed, extracted = None, src.fasta
    return {
        "compressed_fasta": compressed,
        "extracted_fasta": extracted,
        "validation_dir": _join(work, "validation"),
        "shuffled_fasta": _join(work, "shuffled.fasta"),
        "filtered_fasta": _join(work, "filtered.fasta"),
        "splitted_fasta_dir": _join(work, "split"),
        "output_dir": src.data_dir,
    }


def legacy_source(config: dict) -> Source:
    """The old data_dirs layout as source "main", keyed the way the old dvc.yaml was.

    Used for distillation_data.yaml too, so it reads every data_params key it finds.
    """
    dirs = config.get("data_dirs") or {}
    params = dict(config.get("data_params") or {})
    src = Source(name=LEGACY_NAME, legacy=True)
    src.fasta = dirs.get("url") or dirs.get("local_fasta") or None
    src.paths = {k: dirs.get(k) for k in DATA_DIRS_KEYS}
    src.output_dir = dirs.get("output_dir") or ""
    if "seqs_num" in params and "max_seqs" not in params:
        params["max_seqs"] = params.pop("seqs_num")
        src.keys["max_seqs"] = "data_params.seqs_num"
    src.params = params
    for k in params:
        src.keys.setdefault(k, f"data_params.{k}")
    return src


def _scalar(value) -> str:
    """A YAML scalar for the paste-in block: a word (a shortcut, a choice) bare, a path or
    URL quoted."""
    if value is None:
        return "null"
    if isinstance(value, str):
        bare = re.fullmatch(r"[a-z][a-z0-9_]*", value) and value not in (
            "null", "true", "false", "yes", "no", "on", "off")
        return value if bare else f'"{value}"'
    return literal(value)


# data_dirs files that the sources layout no longer reads
_OLD_INTERMEDIATES = ("extracted_fasta", "validation_dir", "shuffled_fasta", "filtered_fasta",
                      "splitted_fasta_dir")


def migration_hint(config: dict) -> str:
    """The deprecation warning for a data_dirs config, with the block to paste in."""
    dirs = config.get("data_dirs") or {}
    fasta = dirs.get("url") or dirs.get("local_fasta")
    shortcut = next((k for k, v in FASTA_SHORTCUTS.items() if v == fasta), None)
    out = dirs.get("output_dir") or "output/data/pretrain_data"
    intro = ("data_dirs is the old data.yaml layout. It still works, read as one source "
             f'named "{LEGACY_NAME}". To switch, replace data_dirs with:')
    lines = [
        intro,
        "",
        f'output_dir: "{out}"',
        "sources:",
        f"  {LEGACY_NAME}:",
        f"    fasta: {shortcut or _scalar(fasta)}",
    ]
    rename = ("and rename data_params.seqs_num to max_seqs. "
              if "seqs_num" in (config.get("data_params") or {}) else "")
    lines += [
        "",
        f"{rename}The new layout keeps intermediate files under output_dir/{LEGACY_NAME}/, "
        "so the first run after switching rebuilds them.",
    ]
    compressed = dirs.get("compressed_fasta")
    old = [dirs[k] for k in _OLD_INTERMEDIATES if dirs.get(k)]
    if dirs.get("url") and compressed:
        lines.append(f'To skip the download, set fasta: "{compressed}" (the file already '
                     "downloaded; it is read where it is, so keep it).")
    elif compressed and dirs.get("local_fasta") and is_compressed(dirs["local_fasta"]):
        old.append(compressed)          # get_file's link to local_fasta
    if old:
        lines.append(f"Then these old files can be deleted: {', '.join(old)}.")
    return "\n".join(lines)


def parse_data_config(config: dict, base_dir=None, name_of: str = "data.yaml") -> DataConfig:
    """Read a parsed pretraining data.yaml. ``base_dir`` is its directory (None = cwd)."""
    if "data_dirs" in config:
        if "sources" in config or "output_dir" in config:
            raise DataConfigError(f"{name_of}: data_dirs is the old layout; use output_dir + "
                                  "sources without data_dirs")
        _check_keys(name_of, config, ("data_dirs", "data_params"))
        dirs = config.get("data_dirs") or {}
        if not isinstance(dirs, dict):
            raise DataConfigError(f"{name_of}: data_dirs must be a mapping")
        _check_keys("data_dirs", dirs, DATA_DIRS_KEYS)
        params = config.get("data_params") or {}
        _check_keys("data_params", [OLD_PARAM_NAMES.get(k, k) for k in params],
                    PARAM_FIELDS + LEGACY_IGNORED)
        if "val_size" in params:
            raise DataConfigError(f"{name_of}: data_params.val_size needs the sources layout; "
                                  "use val_ratio with data_dirs")
        src = legacy_source(config)
        warnings = [migration_hint(config)]
        warnings += [f"data_params.{k} is no longer used and is ignored; remove it"
                     for k in LEGACY_IGNORED if k in params]
        return DataConfig(output_dir=src.output_dir, sources={src.name: src}, legacy=True,
                          warnings=warnings)
    return _parse_sources(config, base_dir, name_of)
