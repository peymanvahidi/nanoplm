"""Read a pretraining data.yaml into sources.

Two layouts are accepted:

  sources layout   ``output_dir`` + ``sources: {name: spec}`` + ``data_params`` defaults.
                   Each source names a ``fasta`` (a UniProt shortcut, a URL that is
                   downloaded, or a local file read in place) or a ``prepared`` folder.
                   Any data_params field may be set on a source to override the default.
  data_dirs layout the old one: ``data_dirs.url`` / ``local_fasta`` and nine paths. It is
                   read as one source named "main" and every stage keeps its old shape.

Only one source is supported for now. Relative paths are relative to data.yaml's
directory, where ``nanoplm data from-yaml`` runs DVC.

The sources layout puts everything under ``output_dir``:

  downloads/<file>          one file per URL (the source FASTA, members, uniref50.xml.gz)
  <source>/                 that source's intermediate FASTAs, validation/ and split/
  train/ val/ .data_manifest   the dataset pretrain.yaml's dataset_dir points at
  cluster_index/            members.tsv and the index (cluster subsampling only)
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

FASTA_SHORTCUTS = {k: DATASET_URLS[k]
                   for k in ("swissprot", "trembl", "uniref50", "uniref90", "uniref100")}
URL_SCHEMES = ("http://", "https://", "ftp://")
# data_params fields of a pretraining run, in template order
PARAM_FIELDS = ("max_seqs", "min_seq_len", "max_seq_len", "val_ratio", "drop_invalid_sequences",
                "shuffle", "shuffle_seed", "samples_per_shard", "max_workers", "register_tokens")
OLD_PARAM_NAMES = {"seqs_num": "max_seqs"}
SOURCE_KEYS = ("fasta", "prepared", "weight", "cluster_subsampling")
CLUSTER_KEYS = ("members", "clustering", "identity", "coverage")
TOP_KEYS = ("output_dir", "sources", "data_params")
DATA_DIRS_KEYS = ("url", "local_fasta", "compressed_fasta", "extracted_fasta", "validation_dir",
                  "shuffled_fasta", "filtered_fasta", "splitted_fasta_dir", "output_dir")
# folders under output_dir that a source name would collide with
RESERVED_NAMES = ("downloads", "train", "val", "cluster_index")
LEGACY_NAME = "main"
WEIGHT_TOLERANCE = 1e-6
ONE_SOURCE = "mixing several sources is not supported yet (one source only)"


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
    return value.startswith(URL_SCHEMES)


def file_name(value: str) -> str:
    """The file name of a URL (query and fragment dropped) or a path."""
    return Path(urlparse(value).path if is_url(value) else value).name


def strip_gz(name: str) -> str:
    return name[:-3] if name.endswith(".gz") else name


def _join(*parts: str) -> str:
    return "/".join(p.rstrip("/") for p in parts if p)


@dataclass
class Source:
    """One source, resolved. ``params`` holds every data_params field under its new name;
    ``keys`` maps each field to where data.yaml writes it, for DVC params and ${...}."""

    name: str
    legacy: bool = False
    fasta: Optional[str] = None            # URL or path; a shortcut is expanded
    fasta_written: Optional[str] = None    # as written in data.yaml
    prepared: Optional[str] = None
    weight: Optional[float] = None
    params: dict = field(default_factory=dict)
    keys: dict = field(default_factory=dict)
    overrides: tuple = ()                  # data_params fields set on this source
    cluster: Optional[dict] = None         # the cluster_subsampling block, None = off
    paths: dict = field(default_factory=dict)
    output_dir: str = ""                   # the top-level output_dir (dataset_dir)
    downloads_dir: str = ""

    @property
    def is_url(self) -> bool:
        return bool(self.fasta) and is_url(self.fasta)

    @property
    def is_compressed(self) -> bool:
        return bool(self.fasta) and file_name(self.fasta).endswith(".gz")

    @property
    def work_dir(self) -> str:
        return _join(self.output_dir, self.name)

    def key(self, name: str) -> str:
        return self.keys[name]

    def ref(self, name: str) -> str:
        """``${data_params.max_seq_len}``, or the source's override."""
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

    @property
    def source(self) -> Source:
        """The one source (several are refused while parsing)."""
        return next(iter(self.sources.values()))


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
            raise DataConfigError(f"{where}.fasta is a directory, not a .fasta/.fasta.gz file: "
                                  f"{fasta}")
    if not (name.endswith(".fasta") or name.endswith(".fasta.gz")):
        raise DataConfigError(f"{where}.fasta must end with .fasta or .fasta.gz, got: {fasta}")
    return fasta


def _cluster_block(block, where: str) -> Optional[dict]:
    where = f"{where}.cluster_subsampling"
    if block is None:
        raise DataConfigError(f"{where} is empty. It needs members (null = this fasta); "
                              "delete the block to turn cluster subsampling off")
    if not isinstance(block, dict):
        raise DataConfigError(f"{where} must be a mapping, got {block!r}")
    if "enabled" in block:
        raise DataConfigError(f"{where}: remove enabled; the block being present turns it on "
                              "(delete or comment out the block to turn it off)")
    _check_keys(where, block, CLUSTER_KEYS)
    if "members" not in block:
        raise DataConfigError(f"{where} needs members (null = this fasta); delete the block "
                              "to turn cluster subsampling off")
    return dict(block)


def _weight(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DataConfigError(f"{where}.weight must be a number, got {value!r}")
    if abs(value - 1) > WEIGHT_TOLERANCE:
        raise DataConfigError(f"{where}.weight is {value:g}; with one source the weight must "
                              "be 1 (or leave it out). Weights are shares of training "
                              "sequences and must sum to 1")
    return float(value)


def _parse_sources(config: dict, base_dir, name_of: str) -> DataConfig:
    warnings: list = []
    _check_keys(name_of, config, TOP_KEYS)
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise DataConfigError(f"{name_of}: output_dir is required (the folder pretrain.yaml's "
                              "dataset_dir points at)")
    output_dir = output_dir.rstrip("/") or "/"
    sources = config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise DataConfigError(f"{name_of}: sources needs one entry, e.g. "
                              "sources: {swissprot: {fasta: swissprot}}")
    if len(sources) > 1:
        raise DataConfigError(f"{name_of}: {ONE_SOURCE}; sources has {len(sources)}: "
                              f"{', '.join(map(str, sources))}")
    defaults_raw = config.get("data_params") or {}
    if not isinstance(defaults_raw, dict):
        raise DataConfigError(f"{name_of}: data_params must be a mapping")
    _check_keys("data_params", [OLD_PARAM_NAMES.get(k, k) for k in defaults_raw], PARAM_FIELDS)
    defaults = _params(defaults_raw, "data_params", warnings)

    out = {}
    for name, spec in sources.items():
        where = f"sources.{name}"
        if name in CLUSTER_KEYS + ("cluster_subsampling",) + SOURCE_KEYS:
            raise DataConfigError(f"{where}: {name} is not a source name; indent it under one")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise DataConfigError(f"{where}: a source name is letters, digits and _ (not "
                                  "starting with a digit)")
        if name in RESERVED_NAMES:
            raise DataConfigError(f"{where}: {name} is a folder name under output_dir; pick "
                                  "another source name")
        if not isinstance(spec, dict):
            raise DataConfigError(f"{where} must be a mapping with fasta or prepared")
        allowed = SOURCE_KEYS + PARAM_FIELDS
        _check_keys(where, [OLD_PARAM_NAMES.get(k, k) for k in spec], allowed)
        src = Source(name=name, output_dir=output_dir,
                     downloads_dir=_join(output_dir, "downloads"))
        over = _params({k: v for k, v in spec.items() if k not in SOURCE_KEYS}, where, warnings)
        if "weight" in spec:
            src.weight = _weight(spec["weight"], where)
        has_fasta, has_prepared = spec.get("fasta") is not None, spec.get("prepared") is not None
        if has_fasta == has_prepared:
            raise DataConfigError(f"{where}: set fasta (a FASTA to build from) or prepared (a "
                                  "folder another data.yaml built), not both"
                                  if has_fasta else f"{where} needs fasta or prepared")
        if has_prepared:
            if over or "cluster_subsampling" in spec:
                extra = sorted(set(over) | ({"cluster_subsampling"} & set(spec)))
                raise DataConfigError(f"{where}: a prepared source is used as it was built; "
                                      f"remove {', '.join(extra)}")
            if not isinstance(spec["prepared"], str):
                raise DataConfigError(f"{where}.prepared must be a folder path")
            src.prepared = spec["prepared"]
            out[name] = src
            continue
        src.fasta_written = spec["fasta"]
        src.fasta = _fasta(spec["fasta"], where, base_dir)
        if "cluster_subsampling" in spec:
            src.cluster = _cluster_block(spec["cluster_subsampling"], where)
        merged = {**defaults, **over}
        missing = [f for f in PARAM_FIELDS if f not in merged]
        if missing:
            raise DataConfigError(f"{name_of}: data_params.{missing[0]} is required")
        src.params = {f: merged[f][0] for f in PARAM_FIELDS}
        src.keys = {f: merged[f][1] for f in PARAM_FIELDS}
        src.overrides = tuple(f for f in PARAM_FIELDS if f in over)
        src.paths = _source_paths(src)
        out[name] = src
    return DataConfig(output_dir=output_dir, sources=out, warnings=warnings)


def _source_paths(src: Source) -> dict:
    work = src.work_dir
    if src.is_url:
        got = src.download_path(src.fasta)
        compressed, extracted = (got, _join(work, strip_gz(file_name(src.fasta)))) \
            if src.is_compressed else (None, got)
    elif src.is_compressed:                 # a local .gz is read in place
        compressed, extracted = src.fasta, _join(work, strip_gz(file_name(src.fasta)))
    else:                                   # a local .fasta is read in place
        compressed, extracted = None, src.fasta
    return {
        "compressed_fasta": compressed,
        "extracted_fasta": extracted,
        "validation_dir": _join(work, "validation"),
        "shuffled_fasta": _join(work, "shuffled.fasta"),
        "filtered_fasta": _join(work, "filtered.fasta"),
        "splitted_fasta_dir": _join(work, "split"),
        "output_dir": src.output_dir,
    }


def legacy_source(config: dict) -> Source:
    """The old data_dirs layout as source "main", keyed the way the old dvc.yaml was.

    Used for distillation_data.yaml too, so it reads every data_params key it finds.
    """
    dirs = config.get("data_dirs") or {}
    params = dict(config.get("data_params") or {})
    src = Source(name=LEGACY_NAME, legacy=True)
    src.fasta = dirs.get("url") or dirs.get("local_fasta") or None
    src.fasta_written = src.fasta
    src.paths = {k: dirs.get(k) for k in DATA_DIRS_KEYS}
    src.output_dir = dirs.get("output_dir") or ""
    if "seqs_num" in params and "max_seqs" not in params:
        params["max_seqs"] = params.pop("seqs_num")
        src.keys["max_seqs"] = "data_params.seqs_num"
    src.params = params
    for k in params:
        src.keys.setdefault(k, f"data_params.{k}")
    return src


def migration_hint(config: dict) -> str:
    """The deprecation warning for a data_dirs config, with the block to paste in."""
    dirs = config.get("data_dirs") or {}
    fasta = dirs.get("url") or dirs.get("local_fasta")
    shortcut = next((k for k, v in FASTA_SHORTCUTS.items() if v == fasta), None)
    out = dirs.get("output_dir") or "output/data/pretrain_data"
    lines = [
        "data_dirs is the old data.yaml layout. It still works, read as one source named "
        f'"{LEGACY_NAME}". To switch, replace data_dirs with:',
        "",
        f'output_dir: "{out}"',
        "sources:",
        f"  {LEGACY_NAME}:",
        f'    fasta: {shortcut if shortcut else repr(fasta)}',
        "",
        "and rename data_params.seqs_num to max_seqs. The new layout keeps intermediate "
        f"files under output_dir/{LEGACY_NAME}/, so the first run after switching rebuilds "
        "everything; the old ones in data_dirs can then be deleted.",
    ]
    compressed = dirs.get("compressed_fasta")
    if dirs.get("url") and compressed:
        lines.append(f'To skip the download, set fasta: "{compressed}" (the file already '
                     "downloaded).")
    return "\n".join(lines)


def parse_data_config(config: dict, base_dir=None, name_of: str = "data.yaml") -> DataConfig:
    """Read a parsed pretraining data.yaml. ``base_dir`` is its directory (None = cwd)."""
    if "cluster_subsampling" in config:
        raise DataConfigError(
            f"{name_of}: cluster_subsampling now sits under its source: "
            "sources.<name>.cluster_subsampling (see `nanoplm data get-yaml`), without "
            "enabled: the block being present turns it on")
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
        _check_keys("data_params", [OLD_PARAM_NAMES.get(k, k) for k in params], PARAM_FIELDS)
        src = legacy_source(config)
        return DataConfig(output_dir=src.output_dir, sources={src.name: src}, legacy=True,
                          warnings=[migration_hint(config)])
    return _parse_sources(config, base_dir, name_of)
