"""One validation split for a whole data.yaml, shared out by the source weights.

data_params sets the val size once: ``val_ratio`` (a share of the pool) or ``val_size``
(a count). The pool sums every built source's units: filtered sequences for a plain
source, universe clusters for a cluster-subsampled one (both after the max_seqs cap).
The total is split across the sources by their weights, so each source's val count
depends on every source's pool. ``nanoplm data from-yaml`` runs it as DVC stages:

  <source>-count   count-units: the source's units into <source>/pool_units.json
  val_split        val-split: every count + the weights -> <source>/val_count.json each
  <source>-split   split / cluster-index read their own val_count.json

A rerun of val_split that writes the same count for a source reruns nothing of that
source. A prepared source keeps the val split it was built with and is not in the pool;
the built sources share the val sequences by their weights scaled to sum to 1.

One source with val_ratio does not use these stages: split and cluster-index take the
ratio itself, as before.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

POOL_UNITS = "pool_units.json"
VAL_COUNT = "val_count.json"
PLAIN, CLUSTER = "sequences", "clusters"


class ValSplitError(ValueError):
    """The val size of data.yaml cannot be met; the message says what to change."""


_HEADER = re.compile(rb"^>(\S*)", re.M)
_CHUNK = 1 << 26


def _header_ids(fasta):
    """The first word of every FASTA header, a chunk's worth of them at a time."""
    tail = b""
    with open(fasta, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            block = tail + block
            cut = block.rfind(b"\n") + 1
            if cut == 0:
                tail = block
                continue
            tail = block[cut:]
            yield _HEADER.findall(block, 0, cut)
    if tail:
        yield _HEADER.findall(tail)


def count_fasta_records(fasta) -> int:
    return sum(len(ids) for ids in _header_ids(fasta))


def count_universe_clusters(fasta, members_tsv, max_clusters: int = -1) -> int:
    """Clusters cluster_index puts in its universe: those whose representative is a
    record of ``fasta`` (the pool), capped to ``max_clusters`` when that is > 0."""
    import numpy as np
    from nanoplm.data.preparation import uniref_hierarchy as uh
    from nanoplm.data.preparation.cluster_table import read_members_table
    table = read_members_table(members_tsv)
    n = 0
    for ids in _header_ids(fasta):
        if not ids:
            continue
        lens = np.fromiter((len(i) for i in ids), dtype=np.int64, count=len(ids))
        keys = uh.hash_accessions(np.frombuffer(b"".join(ids), dtype=np.uint8), lens)
        pos = np.clip(np.searchsorted(table.keys, keys), 0, len(table.keys) - 1)
        hit = np.asarray(table.keys[pos]) == keys
        n += int(np.asarray(table.is_rep[pos[hit]]).sum())
    return min(n, int(max_clusters)) if max_clusters > 0 else n


def write_units(path, units: int, kind: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"units": int(units), "kind": kind}) + "\n")


def write_val_count(path, val: int, units: int, kind: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"val": int(val), "units": int(units), "kind": kind})
                          + "\n")


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def split_by_weight(total: int, weights: dict) -> dict:
    """``total`` shared out by ``weights`` (scaled to sum to 1), rounded so the parts
    sum to ``total``: largest remainder, ties to the source listed first."""
    s = sum(weights.values())
    exact = {n: total * w / s for n, w in weights.items()}
    out = {n: int(v) for n, v in exact.items()}
    order = sorted(weights, key=lambda n: -(exact[n] - out[n]))
    for n in order[:total - sum(out.values())]:
        out[n] += 1
    return out


def val_counts(sources: dict, val_ratio: Optional[float] = None,
               val_size: Optional[int] = None) -> dict:
    """{name: val units} for ``sources`` = {name: (units, kind, weight)}.

    Every source gets at least one val unit and keeps at least one train unit.
    """
    if (val_ratio is None) == (val_size is None):
        raise ValSplitError("set data_params.val_ratio or data_params.val_size, not both")
    pool = sum(u for u, _, _ in sources.values())
    total = int(round(val_ratio * pool)) if val_size is None else int(val_size)
    counts = split_by_weight(total, {n: w for n, (_, _, w) in sources.items()})
    field = "data_params.val_ratio" if val_size is None else "data_params.val_size"
    for name, n in counts.items():
        units, kind, _ = sources[name]
        if n < 1:
            raise ValSplitError(
                f"sources.{name} would get 0 val {kind} ({field} gives {total:,} for the "
                f"whole mix); raise {field} or this source's weight")
        if n >= units:
            fix = ("lower data_params.val_ratio or use data_params.val_size"
                   if val_size is None else "lower data_params.val_size")
            raise ValSplitError(
                f"sources.{name} would need {n:,} val {kind} but has {units:,}; {fix}")
    return counts
