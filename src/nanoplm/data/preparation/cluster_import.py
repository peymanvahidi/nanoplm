"""``cluster_index/members.tsv`` from clusters the user brings (cluster_table has the format).

Two cluster sources, both checked against the member FASTA before the length filter
(as with UniProt's clusters: a sequence the filter drops is simply not trained on, and a
cluster whose representative it drops is left out):

  tsv      a ``representative<TAB>member`` file, one line per member, the representative
           also listed as ``rep<TAB>rep`` (MMseqs2's ``createtsv`` / ``easy-cluster``
           ``_cluster.tsv``). Plain, .gz or .zst. Every member FASTA record must be in
           it exactly once, and every id in it must be a record of the FASTA.
  headers  a ``CL:<cluster id>`` token after the id in every FASTA header (Logan's
           protein files: ``>SRR10247178_697980_1<TAB>CL:DRR000534_5840_1``); the
           representative is the record whose id is its cluster id.

Ids are FASTA ids (the first word of the header). Clusters are ordered by their
representative's FASTA row and members by theirs, representative first, as for MMseqs2
(cluster_table.mmseqs_rows_to_members_tsv), so the order of the input lines does not
change the table or the validation split drawn over it.

This module is separate from cluster_table so that it is a DVC dependency of these two
cluster sources only.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.data.preparation import uniref_hierarchy as uh
from nanoplm.data.preparation.cluster_table import _Ids, write_pairs
from nanoplm.data.preparation.extractor import ZSTD_MAX_WINDOW, ZSTD_MISSING
from nanoplm.utils import logger

CL_TAG = b"CL:"
SOURCES = ("tsv", "headers")
_CHUNK = 1 << 26
_ID = re.compile(rb"^>(\S*)", re.M)
# the id, then a whitespace-separated CL:<value> token anywhere after it (b"" if none)
_ID_CL = re.compile(rb"^>(\S*)(?:[^\n]*?[ \t]CL:(\S*))?", re.M)
EXAMPLES = 3
_MMSEQS_IDS = ("MMseqs2 writes the accession of a UniProt-style header (P00001 for "
               "sp|P00001|NAME), not the first word")


class ClusterImportError(ValueError):
    """A cluster file does not fit the member FASTA; the message says what is wrong."""


class DecompressorMissing(OSError):
    """A .zst cannot be read: neither the zstandard package nor the zstd tool is there."""


# ------------------------------------------------------------------ reading


@contextlib.contextmanager
def open_decompressed(path):
    """A file for reading in binary, decompressed when it is .gz or .zst. A .zst is read
    with the zstandard package (the extractor's window limit), else with the zstd tool;
    a bad .zst raises OSError."""
    path = Path(path)
    if not str(path).endswith(".zst"):
        with (gzip.open if str(path).endswith(".gz") else open)(path, "rb") as f:
            yield f
        return
    try:
        import zstandard
    except ImportError:
        zstandard = None
    if zstandard is not None:
        # a --long=31 file needs more than the 128 MiB default window
        dctx = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW)
        with io.BufferedReader(zstandard.open(path, "rb", dctx=dctx)) as f:
            try:
                yield f
            except zstandard.ZstdError as e:
                raise OSError(f"{path}: {e}") from e
        return
    zstd = shutil.which("zstd")
    if zstd is None:
        raise DecompressorMissing(f"Reading {path} {ZSTD_MISSING}")
    proc = subprocess.Popen([zstd, "-dcq", "--memory=2048MB", str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        yield proc.stdout
    finally:
        proc.kill()
        proc.stdout.close()
        proc.wait()


def _blocks(path):
    """The file in chunks that end at a line end (the last one gets one added)."""
    tail = b""
    with open_decompressed(path) as f:
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
            yield block[:cut]
    if tail:
        yield tail + b"\n"


def _ids_of(parts: list, lens: list) -> _Ids:
    blob = np.frombuffer(b"".join(parts), dtype=np.uint8) if parts else np.zeros(0, np.uint8)
    lens = np.concatenate(lens) if lens else np.zeros(0, np.int32)
    return _Ids(blob, lens)


def fasta_ids(fasta, cl: bool = False):
    """The ids of every FASTA record, in order. With ``cl``, also the value of each
    header's CL: tag (b"" where there is none): returns (ids, cl_ids)."""
    parts, lens, cparts, clens = [], [], [], []
    for block in _blocks(fasta):
        if cl:
            pairs = _ID_CL.findall(block)
            if not pairs:
                continue
            ids, cls = zip(*pairs)
            cparts.append(b"".join(cls))
            clens.append(np.fromiter(map(len, cls), dtype=np.int32, count=len(cls)))
        else:
            ids = _ID.findall(block)
            if not ids:
                continue
        parts.append(b"".join(ids))
        lens.append(np.fromiter(map(len, ids), dtype=np.int32, count=len(ids)))
    ids = _ids_of(parts, lens)
    if len(ids) == 0:
        raise ClusterImportError(f"{fasta} has no FASTA records")
    if not np.all(ids.lens > 0):
        row = int(np.flatnonzero(ids.lens == 0)[0])
        raise ClusterImportError(f"{fasta}: record {row + 1} has a header with no id")
    return (ids, _ids_of(cparts, clens)) if cl else ids


def _keys(ids: _Ids) -> np.ndarray:
    return uh.hash_accessions(ids.blob, ids.lens, offsets=ids.offsets)


def _text(ids: _Ids, rows) -> list:
    return [bytes(ids.blob[ids.offsets[r]:ids.offsets[r] + ids.lens[r]]).decode(
        errors="replace") for r in rows]


def _eg(names) -> str:
    return "e.g. " + ", ".join(names[:EXAMPLES])


class _Index:
    """FASTA row of an id, by its 64-bit hash (as cluster_index joins the pool)."""

    def __init__(self, ids: _Ids, fasta):
        keys = _keys(ids)
        self.order = np.argsort(keys, kind="stable")
        self.sorted = keys[self.order]
        dup = np.flatnonzero(self.sorted[1:] == self.sorted[:-1])
        if dup.size:
            rows = self.order[dup[:EXAMPLES]]
            raise ClusterImportError(
                f"{fasta}: {dup.size:,} record ids appear more than once "
                f"({_eg(_text(ids, rows))}); every FASTA id must be unique")

    def rows(self, keys: np.ndarray) -> np.ndarray:
        """Row of each key, -1 for a key that is no FASTA id."""
        if len(self.sorted) == 0:
            return np.full(len(keys), -1, dtype=np.int64)
        pos = np.clip(np.searchsorted(self.sorted, keys), 0, len(self.sorted) - 1)
        return np.where(self.sorted[pos] == keys, self.order[pos], -1).astype(np.int64)


# ------------------------------------------------------------------ writing


def _write_rows(rep: np.ndarray, mem: np.ndarray, ids: _Ids, out_tsv) -> dict:
    """members.tsv from (representative row, member row) pairs that list every row once,
    each representative as its own member."""
    reps, cid = np.unique(rep, return_inverse=True)
    order = np.lexsort((mem, mem != rep, cid))
    rep, mem = rep[order], mem[order]
    sizes = np.bincount(cid, minlength=len(reps))
    del order, cid
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with atomic_write_path(Path(out_tsv)) as tmp, open(tmp, "wb", buffering=1 << 24) as out:
        write_pairs(out, ids.blob, ids.offsets[rep], ids.lens[rep],
                    ids.blob, ids.offsets[mem], ids.lens[mem])
    return {"clusters": int(len(reps)), "members": int(len(mem)),
            "singletons": int((sizes == 1).sum())}


def _write_info(out_tsv, stats: dict) -> None:
    """members.json: what the table is (cluster_table.build_members_table's rule: no path,
    no timing, so an identical rebuild reruns nothing after it)."""
    with open(Path(out_tsv).with_suffix(".json"), "w") as f:
        json.dump(stats, f, indent=1)


# ------------------------------------------------------------------ the TSV


def _tsv_blocks(tsv):
    """Per chunk of the file: (bytes, representative keys and spans, member keys and
    spans). A line that is not two non-empty ids around one tab is an error; empty lines
    are skipped."""
    line0 = 0
    for block in _blocks(tsv):
        buf = np.frombuffer(block, dtype=np.uint8)
        ends = np.flatnonzero(buf == 10)
        starts = np.empty_like(ends)
        starts[0] = 0
        starts[1:] = ends[:-1] + 1
        stop = ends.copy()
        cr = (stop > starts) & (buf[np.maximum(stop - 1, 0)] == 13)
        stop[cr] -= 1
        tabs = np.flatnonzero(buf == 9)
        first = np.searchsorted(tabs, starts)
        ntabs = np.searchsorted(tabs, stop) - first
        tab = tabs[np.minimum(first, len(tabs) - 1)] if len(tabs) else stop
        empty = stop == starts
        bad = ~empty & ((ntabs != 1) | (tab == starts) | (tab + 1 == stop))
        if bad.any():
            k = int(np.flatnonzero(bad)[0])
            line = bytes(block[starts[k]:stop[k]])[:80].decode(errors="replace")
            raise ClusterImportError(
                f"{tsv}: line {line0 + k + 1:,} is not representative<TAB>member: {line!r}")
        line0 += len(ends)
        s, t, e = starts[~empty], tab[~empty], stop[~empty]
        yield (buf, (uh.hash_accessions(buf, t - s, offsets=s), s, t - s),
               (uh.hash_accessions(buf, e - t - 1, offsets=t + 1), t + 1, e - t - 1))


def _tsv_keys(tsv):
    """(representative keys, member keys) of every line."""
    reps, mems = [], []
    for _, (rk, _, _), (mk, _, _) in _tsv_blocks(tsv):
        reps.append(rk)
        mems.append(mk)
    if not reps or sum(len(r) for r in reps) == 0:
        raise ClusterImportError(f"{tsv} is empty")
    return np.concatenate(reps), np.concatenate(mems)


def _tsv_ids_with_keys(tsv, wanted: np.ndarray) -> list:
    """A few ids of the TSV whose key is in ``wanted``, for an error message."""
    found: list = []
    for buf, *columns in _tsv_blocks(tsv):
        for keys, off, lens in columns:
            for i in np.flatnonzero(np.isin(keys, wanted))[:EXAMPLES]:
                name = bytes(buf[off[i]:off[i] + lens[i]]).decode(errors="replace")
                if name not in found:
                    found.append(name)
        if len(found) >= EXAMPLES:
            break
    return found[:EXAMPLES]


def tsv_members_tsv(tsv, fasta, out_tsv) -> dict:
    """members.tsv from a representative<TAB>member file over the member FASTA."""
    t0 = time.time()
    tsv, fasta = Path(tsv), Path(fasta)
    logger.info(f"cluster table: reading the ids of {fasta}")
    ids = fasta_ids(fasta)
    n = len(ids)
    index = _Index(ids, fasta)
    logger.info(f"cluster table: reading {tsv}")
    rep_keys, mem_keys = _tsv_keys(tsv)
    rep, mem = index.rows(rep_keys), index.rows(mem_keys)
    unknown = np.unique(np.concatenate([rep_keys[rep < 0], mem_keys[mem < 0]]))
    del rep_keys, mem_keys
    if unknown.size:
        raise ClusterImportError(
            f"{unknown.size:,} ids in {tsv} are not records of {fasta} "
            f"({_eg(_tsv_ids_with_keys(tsv, unknown))}; the FASTA's ids look like "
            f"{', '.join(_text(ids, range(min(EXAMPLES, n))))}). The cluster file must use "
            "FASTA ids, the first word of each header without '>'. If it comes from "
            f"MMseqs2: {_MMSEQS_IDS}; give MMseqs2 a FASTA whose headers are plain ids")
    seen = np.bincount(mem, minlength=n)
    twice = np.flatnonzero(seen > 1)
    if twice.size:
        raise ClusterImportError(
            f"{twice.size:,} sequences are listed as a member more than once in {tsv} "
            f"({_eg(_text(ids, twice))}); every sequence belongs to one cluster")
    missing = np.flatnonzero(seen == 0)
    if missing.size:
        raise ClusterImportError(
            f"{missing.size:,} of the {n:,} records of {fasta} are not in {tsv} "
            f"({_eg(_text(ids, missing))}). The cluster file must list every sequence of the "
            "FASTA (before the length filter) once; a partial file is usually one made "
            "from another FASTA")
    del seen, twice, missing
    reps = np.unique(rep)
    no_self = reps[~np.isin(reps, mem[rep == mem])]
    if no_self.size:
        raise ClusterImportError(
            f"{no_self.size:,} representatives in {tsv} have no line of their own "
            f"(representative<TAB>itself; {_eg(_text(ids, no_self))}); every "
            "representative must also be listed as a member of its own cluster")
    stats = _write_rows(rep, mem, ids, out_tsv)
    stats["source"] = "tsv"
    logger.info(f"cluster table: {stats['clusters']:,} clusters of {stats['members']:,} "
                f"sequences from {tsv.name} ({time.time() - t0:.1f}s)")
    return stats


# ------------------------------------------------------------------ CL: headers


def headers_members_tsv(fasta, out_tsv) -> dict:
    """members.tsv from the CL:<cluster id> tag of every FASTA header."""
    t0 = time.time()
    fasta = Path(fasta)
    logger.info(f"cluster table: reading the CL: tags of {fasta}")
    ids, cls = fasta_ids(fasta, cl=True)
    n = len(ids)
    untagged = np.flatnonzero(cls.lens == 0)
    if untagged.size:
        raise ClusterImportError(
            f"{untagged.size:,} of the {n:,} records of {fasta} have no CL: tag "
            f"({_eg(_text(ids, untagged))}). With clusters from the headers, every header "
            "names its cluster after the id: >id<TAB>CL:<cluster id>")
    index = _Index(ids, fasta)
    ckeys = _keys(cls)
    rep = index.rows(ckeys)
    orphan = np.flatnonzero(rep < 0)
    if orphan.size:
        names = list(dict.fromkeys(_text(cls, orphan[:1000])))
        n_clusters = len(np.unique(ckeys[orphan]))
        raise ClusterImportError(
            f"{n_clusters:,} clusters of {fasta} have no representative, no record whose id "
            f"is the cluster id ({_eg(['CL:' + c for c in names])}; {orphan.size:,} records "
            "name them). Every cluster needs its representative in the file")
    mem = np.arange(n, dtype=np.int64)
    reps = np.unique(rep)
    other = reps[rep[reps] != reps]
    if other.size:
        raise ClusterImportError(
            f"{other.size:,} representatives of {fasta} name another cluster in their own "
            f"CL: tag ({_eg(_text(ids, other))}); a representative's tag is its own id")
    stats = _write_rows(rep, mem, ids, out_tsv)
    stats["source"] = "headers"
    logger.info(f"cluster table: {stats['clusters']:,} clusters of {stats['members']:,} "
                f"sequences from the CL: tags ({time.time() - t0:.1f}s)")
    return stats


def import_members_table(source: str, out_tsv, *, fasta, tsv: Optional[str] = None) -> dict:
    """members.tsv and members.json for the tsv or headers cluster source. ``fasta`` is
    the member FASTA before the length filter."""
    if source == "tsv":
        if tsv is None:
            raise ValueError("the tsv cluster source needs the cluster file")
        stats = tsv_members_tsv(tsv, fasta, out_tsv)
    elif source == "headers":
        stats = headers_members_tsv(fasta, out_tsv)
    else:
        raise ValueError(f"unknown cluster source {source!r}")
    _write_info(out_tsv, stats)
    return stats
