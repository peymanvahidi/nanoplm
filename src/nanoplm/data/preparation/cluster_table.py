"""The canonical cluster table, ``cluster_index/members.tsv``, from either cluster source.

Format: one ``cluster_id<TAB>sequence_id`` line per sequence. Lines are grouped by
cluster, and the first line of each group is the cluster's representative. Sequence
ids are FASTA ids (the first word of the header, without ``>``); each appears once.
Cluster order is the order of first appearance (for MMseqs2: the representative's
row in the member FASTA, then the other members in FASTA order).

Two sources write it:
  uniref50.xml    ``uniref_hierarchy.xml_to_members_tsv`` (UniProt UniRef50 clusters)
  MMseqs2         ``mmseqs_members_tsv``: ``easy-linclust`` over the filtered member
                  FASTA with --min-seq-id identity -c coverage --cov-mode 0 (the
                  alignment must cover that fraction of both sequences). MMseqs2 reads
                  its own id out of a header (the accession of ``sp|P00001|X``), so it
                  runs on a copy whose headers are row numbers, mapped back after.

The cluster index is built from this table alone (``read_members_table``), whatever
the source.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.data.preparation import uniref_hierarchy as uh
from nanoplm.utils import logger

COV_MODE = 0
MMSEQS_INSTALL_HINT = (
    "Install it with `nanoplm deps install --mmseqs-only` (static binary into the venv), "
    "or put a binary from https://github.com/soedinglab/MMseqs2/releases on PATH, or set "
    "NANOPLM_MMSEQS=/path/to/mmseqs.")


class ClusterTableError(RuntimeError):
    """A cluster table is malformed or does not fit the member FASTA."""


@dataclass
class MembersTable:
    keys: np.ndarray        # uint64[P], sorted strictly increasing: hash of each sequence id
    cluster: np.ndarray     # int32[P], aligned with keys: cluster index (first appearance)
    is_rep: np.ndarray      # bool[P], aligned with keys
    sizes: np.ndarray       # int32[C]: members per cluster in the table


def _hash_blob(blob: bytearray, lens: array) -> np.ndarray:
    return uh.hash_accessions(np.frombuffer(bytes(blob), dtype=np.uint8),
                              np.frombuffer(bytes(lens), dtype=np.uint16).astype(np.int64))


def read_members_table(path) -> MembersTable:
    """Parse a canonical members.tsv into sorted hash keys + aligned cluster / rep arrays."""
    path = Path(path)
    mblob, mlens = bytearray(), array("H")
    cblob, clens = bytearray(), array("H")
    sizes = array("i")
    prev = None
    n = 0
    with open(path, "rb", buffering=1 << 24) as f:
        for line in f:
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            c, tab, m = line.partition(b"\t")
            if not tab or not c or not m or b"\t" in m:
                raise ClusterTableError(f"{path}: line {n + 1} is not cluster_id<TAB>sequence_id")
            if len(m) > 65535 or len(c) > 65535:
                raise ClusterTableError(f"{path}: line {n + 1} has an id over 64 KB")
            if c != prev:
                cblob.extend(c)
                clens.append(len(c))
                sizes.append(0)
                prev = c
            sizes[-1] += 1
            mblob.extend(m)
            mlens.append(len(m))
            n += 1
    if n == 0:
        raise ClusterTableError(f"{path} is empty")
    sizes = np.frombuffer(bytes(sizes), dtype=np.int32).copy()
    ckeys = _hash_blob(cblob, clens)
    del cblob
    if len(np.unique(ckeys)) != len(ckeys):
        raise ClusterTableError(f"{path}: a cluster id appears in two separate groups; the "
                                "table must be grouped by cluster")
    del ckeys
    keys = _hash_blob(mblob, mlens)
    del mblob
    cluster = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
    is_rep = np.zeros(n, dtype=bool)
    is_rep[np.cumsum(sizes, dtype=np.int64) - sizes] = True
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    dup = np.flatnonzero(keys[1:] == keys[:-1])
    if dup.size:
        raise ClusterTableError(f"{path}: {dup.size} sequence ids appear twice (or collide "
                                "under the 64-bit id hash); every sequence must be listed once")
    return MembersTable(keys, cluster[order], is_rep[order], sizes)


class _Ids:
    """Sequence ids as one byte blob plus offsets (no Python object per id)."""

    def __init__(self, blob: np.ndarray, lens: np.ndarray):
        self.blob = blob
        self.lens = lens
        self.offsets = np.zeros(len(lens), dtype=np.int64)
        np.cumsum(self.lens[:-1], out=self.offsets[1:])

    def __len__(self) -> int:
        return len(self.lens)


def number_fasta(fasta, out_fasta) -> _Ids:
    """Copy ``fasta`` with headers replaced by row numbers (>0, >1, ...); return its ids."""
    blob, lens = bytearray(), array("i")
    n = 0
    with open(fasta, "rb", buffering=1 << 24) as f, \
            open(out_fasta, "wb", buffering=1 << 24) as out:
        for line in f:
            if line[:1] == b">":
                parts = line[1:].split(None, 1)
                if not parts:
                    raise ClusterTableError(f"{fasta}: a header has no id")
                blob.extend(parts[0])
                lens.append(len(parts[0]))
                out.write(b">%d\n" % n)
                n += 1
            else:
                out.write(line)
    return _Ids(np.frombuffer(blob, dtype=np.uint8), np.frombuffer(lens, dtype=np.int32))


def _gather(off: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Positions of the byte runs [off, off+len), concatenated."""
    starts = np.repeat(off - (np.cumsum(lens) - lens), lens)
    return starts + np.arange(int(lens.sum()), dtype=np.int64)


def write_pairs(out, c_src, c_off, c_len, m_src, m_off, m_len, chunk: int = 1 << 14) -> None:
    """Write ``cluster<TAB>member`` lines, the ids sliced out of two byte blobs."""
    for a in range(0, len(c_off), chunk):
        co, cl = c_off[a:a + chunk], c_len[a:a + chunk].astype(np.int64)
        mo, ml = m_off[a:a + chunk], m_len[a:a + chunk].astype(np.int64)
        width = cl + ml + 2
        end = np.cumsum(width)
        start = end - width
        buf = np.empty(int(end[-1]), dtype=np.uint8)
        buf[_gather(start, cl)] = c_src[_gather(co, cl)]
        buf[start + cl] = 9
        buf[_gather(start + cl + 1, ml)] = m_src[_gather(mo, ml)]
        buf[end - 1] = 10
        out.write(buf.tobytes())


def find_mmseqs(explicit: Optional[str] = None) -> Optional[str]:
    """The mmseqs binary: explicit path, $NANOPLM_MMSEQS, PATH, then the venv's bin/."""
    import sys
    for cand in (explicit, os.environ.get("NANOPLM_MMSEQS")):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return str(cand)
    found = shutil.which("mmseqs")
    if found:
        return found
    venv = Path(sys.prefix) / "bin" / "mmseqs"
    return str(venv) if venv.is_file() and os.access(venv, os.X_OK) else None


def mmseqs_rows_to_members_tsv(cluster_tsv, ids: _Ids, out_tsv) -> dict:
    """members.tsv from MMseqs2's rep<TAB>member table over row-numbered headers.

    Every row must be listed exactly once: a sequence MMseqs2 left out would otherwise
    become a silent singleton.
    """
    n = len(ids)
    pairs = np.fromfile(cluster_tsv, dtype=np.int64, sep=" ")
    if pairs.size % 2:
        raise ClusterTableError(f"{cluster_tsv}: not rep<TAB>member pairs of row numbers")
    rep, mem = pairs[0::2], pairs[1::2]
    del pairs
    if rep.size and (min(rep.min(), mem.min()) < 0 or max(rep.max(), mem.max()) >= n):
        raise ClusterTableError(f"{cluster_tsv}: a row number outside 0..{n - 1}")
    seen = np.bincount(mem, minlength=n)
    if not np.all(seen == 1):
        missing, twice = int((seen == 0).sum()), int((seen > 1).sum())
        raise ClusterTableError(
            f"MMseqs2 listed {len(mem):,} members for {n:,} sequences ({missing:,} missing, "
            f"{twice:,} listed more than once); refusing to turn the missing ones into "
            f"singleton clusters")
    del seen
    # np.unique numbers the clusters by their representative's FASTA row, so the
    # order (and the validation split drawn over it) does not depend on how MMseqs2
    # ordered its output
    reps, cid = np.unique(rep, return_inverse=True)
    if not np.all(np.isin(reps, mem[rep == mem])):
        raise ClusterTableError(f"{cluster_tsv}: a representative is not its own member")
    del reps
    order = np.lexsort((mem, mem != rep, cid))
    rep, mem = rep[order], mem[order]
    del order, cid
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with atomic_write_path(Path(out_tsv)) as tmp, open(tmp, "wb", buffering=1 << 24) as out:
        write_pairs(out, ids.blob, ids.offsets[rep], ids.lens[rep],
                    ids.blob, ids.offsets[mem], ids.lens[mem])
    return {"clusters": int(np.count_nonzero(rep == mem)), "members": int(n),
            "singletons_added": 0}


def mmseqs_members_tsv(fasta, out_tsv, identity: float, coverage: float, tmp_dir,
                       threads: Optional[int] = None, mmseqs: Optional[str] = None) -> dict:
    """Cluster the member FASTA with MMseqs2 easy-linclust and write members.tsv."""
    binary = find_mmseqs(mmseqs)
    if binary is None:
        raise ClusterTableError("cluster_subsampling needs MMseqs2 to cluster these "
                                "sequences, but `mmseqs` was not found. " + MMSEQS_INSTALL_HINT)
    t0 = time.time()
    tmp_dir = Path(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    prefix = tmp_dir / "clu"
    numbered = tmp_dir / "rows.fasta"
    cmd = [binary, "easy-linclust", str(numbered), str(prefix), str(tmp_dir / "work"),
           "--min-seq-id", str(identity), "-c", str(coverage), "--cov-mode", str(COV_MODE),
           "--threads", str(threads if threads and threads > 0
                            else max(1, len(os.sched_getaffinity(0))))]
    try:
        ids = number_fasta(fasta, numbered)
        logger.info("cluster table: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ClusterTableError(f"mmseqs easy-linclust failed ({proc.returncode}):\n"
                                    + proc.stderr[-4000:])
        stats = mmseqs_rows_to_members_tsv(Path(f"{prefix}_cluster.tsv"), ids, out_tsv)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    stats.update(source="mmseqs2", input=str(fasta), identity=identity, coverage=coverage, cov_mode=COV_MODE,
                 seconds=round(time.time() - t0, 1))
    return stats


def detect_uniref_level(fasta: Path) -> str:
    """'uniref50', 'uniref90' or 'uniref100', from the first FASTA header."""
    with open(fasta, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                for level, n in (("uniref100", b">UniRef100_"), ("uniref90", b">UniRef90_"),
                                 ("uniref50", b">UniRef50_")):
                    if line.startswith(n):
                        return level
                raise ValueError(f"{fasta}: header {line[:40]!r} is not a UniRef header")
    raise ValueError(f"{fasta}: no FASTA header found")


def build_members_table(source: str, out_tsv: Path, *, fasta: Path, level: Optional[str] = None,
                        xml: Optional[Path] = None, identity: float = 0.5,
                        coverage: float = 0.8, tmp_dir: Optional[Path] = None,
                        mmseqs: Optional[str] = None, threads: Optional[int] = None) -> dict:
    """members.tsv (and members.json, its stats) from one of the two sources. ``fasta``
    is the member FASTA (filtered for mmseqs; any of them for the uniref_xml level check)."""
    out_tsv = Path(out_tsv)
    if source == "uniref_xml":
        got = detect_uniref_level(Path(fasta))
        if got != level:
            raise ValueError(f"{fasta} holds {got} sequences, but the plan expects {level}; "
                             "check cluster_subsampling.members and fasta")
        stats = uh.xml_to_members_tsv(xml, out_tsv, level)
    elif source == "mmseqs":
        stats = mmseqs_members_tsv(fasta, out_tsv, identity, coverage,
                                   tmp_dir or out_tsv.parent / ".mmseqs_tmp",
                                   threads=threads, mmseqs=mmseqs)
    else:
        raise ValueError(f"unknown cluster source {source!r}")
    with open(out_tsv.with_suffix(".json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats
