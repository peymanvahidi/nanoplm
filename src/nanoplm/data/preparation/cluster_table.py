"""The canonical cluster table, ``cluster_index/members.tsv``, from any cluster source.

Format: one ``cluster_id<TAB>sequence_id`` line per sequence. Lines are grouped by
cluster, and the first line of each group is the cluster's representative. Sequence
ids are FASTA ids (the first word of the header, without ``>``); each appears once.
Cluster order is the order of first appearance.

Three sources write it:
  uniref50.xml    ``uniref_hierarchy.xml_to_members_tsv`` (UniProt UniRef50 clusters)
  MMseqs2         ``mmseqs_members_tsv``: ``easy-linclust`` over the filtered member
                  FASTA with --min-seq-id identity -c coverage --cov-mode 0 (the
                  alignment must cover that fraction of both sequences)
  a user table    ``table_members_tsv``: cluster_id<TAB>sequence_id lines in any order;
                  the representative is the member whose id equals the cluster id, else
                  the first one listed. FASTA sequences missing from the table become
                  singleton clusters.

The cluster index is built from this table alone (``read_members_table``), whatever
the source.
"""
from __future__ import annotations

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
    "NANOPLM_MMSEQS=/path/to/mmseqs. Alternatively give your own table with "
    "cluster_subsampling.clusters: <path>.")


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


def fasta_ids(fasta) -> list:
    """FASTA ids (first header word) in file order."""
    out = []
    with open(fasta, "rb") as f:
        for line in f:
            if line[:1] == b">":
                parts = line[1:].split(None, 1)
                if not parts:
                    raise ClusterTableError(f"{fasta}: a header has no id")
                out.append(parts[0])
    return out


def table_members_tsv(table, fasta, out_tsv) -> dict:
    """Canonicalise a user cluster_id<TAB>sequence_id table against the member FASTA."""
    t0 = time.time()
    groups: dict = {}
    where: dict = {}
    with open(table, "rb") as f:
        for k, line in enumerate(f, 1):
            line = line.rstrip(b"\r\n")
            if not line or line.startswith(b"#"):
                continue
            parts = line.split(b"\t")
            if len(parts) < 2 or not parts[0] or not parts[1]:
                raise ClusterTableError(f"{table}: line {k} is not cluster_id<TAB>sequence_id")
            c, m = parts[0], parts[1]
            if m in where:
                if where[m] == c:
                    continue
                raise ClusterTableError(f"{table}: sequence {m.decode()} is in clusters "
                                        f"{where[m].decode()} and {c.decode()}")
            where[m] = c
            groups.setdefault(c, []).append(m)
    if not groups:
        raise ClusterTableError(f"{table} holds no clusters")
    n_table = len(groups)
    singletons = 0
    for m in fasta_ids(fasta):
        if m not in where:
            if m in groups:
                raise ClusterTableError(f"{fasta}: sequence {m.decode()} is missing from the "
                                        "table but its id is used as a cluster id there")
            groups[m] = [m]
            where[m] = m
            singletons += 1
    del where
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with atomic_write_path(Path(out_tsv)) as tmp, open(tmp, "wb", buffering=1 << 24) as out:
        for c, ms in groups.items():
            if c in ms and ms[0] != c:
                ms = [c] + [m for m in ms if m != c]
            out.write(b"".join(c + b"\t" + m + b"\n" for m in ms))
    stats = {"source": "table", "input": str(table), "clusters_in_table": n_table,
             "singletons_added": singletons, "clusters": len(groups),
             "members": sum(len(v) for v in groups.values()),
             "seconds": round(time.time() - t0, 1)}
    if singletons:
        logger.info(f"cluster table: {singletons:,} FASTA sequences are not in {table}; "
                    "each becomes its own cluster")
    return stats


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
    cmd = [binary, "easy-linclust", str(fasta), str(prefix), str(tmp_dir / "work"),
           "--min-seq-id", str(identity), "-c", str(coverage), "--cov-mode", str(COV_MODE),
           "--threads", str(threads or max(1, len(os.sched_getaffinity(0))))]
    logger.info("cluster table: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ClusterTableError(f"mmseqs easy-linclust failed ({proc.returncode}):\n"
                                    + proc.stderr[-4000:])
        stats = table_members_tsv(Path(f"{prefix}_cluster.tsv"), fasta, out_tsv)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    stats.update(source="mmseqs2", identity=identity, coverage=coverage, cov_mode=COV_MODE,
                 seconds=round(time.time() - t0, 1))
    return stats
