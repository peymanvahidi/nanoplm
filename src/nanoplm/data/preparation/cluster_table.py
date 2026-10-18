"""The canonical cluster table, ``cluster_index/members.tsv``, from any cluster source.

Format: one ``cluster_id<TAB>sequence_id`` line per sequence. Lines are grouped by
cluster, and the first line of each group is the cluster's representative. Sequence
ids are FASTA ids (the first word of the header, without ``>``); each appears once.
Cluster order is the order of first appearance.

Three sources write it:
  uniref50.xml    ``uniref_hierarchy.xml_to_members_tsv`` (UniProt UniRef50 clusters)
  MMseqs2         ``mmseqs_members_tsv``: ``easy-linclust`` over the filtered member
                  FASTA with --min-seq-id identity -c coverage --cov-mode 0 (the
                  alignment must cover that fraction of both sequences). MMseqs2 reads
                  its own id out of a header (the accession of ``sp|P00001|X``), so it
                  runs on a copy whose headers are row numbers, mapped back after.
  a user table    ``table_members_tsv``: cluster_id<TAB>sequence_id lines in any order;
                  the representative is the member whose id equals the cluster id, else
                  the first one listed, among the members in the filtered FASTA when
                  any is (so a rep the length filter dropped does not take its whole
                  cluster with it). FASTA sequences missing from the table become
                  singleton clusters. Too few table ids in the FASTA is an error (the
                  ids are probably written another way); a partial match is a warning.

The cluster index is built from this table alone (``read_members_table``), whatever
the source.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from array import array
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.data.preparation import uniref_hierarchy as uh
from nanoplm.utils import logger

COV_MODE = 0
# Refuse a user table when fewer than this fraction of its ids are in the FASTA and
# fewer than this fraction of the FASTA's ids are in the table.
MIN_TABLE_MATCH = 0.5
N_EXAMPLES = 5
# hash_accessions builds an (ids x length) int64 index per batch; keep that bounded
HASH_BATCH = 1 << 16
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
    """FASTA ids (first header word) in file order, as bytes."""
    ids = scan_fasta_ids(fasta)
    return [_id(ids.blob, o, n) for o, n in zip(ids.offsets, ids.lens)]


def _examples(ids) -> str:
    return ", ".join(i.decode(errors="replace") for i in ids[:N_EXAMPLES])


def check_table_match(table, fasta, n_table: int, missing_table: int, ex_table: list,
                      n_fasta: int, missing_fasta: int, ex_fasta: list) -> None:
    """Refuse a table whose ids mostly miss the FASTA; warn about a partial match.

    ``missing_*`` count the ids not found on the other side; ``ex_*`` are a few of them.
    """
    t_frac = 1 - missing_table / n_table if n_table else 1.0
    f_frac = 1 - missing_fasta / n_fasta if n_fasta else 1.0
    if t_frac < MIN_TABLE_MATCH and f_frac < MIN_TABLE_MATCH:
        raise ClusterTableError(
            f"{table}: only {t_frac:.1%} of its sequence ids are in {fasta}, and only "
            f"{f_frac:.1%} of the FASTA's ids are in the table. Sequence ids must be the "
            f"first word of the FASTA header. Table ids not in the FASTA: "
            f"{_examples(ex_table)}; FASTA ids not in the table: {_examples(ex_fasta)}")
    if missing_table:
        logger.warning(
            f"cluster table: {missing_table:,} of {n_table:,} sequence ids in {table} "
            f"({1 - t_frac:.1%}) are not in {fasta} (removed by the check or length filter, "
            f"or written differently), e.g. {_examples(ex_table)}")


class _Ids:
    """Sequence ids as one byte blob plus offsets (no Python object per id)."""

    def __init__(self, blob: np.ndarray, lens: np.ndarray):
        self.blob = blob
        self.lens = lens
        self.offsets = np.zeros(len(lens), dtype=np.int64)
        np.cumsum(self.lens[:-1], out=self.offsets[1:])

    def hashes(self) -> np.ndarray:
        return uh.hash_accessions(self.blob, self.lens, offsets=self.offsets, batch=HASH_BATCH)

    def __len__(self) -> int:
        return len(self.lens)


def scan_fasta_ids(fasta) -> _Ids:
    """FASTA ids (first header word) in file order."""
    return number_fasta(fasta, None)


def number_fasta(fasta, out_fasta) -> _Ids:
    """Copy ``fasta`` with headers replaced by row numbers (>0, >1, ...); return its ids.
    With ``out_fasta`` None, only the ids are read."""
    blob, lens = bytearray(), array("i")
    n = 0
    with open(fasta, "rb", buffering=1 << 24) as f, \
            (open(out_fasta, "wb", buffering=1 << 24) if out_fasta else nullcontext()) as out:
        for line in f:
            if line[:1] == b">":
                parts = line[1:].split(None, 1)
                if not parts:
                    raise ClusterTableError(f"{fasta}: a header has no id")
                blob.extend(parts[0])
                lens.append(len(parts[0]))
                if out is not None:
                    out.write(b">%d\n" % n)
                n += 1
            elif out is not None:
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


def _find(buf: np.ndarray, byte: int, chunk: int = 1 << 26) -> np.ndarray:
    """np.flatnonzero(buf == byte) without a file-sized bool temporary."""
    parts = [np.flatnonzero(buf[a:a + chunk] == byte) + a for a in range(0, len(buf), chunk)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _parse_table(table):
    """A cluster_id<TAB>sequence_id table as its bytes plus per-line id positions.

    Returns (buf, c_off, c_len, m_len): the member id of a line starts at
    c_off + c_len + 1. Blank lines and ``#`` comments are skipped; columns after the
    second are ignored.
    """
    buf = np.fromfile(table, dtype=np.uint8)
    if buf.size and buf[-1] != 10:
        buf = np.append(buf, np.uint8(10))
    ends = _find(buf, 10)
    starts = np.empty_like(ends)
    starts[:1] = 0
    starts[1:] = ends[:-1] + 1
    cr = (ends > starts) & (buf[ends - 1] == 13)
    ends[cr] -= 1
    del cr
    keep = (ends > starts) & (buf[np.minimum(starts, len(buf) - 1)] != ord("#"))
    line_no = np.flatnonzero(keep) + 1
    starts, ends = starts[keep], ends[keep]
    del keep
    tabs = _find(buf, 9)
    if not tabs.size:
        tabs = np.array([len(buf)], dtype=np.int64)
    i = np.searchsorted(tabs, starts)
    np.minimum(i, len(tabs) - 1, out=i)
    t1 = tabs[i]
    t2 = tabs[np.minimum(i + 1, len(tabs) - 1)]
    del i
    m_end = np.where((t2 > t1) & (t2 < ends), t2, ends)
    del t2
    c_len = (t1 - starts).astype(np.int32)
    m_len = (m_end - t1 - 1).astype(np.int32)
    del m_end
    bad = (t1 >= ends) | (c_len <= 0) | (m_len <= 0)
    if bad.any():
        raise ClusterTableError(f"{table}: line {int(line_no[np.argmax(bad)])} is not "
                                "cluster_id<TAB>sequence_id")
    return buf, starts, c_len, m_len


def _id(buf, off, length) -> bytes:
    return buf[int(off):int(off) + int(length)].tobytes()


def table_members_tsv(table, fasta, out_tsv) -> dict:
    """Canonicalise a user cluster_id<TAB>sequence_id table against the member FASTA.

    Arrays, not dicts: ids are hashed (64-bit, as in read_members_table) and joined by
    sorting, so memory stays near the table's own size plus ~60 bytes per line.
    """
    t0 = time.time()
    buf, c_off, c_len, m_len = _parse_table(table)
    if not len(c_off):
        raise ClusterTableError(f"{table} holds no clusters")
    m_off = c_off + c_len + 1
    ch = uh.hash_accessions(buf, c_len, offsets=c_off, batch=HASH_BATCH)
    mh = uh.hash_accessions(buf, m_len, offsets=m_off, batch=HASH_BATCH)

    # a member listed twice: the same cluster again is dropped, another one is refused
    order = np.argsort(mh, kind="stable")
    same = mh[order[1:]] == mh[order[:-1]]
    clash = same & (ch[order[1:]] != ch[order[:-1]])
    if clash.any():
        k = int(np.argmax(clash))
        a, b = order[k], order[k + 1]
        raise ClusterTableError(f"{table}: sequence {_id(buf, m_off[a], m_len[a]).decode()} is "
                                f"in clusters {_id(buf, c_off[a], c_len[a]).decode()} and "
                                f"{_id(buf, c_off[b], c_len[b]).decode()}")
    kept = np.ones(len(mh), dtype=bool)
    kept[order[1:][same]] = False
    del order, same, clash
    if not kept.all():
        c_off, c_len, m_off, m_len = c_off[kept], c_len[kept], m_off[kept], m_len[kept]
        ch, mh = ch[kept], mh[kept]
    del kept

    # clusters numbered by first appearance
    uc, first, cid = np.unique(ch, return_index=True, return_inverse=True)
    rank = np.empty(len(uc), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(uc))
    cid = rank[cid].astype(np.int32)
    del rank, first

    ids = scan_fasta_ids(fasta)
    fh = ids.hashes()
    fs = np.sort(fh)
    pos = np.minimum(np.searchsorted(fs, mh), max(len(fs) - 1, 0))
    in_fasta = fs[pos] == mh if len(fs) else np.zeros(len(mh), dtype=bool)
    del fs, pos
    ms = np.sort(mh)
    pos = np.minimum(np.searchsorted(ms, fh), len(ms) - 1)
    lone = np.flatnonzero(ms[pos] != fh)          # FASTA rows not in the table
    del ms, pos
    miss_t = np.flatnonzero(~in_fasta)
    check_table_match(
        table, fasta, len(mh), len(miss_t),
        [_id(buf, m_off[r], m_len[r]) for r in miss_t[:N_EXAMPLES]],
        len(ids), len(lone),
        [_id(ids.blob, ids.offsets[r], ids.lens[r]) for r in lone[:N_EXAMPLES]])
    del miss_t
    pos = np.minimum(np.searchsorted(uc, fh[lone]), len(uc) - 1)
    named = np.flatnonzero(uc[pos] == fh[lone])
    if named.size:
        r = lone[named[0]]
        raise ClusterTableError(f"{fasta}: sequence "
                                f"{_id(ids.blob, ids.offsets[r], ids.lens[r]).decode()} is "
                                "missing from the table but its id is used as a cluster id there")
    del pos, named, fh

    # grouped by cluster, representative first, the rest in table order. The rep is the
    # member named like its cluster, else the first listed, preferring members that are
    # in the (filtered) FASTA: a rep the filter dropped would take its cluster with it.
    line = np.arange(len(mh))
    named = mh == ch
    pref = np.where(in_fasta, 0, 2) + np.where(named, 0, 1)
    del in_fasta
    order = np.lexsort((line, pref, cid))
    is_rep = np.zeros(len(mh), dtype=bool)
    first = np.ones(len(order), dtype=bool)
    first[1:] = cid[order[1:]] != cid[order[:-1]]
    is_rep[order[first]] = True
    del pref, first, named
    order = np.lexsort((line, ~is_rep, cid))
    del cid, ch, mh, line, is_rep
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with atomic_write_path(Path(out_tsv)) as tmp, open(tmp, "wb", buffering=1 << 24) as out:
        write_pairs(out, buf, c_off[order], c_len[order], buf, m_off[order], m_len[order])
        write_pairs(out, ids.blob, ids.offsets[lone], ids.lens[lone],
                    ids.blob, ids.offsets[lone], ids.lens[lone])
    stats = {"source": "table", "input": str(table), "clusters_in_table": len(uc),
             "singletons_added": len(lone), "clusters": len(uc) + len(lone),
             "members": len(order) + len(lone), "seconds": round(time.time() - t0, 1)}
    if len(lone):
        logger.info(f"cluster table: {len(lone):,} FASTA sequences are not in {table}; "
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
    reps, first, cid = np.unique(rep, return_index=True, return_inverse=True)
    if not np.all(np.isin(reps, mem[rep == mem])):
        raise ClusterTableError(f"{cluster_tsv}: a representative is not its own member")
    rank = np.empty(len(reps), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(reps))
    cid = rank[cid]
    del rank, first, reps
    order = np.lexsort((np.arange(len(mem)), mem != rep, cid))
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
