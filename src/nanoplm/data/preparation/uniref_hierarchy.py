"""UniRef50 cluster membership from uniref50.xml(.gz), written as a members table.

Streams the XML through ``pigz -dc | grep -F`` (``gzip -dc`` without pigz; a
pure-Python filter without grep) on the few line shapes it needs and writes the
canonical cluster table (see ``cluster_table``): one ``cluster_id<TAB>sequence_id``
line per distinct UniRef90 (or UniRef100) member, grouped by UniRef50 entry in XML
order, the cluster's representative first. Ids are the full FASTA ids
(``UniRef50_X``, ``UniRef90_Y``), so the table joins against the member FASTA headers.

Representatives:
  uniref90   the UniRef90 id inside ``<representativeMember>``.
  uniref100  the *paired* rep: ``UniRef100_<acc>`` where ``UniRef90_<acc>`` is the
             entry's UniRef90 representative, which is the same sequence as the
             UniRef90 pool's rep. So a UniRef100 build has the same universe, train and
             val clusters as the UniRef90 build of the same release. If the entry has no
             such UniRef100 id, the literal representativeMember's id is used and counted.

An entry with no or two representative blocks, or a member id before the
representative block, fails loudly. The 64-bit accession hash used for joins
(FNV-1a-64 + murmur3 fmix64) lives here too.

Ported from the data-sampler experiment's extract_hierarchy.py / extract_hierarchy100.py.
"""
from __future__ import annotations

import contextlib
import gzip
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.utils import logger

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
FMIX_C1 = 0xFF51AFD7ED558CCD
FMIX_C2 = 0xC4CEB9FE1A85EC53
MASK64 = (1 << 64) - 1
HASH_NAME = "fnv1a64+fmix64"

UNIREF_LEVELS = ("uniref90", "uniref100")
ENTRY_PREFIX = b'<entry id="UniRef50_'
U90_GREP = b'<property type="UniRef90 ID" value="'
U100_GREP = b'<property type="UniRef100 ID" value="'
U90_PREFIX = U90_GREP + b"UniRef90_"
U100_PREFIX = U100_GREP + b"UniRef100_"
REP_OPEN = b"<representativeMember>"
REP_CLOSE = b"</representativeMember>"
ROOT_CLOSE = b"</UniRef50>"
QUOTE = 0x22
READ_BLOCK = 1 << 22


class HierarchyError(RuntimeError):
    """The XML breaks an invariant the cluster table relies on."""


# ------------------------------------------------------------------- the hash


def fmix64(h: int) -> int:
    h &= MASK64
    h ^= h >> 33
    h = (h * FMIX_C1) & MASK64
    h ^= h >> 33
    h = (h * FMIX_C2) & MASK64
    h ^= h >> 33
    return h


def hash_one(acc: bytes, salt: int = 0) -> int:
    """Reference accession -> uint64 key map."""
    h = (FNV_OFFSET ^ (salt & MASK64)) & MASK64
    for b in acc:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return fmix64(h)


def _fmix64_vec(h: np.ndarray) -> np.ndarray:
    shift = np.uint64(33)
    h = h ^ (h >> shift)
    h = h * np.uint64(FMIX_C1)
    h = h ^ (h >> shift)
    h = h * np.uint64(FMIX_C2)
    return h ^ (h >> shift)


def offsets_from_lengths(lengths: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], dtype=np.int64, out=offsets[1:])
    return offsets


def hash_accessions(blob: np.ndarray, lengths: np.ndarray, salt: int = 0,
                    offsets: Optional[np.ndarray] = None, batch: int = 1 << 23) -> np.ndarray:
    """Vectorized ``hash_one`` over accessions concatenated in ``blob``.

    Rows are grouped by exact length so each group folds as a dense (n, L) byte
    matrix with no padding byte absorbed, which keeps it bitwise equal to hash_one.
    """
    lengths = np.asarray(lengths)
    n = len(lengths)
    if offsets is None:
        offsets = offsets_from_lengths(lengths)
    keys = np.empty(n, dtype=np.uint64)
    h0 = np.uint64(FNV_OFFSET ^ (salt & MASK64))
    prime = np.uint64(FNV_PRIME)
    with np.errstate(over="ignore"):
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            lens = lengths[start:stop].astype(np.int64, copy=False)
            offs = offsets[start:stop]
            out = np.empty(stop - start, dtype=np.uint64)
            for length in np.unique(lens):
                length = int(length)
                rows = np.flatnonzero(lens == length)
                mat = blob[offs[rows, None] + np.arange(length, dtype=np.int64)[None, :]]
                h = np.full(len(rows), h0, dtype=np.uint64)
                for j in range(length):
                    h = (h ^ mat[:, j].astype(np.uint64)) * prime
                out[rows] = _fmix64_vec(h)
            keys[start:stop] = out
    return keys


def hash_strings(accs, salt: int = 0) -> np.ndarray:
    accs = list(accs)
    lengths = np.fromiter((len(a) for a in accs), dtype=np.int64, count=len(accs))
    blob = np.frombuffer(b"".join(accs), dtype=np.uint8)
    return hash_accessions(blob, lengths, salt=salt)


# --------------------------------------------------------- the filtered stream


def _patterns(level: str) -> list:
    # a UniRef100 table also needs the UniRef90 rep id, to pick the paired rep
    props = [U90_GREP] if level == "uniref90" else [U90_GREP, U100_GREP]
    return [ENTRY_PREFIX, *props, REP_OPEN, REP_CLOSE, ROOT_CLOSE]


@contextlib.contextmanager
def _filtered_lines(path: Path, level: str) -> Iterator[tuple]:
    """Yield (blocks, finish): blocks of grep-filtered XML lines; finish() -> (ok, detail)."""
    pats = _patterns(level)
    gz = str(path).endswith(".gz")
    grep = shutil.which("grep")
    if grep is None:
        with _python_filter(path, gz, pats) as src:
            yield src
        return
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    argv = [grep, "-F"] + [a for p in pats for a in ("-e", p.decode())]
    if gz:
        dec_bin = shutil.which("pigz") or shutil.which("gzip")
        if dec_bin is None:
            dec = None
            g = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 bufsize=0, env=env)
            failure: list[str] = []

            def pump():
                try:
                    with gzip.open(path, "rb") as f:
                        shutil.copyfileobj(f, g.stdin, READ_BLOCK)
                except (OSError, EOFError) as exc:
                    failure.append(f"python gzip failed: {exc!r}")
                finally:
                    with contextlib.suppress(OSError):
                        g.stdin.close()

            t = threading.Thread(target=pump, daemon=True)
            t.start()

            def finish():
                g.stdout.close()
                rc = g.wait()
                t.join()
                if failure:
                    return False, failure[0]
                return rc in (0, 1), f"grep exited {rc}"
        else:
            dec = subprocess.Popen([dec_bin, "-dc", str(path)], stdout=subprocess.PIPE,
                                   bufsize=0)
            g = subprocess.Popen(argv, stdin=dec.stdout, stdout=subprocess.PIPE,
                                 bufsize=0, env=env)
            dec.stdout.close()

            def finish():
                g.stdout.close()
                rc_g, rc_d = g.wait(), dec.wait()
                if rc_d != 0:
                    return False, f"{Path(dec_bin).name} exited {rc_d} (truncated or corrupt gzip)"
                return rc_g in (0, 1), f"grep exited {rc_g}"
    else:
        dec = None
        g = subprocess.Popen(argv + [str(path)], stdout=subprocess.PIPE, bufsize=0, env=env)

        def finish():
            g.stdout.close()
            rc = g.wait()
            return rc in (0, 1), f"grep exited {rc}"
    try:
        yield iter(lambda: g.stdout.read(READ_BLOCK), b""), finish
    finally:
        for p in (g, dec):
            if p is not None and p.poll() is None:
                p.kill()


@contextlib.contextmanager
def _python_filter(path: Path, gz: bool, pats: list[bytes]):
    opener = gzip.open if gz else open
    state = {"err": None}

    def blocks():
        try:
            with opener(path, "rb") as f:
                buf = []
                for line in f:
                    if any(p in line for p in pats):
                        buf.append(line)
                        if len(buf) >= 4096:
                            yield b"".join(buf)
                            buf = []
                if buf:
                    yield b"".join(buf)
        except (OSError, EOFError) as exc:
            state["err"] = f"read failed: {exc!r}"

    yield blocks(), lambda: (state["err"] is None, state["err"] or "python filter read to EOF")


def read_release(path: Path) -> dict:
    """``version`` and ``releaseDate`` of the <UniRef50> root element."""
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(path, "rb") as f:
            head = f.read(4096)
    except (OSError, EOFError):
        return {}
    m = re.search(rb"<UniRef50\b([^>]*)>", head)
    if not m:
        return {}
    attrs = dict(re.findall(rb'(\w+)="([^"]*)"', m.group(1)))
    return {k: attrs[kb].decode() for k, kb in (("release_version", b"version"),
                                                 ("release_date", b"releaseDate"))
            if kb in attrs}


# ------------------------------------------------------------------- the scan


class _Entry:
    __slots__ = ("u50", "u90", "u100", "seen90", "seen100", "rep_blocks", "rep90", "rep100",
                 "in_rep")

    def __init__(self, u50: bytes):
        self.u50 = u50
        self.u90: list = []
        self.u100: list = []
        self.seen90: set = set()
        self.seen100: set = set()
        self.rep_blocks = 0
        self.rep90: list = []
        self.rep100: list = []
        self.in_rep = False


def _add(entry: _Entry, acc: bytes, ids: list, seen: set, rep_ids: list, what: str) -> None:
    if acc in seen:
        return
    if not 0 < len(acc) < 256:
        raise HierarchyError(f"implausible {what} accession {acc!r}")
    if entry.in_rep:
        rep_ids.append(acc)
    elif entry.rep_blocks == 0:
        raise HierarchyError(f"entry UniRef50_{entry.u50.decode()} lists a member id "
                             f"({acc.decode()}) before its <representativeMember> block")
    seen.add(acc)
    ids.append(acc)


def iter_entries(blocks) -> Iterator[_Entry]:
    """Parse filtered XML lines into entries; the caller checks completeness."""
    o90, o100, oe = len(U90_PREFIX), len(U100_PREFIX), len(ENTRY_PREFIX)
    cur: Optional[_Entry] = None
    saw_close = False
    tail = b""
    for block in blocks:
        if tail:
            block = tail + block
        lines = block.split(b"\n")
        tail = lines.pop()
        for line in lines:
            line = line.strip()
            if line.startswith(U90_GREP):
                if cur is None:
                    continue
                if not line.startswith(U90_PREFIX):
                    raise HierarchyError(f"unexpected UniRef90 id property: {line[:120]!r}")
                _add(cur, line[o90:line.index(QUOTE, o90)], cur.u90, cur.seen90, cur.rep90,
                     "UniRef90")
            elif line.startswith(U100_GREP):
                if cur is None:
                    continue
                if not line.startswith(U100_PREFIX):
                    raise HierarchyError(f"unexpected UniRef100 id property: {line[:120]!r}")
                _add(cur, line[o100:line.index(QUOTE, o100)], cur.u100, cur.seen100, cur.rep100,
                     "UniRef100")
            elif line.startswith(REP_OPEN):
                if cur is not None:
                    cur.rep_blocks += 1
                    cur.in_rep = True
            elif line.startswith(REP_CLOSE):
                if cur is not None:
                    cur.in_rep = False
            elif line.startswith(ENTRY_PREFIX):
                if cur is not None:
                    yield cur
                cur = _Entry(line[oe:line.index(QUOTE, oe)])
            elif ROOT_CLOSE in line:
                if cur is not None:
                    yield cur
                    cur = None
                saw_close = True
    if ROOT_CLOSE in tail and cur is not None:
        yield cur
        cur = None
        saw_close = True
    if not saw_close:
        # the stream was cut, so the open entry may be partial: never emit it
        raise HierarchyError("the XML ends without </UniRef50>; it is incomplete")


def _members(entry: _Entry, level: str, stats: dict) -> list:
    u50 = entry.u50.decode()
    if entry.rep_blocks != 1:
        raise HierarchyError(f"entry UniRef50_{u50} has {entry.rep_blocks} "
                             "<representativeMember> blocks")
    ids = entry.u90 if level == "uniref90" else entry.u100
    rep = entry.rep90 if level == "uniref90" else entry.rep100
    if not ids:
        raise HierarchyError(f"entry UniRef50_{u50} carries no {level} id")
    if len(rep) != 1:
        raise HierarchyError(f"entry UniRef50_{u50}: its <representativeMember> holds "
                             f"{len(rep)} {level} ids, expected 1")
    if level == "uniref100":
        if len(entry.rep90) != 1:
            raise HierarchyError(f"entry UniRef50_{u50}: <representativeMember> holds "
                                 f"{len(entry.rep90)} UniRef90 ids, expected 1")
        paired = entry.rep90[0]
        if paired in entry.seen100:
            if ids[0] != paired:
                stats["paired_rep_differs_from_literal"] += 1
                ids = [paired] + [a for a in ids if a != paired]
        else:
            stats["paired_rep_missing"] += 1
    return ids


def xml_to_members_tsv(xml_path, out_tsv, level: str = "uniref90") -> dict:
    """Write the canonical members table for ``level`` from uniref50.xml(.gz)."""
    if level not in UNIREF_LEVELS:
        raise ValueError(f"level must be one of {UNIREF_LEVELS}, got {level!r}")
    xml_path, out_tsv = Path(xml_path), Path(out_tsv)
    if not xml_path.exists():
        raise FileNotFoundError(f"UniRef50 XML not found: {xml_path}")
    t0 = time.time()
    seq_prefix = b"UniRef90_" if level == "uniref90" else b"UniRef100_"
    stats = {"clusters": 0, "members": 0, "singletons": 0,
             "paired_rep_differs_from_literal": 0, "paired_rep_missing": 0}
    logger.info(f"cluster table: scanning {xml_path} for UniRef50 -> {level} membership")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write_path(out_tsv) as tmp, open(tmp, "wb", buffering=1 << 24) as out:
        with _filtered_lines(xml_path, level) as (blocks, finish):
            for entry in iter_entries(blocks):
                ids = _members(entry, level, stats)
                c = b"UniRef50_" + entry.u50 + b"\t" + seq_prefix
                out.write(b"".join(c + a + b"\n" for a in ids))
                stats["clusters"] += 1
                stats["members"] += len(ids)
                stats["singletons"] += len(ids) == 1
            clean, detail = finish()
        if not clean:
            raise HierarchyError(f"{xml_path} is incomplete ({detail})")
    if stats["clusters"] == 0:
        raise HierarchyError(f"{xml_path} holds no UniRef50 entries")
    if stats["paired_rep_missing"]:
        logger.warning(f"cluster table: {stats['paired_rep_missing']:,} entries have no "
                       "UniRef100 id paired with their UniRef90 representative; their "
                       "literal representativeMember is used")
    stats.update(level=level, source="uniref50_xml", input=str(xml_path),
                 **read_release(xml_path), seconds=round(time.time() - t0, 1))
    logger.info(f"cluster table: {stats['clusters']:,} UniRef50 clusters, "
                f"{stats['members']:,} {level} members ({stats['seconds']}s)")
    return stats
