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

With ``sub_out`` (UniRef100 level only) the same scan also writes ``submembers.tsv``,
the UniRef90 sub-cluster of every UniRef100 member in the same canonical format:
``UniRef90_X<TAB>UniRef100_Y``, grouped by UniRef90 id (first appearance inside the
entry, members in XML order). Every member record lists its UniRef100 id and, on the
next line, its UniRef90 id; the scan fails if the two do not alternate, if one
UniRef100 id sits under two UniRef90 ids, or if one UniRef90 id appears in two entries.

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
from array import array
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
RECORD_OPEN = b"<dbReference "   # starts every member / representative record
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


def _patterns(level: str, records: bool = False) -> list:
    # a UniRef100 table also needs the UniRef90 rep id, to pick the paired rep;
    # pairing scans (``records``) also need the record boundary
    props = [U90_GREP] if level == "uniref90" else [U90_GREP, U100_GREP]
    if records:
        props.append(RECORD_OPEN)
    return [ENTRY_PREFIX, *props, REP_OPEN, REP_CLOSE, ROOT_CLOSE]


@contextlib.contextmanager
def _filtered_lines(path: Path, level: str, records: bool = False) -> Iterator[tuple]:
    """Yield (blocks, finish): blocks of grep-filtered XML lines; finish() -> (ok, detail)."""
    pats = _patterns(level, records)
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
                 "in_rep", "sub", "pending")

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
        self.sub: Optional[dict] = None      # UniRef100 -> UniRef90 (pairing scans only)
        self.pending: Optional[bytes] = None  # UniRef100 id waiting for its UniRef90 line


class _PairIssues:
    """Pairing problems, counted over the whole scan; raised once at the end."""

    def __init__(self):
        self.count = {"order": 0, "conflict": 0}
        self.examples = {"order": [], "conflict": []}

    def add(self, kind: str, text: str) -> None:
        self.count[kind] += 1
        if len(self.examples[kind]) < 5:
            self.examples[kind].append(text)

    def check(self) -> None:
        msgs = []
        if self.count["order"]:
            msgs.append(f"{self.count['order']:,} member records where the UniRef100 ID and "
                        "UniRef90 ID lines do not alternate (first: "
                        + "; ".join(self.examples["order"]) + ")")
        if self.count["conflict"]:
            msgs.append(f"{self.count['conflict']:,} UniRef100 ids listed under two "
                        "UniRef90 ids (first: " + "; ".join(self.examples["conflict"]) + ")")
        if msgs:
            raise HierarchyError("cannot pair UniRef100 members with UniRef90 sub-clusters: "
                                 + "; ".join(msgs))


def _end_pairs(cur: _Entry, issues: _PairIssues) -> None:
    if cur.pending is not None:
        issues.add("order", f"entry UniRef50_{cur.u50.decode()}: UniRef100_"
                            f"{cur.pending.decode()} has no UniRef90 line")
        cur.pending = None


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


def iter_entries(blocks, pairs: Optional[_PairIssues] = None) -> Iterator[_Entry]:
    """Parse filtered XML lines into entries; the caller checks completeness.

    With ``pairs``, every entry also gets ``sub`` (UniRef100 -> UniRef90 of each member
    record) and pairing problems are counted in ``pairs``."""
    o90, o100, oe = len(U90_PREFIX), len(U100_PREFIX), len(ENTRY_PREFIX)
    cur: Optional[_Entry] = None

    def new_entry(line):
        e = _Entry(line[oe:line.index(QUOTE, oe)])
        if pairs is not None:
            e.sub = {}
        return e

    def done(e):
        if pairs is not None:
            _end_pairs(e, pairs)
        return e

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
                acc = line[o90:line.index(QUOTE, o90)]
                _add(cur, acc, cur.u90, cur.seen90, cur.rep90, "UniRef90")
                if pairs is not None:          # pair it with the UniRef100 line before
                    p = cur.pending
                    if p is None:
                        pairs.add("order", f"entry UniRef50_{cur.u50.decode()}: UniRef90_"
                                           f"{acc.decode()} follows no UniRef100 line")
                    else:
                        old = cur.sub.setdefault(p, acc)
                        if old != acc:
                            pairs.add("conflict", f"entry UniRef50_{cur.u50.decode()}: "
                                      f"UniRef100_{p.decode()} under UniRef90_{old.decode()} "
                                      f"and UniRef90_{acc.decode()}")
                        cur.pending = None
            elif line.startswith(U100_GREP):
                if cur is None:
                    continue
                if not line.startswith(U100_PREFIX):
                    raise HierarchyError(f"unexpected UniRef100 id property: {line[:120]!r}")
                acc = line[o100:line.index(QUOTE, o100)]
                _add(cur, acc, cur.u100, cur.seen100, cur.rep100, "UniRef100")
                if pairs is not None:
                    if cur.pending is not None:
                        pairs.add("order", f"entry UniRef50_{cur.u50.decode()}: UniRef100_"
                                           f"{cur.pending.decode()} has no UniRef90 line")
                    cur.pending = acc
            elif line.startswith(RECORD_OPEN):     # only in pairing scans: close the record
                if cur is not None and pairs is not None:
                    _end_pairs(cur, pairs)
            elif line.startswith(REP_OPEN):
                if cur is not None:
                    cur.rep_blocks += 1
                    cur.in_rep = True
            elif line.startswith(REP_CLOSE):
                if cur is not None:
                    cur.in_rep = False
            elif line.startswith(ENTRY_PREFIX):
                if cur is not None:
                    yield done(cur)
                cur = new_entry(line)
            elif ROOT_CLOSE in line:
                if cur is not None:
                    yield done(cur)
                    cur = None
                saw_close = True
    if ROOT_CLOSE in tail and cur is not None:
        yield done(cur)
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


def _sub_lines(entry: _Entry, sub_hashes: array) -> bytes:
    """submembers.tsv lines of one entry: grouped by UniRef90 id, first appearance first."""
    sub = entry.sub
    if len(entry.u90) == len(entry.u100):    # every sub-cluster a singleton (most entries)
        groups = {sub[a]: (a,) for a in entry.u100}
    else:
        groups = {}
        for a100 in entry.u100:
            groups.setdefault(sub[a100], []).append(a100)
    sub_hashes.extend(map(_id_hash, groups))    # checked for repeats across entries at the end
    out = []
    for a90, a100s in groups.items():
        c = b"UniRef90_" + a90 + b"\tUniRef100_"
        out.extend(c + a + b"\n" for a in a100s)
    return b"".join(out)


_id_hash = hash   # 64-bit, salted per process; a test swaps it to force collisions


def _check_subclusters_unique(sub_hashes: array, members_tsv: Path, sub_tsv: Path) -> None:
    """Fail if one UniRef90 id heads sub-clusters in two UniRef50 entries.

    The hashes only find suspects (a false collision among ~1e8 ids is rare but real);
    suspects are confirmed on the exact ids by reading both tables back, whose rows are
    in the same entry order (each entry writes one row per UniRef100 member to both)."""
    h = np.sort(np.frombuffer(sub_hashes, dtype=np.int64))
    twice = h[1:][h[1:] == h[:-1]]
    if not len(twice):
        return
    suspects = set(np.unique(twice).tolist())
    clusters: dict = {}
    prev, hit = None, False
    with open(members_tsv, "rb") as fm, open(sub_tsv, "rb") as fs:
        for lm, ls in zip(fm, fs):
            a90 = ls[9:ls.index(b"\t")]                      # after "UniRef90_"
            if a90 != prev:
                prev, hit = a90, _id_hash(a90) in suspects
            if hit:
                clusters.setdefault(a90, set()).add(lm[:lm.index(b"\t")].decode())
    bad = sorted((a.decode(), sorted(c)) for a, c in clusters.items() if len(c) > 1)
    if not bad:
        logger.info(f"cluster table: {len(suspects):,} UniRef90 id hash collisions, "
                    "none a real repeat")
        return
    ex = "; ".join(f"UniRef90_{a} in {' and '.join(c)}" for a, c in bad[:5])
    raise HierarchyError(f"{len(bad):,} UniRef90 ids appear in two UniRef50 entries (first: "
                         f"{ex}); a sub-cluster must lie inside one cluster")


def xml_to_members_tsv(xml_path, out_tsv, level: str = "uniref90", sub_out=None) -> dict:
    """Write the canonical members table for ``level`` from uniref50.xml(.gz).

    With ``sub_out`` (``level="uniref100"`` only), also write the UniRef90 sub-cluster
    table there (``subcluster_id<TAB>sequence_id``, same canonical format)."""
    if level not in UNIREF_LEVELS:
        raise ValueError(f"level must be one of {UNIREF_LEVELS}, got {level!r}")
    if sub_out is not None and level != "uniref100":
        raise ValueError("a sub-cluster table (sub_out) needs level uniref100: UniRef90 "
                         "members have no level below them")
    xml_path, out_tsv = Path(xml_path), Path(out_tsv)
    if not xml_path.exists():
        raise FileNotFoundError(f"UniRef50 XML not found: {xml_path}")
    t0 = time.time()
    seq_prefix = b"UniRef90_" if level == "uniref90" else b"UniRef100_"
    stats = {"clusters": 0, "members": 0, "singletons": 0,
             "paired_rep_differs_from_literal": 0, "paired_rep_missing": 0}
    logger.info(f"cluster table: scanning {xml_path} for UniRef50 -> {level} membership")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    issues = _PairIssues() if sub_out is not None else None
    sub_hashes = array("q")
    with contextlib.ExitStack() as stack:
        tmp = stack.enter_context(atomic_write_path(out_tsv))
        out = stack.enter_context(open(tmp, "wb", buffering=1 << 24))
        sub = None
        if sub_out is not None:
            sub_out = Path(sub_out)
            sub_out.parent.mkdir(parents=True, exist_ok=True)
            sub_tmp = stack.enter_context(atomic_write_path(sub_out))
            sub = stack.enter_context(open(sub_tmp, "wb", buffering=1 << 24))
        with _filtered_lines(xml_path, level, records=sub is not None) as (blocks, finish):
            for entry in iter_entries(blocks, issues):
                ids = _members(entry, level, stats)
                c = b"UniRef50_" + entry.u50 + b"\t" + seq_prefix
                out.write(b"".join(c + a + b"\n" for a in ids))
                if sub is not None and not (issues.count["order"] or issues.count["conflict"]):
                    sub.write(_sub_lines(entry, sub_hashes))
                stats["clusters"] += 1
                stats["members"] += len(ids)
                stats["singletons"] += len(ids) == 1
            clean, detail = finish()
        if not clean:
            raise HierarchyError(f"{xml_path} is incomplete ({detail})")
        if issues is not None:
            issues.check()
            out.flush()
            sub.flush()
            _check_subclusters_unique(sub_hashes, Path(tmp), Path(sub_tmp))
    if stats["clusters"] == 0:
        raise HierarchyError(f"{xml_path} holds no UniRef50 entries")
    if stats["paired_rep_missing"]:
        logger.warning(f"cluster table: {stats['paired_rep_missing']:,} entries have no "
                       "UniRef100 id paired with their UniRef90 representative; their "
                       "literal representativeMember is used")
    if sub_out is not None:
        stats["subclusters"] = len(sub_hashes)
    stats.update(level=level, source="uniref50_xml", input=str(xml_path),
                 **read_release(xml_path), seconds=round(time.time() - t0, 1))
    logger.info(f"cluster table: {stats['clusters']:,} UniRef50 clusters, "
                f"{stats['members']:,} {level} members ({stats['seconds']}s)")
    return stats
