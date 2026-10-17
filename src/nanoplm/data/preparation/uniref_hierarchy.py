"""UniRef50 -> UniRef90 / UniRef100 membership from uniref50.xml(.gz).

Streams the XML through ``pigz -dc | grep -F`` (``gzip -dc`` without pigz; a
pure-Python filter without grep) on the few line shapes it needs, dedups the
member ids of each entry, and writes into ``out_dir``:

  {u90,u100}_keys.npy     uint64[P] sorted, strictly increasing: one key per member id
  {u90,u100}_cluster.npy  int32[P]  UniRef50 cluster index (XML entry order), aligned
  {u90,u100}_is_rep.npy   bool[P]   the id inside the entry's <representativeMember>
  cluster_ids.txt         line i: bare UniRef50 accession of cluster i
  cluster_raw_sizes.npy   int32[C]  distinct member ids per cluster, before any filter
  hierarchy_stats.json    counts, release, hash test vectors, verify block

121M accession strings do not fit in a dict, so members are keyed by a 64-bit hash
of the accession with its ``UniRef90_`` / ``UniRef100_`` prefix removed:

    hash_one(b"A0ACN6ENM0")   # for value="UniRef90_A0ACN6ENM0"

FNV-1a-64 over the bytes, then the murmur3 fmix64 finalizer; the salt is XORed into
the FNV offset basis. Collision-freedom is checked at merge time, and the join in
``cluster_index`` refuses any pool accession that is not in the key set, so a
collision cannot silently move a sequence into another cluster.

The representative is the single member id inside ``<representativeMember>``, which
precedes every ``<member>`` in UniProt's files. An entry with no or two
representative blocks, or a member id before the representative block, fails loudly.

Ported from the data-sampler experiment's extract_hierarchy.py /
extract_hierarchy100.py (scan checkpoints and resume were dropped).
"""
from __future__ import annotations

import contextlib
import gzip
import json
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

TOOL_VERSION = 1

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
FMIX_C1 = 0xFF51AFD7ED558CCD
FMIX_C2 = 0xC4CEB9FE1A85EC53
MASK64 = (1 << 64) - 1
HASH_NAME = "fnv1a64+fmix64"
HASH_TEST_VECTORS = (b"P12345", b"A0ACN6ENM0", b"UPI00447321E6")

# pool level -> (file prefix, accession prefix in the XML and in FASTA headers)
LEVELS = {
    "uniref90": ("u90", b"UniRef90_"),
    "uniref100": ("u100", b"UniRef100_"),
}

ENTRY_PREFIX = b'<entry id="UniRef50_'
REP_OPEN = b"<representativeMember>"
REP_CLOSE = b"</representativeMember>"
ROOT_CLOSE = b"</UniRef50>"
QUOTE = 0x22
READ_BLOCK = 1 << 22
STATS_FILE = "hierarchy_stats.json"


class HierarchyError(RuntimeError):
    """The XML or the hierarchy arrays break an invariant the cluster index relies on."""


def _property_grep(level: str) -> bytes:
    n = LEVELS[level][1][len(b"UniRef"):-1].decode()
    return f'<property type="UniRef{n} ID" value="'.encode()


def file_prefix(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
    return LEVELS[level][0]


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


def _patterns(level: str) -> list[bytes]:
    return [ENTRY_PREFIX, _property_grep(level), REP_OPEN, REP_CLOSE, ROOT_CLOSE]


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


def _scan(blocks, level: str):
    prop = _property_grep(level) + LEVELS[level][1]
    prop_off = len(prop)
    entry_off = len(ENTRY_PREFIX)
    blob = bytearray()
    lens = array("H")
    sizes = array("i")
    cluster_accs = bytearray()
    seen: set[bytes] = set()
    recording = False
    cur = b""
    n_members = rep_blocks = rep_ids = 0
    in_rep = saw_close = False
    tail = b""

    def end_entry():
        if n_members == 0:
            raise HierarchyError(f"entry UniRef50_{cur.decode()} carries no {level} id")
        if rep_blocks != 1:
            raise HierarchyError(f"entry UniRef50_{cur.decode()} has {rep_blocks} "
                                 "<representativeMember> blocks")
        if rep_ids != 1:
            raise HierarchyError(f"entry UniRef50_{cur.decode()}: its <representativeMember> "
                                 f"holds {rep_ids} {level} ids, expected 1")
        sizes.append(n_members)
        cluster_accs.extend(cur + b"\n")

    for block in blocks:
        if tail:
            block = tail + block
        lines = block.split(b"\n")
        tail = lines.pop()
        for line in lines:
            line = line.strip()
            if line.startswith(prop[:prop_off - len(LEVELS[level][1])]):
                if not recording:
                    continue
                if not line.startswith(prop):
                    raise HierarchyError(f"unexpected {level} id property: {line[:120]!r}")
                acc = line[prop_off:line.index(QUOTE, prop_off)]
                if acc in seen:
                    continue
                if not 0 < len(acc) < 256:
                    raise HierarchyError(f"implausible accession {acc!r}")
                if in_rep:
                    rep_ids += 1
                elif rep_blocks == 0:
                    raise HierarchyError(
                        f"entry UniRef50_{cur.decode()} lists a member id ({acc.decode()}) "
                        "before its <representativeMember> block")
                seen.add(acc)
                blob.extend(acc)
                lens.append(len(acc))
                n_members += 1
            elif line.startswith(REP_OPEN):
                if recording:
                    rep_blocks += 1
                    in_rep = True
            elif line.startswith(REP_CLOSE):
                in_rep = False
            elif line.startswith(ENTRY_PREFIX):
                if recording:
                    end_entry()
                recording = True
                cur = line[entry_off:line.index(QUOTE, entry_off)]
                seen.clear()
                n_members = rep_blocks = rep_ids = 0
                in_rep = False
            elif ROOT_CLOSE in line:
                saw_close = True
    if ROOT_CLOSE in tail:
        saw_close = True
    return blob, lens, sizes, cluster_accs, recording, end_entry, saw_close


def _save_npy(path: Path, arr: np.ndarray) -> None:
    with atomic_write_path(path) as tmp:
        with open(tmp, "wb") as f:
            np.save(f, arr)


def extract_hierarchy(xml_path, out_dir, level: str = "uniref90", salt: int = 0) -> dict:
    """Scan ``xml_path`` and write the hierarchy arrays for ``level`` into ``out_dir``."""
    xml_path, out_dir = Path(xml_path), Path(out_dir)
    pre = file_prefix(level)
    if not xml_path.exists():
        raise FileNotFoundError(f"UniRef50 XML not found: {xml_path}")
    t0 = time.time()
    logger.info(f"hierarchy: scanning {xml_path} for UniRef50 -> {level} membership")
    with _filtered_lines(xml_path, level) as (blocks, finish):
        blob, lens, sizes, cluster_accs, open_entry, end_entry, saw_close = _scan(blocks, level)
        clean, detail = finish()
    if not (clean and saw_close):
        raise HierarchyError(f"{xml_path} is incomplete ({detail}; root close tag "
                             f"{'seen' if saw_close else 'missing'})")
    if open_entry:
        end_entry()
    sizes = np.frombuffer(bytes(sizes), dtype=np.int32).copy()
    lens = np.frombuffer(bytes(lens), dtype=np.uint16).astype(np.int64)
    blob = np.frombuffer(bytes(blob), dtype=np.uint8)
    n_pairs, n_clusters = len(lens), len(sizes)
    if n_clusters == 0:
        raise HierarchyError(f"{xml_path} holds no UniRef50 entries")
    keys = hash_accessions(blob, lens, salt=salt)
    cluster = np.repeat(np.arange(n_clusters, dtype=np.int32), sizes)
    is_rep = np.zeros(n_pairs, dtype=bool)
    is_rep[np.cumsum(sizes, dtype=np.int64) - sizes] = True
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    dup = np.flatnonzero(keys_sorted[1:] == keys_sorted[:-1])
    if dup.size:
        offs = offsets_from_lengths(lens)
        ex = []
        for d in dup[:5]:
            a, b = int(order[d]), int(order[d + 1])
            sa = blob[offs[a]:offs[a] + lens[a]].tobytes()
            sb = blob[offs[b]:offs[b] + lens[b]].tobytes()
            ex.append(f"{sa.decode()} (cluster {cluster[a]}) / {sb.decode()} (cluster "
                      f"{cluster[b]}): {'same id in two entries' if sa == sb else 'collision'}")
        raise HierarchyError(f"{dup.size} duplicate keys among {n_pairs} {level} ids: "
                             + "; ".join(ex) + ". A collision needs another salt.")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_npy(out_dir / f"{pre}_keys.npy", keys_sorted)
    _save_npy(out_dir / f"{pre}_cluster.npy", cluster[order])
    _save_npy(out_dir / f"{pre}_is_rep.npy", is_rep[order])
    _save_npy(out_dir / "cluster_raw_sizes.npy", sizes)
    with atomic_write_path(out_dir / "cluster_ids.txt") as tmp:
        tmp.write_bytes(bytes(cluster_accs))
    stats = {
        "tool": "nanoplm.data.preparation.uniref_hierarchy",
        "tool_version": TOOL_VERSION,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "level": level,
        "input": {"path": str(xml_path), "size_bytes": xml_path.stat().st_size,
                  **read_release(xml_path)},
        "counts": {"clusters": n_clusters, "members": n_pairs,
                   "singletons": int((sizes == 1).sum()),
                   "max_cluster_size": int(sizes.max())},
        "hash": {"function": HASH_NAME, "salt": salt,
                 "keyed_string": f"the {level} accession with its "
                                 f"'{LEVELS[level][1].decode()}' prefix removed",
                 "test_vectors": {a.decode(): f"0x{hash_one(a, salt):016x}"
                                  for a in HASH_TEST_VECTORS}},
        "seconds": round(time.time() - t0, 1),
    }
    stats["verify"] = verify_hierarchy(out_dir, level, stats=stats)
    _write_stats(out_dir, stats)
    if not stats["verify"]["ok"]:
        raise HierarchyError(f"{out_dir}: hierarchy verify failed: "
                             f"{stats['verify']['failures']}")
    logger.info(f"hierarchy: {n_clusters:,} UniRef50 clusters, {n_pairs:,} {level} members "
                f"({stats['seconds']}s)")
    return stats


def _write_stats(out_dir: Path, stats: dict) -> None:
    with atomic_write_path(out_dir / STATS_FILE) as tmp:
        tmp.write_text(json.dumps(stats, indent=1) + "\n")


def read_stats(out_dir) -> dict:
    return json.loads((Path(out_dir) / STATS_FILE).read_text())


def load_hierarchy(out_dir, level: str, mmap: bool = True):
    """(keys, cluster, is_rep, raw_sizes) of a verified hierarchy."""
    out_dir = Path(out_dir)
    stats = read_stats(out_dir)
    if stats.get("level", "uniref90") != level:
        raise HierarchyError(f"{out_dir} holds a {stats.get('level')} hierarchy, not {level}")
    if stats.get("verify", {}).get("ok") is not True:
        raise HierarchyError(f"{out_dir}/{STATS_FILE}: verify.ok is not true")
    pre = file_prefix(level)
    mode = "r" if mmap else None
    return (np.load(out_dir / f"{pre}_keys.npy", mmap_mode=mode),
            np.load(out_dir / f"{pre}_cluster.npy", mmap_mode=mode),
            np.load(out_dir / f"{pre}_is_rep.npy", mmap_mode=mode),
            np.load(out_dir / "cluster_raw_sizes.npy"))


def hash_self_check(stats: dict) -> int:
    """The salt the hierarchy was keyed with, after checking this module's hash on it."""
    h = stats["hash"]
    salt = int(h.get("salt", 0))
    for acc, want in h["test_vectors"].items():
        if f"0x{hash_one(acc.encode(), salt):016x}" != want:
            raise HierarchyError(f"hash self-check failed on {acc!r}: the hierarchy was keyed "
                                 "with another hash")
    return salt


def verify_hierarchy(out_dir, level: str, stats: Optional[dict] = None) -> dict:
    """Re-check the written arrays. Returns {"ok": bool, "checks": {...}, "failures": [...]}."""
    out_dir = Path(out_dir)
    pre = file_prefix(level)
    stats = stats if stats is not None else read_stats(out_dir)
    keys = np.load(out_dir / f"{pre}_keys.npy", mmap_mode="r")
    cluster = np.load(out_dir / f"{pre}_cluster.npy", mmap_mode="r")
    is_rep = np.load(out_dir / f"{pre}_is_rep.npy", mmap_mode="r")
    sizes = np.load(out_dir / "cluster_raw_sizes.npy")
    with open(out_dir / "cluster_ids.txt", "rb") as f:
        n_ids = sum(1 for _ in f)
    checks: dict[str, bool] = {}
    n, c = len(keys), len(sizes)
    checks["aligned_arrays"] = len(cluster) == n == len(is_rep)
    checks["keys_sorted_strictly_increasing"] = bool(n < 2 or np.all(keys[1:] > keys[:-1]))
    checks["cluster_ids_match_sizes"] = n_ids == c
    cl = np.asarray(cluster)
    in_range = bool(n == 0 or (cl.min() >= 0 and cl.max() < c))
    checks["cluster_index_in_range"] = in_range
    counts = np.bincount(cl, minlength=c) if in_range else np.zeros(c)
    checks["bincount_equals_raw_sizes"] = bool(np.array_equal(counts, sizes))
    checks["every_cluster_nonempty"] = bool(c > 0 and (sizes >= 1).all())
    reps = np.bincount(cl[np.asarray(is_rep)], minlength=c) if in_range else np.zeros(c)
    checks["one_representative_per_cluster"] = bool(np.all(reps == 1))
    try:
        hash_self_check(stats)
        checks["hash_self_check"] = True
    except HierarchyError:
        checks["hash_self_check"] = False
    failures = [k for k, v in checks.items() if not v]
    return {"ok": not failures, "checks": checks, "failures": failures}
