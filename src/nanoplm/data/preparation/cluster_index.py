"""Join a tokenized member pool against its cluster table.

Input: the pool's filtered FASTA (its record order *is* the pool row order, since the
shard writer keeps it), the pool shards written from it, and the canonical cluster
table ``members.tsv`` (``cluster_table``; cluster index = order of first appearance,
first row of a cluster = its representative). Every pool sequence must be in the table.
Output in ``index_dir``, under the file names the experiment's index used (so
ClusterSampledDataset opens either):

  pool_cluster.npy            int32[N]    pool row -> cluster index
  cluster_raw_sizes.npy       int32[C]    members per cluster in the table
  pool_is_rep.npy             bool[N]     the row is its cluster's representative
  pool_lengths.npy            int32[N]    tokens of every pool row (concatenated .idx.npy)
  cluster_sizes_filtered.npy  int32[C]    n_c: surviving members per cluster
  universe_mask.npy           bool[C]     cluster is in the universe (see below)
  cluster_rep_row.npy         int32[C]    pool row of the surviving rep, -1 if none
  universe_clusters.npy       int32[Cu]   universe-local -> global cluster index
  global_to_universe.npy      int32[C]    global -> universe-local, -1 outside
  cluster_offsets.npy         int64[Cu+1] CSR over the universe
  cluster_members.npy         int32[M]    pool rows grouped by universe cluster, ascending
  val_clusters.npy            int32       held-out clusters (global, sorted)
  val_rep_rows.npy            int32       their representatives, aligned with val_clusters
  train_clusters.npy          int32       universe minus val: what training visits
  shard_row_offsets.npy       int64       cumulative rows per shard
  pool_manifest.json          counts, holdout, cap, and the verify block

The universe rule: a cluster is in it iff its representative record survived the
length filter, i.e. is a pool row. Training draws members uniformly over all n_c
surviving members, representative included.

Validation is held out by whole cluster: ``round(Cu * val_ratio)`` universe clusters,
drawn with a fixed seed (never a prefix: FASTA order correlates with cluster index).
``max_clusters`` (the data config's seqs_num) first caps the universe to a seeded random
subset of that many clusters; the holdout is then drawn from the capped universe.

Ported from the data-sampler experiment's build_index.py / build_index100.py and
verify_index.py; scan checkpoints, anchors, alpha arms and the gate subset were dropped.
"""
from __future__ import annotations

import json
import time
from array import array
from pathlib import Path
from typing import Optional

import numpy as np

from nanoplm.data.datasets.cluster_sampled import INDEX_MANIFEST
from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.data.preparation import uniref_hierarchy as uh
from nanoplm.data.preparation.cluster_table import (  # noqa: F401  (re-exported)
    detect_uniref_level, read_members_table)
from nanoplm.utils import logger

TOOL_VERSION = 1
# The experiment's holdout seed: rebuilding on its release with its val_ratio (0.0044)
# reproduces its val_clusters.npy exactly.
HOLDOUT_SEED = 20260918
CAP_SEED_OFFSET = 2
# ShardedDataset finds shards with sorted(glob("*.bin")) and the writer names them
# shard_%04d, so from 10,000 shards on the sorted order stops being the row order.
SHARD_SORT_CLIFF = 10_000
DECODE_SPOT_ROWS = 300
_WS = b" \t\r\n\v\f"


class ClusterIndexError(RuntimeError):
    """The pool, its FASTA and the hierarchy do not fit together."""


def _save(path: Path, arr: np.ndarray) -> None:
    with atomic_write_path(path) as tmp:
        with open(tmp, "wb") as f:
            np.save(f, arr)


def _write_json(path: Path, obj: dict) -> None:
    with atomic_write_path(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=1) + "\n")


# ---------------------------------------------------------------- pool shards


def pool_shard_paths(pool_dir: Path) -> list[Path]:
    bins = sorted(Path(pool_dir).glob("*.bin"))
    if not bins:
        raise ClusterIndexError(f"no shards in {pool_dir}")
    if len(bins) >= SHARD_SORT_CLIFF:
        raise ClusterIndexError(
            f"{pool_dir} holds {len(bins)} shards; from {SHARD_SORT_CLIFF} on the sorted "
            "shard order is no longer the row order. Raise samples_per_shard.")
    return bins


def read_pool_lengths(pool_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(lengths per pool row, shard_row_offsets), from the shard .idx.npy files."""
    parts, rows = [], [0]
    for b in pool_shard_paths(pool_dir):
        idx = np.load(b.with_name(b.stem + ".idx.npy"))
        if int(idx.sum(dtype=np.int64)) != b.stat().st_size:
            raise ClusterIndexError(f"{b.name}: .bin size does not match its index")
        parts.append(idx.astype(np.int32))
        rows.append(rows[-1] + len(idx))
    return np.concatenate(parts), np.asarray(rows, dtype=np.int64)


# ---------------------------------------------------------------- FASTA scan


def scan_pool_fasta(fasta: Path, want_rows: Optional[set] = None):
    """Id keys (hash of the first header word) and residue counts of every record.

    Returns (keys uint64[N], residues int64[N], {row: sequence} for rows in want_rows).
    """
    blob = bytearray()
    acc_len = array("H")
    residues = array("q")
    seqs: dict = {}
    row = -1
    count = 0
    parts = None
    with open(fasta, "rb") as f:
        for line in f:
            if line[:1] == b">":
                if row >= 0:
                    residues.append(count)
                    if parts is not None:
                        seqs[row] = b"".join(parts)
                row += 1
                parts_ = line[1:].split(None, 1)
                acc = parts_[0] if parts_ else b""
                if not 0 < len(acc) < 65536:
                    raise ClusterIndexError(f"{fasta}: record {row} has no id")
                blob.extend(acc)
                acc_len.append(len(acc))
                count = 0
                parts = [] if want_rows is not None and row in want_rows else None
            else:
                s = line.translate(None, _WS)
                count += len(s)
                if parts is not None:
                    parts.append(s)
    if row >= 0:
        residues.append(count)
        if parts is not None:
            seqs[row] = b"".join(parts)
    lens = np.frombuffer(bytes(acc_len), dtype=np.uint16).astype(np.int64)
    keys = uh.hash_accessions(np.frombuffer(bytes(blob), dtype=np.uint8), lens)
    return keys, np.frombuffer(bytes(residues), dtype=np.int64), seqs


def token_length_specials(tokens: np.ndarray, residues: np.ndarray, max_len: int) -> int:
    """The constant s with tokens == min(residues + s, max_len) for every row.

    Proves the FASTA and the shards describe the same rows in the same order without
    knowing the tokenizer's BOS / register-token settings.
    """
    if len(tokens) != len(residues):
        raise ClusterIndexError(f"the shards hold {len(tokens)} rows, the FASTA "
                                f"{len(residues)}")
    short = tokens < max_len
    diff = tokens[short].astype(np.int64) - residues[short]
    if diff.size == 0:
        raise ClusterIndexError("every pool row is truncated; cannot check row alignment")
    s = int(np.bincount(diff - diff.min()).argmax() + diff.min())
    expect = np.minimum(residues + s, max_len)
    bad = np.flatnonzero(expect != tokens)
    if bad.size:
        raise ClusterIndexError(
            f"{bad.size} pool rows break tokens == min(residues + {s}, {max_len}) (first "
            f"row {int(bad[0])}): the FASTA and the shards are not the same rows in order")
    return s


# ------------------------------------------------------------------ the draws


def draw_val_clusters(cu: int, val_ratio: float, seed: int) -> np.ndarray:
    """Universe-local indices of the holdout, sorted (the experiment's exact draw)."""
    n_val = int(round(cu * val_ratio))
    rng = np.random.default_rng(seed)
    pick = rng.choice(cu, size=n_val, replace=False, shuffle=False)
    return np.sort(pick.astype(np.int64))


def draw_cap(n: int, k: int, seed: int) -> np.ndarray:
    """Sorted positions of a uniform random k-subset of range(n)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False, shuffle=False).astype(np.int64))


# ------------------------------------------------------------------- the build


def build_cluster_index(
    fasta: Path,
    pool_dir: Path,
    members_tsv: Path,
    index_dir: Path,
    max_seq_len: int,
    val_ratio: float,
    max_clusters: int = -1,
    holdout_seed: int = HOLDOUT_SEED,
    tokenizer=None,
    level: Optional[str] = None,
    clusters_info: Optional[dict] = None,
) -> dict:
    """Write the cluster index for the pool in ``pool_dir`` (built from ``fasta``).

    ``tokenizer`` (optional) re-tokenizes a few hundred FASTA records with the shard
    writer and compares them to their pool rows byte for byte; the token-length rule
    alone cannot see an equal-length reorder.
    """
    t0 = time.time()
    fasta, pool_dir = Path(fasta), Path(pool_dir)
    members_tsv, index_dir = Path(members_tsv), Path(index_dir)
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    logger.info(f"cluster index: reading {members_tsv}")
    table = read_members_table(members_tsv)
    u_keys, u_cluster, u_is_rep, raw_sizes = table.keys, table.cluster, table.is_rep, table.sizes

    lengths, sro = read_pool_lengths(pool_dir)
    n = len(lengths)
    if n > np.iinfo(np.int32).max:
        raise ClusterIndexError(f"{n} pool rows exceed the int32 row ids of the index")
    spot = set(np.random.default_rng(holdout_seed).choice(
        n, size=min(DECODE_SPOT_ROWS, n), replace=False).tolist()) if tokenizer else None
    logger.info(f"cluster index: scanning {fasta} ({n:,} pool rows)")
    keys, residues, spot_seqs = scan_pool_fasta(fasta, spot)
    specials = token_length_specials(lengths, residues, max_seq_len)
    if tokenizer is not None:
        _decode_check(pool_dir, sro, lengths, spot_seqs, tokenizer, max_seq_len, index_dir)

    # join: sorted probe keys turn the binary search into a near-sequential sweep
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    dup = np.flatnonzero(sk[1:] == sk[:-1])
    if dup.size:
        raise ClusterIndexError(f"{dup.size} accessions appear twice in {fasta}, e.g. rows "
                                f"{sorted(order[[dup[0], dup[0] + 1]].tolist())}")
    pos = np.searchsorted(u_keys, sk)
    np.clip(pos, 0, len(u_keys) - 1, out=pos)
    hit = np.asarray(u_keys[pos]) == sk
    if not hit.all():
        bad = order[np.flatnonzero(~hit)[:5]].tolist()
        raise ClusterIndexError(
            f"{int((~hit).sum())} pool sequences are not in {members_tsv} (rows {bad}); the "
            "member FASTA and the cluster source are probably from different releases")
    pool_cluster = np.empty(n, dtype=np.int32)
    pool_is_rep = np.empty(n, dtype=bool)
    pool_cluster[order] = np.asarray(u_cluster[pos])
    pool_is_rep[order] = np.asarray(u_is_rep[pos])
    del order, sk, pos, hit, keys, table, u_keys, u_cluster, u_is_rep

    arrays, counts, cap = aggregate(pool_cluster, pool_is_rep, raw_sizes, val_ratio,
                                    max_clusters, holdout_seed)
    index_dir.mkdir(parents=True, exist_ok=True)
    arrays.update(pool_cluster=pool_cluster, pool_is_rep=pool_is_rep, pool_lengths=lengths,
                  shard_row_offsets=sro, cluster_raw_sizes=raw_sizes)
    for name, arr in arrays.items():
        _save(index_dir / f"{name}.npy", arr)

    tc = arrays["train_clusters"]
    n_c = arrays["cluster_sizes_filtered"]
    sums = _cluster_length_sums(lengths, arrays["cluster_members"], arrays["cluster_offsets"])
    g2u = arrays["global_to_universe"]
    expected_tokens = float((sums[g2u[tc]] / n_c[tc]).sum())
    counts.update(pool_rows=n, pool_tokens=int(lengths.sum(dtype=np.int64)), shards=len(sro) - 1,
                  expected_tokens_per_pass=int(round(expected_tokens)),
                  val_rep_tokens=int(lengths[arrays["val_rep_rows"]].sum(dtype=np.int64)))
    manifest = {
        "tool": "nanoplm.data.preparation.cluster_index",
        "tool_version": TOOL_VERSION,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "level": level,
        "clusters": {"table": str(members_tsv), "hash_function": uh.HASH_NAME,
                     **(clusters_info or {})},
        "pool": {"fasta": str(fasta), "fasta_size_bytes": fasta.stat().st_size,
                 "shards": str(pool_dir)},
        "filter": {"max_seq_len": max_seq_len,
                   "token_length_rule": f"tokens = min(residues + {specials}, {max_seq_len})"},
        "counts": counts,
        "holdout": {"seed": holdout_seed, "val_ratio": val_ratio,
                    "drawn_from": "universe-local cluster indices, numpy Generator(PCG64)"
                                  ".choice(replace=False, shuffle=False), sorted"},
        "cap": cap,
        "sampling": "one visit per train cluster per pass; the member is uniform over all "
                    "n_c surviving members, representative included",
        "seconds": round(time.time() - t0, 1),
    }
    _write_json(index_dir / INDEX_MANIFEST, manifest)
    result = verify_cluster_index(index_dir, pool_dir)
    manifest["verify"] = result
    _write_json(index_dir / INDEX_MANIFEST, manifest)
    if not result["ok"]:
        raise ClusterIndexError(f"{index_dir}: verify failed: {result['failures']}")
    logger.info(
        f"cluster index: {counts['clusters_in_universe']:,} clusters in the universe, "
        f"{counts['train_clusters']:,} train / {counts['val_clusters']:,} val, "
        f"{n:,} pool rows ({manifest['seconds']}s)")
    return manifest


def aggregate(pool_cluster, pool_is_rep, raw_sizes, val_ratio, max_clusters, holdout_seed):
    """Sizes, universe, cap, CSR and holdout. Returns (arrays, counts, cap record)."""
    n = len(pool_cluster)
    c_total = len(raw_sizes)
    sizes = np.bincount(pool_cluster, minlength=c_total)
    if np.any(sizes > raw_sizes):
        raise ClusterIndexError("a cluster has more surviving members than raw members")
    rep_rows = np.flatnonzero(pool_is_rep)
    rep_clusters = pool_cluster[rep_rows]
    per = np.bincount(rep_clusters, minlength=c_total)
    if per.max(initial=0) > 1:
        raise ClusterIndexError(f"cluster {int(np.argmax(per))} has two representative rows")
    eligible = np.flatnonzero(per > 0)
    cap = {"max_clusters": int(max_clusters), "clusters_before_cap": int(len(eligible)),
           "seed": holdout_seed + CAP_SEED_OFFSET, "applied": False}
    if 0 < max_clusters < len(eligible):
        eligible = eligible[draw_cap(len(eligible), int(max_clusters), cap["seed"])]
        cap["applied"] = True
    universe_mask = np.zeros(c_total, dtype=bool)
    universe_mask[eligible] = True
    cluster_rep_row = np.full(c_total, -1, dtype=np.int32)
    keep = universe_mask[rep_clusters]
    cluster_rep_row[rep_clusters[keep]] = rep_rows[keep].astype(np.int32)
    universe = eligible.astype(np.int32)
    cu = len(universe)
    g2u = np.full(c_total, -1, dtype=np.int32)
    g2u[universe] = np.arange(cu, dtype=np.int32)
    n_u = sizes[universe]

    # stable sort by universe-local cluster: grouped, ascending pool rows within a cluster
    sel = np.flatnonzero(universe_mask[pool_cluster])
    order = np.argsort(g2u[pool_cluster[sel]], kind="stable")
    members = sel[order].astype(np.int32)
    offsets = np.zeros(cu + 1, dtype=np.int64)
    np.cumsum(n_u, dtype=np.int64, out=offsets[1:])

    val_local = draw_val_clusters(cu, val_ratio, holdout_seed)
    if len(val_local) == 0 or len(val_local) == cu:
        raise ClusterIndexError(f"val_ratio={val_ratio} over {cu} clusters holds out "
                                f"{len(val_local)}; need at least one train and one val")
    in_val = np.zeros(cu, dtype=bool)
    in_val[val_local] = True
    val_clusters = universe[val_local]
    train_clusters = universe[~in_val]
    val_rep_rows = cluster_rep_row[val_clusters]
    arrays = {
        "cluster_sizes_filtered": sizes.astype(np.int32),
        "universe_mask": universe_mask,
        "cluster_rep_row": cluster_rep_row,
        "universe_clusters": universe,
        "global_to_universe": g2u,
        "cluster_offsets": offsets,
        "cluster_members": members,
        "val_clusters": val_clusters.astype(np.int32),
        "val_rep_rows": val_rep_rows.astype(np.int32),
        "train_clusters": train_clusters.astype(np.int32),
    }
    counts = {
        "clusters_total": int(c_total),
        "clusters_with_any_survivor": int((sizes > 0).sum()),
        "clusters_in_universe": int(cu),
        "universe_members": int(len(members)),
        "universe_singletons": int((n_u == 1).sum()),
        "val_clusters": int(len(val_clusters)),
        "val_clusters_multimember": int((n_u[val_local] >= 2).sum()),
        "train_clusters": int(len(train_clusters)),
        "train_members": int(n_u[~in_val].sum()),
        "pool_rows_outside_universe": int(n - len(members)),
    }
    return arrays, counts, cap


def _cluster_length_sums(lengths, members, offsets, chunk: int = 1 << 22) -> np.ndarray:
    """Sum of member token counts per universe cluster."""
    cs = np.zeros(len(members) + 1, dtype=np.int64)
    for a in range(0, len(members), chunk):
        b = min(a + chunk, len(members))
        cs[a + 1:b + 1] = np.cumsum(lengths[members[a:b]], dtype=np.int64) + cs[a]
    return cs[offsets[1:]] - cs[offsets[:-1]]


def _decode_check(pool_dir, sro, lengths, spot_seqs, tokenizer, max_seq_len, scratch) -> None:
    """Re-tokenize the spot records with the shard writer; their pool rows must match."""
    import shutil
    import tempfile

    from nanoplm.data.sharding.pretrain_shard_writer import ShardWriter

    rows = sorted(spot_seqs)
    if not rows:
        return
    parent = Path(scratch).parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".decode_check_", dir=parent))
    try:
        fa = tmp / "spot.fasta"
        fa.write_bytes(b"".join(b">r%d\n%s\n" % (r, spot_seqs[r]) for r in rows))
        ShardWriter(str(fa), tokenizer, max_seq_len, str(tmp / "shards"),
                    samples_per_shard=len(rows) + 1, max_workers=1).create_shards()
        want = np.fromfile(tmp / "shards" / "shard_0000.bin", dtype=np.uint8)
        offs = np.concatenate([[0], np.cumsum(np.load(tmp / "shards" / "shard_0000.idx.npy"),
                                              dtype=np.int64)])
        bins = pool_shard_paths(pool_dir)
        for k, r in enumerate(rows):
            s = int(np.searchsorted(sro, r, side="right") - 1)
            start = int(lengths[sro[s]:r].sum(dtype=np.int64))
            got = np.fromfile(bins[s], dtype=np.uint8, count=int(lengths[r]), offset=start)
            if not np.array_equal(got, want[offs[k]:offs[k + 1]]):
                raise ClusterIndexError(
                    f"pool row {r} is not the tokenization of FASTA record {r}: the shards "
                    "and the FASTA are not the same rows in the same order")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def shard_size_for(n_rows: int, samples_per_shard: int, what: str = "shards") -> int:
    """``samples_per_shard``, raised when ``n_rows`` would need SHARD_SORT_CLIFF+ shards.

    The raised value is the smallest 1, 2 or 5 x 10^k that stays below the cliff (10000
    for UniRef90's 97.8M rows), so the data.yaml template's value works at any scale.
    """
    if -(-n_rows // samples_per_shard) < SHARD_SORT_CLIFF:
        return samples_per_shard
    need = -(-n_rows // (SHARD_SORT_CLIFF - 1))
    scale = 10 ** (len(str(need)) - 1)
    size = next(m * scale for m in (1, 2, 5, 10) if m * scale >= need)
    logger.info(f"{what}: {n_rows:,} rows at samples_per_shard={samples_per_shard} would make "
                f"{-(-n_rows // samples_per_shard):,} shards (the limit is "
                f"{SHARD_SORT_CLIFF - 1:,}); using samples_per_shard={size} "
                f"({-(-n_rows // size):,} shards)")
    return size


def write_row_gather(pool_dir: Path, rows: np.ndarray, out_dir: Path,
                     samples_per_shard: int) -> int:
    """Copy pool ``rows`` (already tokenized), in the given order, into new shards.

    Global dataset index i of the result is pool row ``rows[i]``. Returns the row count.
    """
    from nanoplm.data.infra.atomic import atomic_write_dir

    lengths, sro = read_pool_lengths(pool_dir)
    bins = pool_shard_paths(pool_dir)
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        raise ClusterIndexError(f"no rows to write into {out_dir}")
    samples_per_shard = shard_size_for(len(rows), samples_per_shard, str(out_dir))
    shard_of = np.searchsorted(sro, rows, side="right") - 1
    offsets: dict = {}
    maps: dict = {}
    with atomic_write_dir(Path(out_dir)) as tmp:
        for k, a in enumerate(range(0, len(rows), samples_per_shard)):
            chunk = rows[a:a + samples_per_shard]
            parts = []
            for r, s in zip(chunk.tolist(), shard_of[a:a + samples_per_shard].tolist()):
                if s not in offsets:
                    offsets[s] = np.concatenate([[0], np.cumsum(
                        lengths[sro[s]:sro[s + 1]], dtype=np.int64)])
                    maps[s] = np.memmap(bins[s], dtype=np.uint8, mode="r")
                o = offsets[s]
                lr = r - int(sro[s])
                parts.append(maps[s][int(o[lr]):int(o[lr + 1])])
            (tmp / f"shard_{k:04d}.bin").write_bytes(np.concatenate(parts).tobytes())
            with open(tmp / f"shard_{k:04d}.idx.npy", "wb") as f:
                np.save(f, lengths[chunk].astype(np.int32))
    return len(rows)


# ------------------------------------------------------------------- verify


def verify_cluster_index(index_dir: Path, pool_dir: Path) -> dict:
    """Re-check every structural invariant of a written index.

    Returns {"ok": bool, "checks": {name: bool}, "failures": [names]}.
    """
    ix = Path(index_dir)
    man = json.loads((ix / INDEX_MANIFEST).read_text())
    L = lambda name: np.load(ix / f"{name}.npy")  # noqa: E731
    checks: dict[str, bool] = {}

    def check(name, ok):
        checks[name] = bool(ok)

    pc, rep, lengths = L("pool_cluster"), L("pool_is_rep"), L("pool_lengths")
    sizes, umask, rep_row = L("cluster_sizes_filtered"), L("universe_mask"), L("cluster_rep_row")
    universe, g2u = L("universe_clusters"), L("global_to_universe")
    off, mem = L("cluster_offsets"), L("cluster_members")
    val, val_rep, train = L("val_clusters"), L("val_rep_rows"), L("train_clusters")
    sro = L("shard_row_offsets")
    n, c = len(pc), len(sizes)

    check("arrays_aligned", len(rep) == n == len(lengths) and len(umask) == c == len(rep_row)
          == len(g2u))
    check("dtypes", pc.dtype == np.int32 and mem.dtype == np.int32 and off.dtype == np.int64
          and train.dtype == np.int32 and g2u.dtype == np.int32)
    check("pool_cluster_in_range", n == 0 or (pc.min() >= 0 and pc.max() < c))
    check("bincount_matches_sizes", np.array_equal(np.bincount(pc, minlength=c), sizes))
    if (ix / "cluster_raw_sizes.npy").exists():
        raw = L("cluster_raw_sizes")
        check("sizes_le_raw_sizes", len(raw) == c and np.all(sizes <= raw))
    in_u = np.flatnonzero(umask)
    check("universe_is_the_mask", np.array_equal(in_u, universe))
    check("universe_mapping_round_trips",
          np.array_equal(g2u[universe], np.arange(len(universe)))
          and int((g2u >= 0).sum()) == len(universe))
    ok_rep = (np.all(rep_row[~umask] == -1) and np.all(rep_row[universe] >= 0)
              and np.all(rep[rep_row[universe]]) and np.array_equal(pc[rep_row[universe]], universe))
    check("one_surviving_rep_row_per_universe_cluster", ok_rep)
    n_u = sizes[universe].astype(np.int64)
    check("csr_offsets", off[0] == 0 and np.all(np.diff(off) >= 1) and off[-1] == len(mem))
    check("csr_segment_lengths_are_n_c", np.array_equal(np.diff(off), n_u))
    check("csr_members_in_range", len(mem) == 0 or (mem.min() >= 0 and mem.max() < n))
    seg_ok = checks["csr_segment_lengths_are_n_c"] and checks["csr_members_in_range"]
    check("csr_member_cluster_agrees_with_pool_cluster",
          seg_ok and np.array_equal(pc[mem], np.repeat(universe, n_u)))
    inc = np.diff(mem.astype(np.int64))
    within = np.ones(len(inc), dtype=bool)
    if seg_ok and len(inc):
        within[off[1:-1] - 1] = False      # pairs that straddle two clusters
    check("csr_members_strictly_increasing_within_a_cluster", np.all(inc[within] > 0))
    check("csr_covers_every_universe_member", int(umask[pc].sum()) == len(mem)
          and len(np.unique(mem)) == len(mem))
    ho = man.get("holdout", {})
    cu = len(universe)
    want_val = universe[draw_val_clusters(cu, float(ho.get("val_ratio", 0)),
                                          int(ho.get("seed", HOLDOUT_SEED)))] if cu else val
    check("val_reproducible_from_seed", np.array_equal(val, want_val))
    check("val_sorted_unique_in_universe", np.all(np.diff(val) > 0) and np.all(umask[val]))
    check("train_is_universe_minus_val",
          np.array_equal(train, np.setdiff1d(universe, val)) and not np.isin(val, train).any())
    check("val_rep_rows_are_the_held_out_representatives", np.array_equal(val_rep, rep_row[val]))
    lens, sro_shards = read_pool_lengths(pool_dir)
    check("shard_row_offsets_match_the_shards", np.array_equal(sro, sro_shards))
    check("pool_lengths_is_the_concatenated_shard_index", np.array_equal(lengths, lens))
    check("shard_count_below_the_sort_cliff", len(sro) - 1 < SHARD_SORT_CLIFF)
    failures = [k for k, v in checks.items() if not v]
    return {"ok": not failures, "checks": checks, "failures": failures}
