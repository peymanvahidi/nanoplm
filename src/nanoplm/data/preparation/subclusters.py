"""Sub-clusters inside the clusters of a cluster index (UniRef50 -> UniRef90 -> UniRef100).

Input: the canonical sub-cluster table ``submembers.tsv`` (``subcluster_id<TAB>sequence_id``,
grouped by sub-cluster, the same format as ``members.tsv``, read by the same reader), and
the arrays of the cluster index. Output, next to them in the index directory:

  sub_members.npy          int32[M]      pool rows grouped by universe cluster, then by
                                         sub-cluster, ascending inside a sub-cluster
  sub_offsets.npy          uint32[S+1]   start of every sub-cluster in sub_members
                                         (int64 if M >= 2**32)
  cluster_sub_offsets.npy  uint32[Cu+1]  first sub-cluster of every universe cluster
                                         (same dtype rule)
  pool_subcluster.npy      int32[N]      pool row -> sub-cluster index of the table, -1 if
                                         the row is not in it (verify only)

Inside a cluster the sub-clusters are ordered by their smallest pool row, so when every
sub-cluster has one member ``sub_members`` equals ``cluster_members``. Training draws a
sub-cluster uniformly among the cluster's, then a member uniformly inside it.

Every member of a kept (universe) cluster must be in the table, and a sub-cluster must
not span two clusters (over all pool rows): both are build errors.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from nanoplm.data.preparation import cluster_index as ci

SUB_ARRAYS = ("sub_members", "sub_offsets", "cluster_sub_offsets")   # opened in training
ALL_ARRAYS = (*SUB_ARRAYS, "pool_subcluster")
SAMPLING = ("one visit per train cluster per pass; the sub-cluster is uniform over the "
            "cluster's sub-clusters with a surviving member, then the member is uniform over "
            "the sub-cluster's surviving members, representative included")
_EXAMPLES = 5


def offsets_dtype(m: int):
    """uint32 offsets while every position fits, else int64."""
    return np.uint32 if m < 2 ** 32 else np.int64


def join_pool_rows(sorted_keys: np.ndarray, order: np.ndarray, n: int, table,
                   ) -> np.ndarray:
    """pool row -> sub-cluster index of ``table`` (a ``MembersTable``), -1 if absent.

    ``sorted_keys`` / ``order`` are the pool's id hashes sorted and their rows, as the
    cluster join builds them.
    """
    if len(table.sizes) > np.iinfo(np.int32).max:
        raise ci.ClusterIndexError(f"{len(table.sizes)} sub-clusters exceed int32")
    pos = np.searchsorted(table.keys, sorted_keys)
    np.clip(pos, 0, len(table.keys) - 1, out=pos)
    hit = np.asarray(table.keys[pos]) == sorted_keys
    out = np.full(n, -1, dtype=np.int32)
    out[order[hit]] = np.asarray(table.cluster[pos[hit]])
    return out


def build_subcluster_arrays(pool_cluster: np.ndarray, pool_subcluster: np.ndarray,
                            members: np.ndarray, offsets: np.ndarray, n_table: int,
                            table_name: str = "the sub-cluster table") -> tuple[dict, dict]:
    """The sub arrays over the universe CSR (``members``, ``offsets``).

    Returns ({name: array} for the four files, counts).
    """
    m, cu = len(members), len(offsets) - 1
    _check_nesting(pool_cluster, pool_subcluster, n_table, table_name)
    s = pool_subcluster[members]
    miss = np.flatnonzero(s < 0)
    if miss.size:
        raise ci.ClusterIndexError(
            f"{miss.size:,} pool sequences of kept clusters are not in {table_name} (pool "
            f"rows {np.sort(members[miss[:_EXAMPLES]]).tolist()}); the sub-cluster table must "
            "list every member of the cluster table (are both from the same release?)")
    # group by sub-cluster; stable, so every group keeps ascending CSR positions, which
    # are ascending (cluster, row)
    order = np.argsort(s, kind="stable")
    ss = s[order]
    del s
    brk = np.flatnonzero(ss[1:] != ss[:-1])
    del ss
    starts = np.empty(len(brk) + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = brk + 1
    del brk
    sizes = np.diff(starts, append=m)
    first = order[starts]                    # smallest CSR position of every sub-cluster
    seg_order = np.argsort(first)            # positions are unique: no ties
    first = first[seg_order]
    new_sizes = sizes[seg_order]
    new_starts = np.cumsum(new_sizes) - new_sizes
    shift = np.empty_like(starts)
    shift[seg_order] = new_starts - starts[seg_order]
    del new_starts, seg_order, starts
    dest = np.repeat(shift, sizes)
    del shift, sizes
    dest += np.arange(m, dtype=np.int64)
    sub_members = np.empty(m, dtype=np.int32)
    sub_members[dest] = members[order]
    del dest, order

    odt = offsets_dtype(m)
    n_sub = len(new_sizes)
    sub_offsets = np.zeros(n_sub + 1, dtype=odt)
    np.cumsum(new_sizes, out=sub_offsets[1:], dtype=odt)
    seg_cluster = np.searchsorted(offsets, first, side="right") - 1
    del first
    per_cluster = np.bincount(seg_cluster, minlength=cu)
    del seg_cluster
    cluster_sub_offsets = np.zeros(cu + 1, dtype=odt)
    np.cumsum(per_cluster, out=cluster_sub_offsets[1:], dtype=odt)
    single = new_sizes == 1
    n_u = np.diff(offsets)
    counts = {
        "table_subclusters": int(n_table),
        "subclusters": int(n_sub),
        "universe_members": int(m),
        "singleton_subclusters": int(single.sum()),
        "clusters_with_one_subcluster": int((per_cluster == 1).sum()),
        "clusters_all_singleton_subclusters": int((per_cluster == n_u).sum()),
        "max_subclusters_per_cluster": int(per_cluster.max(initial=0)),
        "max_subcluster_size": int(new_sizes.max(initial=0)),
    }
    arrays = {"sub_members": sub_members, "sub_offsets": sub_offsets,
              "cluster_sub_offsets": cluster_sub_offsets, "pool_subcluster": pool_subcluster}
    return arrays, counts


def _check_nesting(pool_cluster, pool_subcluster, n_table, table_name) -> None:
    """Every sub-cluster's pool rows lie in one cluster (any one row's cluster wins the
    scatter; a sub-cluster spans two clusters iff a row disagrees with it)."""
    valid = np.flatnonzero(pool_subcluster >= 0)
    ps, pc = pool_subcluster[valid], pool_cluster[valid]
    owner = np.full(n_table, -1, dtype=np.int32)
    owner[ps] = pc
    bad = np.flatnonzero(owner[ps] != pc)
    if bad.size:
        subs = np.unique(ps[bad])
        k = int(subs[0])
        rows = valid[ps == k]
        raise ci.ClusterIndexError(
            f"{subs.size:,} sub-clusters of {table_name} span two clusters, e.g. sub-cluster "
            f"#{k} (0-based, in table order) holds pool rows {rows[:_EXAMPLES].tolist()} of "
            f"clusters {pool_cluster[rows[:_EXAMPLES]].tolist()}; every sub-cluster must lie "
            "inside one cluster (are both tables from the same release?)")


def subcluster_digest(sub_members, sub_offsets, cluster_sub_offsets) -> str:
    """sha256 over dtype, shape and bytes of the three training arrays, in that order."""
    h = hashlib.sha256()
    for name, a in zip(SUB_ARRAYS, (sub_members, sub_offsets, cluster_sub_offsets)):
        a = np.ascontiguousarray(a)
        h.update(f"{name}:{a.dtype.str}:{a.shape};".encode())
        flat = a.reshape(-1).view(np.uint8)
        step = 1 << 26
        for i in range(0, len(flat), step):
            h.update(memoryview(flat[i:i + step]))
    return "sha256:" + h.hexdigest()


def expected_tokens_per_pass(lengths, sub_members, sub_offsets, cluster_sub_offsets,
                             train_local) -> float:
    """Tokens of one pass under the two-level draw: per train cluster, the mean over its
    sub-clusters of the sub-cluster's mean member length, summed."""
    so = np.asarray(sub_offsets, dtype=np.int64)
    sums = ci._cluster_length_sums(lengths, sub_members, so)
    sub_mean = sums / np.diff(so)
    del sums
    cso = np.asarray(cluster_sub_offsets, dtype=np.int64)
    per = np.add.reduceat(sub_mean, cso[:-1]) / np.diff(cso)
    return float(per[np.asarray(train_local, dtype=np.int64)].sum())


def manifest_block(table: Path, info: dict | None, counts: dict, digest: str,
                   uniform_tokens: int) -> dict:
    return {"table": str(table), **(info or {}), "arrays": list(ALL_ARRAYS),
            "order": "inside a cluster, sub-clusters by smallest pool row; rows ascending "
                     "inside a sub-cluster",
            "counts": counts, "digest": digest,
            "expected_tokens_per_pass_uniform": uniform_tokens}


# ------------------------------------------------------------------- verify


def verify_subclusters(ix: Path, man: dict, pc, universe, off, mem, check) -> None:
    """Sub-array checks of ``verify_cluster_index`` (``check(name, ok)`` records each).

    ``pc``, ``universe``, ``off``, ``mem`` are the index's pool_cluster, universe_clusters,
    cluster_offsets and cluster_members.
    """
    present = [(ix / f"{n}.npy").exists() for n in ALL_ARRAYS]
    check("subcluster_arrays_present", all(present) and "subclusters" in man)
    if not all(present):
        return
    L = lambda name: np.load(ix / f"{name}.npy")  # noqa: E731
    sm, so, cso, ps = (L(n) for n in ALL_ARRAYS)
    n, m, cu = len(pc), len(mem), len(universe)
    odt = offsets_dtype(m)
    check("sub_dtypes", sm.dtype == np.int32 and len(sm) == m and so.dtype == odt
          and cso.dtype == odt and ps.dtype == np.int32 and len(ps) == n
          and len(cso) == cu + 1)
    so64, cso64 = so.astype(np.int64), cso.astype(np.int64)
    s = len(so64) - 1
    so_ok = bool(s >= 0 and so64[0] == 0 and np.all(np.diff(so64) >= 1)
                 and so64[-1] == len(sm))
    check("sub_offsets_monotone", so_ok)
    cso_ok = bool(len(cso64) == cu + 1 and cso64[0] == 0 and np.all(np.diff(cso64) >= 1)
                  and cso64[-1] == s)
    check("cluster_sub_offsets_monotone", cso_ok)
    nested = so_ok and cso_ok and np.array_equal(so64[cso64], off)
    check("sub_offsets_nested_in_cluster_csr", nested)
    in_range = len(sm) == 0 or (sm.min() >= 0 and sm.max() < n)
    if in_range and len(sm) == m and len(off) == cu + 1:
        seen = np.zeros(n, dtype=bool)     # unique + right cluster + n_c rows each = same set
        seen[sm] = True
        same = (int(seen.sum()) == m
                and np.array_equal(pc[sm], np.repeat(universe, np.diff(off))))
        del seen
    else:
        same = False
    check("sub_members_same_rows_per_cluster", same)
    if not nested:
        for name in ("sub_rows_ascending_within_a_subcluster",
                     "subclusters_ordered_by_first_row", "one_table_subcluster_per_segment"):
            check(name, False)
    else:
        inc = np.diff(sm.astype(np.int64))
        within = np.ones(len(inc), dtype=bool)
        within[so64[1:-1] - 1] = False       # pairs straddling two sub-clusters
        check("sub_rows_ascending_within_a_subcluster", np.all(inc[within] > 0))
        del inc
        first = sm[so64[:-1]].astype(np.int64)
        finc = np.diff(first)
        fwithin = np.ones(len(finc), dtype=bool)
        fwithin[cso64[1:-1] - 1] = False     # first sub-clusters of the next cluster
        check("subclusters_ordered_by_first_row", np.all(finc[fwithin] > 0))
        del first, finc, fwithin
        one = False
        if in_range and len(ps) == n:
            seg = ps[sm]
            heads = seg[so64[:-1]]
            if seg.min(initial=0) >= 0 and np.all(seg[1:][within] == seg[:-1][within]):
                # heads distinct: a mark instead of np.unique (110 s -> 0.5 s at 120M)
                mark = np.zeros(int(heads.max(initial=-1)) + 1, dtype=bool)
                mark[heads] = True
                one = int(mark.sum()) == len(heads)
                del mark
        check("one_table_subcluster_per_segment", one)
    want = man.get("subclusters", {}).get("digest")
    check("subcluster_digest_matches", want == subcluster_digest(sm, so, cso))
