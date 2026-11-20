"""subclusters: the two-level arrays of the cluster index (UniRef50 -> UniRef90 -> UniRef100),
against a brute-force construction, through the index build, and their verify checks."""
from __future__ import annotations

import gzip
import json
import re
import shutil

import numpy as np
import pytest

from nanoplm.data.preparation import cluster_index as ci
from nanoplm.data.preparation import subclusters as sc
from nanoplm.data.preparation import uniref_hierarchy as uh
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from tests.data.preparation.test_cluster_index import VAL_RATIO, _pool
from tests.data.preparation.uniref_fixture import MAX_LEN, make_release

# ------------------------------------------------------------ brute force


def _brute(pc, ps, universe):
    sm, so, cso = [], [0], [0]
    for c in universe:
        groups: dict = {}
        for r in np.flatnonzero(pc == c):                 # ascending rows
            groups.setdefault(int(ps[r]), []).append(int(r))
        for g in sorted(groups.values(), key=min):
            sm += g
            so.append(len(sm))
        cso.append(len(so) - 1)
    return np.array(sm, np.int32), np.array(so, np.uint32), np.array(cso, np.uint32)


def _random_case(seed, singletons=False, n_clusters=60):
    """pool_cluster / pool_is_rep / raw_sizes plus a sub-cluster table nested in it."""
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 1, 2, 3, 5, 12], n_clusters)
    pc = np.repeat(np.arange(n_clusters), sizes).astype(np.int32)
    is_rep = np.zeros(len(pc), bool)
    is_rep[np.cumsum(sizes) - sizes] = True
    # sub-clusters inside every cluster: random partitions (or all singletons)
    sub = np.empty(len(pc), np.int64)
    nxt = 0
    for a, b in zip(np.cumsum(sizes) - sizes, np.cumsum(sizes)):
        k = b - a if singletons else int(rng.integers(1, b - a + 1))
        lab = np.arange(b - a) if singletons else rng.integers(0, k, b - a)
        sub[a:b] = nxt + lab
        nxt += k
    sub = rng.permutation(nxt)[sub]                       # table order unrelated to clusters
    # filter: drop rows from the pool (some reps, so some clusters leave the universe)
    keep = rng.random(len(pc)) > 0.15
    perm = rng.permutation(int(keep.sum()))               # pool row order unrelated
    pool_c, pool_rep, pool_s = pc[keep][perm], is_rep[keep][perm], sub[keep][perm].astype(np.int32)
    return pool_c, pool_rep, sizes.astype(np.int32), pool_s, nxt


def _arrays(case, val_ratio=0.2):
    pc, rep, raw, ps, n_table = case
    arrays, _, _ = ci.aggregate(pc, rep, raw, val_ratio, -1, ci.HOLDOUT_SEED)
    sub, counts = sc.build_subcluster_arrays(pc, ps, arrays["cluster_members"],
                                             arrays["cluster_offsets"], n_table)
    return arrays, sub, counts


@pytest.mark.parametrize("seed", range(8))
def test_arrays_equal_brute_force(seed):
    case = _random_case(seed)
    arrays, sub, counts = _arrays(case)
    sm, so, cso = _brute(case[0], case[3], arrays["universe_clusters"])
    assert sub["sub_members"].dtype == np.int32 and sub["sub_offsets"].dtype == np.uint32
    assert sub["cluster_sub_offsets"].dtype == np.uint32
    assert np.array_equal(sub["sub_members"], sm)
    assert np.array_equal(sub["sub_offsets"], so)
    assert np.array_equal(sub["cluster_sub_offsets"], cso)
    assert np.array_equal(sub["pool_subcluster"], case[3])
    assert counts["subclusters"] == len(so) - 1
    assert counts["universe_members"] == len(arrays["cluster_members"])
    n_sub = np.diff(cso.astype(np.int64))
    assert counts["clusters_with_one_subcluster"] == int((n_sub == 1).sum())
    assert counts["max_subclusters_per_cluster"] == int(n_sub.max())
    # the case covers what it should: one-member clusters, one-sub-cluster clusters with
    # several members, clusters with several sub-clusters, and clusters out of the universe
    n_u = np.diff(arrays["cluster_offsets"])
    assert (n_u == 1).any() and ((n_sub == 1) & (n_u > 1)).any() and (n_sub > 1).any()
    assert len(arrays["universe_clusters"]) < len(case[2])


def test_all_singleton_subclusters_equal_cluster_members():
    case = _random_case(11, singletons=True)
    arrays, sub, counts = _arrays(case)
    assert np.array_equal(sub["sub_members"], arrays["cluster_members"])
    m = len(arrays["cluster_members"])
    assert np.array_equal(sub["sub_offsets"], np.arange(m + 1))
    assert np.array_equal(sub["cluster_sub_offsets"], arrays["cluster_offsets"])
    assert counts["clusters_all_singleton_subclusters"] == len(arrays["universe_clusters"])


def test_rows_outside_the_universe_may_miss_the_table():
    pc, rep, raw, ps, n_table = _random_case(4)
    arrays, _, _ = ci.aggregate(pc, rep, raw, 0.2, -1, ci.HOLDOUT_SEED)
    out = ~arrays["universe_mask"][pc]
    assert out.any()
    ps = ps.copy()
    ps[out] = -1
    sub, _ = sc.build_subcluster_arrays(pc, ps, arrays["cluster_members"],
                                        arrays["cluster_offsets"], n_table)
    want = _brute(pc, ps, arrays["universe_clusters"])
    assert np.array_equal(sub["sub_members"], want[0])


def test_a_kept_member_missing_from_the_table_is_an_error():
    pc, rep, raw, ps, n_table = _random_case(5)
    arrays, _, _ = ci.aggregate(pc, rep, raw, 0.2, -1, ci.HOLDOUT_SEED)
    ps = ps.copy()
    ps[arrays["cluster_members"][[3, 7]]] = -1
    with pytest.raises(ci.ClusterIndexError, match=r"2 pool sequences of kept clusters are "
                       r"not in tbl \(pool rows \[\d+, \d+\]\)"):
        sc.build_subcluster_arrays(pc, ps, arrays["cluster_members"],
                                   arrays["cluster_offsets"], n_table, "tbl")


def test_a_subcluster_spanning_two_clusters_is_an_error():
    pc, rep, raw, ps, n_table = _random_case(6)
    arrays, _, _ = ci.aggregate(pc, rep, raw, 0.2, -1, ci.HOLDOUT_SEED)
    ps = ps.copy()
    a, b = np.flatnonzero(pc == pc[0])[0], np.flatnonzero(pc != pc[0])[0]
    ps[b] = ps[a]
    with pytest.raises(ci.ClusterIndexError, match="1 sub-clusters of tbl span two clusters"):
        sc.build_subcluster_arrays(pc, ps, arrays["cluster_members"],
                                   arrays["cluster_offsets"], n_table, "tbl")


def test_spanning_into_a_cluster_outside_the_universe_is_an_error():
    pc, rep, raw, ps, n_table = _random_case(7)
    arrays, _, _ = ci.aggregate(pc, rep, raw, 0.2, -1, ci.HOLDOUT_SEED)
    um = arrays["universe_mask"]
    a, b = np.flatnonzero(um[pc])[0], np.flatnonzero(~um[pc])[0]
    ps = ps.copy()
    ps[b] = ps[a]
    with pytest.raises(ci.ClusterIndexError, match="span two clusters"):
        sc.build_subcluster_arrays(pc, ps, arrays["cluster_members"],
                                   arrays["cluster_offsets"], n_table)


def test_expected_tokens_is_the_mean_over_subclusters_of_their_mean():
    case = _random_case(9)
    arrays, sub, _ = _arrays(case)
    lengths = np.random.default_rng(0).integers(20, 300, len(case[0])).astype(np.int32)
    g2u = arrays["global_to_universe"]
    tc = arrays["train_clusters"]
    sm, so, cso = sub["sub_members"], sub["sub_offsets"], sub["cluster_sub_offsets"]
    want = 0.0
    for u in g2u[tc]:
        means = [lengths[sm[so[j]:so[j + 1]]].mean() for j in range(cso[u], cso[u + 1])]
        want += np.mean(means)
    got = sc.expected_tokens_per_pass(lengths, sm, so, cso, g2u[tc])
    assert got == pytest.approx(want, rel=1e-12)


def test_digest_depends_on_every_array_and_its_dtype():
    case = _random_case(2)
    _, sub, _ = _arrays(case)
    arrs = [sub[k] for k in sc.SUB_ARRAYS]
    d = sc.subcluster_digest(*arrs)
    assert d.startswith("sha256:") and d == sc.subcluster_digest(*[a.copy() for a in arrs])
    for i in range(3):
        other = list(arrs)
        other[i] = other[i].astype(np.int64)
        assert sc.subcluster_digest(*other) != d
    other = list(arrs)
    other[0] = other[0].copy()
    other[0][[0, 1]] = other[0][[1, 0]]
    assert sc.subcluster_digest(*other) != d


# --------------------------------------------------- through the index build


def _submembers_tsv(xml, out):
    """UniRef90<TAB>UniRef100 lines from the fixture XML, grouped by UniRef90."""
    groups: dict = {}
    seen = set()
    u100 = None
    with gzip.open(xml, "rt") as f:
        for line in f:
            m = re.search(r'type="UniRef(100|90) ID" value="(\S+)"', line)
            if not m:
                continue
            if m.group(1) == "100":
                u100 = m.group(2)
            elif u100 not in seen:
                seen.add(u100)
                groups.setdefault(m.group(2), []).append(u100)
    out.write_text("".join(f"{g}\t{a}\n" for g, ms in groups.items() for a in ms))
    return out


def _build(root, release, sub, **kw):
    filtered = _pool(root, release.fasta100, "uniref100")
    uh.xml_to_members_tsv(release.xml, root / "members.tsv", level="uniref100")
    if sub:
        kw["submembers_tsv"] = _submembers_tsv(release.xml, root / "submembers.tsv")
    return ci.build_cluster_index(filtered, root / "train", root / "members.tsv",
                                  root / "cluster_index", MAX_LEN, VAL_RATIO,
                                  level="uniref100", tokenizer=ProtModernBertTokenizer(), **kw)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return make_release(tmp_path_factory.mktemp("release"))


@pytest.fixture(scope="module")
def two(tmp_path_factory, release):
    root = tmp_path_factory.mktemp("two")
    man = _build(root, release, True, subclusters_info={"source": "uniref50_xml"})
    return root, man


@pytest.fixture(scope="module")
def one(tmp_path_factory, release):
    root = tmp_path_factory.mktemp("one")
    return root, _build(root, release, False)


def _L(root, name):
    return np.load(root / "cluster_index" / f"{name}.npy")


def test_build_writes_the_brute_force_arrays(two, release):
    root, man = two
    assert man["verify"]["ok"], man["verify"]["failures"]
    pc = _L(root, "pool_cluster")
    accs = [line[1:].split()[0] for line in (root / "uniref100_filtered.fasta").read_text()
            .splitlines() if line.startswith(">")]
    # the table's sub-cluster index is the order of first appearance of the UniRef90 id
    order = list(dict.fromkeys(
        ln.split("\t")[0] for ln in (root / "submembers.tsv").read_text().splitlines()))
    sub_of = {ln.split("\t")[1]: order.index(ln.split("\t")[0])
              for ln in (root / "submembers.tsv").read_text().splitlines()}
    ps = np.array([sub_of[a] for a in accs], np.int32)
    assert np.array_equal(_L(root, "pool_subcluster"), ps)
    sm, so, cso = _brute(pc, ps, _L(root, "universe_clusters"))
    assert np.array_equal(_L(root, "sub_members"), sm)
    assert np.array_equal(_L(root, "sub_offsets"), so)
    assert np.array_equal(_L(root, "cluster_sub_offsets"), cso)
    assert _L(root, "sub_offsets").dtype == np.uint32
    assert (np.diff(cso.astype(int)) > 1).any() and (np.diff(so.astype(int)) > 1).any()


def test_existing_outputs_are_byte_identical_with_and_without(one, two):
    a, b = one[0] / "cluster_index", two[0] / "cluster_index"
    names = sorted(p.name for p in a.glob("*.npy"))
    assert len(names) == 15
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    extra = {p.name for p in b.glob("*.npy")} - set(names)
    assert extra == {f"{n}.npy" for n in sc.ALL_ARRAYS}


def test_manifest_without_the_table_is_todays(one):
    _, man = one
    assert man["tool_version"] == 1 and "subclusters" not in man
    assert man["sampling"] == ("one visit per train cluster per pass; the member is uniform "
                               "over all n_c surviving members, representative included")
    assert not any(k.startswith("sub") for k in man["verify"]["checks"])


def test_manifest_block(one, two):
    root, man = two
    blk = man["subclusters"]
    assert man["tool_version"] == 2 and man["sampling"] == sc.SAMPLING
    assert blk["table"] == str(root / "submembers.tsv") and blk["source"] == "uniref50_xml"
    assert blk["arrays"] == list(sc.ALL_ARRAYS)
    arrs = [_L(root, k) for k in sc.SUB_ARRAYS]
    assert blk["digest"] == sc.subcluster_digest(*arrs)
    so, cso = arrs[1].astype(int), arrs[2].astype(int)
    c = blk["counts"]
    assert c["subclusters"] == len(so) - 1 and c["universe_members"] == len(arrs[0])
    assert c["table_subclusters"] == len(set(
        ln.split("\t")[0] for ln in (root / "submembers.tsv").read_text().splitlines()))
    g2u, tc = _L(root, "global_to_universe"), _L(root, "train_clusters")
    assert c["train_subclusters"] == int(np.diff(cso)[g2u[tc]].sum())
    # counts: two-level expectation; the uniform one kept in the block equals the old index
    lengths = _L(root, "pool_lengths")
    want = sc.expected_tokens_per_pass(lengths, *arrs, g2u[tc])
    assert man["counts"]["expected_tokens_per_pass"] == int(round(want))
    assert blk["expected_tokens_per_pass_uniform"] == one[1]["counts"]["expected_tokens_per_pass"]
    assert want != one[1]["counts"]["expected_tokens_per_pass"]
    others = {k: v for k, v in man["counts"].items() if k != "expected_tokens_per_pass"}
    assert others == {k: v for k, v in one[1]["counts"].items()
                      if k != "expected_tokens_per_pass"}


def test_build_refuses_a_table_that_misses_a_kept_member(two, release, tmp_path):
    root = two[0]
    pc, umask = _L(root, "pool_cluster"), _L(root, "universe_mask")
    accs = [line[1:].split()[0] for line in (root / "uniref100_filtered.fasta").read_text()
            .splitlines() if line.startswith(">")]
    drop = accs[int(np.flatnonzero(umask[pc])[0])]
    lines = [ln for ln in (root / "submembers.tsv").read_text().splitlines(True)
             if ln.split("\t")[1].strip() != drop]
    (tmp_path / "sub.tsv").write_text("".join(lines))
    with pytest.raises(ci.ClusterIndexError, match="1 pool sequences of kept clusters"):
        ci.build_cluster_index(root / "uniref100_filtered.fasta", root / "train",
                               root / "members.tsv", tmp_path / "ix", MAX_LEN, VAL_RATIO,
                               submembers_tsv=tmp_path / "sub.tsv")
    assert not (tmp_path / "ix" / "cluster_members.npy").exists()


# ------------------------------------------------------------------- verify


def _copy(two, tmp_path):
    root = two[0]
    dst = tmp_path / "copy"
    shutil.copytree(root / "cluster_index", dst / "cluster_index")
    (dst / "train").symlink_to(root / "train", target_is_directory=True)
    return dst / "cluster_index", dst / "train"


def _multi(so, cso, min_subs=2, min_size=1):
    """A cluster with at least min_subs sub-clusters whose first two hold min_size rows."""
    for u in range(len(cso) - 1):
        j = int(cso[u])
        if cso[u + 1] - j >= min_subs and all(so[j + k + 1] - so[j + k] >= min_size
                                              for k in range(min_subs)):
            return u, j
    raise AssertionError("fixture has no such cluster")


@pytest.mark.parametrize("tamper,check", [
    ("missing_array", "subcluster_arrays_present"),
    ("offsets_int64", "sub_dtypes"),
    ("so_flat", "sub_offsets_monotone"),
    ("cso_flat", "cluster_sub_offsets_monotone"),
    ("cso_shift", "sub_offsets_nested_in_cluster_csr"),
    ("foreign_row", "sub_members_same_rows_per_cluster"),
    ("rows_descending", "sub_rows_ascending_within_a_subcluster"),
    ("subs_swapped", "subclusters_ordered_by_first_row"),
    ("two_table_subs", "one_table_subcluster_per_segment"),
    ("ps_truncated", "one_table_subcluster_per_segment"),
    ("digest", "subcluster_digest_matches"),
])
def test_verify_rejects_tampered_sub_arrays(two, tmp_path, tamper, check):
    ix, train = _copy(two, tmp_path)
    assert ci.verify_cluster_index(ix, train)["ok"]
    sm, so, cso = (np.load(ix / f"{n}.npy") for n in sc.SUB_ARRAYS)
    ps = np.load(ix / "pool_subcluster.npy")
    if tamper == "missing_array":
        (ix / "sub_offsets.npy").unlink()
    elif tamper == "offsets_int64":
        np.save(ix / "sub_offsets.npy", so.astype(np.int64))
    elif tamper == "so_flat":
        so[1] = so[0]
        np.save(ix / "sub_offsets.npy", so)
    elif tamper == "cso_flat":
        cso[1] = cso[0]
        np.save(ix / "cluster_sub_offsets.npy", cso)
    elif tamper == "cso_shift":
        u, j = _multi(so, cso)
        cso[u + 1] -= 1                       # monotone still, boundary off the cluster CSR
        assert np.all(np.diff(cso.astype(int)) >= 1)
        np.save(ix / "cluster_sub_offsets.npy", cso)
    elif tamper == "foreign_row":
        umask, pc = np.load(ix / "universe_mask.npy"), np.load(ix / "pool_cluster.npy")
        sm[0] = np.flatnonzero(~umask[pc])[0]
        np.save(ix / "sub_members.npy", sm)
    elif tamper == "rows_descending":
        _, j = _multi(so, cso, 1, 2)
        a = int(so[j])
        sm[[a, a + 1]] = sm[[a + 1, a]]
        np.save(ix / "sub_members.npy", sm)
    elif tamper == "subs_swapped":
        u, j = _multi(so, cso)
        a, b, c = int(so[j]), int(so[j + 1]), int(so[j + 2])
        sm[a:c] = np.concatenate([sm[b:c], sm[a:b]])
        so[j + 1] = a + (c - b)
        np.save(ix / "sub_members.npy", sm)
        np.save(ix / "sub_offsets.npy", so)
    elif tamper == "two_table_subs":
        _, j = _multi(so, cso, 1, 2)
        ps[sm[int(so[j]) + 1]] = ps[sm[int(so[j + 1])]] if j + 1 < len(so) - 1 else ps.max() + 1
        np.save(ix / "pool_subcluster.npy", ps)
    elif tamper == "ps_truncated":
        np.save(ix / "pool_subcluster.npy", ps[:-1])
    else:
        man_p = ix / ci.INDEX_MANIFEST
        man = json.loads(man_p.read_text())
        man["subclusters"]["digest"] = "sha256:" + "0" * 64
        man_p.write_text(json.dumps(man))
    res = ci.verify_cluster_index(ix, train)
    assert not res["ok"] and check in res["failures"], res["failures"]


def test_one_level_rebuild_removes_stale_sub_arrays(release, tmp_path):
    _build(tmp_path, release, True)
    ix = tmp_path / "cluster_index"
    assert all((ix / f"{n}.npy").exists() for n in sc.ALL_ARRAYS)
    man = ci.build_cluster_index(tmp_path / "uniref100_filtered.fasta", tmp_path / "train",
                                 tmp_path / "members.tsv", ix, MAX_LEN, VAL_RATIO,
                                 level="uniref100", tokenizer=ProtModernBertTokenizer())
    assert not any((ix / f"{n}.npy").exists() for n in sc.ALL_ARRAYS)
    assert "subclusters" not in man
    assert "subclusters" not in json.loads((ix / ci.INDEX_MANIFEST).read_text())
    assert ci.verify_cluster_index(ix, tmp_path / "train")["ok"]
