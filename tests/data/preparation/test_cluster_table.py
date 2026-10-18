"""members.tsv from the three cluster sources, and the index built from each."""
from __future__ import annotations

import numpy as np
import pytest

from nanoplm.data.datasets.cluster_sampled import ClusterSampledDataset
from nanoplm.data.preparation import cluster_index as ci
from nanoplm.data.preparation import cluster_table as ct
from nanoplm.data.preparation.cluster_pipeline import build_members_table
from tests.data.datasets.cluster_pool import REPO
from tests.data.preparation.test_cluster_index import MAX_LEN, _pool
from tests.data.preparation.uniref_fixture import make_release

AGENT_MMSEQS = REPO / "agent_space/cluster-sampling-port/mmseqs-dl/mmseqs/bin/mmseqs"
MMSEQS = ct.find_mmseqs(str(AGENT_MMSEQS) if AGENT_MMSEQS.exists() else None)
needs_mmseqs = pytest.mark.skipif(MMSEQS is None, reason="mmseqs not installed")


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return make_release(tmp_path_factory.mktemp("release"), n_clusters=200)


@pytest.fixture(scope="module")
def pool(tmp_path_factory, release):
    root = tmp_path_factory.mktemp("pool")
    return root, _pool(root, release.fasta90)


def _groups(tsv):
    out = {}
    for line in tsv.read_text().splitlines():
        c, m = line.split("\t")
        out.setdefault(c, []).append(m)
    return out


def _index(root, filtered, tsv, name):
    man = ci.build_cluster_index(filtered, root / "train", tsv, root / name, MAX_LEN, 0.1)
    assert man["verify"]["ok"]
    return man


def test_user_table_canonicalised_with_singletons(pool, tmp_path):
    root, filtered = pool
    ids = ct.fasta_ids(filtered)
    # an ungrouped table over the first half of the pool, clusters of three
    rows = [(f"fam{i // 3}", ids[i].decode()) for i in range(len(ids) // 2)]
    rows = rows[1::2] + rows[::2]
    rows.append(("fam0", "UniRef90_NOT_IN_POOL"))
    last = ids[-1].decode()
    rows.append((last, last))                     # a cluster named after its member
    table = tmp_path / "t.tsv"
    table.write_text("# my clusters\n" + "".join(f"{c}\t{m}\textra\n" for c, m in rows))
    out = tmp_path / "members.tsv"
    stats = build_members_table("table", out, fasta=filtered, table=table)
    g = _groups(out)
    half = len(ids) // 2
    assert stats["singletons_added"] == len(ids) - half - 1   # -1: `last` is in the table
    for i in range(half, len(ids)):
        m = ids[i].decode()
        assert g[m] == [m]                                    # missing -> singleton
    assert "UniRef90_NOT_IN_POOL" in g["fam0"]
    man = _index(root, filtered, out, "ix_table")
    assert man["counts"]["universe_singletons"] >= len(ids) - half


def test_user_table_rep_is_the_member_named_like_the_cluster(tmp_path):
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">a\nMKV\n>b\nMKV\n>c\nMKV\n")
    table = tmp_path / "t.tsv"
    table.write_text("b\ta\nb\tb\nx\tc\n")
    ct.table_members_tsv(table, fasta, tmp_path / "m.tsv")
    assert (tmp_path / "m.tsv").read_text() == "b\tb\nb\ta\nx\tc\n"


def test_a_sequence_in_two_clusters_is_refused(tmp_path):
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">a\nMKV\n")
    table = tmp_path / "t.tsv"
    table.write_text("x\ta\ny\ta\n")
    with pytest.raises(ct.ClusterTableError, match="in clusters x and y"):
        ct.table_members_tsv(table, fasta, tmp_path / "m.tsv")


def test_reader_needs_a_grouped_table(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text("x\ta\ny\tb\nx\tc\n")
    with pytest.raises(ct.ClusterTableError, match="grouped by cluster"):
        ct.read_members_table(p)
    p.write_text("x\ta\nx\ta\n")
    with pytest.raises(ct.ClusterTableError, match="appear twice"):
        ct.read_members_table(p)
    p.write_text("x a\n")
    with pytest.raises(ct.ClusterTableError, match="cluster_id<TAB>sequence_id"):
        ct.read_members_table(p)


def test_xml_table_feeds_the_index(release, pool, tmp_path):
    root, filtered = pool
    out = tmp_path / "members.tsv"
    stats = build_members_table("uniref_xml", out, fasta=filtered, level="uniref90",
                                xml=release.xml)
    assert stats["source"] == "uniref50_xml" and out.with_suffix(".json").exists()
    _index(root, filtered, out, "ix_xml")
    with pytest.raises(ValueError, match="holds uniref90 sequences"):
        build_members_table("uniref_xml", out, fasta=filtered, level="uniref100",
                            xml=release.xml)


def test_missing_mmseqs_says_how_to_install(pool, tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "find_mmseqs", lambda explicit=None: None)
    with pytest.raises(ct.ClusterTableError, match="nanoplm deps install --mmseqs-only"):
        ct.mmseqs_members_tsv(pool[1], tmp_path / "m.tsv", 0.5, 0.8, tmp_path / "tmp")


@needs_mmseqs
def test_mmseqs_clusters_a_tiny_fasta(tmp_path):
    rng = np.random.default_rng(0)
    aa = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
    recs = []
    for fam in range(8):
        base = aa[rng.integers(0, 20, 120)]
        for k in range(4):              # 4 close variants per family: ~95% identity
            s = base.copy()
            pos = rng.integers(0, 120, 6)
            s[pos] = aa[rng.integers(0, 20, 6)]
            recs.append((f"fam{fam}_{k}", s.tobytes().decode()))
    for k in range(5):                  # unrelated sequences stay singletons
        recs.append((f"solo{k}", aa[rng.integers(0, 20, 90)].tobytes().decode()))
    fasta = tmp_path / "own.fasta"
    fasta.write_text("".join(f">{n} desc\n{s}\n" for n, s in recs))
    out = tmp_path / "members.tsv"
    stats = build_members_table("mmseqs", out, fasta=fasta, identity=0.5, coverage=0.8,
                                tmp_dir=tmp_path / "mm", mmseqs=MMSEQS, threads=2)
    assert stats["source"] == "mmseqs2" and stats["cov_mode"] == 0
    assert not (tmp_path / "mm").exists()
    g = _groups(out)
    assert sum(len(v) for v in g.values()) == len(recs)
    fams = {frozenset(m.split("_")[0] for m in ms) for ms in g.values()}
    assert all(len(f) == 1 for f in fams)                # no cluster mixes families
    assert len([c for c, ms in g.items() if c.startswith("fam")]) == 8
    assert all(len(g[f"solo{k}"]) == 1 for k in range(5))
    for c, ms in g.items():
        assert ms[0] == c                                 # rep first


@needs_mmseqs
def test_mmseqs_table_feeds_the_index(pool, tmp_path):
    root, filtered = pool
    out = tmp_path / "members.tsv"
    build_members_table("mmseqs", out, fasta=filtered, identity=0.5, coverage=0.8,
                        tmp_dir=tmp_path / "mm", mmseqs=MMSEQS, threads=2)
    _index(root, filtered, out, "ix_mm")
    ds = ClusterSampledDataset(root / "train", root / "ix_mm", seed=1)
    assert len(ds) > 0


def _families(headers_of):
    """4 close variants per family plus unrelated singletons, with the given headers."""
    rng = np.random.default_rng(0)
    aa = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
    recs = []
    for fam in range(6):
        base = aa[rng.integers(0, 20, 120)]
        for k in range(4):
            s = base.copy()
            pos = rng.integers(0, 120, 6)
            s[pos] = aa[rng.integers(0, 20, 6)]
            recs.append((headers_of(fam, k), s.tobytes().decode()))
    for k in range(3):
        recs.append((headers_of(100 + k, 0), aa[rng.integers(0, 20, 90)].tobytes().decode()))
    return recs


PIPE_HEADERS = {
    "uniprot": lambda f, k: f"{'sp' if k % 2 else 'tr'}|P{f:03d}{k:02d}|F{f}_HUMAN Protein",
    "genbank": lambda f, k: f"gi|{1000 * f + k}|ref|NP_{f:04d}{k}.1| protein",
}


@needs_mmseqs
@pytest.mark.parametrize("style", sorted(PIPE_HEADERS))
def test_mmseqs_keeps_clusters_with_pipe_headers(tmp_path, style):
    recs = _families(PIPE_HEADERS[style])
    fasta = tmp_path / "own.fasta"
    fasta.write_text("".join(f">{h}\n{s}\n" for h, s in recs))
    out = tmp_path / "members.tsv"
    stats = build_members_table("mmseqs", out, fasta=fasta, identity=0.5, coverage=0.8,
                                tmp_dir=tmp_path / "mm", mmseqs=MMSEQS, threads=2)
    g = _groups(out)
    ids = [h.split()[0] for h, _ in recs]
    assert sorted(m for ms in g.values() for m in ms) == sorted(ids)
    assert stats["singletons_added"] == 0
    assert stats["clusters"] == len(g) == 6 + 3          # not 27 singletons
    for c, ms in g.items():
        assert ms[0] == c and c in ids


def test_mmseqs_rows_map_back_to_fasta_ids(tmp_path):
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">sp|P1|A x\nMKV\n>tr|Q2|B\nMKV\n>gi|7|ref|NP_1|\nMKV\n")
    ids = ct.number_fasta(fasta, tmp_path / "rows.fasta")
    assert (tmp_path / "rows.fasta").read_text() == ">0\nMKV\n>1\nMKV\n>2\nMKV\n"
    (tmp_path / "c.tsv").write_text("2\t2\n2\t0\n1\t1\n")
    stats = ct.mmseqs_rows_to_members_tsv(tmp_path / "c.tsv", ids, tmp_path / "m.tsv")
    assert (tmp_path / "m.tsv").read_text() == (
        "gi|7|ref|NP_1|\tgi|7|ref|NP_1|\ngi|7|ref|NP_1|\tsp|P1|A\ntr|Q2|B\ttr|Q2|B\n")
    assert stats["clusters"] == 2 and stats["members"] == 3


def test_mmseqs_rows_missing_a_sequence_is_an_error(tmp_path):
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">a\nMKV\n>b\nMKV\n>c\nMKV\n")
    ids = ct.number_fasta(fasta, tmp_path / "rows.fasta")
    (tmp_path / "c.tsv").write_text("0\t0\n0\t1\n")
    with pytest.raises(ct.ClusterTableError, match="1 missing"):
        ct.mmseqs_rows_to_members_tsv(tmp_path / "c.tsv", ids, tmp_path / "m.tsv")


def test_user_table_with_other_ids_is_refused(tmp_path):
    fasta = tmp_path / "f.fasta"
    fasta.write_text("".join(f">sp|P{i:05d}|X_HUMAN d\nMKV\n" for i in range(20)))
    table = tmp_path / "t.tsv"
    table.write_text("".join(f"c{i // 4}\tP{i:05d}\n" for i in range(20)))
    with pytest.raises(ct.ClusterTableError, match=r"only 0\.0%.*P00000.*sp\|P00000\|X_HUMAN"):
        ct.table_members_tsv(table, fasta, tmp_path / "m.tsv")


def test_user_table_partial_match_warns_with_examples(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ct.logger, "warning", lambda msg, *a, **k: seen.append(msg))
    fasta = tmp_path / "f.fasta"
    fasta.write_text("".join(f">s{i}\nMKV\n" for i in range(10)))
    table = tmp_path / "t.tsv"
    table.write_text("".join(f"c\ts{i}\n" for i in range(10)) + "c\tgone1\nc\tgone2\n")
    ct.table_members_tsv(table, fasta, tmp_path / "m.tsv")
    assert len(seen) == 1 and "2 of 12" in seen[0] and "gone1, gone2" in seen[0]


def test_user_table_peak_allocation_stays_near_the_table_size(tmp_path):
    import tracemalloc
    n = 600_000
    table, fasta = tmp_path / "t.tsv", tmp_path / "f.fasta"
    table.write_text("".join(f"UniRef90_A{(i // 3) * 3:09d}\tUniRef90_A{i:09d}\n"
                             for i in range(n)))
    fasta.write_text("".join(f">UniRef90_A{i:09d} x\nMKV\n" for i in range(n)))
    tracemalloc.start()
    try:
        ct.table_members_tsv(table, fasta, tmp_path / "m.tsv")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # the dict version needs ~320 B per line; the table itself is 40 B per line
    assert peak / n < 230, f"{peak / n:.0f} B per line"
