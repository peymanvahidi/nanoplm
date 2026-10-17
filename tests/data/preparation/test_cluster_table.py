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
