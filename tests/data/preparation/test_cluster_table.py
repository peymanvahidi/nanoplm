"""members.tsv from the two cluster sources, and the index built from each."""
from __future__ import annotations

import numpy as np
import pytest

from nanoplm.data.datasets.cluster_sampled import ClusterSampledDataset
from nanoplm.data.preparation import cluster_index as ci
from nanoplm.data.preparation import cluster_table as ct
from nanoplm.data.preparation.cluster_table import build_members_table
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
    # clusters in representative row order, members after their rep in row order
    assert (tmp_path / "m.tsv").read_text() == (
        "tr|Q2|B\ttr|Q2|B\ngi|7|ref|NP_1|\tgi|7|ref|NP_1|\ngi|7|ref|NP_1|\tsp|P1|A\n")
    assert stats["clusters"] == 2 and stats["members"] == 3


def test_mmseqs_line_order_does_not_change_members_tsv(tmp_path):
    """MMseqs2 makes no promise about its output order; the val split is drawn over the
    cluster order in members.tsv, so a shuffled c.tsv must give the same bytes."""
    rng = np.random.default_rng(0)
    n = 500
    fasta = tmp_path / "f.fasta"
    fasta.write_text("".join(f">s{i}\nMKV\n" for i in range(n)))
    ids = ct.number_fasta(fasta, tmp_path / "rows.fasta")
    reps = np.sort(rng.choice(n, 60, replace=False))
    rep_of = np.where(np.isin(np.arange(n), reps), np.arange(n), rng.choice(reps, n))
    lines = [f"{r}\t{m}\n" for m, r in enumerate(rep_of)]
    out = []
    for k in range(3):
        rng.shuffle(lines)
        (tmp_path / "c.tsv").write_text("".join(lines))
        ct.mmseqs_rows_to_members_tsv(tmp_path / "c.tsv", ids, tmp_path / f"m{k}.tsv")
        out.append((tmp_path / f"m{k}.tsv").read_bytes())
    assert out[0] == out[1] == out[2]
    first = [line.split(b"\t") for line in out[0].splitlines()]
    assert first[0] == [f"s{reps[0]}".encode()] * 2


def test_mmseqs_rows_missing_a_sequence_is_an_error(tmp_path):
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">a\nMKV\n>b\nMKV\n>c\nMKV\n")
    ids = ct.number_fasta(fasta, tmp_path / "rows.fasta")
    (tmp_path / "c.tsv").write_text("0\t0\n0\t1\n")
    with pytest.raises(ct.ClusterTableError, match="1 missing"):
        ct.mmseqs_rows_to_members_tsv(tmp_path / "c.tsv", ids, tmp_path / "m.tsv")


def _fake_mmseqs(tmp_path):
    """Records its argv, then makes every (row-numbered) sequence its own cluster."""
    fake = tmp_path / "fake_mmseqs"
    fake.write_text(
        "#!/bin/bash\n"
        f'echo "$@" > "{tmp_path}/argv"\n'
        "grep '^>' \"$2\" | cut -c2- | awk '{print $1\"\\t\"$1}' > \"$3_cluster.tsv\"\n")
    fake.chmod(0o755)
    return fake


@pytest.mark.parametrize("threads, expect", [(3, "3"), (-1, None), (None, None)])
def test_mmseqs_threads(tmp_path, threads, expect):
    import os
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">sp|P1|A\nMKV\n>sp|P2|B\nMKV\n")
    stats = ct.mmseqs_members_tsv(fasta, tmp_path / "m.tsv", 0.5, 0.8, tmp_path / "mm",
                                  threads=threads, mmseqs=str(_fake_mmseqs(tmp_path)))
    argv = (tmp_path / "argv").read_text().split()
    got = argv[argv.index("--threads") + 1]
    assert got == (expect or str(len(os.sched_getaffinity(0))))
    assert stats["clusters"] == 2
    assert (tmp_path / "m.tsv").read_text() == "sp|P1|A\tsp|P1|A\nsp|P2|B\tsp|P2|B\n"


@pytest.mark.parametrize("env, expect", [("3", "3"), (None, None)])
def test_cluster_members_threads_come_from_the_env(tmp_path, monkeypatch, env, expect):
    """from-yaml passes data_params.max_workers as $NANOPLM_MAX_WORKERS, not on the
    command line, so a core-count change does not rerun the stage."""
    import os
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">a\nMKV\n>b\nMKV\n")
    monkeypatch.setenv("NANOPLM_MMSEQS", str(_fake_mmseqs(tmp_path)))
    if env is None:
        monkeypatch.delenv("NANOPLM_MAX_WORKERS", raising=False)
    else:
        monkeypatch.setenv("NANOPLM_MAX_WORKERS", env)
    r = CliRunner().invoke(data, ["cluster-members", "--source", "mmseqs", "--fasta", str(fasta),
                                  "--tmp-dir", str(tmp_path / "mm"), "-o",
                                  str(tmp_path / "m.tsv")])
    assert r.exit_code == 0, r.output
    argv = (tmp_path / "argv").read_text().split()
    got = argv[argv.index("--threads") + 1]
    assert got == (expect or str(max(1, len(os.sched_getaffinity(0)))))
