"""members.tsv from a representative<TAB>member file or from CL: tags in the headers."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from nanoplm.data.preparation import cluster_import as ci
from nanoplm.data.preparation.cluster_table import read_members_table

FASTA = ">a desc\nMKV\n>b\nMKL\n>c\nMMM\n>d\nMMA\n>e\nMMC\n"
TSV = "c\tc\nc\ta\nb\tb\nb\td\ne\te\n"
# MMseqs2's order is not FASTA order; the table puts clusters in the representative's FASTA
# order and members in theirs, representative first
TABLE = "b\tb\nb\td\nc\tc\nc\ta\ne\te\n"


def _files(tmp_path, fasta=FASTA, tsv=TSV, name="clusters.tsv"):
    fa, tab = tmp_path / "m.fasta", tmp_path / name
    fa.write_text(fasta)
    data = tsv.encode()
    if name.endswith(".gz"):
        data = gzip.compress(data)
    elif name.endswith(".zst"):
        import zstandard
        data = zstandard.ZstdCompressor().compress(data)
    tab.write_bytes(data)
    return fa, tab


@pytest.mark.parametrize("name", ["clusters.tsv", "clusters.tsv.gz", "clusters.tsv.zst"])
def test_tsv_to_members_table(tmp_path, name):
    fa, tab = _files(tmp_path, name=name)
    out = tmp_path / "cluster_index/members.tsv"
    stats = ci.import_members_table("tsv", out, fasta=fa, tsv=str(tab))
    assert out.read_text() == TABLE
    assert stats == {"clusters": 3, "members": 5, "singletons": 1, "source": "tsv"}
    assert json.loads(out.with_suffix(".json").read_text()) == stats
    t = read_members_table(out)                     # cluster_index reads it as it is
    assert list(t.sizes) == [2, 2, 1] and t.is_rep.sum() == 3


def test_tsv_line_order_does_not_change_the_table(tmp_path):
    lines = TSV.splitlines()
    fa, tab = _files(tmp_path, tsv="\r\n".join(reversed(lines)) + "\r\n\n")
    out = tmp_path / "members.tsv"
    ci.import_members_table("tsv", out, fasta=fa, tsv=str(tab))
    assert out.read_text() == TABLE


@pytest.mark.parametrize("tsv, match", [
    ("c\tc\nc\ta\nb\tb\n" + "e\te\n",
     r"1 of the 5 records of .*m.fasta are not in .*clusters.tsv \(e.g. d\). The cluster file "
     r"must list every sequence"),
    (TSV + "b\tz\ny\ty\n",
     r"2 ids in .*clusters.tsv are not records of .*m.fasta \(e.g. y, z; the FASTA's ids look "
     r"like a, b, c\). The cluster file must use FASTA ids.*MMseqs2 writes the accession"),
    (TSV + "b\ta\n",
     r"1 sequences are listed as a member more than once in .*clusters.tsv \(e.g. a\)"),
    ("c\tc\nc\ta\nb\tb\nb\td\ne\tb\n" + "e\te\n",
     r"listed as a member more than once .*\(e.g. b\)"),
    ("c\tc\nc\ta\nd\tb\nb\td\ne\te\n",
     r"2 representatives in .*clusters.tsv have no line of their own \(representative<TAB>"
     r"itself; e.g. b, d\)"),
    ("c\tc\nc\ta\tx\n", r"clusters.tsv: line 2 is not representative<TAB>member: 'c\\ta\\tx'"),
    ("c\tc\nc a\n", r"line 2 is not representative<TAB>member"),
    ("c\tc\n\tb\n", r"line 2 is not representative<TAB>member"),
    ("c\tc\nc\t\n", r"line 2 is not representative<TAB>member"),
    ("\n\n", r"clusters.tsv is empty"),
])
def test_tsv_errors(tmp_path, tsv, match):
    fa, tab = _files(tmp_path, tsv=tsv)
    with pytest.raises(ci.ClusterImportError, match=match):
        ci.import_members_table("tsv", tmp_path / "members.tsv", fasta=fa, tsv=str(tab))
    assert not (tmp_path / "members.tsv").exists()


def test_tsv_needs_unique_fasta_ids(tmp_path):
    fa, tab = _files(tmp_path, fasta=FASTA + ">a again\nMKV\n")
    with pytest.raises(ci.ClusterImportError, match=r"1 record ids appear more than once "
                                                    r"\(e.g. a\); every FASTA id must be unique"):
        ci.import_members_table("tsv", tmp_path / "members.tsv", fasta=fa, tsv=str(tab))


# Logan's layout: grouped by cluster, the representative (id == its CL: value) not
# always first
HEADERS = (">x2\tCL:x1\nMKV\n>x1\tCL:x1\nMKL\n>x3 CL:x1 more\nMMM\n"
           ">y1\tCL:y1\nMMA\n>z9\tCL:z1\nMMC\n>z1\tCL:z1\nMMD\n")


def test_headers_to_members_table_moves_the_representative_first(tmp_path):
    fa = tmp_path / "logan.fasta"
    fa.write_text(HEADERS)
    out = tmp_path / "members.tsv"
    stats = ci.import_members_table("headers", out, fasta=fa)
    assert out.read_text() == ("x1\tx1\nx1\tx2\nx1\tx3\ny1\ty1\nz1\tz1\nz1\tz9\n")
    assert stats == {"clusters": 3, "members": 6, "singletons": 1, "source": "headers"}
    assert read_members_table(out).is_rep.sum() == 3


@pytest.mark.parametrize("fasta, match", [
    (HEADERS + ">w1\nMKV\n>w2 CL:\nMKV\n",
     r"2 of the 8 records of .*logan.fasta have no CL: tag \(e.g. w1, w2\). With clusters "
     r"from the headers, every header names its cluster after the id"),
    (HEADERS + ">q1\tCL:q0\nMKV\n>q2\tCL:q0\nMKV\n>r1\tCL:r0\nMKV\n",
     r"2 clusters of .*logan.fasta have no representative, no record whose id is the cluster "
     r"id \(e.g. CL:q0, CL:r0; 3 records name them\)"),
    (">a\tCL:a\nM\n>b\tCL:a\nM\n>c\tCL:b\nM\n",
     r"1 representatives of .*logan.fasta name another cluster in their own CL: tag "
     r"\(e.g. b\)"),
    (">aCL:a\nM\n", r"have no CL: tag"),                 # a token of its own
    (HEADERS + ">x1\tCL:x1\nM\n", r"record ids appear more than once \(e.g. x1\)"),
])
def test_headers_errors(tmp_path, fasta, match):
    fa = tmp_path / "logan.fasta"
    fa.write_text(fasta)
    with pytest.raises(ci.ClusterImportError, match=match):
        ci.import_members_table("headers", tmp_path / "members.tsv", fasta=fa)


def test_the_cluster_members_command_prints_the_error_not_a_traceback(tmp_path):
    from click.testing import CliRunner
    from nanoplm.cli import cli
    fa = tmp_path / "logan.fasta"
    fa.write_text(">w1\nMKV\n")
    r = CliRunner().invoke(cli, ["data", "cluster-members", "--source", "headers",
                                 "--fasta", str(fa), "-o", str(tmp_path / "m.tsv")])
    assert r.exit_code == 1 and "Traceback" not in r.output
    assert "Error: 1 of the 1 records of" in r.output and "have no CL: tag" in r.output
    fa2, tab = _files(tmp_path)
    r = CliRunner().invoke(cli, ["data", "cluster-members", "--source", "tsv", "--fasta",
                                 str(fa2), "--tsv", str(tab), "-o", str(tmp_path / "m.tsv")])
    assert r.exit_code == 0, r.output
    assert "3 clusters, 5 sequences" in r.output
    assert Path(tmp_path / "m.tsv").read_text() == TABLE


@pytest.mark.parametrize("name,zstd_tool", [("clusters.tsv.gz", False),
                                            ("clusters.tsv.zst", False),
                                            ("clusters.tsv.zst", True)])
def test_a_cut_off_compressed_tsv_names_the_file(tmp_path, monkeypatch, name, zstd_tool):
    fa, tab = _files(tmp_path, name=name)
    tab.write_bytes(tab.read_bytes()[:-12])
    if zstd_tool:
        import shutil
        import sys
        if shutil.which("zstd") is None:
            pytest.skip("no zstd tool")
        monkeypatch.setitem(sys.modules, "zstandard", None)  # the import fails
    with pytest.raises(ci.ClusterImportError, match=f"{name} is incomplete .* download it again"):
        ci.import_members_table("tsv", tmp_path / "members.tsv", fasta=fa, tsv=tab)
