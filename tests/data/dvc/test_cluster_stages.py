"""dvc.yaml for cluster_subsampling, and an end-to-end `nanoplm data from-yaml` run."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nanoplm.cli._dvc_yaml_gen import generate_dvc_yaml
from nanoplm.cli.templates import load_template
from nanoplm.data.preparation import cluster_config as cc
from nanoplm.data.preparation.cluster_table import find_mmseqs

REPO = Path(__file__).resolve().parents[3]
AGENT_MMSEQS = REPO / "agent_space/cluster-sampling-port/mmseqs-dl/mmseqs/bin/mmseqs"
MMSEQS = find_mmseqs(str(AGENT_MMSEQS) if AGENT_MMSEQS.exists() else None)


OUT = "output/data/pretrain_data"
BLOCK = "sources.src.cluster_subsampling"


def _template():
    return yaml.safe_load(load_template("data_pretrain"))


def _stages(cfg):
    return yaml.safe_load(generate_dvc_yaml(cfg, "pretrain", "data.yaml"))["stages"]


def _source(fasta, block=None, **params):
    """The template with one source "src"; block=None leaves cluster subsampling off."""
    cfg = _template()
    cfg["sources"] = {"src": {"fasta": fasta}}
    if block is not None:
        cfg["sources"]["src"]["cluster_subsampling"] = {"members": None, **block}
    cfg["data_params"].update(params)
    return cfg


def test_the_template_has_no_cluster_stages():
    st = _stages(_template())
    assert list(st) == ["get_file", "extract", "check", "shuffle", "filter", "split",
                        "pre-tokenize", "write_manifest"]


def test_the_cluster_block_never_refetches_the_source():
    """Adding or removing the block keeps get_file / extract / check as they are when the
    members are the source fasta, so DVC does not download or extract it again."""
    u90 = cc.UNIPROT_FASTA["uniref90"]
    plain, clustered = _stages(_source(u90)), _stages(_source(u90, {}))
    for name in ("get_file", "extract", "check"):
        assert plain[name] == clustered[name], name
    assert plain["get_file"] == {
        "cmd": f'nanoplm data get-file --url "{u90}" -o {OUT}/downloads/uniref90.fasta.gz',
        "deps": [plain["get_file"]["deps"][0]], "outs": [f"{OUT}/downloads/uniref90.fasta.gz"]}
    # the shortcut and its URL are the same download
    assert _stages(_source("uniref90"))["get_file"] == plain["get_file"]


def test_mmseqs2_on_uniref_data_fetches_neither_the_xml_nor_the_url():
    st = _stages(_source("uniref50", {"members": "uniref90", "clustering": "mmseqs2"}))
    assert list(st) == ["get_members_fasta", "extract_members", "check", "filter",
                        "cluster_pool", "cluster_members", "cluster_index", "write_manifest"]
    assert "uniref50.fasta" not in yaml.safe_dump(st) and "xml" not in yaml.safe_dump(st)
    assert "--source mmseqs" in st["cluster_members"]["cmd"]
    assert f"{BLOCK}.clustering" in st["cluster_members"]["params"][0]["data.yaml"]


def test_a_block_without_clustering_lists_no_clustering_param():
    st = _stages(_source("uniref50", {"members": "uniref90"}))
    assert "params" not in st["cluster_members"] and "get_cluster_xml" in st


def test_uniref50_url_uniref90_members_stages():
    st = _stages(_source("uniref50", {"members": "uniref90", "clustering": "auto"}))
    assert list(st) == ["get_members_fasta", "extract_members", "check", "filter",
                        "cluster_pool", "get_cluster_xml", "cluster_members", "cluster_index",
                        "write_manifest"]
    assert cc.UNIPROT_FASTA["uniref90"] in st["get_members_fasta"]["cmd"]
    assert "params" not in st["get_members_fasta"]          # depends only on the URL
    assert st["get_cluster_xml"]["outs"] == [
        {f"{OUT}/downloads/uniref50.xml.gz": {"persist": True, "cache": False}}]
    # identity / coverage are ignored here, and clustering: auto is the default
    assert "params" not in st["cluster_members"]
    assert "--source uniref_xml --level uniref90" in st["cluster_members"]["cmd"]
    # a length-filter change must not rerun the XML scan
    assert "filtered.fasta" not in str(st["cluster_members"])
    assert f"{OUT}/src/uniref90.fasta" in st["cluster_members"]["deps"]
    assert st["cluster_members"]["outs"] == [
        f"{OUT}/cluster_index/members.tsv", f"{OUT}/cluster_index/members.json"]
    assert f"{OUT}/cluster_index/index" in st["cluster_index"]["outs"]
    assert "--cluster-index" in st["write_manifest"]["cmd"]
    # named in the release-mismatch error; members.json no longer records it
    assert st["cluster_index"]["cmd"].endswith(
        f" --cluster-xml {OUT}/downloads/uniref50.xml.gz")
    assert "--cluster-xml" not in _stages(_source("uniref50", {}))["cluster_index"]["cmd"]


def test_writing_a_cluster_default_out_changes_nothing(tmp_path):
    """identity 0.5, coverage 0.8 and clustering auto are what a left-out key means, so
    writing them gives the same dvc.yaml (no new DVC param, no rerun)."""
    own = tmp_path / "mine.fasta"
    own.write_text(">a\nMKV\n")
    for fasta, block in ((str(own), {}), ("uniref50", {"members": "uniref90"})):
        written = {**block, "clustering": "auto", "identity": 0.5, "coverage": 0.8}
        assert _stages(_source(fasta, written)) == _stages(_source(fasta, block))
    changed = _stages(_source(str(own), {"identity": 0.6}))["cluster_members"]
    assert "--identity ${" + BLOCK + ".identity} --coverage 0.8" in changed["cmd"]
    assert changed["params"] == [{"data.yaml": [f"{BLOCK}.identity"]}]


@pytest.mark.parametrize("block", [{"members": "uniref90"}, {"clustering": "mmseqs2"}])
def test_each_cluster_stage_depends_on_the_modules_it_runs(block):
    """An edit to cluster_pipeline.py reruns the pool and the index, never the members."""
    st = _stages(_source("uniref50", block))

    def mods(name):
        return {Path(d).name for d in st[name]["deps"] if d.endswith(".py")}

    assert {"cluster_pipeline.py", "cluster_index.py"} <= mods("cluster_pool")
    assert {"cluster_pipeline.py", "cluster_index.py", "cluster_table.py",
            "uniref_hierarchy.py"} <= mods("cluster_index")
    assert "cluster_pipeline.py" not in mods("cluster_members")
    assert "cluster_index.py" not in mods("cluster_members")


def test_an_override_is_referenced_where_it_is_set():
    cfg = _source("uniref50", {"members": "uniref90"})
    cfg["sources"]["src"]["max_seq_len"] = 1024
    st = _stages(cfg)
    assert "--max-seq-len ${sources.src.max_seq_len}" in st["cluster_pool"]["cmd"]
    assert "sources.src.max_seq_len" in st["cluster_pool"]["params"][0]["data.yaml"]
    assert "--val-ratio ${data_params.val_ratio}" in st["cluster_index"]["cmd"]


def test_mmseqs_params_only_reach_cluster_members(tmp_path):
    own = tmp_path / "mine.fasta"
    own.write_text(">a\nMKV\n")
    st = _stages(_source(str(own), {"clustering": "mmseqs2", "identity": 0.6, "coverage": 0.9}))
    assert "get_cluster_xml" not in st and "get_members_fasta" not in st
    # a local file is read in place: no get_file, and check reads it directly
    assert "get_file" not in st and st["check"]["deps"][0] == str(own)
    assert f"--identity ${{{BLOCK}.identity}}" in st["cluster_members"]["cmd"]
    assert st["cluster_members"]["params"] == [
        {"data.yaml": [f"{BLOCK}.clustering", f"{BLOCK}.identity", f"{BLOCK}.coverage"]}]
    # the core count comes from $NANOPLM_MAX_WORKERS, so changing it reruns nothing
    assert "max_workers" not in yaml.safe_dump(st)
    assert "--threads" not in st["cluster_members"]["cmd"]
    for name, stage in st.items():
        if name != "cluster_members":
            flat = str(stage.get("params", "")) + str(stage.get("deps", ""))
            assert "cluster_subsampling" not in flat, name
    assert "members.tsv" not in str(st["cluster_pool"]["deps"])
    assert not any("cluster_pipeline.py" in d for d in st["cluster_members"]["deps"])
    # a key left out is its default, in the command itself
    st = _stages(_source(str(own), {}))
    assert "--identity 0.5 --coverage 0.8" in st["cluster_members"]["cmd"]
    assert "params" not in st["cluster_members"]


# ------------------------------------------------------------------ end to end


def _run(args, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)


def _project(tmp_path, block, fasta, **params):
    """data.yaml with one source "src" and small data_params; block=None = no cluster block."""
    cfg = _source(fasta, block, **{**dict(max_seqs=-1, min_seq_len=20, max_seq_len=128,
                                          val_ratio=0.1, samples_per_shard=101,
                                          max_workers=1), **params})
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}",
               CUDA_VISIBLE_DEVICES="")
    if MMSEQS:
        env["NANOPLM_MMSEQS"] = MMSEQS
    return env


def _from_yaml(tmp_path, env, cwd=None, config="data.yaml"):
    r = _run([str(Path(sys.executable).parent / "nanoplm"), "data", "from-yaml", config],
             cwd or tmp_path, env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def _status(tmp_path, env):
    r = _run([sys.executable, "-m", "dvc", "status", "--json"], tmp_path, env)
    assert r.returncode == 0, r.stderr
    import json
    return set(json.loads(r.stdout or "{}"))


def _edit(tmp_path, fn):
    """Edit data.yaml and regenerate dvc.yaml, as `nanoplm data from-yaml` does before it
    runs DVC (a value at its default is in the command, not a ${...} reference)."""
    cfg = yaml.safe_load((tmp_path / "data.yaml").read_text())
    fn(cfg)
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (tmp_path / "dvc.yaml").write_text(
        generate_dvc_yaml(cfg, "pretrain", "data.yaml", base_dir=tmp_path))


def _assert_source_line_first(r, text):
    """The pre-run table names the cluster source on the source's row, before DVC runs."""
    row = next(ln for ln in r.stdout.splitlines() if ln.startswith("src "))
    assert row.endswith(text), row
    assert "Running stage" not in r.stdout[:r.stdout.index(row)]


@pytest.mark.skipif(MMSEQS is None, reason="mmseqs not installed")
def test_from_yaml_end_to_end_with_mmseqs(tmp_path):
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=120)
    # not a UniRef file name, and no UniRef headers: an "own FASTA"
    own = tmp_path / "src/own.fasta"
    own.write_text(rel.fasta90.read_text().replace(">UniRef90_", ">seq_"))
    env = _project(tmp_path, {"clustering": "auto", "identity": 0.5, "coverage": 0.8}, str(own))
    r = _from_yaml(tmp_path, env)
    _assert_source_line_first(r, f"MMseqs2 linclust, identity 0.5, coverage 0.8 "
                                 f"(members: {own})")
    assert f"  Cluster source:  MMseqs2 linclust, identity 0.5" in r.stdout   # summary
    out = tmp_path / OUT
    assert (out / "cluster_index/members.tsv").exists()
    assert (out / ".data_manifest").exists()
    assert not (out / "cluster_index/.mmseqs_tmp").exists()
    assert _status(tmp_path, env) == set()
    # the local FASTA is read in place: DVC neither links nor protects it
    assert own.stat().st_nlink == 1 and os.access(own, os.W_OK)
    assert (tmp_path / ".dvc/config").read_text().count("hardlink,symlink") == 1
    # a core-count change reruns nothing
    _edit(tmp_path, lambda c: c["data_params"].update(max_workers=3))
    assert _status(tmp_path, env) == set()
    # identity changes only cluster_members (and, once it reruns, what follows it)
    _edit(tmp_path, lambda c: c["sources"]["src"]["cluster_subsampling"].update(identity=0.9))
    assert _status(tmp_path, env) == {"cluster_members"}
    # so does clustering, even when the source it resolves to stays MMseqs2
    _edit(tmp_path, lambda c: c["sources"]["src"]["cluster_subsampling"].update(
        identity=0.5, clustering="mmseqs2"))
    assert _status(tmp_path, env) == {"cluster_members"}


def test_from_yaml_end_to_end_with_uniprot_clusters(tmp_path):
    from nanoplm.data.datasets import ClusterSampledDataset, open_train_dataset
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=120)
    downloads = tmp_path / OUT / "downloads"
    downloads.mkdir(parents=True)
    shutil.copy(rel.xml, downloads / "uniref50.xml.gz")       # reused, never downloaded
    env = _project(tmp_path, {"members": None}, str(rel.fasta90))
    r = _from_yaml(tmp_path, env)
    assert "Reusing" in r.stdout + r.stderr
    _assert_source_line_first(r, "UniProt uniref50.xml.gz (members: UniRef90)")
    assert "  Cluster source:  UniProt uniref50.xml.gz (members: UniRef90)" in r.stdout
    out = tmp_path / OUT
    m = read_manifest(out)
    assert m.cluster_level == "uniref90" and m.cluster_index_dir == "cluster_index/index"
    ds = open_train_dataset(out, m, seed=3)
    assert isinstance(ds, ClusterSampledDataset) and len(ds) == m.train_sequences
    # removing the block goes back to the plain pipeline over the same local FASTA
    _edit(tmp_path, lambda c: c["sources"]["src"].pop("cluster_subsampling"))
    _from_yaml(tmp_path, env)
    m = read_manifest(out)
    assert m.cluster_index_dir is None and m.train_sequences > 0


@pytest.mark.skipif(MMSEQS is None, reason="mmseqs not installed")
@pytest.mark.parametrize("layout", ["uniref90 url", "uniref50 url + uniref90 members"])
def test_from_yaml_mmseqs2_on_uniref_data_never_fetches_the_xml(tmp_path, layout):
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=60)
    if layout == "uniref90 url":
        block, fasta = {}, str(rel.fasta90)
    else:
        u50 = tmp_path / "src/reps.fasta"          # UniRef50 by its headers only
        u50.write_text(rel.fasta90.read_text().replace(">UniRef90_", ">UniRef50_"))
        block, fasta = {"members": str(rel.fasta90)}, str(u50)
    env = _project(tmp_path, {"clustering": "mmseqs2", **block}, fasta)
    env.update(HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9",
               https_proxy="http://127.0.0.1:9", http_proxy="http://127.0.0.1:9")
    r = _from_yaml(tmp_path, env)
    _assert_source_line_first(r, f"MMseqs2 linclust, identity 0.5, coverage 0.8 "
                                 f"(members: {rel.fasta90})")
    dvc_yaml = (tmp_path / "dvc.yaml").read_text()
    assert "get_cluster_xml" not in dvc_yaml and "uniref50.xml" not in dvc_yaml
    assert not list((tmp_path / OUT).rglob("*.xml*"))
    if layout != "uniref90 url":
        assert "is not needed and is not downloaded" in r.stdout
        assert "reps.fasta" not in dvc_yaml               # the source FASTA is never read
    m = read_manifest(tmp_path / OUT)
    assert m.cluster_level == "mmseqs2" and m.seqs_num > 0


def test_from_yaml_from_another_directory_reads_relative_paths_there(tmp_path):
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=60)
    # UniRef90 only by its headers, given relative to data.yaml
    shutil.copy(rel.fasta90, tmp_path / "src/mine.fasta")
    downloads = tmp_path / OUT / "downloads"
    downloads.mkdir(parents=True)
    shutil.copy(rel.xml, downloads / "uniref50.xml.gz")
    env = _project(tmp_path, {}, "src/mine.fasta")
    env.pop("NANOPLM_MMSEQS", None)        # the UniProt route needs no MMseqs2
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _from_yaml(tmp_path, env, cwd=elsewhere, config="../data.yaml")
    m = read_manifest(tmp_path / OUT)
    assert m.cluster_level == "uniref90"
    assert not (elsewhere / "output/data").exists()


def test_cluster_mode_needs_a_valid_max_workers(tmp_path):
    import click
    from nanoplm.cli.data import _validate_cluster_block
    from nanoplm.data.preparation.data_config import parse_data_config
    src = parse_data_config(_source("https://example.org/mine.fasta.gz", {},
                                    max_workers=0)).source
    with pytest.raises(click.ClickException, match="data_params.max_workers must be -1"):
        _validate_cluster_block(src, tmp_path / "data.yaml")


def test_from_yaml_with_the_template_data_params(tmp_path):
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=80)
    downloads = tmp_path / OUT / "downloads"
    downloads.mkdir(parents=True)
    shutil.copy(rel.xml, downloads / "uniref50.xml.gz")
    cfg = _source(str(rel.fasta90), {})                 # data_params exactly as shipped
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}",
               CUDA_VISIBLE_DEVICES="")
    r = _from_yaml(tmp_path, env)
    err = r.stdout + r.stderr
    assert "data_params.max_seqs=20000 trains on at most 20,000 clusters" in err
    assert "data_params.val_ratio=0.1 holds out 10.0% of the clusters" in err
    m = read_manifest(tmp_path / OUT)
    assert m.cluster_level == "uniref90" and m.samples_per_shard == 2000
    assert m.val_sequences == round(m.seqs_num * 0.1)


def test_summary_counts_clusters_and_names_the_members(tmp_path, capsys):
    from nanoplm.cli.data import _print_summary
    from nanoplm.data.preparation.data_config import parse_data_config
    raw = _source("uniref50", {"members": "uniref90"})
    out = tmp_path / OUT
    out.mkdir(parents=True)
    (out / ".data_manifest").write_text(yaml.safe_dump({
        "seqs_num": 120, "train_sequences": 108, "val_sequences": 12,
        "cluster_index_dir": "cluster_index/index", "cluster_level": "uniref90",
        "pool_sequences": 400}))
    _print_summary(tmp_path, raw, "pretrain", tmp_path / "data.yaml",
                   parse_data_config(raw, tmp_path))
    text = capsys.readouterr().out
    assert "uniref50.fasta" not in text and "Total sequences" not in text
    assert f"Members:         {cc.UNIPROT_FASTA['uniref90']}" in text
    assert "Cluster source:  UniProt uniref50.xml.gz (members: UniRef90)" in text
    assert "Clusters:        120 (uniref90)" in text and "Pool sequences:  400" in text


def test_from_yaml_says_how_to_install_mmseqs(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    from nanoplm.data.preparation import cluster_table as ct
    (tmp_path / "src").mkdir()
    (tmp_path / "src/own.fasta").write_text(">p1\nMKVLAAGIVGLLLAAGSAQAEELK\n")
    _project(tmp_path, {}, "src/own.fasta")
    monkeypatch.setattr(ct, "find_mmseqs", lambda explicit=None: None)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("DVC must not run"))
    r = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
    assert r.exit_code != 0
    assert "clustered with MMseqs2, but `mmseqs` was not found" in r.output
    assert "nanoplm deps install --mmseqs-only" in r.output
    assert not (tmp_path / "dvc.yaml").exists()


@pytest.mark.parametrize("block, message", [
    ({"clustering": "uniprot"}, "data.yaml: sources.src.cluster_subsampling.clustering: "
     "uniprot: UniProt clusters need UniRef data (fasta UniRef50 + members uniref90/uniref100, "
     "or a UniRef90/100 fasta)"),
    ({"clustering": "linclust"}, "data.yaml: sources.src.cluster_subsampling.clustering must "
     "be one of auto, uniprot, mmseqs2, got 'linclust'"),
])
def test_from_yaml_refuses_a_bad_clustering_before_dvc(tmp_path, monkeypatch, block, message):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    (tmp_path / "src").mkdir()
    (tmp_path / "src/own.fasta").write_text(">p1\nMKVLAAGIVGLLLAAGSAQAEELK\n")
    _project(tmp_path, block, "src/own.fasta")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("DVC must not run"))
    r = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
    assert r.exit_code != 0 and message in " ".join(r.output.split())
    assert not (tmp_path / "dvc.yaml").exists()


def test_forced_mmseqs2_on_uniref_data_needs_the_binary(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    from nanoplm.data.preparation import cluster_table as ct
    (tmp_path / "src").mkdir()
    (tmp_path / "src/u90.fasta").write_text(">UniRef90_A x\nMKVLAAGIVGLLLAAGSAQAEELK\n")
    _project(tmp_path, {"clustering": "mmseqs2"}, "src/u90.fasta")
    monkeypatch.setattr(ct, "find_mmseqs", lambda explicit=None: None)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("DVC must not run"))
    r = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
    assert r.exit_code != 0 and "clustered with MMseqs2, but `mmseqs` was not found" in r.output
