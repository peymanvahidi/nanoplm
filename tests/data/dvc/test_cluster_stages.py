"""dvc.yaml for cluster_subsampling, and an end-to-end `nanoplm data from-yaml` run."""
from __future__ import annotations

import copy
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


def _template():
    return yaml.safe_load(load_template("data_pretrain"))


def _stages(cfg):
    return yaml.safe_load(generate_dvc_yaml(cfg, "pretrain", "data.yaml"))["stages"]


def _enabled(**over):
    cfg = _template()
    cfg["cluster_subsampling"]["enabled"] = True
    for k, v in over.items():
        sec, key = k.split("__")
        cfg[sec][key] = v
    return cfg


def test_disabled_block_leaves_dvc_yaml_unchanged():
    cfg = _template()
    without = copy.deepcopy(cfg)
    del without["cluster_subsampling"]
    assert generate_dvc_yaml(cfg, "pretrain", "data.yaml") == \
        generate_dvc_yaml(without, "pretrain", "data.yaml")
    for clustering in cc.CLUSTERINGS:      # the key is not read while disabled
        cfg["cluster_subsampling"]["clustering"] = clustering
        assert generate_dvc_yaml(cfg, "pretrain", "data.yaml") == \
            generate_dvc_yaml(without, "pretrain", "data.yaml")


def test_mmseqs2_on_uniref_data_fetches_neither_the_xml_nor_the_url():
    st = _stages(_enabled(data_dirs__url=cc.UNIPROT_FASTA["uniref50"],
                          cluster_subsampling__members="uniref90",
                          cluster_subsampling__clustering="mmseqs2"))
    assert list(st) == ["get_members_fasta", "extract_members", "check", "filter",
                        "cluster_pool", "cluster_members", "cluster_index", "write_manifest"]
    assert "uniref50" not in yaml.safe_dump(st)
    assert "--source mmseqs" in st["cluster_members"]["cmd"]
    assert "cluster_subsampling.identity" in st["cluster_members"]["params"][0]["data.yaml"]


def test_a_block_without_clustering_lists_no_clustering_param():
    cfg = _enabled(data_dirs__url=cc.UNIPROT_FASTA["uniref50"],
                   cluster_subsampling__members="uniref90")
    del cfg["cluster_subsampling"]["clustering"]
    st = _stages(cfg)
    assert "params" not in st["cluster_members"] and "get_cluster_xml" in st


def test_uniref50_url_uniref90_members_stages():
    st = _stages(_enabled(data_dirs__url=cc.UNIPROT_FASTA["uniref50"],
                          cluster_subsampling__members="uniref90"))
    assert list(st) == ["get_members_fasta", "extract_members", "check", "filter",
                        "cluster_pool", "get_cluster_xml", "cluster_members", "cluster_index",
                        "write_manifest"]
    assert cc.UNIPROT_FASTA["uniref90"] in st["get_members_fasta"]["cmd"]
    assert st["get_cluster_xml"]["outs"] == [
        {"output/data/raw/uniref50.xml.gz": {"persist": True, "cache": False}}]
    # identity / coverage are ignored here, so they must not trigger a rerun
    assert st["cluster_members"]["params"] == [{"data.yaml": ["cluster_subsampling.clustering"]}]
    assert "--source uniref_xml --level uniref90" in st["cluster_members"]["cmd"]
    # a length-filter change must not rerun the XML scan
    assert "filtered_fasta" not in str(st["cluster_members"])
    assert "output/data/raw/uniref90.fasta" in st["cluster_members"]["deps"]
    assert st["cluster_members"]["outs"] == [
        "output/data/pretrain_data/cluster_index/members.tsv",
        "output/data/pretrain_data/cluster_index/members.json"]
    assert "output/data/pretrain_data/cluster_index/index" in st["cluster_index"]["outs"]
    assert "--cluster-index" in st["write_manifest"]["cmd"]


def test_mmseqs_params_only_reach_cluster_members(tmp_path):
    own = tmp_path / "mine.fasta"
    own.write_text(">a\nMKV\n")
    st = _stages(_enabled(data_dirs__url=None, data_dirs__local_fasta=str(own)))
    assert "get_cluster_xml" not in st and "get_members_fasta" not in st
    assert st["get_file"]["outs"] == ["output/data/raw/sequences.fasta"]
    assert st["cluster_members"]["params"] == [
        {"data.yaml": ["cluster_subsampling.clustering", "cluster_subsampling.identity",
                       "cluster_subsampling.coverage"]}]
    # the core count comes from $NANOPLM_MAX_WORKERS, so changing it reruns nothing
    assert "max_workers" not in yaml.safe_dump(st)
    assert "--threads" not in st["cluster_members"]["cmd"]
    for name, stage in st.items():
        if name != "cluster_members":
            flat = str(stage.get("params", "")) + str(stage.get("deps", ""))
            assert "cluster_subsampling" not in flat, name
    assert "members.tsv" not in str(st["cluster_pool"]["deps"])
    assert not any("cluster_pipeline.py" in d for d in st["cluster_members"]["deps"])


# ------------------------------------------------------------------ end to end


def _run(args, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)


def _project(tmp_path, block, data_dirs):
    cfg = _template()
    cfg["data_dirs"].update(data_dirs)
    cfg["data_params"].update(seqs_num=-1, min_seq_len=20, max_seq_len=128, val_ratio=0.1,
                              samples_per_shard=101, max_workers=1)
    cfg["cluster_subsampling"].update(block)
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


@pytest.mark.skipif(MMSEQS is None, reason="mmseqs not installed")
def _assert_source_line_first(r, line):
    """The source line is printed once, before DVC runs any stage."""
    text = r.stdout
    assert text.count(line) == 1, text[-3000:]
    assert "Running stage" not in text[:text.index(line)]


def test_from_yaml_end_to_end_with_mmseqs(tmp_path):
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=120)
    # not a UniRef file name, and no UniRef headers: an "own FASTA"
    own = tmp_path / "src/own.fasta"
    own.write_text(rel.fasta90.read_text().replace(">UniRef90_", ">seq_"))
    env = _project(tmp_path, {"enabled": True},
                   {"url": None, "local_fasta": str(own)})
    r = _from_yaml(tmp_path, env)
    _assert_source_line_first(r, f"Cluster source: MMseqs2 linclust, identity 0.5, "
                                 f"coverage 0.8 (members: {own})")
    assert f"  Cluster source:  MMseqs2 linclust, identity 0.5" in r.stdout   # summary
    out = tmp_path / "output/data/pretrain_data"
    assert (out / "cluster_index/members.tsv").exists()
    assert (out / ".data_manifest").exists()
    assert not (out / "cluster_index/.mmseqs_tmp").exists()
    assert _status(tmp_path, env) == set()
    # identity changes only cluster_members (and, once it reruns, what follows it)
    cfg = yaml.safe_load((tmp_path / "data.yaml").read_text())
    cfg["cluster_subsampling"]["identity"] = 0.9
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert _status(tmp_path, env) == {"cluster_members"}
    # so does clustering, even when the source it resolves to stays MMseqs2
    cfg["cluster_subsampling"].update(identity=0.5, clustering="mmseqs2")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert _status(tmp_path, env) == {"cluster_members"}


def test_from_yaml_end_to_end_with_uniprot_clusters(tmp_path):
    from nanoplm.data.datasets import ClusterSampledDataset, open_train_dataset
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=120)
    raw = tmp_path / "output/data/raw"
    raw.mkdir(parents=True)
    shutil.copy(rel.xml, raw / "uniref50.xml.gz")       # reused, never downloaded
    env = _project(tmp_path, {"enabled": True, "members": None},
                   {"url": None, "local_fasta": str(rel.fasta90)})
    r = _from_yaml(tmp_path, env)
    assert "Reusing" in r.stdout + r.stderr
    _assert_source_line_first(r, "Cluster source: UniProt uniref50.xml.gz (members: UniRef90)")
    assert "  Cluster source:  UniProt uniref50.xml.gz (members: UniRef90)" in r.stdout
    out = tmp_path / "output/data/pretrain_data"
    m = read_manifest(out)
    assert m.cluster_level == "uniref90" and m.cluster_index_dir == "cluster_index/index"
    ds = open_train_dataset(out, m, seed=3)
    assert isinstance(ds, ClusterSampledDataset) and len(ds) == m.train_sequences


@pytest.mark.skipif(MMSEQS is None, reason="mmseqs not installed")
@pytest.mark.parametrize("layout", ["uniref90 url", "uniref50 url + uniref90 members"])
def test_from_yaml_mmseqs2_on_uniref_data_never_fetches_the_xml(tmp_path, layout):
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=60)
    if layout == "uniref90 url":
        block, dirs = {}, {"url": None, "local_fasta": str(rel.fasta90)}
    else:
        u50 = tmp_path / "src/reps.fasta"          # UniRef50 by its headers only
        u50.write_text(rel.fasta90.read_text().replace(">UniRef90_", ">UniRef50_"))
        block, dirs = {"members": str(rel.fasta90)}, {"url": None, "local_fasta": str(u50)}
    env = _project(tmp_path, {"enabled": True, "clustering": "mmseqs2", **block}, dirs)
    env.update(HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9",
               https_proxy="http://127.0.0.1:9", http_proxy="http://127.0.0.1:9")
    r = _from_yaml(tmp_path, env)
    _assert_source_line_first(r, f"Cluster source: MMseqs2 linclust, identity 0.5, "
                                 f"coverage 0.8 (members: {rel.fasta90})")
    dvc_yaml = (tmp_path / "dvc.yaml").read_text()
    assert "get_cluster_xml" not in dvc_yaml and "uniref50.xml" not in dvc_yaml
    raw = tmp_path / "output/data/raw"
    assert not list(raw.glob("*.xml*"))
    if layout != "uniref90 url":
        assert "is not needed and is not downloaded" in r.stdout
        assert not (raw / "sequences.fasta").exists()       # url's FASTA never linked
        assert (raw / "uniref90.fasta").exists()
    m = read_manifest(tmp_path / "output/data/pretrain_data")
    assert m.cluster_level == "mmseqs2" and m.seqs_num > 0


def test_from_yaml_from_another_directory_reads_relative_paths_there(tmp_path):
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=60)
    # UniRef90 only by its headers, given relative to data.yaml
    shutil.copy(rel.fasta90, tmp_path / "src/mine.fasta")
    raw = tmp_path / "output/data/raw"
    raw.mkdir(parents=True)
    shutil.copy(rel.xml, raw / "uniref50.xml.gz")
    env = _project(tmp_path, {"enabled": True}, {"url": None, "local_fasta": "src/mine.fasta"})
    env.pop("NANOPLM_MMSEQS", None)        # the UniProt route needs no MMseqs2
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _from_yaml(tmp_path, env, cwd=elsewhere, config="../data.yaml")
    m = read_manifest(tmp_path / "output/data/pretrain_data")
    assert m.cluster_level == "uniref90"
    assert not (elsewhere / "output/data").exists()


def test_cluster_mode_needs_a_valid_max_workers(tmp_path):
    import click
    from nanoplm.cli.data import _validate_cluster_block
    cfg = _enabled(data_dirs__url="https://example.org/mine.fasta.gz",
                   data_params__max_workers=0)
    with pytest.raises(click.ClickException, match="max_workers must be -1"):
        _validate_cluster_block(cfg, tmp_path / "data.yaml")


def test_from_yaml_with_the_template_data_params(tmp_path):
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=80)
    raw = tmp_path / "output/data/raw"
    raw.mkdir(parents=True)
    shutil.copy(rel.xml, raw / "uniref50.xml.gz")
    cfg = _template()                                   # data_params exactly as shipped
    cfg["data_dirs"].update(url=None, local_fasta=str(rel.fasta90))
    cfg["cluster_subsampling"]["enabled"] = True
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}",
               CUDA_VISIBLE_DEVICES="")
    r = _from_yaml(tmp_path, env)
    err = r.stdout + r.stderr
    assert "seqs_num=20000 trains on at most 20,000 clusters" in err
    assert "val_ratio=0.1 holds out 10.0% of the clusters" in err
    m = read_manifest(tmp_path / "output/data/pretrain_data")
    assert m.cluster_level == "uniref90" and m.samples_per_shard == 2000
    assert m.val_sequences == round(m.seqs_num * 0.1)


def test_summary_counts_clusters_and_names_the_members(tmp_path, capsys):
    from nanoplm.cli.data import _print_summary
    cfg = _enabled(data_dirs__url=cc.UNIPROT_FASTA["uniref50"],
                   cluster_subsampling__members="uniref90")
    out = tmp_path / "output/data/pretrain_data"
    out.mkdir(parents=True)
    (out / ".data_manifest").write_text(yaml.safe_dump({
        "seqs_num": 120, "train_sequences": 108, "val_sequences": 12,
        "cluster_index_dir": "cluster_index/index", "cluster_level": "uniref90",
        "pool_sequences": 400}))
    _print_summary(tmp_path, cfg, "pretrain", tmp_path / "data.yaml")
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
    _project(tmp_path, {"enabled": True}, {"url": None, "local_fasta": "src/own.fasta"})
    monkeypatch.setattr(ct, "find_mmseqs", lambda explicit=None: None)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("DVC must not run"))
    r = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
    assert r.exit_code != 0
    assert "clustered with MMseqs2, but `mmseqs` was not found" in r.output
    assert "nanoplm deps install --mmseqs-only" in r.output
    assert not (tmp_path / "dvc.yaml").exists()


@pytest.mark.parametrize("block, message", [
    ({"clustering": "uniprot"}, "data.yaml: cluster_subsampling.clustering: uniprot: UniProt "
     "clusters need UniRef data (url UniRef50 + members uniref90/uniref100, or a UniRef90/100 url)"),
    ({"clustering": "linclust"}, "data.yaml: cluster_subsampling.clustering must be one of "
     "auto, uniprot, mmseqs2, got 'linclust'"),
])
def test_from_yaml_refuses_a_bad_clustering_before_dvc(tmp_path, monkeypatch, block, message):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    (tmp_path / "src").mkdir()
    (tmp_path / "src/own.fasta").write_text(">p1\nMKVLAAGIVGLLLAAGSAQAEELK\n")
    _project(tmp_path, {"enabled": True, **block}, {"url": None, "local_fasta": "src/own.fasta"})
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
    _project(tmp_path, {"enabled": True, "clustering": "mmseqs2"},
             {"url": None, "local_fasta": "src/u90.fasta"})
    monkeypatch.setattr(ct, "find_mmseqs", lambda explicit=None: None)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("DVC must not run"))
    r = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
    assert r.exit_code != 0 and "clustered with MMseqs2, but `mmseqs` was not found" in r.output
