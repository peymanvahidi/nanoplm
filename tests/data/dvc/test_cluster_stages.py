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
    assert "params" not in st["cluster_members"]
    assert "--source uniref_xml --level uniref90" in st["cluster_members"]["cmd"]
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
        {"data.yaml": ["cluster_subsampling.identity", "cluster_subsampling.coverage"]}]
    for name, stage in st.items():
        if name != "cluster_members":
            flat = str(stage.get("params", "")) + str(stage.get("deps", ""))
            assert "cluster_subsampling" not in flat, name
    assert "members.tsv" not in str(st["cluster_pool"]["deps"])


def test_user_table_is_a_dependency(tmp_path):
    table = tmp_path / "t.tsv"
    table.write_text("c\ta\n")
    st = _stages(_enabled(data_dirs__url="https://example.org/mine.fasta.gz",
                          cluster_subsampling__clusters=str(table)))
    assert str(table) in st["cluster_members"]["deps"]
    assert "--source table" in st["cluster_members"]["cmd"]


# ------------------------------------------------------------------ end to end


def _run(args, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)


def _project(tmp_path, block, data_dirs):
    cfg = _template()
    cfg["data_dirs"].update(data_dirs)
    cfg["data_params"].update(seqs_num=-1, min_seq_len=20, max_seq_len=128, val_ratio=0.1,
                              samples_per_shard=101, max_workers=1, shuffle=False,
                              shuffle_seed=None)
    cfg["cluster_subsampling"].update(block)
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}",
               CUDA_VISIBLE_DEVICES="")
    if MMSEQS:
        env["NANOPLM_MMSEQS"] = MMSEQS
    return env


def _from_yaml(tmp_path, env):
    r = _run([str(Path(sys.executable).parent / "nanoplm"), "data", "from-yaml", "data.yaml"],
             tmp_path, env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def _status(tmp_path, env):
    r = _run([sys.executable, "-m", "dvc", "status", "--json"], tmp_path, env)
    assert r.returncode == 0, r.stderr
    import json
    return set(json.loads(r.stdout or "{}"))


@pytest.mark.skipif(MMSEQS is None, reason="mmseqs not installed")
def test_from_yaml_end_to_end_with_mmseqs(tmp_path):
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=120)
    # not a UniRef file name, and no UniRef headers: an "own FASTA"
    own = tmp_path / "src/own.fasta"
    own.write_text(rel.fasta90.read_text().replace(">UniRef90_", ">seq_"))
    env = _project(tmp_path, {"enabled": True},
                   {"url": None, "local_fasta": str(own)})
    _from_yaml(tmp_path, env)
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
    out = tmp_path / "output/data/pretrain_data"
    m = read_manifest(out)
    assert m.cluster_level == "uniref90" and m.cluster_index_dir == "cluster_index/index"
    ds = open_train_dataset(out, m, seed=3)
    assert isinstance(ds, ClusterSampledDataset) and len(ds) == m.train_sequences
