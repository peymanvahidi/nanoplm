"""cluster_subsampling config resolution: the six documented scenarios and every error."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from nanoplm.data.preparation import cluster_config as cc

TEMPLATE = Path(cc.__file__).resolve().parents[2] / "cli/templates/data_pretrain.yaml"
U50, U90, U100 = (cc.UNIPROT_FASTA[k] for k in ("uniref50", "uniref90", "uniref100"))
TREMBL = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz"


def _config(url=None, local=None, **block):
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg = copy.deepcopy(cfg)
    cfg["data_dirs"]["url"] = url
    cfg["data_dirs"]["local_fasta"] = local
    cfg["data_params"]["shuffle"] = False
    cfg["data_params"]["shuffle_seed"] = None
    cfg["cluster_subsampling"].update({"enabled": True, **block})
    return cfg


def test_template_block_is_off_by_default():
    cfg = yaml.safe_load(TEMPLATE.read_text())
    assert cfg["cluster_subsampling"] == {"enabled": False, "members": None, "clusters": "auto",
                                          "identity": 0.5, "coverage": 0.8}
    assert cc.resolve_cluster_plan(cfg) is None                      # scenario 1


def test_scenario2_uniref50_url_uniref90_members():
    p = cc.resolve_cluster_plan(_config(url=U50, members="uniref90"))
    raw = Path("output/data/raw")
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"
    assert not p.members_from_source and p.members == U90
    assert p.members_compressed == raw / "uniref90.fasta.gz"
    assert p.members_fasta == raw / "uniref90.fasta"
    assert p.xml_path == raw / "uniref50.xml.gz" and p.xml_url == cc.UNIREF50_XML_URL
    assert p.members_tsv == Path("output/data/pretrain_data/cluster_index/members.tsv")
    assert p.index_dir == Path("output/data/pretrain_data/cluster_index/index")
    assert p.warnings == []


def test_scenario3_uniref100_members():
    p = cc.resolve_cluster_plan(_config(url=U50, members="uniref100"))
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref100"
    assert p.members_fasta.name == "uniref100.fasta"


def test_scenario4_own_fasta_is_clustered_with_mmseqs(tmp_path):
    own = tmp_path / "my_proteins.fasta"
    own.write_text(">p1 x\nMKV\n")
    p = cc.resolve_cluster_plan(_config(local=str(own)))
    assert p.cluster_source == "mmseqs" and p.members_from_source
    assert p.members_level is None and p.identity == 0.5 and p.coverage == 0.8


def test_scenario5_own_fasta_with_a_table(tmp_path):
    table = tmp_path / "clusters.tsv"
    table.write_text("c1\tp1\n")
    p = cc.resolve_cluster_plan(_config(url="https://example.org/mine.fasta.gz",
                                        clusters=str(table)))
    assert p.cluster_source == "table" and p.table == table


def test_scenario6_uniref90_url_or_local_fasta(tmp_path):
    p = cc.resolve_cluster_plan(_config(url=U90))
    assert p.cluster_source == "uniref_xml" and p.members_from_source
    assert p.members_fasta == Path("output/data/raw/sequences.fasta")
    # a local file is classified by its headers, whatever its name
    local = tmp_path / "seqs.fasta"
    local.write_text(">UniRef90_A0A1 desc\nMKV\n")
    p = cc.resolve_cluster_plan(_config(local=str(local)))
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"
    # members: uniref90 on a uniref90 url is the url itself, not a second download
    p = cc.resolve_cluster_plan(_config(url=U90, members="uniref90"))
    assert p.members_from_source


def test_uniref_members_need_a_uniref50_or_same_set_url():
    with pytest.raises(cc.ClusterConfigError, match="must be UniRef50 or uniref90 itself"):
        cc.resolve_cluster_plan(_config(url=TREMBL, members="uniref90"))
    with pytest.raises(cc.ClusterConfigError, match="must be UniRef50 or uniref100 itself"):
        cc.resolve_cluster_plan(_config(url=U90, members="uniref100"))


def test_a_non_uniref_members_fasta_must_go_in_url(tmp_path):
    with pytest.raises(cc.ClusterConfigError, match=r"Put that FASTA in data_dirs.url .*"
                                                    r"members: null"):
        cc.resolve_cluster_plan(_config(url=U50, members="https://example.org/mine.fasta.gz"))


def test_identity_and_coverage_are_ignored_with_uniprot_clusters():
    p = cc.resolve_cluster_plan(_config(url=U50, members="uniref90", identity=0.3))
    assert any("ignored with UniProt clusters" in w for w in p.warnings)
    p = cc.resolve_cluster_plan(_config(url=U50, members="uniref90"))   # template values
    assert p.warnings == []


def test_shuffle_is_ignored_with_a_warning():
    cfg = _config(url=U90)
    cfg["data_params"].update(shuffle=True, shuffle_seed=24)
    assert any("ignores data_params.shuffle" in w
               for w in cc.resolve_cluster_plan(cfg).warnings)


@pytest.mark.parametrize("block,match", [
    ({"identity": 0}, "identity must be a number in"),
    ({"coverage": 1.5}, "coverage must be a number in"),
    ({"clusters": "/nope/table.tsv"}, "not found"),
    ({"clusters": 3}, "auto or a path"),
    ({"members": 5}, "string or null"),
    ({"enabled": "yes"}, "true or false"),
    ({"cluster_url": "x"}, "unknown cluster_subsampling keys"),
])
def test_bad_values_are_refused(block, match):
    with pytest.raises(cc.ClusterConfigError, match=match):
        cc.resolve_cluster_plan(_config(url=U90, **block))


def test_experiment_dirs_are_refused(tmp_path):
    from nanoplm.data.datasets import open_train_dataset
    from tests.data.datasets.test_cluster_sampled import _manifest
    d = tmp_path / "old"
    (d / "train").mkdir(parents=True)
    (d / "RUNTIME_SAMPLER_REQUIRED").write_text("{}\n")
    with pytest.raises(cc.ClusterConfigError, match="Rebuild it with `nanoplm data"):
        open_train_dataset(d, _manifest(), seed=1)
    # no marker, but train/ is a pool with the experiment's index next to it
    pool = tmp_path / "exp/pool"
    (pool / "shards").mkdir(parents=True)
    (pool / "index").mkdir()
    (pool / "index/pool_manifest.json").write_text("{}")
    d2 = tmp_path / "old2"
    d2.mkdir()
    (d2 / "train").symlink_to(pool / "shards", target_is_directory=True)
    with pytest.raises(cc.ClusterConfigError, match="data-sampler experiment"):
        open_train_dataset(d2, _manifest(), seed=1)


def test_relative_paths_resolve_against_the_config_dir(tmp_path, monkeypatch):
    cfg_dir, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    (cfg_dir / "src").mkdir(parents=True)
    elsewhere.mkdir()
    # UniRef90 by its headers only: the file name says nothing
    (cfg_dir / "src/mine.fasta").write_text(">UniRef90_A0A000 x\nMKV\n")
    (cfg_dir / "src/table.tsv").write_text("c\tUniRef90_A0A000\n")
    monkeypatch.chdir(elsewhere)
    p = cc.resolve_cluster_plan(_config(local="src/mine.fasta"), cfg_dir)
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"
    assert p.members_fasta == Path("output/data/raw/sequences.fasta")   # kept as written
    p = cc.resolve_cluster_plan(_config(local="src/mine.fasta", clusters="src/table.tsv"),
                                cfg_dir)
    assert p.cluster_source == "table" and p.table == Path("src/table.tsv")
    with pytest.raises(cc.ClusterConfigError, match="not found"):
        cc.resolve_cluster_plan(_config(local="src/mine.fasta", clusters="src/table.tsv"))
