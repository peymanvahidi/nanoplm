"""cluster_subsampling config resolution: the six documented scenarios and every error."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from nanoplm.data.preparation import cluster_config as cc
from nanoplm.data.preparation.data_config import DataConfigError, parse_data_config

TEMPLATE = Path(cc.__file__).resolve().parents[2] / "cli/templates/data_pretrain.yaml"
U50, U90, U100 = (cc.UNIPROT_FASTA[k] for k in ("uniref50", "uniref90", "uniref100"))
TREMBL = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz"


OUT = Path("output/data/pretrain_data")
DOWNLOADS, WORK = OUT / "downloads", OUT / "src"


def _config(url=None, local=None, **block):
    """The template with one source "src" holding a cluster block."""
    cfg = copy.deepcopy(yaml.safe_load(TEMPLATE.read_text()))
    cfg["data_params"].update(max_seqs=-1, val_ratio=0.0044)
    cfg["sources"] = {"src": {"fasta": url or local,
                              "cluster_subsampling": {"members": None, **block}}}
    return cfg


def _plan(cfg, base_dir=None):
    return cc.resolve_cluster_plan(parse_data_config(cfg, base_dir).source, base_dir)


def test_template_source_has_no_cluster_block():
    cfg = yaml.safe_load(TEMPLATE.read_text())
    assert "cluster_subsampling" not in cfg["sources"]["swissprot"]
    assert _plan(cfg) is None                                        # scenario 1


def test_scenario2_uniref50_url_uniref90_members():
    p = _plan(_config(url=U50, members="uniref90"))
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"
    assert not p.members_from_source and p.members == U90
    assert p.members_compressed == DOWNLOADS / "uniref90.fasta.gz"
    assert p.members_fasta == WORK / "uniref90.fasta"
    assert p.xml_path == DOWNLOADS / "uniref50.xml.gz" and p.xml_url == cc.UNIREF50_XML_URL
    assert p.members_tsv == Path("output/data/pretrain_data/cluster_index/members.tsv")
    assert p.index_dir == Path("output/data/pretrain_data/cluster_index/index")
    assert p.warnings == []


def test_scenario3_uniref100_members():
    p = _plan(_config(url=U50, members="uniref100"))
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref100"
    assert p.members_fasta.name == "uniref100.fasta"


def test_scenario4_own_fasta_is_clustered_with_mmseqs(tmp_path):
    own = tmp_path / "my_proteins.fasta"
    own.write_text(">p1 x\nMKV\n")
    p = _plan(_config(local=str(own)))
    assert p.cluster_source == "mmseqs" and p.members_from_source
    assert p.members_level is None and p.identity == 0.5 and p.coverage == 0.8


def test_scenario5_uniref90_url_or_local_fasta(tmp_path):
    p = _plan(_config(url=U90))
    assert p.cluster_source == "uniref_xml" and p.members_from_source
    assert p.members_fasta == WORK / "uniref90.fasta"
    # a local file is classified by its headers, whatever its name
    local = tmp_path / "seqs.fasta"
    local.write_text(">UniRef90_A0A1 desc\nMKV\n")
    p = _plan(_config(local=str(local)))
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"
    # members: uniref90 on a uniref90 url is the url itself, not a second download
    p = _plan(_config(url=U90, members="uniref90"))
    assert p.members_from_source


def test_uniref_members_need_a_uniref50_or_same_set_url():
    with pytest.raises(cc.ClusterConfigError, match="must be UniRef50 or uniref90 itself"):
        _plan(_config(url=TREMBL, members="uniref90"))
    with pytest.raises(cc.ClusterConfigError, match="must be UniRef50 or uniref100 itself"):
        _plan(_config(url=U90, members="uniref100"))


@pytest.mark.parametrize("members", ["uniref50", "https://example.org/x/uniref50.fasta.gz"])
def test_uniref50_members_are_refused_clearly(members):
    with pytest.raises(cc.ClusterConfigError, match="UniRef50 entries are the clusters "
                                                    "themselves. Use members: uniref90"):
        _plan(_config(url=U50, members=members))


def test_uniref50_url_alone_warns_about_mmseqs_over_all_of_it():
    p = _plan(_config(url=U50))
    assert p.cluster_source == "mmseqs"
    assert any("~60M UniRef50 sequences" in w and "members: uniref90" in w
               and "clustering: auto or uniprot" in w for w in p.warnings)
    # the suggestion resolves cleanly, with either value it names
    for clustering in ("auto", "uniprot"):
        p = _plan(_config(url=U50, members="uniref90", clustering=clustering))
        assert p.cluster_source == "uniref_xml" and not any("~60M" in w for w in p.warnings)
    # an explicit mmseqs2 is the user's choice: no warning
    p = _plan(_config(url=U50, clustering="mmseqs2"))
    assert p.cluster_source == "mmseqs" and not any("~60M" in w for w in p.warnings)


def test_a_missing_local_members_file_is_caught_at_config_time(tmp_path):
    with pytest.raises(cc.ClusterConfigError, match="members: data/uniref90.fasta not found"):
        _plan(_config(url=U50, members="data/uniref90.fasta"), tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/uniref90.fasta").write_text(">UniRef90_A x\nMKV\n")
    p = _plan(_config(url=U50, members="data/uniref90.fasta"), tmp_path)
    assert p.members_level == "uniref90" and p.cluster_source == "uniref_xml"


def test_a_non_uniref_members_fasta_must_go_in_url(tmp_path):
    with pytest.raises(cc.ClusterConfigError, match=r"Put that FASTA in fasta and set "
                                                    r"members: null"):
        _plan(_config(url=U50, members="https://example.org/mine.fasta.gz"))


def test_identity_and_coverage_are_ignored_with_uniprot_clusters():
    p = _plan(_config(url=U50, members="uniref90", identity=0.3))
    assert any("ignored with UniProt clusters" in w and "clustering: mmseqs2 to use them" in w
               for w in p.warnings)
    p = _plan(_config(url=U50, members="uniref90", identity=0.3,
                                        clustering="mmseqs2"))
    assert p.cluster_source == "mmseqs" and p.identity == 0.3
    assert not any("identity / coverage" in w for w in p.warnings)
    p = _plan(_config(url=U50, members="uniref90"))   # template values
    assert p.warnings == []


@pytest.mark.parametrize("shuffle, seed, warns", [
    (True, 24, False), (False, 24, True), (True, 7, True), (True, None, True)])
def test_shuffle_warns_only_when_changed_from_the_template(shuffle, seed, warns):
    cfg = _config(url=U90)
    cfg["data_params"].update(shuffle=shuffle, shuffle_seed=seed)
    assert any("ignores shuffle / shuffle_seed" in w
               for w in _plan(cfg).warnings) == warns


@pytest.mark.parametrize("block,match", [
    ({"identity": 0}, "sources.src.cluster_subsampling.identity must be a number in"),
    ({"coverage": 1.5}, "coverage must be a number in"),
    ({"clusters": "auto"}, "sources.src.cluster_subsampling: unknown key clusters"),
    ({"members": 5}, "string or null"),
    ({"enabled": True}, "remove enabled; the block being present turns it on"),
    ({"enabled": False}, "remove enabled; the block being present turns it on"),
    ({"cluster_url": "x"}, "unknown key cluster_url"),
    ({"clustering": "mmseqs"}, "clustering must be one of auto, uniprot, mmseqs2, got 'mmseqs'"),
    ({"clustering": None}, "clustering must be one of auto, uniprot, mmseqs2"),
])
def test_bad_values_are_refused(block, match):
    with pytest.raises((cc.ClusterConfigError, DataConfigError), match=match):
        _plan(_config(url=U90, **block))


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
    monkeypatch.chdir(elsewhere)
    p = _plan(_config(local="src/mine.fasta"), cfg_dir)
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"
    assert p.members_fasta == Path("src/mine.fasta")   # read in place, kept as written


def test_template_data_params_warn_at_config_time():
    cfg = _config(url=U50, members="uniref90")
    cfg["data_params"].update(max_seqs=20000, val_ratio=0.1)        # the template's
    w = _plan(cfg).warnings
    assert ("sources.src: max_seqs=20000 (from data_params) trains on at most 20,000 "
            "clusters; set sources.src.max_seqs: -1 to use all of them") in w
    assert ("sources.src: val_ratio=0.1 (from data_params) holds out about 10.0% of the "
            "clusters for validation; the UniRef90 experiment used data_params.val_ratio: "
            "0.0044") in w


def test_cluster_warnings_name_the_source_and_its_own_override(tmp_path):
    """A mix shares data_params, so every warning says which source it is about and
    the fix is set on that source; the 0.0044 hint is for UniProt clusters only."""
    cfg = _config(local=_own(tmp_path))
    cfg["data_params"]["val_ratio"] = 0.2
    w = _plan(cfg).warnings
    assert ("sources.src: val_ratio=0.2 (from data_params) holds out about 20.0% of the "
            "clusters for validation; set data_params.val_ratio lower to train on more of "
            "them") in w
    assert not any("0.0044" in x for x in w)
    p = _plan(_config(url=U50, members="uniref90", identity=0.3))
    assert any(x.startswith("sources.src: ") and
               "Set sources.src.cluster_subsampling.clustering: mmseqs2" in x
               for x in p.warnings)
    p = _plan(_config(url=U50))
    assert any(x.startswith("sources.src: fasta is UniRef50 and members is null") and
               "set sources.src.cluster_subsampling.members: uniref90" in x
               for x in p.warnings)


def test_uniprot_error_says_members_null(tmp_path):
    with pytest.raises(cc.ClusterConfigError, match="and members null. Use clustering"):
        _plan(_config(local=_own(tmp_path), clustering="uniprot"))


# ------------------------------------------------------------------ clustering


def _own(tmp_path):
    own = tmp_path / "my_proteins.fasta"
    own.write_text(">p1 x\nMKV\n")
    return str(own)


# (url, local, members) -> what auto resolves to; UniRef data first, then non-UniRef
UNIREF_CASES = {
    "u50+uniref90": (dict(url=U50, members="uniref90"), "uniref90"),
    "u50+uniref100": (dict(url=U50, members="uniref100"), "uniref100"),
    "u90 url": (dict(url=U90), "uniref90"),
    "u100 url": (dict(url=U100), "uniref100"),
}
NON_UNIREF_CASES = {
    "u50 alone": dict(url=U50),
    "trembl": dict(url=TREMBL),
    "own fasta": "own",
}


@pytest.mark.parametrize("case", list(UNIREF_CASES))
@pytest.mark.parametrize("clustering", cc.CLUSTERINGS)
def test_clustering_on_uniref_data(case, clustering):
    kw, level = UNIREF_CASES[case]
    p = _plan(_config(clustering=clustering, **kw))
    assert p.clustering == clustering and p.members_level == level
    if clustering == "mmseqs2":
        assert p.cluster_source == "mmseqs" and p.xml_url is None and p.xml_path is None
        assert p.source_line() == "MMseqs2 linclust, identity 0.5, coverage 0.8"
    else:
        assert p.cluster_source == "uniref_xml" and p.xml_url == cc.UNIREF50_XML_URL
        assert p.source_line() == (f"UniProt uniref50.xml.gz "
                                   f"(members: UniRef{level[len('uniref'):]})")


@pytest.mark.parametrize("case", list(NON_UNIREF_CASES))
@pytest.mark.parametrize("clustering", cc.CLUSTERINGS)
def test_clustering_on_non_uniref_data(case, clustering, tmp_path):
    kw = NON_UNIREF_CASES[case]
    kw = dict(local=_own(tmp_path)) if kw == "own" else kw
    if clustering == "uniprot":
        with pytest.raises(cc.ClusterConfigError) as e:
            _plan(_config(clustering=clustering, **kw))
        assert ("clustering: uniprot: UniProt clusters need UniRef data (fasta UniRef50 + "
                "members uniref90/uniref100, or a UniRef90/100 fasta)") in str(e.value)
        return
    p = _plan(_config(clustering=clustering, **kw))
    assert p.cluster_source == "mmseqs" and p.members_from_source and p.xml_url is None
    assert p.source_line() == "MMseqs2 linclust, identity 0.5, coverage 0.8"


def test_a_missing_clustering_key_means_auto():
    cfg = _config(url=U50, members="uniref90")
    assert "clustering" not in cfg["sources"]["src"]["cluster_subsampling"]
    p = _plan(cfg)
    assert p.clustering == "auto" and p.cluster_source == "uniref_xml"


def test_mmseqs2_on_uniref50_url_clusters_the_members_without_the_url_fasta():
    p = _plan(_config(url=U50, members="uniref90", clustering="mmseqs2",
                                        identity=0.3))
    assert p.cluster_source == "mmseqs" and not p.members_from_source and p.members == U90
    assert p.members_fasta == WORK / "uniref90.fasta"
    assert p.source_line() == "MMseqs2 linclust, identity 0.3, coverage 0.8"
    assert p.notes == [f"sources.src: clustering: mmseqs2 clusters the members ({U90}); the source "
                       f"fasta ({U50}) is not needed and is not downloaded"]
    # the member/url pairing rules hold under every clustering value
    with pytest.raises(cc.ClusterConfigError, match="must be UniRef50 or uniref90 itself"):
        _plan(_config(url=TREMBL, members="uniref90", clustering="mmseqs2"))


def test_members_are_read_in_place_or_downloaded(tmp_path):
    """A local members file is a dep, never copied; a URL goes to downloads/."""
    gz = tmp_path / "uniref90.fasta.gz"
    import gzip
    gz.write_bytes(gzip.compress(b">UniRef90_A x\nMKV\n"))
    p = _plan(_config(url=U50, members=str(gz)))
    assert p.members_compressed == gz and p.members_fasta == WORK / "uniref90.fasta"
    plain = tmp_path / "u90.fasta"
    plain.write_text(">UniRef90_A x\nMKV\n")
    p = _plan(_config(url=U50, members=str(plain)))
    assert p.members_compressed is None and p.members_fasta == plain
