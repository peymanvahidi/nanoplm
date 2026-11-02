"""cluster_subsampling config resolution: the six documented scenarios and every error."""
from __future__ import annotations

import copy
import gzip
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
    p = _plan(_config(url=U50, members="uniref90", clusters="uniprot"))
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"
    assert not p.members_from_source and p.members == U90
    assert p.members_compressed == DOWNLOADS / "uniref90.fasta.gz"
    assert p.members_fasta == WORK / "uniref90.fasta"
    assert p.xml_path == DOWNLOADS / "uniref50.xml.gz" and p.xml_url == cc.UNIREF50_XML_URL
    assert p.members_tsv == Path("output/data/pretrain_data/cluster_index/members.tsv")
    assert p.index_dir == Path("output/data/pretrain_data/cluster_index/index")
    assert p.warnings == []


def test_scenario3_uniref100_members():
    p = _plan(_config(url=U50, members="uniref100", clusters="uniprot"))
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref100"
    assert p.members_fasta.name == "uniref100.fasta"


def test_scenario4_own_fasta_is_clustered_with_mmseqs(tmp_path):
    own = tmp_path / "my_proteins.fasta"
    own.write_text(">p1 x\nMKV\n")
    p = _plan(_config(local=str(own), clusters="mmseqs2"))
    assert p.cluster_source == "mmseqs" and p.members_from_source
    assert p.members_level is None and p.identity == 0.5 and p.coverage == 0.8


def test_scenario5_uniref90_url_or_local_fasta(tmp_path):
    p = _plan(_config(url=U90, clusters="uniprot"))
    assert p.cluster_source == "uniref_xml" and p.members_from_source
    assert p.members_fasta == WORK / "uniref90.fasta"
    # a local file is classified by its headers, whatever its name
    local = tmp_path / "seqs.fasta"
    local.write_text(">UniRef90_A0A1 desc\nMKV\n")
    p = _plan(_config(local=str(local), clusters="uniprot"))
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"
    # members: uniref90 on a uniref90 url is the url itself, not a second download
    p = _plan(_config(url=U90, members="uniref90", clusters="uniprot"))
    assert p.members_from_source


def test_uniref_members_need_a_uniref50_or_same_set_url():
    with pytest.raises(cc.ClusterConfigError, match="must be UniRef50 or uniref90 itself"):
        _plan(_config(url=TREMBL, members="uniref90", clusters="uniprot"))
    with pytest.raises(cc.ClusterConfigError, match="must be UniRef50 or uniref100 itself"):
        _plan(_config(url=U90, members="uniref100", clusters="uniprot"))


@pytest.mark.parametrize("members", ["uniref50", "https://example.org/x/uniref50.fasta.gz"])
def test_uniref50_members_are_refused_clearly(members):
    with pytest.raises(cc.ClusterConfigError, match="UniRef50 entries are the clusters "
                                                    "themselves. Use members: uniref90"):
        _plan(_config(url=U50, members=members, clusters="uniprot"))


def test_uniref50_url_alone_with_mmseqs2_is_the_users_choice():
    # mmseqs2 runs only when asked, so it draws no warning, even on all of UniRef50
    p = _plan(_config(url=U50, clusters="mmseqs2"))
    assert p.cluster_source == "mmseqs" and p.warnings == []
    # uniprot on UniRef50 alone has no finer level to sample
    with pytest.raises(cc.ClusterConfigError, match="UniProt clusters need UniRef data"):
        _plan(_config(url=U50, clusters="uniprot"))


def test_a_missing_local_members_file_is_caught_at_config_time(tmp_path):
    with pytest.raises(cc.ClusterConfigError, match="members: data/uniref90.fasta not found"):
        _plan(_config(url=U50, members="data/uniref90.fasta", clusters="uniprot"), tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/uniref90.fasta").write_text(">UniRef90_A x\nMKV\n")
    p = _plan(_config(url=U50, members="data/uniref90.fasta", clusters="uniprot"), tmp_path)
    assert p.members_level == "uniref90" and p.cluster_source == "uniref_xml"


def test_a_non_uniref_members_fasta_must_go_in_url(tmp_path):
    with pytest.raises(cc.ClusterConfigError, match=r"Put that FASTA in fasta and leave "
                                                    r"members out"):
        _plan(_config(url=U50, members="https://example.org/mine.fasta.gz", clusters="mmseqs2"))


def test_identity_and_coverage_are_an_error_with_uniprot_clusters():
    with pytest.raises(cc.ClusterConfigError,
                       match=r"identity / coverage only apply when MMseqs2 makes the clusters, "
                             r"and these come from UniProt's clusters\. .*set "
                             r"sources.src.cluster_subsampling.clusters: mmseqs2"):
        _plan(_config(url=U50, members="uniref90", identity=0.3, clusters="uniprot"))
    with pytest.raises(cc.ClusterConfigError, match="come from UniProt's clusters. Remove"):
        _plan(_config(url=U50, members="uniref90", coverage=0.9, clusters="uniprot"))
    p = _plan(_config(url=U50, members="uniref90", identity=0.3, clusters="mmseqs2"))
    assert p.cluster_source == "mmseqs" and p.identity == 0.3
    assert not any("identity / coverage" in w for w in p.warnings)
    # the old template wrote the defaults out: no error, no warning
    p = _plan(_config(url=U50, members="uniref90", identity=0.5, coverage=0.8, clusters="uniprot"))
    assert p.cluster_source == "uniref_xml" and p.warnings == []


@pytest.mark.parametrize("shuffle, seed, warns", [
    (True, 24, False), (False, 24, True), (True, 7, True), (True, None, True)])
def test_shuffle_warns_only_when_changed_from_the_template(shuffle, seed, warns):
    cfg = _config(url=U90, clusters="uniprot")
    cfg["data_params"].update(shuffle=shuffle, shuffle_seed=seed)
    assert any("ignores shuffle / shuffle_seed" in w
               for w in _plan(cfg).warnings) == warns


@pytest.mark.parametrize("block,match", [
    ({"clusters": "uniprot", "identity": 0},
     "sources.src.cluster_subsampling.identity must be a number in"),
    ({"clusters": "uniprot", "coverage": 1.5}, "coverage must be a number in"),
    ({"clusters": "uniprot", "members": 5}, "string or null"),
    ({"enabled": True}, "remove enabled; the block turns it on"),
    ({"enabled": False}, "remove enabled; the block turns it on"),
    ({"clusters": "uniprot", "cluster_url": "x"}, "unknown key cluster_url"),
    ({"clustering": "mmseqs"}, "sources.src.cluster_subsampling.clustering must be uniprot, "
                               "headers, mmseqs2, or a cluster file ending in .tsv "
                               r"\(.tsv.gz, .tsv.zst\) with representative<TAB>member lines, "
                               "got 'mmseqs'"),
    ({"clusters": 5}, "sources.src.cluster_subsampling.clusters must be uniprot"),
    ({"clusters": "clusters.txt"}, "or a cluster file ending in .tsv"),
    ({"clusters": "uniprot", "clustering": "uniprot"}, "set clusters or clustering, not both"),
    ({"clusters": "auto"}, "sources.src.cluster_subsampling.clusters: auto was removed; "
                           "choose where the clusters come from: uniprot"),
    ({"clustering": "auto"}, "sources.src.cluster_subsampling.clustering: auto was removed"),
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
    p = _plan(_config(local="src/mine.fasta", clusters="uniprot"), cfg_dir)
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"
    assert p.members_fasta == Path("src/mine.fasta")   # read in place, kept as written


def test_template_data_params_warn_at_config_time():
    cfg = _config(url=U50, members="uniref90", clusters="uniprot")
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
    cfg = _config(local=_own(tmp_path), clusters="mmseqs2")
    cfg["data_params"]["val_ratio"] = 0.2
    w = _plan(cfg).warnings
    assert ("sources.src: val_ratio=0.2 (from data_params) holds out about 20.0% of the "
            "clusters for validation; set data_params.val_ratio lower to train on more of "
            "them") in w
    assert not any("0.0044" in x for x in w)


def test_uniprot_error_says_members_null(tmp_path):
    with pytest.raises(cc.ClusterConfigError, match="and members null. Use clusters: headers, mmseqs2"):
        _plan(_config(local=_own(tmp_path), clusters="uniprot"))


# ------------------------------------------------------------------ clusters


def _own(tmp_path):
    own = tmp_path / "my_proteins.fasta"
    own.write_text(">p1 x\nMKV\n")
    return str(own)


# (url, local, members) -> the UniRef level; UniRef data first, then non-UniRef
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


CHOICES = ("uniprot", "mmseqs2")


@pytest.mark.parametrize("case", list(UNIREF_CASES))
@pytest.mark.parametrize("clusters", CHOICES)
@pytest.mark.parametrize("key", ["clusters", "clustering"])     # the old name still works
def test_clusters_on_uniref_data(case, clusters, key):
    kw, level = UNIREF_CASES[case]
    p = _plan(_config(**{key: clusters}, **kw))
    assert p.clusters == clusters and p.members_level == level
    assert p.clusters_key == f"sources.src.cluster_subsampling.{key}"
    trains = "this fasta" if "url" in kw and "members" not in kw else \
        f"UniRef{level[len('uniref'):]} members"
    if clusters == "mmseqs2":
        assert p.cluster_source == "mmseqs" and p.xml_url is None and p.xml_path is None
        assert p.source_line() == ("clusters: MMseqs2 (50% identity, 80% coverage) | "
                                   f"trains on: {trains}")
    else:
        assert p.cluster_source == "uniref_xml" and p.xml_url == cc.UNIREF50_XML_URL
        assert p.source_line() == f"clusters: UniProt uniref50.xml.gz | trains on: {trains}"


@pytest.mark.parametrize("case", list(NON_UNIREF_CASES))
@pytest.mark.parametrize("clusters", CHOICES)
def test_clusters_on_non_uniref_data(case, clusters, tmp_path):
    kw = NON_UNIREF_CASES[case]
    kw = dict(local=_own(tmp_path)) if kw == "own" else kw
    if clusters == "uniprot":
        with pytest.raises(cc.ClusterConfigError) as e:
            _plan(_config(clusters=clusters, **kw))
        assert ("clusters: uniprot: UniProt clusters need UniRef data (fasta UniRef50 + "
                "members uniref90/uniref100, or a UniRef90/100 fasta)") in str(e.value)
        return
    p = _plan(_config(clusters=clusters, **kw))
    assert p.cluster_source == "mmseqs" and p.members_from_source and p.xml_url is None
    assert p.source_line() == ("clusters: MMseqs2 (50% identity, 80% coverage) | "
                               "trains on: this fasta")


CHOICES_TEXT = ("uniprot (UniProt's UniRef50 clusters, for UniRef90/UniRef100 members), "
                "headers (CL: tags in the FASTA headers), mmseqs2 (clusters with MMseqs2; "
                "slow: hours on large files), or a .tsv file of representative<TAB>member lines")


@pytest.mark.parametrize("block", [{"members": "uniref90"}, {"members": "uniref90",
                                                              "clusters": None},
                                   {"members": "uniref90", "clustering": None}])
def test_a_block_without_clusters_lists_the_four_choices(block):
    cfg = _config(url=U50)
    cfg["sources"]["src"]["cluster_subsampling"] = block
    with pytest.raises(DataConfigError) as e:
        _plan(cfg)
    assert str(e.value) == ("sources.src.cluster_subsampling.clusters is required: "
                            + CHOICES_TEXT)


def test_cluster_subsampling_true_or_empty_lists_the_four_choices():
    cfg = _config(url=U90)
    cfg["sources"]["src"]["cluster_subsampling"] = True
    with pytest.raises(DataConfigError) as e:
        _plan(cfg)
    assert str(e.value) == ("sources.src.cluster_subsampling: true no longer turns it on; "
                            "write a block that sets clusters to " + CHOICES_TEXT)
    cfg["sources"]["src"]["cluster_subsampling"] = None
    with pytest.raises(DataConfigError, match="is empty. Set clusters in it to uniprot "):
        _plan(cfg)
    cfg["sources"]["src"]["cluster_subsampling"] = False
    assert _plan(cfg) is None


def test_mmseqs2_on_uniref50_url_clusters_the_members_without_the_url_fasta():
    p = _plan(_config(url=U50, members="uniref90", clusters="mmseqs2",
                                        identity=0.3))
    assert p.cluster_source == "mmseqs" and not p.members_from_source and p.members == U90
    assert p.members_fasta == WORK / "uniref90.fasta"
    assert p.source_line() == ("clusters: MMseqs2 (30% identity, 80% coverage) | "
                               "trains on: UniRef90 members")
    assert p.notes == [f"sources.src: clusters: mmseqs2 clusters the members ({U90}); the source "
                       f"fasta ({U50}) is not needed and is not downloaded"]
    # the member/url pairing rules hold under every clusters value
    with pytest.raises(cc.ClusterConfigError, match="must be UniRef50 or uniref90 itself"):
        _plan(_config(url=TREMBL, members="uniref90", clusters="mmseqs2"))


def test_members_are_read_in_place_or_downloaded(tmp_path):
    """A local members file is a dep, never copied; a URL goes to downloads/."""
    gz = tmp_path / "uniref90.fasta.gz"
    import gzip
    gz.write_bytes(gzip.compress(b">UniRef90_A x\nMKV\n"))
    p = _plan(_config(url=U50, members=str(gz), clusters="uniprot"))
    assert p.members_compressed == gz and p.members_fasta == WORK / "uniref90.fasta"
    plain = tmp_path / "u90.fasta"
    plain.write_text(">UniRef90_A x\nMKV\n")
    p = _plan(_config(url=U50, members=str(plain), clusters="uniprot"))
    assert p.members_compressed is None and p.members_fasta == plain


# ------------------------------------------------------------------ .zst / .fa sources


UNIREF90_TEXT = b"".join(b">UniRef90_P%05d n=1\n%s\n" % (i, b"MAGVKLLW" * 20) for i in range(50))


def _long31_zst(path: Path) -> Path:
    """A .zst as `zstd --long=31` writes it: a 2 GiB window, more than the 128 MiB a
    decoder accepts by default."""
    import zstandard
    params = zstandard.ZstdCompressionParameters.from_level(
        3, window_log=31, enable_ldm=True, write_content_size=False)
    obj = zstandard.ZstdCompressor(compression_params=params).compressobj()
    path.write_bytes(obj.compress(UNIREF90_TEXT) + obj.flush())
    assert zstandard.get_frame_parameters(path.read_bytes()).window_size > 1 << 27
    return path


def test_a_long31_uniref90_zst_gets_uniprot_clusters(tmp_path):
    """A local UniRef90 .fasta.zst made with --long=31 is UniRef90 by its headers, so
    clusters: uniprot accepts it."""
    local = _long31_zst(tmp_path / "seqs.fasta.zst")
    assert cc.level_of(str(local)) == "uniref90"
    p = _plan(_config(local=str(local), clusters="uniprot"))
    assert p.cluster_source == "uniref_xml" and p.members_level == "uniref90"


@pytest.mark.parametrize("name, level", [
    ("uniref90.fasta", "uniref90"), ("uniref90.fasta.gz", "uniref90"),
    ("uniref90.fa", "uniref90"), ("uniref50.fa.zst", "uniref50"),
    ("UniRef100.fasta.zst", "uniref100"), ("my_uniref90.fasta", None),
    ("uniref90.faa", None), ("uniref90.fasta.bz2", None)])
def test_the_level_by_file_name_takes_fa_and_zst(name, level):
    assert cc.level_of(f"https://example.org/{name}") == level
    assert cc.level_of(f"not/on/disk/{name}") == level


def test_zst_headers_without_zstandard_use_the_zstd_tool(tmp_path, monkeypatch):
    import shutil
    import sys
    if not shutil.which("zstd"):
        pytest.skip("zstd not installed in test environment")
    local = _long31_zst(tmp_path / "seqs.fa.zst")
    monkeypatch.setitem(sys.modules, "zstandard", None)
    assert cc.level_of(str(local)) == "uniref90"


def test_zst_headers_without_zstandard_or_zstd_say_what_to_install(tmp_path, monkeypatch):
    import sys
    local = _long31_zst(tmp_path / "seqs.fa.zst")
    monkeypatch.setitem(sys.modules, "zstandard", None)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(cc.ClusterConfigError, match="pip install zstandard") as e:
        _plan(_config(local=str(local), clusters="uniprot"))
    assert str(local) in str(e.value) and "zstd" in str(e.value)


# ------------------------------------------------------------- headers / a .tsv / CL: tags


def _on(fasta, block, **params):
    """The template with one source "src" and this cluster_subsampling block."""
    cfg = copy.deepcopy(yaml.safe_load(TEMPLATE.read_text()))
    cfg["data_params"].update(max_seqs=-1, val_ratio=0.0044, **params)
    cfg["sources"] = {"src": {"fasta": fasta, "cluster_subsampling": block}}
    return cfg


CL_FASTA = ">x2\tCL:x1\nMKV\n>x1\tCL:x1\nMKL\n>y1 CL:y1\nMMA\n"


def test_each_choice_on_this_fasta(tmp_path):
    own = tmp_path / "mine.fasta"
    own.write_text(">p1 x\nMKV\n>p2\nMKL\n")
    p = _plan(_on(str(own), {"clusters": "mmseqs2"}))
    assert p.cluster_source == "mmseqs" and p.members_from_source and p.members == str(own)
    assert p.clusters == "mmseqs2"
    assert p.clusters_key == "sources.src.cluster_subsampling.clusters"
    assert p.source_line() == ("clusters: MMseqs2 (50% identity, 80% coverage) | "
                               "trains on: this fasta")
    tagged = tmp_path / "logan.fasta"
    tagged.write_text(CL_FASTA)
    p = _plan(_on(str(tagged), {"clusters": "headers"}))
    assert p.cluster_source == "headers" and p.members_from_source
    assert p.source_line() == "clusters: CL: tags in headers | trains on: this fasta"
    u90 = tmp_path / "seqs.fasta"
    u90.write_text(">UniRef90_A0A000 x\nMKV\n")
    p = _plan(_on(str(u90), {"clusters": "uniprot"}))
    assert p.cluster_source == "uniref_xml"
    assert p.source_line() == "clusters: UniProt uniref50.xml.gz | trains on: this fasta"
    assert _plan(_on(U50, {"members": "uniref90", "clusters": "uniprot"})).source_line() == (
        "clusters: UniProt uniref50.xml.gz | trains on: UniRef90 members")


def test_headers_reads_a_compressed_local_fasta(tmp_path):
    import zstandard
    gz = tmp_path / "logan.fasta.gz"
    gz.write_bytes(gzip.compress(CL_FASTA.encode()))
    assert _plan(_on(str(gz), {"clusters": "headers"})).cluster_source == "headers"
    zst = tmp_path / "logan.fa.zst"
    zst.write_bytes(zstandard.ZstdCompressor().compress(CL_FASTA.encode()))
    assert _plan(_on(str(zst), {"clusters": "headers"})).cluster_source == "headers"


def test_headers_refuses_a_fasta_with_some_cl_tags(tmp_path):
    mixed = tmp_path / "mixed.fasta"
    mixed.write_text(CL_FASTA + ">z1\nMKV\n")
    with pytest.raises(cc.ClusterConfigError, match=r"1 of the first 4 headers of mixed.fasta "
                                                    r"have none \(e.g. >z1\)"):
        _plan(_on(str(mixed), {"clusters": "headers"}))
    # mmseqs2 ignores the tags
    p = _plan(_on(str(mixed), {"clusters": "mmseqs2"}))
    assert p.cluster_source == "mmseqs"


def test_headers_needs_cl_tags(tmp_path):
    own = tmp_path / "mine.fasta"
    own.write_text(">p1 x\nMKV\n>p2\nMKL\n")
    with pytest.raises(cc.ClusterConfigError,
                       match=r"sources.src.cluster_subsampling.clusters: headers needs a "
                             r"CL:<cluster id> tag after the id in every FASTA header; 2 of the "
                             r"first 2 headers of mine.fasta have none \(e.g. >p1 x\)"):
        _plan(_on(str(own), {"clusters": "headers"}))
    with pytest.raises(cc.ClusterConfigError, match="headers needs a CL:<cluster id> tag"):
        _plan(_on(U50, {"members": "uniref90", "clusters": "headers"}))
    tagged = tmp_path / "logan.fasta"
    tagged.write_text(CL_FASTA)
    assert _plan(_on(str(tagged), {"clusters": "headers"})).cluster_source == "headers"


def test_uniprot_needs_uniref_data(tmp_path):
    tagged = tmp_path / "logan.fasta"
    tagged.write_text(CL_FASTA)
    with pytest.raises(cc.ClusterConfigError, match="clusters: uniprot: UniProt clusters need "
                                                    "UniRef data"):
        _plan(_on(str(tagged), {"clusters": "uniprot"}))


def test_identity_with_cl_headers_is_an_error(tmp_path):
    tagged = tmp_path / "logan.fasta"
    tagged.write_text(CL_FASTA)
    with pytest.raises(cc.ClusterConfigError, match=r"these come from the CL: tags of "
                                                    r"logan.fasta\. Remove identity"):
        _plan(_on(str(tagged), {"clusters": "headers", "identity": 0.3}))


def test_a_cluster_tsv_path_or_url(tmp_path):
    own = tmp_path / "mine.fasta"
    own.write_text(">p1\nMKV\n")
    with pytest.raises(cc.ClusterConfigError, match=r"sources.src.cluster_subsampling.clusters:"
                                                    r" data/clu.tsv not found \(a relative path"):
        _plan(_on(str(own), {"clusters": "data/clu.tsv"}), tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/clu.tsv").write_text("p1\tp1\n")
    p = _plan(_on(str(own), {"clusters": "data/clu.tsv"}), tmp_path)
    assert p.cluster_source == "tsv" and p.tsv_path == Path("data/clu.tsv") and not p.tsv_url
    assert p.source_line() == "clusters: data/clu.tsv | trains on: this fasta"
    url = "https://example.org/c/clu_cluster.tsv.gz"
    p = _plan(_on(str(own), {"clusters": url}), tmp_path)
    assert p.tsv_url == url and p.tsv_path == DOWNLOADS / "clu_cluster.tsv.gz"
    assert p.source_line() == "clusters: clu_cluster.tsv.gz | trains on: this fasta"
    # a cluster file is chosen as it is: on UniRef data too, and identity is an error
    p = _plan(_on(U50, {"members": "uniref90", "clusters": url}))
    assert p.cluster_source == "tsv" and p.trains_on() == "UniRef90 members"
    with pytest.raises(cc.ClusterConfigError, match="these come from " + url):
        _plan(_on(str(own), {"clusters": url, "coverage": 0.5}), tmp_path)


def test_mmseqs2_runs_at_any_size(tmp_path):
    """mmseqs2 is only run when asked, so there is no size guard."""
    gz = tmp_path / "big.fasta.gz"
    gz.write_bytes(gzip.compress(b">p1\nMKV\n" * 1000))
    p = _plan(_on(str(gz), {"clusters": "mmseqs2"}))
    assert p.cluster_source == "mmseqs" and p.warnings == []
    assert not hasattr(cc, "MMSEQS_AUTO_MAX_BYTES")


# ------------------------------------------------------------------ headers on a URL


@pytest.fixture
def server(monkeypatch):
    """Fake range requests: url -> (bytes served, total size or None); a missing url fails."""
    files: dict = {}
    calls: list = []

    def http_start(url):
        calls.append(url)
        if url not in files:
            raise OSError("HTTP Error 404: Not Found")
        data, total = files[url]
        return data[:cc.PEEK_FETCH], total

    monkeypatch.setattr(cc, "_http_start", http_start)
    monkeypatch.setattr(cc, "_URL_PEEKS", {})
    return files, calls


def test_headers_on_a_url_reads_its_start(server):
    import zstandard
    files, calls = server
    tagged = "https://example.org/logan.fasta.zst"
    files[tagged] = (zstandard.ZstdCompressor().compress(CL_FASTA.encode() * 5000), 10**12)
    p = _plan(_on(tagged, {"clusters": "headers"}))
    assert p.cluster_source == "headers" and calls == [tagged]
    _plan(_on(tagged, {"clusters": "headers"}))
    assert calls == [tagged]                        # read once per process
    plain = "https://example.org/mine.fasta.gz"
    files[plain] = (gzip.compress(b">p1 x\nMKV\n" * 1000), 5000)
    with pytest.raises(cc.ClusterConfigError, match=r"headers needs a CL:<cluster id> tag .* "
                                                    r"headers of mine.fasta.gz have none"):
        _plan(_on(plain, {"clusters": "headers"}))


def test_a_url_that_cannot_be_read_is_left_to_the_stage(server):
    url = "https://example.org/gone.fasta.gz"
    assert _plan(_on(url, {"clusters": "headers"})).cluster_source == "headers"


def test_headers_reads_the_downloaded_copy_before_the_url(server, tmp_path):
    files, calls = server
    url = "https://example.org/logan.fasta.gz"
    got = tmp_path / DOWNLOADS / "logan.fasta.gz"
    got.parent.mkdir(parents=True)
    got.write_bytes(gzip.compress(CL_FASTA.encode()))
    assert _plan(_on(url, {"clusters": "headers"}), tmp_path).cluster_source == "headers"
    assert calls == []


def test_only_headers_reads_a_url(server):
    """uniprot, mmseqs2 and a .tsv need no request; headers on a UniProt shortcut is refused
    without one."""
    _, calls = server
    url = "https://example.org/mine.fasta.gz"
    _plan(_on(U50, {"members": "uniref90", "clusters": "uniprot"}))
    _plan(_on("uniref90", {"clusters": "uniprot"}))
    _plan(_on(U50, {"clusters": "mmseqs2"}))
    _plan(_on("trembl", {"clusters": "mmseqs2"}))
    _plan(_on(url, {"clusters": "mmseqs2"}))
    _plan(_on(url, {"clusters": "https://example.org/c.tsv.gz"}))
    for shortcut in ("swissprot", "trembl", "uniref50"):
        with pytest.raises(cc.ClusterConfigError, match=f"UniProt's .* has none"):
            _plan(_on(shortcut, {"clusters": "headers"}))
    assert calls == []
