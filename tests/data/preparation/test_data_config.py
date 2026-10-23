"""data.yaml parsing: the sources layout, the old data_dirs layout, and every error."""
from __future__ import annotations

import copy
import textwrap

import pytest
import yaml

from nanoplm.cli.templates import load_template
from nanoplm.data.preparation import data_config as dc

SPROT = dc.FASTA_SHORTCUTS["swissprot"]


def _template():
    return yaml.safe_load(load_template("data_pretrain"))


def _parse(cfg, base_dir=None):
    return dc.parse_data_config(cfg, base_dir)


def _with_source(spec, **top):
    cfg = _template()
    cfg["sources"] = {"src": spec}
    cfg.update(top)
    return cfg


# ------------------------------------------------------------------ the template


def test_the_template_is_one_plain_swissprot_source():
    cfg = _parse(_template())
    assert not cfg.legacy and list(cfg.sources) == ["swissprot"]
    src = cfg.source
    assert src.fasta == SPROT and src.fasta_written == "swissprot" and src.weight == 1.0
    assert src.cluster is None and src.prepared is None and src.overrides == ()
    assert src.params["max_seqs"] == 20000 and src.key("max_seqs") == "data_params.max_seqs"
    out = "output/data/pretrain_data"
    assert cfg.output_dir == out
    assert src.paths == {
        "compressed_fasta": f"{out}/downloads/uniprot_sprot.fasta.gz",
        "extracted_fasta": f"{out}/swissprot/uniprot_sprot.fasta",
        "validation_dir": f"{out}/swissprot/validation",
        "shuffled_fasta": f"{out}/swissprot/shuffled.fasta",
        "filtered_fasta": f"{out}/swissprot/filtered.fasta",
        "splitted_fasta_dir": f"{out}/swissprot/split",
        "output_dir": out,
    }


def test_the_template_comments_say_what_the_design_asks():
    text = load_template("data_pretrain")
    assert "weight = share of training sequences (not tokens); weights must sum to 1" in text
    assert "max_seqs: 20000        # -1 = all" in text
    assert "# Rarely changed" in text
    assert "#   cluster_subsampling:" in text and "#   prepared:" in text
    assert "enabled" not in text and "data_dirs" not in text


@pytest.mark.parametrize("name", list(dc.FASTA_SHORTCUTS))
def test_shortcuts_expand_to_the_uniprot_url(name):
    src = _parse(_with_source({"fasta": name})).source
    assert src.fasta == dc.FASTA_SHORTCUTS[name] and src.is_url


@pytest.mark.parametrize("url", ["https://example.org/a.fasta.gz", "http://example.org/a.fasta",
                                 "ftp://ftp.example.org/pub/a.fasta.gz"])
def test_urls_are_downloaded(url):
    src = _parse(_with_source({"fasta": url})).source
    assert src.is_url and src.paths["output_dir"] == "output/data/pretrain_data"
    got = src.paths["compressed_fasta"] or src.paths["extracted_fasta"]
    assert got == f"output/data/pretrain_data/downloads/{url.rsplit('/', 1)[1]}"


def test_local_files_are_read_in_place(tmp_path):
    (tmp_path / "a.fasta").write_text(">a\nMKV\n")
    src = _parse(_with_source({"fasta": "a.fasta"}), tmp_path).source
    assert not src.is_url and src.paths["extracted_fasta"] == "a.fasta"
    assert src.paths["compressed_fasta"] is None
    src = _parse(_with_source({"fasta": "/data/b.fasta.gz"})).source
    assert src.paths["compressed_fasta"] == "/data/b.fasta.gz"
    assert src.paths["extracted_fasta"] == "output/data/pretrain_data/src/b.fasta"


def test_a_directory_is_not_a_fasta(tmp_path):
    (tmp_path / "seqs.fasta").mkdir()
    with pytest.raises(dc.DataConfigError, match="sources.src.fasta is a directory"):
        _parse(_with_source({"fasta": "seqs.fasta"}), tmp_path)


def test_overrides_are_keyed_where_they_are_set():
    src = _parse(_with_source({"fasta": "swissprot", "max_seq_len": 1024, "val_ratio": 0.2})).source
    assert src.params["max_seq_len"] == 1024 and src.params["min_seq_len"] == 20
    assert src.key("max_seq_len") == "sources.src.max_seq_len"
    assert src.key("min_seq_len") == "data_params.min_seq_len"
    assert src.overrides == ("max_seq_len", "val_ratio")


# ------------------------------------------------------------------ max_seqs / seqs_num


def test_seqs_num_is_the_old_name_of_max_seqs():
    cfg = _template()
    cfg["data_params"]["seqs_num"] = cfg["data_params"].pop("max_seqs")
    parsed = _parse(cfg)
    assert parsed.source.params["max_seqs"] == 20000
    assert parsed.source.key("max_seqs") == "data_params.seqs_num"     # what DVC reads
    assert any("data_params.seqs_num is now called max_seqs" in w for w in parsed.warnings)
    with pytest.raises(dc.DataConfigError, match="set max_seqs or seqs_num, not both"):
        cfg["data_params"]["max_seqs"] = 5
        _parse(cfg)


def test_seqs_num_works_as_a_source_override_too():
    parsed = _parse(_with_source({"fasta": "swissprot", "seqs_num": -1}))
    assert parsed.source.params["max_seqs"] == -1
    assert parsed.source.key("max_seqs") == "sources.src.seqs_num"


# ------------------------------------------------------------------ weight / sources


@pytest.mark.parametrize("w", [1, 1.0, 1 - 1e-9])
def test_one_source_weight_is_1(w):
    assert _parse(_with_source({"fasta": "swissprot", "weight": w})).source.weight == pytest.approx(1)


@pytest.mark.parametrize("w, match", [(0.5, "the weight must be 1"), (100, "the weight must be 1"),
                                      ("1", "must be a number"), (True, "must be a number")])
def test_a_wrong_weight_is_refused(w, match):
    with pytest.raises(dc.DataConfigError, match=match):
        _parse(_with_source({"fasta": "swissprot", "weight": w}))


def test_weight_may_be_left_out():
    assert _parse(_with_source({"fasta": "swissprot"})).source.weight is None


def test_several_sources_are_not_supported_yet():
    cfg = _template()
    cfg["sources"]["trembl"] = {"fasta": "trembl"}
    with pytest.raises(dc.DataConfigError,
                       match=r"mixing several sources is not supported yet \(one source only\)"):
        _parse(cfg)


@pytest.mark.parametrize("cfg_edit, match", [
    (lambda c: c.pop("sources"), "sources needs one entry"),
    (lambda c: c.update(sources={}), "sources needs one entry"),
    (lambda c: c.pop("output_dir"), "output_dir is required"),
    (lambda c: c.update(sources={"src": {}}), "sources.src needs fasta or prepared"),
    (lambda c: c.update(sources={"src": {"fasta": "swissprot", "prepared": "x"}}),
     "set fasta .* or prepared .*, not both"),
    (lambda c: c.update(sources={"src": {"fasta": "x.txt"}}), "must end with .fasta or .fasta.gz"),
    (lambda c: c.update(sources={"train": {"fasta": "swissprot"}}), "train is a folder name"),
    (lambda c: c.update(sources={"my.src": {"fasta": "swissprot"}}), "letters, digits and _"),
    (lambda c: c.update(sources={"cluster_subsampling": {"members": None}}),
     "cluster_subsampling is not a source name; indent it under one"),
    (lambda c: c["data_params"].pop("val_ratio"), "data_params.val_ratio is required"),
])
def test_layout_errors(cfg_edit, match):
    cfg = _template()
    cfg_edit(cfg)
    with pytest.raises(dc.DataConfigError, match=match):
        _parse(cfg)


# ------------------------------------------------------------------ unknown keys


@pytest.mark.parametrize("cfg_edit, match", [
    (lambda c: c.update(outptu_dir="x"), "data.yaml: unknown key outptu_dir"),
    (lambda c: c["data_params"].update(max_seq_length=5), "data_params: unknown key max_seq_length"),
    (lambda c: c["sources"]["swissprot"].update(max_seq_length=5),
     "sources.swissprot: unknown key max_seq_length"),
    (lambda c: c["sources"]["swissprot"].update(cluster_subsampling={"members": None, "x": 1}),
     "sources.swissprot.cluster_subsampling: unknown key x"),
])
def test_unknown_keys_at_every_level_are_errors(cfg_edit, match):
    cfg = _template()
    cfg_edit(cfg)
    with pytest.raises(dc.DataConfigError, match=match):
        _parse(cfg)


def test_duplicate_keys_are_errors(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(textwrap.dedent("""\
        output_dir: out
        sources:
          a:
            fasta: swissprot
          a:
            fasta: trembl
        """))
    with pytest.raises(dc.DataConfigError, match=r"data.yaml:5: duplicate key 'a' \(first on line 3\)"):
        dc.load_strict(p)
    p.write_text("data_params:\n  max_seqs: 1\n  max_seqs: 2\n")
    with pytest.raises(dc.DataConfigError, match="duplicate key 'max_seqs'"):
        dc.load_strict(p)
    p.write_text(load_template("data_pretrain"))
    assert dc.load_strict(p) == _template()


# ------------------------------------------------------------------ cluster block


def test_the_block_being_present_turns_it_on():
    src = _parse(_with_source({"fasta": "uniref50",
                               "cluster_subsampling": {"members": "uniref90"}})).source
    assert src.cluster == {"members": "uniref90"}


@pytest.mark.parametrize("block, match", [
    (None, "cluster_subsampling is empty. It needs members"),          # bare `key:` / all commented
    ({}, "cluster_subsampling needs members"),
    ({"clustering": "auto"}, "cluster_subsampling needs members"),
    ({"members": None, "enabled": True},
     "remove enabled; the block being present turns it on"),
    ({"enabled": False}, "remove enabled; the block being present turns it on"),
    ("yes", "must be a mapping"),
])
def test_cluster_block_errors(block, match):
    with pytest.raises(dc.DataConfigError, match=match):
        _parse(_with_source({"fasta": "uniref50", "cluster_subsampling": block}))


def test_members_null_means_this_fasta():
    src = _parse(_with_source({"fasta": "uniref90",
                               "cluster_subsampling": {"members": None}})).source
    assert src.cluster == {"members": None}


def test_a_top_level_cluster_block_points_at_its_new_place():
    cfg = _template()
    cfg["cluster_subsampling"] = {"enabled": True, "members": "uniref90"}
    with pytest.raises(dc.DataConfigError, match="sources.<name>.cluster_subsampling"):
        _parse(cfg)


# ------------------------------------------------------------------ prepared


def test_prepared_takes_no_build_settings():
    src = _parse(_with_source({"prepared": "../built"})).source
    assert src.prepared == "../built" and src.fasta is None
    with pytest.raises(dc.DataConfigError, match="prepared source is used as it was built; "
                                                  "remove max_seq_len"):
        _parse(_with_source({"prepared": "../built", "max_seq_len": 5}))
    with pytest.raises(dc.DataConfigError, match="remove cluster_subsampling"):
        _parse(_with_source({"prepared": "../built", "cluster_subsampling": {"members": None}}))


# ------------------------------------------------------------------ old layout


OLD = {
    "data_dirs": {
        "url": SPROT, "local_fasta": None,
        "compressed_fasta": "output/data/raw/sequences.fasta.gz",
        "extracted_fasta": "output/data/raw/sequences.fasta",
        "validation_dir": "output/data/validation",
        "shuffled_fasta": "output/data/raw/sequences_shuffled.fasta",
        "filtered_fasta": "output/data/filter/sequences_filtered.fasta",
        "splitted_fasta_dir": "output/data/split", "output_dir": "output/data/pretrain_data",
    },
    "data_params": {"seqs_num": 20000, "min_seq_len": 20, "max_seq_len": 512, "val_ratio": 0.1,
                    "shuffle": True, "shuffle_seed": 24, "drop_invalid_sequences": False,
                    "samples_per_shard": 2000, "max_workers": 2, "register_tokens": 0},
}


def test_the_old_layout_is_source_main():
    cfg = _parse(copy.deepcopy(OLD))
    assert cfg.legacy and list(cfg.sources) == ["main"]
    src = cfg.source
    assert src.fasta == SPROT and src.params["max_seqs"] == 20000
    assert src.ref("max_seqs") == "${data_params.seqs_num}"
    assert src.path("extracted_fasta") == "${data_dirs.extracted_fasta}"
    hint = cfg.warnings[0]
    assert "data_dirs is the old data.yaml layout" in hint and 'source named "main"' in hint
    assert ('output_dir: "output/data/pretrain_data"\nsources:\n  main:\n    fasta: swissprot'
            in hint)
    assert 'To skip the download, set fasta: "output/data/raw/sequences.fasta.gz"' in hint


def test_the_old_layout_local_fasta_hint():
    old = copy.deepcopy(OLD)
    old["data_dirs"].update(url=None, local_fasta="/data/mine.fasta")
    hint = _parse(old).warnings[0]
    assert "    fasta: '/data/mine.fasta'" in hint and "skip the download" not in hint


@pytest.mark.parametrize("edit, match", [
    (lambda c: c.update(cluster_subsampling={"enabled": False}),
     "cluster_subsampling now sits under its source"),
    (lambda c: c.update(sources={}), "data_dirs is the old layout; use output_dir"),
    (lambda c: c["data_dirs"].update(urll="x"), "data_dirs: unknown key urll"),
    (lambda c: c["data_params"].update(seqs_numm=1), "data_params: unknown key seqs_numm"),
])
def test_old_layout_errors(edit, match):
    cfg = copy.deepcopy(OLD)
    edit(cfg)
    with pytest.raises(dc.DataConfigError, match=match):
        _parse(cfg)


def test_old_layout_keys_that_older_templates_shipped_are_ignored():
    old = copy.deepcopy(OLD)
    old["data_params"].update(pipeline_backend="fast", device="auto")
    cfg = _parse(old)
    assert any("data_params.pipeline_backend is no longer used" in w for w in cfg.warnings)
    assert any("data_params.device is no longer used" in w for w in cfg.warnings)
