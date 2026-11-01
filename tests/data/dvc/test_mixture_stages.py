"""dvc.yaml and `nanoplm data from-yaml` for a data.yaml with several sources (a mix)."""
from __future__ import annotations

import json
import shutil

import pytest
import yaml

from nanoplm.cli._dvc_yaml_gen import _NANOPLM_PKG_DIR, generate_dvc_yaml
from nanoplm.cli.templates import load_template
from nanoplm.data.preparation.data_config import DataConfigError
from nanoplm.data.preparation.val_split import split_by_weight

from tests.data.dvc.test_cluster_stages import _from_yaml, _project, _status


def _mix(**sources):
    cfg = yaml.safe_load(load_template("data_pretrain"))
    cfg["output_dir"] = "out"
    cfg["sources"] = sources
    return cfg


def _stages(cfg, base_dir=None):
    text = generate_dvc_yaml(cfg, "pretrain", "data.yaml", base_dir)
    return yaml.safe_load(text.replace(str(_NANOPLM_PKG_DIR), "<PKG>"))["stages"]


def _params(stage):
    return [k for p in stage.get("params", []) for k in p["data.yaml"]]


THREE = dict(
    swissprot={"fasta": "swissprot", "weight": 0.2},
    uniref50={"fasta": "uniref50", "weight": 0.7, "max_seq_len": 1024,
              "cluster_subsampling": {"members": "uniref90"}},
    u90={"fasta": "uniref90", "weight": 0.1},
)


def test_three_sources_one_download_per_url_and_a_mixture_stage():
    st = _stages(_mix(**THREE))
    downloads = [n for n in st if n.startswith("download-")]
    assert downloads == ["download-uniprot_sprot_fasta_gz", "download-uniref90_fasta_gz",
                         "download-uniref50_xml_gz"]
    # uniref50's members and u90's fasta are the same file: fetched, extracted and
    # checked once, read twice
    assert [n for n in st if n.startswith(("extract", "check")) or "-extract" in n
            or n.endswith("-check")] == [
        "extract-uniprot_sprot_fasta", "check-uniprot_sprot_fasta",
        "extract-uniref90_fasta", "check-uniref90_fasta"]
    assert st["extract-uniref90_fasta"]["deps"][0] == "out/downloads/uniref90.fasta.gz"
    assert st["extract-uniref90_fasta"]["outs"] == ["out/downloads/uniref90.fasta"]
    assert st["check-uniref90_fasta"]["outs"] == ["out/downloads/uniref90_fasta_validation"]
    for name in ("uniref50-filter", "u90-shuffle"):
        assert st[name]["deps"][:2] == ["out/downloads/uniref90.fasta",
                                        "out/downloads/uniref90_fasta_validation"]
    assert st["download-uniref50_xml_gz"]["outs"] == [
        {"out/downloads/uniref50.xml.gz": {"persist": True, "cache": False}}]
    for name in ("swissprot", "uniref50", "u90"):
        assert st[f"{name}-write_manifest"]["outs"] == [f"out/{name}/.data_manifest"]
    assert st["uniref50-cluster_index"]["outs"] == ["out/uniref50/cluster_index/index",
                                                   "out/uniref50/val"]
    assert list(st)[-1] == "mixture"
    mix = st["mixture"]
    assert mix["outs"] == ["out/.data_manifest"]
    assert mix["deps"][:3] == [f"out/{n}/.data_manifest" for n in THREE]
    assert _params(mix) == [f"sources.{n}.weight" for n in THREE]
    assert "--source u90 out/u90 ${sources.u90.weight}" in mix["cmd"]
    # the weights are a param of the mixture and the val split only
    vsplit = st["val_split"]
    assert _params(vsplit) == ["data_params.val_ratio"] + [f"sources.{n}.weight" for n in THREE]
    assert vsplit["outs"] == [f"out/{n}/val_count.json" for n in THREE]
    assert vsplit["deps"][:3] == [f"out/{n}/pool_units.json" for n in THREE]
    assert "--members-tsv out/uniref50/cluster_index/members.tsv" in \
        st["uniref50-count"]["cmd"] and "--members-tsv" not in st["u90-count"]["cmd"]
    for name in ("swissprot-split", "uniref50-cluster_index"):
        assert f"--val-count-file out/{name.split('-')[0]}/val_count.json" in st[name]["cmd"]
        assert "val_ratio" not in st[name]["cmd"] and "val_ratio" not in str(_params(st[name]))
    for name, stage in st.items():
        stage = stage.get("do", stage)
        if name not in ("mixture", "val_split"):
            assert not any("weight" in k for k in _params(stage)), name


def test_two_urls_with_one_file_name_are_refused():
    cfg = _mix(a={"fasta": "https://x.org/a/prot.fasta.gz", "weight": 0.5},
               b={"fasta": "https://y.org/b/prot.fasta.gz", "weight": 0.5})
    with pytest.raises(DataConfigError, match="sources.a and sources.b download different "
                                              "URLs to the same file out/downloads/prot"):
        _stages(cfg)


def test_a_shared_gz_is_extracted_and_checked_once_in_any_source_order():
    a = {"fasta": "prot.fasta.gz", "weight": 0.5}
    b = {"fasta": "prot.fasta.gz", "weight": 0.4, "max_seq_len": 100}
    c = {"fasta": "small.fasta", "weight": 0.1}
    st = _stages(_mix(a=a, b=b, c=c))
    assert [n for n in st if n.startswith(("extract", "check"))] == [
        "extract-prot_fasta", "check-prot_fasta"]
    assert st["extract-prot_fasta"]["cmd"] == \
        "nanoplm data extract -i prot.fasta.gz -o out/downloads/prot.fasta"
    assert st["a-shuffle"]["deps"][:2] == st["b-shuffle"]["deps"][:2] == [
        "out/downloads/prot.fasta", "out/downloads/prot_fasta_validation"]
    assert "c-check" in st                  # a local FASTA is checked per source
    assert st["c-check"]["outs"] == ["out/c/validation"]
    flipped = _stages(_mix(c=c, b=b, a=a))
    assert {n: flipped[n] for n in ("extract-prot_fasta", "check-prot_fasta")} == \
        {n: st[n] for n in ("extract-prot_fasta", "check-prot_fasta")}


def test_a_shared_zst_is_extracted_and_checked_once_like_a_gz():
    url = "https://s3.amazonaws.com/logan-pub/p/logan50_c1.0_human_complete.fa.zst"
    st = _stages(_mix(a={"fasta": url, "weight": 0.5},
                      b={"fasta": url, "weight": 0.3, "max_seq_len": 100},
                      c={"fasta": "small.fa", "weight": 0.2}))
    assert [n for n in st if n.startswith(("download", "extract", "check"))] == [
        "download-logan50_c1_0_human_complete_fa_zst", "extract-logan50_c1_0_human_complete_fa",
        "check-logan50_c1_0_human_complete_fa"]
    assert st["extract-logan50_c1_0_human_complete_fa"]["cmd"] == (
        "nanoplm data extract -i out/downloads/logan50_c1.0_human_complete.fa.zst "
        "-o out/downloads/logan50_c1.0_human_complete.fa")
    assert st["a-shuffle"]["deps"][:2] == st["b-shuffle"]["deps"][:2] == [
        "out/downloads/logan50_c1.0_human_complete.fa",
        "out/downloads/logan50_c1_0_human_complete_fa_validation"]
    assert st["c-shuffle"]["cmd"].startswith("nanoplm data shuffle -i small.fa ")


def test_a_per_source_drop_invalid_keeps_its_own_check():
    st = _stages(_mix(a={"fasta": "prot.fasta.gz", "weight": 0.5},
                      b={"fasta": "prot.fasta.gz", "weight": 0.5,
                         "drop_invalid_sequences": True}))
    assert "check-prot_fasta" in st and st["b-check"]["outs"] == ["out/b/validation"]
    assert st["b-check"]["deps"][0] == "out/downloads/prot.fasta"


def test_two_local_gz_with_one_file_name_are_refused():
    cfg = _mix(a={"fasta": "x/prot.fasta.gz", "weight": 0.5},
               b={"fasta": "y/prot.fasta.gz", "weight": 0.5})
    with pytest.raises(DataConfigError, match="sources.a and sources.b extract different "
                                              "files to the same file out/downloads/prot"):
        _stages(cfg)


def test_the_mixture_stage_reads_every_source_manifest():
    st = _stages(_mix(own={"fasta": "x/own.fasta", "weight": 0.4},
                      swissprot={"fasta": "swissprot", "weight": 0.6}))
    assert st["mixture"]["deps"][:2] == ["out/own/.data_manifest",
                                         "out/swissprot/.data_manifest"]
    assert '--source own out/own ${sources.own.weight} "x/own.fasta"' in st["mixture"]["cmd"]


def test_a_single_source_keeps_its_stage_names():
    st = _stages(yaml.safe_load(load_template("data_pretrain")))
    assert "mixture" not in st and "get_file" in st and "write_manifest" in st


# ------------------------------------------------------------------ end to end


def _two_source_project(tmp_path):
    """plain: a local FASTA; clus: UniProt clusters over a tiny UniRef90 release (the
    XML is placed in downloads/ so nothing is fetched)."""
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=120)
    plain = tmp_path / "src/plain.fasta"
    plain.write_text(rel.fasta100.read_text().replace(">UniRef100_", ">p_"))
    env = _project(tmp_path, None, "x.fasta")
    cfg = yaml.safe_load((tmp_path / "data.yaml").read_text())
    cfg["sources"] = {
        "plain": {"fasta": str(plain), "weight": 0.3},
        "clus": {"fasta": str(rel.fasta90), "weight": 0.7,
                 "cluster_subsampling": {"members": None}},
    }
    cfg["data_params"]["val_ratio"] = 0.05
    cfg["output_dir"] = "out"
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (tmp_path / "out/downloads").mkdir(parents=True)
    shutil.copy(rel.xml, tmp_path / "out/downloads/uniref50.xml.gz")
    return env


def _set_weights(tmp_path, **w):
    cfg = yaml.safe_load((tmp_path / "data.yaml").read_text())
    for k, v in w.items():
        cfg["sources"][k]["weight"] = v
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def test_from_yaml_builds_a_two_source_mix_with_one_val_split_by_weight(tmp_path):
    from nanoplm.data.datasets import open_train_dataset
    from nanoplm.data.datasets.mixture import MixtureDataset
    from nanoplm.data.infra.manifest import read_manifest
    env = _two_source_project(tmp_path)
    r = _from_yaml(tmp_path, env)
    table = r.stdout[:r.stdout.index("Regenerated")]
    assert "data.yaml: 2 sources" in table and "weight = share of training sequences" in table
    head = next(ln for ln in table.splitlines() if ln.startswith("source"))
    assert head.split()[:3] == ["source", "weight", "fasta"]
    assert next(ln for ln in table.splitlines() if ln.startswith("clus ")).split()[1] == "0.7"
    assert "Pretraining data mixture complete (2 sources" in r.stdout
    out = tmp_path / "out"
    m = read_manifest(out)
    assert m.is_mixture and [s["name"] for s in m.sources] == ["plain", "clus"]
    rows = {s["name"]: s for s in m.sources}
    assert rows["clus"]["kind"] == "cluster" and rows["plain"]["kind"] == "plain"
    assert rows["clus"]["dir"] == "clus" and (out / "clus/cluster_index/index").is_dir()
    assert rows["plain"]["fasta"].endswith("plain.fasta")
    assert not (out / "train").exists()                 # every source has its own folder
    ds = open_train_dataset(out, m, seed=1)
    assert isinstance(ds, MixtureDataset) and ds.counts[0] == round(0.3 * len(ds))
    assert _status(tmp_path, env) == set()
    # one val split: 0.05 of the pool (plain sequences + clus clusters), shared 0.3 / 0.7
    units = {n: json.loads((out / n / "pool_units.json").read_text()) for n in rows}
    assert units["plain"]["kind"] == "sequences" and units["clus"]["kind"] == "clusters"
    pool = units["plain"]["units"] + units["clus"]["units"]
    assert rows["clus"]["train_sequences"] + rows["clus"]["val_sequences"] == \
        units["clus"]["units"]
    val = {n: rows[n]["val_sequences"] for n in rows}
    assert sum(val.values()) == round(0.05 * pool)
    assert val == split_by_weight(round(0.05 * pool), {"plain": 0.3, "clus": 0.7})

    _set_weights(tmp_path, plain=0.6, clus=0.4)
    (tmp_path / "dvc.yaml").write_text(generate_dvc_yaml(
        yaml.safe_load((tmp_path / "data.yaml").read_text()), "pretrain", "data.yaml",
        base_dir=tmp_path))
    assert _status(tmp_path, env) == {"val_split", "mixture"}
    r = _from_yaml(tmp_path, env)
    # the val counts move with the weights, so both sources split again (not the pool)
    assert "Running stage 'val_split'" in r.stdout and "Running stage 'mixture'" in r.stdout
    assert "Running stage 'clus-cluster_index'" in r.stdout
    assert "Running stage 'clus-cluster_pool'" not in r.stdout
    assert "Running stage 'plain-split'" in r.stdout and "Running stage 'plain-filter'" not in r.stdout
    assert "Running stage 'plain-pre-tokenize@train'" in r.stdout
    m = read_manifest(out)
    assert [s["weight"] for s in m.sources] == [0.6, 0.4]
    assert {s["name"]: s["val_sequences"] for s in m.sources} == split_by_weight(
        round(0.05 * pool), {"plain": 0.6, "clus": 0.4})

    _set_weights(tmp_path, plain=0.6, clus=0.5)
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    res = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
    assert res.exit_code != 0
    assert "0.6 (plain) + 0.5 (clus) = 1.1" in res.output


def test_the_table_has_a_weight_column_and_lists_a_shared_download_once(tmp_path):
    from nanoplm.cli.data import plan_table
    from nanoplm.data.preparation.cluster_config import resolve_cluster_plan
    from nanoplm.data.preparation.data_config import parse_data_config
    cfg = parse_data_config(_mix(**THREE))
    plans = {n: resolve_cluster_plan(s) for n, s in cfg.sources.items()}
    lines = plan_table(cfg, plans, tmp_path, tmp_path / "data.yaml").splitlines()
    assert lines[1] == "weight = share of training sequences"
    assert lines[2].split()[:3] == ["source", "weight", "fasta"]
    rows = {ln.split()[0]: ln for ln in lines[3:]}
    assert rows["uniref50"].split()[1] == "0.7"
    assert "uniref90.fasta.gz (url) + uniref50.xml.gz (url)" in rows["uniref50"]
    assert "uniref90.fasta.gz (same download as uniref50)" in rows["u90"]


def test_built_sources_with_other_register_tokens_fail_before_dvc(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    cfg = _mix(a={"fasta": "swissprot", "weight": 0.5},
               b={"fasta": "trembl", "weight": 0.5, "register_tokens": 1})
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("DVC must not run"))
    r = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
    assert r.exit_code != 0
    assert "the sources of a mix must use the same register_tokens; got a 0, b 1" in r.output


def _lock_from(tmp_path, cfg):
    """The dvc.lock a build of ``cfg`` leaves, and its outputs on disk."""
    from nanoplm.cli.data import _stage_graph
    graph = _stage_graph(generate_dvc_yaml(cfg, "pretrain", "data.yaml", tmp_path), cfg)
    stages = {n: {"cmd": "x", "outs": [{"path": p} for p in outs]}
              for n, (_, outs) in graph.items()}
    (tmp_path / "dvc.lock").write_text(yaml.safe_dump({"schema": "2.0", "stages": stages}))
    for _, outs in graph.values():
        for p in map(tmp_path.joinpath, outs):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch() if "." in p.name else p.mkdir(exist_ok=True)


def _orphans(tmp_path, cfg):
    from nanoplm.cli.data import orphaned_outputs
    found = orphaned_outputs(tmp_path, generate_dvc_yaml(cfg, "pretrain", "data.yaml",
                                                         tmp_path), cfg)
    return [str(p.relative_to(tmp_path / "out")) for p in found]


A = {"fasta": "swissprot", "weight": 0.5}


def _one(**src):
    return _mix(**{n: {k: v for k, v in s.items() if k != "weight"} for n, s in src.items()})


def test_a_mix_names_what_an_earlier_single_source_left(tmp_path, capsys):
    from nanoplm.cli.data import _warn_orphaned_outputs
    _lock_from(tmp_path, _one(a=A))
    (tmp_path / "out/.gitignore").write_text("/train\n/val\n/.data_manifest\n")
    mix = _mix(a=A, b={"fasta": "trembl", "weight": 0.5})
    got = _orphans(tmp_path, mix)
    # one source writes the dataset into output_dir and extracts the .gz in its own
    # folder; a mix uses output_dir/a/ and downloads/
    assert sorted(got) == ["a/uniprot_sprot.fasta", "a/validation", "train", "val"]
    _warn_orphaned_outputs(tmp_path, generate_dvc_yaml(mix, "pretrain", "data.yaml"), mix)
    err = capsys.readouterr().err
    assert f"  {tmp_path / 'out/train'}\n" in err and ".data_manifest" not in err.replace(
        ".gitignore", "")                               # the mixture's own
    assert f"their lines in {tmp_path / 'out/.gitignore'}" in err
    assert "delete them and run `dvc gc --workspace`" in err
    for p in got:
        (tmp_path / "out" / p).rmdir() if (tmp_path / "out" / p).is_dir() else \
            (tmp_path / "out" / p).unlink()
    assert _orphans(tmp_path, mix) == []               # listed, never deleted


def test_one_source_names_what_an_earlier_mix_left(tmp_path):
    _lock_from(tmp_path, _mix(a=A, b={"fasta": "trembl", "weight": 0.5}))
    got = _orphans(tmp_path, _one(a=A))
    assert [p for p in got if p.startswith("b/")] and "b" not in got
    assert {"a/train", "a/val", "a/.data_manifest", "downloads/uniprot_sprot.fasta",
            "downloads/uniprot_sprot_fasta_validation"} <= set(got)
    assert not any(p.startswith("downloads/uniprot_sprot.fasta.gz") for p in got)


def test_removing_or_renaming_a_source_names_its_outputs(tmp_path):
    three = _mix(a={"fasta": "swissprot", "weight": 0.4}, b={"fasta": "trembl", "weight": 0.3},
                 c={"fasta": "local.fasta", "weight": 0.3})
    _lock_from(tmp_path, three)
    two = _mix(a={"fasta": "swissprot", "weight": 0.5}, b={"fasta": "trembl", "weight": 0.5})
    got = _orphans(tmp_path, two)
    assert got and all(p.startswith("c/") for p in got) and "c/train" in got
    renamed = _mix(a={"fasta": "swissprot", "weight": 0.5}, d={"fasta": "trembl", "weight": 0.5})
    got = _orphans(tmp_path, renamed)
    assert "b/train" in got and not any(p.startswith(("a/", "downloads/")) for p in got)


def test_dropping_a_cluster_block_names_its_index_and_the_xml(tmp_path):
    u = {"fasta": "uniref50", "weight": 0.5, "max_seqs": -1,
         "cluster_subsampling": {"members": "uniref90"}}
    _lock_from(tmp_path, _mix(a=A, u=u))
    assert _orphans(tmp_path, _mix(a=A, u=u)) == []
    off = _mix(a=A, u={"fasta": "uniref50", "weight": 0.5})
    got = _orphans(tmp_path, off)
    assert "downloads/uniref50.xml.gz" in got and "u/cluster_index/index" in got
    assert "downloads/uniref90.fasta.gz" in got         # the members download, unused now
    assert (tmp_path / "out/downloads/uniref50.xml.gz").exists()


def test_no_dvc_lock_no_orphans(tmp_path):
    (tmp_path / "out/b/train").mkdir(parents=True)
    assert _orphans(tmp_path, _mix(a=A, c={"fasta": "trembl", "weight": 0.5})) == []


def test_the_rebuild_line_of_a_mix_is_one_summary_per_source():
    from nanoplm.cli.data import _rebuild_summary
    names = ["download-uniref50_xml_gz", "swissprot-check", "swissprot-filter",
             "uniref50-check", "uniref50-filter", "uniref50-cluster_index", "mixture"]
    changed = {"uniref50-filter", "uniref50-cluster_index", "mixture"}
    assert _rebuild_summary(names, changed, ["swissprot", "uniref50"]) == (
        "Rebuild: downloads: up to date; swissprot: up to date; uniref50: 2 of 3 stages may "
        "rerun; mixture: may rerun. (DVC skips a stage its run cache already has.)")


# ------------------------------------------------------------------ one val split


def test_one_source_with_val_size_uses_the_val_split_stages():
    cfg = _one(a=A)
    del cfg["data_params"]["val_ratio"]
    cfg["data_params"]["val_size"] = 500
    st = _stages(cfg)
    assert st["val_split"]["cmd"] == ("nanoplm data val-split --val-size "
                                      "${data_params.val_size} --source a "
                                      "out/a/pool_units.json 1 out/a/val_count.json")
    assert st["count"]["outs"] == ["out/a/pool_units.json"]
    assert "--val-count-file out/a/val_count.json" in st["split"]["cmd"]
    assert "--val-ratio" not in st["write_manifest"]["cmd"]
    assert "val_ratio" not in yaml.safe_dump(st)


def test_one_source_with_val_ratio_has_no_val_split_stages():
    st = _stages(_one(a=A))
    assert "val_split" not in st and "count" not in st
    assert "--val-ratio ${data_params.val_ratio}" in st["split"]["cmd"]


def test_split_and_val_split_commands_hit_the_count(tmp_path):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    fasta = tmp_path / "f.fasta"
    fasta.write_text("".join(f">s{i}\nMKV\n" for i in range(97)))
    run = CliRunner().invoke
    r = run(data, ["count-units", "-i", str(fasta), "-o", str(tmp_path / "u.json")])
    assert r.exit_code == 0, r.output
    r = run(data, ["val-split", "--val-size", "13", "--source", "a", str(tmp_path / "u.json"),
                   "1", str(tmp_path / "v.json")])
    assert r.exit_code == 0, r.output
    assert json.loads((tmp_path / "v.json").read_text()) == {
        "val": 13, "units": 97, "kind": "sequences"}
    r = run(data, ["split", "-i", str(fasta), "-o", str(tmp_path / "split"),
                   "--val-count-file", str(tmp_path / "v.json")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "split/val.fasta").read_text().count(">") == 13
    r = run(data, ["val-split", "--val-size", "97", "--source", "a", str(tmp_path / "u.json"),
                   "1", str(tmp_path / "v.json")])
    assert r.exit_code != 0
    assert "sources.a would need 97 val sequences but has 97; lower data_params.val_size" \
        in r.output


def test_from_yaml_one_cluster_source_with_val_size(tmp_path):
    """val_size on one cluster source: the count and val_split stages, then exactly that
    many held-out clusters."""
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.preparation.uniref_fixture import make_release
    rel = make_release(tmp_path / "src", n_clusters=120)
    env = _project(tmp_path, {"members": None}, str(rel.fasta90), val_size=7)
    cfg = yaml.safe_load((tmp_path / "data.yaml").read_text())
    del cfg["data_params"]["val_ratio"]
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = tmp_path / cfg["output_dir"]
    (out / "downloads").mkdir(parents=True)
    shutil.copy(rel.xml, out / "downloads/uniref50.xml.gz")
    r = _from_yaml(tmp_path, env)
    assert "val = data_params.val_size: 7 sequences / clusters, shared out by weight" in r.stdout
    m = read_manifest(out)
    assert m.val_sequences == 7 and m.is_cluster_sampled
    units = json.loads((out / "src/pool_units.json").read_text())
    assert units["kind"] == "clusters" and m.train_sequences + 7 == units["units"]
    assert m.val_ratio == round(7 / units["units"], 6)
