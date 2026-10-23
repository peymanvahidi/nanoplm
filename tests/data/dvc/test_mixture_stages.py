"""dvc.yaml and `nanoplm data from-yaml` for a data.yaml with several sources (a mix)."""
from __future__ import annotations

import shutil

import pytest
import yaml

from nanoplm.cli._dvc_yaml_gen import _NANOPLM_PKG_DIR, generate_dvc_yaml
from nanoplm.cli.templates import load_template
from nanoplm.data.preparation.data_config import DataConfigError

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
    # the weights are a param of the mixture stage only
    for name, stage in st.items():
        stage = stage.get("do", stage)
        if name != "mixture":
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


def test_a_prepared_source_has_no_stages_and_feeds_the_mixture():
    st = _stages(_mix(own={"prepared": "../built/", "weight": 0.4},
                      swissprot={"fasta": "swissprot", "weight": 0.6}))
    assert not any(n.startswith("own-") for n in st)
    assert st["mixture"]["deps"][0] == "../built/.data_manifest"
    assert "--source own ../built ${sources.own.weight} -" in st["mixture"]["cmd"]


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
        "clus": {"fasta": str(rel.fasta90), "weight": 0.7, "val_ratio": 0.05,
                 "cluster_subsampling": {"members": None}},
    }
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


def test_from_yaml_builds_a_two_source_mix_and_a_weight_change_reruns_only_the_mix(tmp_path):
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

    _set_weights(tmp_path, plain=0.6, clus=0.4)
    (tmp_path / "dvc.yaml").write_text(generate_dvc_yaml(
        yaml.safe_load((tmp_path / "data.yaml").read_text()), "pretrain", "data.yaml",
        base_dir=tmp_path))
    assert _status(tmp_path, env) == {"mixture"}
    r = _from_yaml(tmp_path, env)
    assert "Rebuild: mixture." in r.stdout
    assert r.stdout.count("Running stage") == 1 and "Running stage 'mixture'" in r.stdout
    assert [s["weight"] for s in read_manifest(out).sources] == [0.6, 0.4]

    _set_weights(tmp_path, plain=0.6, clus=0.5)
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    res = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
    assert res.exit_code != 0
    assert "0.6 (plain) + 0.5 (clus) = 1.1" in res.output


def test_from_yaml_mixes_a_prepared_source_with_a_built_one(tmp_path):
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.datasets.mixture_sources import plain_source
    built = plain_source(tmp_path / "elsewhere/built", 60, tag=5)
    proj = tmp_path / "proj"
    proj.mkdir()
    env = _project(proj, None, "x.fasta")
    (proj / "local.fasta").write_text("".join(f">s{i}\n{'MKV' * (8 + i % 9)}\n"
                                              for i in range(80)))
    cfg = yaml.safe_load((proj / "data.yaml").read_text())
    cfg["output_dir"] = "out"
    cfg["sources"] = {"mine": {"prepared": "../elsewhere/built", "weight": 0.25},
                      "local": {"fasta": "local.fasta", "weight": 0.75}}
    (proj / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    r = _from_yaml(proj, env)
    assert "../elsewhere/built (prepared)" in r.stdout
    assert "Running stage 'mine-" not in r.stdout
    m = read_manifest(proj / "out")
    assert m.sources[0]["dir"] == str(built.resolve()) and m.sources[0]["fasta"] is None
    assert _status(proj, env) == set()
    plain_source(built, 61, tag=5)                    # the prepared folder is rebuilt
    assert _status(proj, env) == {"mixture"}


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


def test_a_prepared_source_with_other_register_tokens_fails_before_dvc(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    from tests.data.datasets.mixture_sources import plain_source
    plain_source(tmp_path / "built", 20, tag=5, register_tokens=1)
    cfg = _mix(mine={"prepared": "built", "weight": 0.5},
               swissprot={"fasta": "swissprot", "weight": 0.5})
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("DVC must not run"))
    r = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
    assert r.exit_code != 0
    assert "the sources of a mix must use the same register_tokens; got mine 1, swissprot 0" \
        in r.output


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


def _built(folder, manifest=True):
    for n in ("train", "val"):
        (folder / n).mkdir(parents=True)
    if manifest:
        (folder / ".data_manifest").write_text("x")


def test_a_mix_names_the_train_and_val_an_earlier_single_source_left(tmp_path, capsys):
    from nanoplm.cli.data import _warn_orphaned_outputs, orphaned_outputs
    from nanoplm.data.preparation.data_config import parse_data_config
    out = tmp_path / "out"
    _built(out)
    (out / "cluster_index").mkdir()
    (out / ".gitignore").write_text("/train\n/val\n/.data_manifest\n")
    cfg = parse_data_config(_mix(a={"fasta": "swissprot", "weight": 0.5},
                                 b={"fasta": "trembl", "weight": 0.5}))
    assert orphaned_outputs(cfg, tmp_path) == [out / "train", out / "val", out / "cluster_index"]
    _warn_orphaned_outputs(cfg, tmp_path)
    err = capsys.readouterr().err
    assert "earlier build with one source" in err and f"  {out / 'train'}\n" in err
    assert f"their lines in {out / '.gitignore'}" in err
    assert ".data_manifest" not in err.replace(".gitignore", "")   # the mixture's own
    for n in ("train", "val", "cluster_index"):
        shutil.rmtree(out / n)
    assert orphaned_outputs(cfg, tmp_path) == []


def test_one_source_names_the_folders_an_earlier_mix_left(tmp_path):
    from nanoplm.cli.data import orphaned_outputs
    from nanoplm.data.preparation.data_config import parse_data_config
    out = tmp_path / "out"
    _built(out / "a")
    _built(out / "b")
    (out / "a/split").mkdir()                 # an intermediate: one source keeps it
    (out / "downloads").mkdir()
    cfg = yaml.safe_load(load_template("data_pretrain"))
    cfg["output_dir"] = "out"
    cfg["sources"] = {"a": {"fasta": "swissprot"}}
    cfg = parse_data_config(cfg)
    got = orphaned_outputs(cfg, tmp_path)
    assert got == [out / "a/train", out / "a/val", out / "a/.data_manifest", out / "b"]
    shutil.rmtree(out / "b")
    for p in got[:3]:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    assert orphaned_outputs(cfg, tmp_path) == []


def test_a_layout_switch_names_the_extract_and_check_it_moved(tmp_path):
    from nanoplm.cli.data import orphaned_outputs
    from nanoplm.data.preparation.data_config import parse_data_config
    out = tmp_path / "out"
    for p in ("a/validation", "b/validation", "c/validation", "downloads/prot_fasta_validation"):
        (out / p).mkdir(parents=True)
    for p in ("a/prot.fasta", "b/prot.fasta", "downloads/prot.fasta"):
        (out / p).write_text(">x\nMK\n")
    a = {"fasta": "prot.fasta.gz", "weight": 0.4}
    b = {"fasta": "prot.fasta.gz", "weight": 0.4, "drop_invalid_sequences": True}
    c = {"fasta": "small.fasta", "weight": 0.2}
    mix = parse_data_config(_mix(a=a, b=b, c=c))
    # b keeps its own check, c (a local FASTA) its own check and no extract
    assert orphaned_outputs(mix, tmp_path) == [out / "a/prot.fasta", out / "a/validation",
                                               out / "b/prot.fasta"]
    one = yaml.safe_load(load_template("data_pretrain"))
    one["output_dir"] = "out"
    one["sources"] = {"a": {"fasta": "prot.fasta.gz"}}
    assert orphaned_outputs(parse_data_config(one), tmp_path) == [
        out / "downloads/prot.fasta", out / "downloads/prot_fasta_validation"]
