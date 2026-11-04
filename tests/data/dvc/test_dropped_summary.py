"""Invalid records are dropped with a warning by default; the from-yaml summary counts them."""
from __future__ import annotations

import logging

import yaml
from click.testing import CliRunner

from nanoplm.cli._dvc_yaml_gen import generate_dvc_yaml
from nanoplm.cli.data import _dropped_lines, data
from nanoplm.cli.templates import load_template
from nanoplm.data.preparation.checker import Checker


def _logged(fn):
    """Run fn, return the warnings it logged to the nanoplm logger (it does not propagate)."""
    msgs = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda rec: msgs.append(rec.getMessage())
    logging.getLogger("nanoplm").addHandler(handler)
    try:
        fn()
    finally:
        logging.getLogger("nanoplm").removeHandler(handler)
    return msgs


def test_check_drops_by_default_and_warns_with_the_first_ids(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">ok\nMAGV\n>b1\nMAG!\n>b2\n\n>b3\nMA*GV\n>b4\nMA1\n")
    out = tmp_path / "validation"
    msgs = _logged(lambda: Checker(src, out).check())
    assert (out / "invalid_ids.txt").read_text().split() == ["b1", "b2", "b3", "b4"]
    assert msgs == [f"check: dropping 4 records with characters outside the amino-acid "
                    f"alphabet or empty (first: b1, b2, b3); see {out / 'invalid.fasta'}"]



def test_one_dropped_record_is_singular(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">ok\nMAGV\n>b1\nMAG!\n")
    out = tmp_path / "validation"
    msgs = _logged(lambda: Checker(src, out).check())
    assert msgs[0].startswith("check: dropping 1 record with characters outside")

def test_check_warns_about_trailing_stop_codons(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">a\nMAGV*\n>b\nMAGV\n>c\nMA*\n>d\nGV*\n>e\nMK*\n>f\nMAG**\n")
    msgs = _logged(lambda: Checker(src, tmp_path / "validation").check())
    assert msgs[0] == ("check: removing the trailing '*' (stop codon) from 4 records "
                       "(first: a, c, d)")
    assert msgs[1].startswith("check: dropping 1 record ")      # "**" is invalid


def test_one_stop_codon_is_singular_and_none_is_silent(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">a\nMAGV*\n")
    msgs = _logged(lambda: Checker(src, tmp_path / "v1").check())
    assert msgs == ["check: removing the trailing '*' (stop codon) from 1 record (first: a)"]
    src.write_text(">a\nMAGV\n")
    assert _logged(lambda: Checker(src, tmp_path / "v2").check()) == []


def test_the_check_command_drops_by_default_and_false_still_stops(tmp_path):
    src = tmp_path / "in.fasta"
    src.write_text(">ok\nMAGV\n>bad\nMAG!\n")
    r = CliRunner().invoke(data, ["check", "-i", str(src), "-o", str(tmp_path / "v1")])
    assert r.exit_code == 0 and "Validation found" not in r.output
    assert (tmp_path / "v1" / "invalid_ids.txt").read_text().split() == ["bad"]
    r = CliRunner().invoke(data, ["check", "-i", str(src), "-o", str(tmp_path / "v2"),
                                  "--drop-invalid", "false"])
    assert r.exit_code != 0 and "from-yaml --fix" in r.output
    assert (tmp_path / "v2" / "invalid_ids.txt").read_text() == ""


def test_the_template_default_is_a_value_not_a_param():
    cfg = yaml.safe_load(load_template("data_pretrain"))
    st = yaml.safe_load(generate_dvc_yaml(cfg, "pretrain", "data.yaml"))["stages"]
    assert st["check"]["cmd"].endswith("--drop-invalid true")
    cfg["data_params"]["drop_invalid_sequences"] = False
    st = yaml.safe_load(generate_dvc_yaml(cfg, "pretrain", "data.yaml"))["stages"]
    assert st["check"]["cmd"].endswith("--drop-invalid ${data_params.drop_invalid_sequences}")


def test_the_summary_says_how_many_were_dropped(tmp_path, monkeypatch):
    cfg = yaml.safe_load(load_template("data_pretrain"))
    cfg["output_dir"] = "out"
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    out.mkdir()
    (out / ".data_manifest").write_text("seqs_num: 42\ntrain_sequences: 38\n"
                                        "val_sequences: 4\n")
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".dvc").mkdir()

    def run():
        r = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml")])
        assert r.exit_code == 0, r.output
        return r.output

    assert "Dropped" not in run()                   # no validation folder yet
    check_out = yaml.safe_load((tmp_path / "dvc.yaml").read_text())["stages"]["check"]["outs"][0]
    val = tmp_path / check_out
    val.mkdir(parents=True)
    (val / "invalid_ids.txt").write_text("")
    assert "Dropped" not in run()                   # nothing dropped
    (val / "invalid_ids.txt").write_text("a\nb\nc\n")
    assert f"  Dropped:         3 invalid records, listed in {val / 'invalid.fasta'}\n" in run()
    (val / "validation.json").write_text('{"trailing_stop_count": 0}')
    assert "Stop codons" not in run()
    (val / "validation.json").write_text('{"trailing_stop_count": 1234}')
    assert "  Stop codons:     1,234 trailing '*' removed\n" in run()


def test_a_mix_says_it_per_source_and_a_shared_check_for_each_reader(tmp_path):
    cfg = yaml.safe_load(load_template("data_pretrain"))
    cfg["output_dir"] = "out"
    cfg["sources"] = {
        "swissprot": {"fasta": "swissprot", "weight": 0.2},
        "uniref50": {"fasta": "uniref50", "weight": 0.7,
                     "cluster_subsampling": {"members": "uniref90", "clusters": "uniprot"}},
        "u90": {"fasta": "uniref90", "weight": 0.1},
    }
    (tmp_path / "dvc.yaml").write_text(generate_dvc_yaml(cfg, "pretrain", "data.yaml"))
    for name, ids in (("uniprot_sprot_fasta", ""), ("uniref90_fasta", "x\ny\n")):
        val = tmp_path / "out" / "downloads" / f"{name}_validation"
        val.mkdir(parents=True)
        (val / "invalid_ids.txt").write_text(ids)
    (tmp_path / "out/downloads/uniprot_sprot_fasta_validation/validation.json").write_text(
        '{"trailing_stop_count": 5}')
    lines = _dropped_lines(tmp_path, cfg, list(cfg["sources"])).splitlines()
    shared = tmp_path / "out/downloads/uniref90_fasta_validation/invalid.fasta"
    assert lines == ["  Stop codons (swissprot): 5 trailing '*' removed",
                     f"  Dropped (uniref50):      2 invalid records, listed in {shared}",
                     f"  Dropped (u90):           2 invalid records, listed in {shared}"]
