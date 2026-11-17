"""A stage given -o <directory> names its output after the input, and never the input
itself: x.fa, x.fasta.gz and x.fa.zst inputs included."""
import importlib

import pytest
from click.testing import CliRunner

from nanoplm.cli.data import data
from nanoplm.data.preparation.data_config import fasta_stem

FASTA = ">a\nMAGVKLLW\n>b\nMKV\n"


@pytest.mark.parametrize("name, stem", [
    ("x.fasta", "x"), ("x.fa", "x"), ("x.fa.zst", "x"), ("x.fasta.gz", "x"),
    ("x.y.fa", "x.y"), ("x_filtered.fasta", "x_filtered"), ("x", "x")])
def test_fasta_stem(name, stem):
    assert fasta_stem(name) == stem


@pytest.mark.parametrize("name", ["x.fa", "x.fasta"])
def test_filter_into_the_input_directory_keeps_the_input(tmp_path, name):
    src = tmp_path / name
    src.write_text(FASTA)
    res = CliRunner().invoke(data, ["filter", "-i", str(src), "-o", str(tmp_path),
                                    "--min-seq-len", "4"])
    assert res.exit_code == 0, res.output
    assert src.read_text() == FASTA
    assert (tmp_path / "x_filtered.fasta").read_text() == ">a\nMAGVKLLW\n"


def test_shuffle_into_the_input_directory_keeps_the_input(tmp_path):
    src = tmp_path / "x.fa"
    src.write_text(FASTA)
    res = CliRunner().invoke(data, ["shuffle", "-i", str(src), "-o", str(tmp_path),
                                    "--seed", "1"])
    assert res.exit_code == 0, res.output
    assert src.read_text() == FASTA
    assert (tmp_path / "x_shuffled.fasta").exists()


@pytest.mark.parametrize("name", ["x.fa", "x.fasta"])
def test_kd_embed_names_the_h5_after_the_input(tmp_path, monkeypatch, name):
    """Without the teacher: the stubs record the output path kd-embed derives."""
    got = {}

    class FakeKD:
        def __init__(self, **kw):
            got.update(kw)

        def process_dataset(self):
            return got["output_path"]

    cli = importlib.import_module("nanoplm.cli.data")    # nanoplm.cli.data is the group
    monkeypatch.setattr(cli, "ProtT5", lambda: None)
    monkeypatch.setattr(cli, "SaveKDDataset", FakeKD)
    src = tmp_path / name
    src.write_text(FASTA)
    res = CliRunner().invoke(data, ["kd-embed", "-i", str(src), "-o", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert got["output_path"] == tmp_path / "x_kd_dataset.h5"
    assert src.read_text() == FASTA


def test_a_stage_refuses_an_output_that_is_its_input(tmp_path):
    src = tmp_path / "x.fa"
    src.write_text(FASTA)
    for args in (["filter", "-i", str(src), "-o", str(src)],
                 ["get-file", "--local-fasta", str(src), "-o", str(tmp_path), "--force"]):
        res = CliRunner().invoke(data, args)
        assert res.exit_code != 0 and "is the input file" in res.output, res.output
        assert src.read_text() == FASTA


def test_extract_refuses_a_plain_fasta_before_touching_it(tmp_path):
    """extract -i dir/x.fa -o dir --force used to unlink dir/x.fa: the output it derives
    from a name without .gz / .zst is the input itself."""
    src = tmp_path / "x.fa"
    src.write_text(FASTA)
    res = CliRunner().invoke(data, ["extract", "-i", str(src), "-o", str(tmp_path), "--force"])
    assert res.exit_code != 0 and "must be a .gz or .zst file" in res.output, res.output
    assert src.read_text() == FASTA


def test_a_symlinked_output_from_an_earlier_run_is_not_the_input(tmp_path):
    """get-file links a local FASTA; a rerun replaces the link, not the file."""
    src = tmp_path / "src" / "x.fa"
    src.parent.mkdir()
    src.write_text(FASTA)
    out = tmp_path / "out" / "x.fa"
    for _ in range(2):
        res = CliRunner().invoke(data, ["get-file", "--local-fasta", str(src), "-o", str(out),
                                        "--force"])
        assert res.exit_code == 0, res.output
    assert out.read_text() == FASTA and src.read_text() == FASTA
