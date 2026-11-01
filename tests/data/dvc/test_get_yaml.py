# tests/data/dvc/test_get_yaml.py
from pathlib import Path
from click.testing import CliRunner
import pytest
from nanoplm.cli.data import data


def test_get_yaml_default_emits_data_yaml_and_dvc_yaml(tmp_path):
    runner = CliRunner()
    result = runner.invoke(data, ["get-yaml", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data.yaml").exists()
    assert (tmp_path / "dvc.yaml").exists()
    assert not (tmp_path / "distillation_data.yaml").exists()
    content = (tmp_path / "data.yaml").read_text()
    assert "register_tokens" in content
    assert "teacher_model" not in content


def test_get_yaml_distillation_emits_distillation_data_yaml(tmp_path):
    runner = CliRunner()
    result = runner.invoke(data, ["get-yaml", str(tmp_path), "--distillation"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "distillation_data.yaml").exists()
    assert (tmp_path / "dvc.yaml").exists()
    assert not (tmp_path / "data.yaml").exists()
    content = (tmp_path / "distillation_data.yaml").read_text()
    assert "teacher_model" in content
    assert "register_tokens" not in content


def test_get_yaml_errors_when_data_yaml_exists(tmp_path):
    (tmp_path / "data.yaml").write_text("existing")
    runner = CliRunner()
    result = runner.invoke(data, ["get-yaml", str(tmp_path)])
    assert result.exit_code != 0


def test_get_yaml_force_overwrites(tmp_path):
    (tmp_path / "data.yaml").write_text("existing")
    runner = CliRunner()
    result = runner.invoke(data, ["get-yaml", str(tmp_path), "--force"])
    assert result.exit_code == 0


def test_get_yaml_subsampling_emits_the_cluster_template(tmp_path):
    from nanoplm.cli.templates import load_template
    result = CliRunner().invoke(data, ["get-yaml", str(tmp_path), "--subsampling"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data.yaml").read_text() == load_template("data_pretrain_subsampling")
    dvc = (tmp_path / "dvc.yaml").read_text()
    assert "cluster_members" in dvc and "get_cluster_xml" in dvc
    assert not (tmp_path / "distillation_data.yaml").exists()


def test_get_yaml_subsampling_and_distillation_are_exclusive(tmp_path):
    result = CliRunner().invoke(data, ["get-yaml", str(tmp_path), "--subsampling",
                                       "--distillation"])
    assert result.exit_code != 0
    assert "--subsampling and --distillation cannot be used together" in result.output
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["data_pretrain", "data_pretrain_subsampling", "data_distill"])
def test_every_data_template_parses_and_generates_a_dvc_yaml(tmp_path, name):
    """Each template as get-yaml writes it goes through the checks from-yaml runs before
    DVC, and gives a dvc.yaml; no network (the UniRef plan is decided by name)."""
    import yaml
    from nanoplm.cli._dvc_yaml_gen import generate_dvc_yaml
    from nanoplm.cli.templates import load_template
    text = load_template(name)
    if name == "data_distill":
        from nanoplm.cli.data import _validate_pipeline_fields, _validate_source_fields
        cfg = yaml.safe_load(text)
        _validate_source_fields(cfg["data_dirs"])
        _validate_pipeline_fields(cfg["data_dirs"], cfg["data_params"], "distillation",
                                  tmp_path / "distillation_data.yaml")
        stages = yaml.safe_load(generate_dvc_yaml(cfg, "distillation",
                                                  "distillation_data.yaml"))["stages"]
        assert "kd-embed" in stages
        return
    from nanoplm.cli.data import _validate_sources
    from nanoplm.data.preparation.data_config import load_strict, parse_data_config
    path = tmp_path / "data.yaml"
    path.write_text(text)
    raw = load_strict(path)
    cfg = parse_data_config(raw, tmp_path, path.name)
    assert cfg.warnings == []
    plans = _validate_sources(cfg, tmp_path, path)
    stages = yaml.safe_load(generate_dvc_yaml(raw, "pretrain", "data.yaml", tmp_path))["stages"]
    assert "write_manifest" in stages
    if name == "data_pretrain_subsampling":
        plan = plans["uniref50"]
        assert plan.source_line() == ("clusters: UniProt uniref50.xml.gz | "
                                      "trains on: UniRef90 members")
        assert "cluster_index" in stages and "split" not in stages
    else:
        assert plans == {"swissprot": None} and "split" in stages


def test_template_doc_links_point_at_real_pages():
    import re
    from pathlib import Path
    from nanoplm.cli.templates import load_template
    docs = Path(__file__).resolve().parents[3] / "website/src/content/docs"
    for name in ("data_pretrain", "data_pretrain_subsampling"):
        links = re.findall(r"https://peymanvahidi\.github\.io/nanoplm/(\S+?)/?(?:\s|$)",
                           load_template(name))
        assert links, name
        for slug in links:
            assert any((docs / f"{slug}{ext}").is_file() for ext in (".md", ".mdx")), slug
