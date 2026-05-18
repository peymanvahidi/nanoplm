# tests/data/dvc/test_get_yaml.py
from pathlib import Path
from click.testing import CliRunner
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
