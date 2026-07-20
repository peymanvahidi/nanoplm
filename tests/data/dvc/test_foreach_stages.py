# tests/data/dvc/test_foreach_stages.py
import subprocess
import yaml
from pathlib import Path
from click.testing import CliRunner
from nanoplm.cli.data import data


def test_dvc_dag_contains_foreach_variants(tmp_path, monkeypatch):
    """Verify the regenerated dvc.yaml has pre-tokenize@train / @val."""
    from nanoplm.cli.templates import load_template
    cfg = tmp_path / "data.yaml"
    cfg.write_text(load_template("data_pretrain"))
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(data, ["from-yaml", str(cfg), "--dry-run"])
    dvc = yaml.safe_load((tmp_path / "dvc.yaml").read_text())
    pretokenize = dvc["stages"]["pre-tokenize"]
    assert "foreach" in pretokenize
    assert set(pretokenize["foreach"].keys()) == {"train", "val"}
