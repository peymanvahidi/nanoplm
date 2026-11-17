# tests/data/preparation/test_check_fix_flow.py
from pathlib import Path
from click.testing import CliRunner
from nanoplm.cli.data import data


def test_fix_flag_flips_drop_invalid_in_config(tmp_path, monkeypatch):
    """from-yaml --fix should set drop_invalid_sequences=true in data.yaml."""
    from nanoplm.cli.templates import load_template
    cfg = tmp_path / "data.yaml"
    cfg.write_text(load_template("data_pretrain").replace(
        "drop_invalid_sequences: true", "drop_invalid_sequences: false"))
    assert "drop_invalid_sequences: true" not in cfg.read_text()
    # Stub dvc — return object must expose .returncode AND .stdout (used by is_git_subdir)
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(data, ["from-yaml", str(cfg), "--fix", "--dry-run"])
    new_text = cfg.read_text()
    assert "drop_invalid_sequences: true" in new_text
