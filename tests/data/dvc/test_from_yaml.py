# tests/data/dvc/test_from_yaml.py
import yaml
from pathlib import Path
from click.testing import CliRunner
from nanoplm.cli.data import data


def test_from_yaml_errors_on_legacy_pipeline_mode(tmp_path):
    cfg = tmp_path / "data.yaml"
    cfg.write_text("data_params:\n  pipeline_mode: pretrain\n")
    (tmp_path / "dvc.yaml").write_text("stages: {}\n")
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg)])
    assert result.exit_code != 0
    assert "pipeline_mode" in result.output


def test_from_yaml_errors_when_both_url_and_local_fasta(tmp_path):
    cfg_text = """\
data_dirs:
  url: "https://x.com/a.gz"
  local_fasta: "/tmp/x.gz"
  compressed_fasta: "out.gz"
  extracted_fasta: "out.fasta"
  validation_dir: "val"
  shuffled_fasta: "s.fasta"
  filtered_fasta: "f.fasta"
  splitted_fasta_dir: "split"
  output_dir: "out"
data_params:
  seqs_num: 100
  min_seq_len: 1
  max_seq_len: 100
  val_ratio: 0.1
  shuffle: true
  shuffle_seed: 24
  drop_invalid_sequences: false
  samples_per_shard: 10
  max_workers: 1
  register_tokens: 0
"""
    cfg = tmp_path / "data.yaml"
    cfg.write_text(cfg_text)
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg)])
    assert result.exit_code != 0
    assert "url" in result.output.lower() or "local_fasta" in result.output.lower()


def test_from_yaml_regenerates_dvc_yaml(tmp_path, monkeypatch):
    """from-yaml should rewrite dvc.yaml to match the current data.yaml."""
    from nanoplm.cli._templates import PRETRAIN_DATA_YAML
    cfg = tmp_path / "data.yaml"
    cfg.write_text(PRETRAIN_DATA_YAML)
    # Stub out dvc subprocess
    monkeypatch.setattr("subprocess.run", lambda *a, **k: type("R", (), {"returncode": 0})())
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg), "--dry-run"])
    assert (tmp_path / "dvc.yaml").exists()
    parsed = yaml.safe_load((tmp_path / "dvc.yaml").read_text())
    assert "pre-tokenize" in parsed["stages"]


def test_from_yaml_summary_uses_yaml_manifest(tmp_path, monkeypatch):
    """_print_summary should successfully read the YAML-formatted manifest."""
    import yaml as _yaml
    from nanoplm.cli._templates import PRETRAIN_DATA_YAML

    # Materialize config and override output_dir to tmp_path
    cfg_dict = _yaml.safe_load(PRETRAIN_DATA_YAML)
    cfg_dict["data_dirs"]["output_dir"] = str(tmp_path / "pretrain_out")
    cfg = tmp_path / "data.yaml"
    cfg.write_text(_yaml.safe_dump(cfg_dict))

    # Pre-create the manifest as the real write-manifest stage would
    out_dir = Path(cfg_dict["data_dirs"]["output_dir"])
    out_dir.mkdir(parents=True)
    manifest_text = (
        "# This file is AUTO-GENERATED\n"
        "pipeline_mode: pretrain\n"
        "seqs_num: 42\n"
        "min_seq_len: 20\n"
        "max_seq_len: 512\n"
        "val_ratio: 0.1\n"
        "train_dir: train\n"
        "val_dir: val\n"
        "train_sequences: 38\n"
        "val_sequences: 4\n"
        "created_at: '2026-01-01T00:00:00'\n"
        "sharded: true\n"
        "samples_per_shard: 2000\n"
    )
    (out_dir / ".data_manifest").write_text(manifest_text)

    monkeypatch.setattr("subprocess.run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg)])
    assert result.exit_code == 0, result.output
    # Summary must include the real numbers, not a fallback message
    assert "Total sequences: 42" in result.output
    assert "Train:         38" in result.output
    assert "Val:           4" in result.output


def test_from_yaml_fix_preserves_comments(tmp_path):
    """--fix must not destroy YAML comments in the user's data.yaml."""
    from nanoplm.cli._templates import PRETRAIN_DATA_YAML
    cfg = tmp_path / "data.yaml"
    cfg.write_text(PRETRAIN_DATA_YAML)
    # PRETRAIN_DATA_YAML has commented-out alt-URL lines starting with "# swissprot"
    assert "# swissprot:" in cfg.read_text()

    # Stub dvc subprocess
    import subprocess as _sp
    orig_run = _sp.run

    def _fake_run(*args, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": ""})()

    import unittest.mock as _mock
    with _mock.patch("subprocess.run", _fake_run):
        runner = CliRunner()
        result = runner.invoke(data, ["from-yaml", str(cfg), "--fix", "--dry-run"])
    assert result.exit_code == 0, result.output
    # Comments preserved + key flipped
    after = cfg.read_text()
    assert "# swissprot:" in after
    assert "drop_invalid_sequences: true" in after
