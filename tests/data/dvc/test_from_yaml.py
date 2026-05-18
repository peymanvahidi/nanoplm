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
