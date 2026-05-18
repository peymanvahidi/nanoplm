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
  local_fasta: "/tmp/x.fasta.gz"
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


def _make_cfg(tmp_path, **overrides):
    """Build a minimal data.yaml. Pass overrides for data_dirs fields."""
    import yaml as _yaml
    data_dirs = {
        "url": None,
        "local_fasta": None,
        "compressed_fasta": "out/raw/x.fasta.gz",
        "extracted_fasta": "out/raw/x.fasta",
        "validation_dir": "out/val",
        "shuffled_fasta": "out/s.fasta",
        "filtered_fasta": "out/f.fasta",
        "splitted_fasta_dir": "out/split",
        "output_dir": "out/pretrain",
    }
    data_dirs.update(overrides)
    cfg = {
        "data_dirs": data_dirs,
        "data_params": {
            "seqs_num": 100, "min_seq_len": 1, "max_seq_len": 100,
            "val_ratio": 0.1, "shuffle": True, "shuffle_seed": 24,
            "drop_invalid_sequences": False,
            "samples_per_shard": 10, "max_workers": 1, "register_tokens": 0,
        },
    }
    p = tmp_path / "data.yaml"
    p.write_text(_yaml.safe_dump(cfg))
    return p


def test_from_yaml_rejects_bad_local_fasta_extension(tmp_path):
    cfg = _make_cfg(tmp_path, local_fasta="data.txt")
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg)])
    assert result.exit_code != 0
    assert ".fasta" in result.output


def test_from_yaml_rejects_bad_url_extension(tmp_path):
    cfg = _make_cfg(tmp_path, url="https://example.com/data.zip")
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg)])
    assert result.exit_code != 0
    assert ".fasta" in result.output


def test_from_yaml_requires_compressed_fasta_in_gz_mode(tmp_path):
    cfg = _make_cfg(tmp_path, url="https://example.com/x.fasta.gz",
                    compressed_fasta=None)
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg)])
    assert result.exit_code != 0
    assert "compressed_fasta" in result.output


def test_from_yaml_requires_extracted_fasta(tmp_path):
    cfg = _make_cfg(tmp_path, url="https://example.com/x.fasta.gz",
                    extracted_fasta=None)
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg)])
    assert result.exit_code != 0
    assert "extracted_fasta" in result.output


def test_from_yaml_warns_when_compressed_fasta_set_in_plain_mode(tmp_path, monkeypatch):
    """Plain .fasta source + compressed_fasta set => succeeds with warning."""
    cfg = _make_cfg(tmp_path,
                    local_fasta=str(tmp_path / "x.fasta"),
                    compressed_fasta="out/raw/x.fasta.gz")
    (tmp_path / "x.fasta").write_text(">a\nMAGV\n")
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.chdir(tmp_path)

    # Spy on the nanoplm logger (caplog doesn't work because propagate=False)
    warnings_seen = []
    from nanoplm.utils.logger import logger as _nplm_logger
    monkeypatch.setattr(_nplm_logger, "warning",
                        lambda msg, *a, **k: warnings_seen.append(msg))

    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert any("unused" in w for w in warnings_seen)


def test_from_yaml_plain_local_omits_compressed_fasta_ok(tmp_path, monkeypatch):
    """Plain .fasta source with compressed_fasta missing => succeeds (it's optional here)."""
    cfg = _make_cfg(tmp_path,
                    local_fasta=str(tmp_path / "x.fasta"),
                    compressed_fasta=None)
    (tmp_path / "x.fasta").write_text(">a\nMAGV\n")
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output


def test_from_yaml_plain_local_dryrun_emits_pipeline_without_extract(tmp_path, monkeypatch):
    """Smoke: from-yaml in plain-.fasta mode produces a dvc.yaml that:
      - has no `extract` stage,
      - has all downstream stages,
      - has get_file writing extracted_fasta directly.
    """
    import yaml as _yaml
    # Materialize a plain-fasta local file
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">a\nMAGVAGM\n>b\nVVAGM\n")
    cfg = _make_cfg(
        tmp_path,
        url=None,
        local_fasta=str(fasta),
        compressed_fasta=None,
    )
    # Stub DVC subprocess — we only need from-yaml to write dvc.yaml.
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": ""})())
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(data, ["from-yaml", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output

    dvc = _yaml.safe_load((tmp_path / "dvc.yaml").read_text())
    stages = dvc["stages"]
    assert "extract" not in stages
    for required in ("get_file", "check", "shuffle", "filter", "split",
                     "pre-tokenize", "write_manifest"):
        assert required in stages, f"missing stage: {required}"
    assert stages["get_file"]["outs"] == ["${data_dirs.extracted_fasta}"]
    assert "${data_dirs.local_fasta}" in stages["get_file"]["deps"]
