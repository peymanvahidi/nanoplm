"""Verify that the flag-based `nanoplm pretrain run` command is fully wired:

1. Every `run()` signature parameter has a matching `@click.option` (a missing
   option makes Click invoke the callback without that kwarg -> TypeError). The
   tests INVOKE the command (not just `--help`) so such mismatches are caught.
2. The token-budget + WSD duration flags build the expected PretrainingConfig.
3. `_validate_duration_config` is wired: an inconsistent schedule/budget combo
   exits non-zero with a clear message.

Mirrors the patching style of test_pretrain_manifest_injection.py: a real tiny
dataset + manifest is provided so dataset/manifest validation passes, while the
heavy model construction and training entry point are patched out.
"""

import numpy as np
import yaml
from click.testing import CliRunner
from unittest.mock import patch

from nanoplm.cli.pretrain import pretrain


def _make_pretrain_shard_dir(d, sequence_counts):
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(sequence_counts):
        # validate_pretrain_shards enforces sum(idx_sizes) == len(.bin).
        # Zero-length sequence sizes + a zero-byte .bin pass shard validation;
        # the pipeline itself is patched out below.
        (d / f"shard_{i:04d}.bin").write_bytes(b"")
        np.save(d / f"shard_{i:04d}.idx.npy", np.zeros(n, dtype=np.int32))


def _write_manifest(dataset_dir):
    manifest = {
        "pipeline_mode": "pretrain",
        "seqs_num": 30,
        "min_seq_len": 20,
        "max_seq_len": 64,
        "val_ratio": 0.1,
        "train_dir": "train",
        "val_dir": "val",
        "train_sequences": 20,
        "val_sequences": 10,
        "num_register_tokens": 2,
        "sharded": True,
        "samples_per_shard": 10,
    }
    (dataset_dir / ".data_manifest").write_text(yaml.dump(manifest))


def _make_dataset(tmp_path):
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    _make_pretrain_shard_dir(dataset_dir / "train", [20])
    _make_pretrain_shard_dir(dataset_dir / "val", [10])
    _write_manifest(dataset_dir)
    return dataset_dir


class _FakeModel:
    """Stand-in for PureProtModernBertMLM: cheap, no real params/layers."""

    def __init__(self, model_config):
        self.model_config = model_config

    def parameters(self):
        return iter(())

    def modules(self):
        return iter(())


def _base_args(dataset_dir, tmp_path):
    return [
        "run",
        "--pure-torch",
        "--world-size",
        "1",
        "--dataset-dir",
        str(dataset_dir),
        "--ckp-dir",
        str(tmp_path / "ckpt"),
        "--micro-batch-seqs",
        "2",
        "--global-batch-tokens",
        "2",
    ]


def test_run_wsd_builds_config_and_invokes_pipeline(tmp_path):
    dataset_dir = _make_dataset(tmp_path)

    seen = {}

    def fake_run(model, pretrain_config, manifest, resume_config=None):
        seen["cfg"] = pretrain_config
        seen["manifest_n"] = manifest.num_register_tokens

    args = _base_args(dataset_dir, tmp_path) + [
        "--lr-schedule",
        "wsd",
        "--stable-tokens",
        "9000",
        "--decay-tokens",
        "1000",
        "--decay-shape",
        "1-sqrt",
    ]

    with patch("nanoplm.cli.pretrain.run_pure_pretraining", side_effect=fake_run), patch(
        "nanoplm.cli.pretrain.PureProtModernBertMLM", _FakeModel
    ):
        runner = CliRunner()
        result = runner.invoke(pretrain, args)

    # exit_code == 0 proves NO param<->option mismatch and a valid config built.
    assert result.exit_code == 0, result.output
    cfg = seen["cfg"]
    assert cfg.lr_schedule == "wsd"
    assert cfg.stable_tokens == 9000
    assert cfg.decay_tokens == 1000
    assert cfg.decay_shape == "1-sqrt"
    assert cfg.train_tokens is None
    assert cfg.max_steps is None
    assert seen["manifest_n"] == 2


def test_run_single_phase_train_tokens(tmp_path):
    dataset_dir = _make_dataset(tmp_path)

    seen = {}

    def fake_run(model, pretrain_config, manifest, resume_config=None):
        seen["cfg"] = pretrain_config

    args = _base_args(dataset_dir, tmp_path) + [
        "--lr-schedule",
        "cosine",
        "--train-tokens",
        "50000",
    ]

    with patch("nanoplm.cli.pretrain.run_pure_pretraining", side_effect=fake_run), patch(
        "nanoplm.cli.pretrain.PureProtModernBertMLM", _FakeModel
    ):
        runner = CliRunner()
        result = runner.invoke(pretrain, args)

    assert result.exit_code == 0, result.output
    cfg = seen["cfg"]
    assert cfg.lr_schedule == "cosine"
    assert cfg.train_tokens == 50000
    assert cfg.stable_tokens is None
    assert cfg.decay_tokens is None


def test_run_wsd_missing_budget_fails(tmp_path):
    """wsd with no stable/decay budget must exit non-zero via _validate_duration_config."""
    dataset_dir = _make_dataset(tmp_path)

    args = _base_args(dataset_dir, tmp_path) + ["--lr-schedule", "wsd"]

    with patch("nanoplm.cli.pretrain.run_pure_pretraining") as run_mock, patch(
        "nanoplm.cli.pretrain.PureProtModernBertMLM", _FakeModel
    ):
        runner = CliRunner()
        result = runner.invoke(pretrain, args)

    assert result.exit_code != 0, result.output
    run_mock.assert_not_called()
    # message should reference the missing stable/decay budget
    assert "stable" in result.output.lower() or "decay" in result.output.lower()
