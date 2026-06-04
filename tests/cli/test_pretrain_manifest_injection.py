"""Verify that `nanoplm pretrain from-yaml` injects manifest.num_register_tokens
into model_config before constructing the model, so the resulting tokenizer
appends the right number of register tokens."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from click.testing import CliRunner

from nanoplm.cli.pretrain import pretrain


def _make_pretrain_shard_dir(d: Path, sequence_counts):
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(sequence_counts):
        # validate_pretrain_shards enforces sum(idx_sizes) == len(.bin).
        # Use zero-length sequence sizes and a zero-byte .bin so the fixture
        # passes shard validation; the pipeline itself is patched out below.
        (d / f"shard_{i:04d}.bin").write_bytes(b"")
        np.save(d / f"shard_{i:04d}.idx.npy", np.zeros(n, dtype=np.int32))


def _write_manifest_with_register_tokens(dataset_dir: Path, n: int) -> None:
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
        "num_register_tokens": n,
        "sharded": True,
        "samples_per_shard": 10,
    }
    (dataset_dir / ".data_manifest").write_text(yaml.dump(manifest))


def test_from_yaml_injects_register_tokens_from_manifest(tmp_path):
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    _make_pretrain_shard_dir(dataset_dir / "train", [20])
    _make_pretrain_shard_dir(dataset_dir / "val", [10])
    _write_manifest_with_register_tokens(dataset_dir, n=2)

    yaml_cfg = {
        "pure_torch": True,
        "model": {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
        },
        "pretraining": {
            "dataset_dir": str(dataset_dir),
            "ckp_dir": str(tmp_path / "ckpt"),
            "micro_batch_seqs": 2,
            "global_batch_tokens": 2,
            "num_epochs": 1,
        },
        "resume": {"is_resume": False, "checkpoint_dir": ""},
    }
    cfg_path = tmp_path / "pretrain.yaml"
    cfg_path.write_text(yaml.dump(yaml_cfg))

    seen = {}

    def fake_run(model, pretrain_config, manifest, resume_config=None):
        seen["num_register_tokens_in_cfg"] = model.model_config.num_register_tokens
        seen["tokenizer_n"] = model.tokenizer.num_register_tokens
        seen["manifest_n"] = manifest.num_register_tokens

    with patch("nanoplm.cli.pretrain.run_pure_pretraining", side_effect=fake_run):
        runner = CliRunner()
        result = runner.invoke(pretrain, ["from-yaml", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert seen == {
        "num_register_tokens_in_cfg": 2,
        "tokenizer_n": 2,
        "manifest_n": 2,
    }
