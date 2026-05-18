"""Distillation pipeline accepts a PretrainManifest for on-the-fly KD."""
from pathlib import Path
import pytest
from nanoplm.data.infra.manifest import PretrainManifest, write_manifest


def _write_pretrain_manifest(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = PretrainManifest(
        pipeline_mode="pretrain",
        seqs_num=100,
        min_seq_len=20,
        max_seq_len=512,
        val_ratio=0.1,
        train_dir="train",
        val_dir="val",
        train_sequences=90,
        val_sequences=10,
        sharded=True,
        samples_per_shard=50,
    )
    write_manifest(dataset_dir, manifest)


def test_resolve_config_uses_pretrain_manifest_for_on_the_fly_kd(tmp_path):
    from nanoplm.distillation.pipeline import (
        DistillationConfig, _resolve_config_from_manifest,
    )
    _write_pretrain_manifest(tmp_path)

    # Stand up a tiny FASTA pair the on-the-fly path needs.
    train_fasta = tmp_path / "train.fasta"
    val_fasta = tmp_path / "val.fasta"
    train_fasta.write_text(">a\nMAGV\n")
    val_fasta.write_text(">b\nVVAGM\n")

    cfg = DistillationConfig(
        dataset_dir=str(tmp_path),
        train_fasta=str(train_fasta),
        val_fasta=str(val_fasta),
    )
    out = _resolve_config_from_manifest(cfg)
    assert out.on_the_fly is True
    assert out.max_seq_len == 512
    assert out.train_sequences == 90
    assert out.val_sequences == 10


def test_resolve_config_errors_when_pretrain_manifest_missing_fasta_paths(tmp_path):
    """Pretrain manifest doesn't record FASTA paths — caller MUST supply them."""
    from nanoplm.distillation.pipeline import (
        DistillationConfig, _resolve_config_from_manifest,
    )
    _write_pretrain_manifest(tmp_path)

    cfg = DistillationConfig(dataset_dir=str(tmp_path))  # no train_fasta/val_fasta

    with pytest.raises(ValueError, match="train_fasta"):
        _resolve_config_from_manifest(cfg)
