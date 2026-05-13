"""Checkpoint resolution and loading for downstream evaluation."""

from __future__ import annotations

import json
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.utils.logger import logger


def _safe_load_checkpoint_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_config(yaml_path: Union[str, Path] = "pretrain.yaml") -> ProtModernBertMLMConfig:
    """Build a ProtModernBertMLMConfig from a pretrain.yaml's ``model:`` block."""
    cfg = _safe_load_checkpoint_yaml(Path(yaml_path))
    return _model_config_from_dict(cfg["model"])


def load_model_config_from_checkpoint(
    checkpoint_dir: Union[str, Path],
) -> Optional[ProtModernBertMLMConfig]:
    """Try to load model_config.yaml from a checkpoint directory.

    Returns ``None`` if the file does not exist (older checkpoints).
    """
    cfg_path = Path(checkpoint_dir) / "model_config.yaml"
    if not cfg_path.exists():
        return None
    return _model_config_from_dict(_safe_load_checkpoint_yaml(cfg_path))


def _model_config_from_dict(m: dict) -> ProtModernBertMLMConfig:
    """Construct a ProtModernBertMLMConfig from a flat dict of model fields."""
    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []

    for field in fields(ProtModernBertMLMConfig):
        if not field.init:
            continue
        if field.name in m:
            kwargs[field.name] = m[field.name]
        elif field.default is not MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:
            kwargs[field.name] = field.default_factory()
        else:
            missing_required.append(field.name)

    if missing_required:
        raise KeyError(
            "Missing required model config field(s): "
            + ", ".join(sorted(missing_required))
        )

    return ProtModernBertMLMConfig(**kwargs)


def _checkpoint_step(checkpoint_dir: Union[str, Path]) -> int:
    name = Path(checkpoint_dir).name
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.split("-", 1)[1])
    except ValueError:
        return -1


def _find_latest_checkpoint(checkpoint_root: Union[str, Path]) -> Path:
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    checkpoints = [
        path for path in checkpoint_root.glob("run-*/checkpoint-*")
        if path.is_dir()
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint-* directories found under {checkpoint_root}"
        )

    return max(
        checkpoints,
        key=lambda path: (path.stat().st_mtime, _checkpoint_step(path), str(path)),
    )


def resolve_checkpoint_dir(
    checkpoint_dir: Union[str, Path],
    checkpoint_root: Union[str, Path] = "output/pretraining_checkpoints",
) -> Path:
    """Resolve 'latest', run-* directories, or explicit checkpoint paths."""
    checkpoint_arg = Path(checkpoint_dir)

    if str(checkpoint_dir).strip().lower() == "latest":
        return _find_latest_checkpoint(checkpoint_root).resolve()

    if checkpoint_arg.is_dir() and checkpoint_arg.name.startswith("checkpoint-"):
        return checkpoint_arg.resolve()

    if checkpoint_arg.is_dir():
        checkpoints = [
            path for path in checkpoint_arg.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ]
        if checkpoints:
            return max(
                checkpoints,
                key=lambda path: (_checkpoint_step(path), path.stat().st_mtime, str(path)),
            ).resolve()

    raise FileNotFoundError(
        "Checkpoint directory must be a checkpoint-* directory, a run-* directory, "
        f"or the keyword 'latest'. Got: {checkpoint_dir}"
    )


def load_checkpoint(
    checkpoint_dir: Union[str, Path],
    config: ProtModernBertMLMConfig,
    device: torch.device,
) -> PureProtModernBertMLM:
    """Instantiate PureProtModernBertMLM and load checkpoint weights."""
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "pytorch_model.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    logger.info(f"Loading pure-torch checkpoint from {model_path} ...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    model = PureProtModernBertMLM(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    training_state_path = checkpoint_dir / "training_state.json"
    checkpoint_step = 0
    checkpoint_state: dict[str, Any] = {}
    if training_state_path.exists():
        checkpoint_state = json.loads(training_state_path.read_text(encoding="utf-8"))
        checkpoint_step = int(checkpoint_state.get("global_step", 0))

    # Restore step-dependent runtime state that is intentionally not persisted
    # in the model weights.
    if getattr(config, "use_prores", False) and hasattr(model.model, "update_prores_alphas"):
        model.model.update_prores_alphas(checkpoint_step)
        logger.info(f"  ProRes restored for global_step={checkpoint_step}")

    if config.use_repo:
        repo_warmup_steps = 0
        pretrain_cfg_path = checkpoint_dir / "pretrain_config.yaml"
        if pretrain_cfg_path.exists():
            pretrain_cfg = _safe_load_checkpoint_yaml(pretrain_cfg_path) or {}
            repo_warmup_steps = int(
                pretrain_cfg.get(
                    "repo_rope_warmup_steps",
                    pretrain_cfg.get("warmup_steps", 0),
                )
            )
        elif training_state_path.exists():
            repo_warmup_steps = int(checkpoint_state.get("warmup_steps", 0))

        model.model.repo_active = checkpoint_step >= repo_warmup_steps
        state = "activated" if model.model.repo_active else "inactive"
        logger.info(
            "  RePO "
            f"{state} (step={checkpoint_step}, repo_rope_warmup_steps={repo_warmup_steps})"
        )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded: {n_params:.1f}M parameters on {device}")
    return model
