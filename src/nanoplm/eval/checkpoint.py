"""Checkpoint resolution and loading for downstream evaluation."""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml

from nanoplm.data.infra.atomic import atomic_write_path
from nanoplm.pretraining.fp8 import convert_fp8_targets
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.modeling import check_mhc_checkpoint
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.utils import checkpoint_step as _checkpoint_step
from nanoplm.utils.common import git_sha
from nanoplm.utils.logger import logger


def _safe_load_checkpoint_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def _read_run_manifest(run_root: Path) -> Optional[dict[str, Any]]:
    """Read the training run's ``run.json``.

    A run predating run.json left none, so absence is normal rather than an
    error. A manifest that is present but unreadable is not, so it warns.
    """
    manifest_path = run_root / "run.json"
    if not manifest_path.exists():
        return None
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"Ignoring unreadable run manifest {manifest_path}: {exc}")
        return None
    return loaded if isinstance(loaded, dict) else None


def _resolved_step(checkpoint_dir: Path) -> Optional[int]:
    """The one place ``checkpoint_step``'s ``-1`` sentinel becomes ``None``.

    A sidecar reader would take ``-1`` for a step.
    """
    step = _checkpoint_step(checkpoint_dir)
    return None if step < 0 else step


def default_embedder_name(checkpoint_dir: Union[str, Path]) -> str:
    """Name a checkpoint's eval after its run and step.

    ``checkpoint-27500`` is the same name under every run, so the bare
    directory stem collides between runs evaluated into one ``output_dir``.
    ``<run>-ckpt<step>`` is the convention the cluster-experiments skill
    documents and every eval config under the gitignored ``experiments/`` types
    by hand. A step no name carries falls back to the checkpoint's role, so a
    ``checkpoint-final`` with no readable ``training_state.json`` gives
    ``<run>-final``.
    """
    path = Path(checkpoint_dir).resolve()
    step = _resolved_step(path)
    suffix = (
        f"ckpt{step}" if step is not None else path.name.removeprefix("checkpoint-")
    )
    return f"{path.parent.name}-{suffix}"


@dataclass(frozen=True)
class EvalProvenance:
    """Which artifact an autoeval report came from.

    biotrainer's ``AutoEvalReport`` carries only ``embedder_name`` and a
    ``training_date`` stamped with the eval date, so the report names neither
    the checkpoint, the step, nor the training run. This is the record that
    does.

    ``run_name`` is the directory the checkpoint sits in, which still names the
    run when ``run.json`` is absent, so it can disagree with
    ``training_run["run_name"]`` after the run directory was moved.

    ``biotrainer_version`` is what decides whether a later run may reuse the
    results beside this record. It is ``None`` in sidecars written before the
    field existed.
    """

    embedder_name: str
    checkpoint_dir: Path
    checkpoint_step: Optional[int]
    run_name: str
    loop_count: Optional[int]
    eval_git_sha: Optional[str]
    eval_started_at: str
    training_run: Optional[dict[str, Any]]
    biotrainer_version: Optional[str] = None

    @classmethod
    def resolve(
        cls,
        *,
        checkpoint_dir: Union[str, Path],
        embedder_name: Optional[str],
        loop_count: Optional[int],
        biotrainer_version: Optional[str] = None,
    ) -> "EvalProvenance":
        checkpoint_dir = Path(checkpoint_dir).resolve()
        return cls(
            embedder_name=embedder_name or default_embedder_name(checkpoint_dir),
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=_resolved_step(checkpoint_dir),
            run_name=checkpoint_dir.parent.name,
            loop_count=loop_count,
            eval_git_sha=git_sha(Path(__file__).resolve().parent),
            eval_started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            training_run=_read_run_manifest(checkpoint_dir.parent),
            biotrainer_version=biotrainer_version,
        )


def eval_provenance_path(report_path: Union[str, Path]) -> Path:
    """The sidecar path for a report: its stem plus ``.provenance.json``."""
    report_path = Path(report_path)
    return report_path.with_name(
        report_path.name.removesuffix(".json") + ".provenance.json"
    )


def write_eval_provenance(
    provenance: EvalProvenance, report_path: Union[str, Path]
) -> Optional[Path]:
    """Write the provenance record beside the report it annotates.

    It shares the report's stem so a reader holding a report path finds it by
    appending one suffix.
    """
    sidecar_path = eval_provenance_path(report_path)
    record = asdict(provenance) | {"checkpoint_dir": str(provenance.checkpoint_dir)}
    try:
        with atomic_write_path(sidecar_path) as tmp:
            tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not write eval provenance to {sidecar_path}: {exc}")
        return None
    return sidecar_path


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

    # Checkpoints written before use_huginn_looping existed encode "recycling is
    # on" solely as recycling_loop_count > 1. Infer the flag so those configs
    # still validate under the now-explicit switch.
    if "use_huginn_looping" not in m:
        kwargs["use_huginn_looping"] = int(m.get("recycling_loop_count", 1) or 1) > 1

    return ProtModernBertMLMConfig(**kwargs)


def resolve_checkpoint_dir(checkpoint_dir: Union[str, Path]) -> Path:
    """Resolve a checkpoint-* directory or a run-* directory to a concrete checkpoint path."""
    checkpoint_arg = Path(checkpoint_dir)

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
        "Checkpoint directory must be a checkpoint-* directory or a run-* "
        f"directory containing one. Got: {checkpoint_dir}"
    )


def _pretrain_config_from_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    pretrain_cfg_path = checkpoint_dir / "pretrain_config.yaml"
    if not pretrain_cfg_path.exists():
        return {}
    loaded = _safe_load_checkpoint_yaml(pretrain_cfg_path)
    return loaded if isinstance(loaded, dict) else {}


def _apply_training_mlp_layout(
    model: PureProtModernBertMLM,
    pretrain_cfg: dict[str, Any],
    device: torch.device,
) -> None:
    """Match FP8 TE MLP module layout saved during training."""
    use_fp8 = bool(pretrain_cfg.get("fp8", False))
    if not use_fp8:
        return
    if device.type != "cuda":
        raise ValueError(
            "FP8 checkpoints require CUDA for eval "
            "(TE MLP layout is built on GPU)."
        )

    model.to(device)
    fp8_target = pretrain_cfg.get("fp8_target", "mlp")
    fp8_skip_last = int(pretrain_cfg.get("fp8_skip_last_n_layers", 0))
    n_converted = convert_fp8_targets(
        model,
        target=fp8_target,
        skip_last_n_layers=fp8_skip_last,
    )
    if n_converted == 0:
        raise ValueError(
            "Checkpoint has fp8=True but no FP8 module blocks were converted for eval."
        )
    logger.info(
        f"FP8 eval: converted {n_converted} block(s) to TE layout "
        f"(target={fp8_target!r}, skip_last_n_layers={fp8_skip_last})"
    )


def _cast_sonicmoe_experts_to_bf16(
    model: PureProtModernBertMLM,
    config: ProtModernBertMLMConfig,
) -> None:
    """Give the sonicmoe expert GEMMs operands their kernel accepts.

    quack's SM90 grouped GEMM only takes 16- or 8-bit operands, and eval holds
    the model in fp32 and leans on autocast, which never reaches a custom op.
    Training multiplied these same weights in bf16 under FSDP's bf16 policy, so
    bf16 is their trained compute dtype; casting only ``Wi``/``Wo`` leaves the
    rest of the graph at the precision every other checkpoint is evaluated at.
    """
    if str(getattr(config, "moe_kernel_backend", "cutlass")).lower() != "sonicmoe":
        return

    from nanoplm.pretraining.models.modern_bert.moe import MoELayer

    n_cast = 0
    for module in model.modules():
        if isinstance(module, MoELayer):
            module.Wi.data = module.Wi.data.to(torch.bfloat16)
            module.Wo.data = module.Wo.data.to(torch.bfloat16)
            n_cast += 1
    logger.info(f"sonicmoe eval: cast Wi/Wo of {n_cast} MoE layer(s) to bfloat16")


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
    pretrain_cfg = _pretrain_config_from_checkpoint(checkpoint_dir)
    _apply_training_mlp_layout(model, pretrain_cfg, device)
    check_mhc_checkpoint(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    if device.type == "cuda" and next(model.parameters()).device.type != "cuda":
        model.to(device)
    _cast_sonicmoe_experts_to_bf16(model, config)
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
