"""Downstream evaluation pipeline: runs biotrainer autoeval on a pure-torch checkpoint."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch

from nanoplm.eval.checkpoint import (
    load_checkpoint,
    load_model_config_from_checkpoint,
    resolve_checkpoint_dir,
)
from nanoplm.eval.config import EvalConfig
from nanoplm.eval.zero_shot import NanoPLMBioEngineer
from nanoplm.utils.logger import logger

# nanoPLM framework key -> biotrainer's AutoEvalFramework.get_name().
FRAMEWORK_REPORT_NAMES = {
    "pgym": "PGYM",
    "pbc_zeroshot_contact": "PBC_ZEROSHOT_CONTACT",
}

# Frameworks that existed before the biotrainer 2.0.0 migration. Kept only to
# produce an actionable error instead of a bare "unsupported framework".
REMOVED_FRAMEWORKS = {"pbc", "pbc_supervised", "flip"}


def _load_biotrainer_symbols() -> dict[str, Any]:
    """Import the biotrainer 2.0.0 symbols the eval pipeline depends on."""
    try:
        from biotrainer.autoeval import AutoEval
        from biotrainer.bioengineer import BioEngineer, CustomBioEngineerModel
        from biotrainer.bioengineer.bioengineer_custom_model import (
            CustomBioEngineerModelWrapper,
        )
        from biotrainer.bioengineer.bioengineer_utils import get_optimal_window
        from biotrainer_core.data_classes import ZeroShotMethod
        from biotrainer_core.data_classes.autoeval import AutoEvalReport
        from biotrainer_core.utils.constants import STANDARD_AAS
    except ImportError as e:
        raise ImportError(
            "biotrainer 2.0.x is required for nanoplm eval but is not installed "
            f"or has an incompatible API. Install with: pip install nanoplm[eval]. "
            f"Original error: {e}"
        ) from e

    return {
        "AutoEval": AutoEval,
        "AutoEvalReport": AutoEvalReport,
        "BioEngineer": BioEngineer,
        "CustomBioEngineerModel": CustomBioEngineerModel,
        "CustomBioEngineerModelWrapper": CustomBioEngineerModelWrapper,
        "get_optimal_window": get_optimal_window,
        "ZeroShotMethod": ZeroShotMethod,
        "STANDARD_AAS": STANDARD_AAS,
    }


def _resolve_device(device_override: Optional[str]) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _disable_h100_triton_srelu() -> None:
    """Disable fused Triton SReLU before model construction.

    The fused kernel can OOM on H100 in eval-only flows; plain PyTorch ops are
    sufficient for inference and avoid the failure.
    """
    import nanoplm.pretraining.models.modern_bert.modeling as _bert_modeling
    _bert_modeling.USE_TRITON_SRELU = False


def _resolve_zero_shot_method(name: str, biotrainer: dict[str, Any]):
    """Map the config's ``zero_shot_method`` string onto biotrainer's enum."""
    ZeroShotMethod = biotrainer["ZeroShotMethod"]
    method_map = {
        "masked_marginals": ZeroShotMethod.MASKED_MARGINALS,
        "wt_marginals": ZeroShotMethod.WT_MARGINALS,
        "pseudoperplexity": ZeroShotMethod.PSEUDOPERPLEXITY,
    }
    if name not in method_map:
        raise ValueError(
            f"Invalid zero_shot_method: {name!r}. "
            f"Expected one of: {sorted(method_map)}"
        )
    return method_map[name]


def _warn_about_cached_frameworks(
    *,
    biotrainer: dict[str, Any],
    embedder_name: str,
    output_dir: Path,
    frameworks: list[str],
) -> None:
    """Warn when an existing report will short-circuit a framework.

    ``AutoEval`` skips any framework already present in the report on disk and
    registers no runner for it, so a re-run into the same ``output_dir`` can
    silently evaluate nothing. Surface that up front.
    """
    report_path = output_dir / biotrainer["AutoEvalReport"].get_file_name(embedder_name)
    if not report_path.exists():
        return
    try:
        existing = biotrainer["AutoEvalReport"].from_json_file(report_path)
        cached = [f for f in frameworks
                  if existing.maybe_framework_result(FRAMEWORK_REPORT_NAMES[f])]
    except Exception:
        logger.warning(f"Could not read the existing report at {report_path}.")
        return
    if cached:
        logger.warning(
            f"Existing report {report_path} already holds results for {cached}; "
            "those frameworks will be reused, not recomputed. Delete the report "
            "or use a fresh output_dir to force a re-run."
        )


def run_eval(eval_config: EvalConfig) -> None:
    """Run biotrainer autoeval on a pure-torch nanoPLM checkpoint.

    Supports one or more zero-shot frameworks per invocation: 'pgym' scores
    ProteinGym DMS variants, 'pbc_zeroshot_contact' predicts contact maps from
    the categorical Jacobian. Both run against the same GPU-resident model
    through a single ``BioEngineer``, and accumulate into one ``AutoEvalReport``
    at ``<output_dir>/autoeval_report_<embedder>.json``. Per-framework artefacts
    land under ``<output_dir>/<embedder>/<FRAMEWORK_NAME>/``.
    """
    _disable_h100_triton_srelu()

    biotrainer = _load_biotrainer_symbols()

    frameworks = list(eval_config.framework)
    removed = [f for f in frameworks if f in REMOVED_FRAMEWORKS]
    if removed:
        raise ValueError(
            f"Framework(s) {removed} are no longer supported. nanoplm eval moved to "
            "biotrainer 2.0.0 and now covers the zero-shot frameworks only: "
            f"{sorted(FRAMEWORK_REPORT_NAMES)}."
        )
    unknown = [f for f in frameworks if f not in FRAMEWORK_REPORT_NAMES]
    if unknown:
        raise ValueError(
            f"Unsupported framework(s): {unknown}. "
            f"Expected one of: {sorted(FRAMEWORK_REPORT_NAMES)}"
        )
    if eval_config.force_download and eval_config.dataset_dir:
        raise ValueError(
            "force_download and dataset_dir are mutually exclusive: force_download "
            "only clears the download cache, which a pre-populated dataset_dir "
            "does not use."
        )

    checkpoint_dir = resolve_checkpoint_dir(eval_config.checkpoint_dir)
    logger.info(f"Resolved checkpoint: {checkpoint_dir}")

    device = _resolve_device(eval_config.device)
    logger.info(f"Using device: {device}")

    model_config = load_model_config_from_checkpoint(checkpoint_dir)
    if model_config is None:
        raise FileNotFoundError(
            f"No model_config.yaml in checkpoint: {checkpoint_dir}. "
            "Re-train the checkpoint with a nanoPLM version that writes "
            "model_config.yaml into the checkpoint directory."
        )
    logger.info(
        f"Loaded model config from checkpoint: {checkpoint_dir}/model_config.yaml"
    )

    model = load_checkpoint(checkpoint_dir, model_config, device)
    tokenizer = model.tokenizer

    embedder_name = eval_config.embedder_name or checkpoint_dir.stem
    logger.info(f"Embedder name: {embedder_name}")

    output_dir = Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_about_cached_frameworks(
        biotrainer=biotrainer,
        embedder_name=embedder_name,
        output_dir=output_dir,
        frameworks=frameworks,
    )

    bio_engineer = NanoPLMBioEngineer.build(
        model,
        tokenizer,
        device,
        biotrainer,
        batch_size=eval_config.zero_shot_batch_size,
    )

    autoeval = biotrainer["AutoEval"](
        embedder_name=embedder_name,
        output_dir=output_dir,
        force_download=eval_config.force_download,
        custom_storage_path=eval_config.dataset_dir,
        custom_bioengineer=bio_engineer,
        development_mode=eval_config.development_mode,
    )

    for framework in frameworks:
        if framework == "pgym":
            method = _resolve_zero_shot_method(eval_config.zero_shot_method, biotrainer)
            logger.info(
                f"Adding PGYM zero-shot: method={eval_config.zero_shot_method}, "
                f"batch_size={eval_config.zero_shot_batch_size}"
            )
            autoeval.pgym(zero_shot_method=method)
        elif framework == "pbc_zeroshot_contact":
            logger.info(
                "Adding PBC zero-shot contact (categorical Jacobian), "
                f"batch_size={eval_config.zero_shot_batch_size}"
            )
            autoeval.pbc_zeroshot_contact()

    logger.info(
        f"Starting autoeval: frameworks={frameworks}, "
        f"development_mode={eval_config.development_mode}"
    )
    logger.info("=" * 60)
    started = time.time()

    report = autoeval.run(device=device)

    logger.info("=" * 60)
    logger.info(f"AUTOEVAL COMPLETE in {time.time() - started:.1f}s — Final Report Summary")
    logger.info(f"Frameworks in report: {sorted(report.all_framework_names())}")
    logger.info("=" * 60)
    report.summary(development_mode=eval_config.development_mode)
