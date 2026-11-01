"""Downstream evaluation pipeline: runs biotrainer autoeval on a pure-torch checkpoint."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Optional

import torch

from nanoplm.eval.checkpoint import (
    EvalProvenance,
    eval_provenance_path,
    load_checkpoint,
    load_model_config_from_checkpoint,
    resolve_checkpoint_dir,
    write_eval_provenance,
)
from nanoplm.eval.config import EvalConfig
from nanoplm.eval.embedder import NanoPLMCustomEmbedder
from nanoplm.eval.zero_shot import NanoPLMBioEngineer
from nanoplm.utils.logger import logger

# nanoPLM framework key -> biotrainer's AutoEvalFramework.get_name().
FRAMEWORK_REPORT_NAMES = {
    "pgym": "PGYM",
    "pbc_zeroshot_contact": "PBC_ZEROSHOT_CONTACT",
    "pbc_supervised": "PBC_SUPERVISED",
    "pbc_supervised_contact": "PBC_SUPERVISED_CONTACT",
}

# Frameworks that existed before the biotrainer 2.0.0 migration. Kept only to
# produce an actionable error instead of a bare "unsupported framework".
REMOVED_FRAMEWORKS = {"pbc", "flip"}


def _load_biotrainer_symbols() -> dict[str, Any]:
    """Import the biotrainer 2.1.1 symbols the eval pipeline depends on."""
    try:
        import biotrainer
        from biotrainer.autoeval import AutoEval
        from biotrainer.bioengineer import BioEngineer, CustomBioEngineerModel
        from biotrainer.bioengineer.models import (
            CustomBioEngineerModelWrapper,
        )
        from biotrainer.bioengineer.bioengineer_utils import get_optimal_window
        from biotrainer.embedding import CustomEmbedder
        from biotrainer_core.data_classes import ZeroShotMethod
        from biotrainer_core.data_classes.autoeval import AutoEvalReport
        from biotrainer.shared import SequenceTooLongError
        from biotrainer_core.utils.constants import STANDARD_AAS
    except ImportError as e:
        raise ImportError(
            "biotrainer 2.1.1 is required for nanoplm eval but is not installed "
            f"or has an incompatible API. Reinstall nanoplm to pull it in. "
            f"Original error: {e}"
        ) from e

    return {
        "AutoEval": AutoEval,
        "AutoEvalReport": AutoEvalReport,
        "BioEngineer": BioEngineer,
        "CustomBioEngineerModel": CustomBioEngineerModel,
        "CustomBioEngineerModelWrapper": CustomBioEngineerModelWrapper,
        "CustomEmbedder": CustomEmbedder,
        "get_optimal_window": get_optimal_window,
        "ZeroShotMethod": ZeroShotMethod,
        "SequenceTooLongError": SequenceTooLongError,
        "STANDARD_AAS": STANDARD_AAS,
        "version": biotrainer.__version__,
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


def _reused_frameworks(
    *,
    biotrainer: dict[str, Any],
    report_path: Path,
    frameworks: list[str],
    development_mode: bool,
) -> list[str]:
    """The frameworks ``AutoEval`` will reuse from an existing report.

    Mirrors ``AutoEval._general_task_setup``, where a stored result stands in
    for this run unless it covers only a filtered subset of tasks or was
    produced in development mode while this run wants the full benchmark. A
    reused framework registers no runner, so an invocation whose every framework
    is reused evaluates nothing.
    """
    if not report_path.exists():
        return []
    reused: list[str] = []
    try:
        existing = biotrainer["AutoEvalReport"].from_json_file(report_path)
        for framework in frameworks:
            result = existing.maybe_framework_result(FRAMEWORK_REPORT_NAMES[framework])
            if not result or result.task_filter_applied:
                continue
            if result.used_development_mode() and not development_mode:
                continue
            reused.append(framework)
    except Exception:
        logger.warning(f"Could not read the existing report at {report_path}.")
        return []
    if reused:
        logger.warning(
            f"Existing report {report_path} already holds results for {reused}; "
            "those frameworks will be reused, not recomputed. Delete the report "
            "or use a fresh output_dir to force a re-run."
        )
    return reused


def _refuse_stale_dataset_cache(root: Path) -> None:
    """Raise if ``root`` holds the dataset layout of biotrainer < 2.1.

    Without ``dataset_dir``, biotrainer 2.1 reads ``<root>/PBC`` and
    ``<root>/PGYM`` and downloads only into empty directories. A cache left by
    the PR #192 / 2.0 stack is therefore never refreshed: supervised PBC would
    silently score the old files, and the contact frameworks and PGYM fail
    later with unrelated-looking errors.
    """
    stale = []
    if (root / "CONTACT").is_dir():
        stale.append(root / "CONTACT")
    pbc = root / "PBC"
    if pbc.is_dir() and any(pbc.iterdir()) and not (pbc / "contacts").is_dir():
        stale.append(pbc)
    # A PGYM-only cache: 2.1 reads row["pbc_dev_mode"] from this file.
    reference = root / "PGYM" / "reference" / "reference.csv"
    if reference.is_file():
        with reference.open(newline="") as f:
            if "pbc_dev_mode" not in next(csv.reader(f), []):
                stale.append(reference)
    if stale:
        raise RuntimeError(
            f"The biotrainer dataset cache {root} has the layout of biotrainer "
            f"< 2.1 ({', '.join(str(p) for p in stale)}), which 2.1 would reuse "
            "without downloading the current bundle. Set dataset_dir to a "
            "directory for the 2.1 datasets, or delete the cache so 2.1 downloads "
            "the current bundle."
        )


def _refuse_results_from_another_biotrainer(
    *, output_dir: Path, embedder_name: str, report_path: Path, version: str
) -> None:
    """Raise if this embedder's results in ``output_dir`` came from another biotrainer.

    AutoEval reuses a stored framework result and the per-protein contact
    cache without asking which biotrainer wrote them, and 2.1 reads a report
    from an older version's development mode as a full result. Every run
    writes its provenance sidecar before evaluating, so results whose sidecar
    is missing or names another version are refused rather than reused.
    Other embedders in the same ``output_dir`` are never read by this run and
    are not checked.
    """
    framework_dir = output_dir / embedder_name.replace("/", "-")
    has_results = report_path.exists() or (
        framework_dir.is_dir() and any(framework_dir.iterdir())
    )
    if not has_results:
        return
    sidecar = eval_provenance_path(report_path)
    try:
        found = json.loads(sidecar.read_text(encoding="utf-8")).get(
            "biotrainer_version"
        )
    except (OSError, ValueError, AttributeError):
        found = None
    if found == version:
        return
    if found:
        evidence = f"{sidecar} records biotrainer {found}"
    elif sidecar.exists():
        evidence = (
            f"{sidecar} has no biotrainer_version (written before nanoplm "
            "recorded it)"
        )
    else:
        evidence = f"there is no provenance sidecar at {sidecar}"
    raise RuntimeError(
        f"{output_dir} already holds results for {embedder_name!r} that this "
        f"run cannot attribute to biotrainer {version}: {evidence}. biotrainer "
        "would reuse them as if they were current. Choose a fresh output_dir, "
        f"or move {report_path.name} and {framework_dir.name}/ out of "
        f"{output_dir}."
    )


def _completion_message(
    *, computed: list[str], reused: list[str], elapsed_s: float
) -> str:
    """Say what this invocation computed, so a 0.0s finish cannot read as a run."""
    if not computed:
        return (
            f"AUTOEVAL COMPUTED NOTHING in {elapsed_s:.1f}s: every requested "
            f"framework {reused} was reused from the existing report. Delete "
            "the report or use a fresh output_dir to force a re-run."
        )
    if reused:
        return (
            f"AUTOEVAL COMPLETE in {elapsed_s:.1f}s. "
            f"Computed {computed}, reused {reused}."
        )
    return f"AUTOEVAL COMPLETE in {elapsed_s:.1f}s. Computed {computed}."


def run_eval(eval_config: EvalConfig) -> None:
    """Run biotrainer autoeval on a pure-torch nanoPLM checkpoint.

    Supports one or more frameworks per invocation: 'pgym' scores ProteinGym DMS
    variants zero-shot, 'pbc_zeroshot_contact' predicts contact maps from the
    categorical Jacobian, 'pbc_supervised_contact' fits a probe on the model's
    attention maps, and 'pbc_supervised' trains task heads on the frozen
    encoder's embeddings. The zero-shot pair runs through a ``BioEngineer`` and
    the supervised one through a ``CustomEmbedder``, both wrapping the same
    GPU-resident model, and all accumulate into one ``AutoEvalReport`` at
    ``<output_dir>/autoeval_report_<embedder>.json``, beside an
    ``autoeval_report_<embedder>.provenance.json`` sidecar naming the checkpoint
    behind it. Per-framework artefacts land under
    ``<output_dir>/<embedder>/<FRAMEWORK_NAME>/``. Results for this embedder
    left by another biotrainer version stop the run before the model loads.
    """
    # MoE checkpoints route the MLP through sonicmoe's quack GEMM, whose
    # autotuner keys on tensor shapes. Eval's variable-length batches make
    # almost every batch a new key, at ~11 s a search for no measurable
    # steady-state gain. Must run before the model is loaded.
    from nanoplm.eval._quack_untuned import disable_quack_autotune

    if disable_quack_autotune():
        logger.info("quack GEMM autotuning disabled for eval "
                    "(set NANOPLM_EVAL_QUACK_TUNED=1 to restore)")
    _disable_h100_triton_srelu()

    biotrainer = _load_biotrainer_symbols()

    frameworks = list(eval_config.framework)
    removed = [f for f in frameworks if f in REMOVED_FRAMEWORKS]
    if removed:
        raise ValueError(
            f"Framework(s) {removed} are no longer supported. nanoplm eval runs "
            "on biotrainer 2.1.1 and covers: "
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
    if not eval_config.dataset_dir and not eval_config.force_download:
        # force_download clears the whole cache, stale layout included.
        from biotrainer.autoeval.core.autoeval_data_handler import get_autoeval_path

        _refuse_stale_dataset_cache(get_autoeval_path())

    checkpoint_dir = resolve_checkpoint_dir(eval_config.checkpoint_dir)
    logger.info(f"Resolved checkpoint: {checkpoint_dir}")

    provenance = EvalProvenance.resolve(
        checkpoint_dir=checkpoint_dir,
        embedder_name=eval_config.embedder_name,
        loop_count=eval_config.loop_count,
        biotrainer_version=biotrainer["version"],
    )
    embedder_name = provenance.embedder_name
    logger.info(f"Embedder name: {embedder_name}")

    output_dir = Path(eval_config.output_dir)
    report_path = output_dir / biotrainer["AutoEvalReport"].get_file_name(embedder_name)
    _refuse_results_from_another_biotrainer(
        output_dir=output_dir,
        embedder_name=embedder_name,
        report_path=report_path,
        version=biotrainer["version"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reused = _reused_frameworks(
        biotrainer=biotrainer,
        report_path=report_path,
        frameworks=frameworks,
        development_mode=eval_config.development_mode,
    )

    # Written before the run so an eval that dies partway still records which
    # checkpoint it was running, and which biotrainer its results came from.
    sidecar = write_eval_provenance(provenance, report_path)
    if sidecar is not None:
        logger.info(f"Wrote eval provenance sidecar: {sidecar}")

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
    if eval_config.torch_compile:
        # dynamic=True on purpose: shapes vary per protein and per batch, and a
        # static compile would recompile for each one. Attribute access falls
        # through to the wrapped module, so `.tokenizer` and friends still work.
        logger.info("Compiling the model (dynamic shapes) ...")
        model = torch.compile(model, dynamic=True)

    bio_engineer = NanoPLMBioEngineer.build(
        model,
        tokenizer,
        device,
        biotrainer,
        batch_size=eval_config.zero_shot_batch_size,
        loop_count=eval_config.loop_count,
    )
    custom_embedder = NanoPLMCustomEmbedder.build(
        model,
        tokenizer,
        device,
        biotrainer,
        batch_size=eval_config.zero_shot_batch_size,
        loop_count=eval_config.loop_count,
    )

    autoeval = biotrainer["AutoEval"](
        embedder_name=embedder_name,
        output_dir=output_dir,
        force_download=eval_config.force_download,
        custom_storage_path=eval_config.dataset_dir,
        custom_bioengineer=bio_engineer,
        custom_embedder=custom_embedder,
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
            autoeval.pbc_zeroshot_contact(
                batch_size=eval_config.zero_shot_batch_size
            )
        elif framework == "pbc_supervised":
            wanted = eval_config.pbc_supervised_datasets
            # Restricting the datasets also restricts the embedding: biotrainer
            # only embeds the sequences the selected tasks need, which is the
            # whole reason to filter rather than run all seven and ignore five.
            task_filter = None
            if wanted:
                wanted_set = set(wanted)
                task_filter = lambda task: task.dataset_name in wanted_set  # noqa: E731
            logger.info(
                "Adding PBC supervised: task heads trained on frozen "
                f"per-residue/per-sequence embeddings, datasets="
                f"{sorted(wanted_set) if wanted else 'all'}, batch_size="
                f"{eval_config.zero_shot_batch_size}"
            )
            autoeval.pbc_supervised(task_filter=task_filter)
        elif framework == "pbc_supervised_contact":
            logger.info(
                "Adding PBC supervised contact: a logistic probe over "
                "symmetrized, APC-corrected attention maps"
            )
            autoeval.pbc_supervised_contact()

    logger.info(
        f"Starting autoeval: frameworks={frameworks}, "
        f"development_mode={eval_config.development_mode}"
    )
    logger.info("=" * 60)
    started = time.time()

    report = autoeval.run(device=device)

    logger.info("=" * 60)
    computed = [f for f in frameworks if f not in reused]
    message = _completion_message(
        computed=computed, reused=reused, elapsed_s=time.time() - started
    )
    if computed:
        logger.info(message)
    else:
        logger.warning(message)
    logger.info(f"Frameworks in report: {sorted(report.all_framework_names())}")
    logger.info("=" * 60)
    report.summary(development_mode=eval_config.development_mode)
