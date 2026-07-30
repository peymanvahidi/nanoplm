"""Downstream evaluation pipeline: runs biotrainer autoeval on a pure-torch checkpoint."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from nanoplm.eval.checkpoint import (
    load_checkpoint,
    load_model_config_from_checkpoint,
    resolve_checkpoint_dir,
)
from nanoplm.eval.config import EvalConfig
from nanoplm.eval.zero_shot import NanoPLMBioEngineer, NanoPLMEmbedder
from nanoplm.utils.logger import logger

# nanoPLM framework name -> biotrainer AvailableFramework member name.
# Contact frameworks are intentionally unsupported:
#   PBC_SUPERVISED_CONTACT requires a HuggingfaceTransformerEmbedder (rejects custom embedders)
#   PBC_ZEROSHOT_CONTACT uses the categorical Jacobian, which assumes a leading BOS token
_FRAMEWORK_TO_AVAILABLE = {
    "pbc": "PBC_SUPERVISED",
    "flip": "FLIP",
    "pgym": "PGYM",
}


def _load_biotrainer_symbols() -> dict[str, Any]:
    """Import the biotrainer 2.0.0 symbols used by both the sequential and parallel paths."""
    try:
        from biotrainer.autoeval import AutoEval, AvailableFramework
        from biotrainer.autoeval.core.autoeval_config_bank import AutoEvalConfigBank
        from biotrainer.autoeval.pipelines import (
            check_h5_file,
            get_unique_framework_sequences,
            setup_embedder,
            setup_output_dir,
            validate_input,
        )
        from biotrainer.bioengineer import BioEngineer, CustomBioEngineerModel
        from biotrainer.embedding import CustomEmbedder
        from biotrainer.training.utilities.executer import parse_config_file_and_execute_run
        from biotrainer_core.data_classes import Protocol, ZeroShotMethod
        from biotrainer_core.data_classes.autoeval import AutoEvalReport
        from biotrainer_core.data_classes.autoeval.autoeval_report import (
            SupervisedFrameworkReport,
        )
        from biotrainer_core.utils.constants import STANDARD_AAS
    except ImportError as e:
        raise ImportError(
            "biotrainer 2.0.x is required for nanoplm eval but is not installed "
            f"or has an incompatible API. Install with: pip install nanoplm[eval]. "
            f"Original error: {e}"
        ) from e

    return {
        "AutoEval": AutoEval,
        "AvailableFramework": AvailableFramework,
        "AutoEvalConfigBank": AutoEvalConfigBank,
        "AutoEvalReport": AutoEvalReport,
        "BioEngineer": BioEngineer,
        "CustomBioEngineerModel": CustomBioEngineerModel,
        "CustomEmbedder": CustomEmbedder,
        "Protocol": Protocol,
        "STANDARD_AAS": STANDARD_AAS,
        "SupervisedFrameworkReport": SupervisedFrameworkReport,
        "ZeroShotMethod": ZeroShotMethod,
        "check_h5_file": check_h5_file,
        "get_unique_framework_sequences": get_unique_framework_sequences,
        "parse_config_file_and_execute_run": parse_config_file_and_execute_run,
        "setup_embedder": setup_embedder,
        "setup_output_dir": setup_output_dir,
        "validate_input": validate_input,
    }


def _run_single_task(
    task_name: str,
    config: dict,
    parse_and_execute,
) -> tuple[str, dict | None]:
    """Run a single biotrainer task in a worker thread. Returns ``(task_name, result_dict)``."""
    try:
        result = parse_and_execute(config=config)
        return task_name, result
    except Exception:
        logger.exception(f"Task {task_name} failed")
        return task_name, None


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


def _force_cutlass_moe_backend(model_config) -> None:
    """Fall back to the cutlass MoE backend for eval.

    The sonicmoe backend routes experts through quack's fused GEMM, which rejects
    bfloat16 inputs ("a_dtype should be float16 or float8") and so raises on every
    forward under the bf16 autocast used here. cutlass computes the same math with
    device-side grouped GEMMs, so eval results are unaffected up to reduction order.
    """
    if getattr(model_config, "moe_kernel_backend", None) == "sonicmoe":
        logger.warning(
            "moe_kernel_backend='sonicmoe' is not usable under bf16 autocast "
            "(quack fused GEMM requires fp16/fp8); falling back to 'cutlass' for eval."
        )
        model_config.moe_kernel_backend = "cutlass"


def _resolve_zero_shot_method(eval_config: EvalConfig, biotrainer: dict[str, Any]):
    zero_shot_method_map = {
        "masked_marginals": biotrainer["ZeroShotMethod"].MASKED_MARGINALS,
        "wt_marginals": biotrainer["ZeroShotMethod"].WT_MARGINALS,
        "pseudoperplexity": biotrainer["ZeroShotMethod"].PSEUDOPERPLEXITY,
    }
    if eval_config.zero_shot_method not in zero_shot_method_map:
        raise ValueError(
            f"Invalid zero_shot_method: {eval_config.zero_shot_method!r}. "
            f"Expected one of: {sorted(zero_shot_method_map)}"
        )
    return zero_shot_method_map[eval_config.zero_shot_method]


def _build_autoeval(
    *,
    eval_config: EvalConfig,
    biotrainer: dict[str, Any],
    embedder_name: str,
    custom_embedder,
    custom_bioengineer,
):
    """Construct an ``AutoEval`` instance shared by all frameworks in this run."""
    return biotrainer["AutoEval"](
        embedder_name=embedder_name,
        output_dir=str(eval_config.output_dir),
        force_download=eval_config.force_download,
        min_seq_length=eval_config.min_seq_length,
        max_seq_length=eval_config.max_seq_length,
        custom_embedder=custom_embedder,
        custom_bioengineer=custom_bioengineer,
        # biotrainer defaults this to True, which silently evaluates on data subsets.
        development_mode=eval_config.development_mode,
    )


def _run_supervised_parallel(
    *,
    eval_config: EvalConfig,
    biotrainer: dict[str, Any],
    framework: str,
    embedder_name: str,
    custom_embedder,
    device: torch.device,
    autoeval_report,
) -> None:
    """Run one supervised framework via the 3-phase embed-then-CPU-probe parallel flow.

    biotrainer's own ``AutoEval.run()`` executes tasks sequentially. This path keeps
    nanoPLM's parallel probe training by driving biotrainer's pipeline helpers directly,
    so it is coupled to biotrainer internals and may need updating on a biotrainer bump.
    """
    logger.info(
        f"Starting autoeval pipeline (PARALLEL, "
        f"{eval_config.parallel_tasks} workers): framework={framework}"
    )
    logger.info("=" * 60)
    overall_start = time.time()

    available_framework = getattr(
        biotrainer["AvailableFramework"], _FRAMEWORK_TO_AVAILABLE[framework]
    )
    framework_obj = biotrainer["validate_input"](
        available_framework,
        zero_shot_method=None,
        min_seq_length=eval_config.min_seq_length,
        max_seq_length=eval_config.max_seq_length,
    )
    framework_output_dir = biotrainer["setup_output_dir"](
        base_dir=Path(eval_config.output_dir),
        embedder_name=embedder_name,
        framework_name=framework_obj.get_name(),
    )

    # --- Phase 1: Compute embeddings (GPU) -----------------------
    logger.info(">>> Phase 1: Computing embeddings …")
    t0 = time.time()

    task_config_tuples, unique_per_residue, unique_per_sequence = (
        biotrainer["get_unique_framework_sequences"](
            framework=framework_obj,
            min_seq_length=eval_config.min_seq_length,
            max_seq_length=eval_config.max_seq_length,
            force_download=eval_config.force_download,
            development_mode=eval_config.development_mode,
        )
    )

    if eval_config.tasks:
        task_config_tuples = [
            (t, c) for t, c in task_config_tuples
            if t.combined_name() in eval_config.tasks
        ]
        logger.info(
            "  Filtered to tasks: "
            f"{[t.combined_name() for t, _ in task_config_tuples]}"
        )

    embedder = biotrainer["setup_embedder"](
        embedder_name=embedder_name,
        output_dir=framework_output_dir,
        use_half_precision=False,
        custom_embedder=custom_embedder,
        device=device,
    )

    embeddings_file_per_residue = embedder.per_residue_path(
        [sr.seq for _, sr in unique_per_residue.items()]
    )
    embeddings_file_per_sequence = embedder.per_sequence_path(
        [sr.seq for _, sr in unique_per_sequence.items()]
    )

    biotrainer["check_h5_file"](
        name="per-residue",
        h5_path=embeddings_file_per_residue,
        expected_length=len(unique_per_residue),
    )
    biotrainer["check_h5_file"](
        name="per-sequence",
        h5_path=embeddings_file_per_sequence,
        expected_length=len(unique_per_sequence),
    )

    logger.info(f"Embeddings done in {time.time() - t0:.1f}s")

    # --- Phase 2: Prepare task configs -------------------------------
    logger.info(">>> Phase 2: Preparing task configs …")
    framework_report = biotrainer["SupervisedFrameworkReport"].empty(
        min_seq_len=eval_config.min_seq_length,
        max_seq_len=eval_config.max_seq_length,
    )
    prepared_tasks: list[tuple] = []  # (task_name, config_dict)

    for task, cfg in task_config_tuples:
        task_name = task.combined_name()
        task_output_dir = framework_output_dir / task_name

        protocol = biotrainer["Protocol"]
        if protocol.from_string(cfg["protocol"]) in protocol.using_per_sequence_embeddings():
            task_emb_file = embeddings_file_per_sequence
        else:
            task_emb_file = embeddings_file_per_residue

        cfg = biotrainer["AutoEvalConfigBank"].add_custom_values_to_config(
            config=cfg,
            embedder_name=embedder_name,
            embeddings_file=task_emb_file,
            input_file=task.input_files[0],
            output_dir=task_output_dir,
            device="cpu",  # probe training is CPU-only
        )

        maybe_result = biotrainer["SupervisedFrameworkReport"].maybe_load_existing_result(
            embedder_name=embedder_name,
            task_output_dir=task_output_dir,
        )
        if maybe_result:
            logger.info(f"  Loaded existing result for {task_name}, skipping …")
            framework_report.update_result(task_name, maybe_result)
        else:
            prepared_tasks.append((task_name, cfg))

    if not prepared_tasks:
        logger.info("All tasks already have results — nothing to run!")
    else:
        task_names = [tn for tn, _ in prepared_tasks]
        logger.info(f"  Tasks to run ({len(prepared_tasks)}): {task_names}")

        # --- Phase 3: Run tasks in parallel --------------------------
        logger.info(
            f">>> Phase 3: Running {len(prepared_tasks)} tasks in parallel "
            f"({eval_config.parallel_tasks} workers) …"
        )
        t1 = time.time()
        parse_and_execute = biotrainer["parse_config_file_and_execute_run"]

        # NOTE: biotrainer's setup_logging mutates a global logger
        # (removes handlers, attaches a FileHandler per task). With
        # parallel_tasks > 1, log lines from concurrent tasks may
        # interleave or land in the wrong logger_out.log file.
        with ThreadPoolExecutor(max_workers=eval_config.parallel_tasks) as pool:
            futures = [
                pool.submit(_run_single_task, tn, cfg, parse_and_execute)
                for tn, cfg in prepared_tasks
            ]
            for future in as_completed(futures):
                returned_name, result = future.result()
                if result is not None:
                    framework_report.update_result(returned_name, result)
                    logger.info(f"  {returned_name} done")
                else:
                    logger.error(f"  {returned_name} FAILED")

        logger.info(f"All tasks finished in {time.time() - t1:.1f}s")

    autoeval_report.add_framework_report(
        framework_name=framework_obj.get_name(),
        framework_mode=framework_obj.get_mode(),
        report=framework_report,
    )
    logger.info(
        f"{framework_obj.get_name()} results added to combined report. "
        f"(total: {time.time() - overall_start:.1f}s)"
    )


def run_eval(eval_config: EvalConfig) -> None:
    """Run biotrainer autoeval on a pure-torch nanoPLM checkpoint.

    Supports one or more frameworks per invocation. When ``parallel_tasks <= 0``
    every framework is handed to a single ``biotrainer.autoeval.AutoEval``
    instance, which deduplicates and computes embeddings once across all of them
    before running their tasks. When ``parallel_tasks > 0``, supervised
    frameworks (pbc, flip) instead go through a 3-phase embed-then-CPU-probe flow
    that trains probes in a ThreadPoolExecutor; 'pgym' still goes through
    ``AutoEval``.

    All frameworks accumulate into a single ``AutoEvalReport`` written at the
    top-level output directory.
    """
    _disable_h100_triton_srelu()

    biotrainer = _load_biotrainer_symbols()

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
    _force_cutlass_moe_backend(model_config)

    model = load_checkpoint(checkpoint_dir, model_config, device)
    tokenizer = model.tokenizer

    embedder_name = eval_config.embedder_name or checkpoint_dir.stem
    logger.info(f"Embedder name: {embedder_name}")

    if not eval_config.development_mode:
        logger.info("development_mode=False — evaluating on the full datasets.")
    else:
        logger.warning(
            "development_mode=True — biotrainer evaluates on data SUBSETS only. "
            "Set development_mode: false for a full evaluation."
        )

    frameworks = list(eval_config.framework)

    # The model stays resident for the whole run: the custom embedder and the
    # bioengineer both close over it.
    custom_embedder = NanoPLMEmbedder.build(
        model, tokenizer, device, eval_config.batch_size, biotrainer
    )
    custom_bioengineer = (
        NanoPLMBioEngineer.build(model, tokenizer, device, biotrainer)
        if "pgym" in frameworks
        else None
    )

    supervised_frameworks = [f for f in frameworks if f != "pgym"]
    run_supervised_in_parallel = eval_config.parallel_tasks > 0 and supervised_frameworks

    if eval_config.tasks and not run_supervised_in_parallel and supervised_frameworks:
        logger.warning(
            "--tasks filter is only supported in parallel mode; ignoring."
        )

    # ── Frameworks handled by biotrainer's own AutoEval ──────────────────────
    autoeval_frameworks = ["pgym"] if "pgym" in frameworks else []
    if not run_supervised_in_parallel:
        autoeval_frameworks += supervised_frameworks

    if autoeval_frameworks:
        autoeval = _build_autoeval(
            eval_config=eval_config,
            biotrainer=biotrainer,
            embedder_name=embedder_name,
            custom_embedder=custom_embedder,
            custom_bioengineer=custom_bioengineer,
        )
        for framework in autoeval_frameworks:
            if framework == "pgym":
                zero_shot_method = _resolve_zero_shot_method(eval_config, biotrainer)
                logger.info(
                    "Adding ProteinGym zero-shot evaluation: "
                    f"method={eval_config.zero_shot_method}"
                )
                autoeval.pgym(zero_shot_method=zero_shot_method)
            elif framework == "pbc":
                logger.info("Adding PBC supervised evaluation")
                autoeval.pbc_supervised()
            elif framework == "flip":
                logger.info("Adding FLIP supervised evaluation")
                autoeval.flip()
            else:
                raise ValueError(f"Unsupported framework: {framework!r}")

        logger.info("=" * 60)
        # AutoEval pre-embeds once across every framework registered above, then runs
        # each framework's tasks and writes the combined report after each one.
        autoeval.run(device=device)

    # ── Supervised frameworks via the parallel probe-training path ───────────
    if run_supervised_in_parallel:
        # loaded_or_empty picks up whatever AutoEval already wrote above, so pgym
        # results are preserved when both paths run in one invocation.
        autoeval_report = biotrainer["AutoEvalReport"].loaded_or_empty(
            embedder_name=embedder_name,
            training_date=str(datetime.now().date().isoformat()),
            output_dir=Path(eval_config.output_dir),
        )
        for framework in supervised_frameworks:
            _run_supervised_parallel(
                eval_config=eval_config,
                biotrainer=biotrainer,
                framework=framework,
                embedder_name=embedder_name,
                custom_embedder=custom_embedder,
                device=device,
                autoeval_report=autoeval_report,
            )
        autoeval_report.write(Path(eval_config.output_dir))
    else:
        autoeval_report = autoeval.autoeval_report

    logger.info("=" * 60)
    logger.info("AUTOEVAL COMPLETE — Final Report Summary")
    logger.info("=" * 60)
    autoeval_report.summary(development_mode=eval_config.development_mode)
