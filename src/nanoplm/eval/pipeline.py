"""Downstream evaluation pipeline: runs biotrainer autoeval on a pure-torch checkpoint."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from nanoplm.eval.checkpoint import (
    load_checkpoint,
    load_model_config,
    load_model_config_from_checkpoint,
    resolve_checkpoint_dir,
)
from nanoplm.eval.config import EvalConfig
from nanoplm.eval.embedders import embed_per_residue, embed_per_sequence
from nanoplm.eval.zero_shot import NanoPLMBioEngineer
from nanoplm.utils.logger import logger


def _load_biotrainer_symbols() -> dict[str, Any]:
    """Import biotrainer 1.4.0 symbols used by both sequential and parallel paths."""
    from biotrainer.autoeval import autoeval_pipeline, AutoEvalReport
    from biotrainer.autoeval.pipelines.autoeval_supervised import (
        _check_h5_file,
        _setup_embedding_functions,
        get_unique_framework_sequences,
    )
    from biotrainer.autoeval.pipelines.autoeval_report import SupervisedFrameworkReport
    from biotrainer.autoeval.core.autoeval_config_bank import AutoEvalConfigBank
    from biotrainer.protocols import Protocol
    from biotrainer.utilities.executer import parse_config_file_and_execute_run
    from biotrainer.bioengineer import (
        CustomBioEngineerModel,
        BioEngineer,
        ZeroShotMethod,
    )
    from biotrainer.utilities import STANDARD_AAS

    return {
        "autoeval_pipeline": autoeval_pipeline,
        "AutoEvalReport": AutoEvalReport,
        "get_unique_framework_sequences": get_unique_framework_sequences,
        "_setup_embedding_functions": _setup_embedding_functions,
        "_check_h5_file": _check_h5_file,
        "SupervisedFrameworkReport": SupervisedFrameworkReport,
        "AutoEvalConfigBank": AutoEvalConfigBank,
        "Protocol": Protocol,
        "parse_config_file_and_execute_run": parse_config_file_and_execute_run,
        "CustomBioEngineerModel": CustomBioEngineerModel,
        "BioEngineer": BioEngineer,
        "ZeroShotMethod": ZeroShotMethod,
        "STANDARD_AAS": STANDARD_AAS,
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
    except Exception as e:
        logger.error(f"[ERROR] Task {task_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return task_name, None


def _resolve_device(device_override: str | None) -> torch.device:
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


def run_eval(eval_config: EvalConfig) -> None:
    """Run biotrainer autoeval on a pure-torch nanoPLM checkpoint.

    Supports one or more frameworks per invocation. 'pgym' is zero-shot
    ProteinGym (runs first while the model is GPU-resident). Supervised
    frameworks (pbc, flip) run after pgym either via
    ``biotrainer.autoeval.autoeval_pipeline`` (sequential, when
    ``parallel_tasks <= 0``) or via a 3-phase embed-then-CPU-probe flow
    (parallel, when ``parallel_tasks > 0``). All frameworks accumulate into a
    single ``AutoEvalReport`` at the embedder-level output directory.
    """
    _disable_h100_triton_srelu()

    biotrainer = _load_biotrainer_symbols()

    checkpoint_dir = resolve_checkpoint_dir(
        eval_config.checkpoint_dir,
        eval_config.checkpoint_root,
    )
    logger.info(f"Resolved checkpoint: {checkpoint_dir}")

    device = _resolve_device(eval_config.device)
    logger.info(f"Using device: {device}")

    # Prefer model_config.yaml inside the checkpoint; fall back to pretrain.yaml.
    model_config = load_model_config_from_checkpoint(checkpoint_dir)
    if model_config is not None:
        logger.info(
            f"Loaded model config from checkpoint: {checkpoint_dir}/model_config.yaml"
        )
    else:
        logger.info(
            f"No model_config.yaml in checkpoint, falling back to {eval_config.config_yaml}"
        )
        model_config = load_model_config(eval_config.config_yaml)

    model = load_checkpoint(checkpoint_dir, model_config, device)
    tokenizer = model.tokenizer

    embedder_name = eval_config.embedder_name or checkpoint_dir.stem
    logger.info(f"Embedder name: {embedder_name}")

    batch_size = eval_config.batch_size
    frameworks = list(eval_config.framework)

    # Embedding closures — captured by reference. Do NOT `del model` later or
    # these closures will raise NameError on next call.
    def _embed_residue(seqs):
        return embed_per_residue(model, tokenizer, seqs, device, batch_size)

    def _embed_sequence(seqs):
        return embed_per_sequence(model, tokenizer, seqs, device, batch_size)

    # Embedder-level output directory: all framework subdirs + combined report live here.
    embedder_dir_name = embedder_name.replace("/", "-")
    embedder_output_dir = Path(eval_config.output_dir) / embedder_dir_name
    os.makedirs(embedder_output_dir, exist_ok=True)

    autoeval_report = biotrainer["AutoEvalReport"].loaded_or_empty(
        embedder_name=embedder_name,
        training_date=str(datetime.now().date().isoformat()),
        output_dir=embedder_output_dir,
    )

    # ── pgym zero-shot (runs first; needs the GPU model throughout) ──────────
    # TODO(future): consider running supervised frameworks first and pgym last
    # as best-effort, so a pgym failure doesn't take down the supervised
    # results. Trade-off: pgym holds the model on GPU throughout its run;
    # running it last requires keeping the model resident for the whole
    # supervised loop too. Today we already keep the model resident, so the
    # trade is just about which failure blocks which results.
    if "pgym" in frameworks:
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
        zero_shot_method = zero_shot_method_map[eval_config.zero_shot_method]

        logger.info(
            f"Starting ProteinGym zero-shot evaluation: method={eval_config.zero_shot_method}"
        )
        logger.info("=" * 60)

        bio_engineer = NanoPLMBioEngineer.build(model, tokenizer, device, biotrainer)

        current_progress = None
        for progress in biotrainer["autoeval_pipeline"](
            embedder_name=embedder_name,
            framework="pgym",
            zero_shot_method=zero_shot_method,
            custom_bioengineer=bio_engineer,
            output_dir=str(eval_config.output_dir),
            force_download=eval_config.force_download,
        ):
            current_progress = progress
            logger.info(
                f"[{progress.completed_tasks}/{progress.total_tasks}] "
                f"{progress.current_framework_name}: {progress.current_task_name}"
            )

        if current_progress and current_progress.final_report:
            pgym_report = current_progress.final_report.zeroshot_results.get("PGYM")
            if pgym_report:
                autoeval_report.add_zeroshot_result("PGYM", pgym_report)
                logger.info("PGYM results added to combined report.")
            else:
                logger.warning("No PGYM zeroshot results found in final report.")
        else:
            logger.warning("No final report generated for PGYM.")

    # ── Supervised frameworks (pbc, flip) ────────────────────────────────────
    # Model is held alive across all supervised frameworks. Probe training runs
    # on cfg["device"] = "cpu", so the resident GPU model doesn't contend with
    # anything; the cost is a few hundred MB of GPU memory held during the CPU
    # probe phase. Freeing inside the loop would invalidate the closure cells
    # captured by _embed_residue / _embed_sequence and crash the next
    # framework's Phase 1.
    supervised_frameworks = [f for f in frameworks if f != "pgym"]

    for framework in supervised_frameworks:
        framework_key = framework.upper()
        framework_dir = (
            f"{framework}_{eval_config.min_seq_length}_{eval_config.max_seq_length}"
        )
        framework_output_dir = embedder_output_dir / framework_dir
        os.makedirs(framework_output_dir, exist_ok=True)

        if eval_config.parallel_tasks <= 0:
            # ── Sequential mode ──────────────────────────────────────────────
            if eval_config.tasks:
                logger.warning(
                    "--tasks filter is only supported in parallel mode; ignoring."
                )
            logger.info(
                f"Starting autoeval pipeline (sequential): framework={framework}"
            )
            logger.info("=" * 60)

            current_progress = None
            for progress in biotrainer["autoeval_pipeline"](
                embedder_name=embedder_name,
                framework=framework,
                output_dir=str(eval_config.output_dir),
                min_seq_length=eval_config.min_seq_length,
                max_seq_length=eval_config.max_seq_length,
                custom_embedding_function_per_residue=_embed_residue,
                custom_embedding_function_per_sequence=_embed_sequence,
                force_download=eval_config.force_download,
            ):
                current_progress = progress
                logger.info(
                    f"[{progress.completed_tasks}/{progress.total_tasks}] "
                    f"{progress.current_framework_name}: {progress.current_task_name}"
                )

            if current_progress and current_progress.final_report:
                supervised_report = current_progress.final_report.supervised_results.get(
                    framework_key
                )
                if supervised_report:
                    autoeval_report.add_supervised_result(framework_key, supervised_report)
                else:
                    logger.warning(
                        f"No {framework_key} supervised results in final report."
                    )
            else:
                logger.warning(f"No final report generated for {framework}.")

        else:
            # ── Parallel mode ────────────────────────────────────────────────
            logger.info(
                f"Starting autoeval pipeline (PARALLEL, "
                f"{eval_config.parallel_tasks} workers): framework={framework}"
            )
            logger.info("=" * 60)
            overall_start = time.time()

            # --- Phase 1: Compute embeddings (GPU) -----------------------
            logger.info(">>> Phase 1: Computing embeddings …")
            t0 = time.time()

            task_config_tuples, unique_per_residue, unique_per_sequence = (
                biotrainer["get_unique_framework_sequences"](
                    framework=framework,
                    min_seq_length=eval_config.min_seq_length,
                    max_seq_length=eval_config.max_seq_length,
                    force_download=eval_config.force_download,
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

            embedding_fn_per_residue, embedding_fn_per_sequence = biotrainer[
                "_setup_embedding_functions"
            ](
                embedder_name=embedder_name,
                output_dir=framework_output_dir,
                use_half_precision=False,
                custom_embedding_function_per_residue=_embed_residue,
                custom_embedding_function_per_sequence=_embed_sequence,
            )

            embeddings_file_per_residue = embedding_fn_per_residue(
                [sr.seq for _, sr in unique_per_residue.items()]
            )
            embeddings_file_per_sequence = embedding_fn_per_sequence(
                [sr.seq for _, sr in unique_per_sequence.items()]
            )

            biotrainer["_check_h5_file"](
                "per-residue", embeddings_file_per_residue, len(unique_per_residue)
            )
            biotrainer["_check_h5_file"](
                "per-sequence", embeddings_file_per_sequence, len(unique_per_sequence)
            )

            logger.info(f"Embeddings done in {time.time() - t0:.1f}s")

            # --- Phase 2: Prepare task configs -------------------------------
            logger.info(">>> Phase 2: Preparing task configs …")
            framework_report = biotrainer["SupervisedFrameworkReport"].empty(
                min_seq_len=eval_config.min_seq_length,
                max_seq_len=eval_config.max_seq_length,
            )
            prepared_tasks: list[tuple] = []  # (task, task_name, config_dict)

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
                    input_file=task.input_files[0],  # 1.4.0 API: input_files is a list
                    output_dir=task_output_dir,
                )
                cfg["device"] = "cpu"  # probe training is CPU-only

                maybe_result = biotrainer["SupervisedFrameworkReport"].maybe_load_existing_result(
                    embedder_name=embedder_name,
                    task_output_dir=task_output_dir,
                )
                if maybe_result:
                    logger.info(f"  Loaded existing result for {task_name}, skipping …")
                    framework_report.update_result(task_name, maybe_result)
                else:
                    prepared_tasks.append((task, task_name, cfg))

            if not prepared_tasks:
                logger.info("All tasks already have results — nothing to run!")
            else:
                task_names = [tn for _, tn, _ in prepared_tasks]
                logger.info(f"  Tasks to run ({len(prepared_tasks)}): {task_names}")

                # --- Phase 3: Run tasks in parallel --------------------------
                logger.info(
                    f">>> Phase 3: Running {len(prepared_tasks)} tasks in parallel "
                    f"({eval_config.parallel_tasks} workers) …"
                )
                t1 = time.time()
                parse_and_execute = biotrainer["parse_config_file_and_execute_run"]

                with ThreadPoolExecutor(max_workers=eval_config.parallel_tasks) as pool:
                    futures = {
                        pool.submit(_run_single_task, tn, cfg, parse_and_execute): (task, tn)
                        for task, tn, cfg in prepared_tasks
                    }
                    for future in as_completed(futures):
                        task, task_name = futures[future]
                        returned_name, result = future.result()
                        if result is not None:
                            framework_report.update_result(returned_name, result)
                            logger.info(f"  {returned_name} done")
                        else:
                            logger.error(f"  {returned_name} FAILED")

                logger.info(f"All tasks finished in {time.time() - t1:.1f}s")

            autoeval_report.add_supervised_result(framework_key, framework_report)
            logger.info(
                f"{framework_key} results added to combined report. "
                f"(total: {time.time() - overall_start:.1f}s)"
            )

    # ── Write combined report ────────────────────────────────────────────────
    autoeval_report.write(output_dir=embedder_output_dir)

    logger.info("=" * 60)
    logger.info("AUTOEVAL COMPLETE — Final Report Summary")
    logger.info("=" * 60)
    autoeval_report.summary()
