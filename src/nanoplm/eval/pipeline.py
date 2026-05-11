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
from nanoplm.utils.logger import logger


def _load_sequential_symbols() -> dict[str, Any]:
    """Import biotrainer symbols needed for sequential mode only.

    Sequential mode goes through the public ``autoeval_pipeline`` generator,
    which is the supported, stable surface of biotrainer's autoeval package.
    """
    from biotrainer.autoeval import autoeval_pipeline

    return {"autoeval_pipeline": autoeval_pipeline}


def _load_parallel_symbols() -> dict[str, Any]:
    """Import private biotrainer symbols needed for parallel mode.

    Parallel mode reaches into biotrainer internals to compute embeddings once
    and then run each downstream task in a ThreadPoolExecutor. These private
    names were available in older biotrainer releases; if they are missing
    here, the installed biotrainer is too new (or too old) for parallel mode.
    """
    try:
        from biotrainer.autoeval.autoeval import (
            _check_h5_file,
            _setup_embedding_functions,
            get_unique_framework_sequences,
        )
        from biotrainer.autoeval.config_bank import AutoEvalConfigBank
        from biotrainer.autoeval.report_manager import ReportManager
        from biotrainer.protocols import Protocol
        from biotrainer.utilities.executer import parse_config_file_and_execute_run
    except ImportError as exc:
        raise RuntimeError(
            "Parallel-mode eval (parallel_tasks > 0) is incompatible with the "
            "installed biotrainer version. The required private symbols "
            "(_check_h5_file, _setup_embedding_functions, AutoEvalConfigBank, "
            "ReportManager, Protocol, parse_config_file_and_execute_run) are "
            "no longer exposed. Use parallel_tasks: 0 (sequential mode) or pin "
            "an older biotrainer release."
        ) from exc

    return {
        "get_unique_framework_sequences": get_unique_framework_sequences,
        "_setup_embedding_functions": _setup_embedding_functions,
        "_check_h5_file": _check_h5_file,
        "AutoEvalConfigBank": AutoEvalConfigBank,
        "ReportManager": ReportManager,
        "Protocol": Protocol,
        "parse_config_file_and_execute_run": parse_config_file_and_execute_run,
    }


def _run_single_task(task_name: str, config: dict) -> tuple[str, dict | None]:
    """Run a single biotrainer task. Returns ``(task_name, result_dict)``."""
    try:
        parse_config_file_and_execute_run = _load_parallel_symbols()[
            "parse_config_file_and_execute_run"
        ]
        result = parse_config_file_and_execute_run(config=config)
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


def run_eval(eval_config: EvalConfig) -> None:
    """Run biotrainer autoeval on a pure-torch nanoPLM checkpoint."""
    checkpoint_dir = resolve_checkpoint_dir(
        eval_config.checkpoint_dir,
        eval_config.checkpoint_root,
    )
    logger.info(f"Resolved checkpoint: {checkpoint_dir}")

    # Sequential mode uses the public autoeval_pipeline; parallel mode reaches
    # into biotrainer privates so load those only when parallel_tasks > 0.
    sequential_syms = _load_sequential_symbols()
    parallel_syms: dict[str, Any] = {}
    if eval_config.parallel_tasks > 0:
        parallel_syms = _load_parallel_symbols()

    device = _resolve_device(eval_config.device)
    logger.info(f"Using device: {device}")

    # Prefer model_config.yaml inside the checkpoint (self-contained), fall back
    # to the external pretrain.yaml.
    model_config = load_model_config_from_checkpoint(checkpoint_dir)
    if model_config is not None:
        logger.info(f"Loaded model config from checkpoint: {checkpoint_dir}/model_config.yaml")
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

    def _embed_residue(seqs):
        return embed_per_residue(model, tokenizer, seqs, device, batch_size)

    def _embed_sequence(seqs):
        return embed_per_sequence(model, tokenizer, seqs, device, batch_size)

    # ── Sequential mode ─────────────────────────────────────────────────────
    if eval_config.parallel_tasks <= 0:
        logger.info(
            f"Starting autoeval pipeline (sequential): framework={eval_config.framework}"
        )
        logger.info("=" * 60)

        current_progress = None
        for progress in sequential_syms["autoeval_pipeline"](
            embedder_name=embedder_name,
            framework=eval_config.framework,
            output_dir=str(eval_config.output_dir),
            min_seq_length=eval_config.min_seq_length,
            max_seq_length=eval_config.max_seq_length,
            custom_embedding_function_per_residue=_embed_residue,
            custom_embedding_function_per_sequence=_embed_sequence,
        ):
            current_progress = progress
            logger.info(
                f"[{progress.completed_tasks}/{progress.total_tasks}] "
                f"{progress.current_framework_name}: {progress.current_task_name}"
            )

        if current_progress and current_progress.final_report:
            logger.info("=" * 60)
            logger.info("AUTOEVAL COMPLETE — Final Report Summary")
            logger.info("=" * 60)
            current_progress.final_report.summary()
        else:
            logger.warning("No final report generated.")
        return

    # ── Parallel mode ───────────────────────────────────────────────────────
    logger.info(
        f"Starting autoeval pipeline (PARALLEL, {eval_config.parallel_tasks} workers): "
        f"framework={eval_config.framework}"
    )
    logger.info("=" * 60)
    overall_start = time.time()

    embedder_dir_name = embedder_name.replace("/", "-")
    framework_dir = (
        f"{eval_config.framework}_{eval_config.min_seq_length}_{eval_config.max_seq_length}"
    )
    output_dir = Path(eval_config.output_dir) / embedder_dir_name / framework_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Phase 1: Compute embeddings (sequential, GPU) ---------------------
    logger.info(">>> Phase 1: Computing embeddings …")
    t0 = time.time()

    task_config_tuples, unique_per_residue, unique_per_sequence = (
        parallel_syms["get_unique_framework_sequences"](
            framework=eval_config.framework,
            min_seq_length=eval_config.min_seq_length,
            max_seq_length=eval_config.max_seq_length,
        )
    )

    embedding_fn_per_residue, embedding_fn_per_sequence = parallel_syms[
        "_setup_embedding_functions"
    ](
        embedder_name=embedder_name,
        output_dir=output_dir,
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

    parallel_syms["_check_h5_file"](
        "per-residue", embeddings_file_per_residue, len(unique_per_residue)
    )
    parallel_syms["_check_h5_file"](
        "per-sequence", embeddings_file_per_sequence, len(unique_per_sequence)
    )

    logger.info(f"Embeddings done in {time.time() - t0:.1f}s")

    # Free GPU memory — the model is no longer needed for downstream training.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Phase 2: Prepare task configs -------------------------------------
    logger.info(">>> Phase 2: Preparing task configs …")
    prepared_tasks: list[tuple] = []  # (task, task_name, config_dict)
    report_manager = parallel_syms["ReportManager"](
        embedder_name=embedder_name,
        training_date=str(datetime.now().date().isoformat()),
        min_seq_len=eval_config.min_seq_length,
        max_seq_len=eval_config.max_seq_length,
    )

    for task, cfg in task_config_tuples:
        task_name = task.combined_name()
        task_output_dir = output_dir / task_name

        protocol = parallel_syms["Protocol"]
        if protocol.from_string(cfg["protocol"]) in protocol.using_per_sequence_embeddings():
            task_emb_file = embeddings_file_per_sequence
        else:
            task_emb_file = embeddings_file_per_residue

        cfg = parallel_syms["AutoEvalConfigBank"].add_custom_values_to_config(
            config=cfg,
            embedder_name=embedder_name,
            embeddings_file=task_emb_file,
            input_file=task.input_file,
            output_dir=task_output_dir,
        )

        maybe_result = report_manager.maybe_load_existing_result(
            task_output_dir=task_output_dir
        )
        if maybe_result:
            logger.info(f"  Loaded existing result for {task_name}, skipping …")
            report_manager.add_result(task=task, result_dict=maybe_result)
        else:
            prepared_tasks.append((task, task_name, cfg))

    if not prepared_tasks:
        logger.info("All tasks already have results — nothing to run!")
    else:
        task_names = [tn for _, tn, _ in prepared_tasks]
        logger.info(f"  Tasks to run ({len(prepared_tasks)}): {task_names}")

        # --- Phase 3: Run tasks in parallel --------------------------------
        logger.info(
            f">>> Phase 3: Running {len(prepared_tasks)} tasks in parallel "
            f"({eval_config.parallel_tasks} workers) …"
        )
        t1 = time.time()

        with ThreadPoolExecutor(max_workers=eval_config.parallel_tasks) as pool:
            futures = {
                pool.submit(_run_single_task, task_name, cfg): (task, task_name)
                for task, task_name, cfg in prepared_tasks
            }
            for future in as_completed(futures):
                task, task_name = futures[future]
                returned_name, result = future.result()
                if result is not None:
                    report_manager.add_result(task=task, result_dict=result)
                    logger.info(f"  ✓ {returned_name} done")
                else:
                    logger.error(f"  ✗ {returned_name} FAILED")

        logger.info(f"All tasks finished in {time.time() - t1:.1f}s")

    # --- Phase 4: Write report ---------------------------------------------
    report = report_manager.write(output_dir=output_dir.parent)

    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info(f"AUTOEVAL COMPLETE — Final Report Summary  (total: {total_time:.1f}s)")
    logger.info("=" * 60)
    report.summary()
