#!/usr/bin/env python3
"""Downstream benchmark script for pure-torch nanoPLM checkpoints.

Loads a pure-torch checkpoint (PureProtModernBertMLM) and runs biotrainer's
autoeval pipeline to measure downstream task performance on PBC or FLIP
benchmarks.

Supports both sequential execution and parallel task training (GPU embedding
first, then CPU-based biotrainer training tasks in parallel).

Usage
-----
    # Sequential (default):
    python downstream_benchmark.py \
        --checkpoint-dir output/pretraining_checkpoints/run-04031843/checkpoint-16066 \
        --framework pbc

    # Parallel (4 workers):
    python downstream_benchmark.py \
        --checkpoint-dir output/pretraining_checkpoints/run-04031843/checkpoint-16066 \
        --framework pbc \
        --parallel-tasks 4

    # Custom settings:
    python downstream_benchmark.py \
        --checkpoint-dir output/pretraining_checkpoints/run-04031843/checkpoint-16066 \
        --framework pbc \
        --batch-size 32 \
        --max-seq-length 2000 \
        --output-dir autoeval_output
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, Generator, Tuple

import torch
import yaml
from tqdm import tqdm

# -- nanoplm imports ---------------------------------------------------------
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

# -- biotrainer imports ------------------------------------------------------
from biotrainer.autoeval import autoeval_pipeline
from biotrainer.autoeval.autoeval import (
    get_unique_framework_sequences,
    _setup_embedding_functions,
    _check_h5_file,
)
from biotrainer.autoeval.config_bank import AutoEvalConfigBank
from biotrainer.autoeval.report_manager import ReportManager
from biotrainer.protocols import Protocol
from biotrainer.utilities.executer import parse_config_file_and_execute_run


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_model_config(yaml_path: str | Path = "pretrain.yaml") -> ProtModernBertMLMConfig:
    """Build a ProtModernBertMLMConfig from pretrain.yaml model section.

    All fields in the YAML ``model:`` block map 1:1 to dataclass fields.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    m = cfg["model"]
    return _model_config_from_dict(m)


def load_model_config_from_checkpoint(checkpoint_dir: str | Path) -> ProtModernBertMLMConfig | None:
    """Try to load model_config.yaml from a checkpoint directory.

    Returns ``None`` if the file does not exist (older checkpoints).
    """
    cfg_path = Path(checkpoint_dir) / "model_config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        m = yaml.safe_load(f)
    return _model_config_from_dict(m)


def _model_config_from_dict(m: dict) -> ProtModernBertMLMConfig:
    """Construct a ProtModernBertMLMConfig from a flat dict of model fields."""
    return ProtModernBertMLMConfig(
        vocab_size=m.get("vocab_size", 32),
        hidden_size=m["hidden_size"],
        intermediate_size=m["intermediate_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        mlp_activation=m.get("mlp_activation", "swiglu"),
        mlp_dropout=m.get("mlp_dropout", 0.0),
        mlp_bias=m.get("mlp_bias", False),
        no_mlp_on_first_layer=m.get("no_mlp_on_first_layer", True),
        attention_bias=m.get("attention_bias", False),
        attention_dropout=m.get("attention_dropout", 0.0),
        classifier_activation=m.get("classifier_activation", "gelu"),
        use_resid_lambdas=m.get("use_resid_lambdas", False),
        use_x0_lambdas=m.get("use_x0_lambdas", False),
        use_qk_norm=m.get("use_qk_norm", False),
        use_canon_layers=m.get("use_canon_layers", False),
        canon_layers_mode=m.get("canon_layers_mode", "abcd"),
        canon_layers_kernel_size=m.get("canon_layers_kernel_size", None),
        use_repo=m.get("use_repo", False),
        repo_after_n_layers=m.get("repo_after_n_layers", 3),
        use_mhc_lite=m.get("use_mhc_lite", False),
        mhc_n_streams=m.get("mhc_n_streams", 4),
        mhc_triton_fused=m.get("mhc_triton_fused", False),
        mhc_lite_wrapping_level=m.get("mhc_lite_wrapping_level", "layer"),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_dir: str | Path,
    config: ProtModernBertMLMConfig,
    device: torch.device,
) -> PureProtModernBertMLM:
    """Instantiate PureProtModernBertMLM and load checkpoint weights."""
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "pytorch_model.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print(f"Loading pure-torch checkpoint from {model_path} ...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    model = PureProtModernBertMLM(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Activate RePO learned positions if the model was trained with it
    if config.use_repo:
        model.model.repo_active = True
        print(f"  RePO activated (repo_after_n_layers={config.repo_after_n_layers})")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {n_params:.1f}M parameters on {device}")
    return model


# ---------------------------------------------------------------------------
# Embedding functions for biotrainer autoeval
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_per_residue(
    model: PureProtModernBertMLM,
    tokenizer: ProtModernBertTokenizer,
    sequences: Iterable[str],
    device: torch.device,
    batch_size: int = 16,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield (sequence, per_residue_embedding) for each input sequence.

    Per-residue embedding shape: ``(seq_len, hidden_size)``.
    Special tokens (EOS) are stripped; only amino-acid positions are returned.
    """
    seq_list = list(sequences)

    for i in tqdm(range(0, len(seq_list), batch_size), desc="Embedding per-residue"):
        batch_seqs = seq_list[i : i + batch_size]
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Run backbone (encoder only, no MLM head) via the SDPA fallback path.
        # This path expects (batch, seq_len) padded tensors with attention_mask.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.model(input_ids=input_ids, attention_mask=attention_mask)

        # hidden shape: (batch, padded_seq_len, hidden_size)
        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            # Extract only amino-acid token embeddings (skip trailing EOS + padding)
            emb = hidden[j, :seq_len, :].float().cpu()
            yield seq, emb


@torch.no_grad()
def embed_per_sequence(
    model: PureProtModernBertMLM,
    tokenizer: ProtModernBertTokenizer,
    sequences: Iterable[str],
    device: torch.device,
    batch_size: int = 16,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield (sequence, per_sequence_embedding) for each input sequence.

    Per-sequence embedding shape: ``(hidden_size,)`` — mean pool over residue
    token positions (excluding special tokens).
    """
    seq_list = list(sequences)

    for i in tqdm(range(0, len(seq_list), batch_size), desc="Embedding per-sequence"):
        batch_seqs = seq_list[i : i + batch_size]
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.model(input_ids=input_ids, attention_mask=attention_mask)

        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            # Mean pool over amino-acid tokens only
            emb = hidden[j, :seq_len, :].mean(dim=0).float().cpu()
            yield seq, emb


# ---------------------------------------------------------------------------
# Parallel task execution
# ---------------------------------------------------------------------------

def _run_single_task(task_name: str, config: dict) -> tuple[str, dict | None]:
    """Run a single biotrainer task. Returns ``(task_name, result_dict)``."""
    try:
        result = parse_config_file_and_execute_run(config=config)
        return task_name, result
    except Exception as e:
        print(f"[ERROR] Task {task_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return task_name, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run downstream benchmarks on a pure-torch nanoPLM checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory (contains pytorch_model.bin).",
    )
    p.add_argument(
        "--framework",
        type=str,
        default="pbc",
        choices=["pbc", "flip"],
        help="Benchmark framework to evaluate on.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="autoeval_output",
        help="Directory to save autoeval results.",
    )
    p.add_argument(
        "--config-yaml",
        type=str,
        default="pretrain.yaml",
        help="Path to pretrain.yaml for model config.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding computation.",
    )
    p.add_argument(
        "--min-seq-length",
        type=int,
        default=0,
        help="Minimum sequence length filter for benchmark datasets.",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=2000,
        help="Maximum sequence length filter for benchmark datasets.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detected if not specified).",
    )
    p.add_argument(
        "--embedder-name",
        type=str,
        default=None,
        help="Name for the embedder in the report. Defaults to checkpoint dir name.",
    )
    p.add_argument(
        "--parallel-tasks",
        type=int,
        default=0,
        help="Number of parallel workers for downstream task training. "
             "0 = sequential (via autoeval_pipeline), >0 = parallel mode.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load config: prefer model_config.yaml inside the checkpoint (self-contained),
    # fall back to the external pretrain.yaml.
    model_config = load_model_config_from_checkpoint(args.checkpoint_dir)
    if model_config is not None:
        print(f"Loaded model config from checkpoint: {args.checkpoint_dir}/model_config.yaml")
    else:
        print(f"No model_config.yaml in checkpoint, falling back to {args.config_yaml}")
        model_config = load_model_config(args.config_yaml)
    model = load_checkpoint(args.checkpoint_dir, model_config, device)
    tokenizer = model.tokenizer

    # Embedder name for the report
    embedder_name = args.embedder_name or Path(args.checkpoint_dir).resolve().stem
    print(f"Embedder name: {embedder_name}")

    # Build embedding closures
    def _embed_residue(seqs):
        return embed_per_residue(model, tokenizer, seqs, device, args.batch_size)

    def _embed_sequence(seqs):
        return embed_per_sequence(model, tokenizer, seqs, device, args.batch_size)

    # ── Sequential mode (original behavior via autoeval_pipeline) ─────────
    if args.parallel_tasks <= 0:
        print(f"\nStarting autoeval pipeline (sequential): framework={args.framework}")
        print("=" * 60)

        current_progress = None
        for progress in autoeval_pipeline(
            embedder_name=embedder_name,
            framework=args.framework,
            output_dir=args.output_dir,
            min_seq_length=args.min_seq_length,
            max_seq_length=args.max_seq_length,
            custom_embedding_function_per_residue=_embed_residue,
            custom_embedding_function_per_sequence=_embed_sequence,
        ):
            current_progress = progress
            print(
                f"[{progress.completed_tasks}/{progress.total_tasks}] "
                f"{progress.current_framework_name}: {progress.current_task_name}"
            )

        # Final report
        if current_progress and current_progress.final_report:
            print("\n" + "=" * 60)
            print("AUTOEVAL COMPLETE — Final Report Summary")
            print("=" * 60)
            current_progress.final_report.summary()
        else:
            print("\n[WARN] No final report generated.")
        return

    # ── Parallel mode ────────────────────────────────────────────────────
    print(
        f"\nStarting autoeval pipeline (PARALLEL, {args.parallel_tasks} workers): "
        f"framework={args.framework}"
    )
    print("=" * 60)
    overall_start = time.time()

    # Build the same output_dir structure that autoeval_pipeline uses
    embedder_dir_name = embedder_name.replace("/", "-")
    framework_dir = f"{args.framework}_{args.min_seq_length}_{args.max_seq_length}"
    output_dir = Path(args.output_dir) / embedder_dir_name / framework_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Phase 1: Compute embeddings (sequential, GPU) -------------------
    print("\n>>> Phase 1: Computing embeddings …")
    t0 = time.time()

    # Use biotrainer's internal helpers to get tasks, sequences, and setup
    task_config_tuples, unique_per_residue, unique_per_sequence = (
        get_unique_framework_sequences(
            framework=args.framework,
            min_seq_length=args.min_seq_length,
            max_seq_length=args.max_seq_length,
        )
    )

    embedding_fn_per_residue, embedding_fn_per_sequence = _setup_embedding_functions(
        embedder_name=embedder_name,
        output_dir=output_dir,
        use_half_precision=False,
        custom_embedding_function_per_residue=_embed_residue,
        custom_embedding_function_per_sequence=_embed_sequence,
    )

    # Actually compute embeddings (or skip if H5 already exists)
    embeddings_file_per_residue = embedding_fn_per_residue(
        [sr.seq for _, sr in unique_per_residue.items()]
    )
    embeddings_file_per_sequence = embedding_fn_per_sequence(
        [sr.seq for _, sr in unique_per_sequence.items()]
    )

    _check_h5_file("per-residue", embeddings_file_per_residue, len(unique_per_residue))
    _check_h5_file("per-sequence", embeddings_file_per_sequence, len(unique_per_sequence))

    print(f"Embeddings done in {time.time() - t0:.1f}s")

    # Free GPU memory — the model is no longer needed for downstream training
    del model
    torch.cuda.empty_cache()

    # --- Phase 2: Prepare task configs -----------------------------------
    print("\n>>> Phase 2: Preparing task configs …")
    prepared_tasks: list[tuple] = []  # (task, task_name, config_dict)
    report_manager = ReportManager(
        embedder_name=embedder_name,
        training_date=str(datetime.now().date().isoformat()),
        min_seq_len=args.min_seq_length,
        max_seq_len=args.max_seq_length,
    )

    for task, cfg in task_config_tuples:
        task_name = task.combined_name()
        task_output_dir = output_dir / task_name

        # Pick the right embeddings file based on protocol
        if Protocol.from_string(cfg["protocol"]) in Protocol.using_per_sequence_embeddings():
            task_emb_file = embeddings_file_per_sequence
        else:
            task_emb_file = embeddings_file_per_residue

        cfg = AutoEvalConfigBank.add_custom_values_to_config(
            config=cfg,
            embedder_name=embedder_name,
            embeddings_file=task_emb_file,
            input_file=task.input_file,
            output_dir=task_output_dir,
        )

        # Check for existing results (resume support)
        maybe_result = report_manager.maybe_load_existing_result(
            task_output_dir=task_output_dir
        )
        if maybe_result:
            print(f"  Loaded existing result for {task_name}, skipping …")
            report_manager.add_result(task=task, result_dict=maybe_result)
        else:
            prepared_tasks.append((task, task_name, cfg))

    if not prepared_tasks:
        print("All tasks already have results — nothing to run!")
    else:
        task_names = [tn for _, tn, _ in prepared_tasks]
        print(f"  Tasks to run ({len(prepared_tasks)}): {task_names}")

        # --- Phase 3: Run tasks in parallel ------------------------------
        print(
            f"\n>>> Phase 3: Running {len(prepared_tasks)} tasks in parallel "
            f"({args.parallel_tasks} workers) …"
        )
        t1 = time.time()

        with ThreadPoolExecutor(max_workers=args.parallel_tasks) as pool:
            futures = {
                pool.submit(_run_single_task, task_name, cfg): (task, task_name)
                for task, task_name, cfg in prepared_tasks
            }
            for future in as_completed(futures):
                task, task_name = futures[future]
                returned_name, result = future.result()
                if result is not None:
                    report_manager.add_result(task=task, result_dict=result)
                    print(f"  ✓ {returned_name} done")
                else:
                    print(f"  ✗ {returned_name} FAILED")

        print(f"All tasks finished in {time.time() - t1:.1f}s")

    # --- Phase 4: Write report -------------------------------------------
    report = report_manager.write(output_dir=output_dir.parent)

    total_time = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"AUTOEVAL COMPLETE — Final Report Summary  (total: {total_time:.1f}s)")
    print(f"{'=' * 60}")
    report.summary()


if __name__ == "__main__":
    main()
