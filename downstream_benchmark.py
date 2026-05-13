#!/usr/bin/env python3
"""Downstream benchmark script for pure-torch nanoPLM checkpoints.

Loads a pure-torch checkpoint (PureProtModernBertMLM) and runs biotrainer's
autoeval pipeline to measure downstream task performance on PBC, FLIP, or
ProteinGym (pgym) benchmarks.

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

    # ProteinGym zero-shot:
    python downstream_benchmark.py \
        --checkpoint-dir output/pretraining_checkpoints/run-04031843/checkpoint-16066 \
        --framework pgym \
        --zero-shot-method masked_marginals

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
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import MISSING, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Generator, List, Tuple

import torch
import yaml
from tqdm import tqdm

# -- nanoplm imports ---------------------------------------------------------
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
import nanoplm.pretraining.models.modern_bert.modeling as _bert_modeling
_bert_modeling.USE_TRITON_SRELU = False  # Fused srelu kernel may OOM on some H100s; plain PyTorch is sufficient for inference

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


# ---------------------------------------------------------------------------
# biotrainer imports
# ---------------------------------------------------------------------------

def _load_biotrainer_symbols() -> dict[str, Any]:
    """Import biotrainer lazily so checkpoint helpers remain importable in tests."""
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
    from biotrainer.bioengineer import CustomBioEngineerModel, BioEngineer, ZeroShotMethod
    from biotrainer.utilities import STANDARD_AAS

    return {
        "autoeval_pipeline": autoeval_pipeline,
        "get_unique_framework_sequences": get_unique_framework_sequences,
        "_setup_embedding_functions": _setup_embedding_functions,
        "_check_h5_file": _check_h5_file,
        "AutoEvalConfigBank": AutoEvalConfigBank,
        "AutoEvalReport": AutoEvalReport,
        "SupervisedFrameworkReport": SupervisedFrameworkReport,
        "Protocol": Protocol,
        "parse_config_file_and_execute_run": parse_config_file_and_execute_run,
        "CustomBioEngineerModel": CustomBioEngineerModel,
        "BioEngineer": BioEngineer,
        "ZeroShotMethod": ZeroShotMethod,
        "STANDARD_AAS": STANDARD_AAS,
    }


# ---------------------------------------------------------------------------
# ProteinGym BioEngineer wrapper
# ---------------------------------------------------------------------------

class NanoPLMBioEngineer:
    """Wraps PureProtModernBertMLM for biotrainer's zero-shot BioEngineer interface.

    Instantiated lazily inside _build_bioengineer() so that biotrainer symbols
    are already loaded before the class body is evaluated.
    """

    @staticmethod
    def build(model: PureProtModernBertMLM, tokenizer: ProtModernBertTokenizer, device: torch.device, biotrainer: dict):
        """Return a BioEngineer wrapping ``model`` for ProteinGym zero-shot scoring."""
        CustomBioEngineerModel = biotrainer["CustomBioEngineerModel"]
        BioEngineer = biotrainer["BioEngineer"]
        STANDARD_AAS = biotrainer["STANDARD_AAS"]

        import re as _re

        class _Impl(CustomBioEngineerModel):
            def __init__(self):
                super().__init__()
                self._model = model
                self._tokenizer = tokenizer
                self._device = device

            def get_name(self) -> str:
                return "nanoPLM"

            def supported_methods(self):
                ZeroShotMethod = biotrainer["ZeroShotMethod"]
                return [
                    ZeroShotMethod.MASKED_MARGINALS,
                    ZeroShotMethod.WT_MARGINALS,
                    ZeroShotMethod.PSEUDOPERPLEXITY,
                ]

            def tokenize(self, batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
                enc = self._tokenizer(batch, padding="longest", return_tensors="pt")
                return enc["input_ids"].to(self._device), enc["attention_mask"].to(self._device)

            def run_model(self, input_ids, attention_mask=None):
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
                    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs["logits"].squeeze(0)

            def get_mask_token_id(self):
                return self._tokenizer.mask_token_id

            def preprocess_sequences(self, sequences: List[str]) -> List[str]:
                return [_re.sub(r"[UZOB]", "X", s) for s in sequences]

            def aa_to_idx(self):
                vocab = self._tokenizer.get_vocab()
                return {aa: vocab[aa] for aa in STANDARD_AAS}

            def strip_special_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
                return tensor[0:-1]  # strip trailing EOS; no leading CLS in this tokenizer

        return BioEngineer.from_custom_model(_Impl(), device=device)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _checkpoint_step(checkpoint_dir: str | Path) -> int:
    name = Path(checkpoint_dir).name
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.split("-", 1)[1])
    except ValueError:
        return -1


def _find_latest_checkpoint(checkpoint_root: str | Path) -> Path:
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
    checkpoint_dir: str | Path,
    checkpoint_root: str | Path = "output/pretraining_checkpoints",
) -> Path:
    """Resolve `latest`, run directories, or explicit checkpoint paths."""
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

    training_state_path = checkpoint_dir / "training_state.json"
    checkpoint_step = 0
    if training_state_path.exists():
        checkpoint_state = json.loads(training_state_path.read_text(encoding="utf-8"))
        checkpoint_step = int(checkpoint_state.get("global_step", 0))

    # Restore step-dependent runtime state that is intentionally not persisted
    # in the model weights.
    if getattr(config, "use_prores", False) and hasattr(model.model, "update_prores_alphas"):
        model.model.update_prores_alphas(checkpoint_step)
        print(f"  ProRes restored for global_step={checkpoint_step}")

    if config.use_repo:
        repo_warmup_steps = 0
        pretrain_cfg_path = checkpoint_dir / "pretrain_config.yaml"
        if pretrain_cfg_path.exists():
            with open(pretrain_cfg_path) as f:
                pretrain_cfg = yaml.safe_load(f) or {}
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
        print(
            "  RePO "
            f"{state} (step={checkpoint_step}, repo_rope_warmup_steps={repo_warmup_steps})"
        )

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
        parse_config_file_and_execute_run = _load_biotrainer_symbols()[
            "parse_config_file_and_execute_run"
        ]
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
        help=(
            "Checkpoint path. Accepts a checkpoint-* directory, a run-* directory, "
            "or the keyword 'latest'."
        ),
    )
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default="output/pretraining_checkpoints",
        help="Root searched when --checkpoint-dir=latest.",
    )
    p.add_argument(
        "--framework",
        type=str,
        nargs="+",
        default=["pbc"],
        choices=["pbc", "flip", "pgym"],
        help="Benchmark framework(s) to evaluate on. Can specify multiple, e.g. --framework pbc pgym. "
             "'pgym' runs zero-shot ProteinGym evaluation.",
    )
    p.add_argument(
        "--zero-shot-method",
        type=str,
        default="masked_marginals",
        choices=["masked_marginals", "wt_marginals", "pseudoperplexity"],
        help="Zero-shot scoring method for ProteinGym (--framework pgym only).",
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
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Whitelist of task names to run (e.g. PBC-secondary_structure). "
             "If not set, all tasks are run.",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Force re-download of framework datasets (use when new tasks are added).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    frameworks = args.framework  # list, e.g. ["pbc"] or ["pbc", "pgym"]

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir, args.checkpoint_root)
    print(f"Resolved checkpoint: {checkpoint_dir}")
    biotrainer = _load_biotrainer_symbols()

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
    model_config = load_model_config_from_checkpoint(checkpoint_dir)
    if model_config is not None:
        print(f"Loaded model config from checkpoint: {checkpoint_dir}/model_config.yaml")
    else:
        print(f"No model_config.yaml in checkpoint, falling back to {args.config_yaml}")
        model_config = load_model_config(args.config_yaml)
    model = load_checkpoint(checkpoint_dir, model_config, device)
    tokenizer = model.tokenizer

    # Embedder name for the report
    embedder_name = args.embedder_name or checkpoint_dir.stem
    print(f"Embedder name: {embedder_name}")

    # Embedder-level output directory — all framework subdirs and the combined report live here
    embedder_dir_name = embedder_name.replace("/", "-")
    embedder_output_dir = Path(args.output_dir) / embedder_dir_name
    os.makedirs(embedder_output_dir, exist_ok=True)

    # Load existing combined report or start fresh (supports incremental runs)
    autoeval_report = biotrainer["AutoEvalReport"].loaded_or_empty(
        embedder_name=embedder_name,
        training_date=str(datetime.now().date().isoformat()),
        output_dir=embedder_output_dir,
    )

    # Build embedding closures
    def _embed_residue(seqs):
        return embed_per_residue(model, tokenizer, seqs, device, args.batch_size)

    def _embed_sequence(seqs):
        return embed_per_sequence(model, tokenizer, seqs, device, args.batch_size)

    # ── ProteinGym zero-shot (runs first — needs the GPU model throughout) ─
    if "pgym" in frameworks:
        zero_shot_method_map = {
            "masked_marginals": biotrainer["ZeroShotMethod"].MASKED_MARGINALS,
            "wt_marginals": biotrainer["ZeroShotMethod"].WT_MARGINALS,
            "pseudoperplexity": biotrainer["ZeroShotMethod"].PSEUDOPERPLEXITY,
        }
        zero_shot_method = zero_shot_method_map[args.zero_shot_method]
        print(f"\nStarting ProteinGym zero-shot evaluation: method={args.zero_shot_method}")
        print("=" * 60)

        bio_engineer = NanoPLMBioEngineer.build(model, tokenizer, device, biotrainer)

        current_progress = None
        for progress in biotrainer["autoeval_pipeline"](
            embedder_name=embedder_name,
            framework="pgym",
            zero_shot_method=zero_shot_method,
            custom_bioengineer=bio_engineer,
            output_dir=args.output_dir,
            force_download=args.force_download,
        ):
            current_progress = progress
            print(
                f"[{progress.completed_tasks}/{progress.total_tasks}] "
                f"{progress.current_framework_name}: {progress.current_task_name}"
            )

        if current_progress and current_progress.final_report:
            pgym_report = current_progress.final_report.zeroshot_results.get("PGYM")
            if pgym_report:
                autoeval_report.add_zeroshot_result("PGYM", pgym_report)
                print("\nPGYM results added to combined report.")
            else:
                print("\n[WARN] No PGYM zeroshot results found in final report.")
        else:
            print("\n[WARN] No final report generated for PGYM.")

    # ── Supervised frameworks (pbc, flip) ────────────────────────────────
    supervised_frameworks = [f for f in frameworks if f != "pgym"]
    model_freed = False

    for framework in supervised_frameworks:
        framework_key = framework.upper()
        framework_dir = f"{framework}_{args.min_seq_length}_{args.max_seq_length}"
        output_dir = embedder_output_dir / framework_dir
        os.makedirs(output_dir, exist_ok=True)

        # ── Sequential mode ───────────────────────────────────────────────
        if args.parallel_tasks <= 0:
            if args.tasks:
                print(f"[WARN] --tasks filter is not supported in sequential mode; ignoring.")
            print(f"\nStarting autoeval pipeline (sequential): framework={framework}")
            print("=" * 60)

            current_progress = None
            for progress in biotrainer["autoeval_pipeline"](
                embedder_name=embedder_name,
                framework=framework,
                output_dir=args.output_dir,
                min_seq_length=args.min_seq_length,
                max_seq_length=args.max_seq_length,
                custom_embedding_function_per_residue=_embed_residue,
                custom_embedding_function_per_sequence=_embed_sequence,
                force_download=args.force_download,
            ):
                current_progress = progress
                print(
                    f"[{progress.completed_tasks}/{progress.total_tasks}] "
                    f"{progress.current_framework_name}: {progress.current_task_name}"
                )

            if current_progress and current_progress.final_report:
                supervised_report = current_progress.final_report.supervised_results.get(framework_key)
                if supervised_report:
                    autoeval_report.add_supervised_result(framework_key, supervised_report)
                else:
                    print(f"\n[WARN] No {framework_key} supervised results in final report.")
            else:
                print(f"\n[WARN] No final report generated for {framework}.")

        # ── Parallel mode ─────────────────────────────────────────────────
        else:
            print(
                f"\nStarting autoeval pipeline (PARALLEL, {args.parallel_tasks} workers): "
                f"framework={framework}"
            )
            print("=" * 60)
            overall_start = time.time()

            # --- Phase 1: Compute embeddings (sequential, GPU) ---------------
            print("\n>>> Phase 1: Computing embeddings …")
            t0 = time.time()

            task_config_tuples, unique_per_residue, unique_per_sequence = (
                biotrainer["get_unique_framework_sequences"](
                    framework=framework,
                    min_seq_length=args.min_seq_length,
                    max_seq_length=args.max_seq_length,
                    force_download=args.force_download,
                )
            )

            if args.tasks:
                task_config_tuples = [
                    (task, cfg) for task, cfg in task_config_tuples
                    if task.combined_name() in args.tasks
                ]
                print(f"  Filtered to tasks: {[t.combined_name() for t, _ in task_config_tuples]}")

            embedding_fn_per_residue, embedding_fn_per_sequence = biotrainer["_setup_embedding_functions"](
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

            biotrainer["_check_h5_file"](
                "per-residue", embeddings_file_per_residue, len(unique_per_residue)
            )
            biotrainer["_check_h5_file"](
                "per-sequence", embeddings_file_per_sequence, len(unique_per_sequence)
            )

            print(f"Embeddings done in {time.time() - t0:.1f}s")

            # Free GPU memory after the last framework that needs the model
            if not model_freed:
                del model
                torch.cuda.empty_cache()
                model_freed = True

            # --- Phase 2: Prepare task configs --------------------------------
            print("\n>>> Phase 2: Preparing task configs …")
            prepared_tasks: list[tuple] = []
            framework_report = biotrainer["SupervisedFrameworkReport"].empty(
                min_seq_len=args.min_seq_length,
                max_seq_len=args.max_seq_length,
            )

            for task, cfg in task_config_tuples:
                task_name = task.combined_name()
                task_output_dir = output_dir / task_name

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
                )

                cfg["device"] = "cpu"  # probe training runs on CPU; GPU was only needed for embedding

                maybe_result = biotrainer["SupervisedFrameworkReport"].maybe_load_existing_result(
                    embedder_name=embedder_name,
                    task_output_dir=task_output_dir,
                )
                if maybe_result:
                    print(f"  Loaded existing result for {task_name}, skipping …")
                    framework_report.update_result(task_name, maybe_result)
                else:
                    prepared_tasks.append((task, task_name, cfg))

            if not prepared_tasks:
                print("All tasks already have results — nothing to run!")
            else:
                task_names = [tn for _, tn, _ in prepared_tasks]
                print(f"  Tasks to run ({len(prepared_tasks)}): {task_names}")

                # --- Phase 3: Run tasks in parallel ---------------------------
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
                            framework_report.update_result(task_name, result)
                            print(f"{returned_name} done")
                        else:
                            print(f"{returned_name} FAILED")

                print(f"All tasks finished in {time.time() - t1:.1f}s")

            autoeval_report.add_supervised_result(framework_key, framework_report)
            print(f"{framework_key} results added to combined report. (total: {time.time() - overall_start:.1f}s)")

    # Write the combined report (PBC + PGYM in one file)
    autoeval_report.write(output_dir=embedder_output_dir)

    print(f"\n{'=' * 60}")
    print(f"AUTOEVAL COMPLETE — Final Report Summary")
    print(f"{'=' * 60}")
    autoeval_report.summary()


if __name__ == "__main__":
    main()
