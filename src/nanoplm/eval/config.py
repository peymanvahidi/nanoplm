"""Configuration dataclass for the downstream evaluation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class EvalConfig:
    # Checkpoint path. Accepts a checkpoint-* directory or a run-* directory
    # (the latest checkpoint inside the run-* is used).
    checkpoint_dir: Union[str, Path]

    # Benchmark framework(s) to evaluate on. One or more of: pbc, flip, pgym.
    framework: List[str] = field(default_factory=lambda: ["pbc"])

    # Directory where autoeval results, H5 embeddings, and per-task outputs are written.
    output_dir: Union[str, Path] = "autoeval_output"

    # Batch size for embedding computation.
    batch_size: int = 16

    # Sequence length filters applied to benchmark datasets.
    min_seq_length: int = 0
    max_seq_length: int = 2000

    # Device override. When None, picks 'cuda' if available else 'cpu'.
    device: Optional[str] = None

    # Embedder name shown in the autoeval report. When None, defaults to the
    # checkpoint directory stem.
    embedder_name: Optional[str] = None

    # Number of parallel workers for downstream task training.
    # 0 = sequential (runs through biotrainer.autoeval_pipeline).
    # >0 = parallel mode (GPU embedding first, then CPU task training in a ThreadPoolExecutor).
    parallel_tasks: int = 0

    # Zero-shot scoring method when framework includes 'pgym'.
    # One of: masked_marginals, wt_marginals, pseudoperplexity.
    zero_shot_method: str = "masked_marginals"

    # Optional whitelist of task names (e.g. PBC-secondary_structure).
    # Parallel mode only — sequential mode warns and ignores.
    tasks: Optional[List[str]] = None

    # Force re-download of framework datasets (use when new tasks land upstream).
    force_download: bool = False
