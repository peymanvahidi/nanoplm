"""Configuration dataclass for the downstream evaluation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class EvalConfig:
    # Checkpoint path. Accepts a checkpoint-* directory, a run-* directory, or
    # the keyword 'latest' (which scans checkpoint_root for the newest checkpoint).
    checkpoint_dir: Union[str, Path]

    # Root searched when checkpoint_dir == 'latest'.
    checkpoint_root: Union[str, Path] = "output/pretraining_checkpoints"

    # Benchmark framework to evaluate on. One of: pbc, flip.
    framework: str = "pbc"

    # Directory where autoeval results, H5 embeddings, and per-task outputs are written.
    output_dir: Union[str, Path] = "autoeval_output"

    # Fallback model config when the checkpoint directory has no model_config.yaml
    # (older checkpoints). The 'model:' block of this YAML is parsed.
    config_yaml: Union[str, Path] = "pretrain.yaml"

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
