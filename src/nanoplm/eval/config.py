"""Configuration dataclass for the downstream evaluation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class EvalConfig:
    # Checkpoint path. Accepts a checkpoint-* directory or a run-* directory
    # (the latest checkpoint inside the run-* is used).
    checkpoint_dir: Union[str, Path]

    # Benchmark framework(s) to evaluate on. One or more of: pgym,
    # pbc_zeroshot_contact. Both are zero-shot: 'pgym' scores ProteinGym DMS
    # variants, 'pbc_zeroshot_contact' predicts contact maps from the
    # categorical Jacobian.
    framework: List[str] = field(default_factory=lambda: ["pgym"])

    # Directory where autoeval results and per-framework outputs are written.
    output_dir: Union[str, Path] = "autoeval_output"

    # Device override. When None, picks 'cuda' if available else 'cpu'.
    device: Optional[str] = None

    # Embedder name shown in the autoeval report. When None, defaults to the
    # checkpoint directory stem.
    embedder_name: Optional[str] = None

    # Zero-shot scoring method for the 'pgym' framework. One of:
    # masked_marginals, wt_marginals, pseudoperplexity. Ignored by
    # pbc_zeroshot_contact, which always uses the categorical Jacobian.
    zero_shot_method: str = "masked_marginals"

    # Sequences scored per forward pass. Masked-marginals masks one position at
    # a time, so this batches that many masked positions into a single forward
    # (biotrainer scores them one-by-one); it also sizes the Jacobian batches.
    zero_shot_batch_size: int = 60

    # Evaluate on subsampled task sets instead of the full benchmark. Much
    # faster; use false for final numbers.
    development_mode: bool = True

    # Pre-downloaded benchmark dataset directory, passed to biotrainer as
    # custom_storage_path. Set this on machines without internet access.
    # Mutually exclusive with force_download.
    dataset_dir: Optional[str] = None

    # Force re-download of framework datasets (use when new tasks land upstream).
    force_download: bool = False
