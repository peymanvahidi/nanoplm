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
    # pbc_zeroshot_contact, pbc_supervised, pbc_supervised_contact. 'pgym'
    # scores ProteinGym DMS variants zero-shot, 'pbc_zeroshot_contact' predicts
    # contact maps from the categorical Jacobian, 'pbc_supervised_contact' fits
    # a probe on the attention maps, and 'pbc_supervised' trains task heads on
    # the frozen encoder's embeddings.
    framework: List[str] = field(default_factory=lambda: ["pgym"])

    # Which PBC_SUPERVISED datasets to run, when 'pbc_supervised' is in
    # `framework`. The framework holds seven task heads and embeds every
    # sequence they need, which is ~13 min and ~30 GB of per-residue embeddings
    # for the full set; the standing battery only wants two of them, so the
    # default restricts it. None means every dataset.
    #   secondary_structure -> carries the newPISCES364 test split
    #   scl                 -> subcellular location
    pbc_supervised_datasets: Optional[List[str]] = field(
        default_factory=lambda: ["secondary_structure", "scl"]
    )

    # Directory where autoeval results and per-framework outputs are written.
    output_dir: Union[str, Path] = "autoeval_output"

    # Device override. When None, picks 'cuda' if available else 'cpu'.
    device: Optional[str] = None

    # Embedder name shown in the autoeval report and in its filename. When
    # None, defaults to '<run-dir>-ckpt<step>' resolved from the checkpoint, so
    # two runs evaluated into one output_dir do not overwrite each other's
    # report; a role-named checkpoint takes its step from training_state.json.
    embedder_name: Optional[str] = None

    # Zero-shot scoring method for the 'pgym' framework. One of:
    # masked_marginals, wt_marginals, pseudoperplexity. Ignored by
    # pbc_zeroshot_contact, which always uses the categorical Jacobian, and by
    # pbc_supervised, which trains on embeddings.
    zero_shot_method: str = "masked_marginals"

    # Sequences scored per forward pass. Masked-marginals masks one position at
    # a time, so this batches that many masked positions into a single forward
    # (biotrainer scores them one-by-one); it also sizes the Jacobian batches
    # and the pbc_supervised embedding batches.
    zero_shot_batch_size: int = 60

    # Evaluate on subsampled task sets instead of the full benchmark. Much
    # faster; use false for final numbers.
    development_mode: bool = True

    # Pre-downloaded benchmark dataset directory, passed to biotrainer as
    # custom_storage_path. Set this on machines without internet access, and
    # while ~/.cache/biotrainer/autoeval holds datasets from biotrainer < 2.1.
    # Mutually exclusive with force_download.
    dataset_dir: Optional[str] = None

    # Delete the whole ~/.cache/biotrainer/autoeval directory and download the
    # datasets again. For a fresh copy elsewhere, use an empty dataset_dir.
    force_download: bool = False

    # Refinement depth to run this checkpoint at, overriding the depth it was
    # trained with. None = the depth in the checkpoint's model_config.yaml.
    # Works for both loopie and huginn checkpoints; a checkpoint with no
    # looping rejects it.
    loop_count: Optional[int] = None

    # Run the model through torch.compile. Zero-shot scoring is thousands of
    # forward passes, and eager spends only ~28% of its time in matmuls: the
    # rest is unfused elementwise work (RoPE, gating, residuals). Compiling
    # fuses it and roughly doubles throughput (185k -> 375k tok/s on a GH200).
    # Compiled with dynamic shapes, because every protein is a different
    # length; warmup is a few seconds and does not recur per shape.
    torch_compile: bool = True
