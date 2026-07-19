"""Where each config surface's documentation comes from.

Paths are absolute, anchored to the repo root via this file's location, so the
generator and its tests resolve sources correctly regardless of the current
working directory (matching the rest of the test suite's ``Path(__file__)``
convention).
"""
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SRC = REPO_ROOT / "src" / "nanoplm"

@dataclass(frozen=True)
class TemplateSource:
    # Path to a packaged ``.yaml`` template, or a ``.py`` file that embeds one.
    file: str
    # For ``.py`` sources: the module-level constant or function that holds the
    # template string. For ``.yaml`` sources: unused (pass ``""``).
    symbol: str = ""

@dataclass(frozen=True)
class DataclassSource:
    file: str
    classname: str

# Surfaces whose docs live in a comment-rich get-yaml template string.
TEMPLATE_SOURCES = {
    "data":     TemplateSource(str(REPO_SRC / "cli/templates/data_pretrain.yaml")),
    "pretrain": TemplateSource(str(REPO_SRC / "cli/templates/pretrain.yaml")),
    # Distill still ships as an embedded string in distill.get_yaml (for now).
    "distill":  TemplateSource(str(REPO_SRC / "cli/distill.py"), "get_yaml"),
}

# Surfaces whose docs live in a dataclass with per-field leading comments.
DATACLASS_SOURCES = {
    "eval": DataclassSource(str(REPO_SRC / "eval/config.py"), "EvalConfig"),
}

# Dataclasses used by the parity gate to prove no field is undocumented/invented.
PARITY_SOURCES = {
    "pretraining":  DataclassSource(str(REPO_SRC / "pretraining/config.py"),     "PretrainingConfig"),
    "resume":       DataclassSource(str(REPO_SRC / "pretraining/config.py"),     "ResumeConfig"),
    "distillation": DataclassSource(str(REPO_SRC / "distillation/pipeline.py"),  "DistillationConfig"),
    "eval":         DataclassSource(str(REPO_SRC / "eval/config.py"),            "EvalConfig"),
}

COMPAT_MATRIX = str(REPO_SRC / "config/recycling_compatibility.yaml")

# Fields that exist on the dataclass but are intentionally NOT user-facing
# (auto-computed/internal) — excluded from the parity check.
PARITY_INTERNAL_ALLOWLIST = {
    # Derived at runtime by cli/pretrain.py::_populate_batch_setup from
    # global_batch_tokens / micro_batch_seqs / manifest.max_seq_len / world_size.
    # They exist on PretrainingConfig only so the pipelines can read them back;
    # a user never sets them (and _load_pretrain_config would happily accept
    # them, but the get-yaml template never emits them).
    "pretraining": {"inferred_grad_accum_steps", "global_batch_seqs",
                    "achieved_global_batch_tokens"},
    "resume": set(),
    "distillation": {
        # Auto-populated from the .data_manifest / data-prep params.yaml — not
        # set by the user in distill.yaml's `distillation:` section.
        "train_fasta", "val_fasta", "train_h5_prefix", "val_h5_prefix",
        "train_sequences", "val_sequences", "max_seq_len", "max_seqs_num",
        "val_ratio", "on_the_fly", "sharded",
    },
    "eval": set(),
}
