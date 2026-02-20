from nanoplm.pretraining.pipeline import (
    PretrainingConfig,
    run_pretraining,
)
from nanoplm.pretraining.pure_pipeline import run_pure_pretraining

_TE_IMPORT_ERROR = None
try:
    from nanoplm.pretraining.te_pipeline import run_te_pretraining
except Exception as exc:  # pragma: no cover - depends on TE/FA availability
    _TE_IMPORT_ERROR = exc

    def run_te_pretraining(*_args, **_kwargs):
        raise ImportError(
            "Transformer Engine pipeline requested but unavailable in this environment."
        ) from _TE_IMPORT_ERROR

__all__ = [
    "PretrainingConfig",
    "run_pretraining",
    "run_pure_pretraining",
    "run_te_pretraining",
]
