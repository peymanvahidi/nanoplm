"""Task 15 verification: TE pipeline token-budget duration.

The real Task-15 check is a source-text assertion that the TE pipeline no
longer references the removed epoch-based duration machinery (``num_epochs``)
or the old batch-field name (``global_batch_size``).  That check does NOT need
``transformer_engine`` installed: ``find_spec(...).origin`` locates the module
file without executing its body, so we can read and grep the source even when
TE's heavy CUDA deps are absent.

A separate smoke test actually imports the module to confirm it loads; it is
skipped when ``transformer_engine`` is unavailable (the module does
``import transformer_engine.pytorch as te`` at the top).
"""

import importlib
import importlib.util

import pytest


def _te_pipeline_source() -> str:
    spec = importlib.util.find_spec("nanoplm.pretraining.te_pipeline")
    assert spec is not None and spec.origin is not None, (
        "Could not locate nanoplm.pretraining.te_pipeline on the import path"
    )
    with open(spec.origin, encoding="utf-8") as fh:
        return fh.read()


def test_te_pipeline_drops_legacy_duration_names():
    # Always runs: pure source-text check, no TE import required.
    text = _te_pipeline_source()
    assert "num_epochs" not in text, "TE pipeline must no longer reference num_epochs"
    assert "global_batch_size" not in text, "TE must use renamed global_batch_tokens"


def test_te_pipeline_imports():
    # TE deps may be absent in this environment; skip the actual import then.
    pytest.importorskip("transformer_engine")
    mod = importlib.import_module("nanoplm.pretraining.te_pipeline")
    assert mod is not None
