"""Disable quack's GEMM autotuner for eval.

Why. Scoring a MoE checkpoint runs the sonicmoe/quack grouped GEMM, whose
autotuner keys on every tensor's shape (`quack/autotuner.py:613-620`). Eval
batches variable-length proteins under `dynamic=True`, so almost every batch is
a new key and pays a fresh search. Measured on one GH200 at E=47/top_k 3/
inter 672: ~11 s per new shape against 6 ms with the search off, for the same
steady-state latency (2.48-3.26 ms tuned, 1.61-2.76 ms untuned). On the first
MoE eval this was 24 of 25 py-spy stacks.

Training is unaffected: it sees a small fixed set of shapes, pays the search
once, and reuses `$FSROOT/cache/quack`. This module lives in the eval-only tree
and is never imported by the training pipeline.

`sonicmoe.functional` binds the four entry points at import time
(`from quack.gemm_interface import gemm, gemm_act, gemm_dgated, gemm_gated`),
so patching `quack.gemm_interface` alone would not reach it; both module
namespaces are rebound here.

Set NANOPLM_EVAL_QUACK_TUNED=1 to restore the autotuner.
"""
import functools
import os

_NAMES = ("gemm", "gemm_act", "gemm_gated", "gemm_dgated")
_applied = False


def _force_untuned(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        kwargs["tuned"] = False
        return fn(*args, **kwargs)
    return wrapper


def disable_quack_autotune() -> bool:
    """Rebind quack's GEMM entry points to skip autotuning. Returns True if applied."""
    global _applied
    if _applied or os.environ.get("NANOPLM_EVAL_QUACK_TUNED") == "1":
        return False
    try:
        import quack.gemm_interface as gi
    except Exception:
        return False  # no quack installed: dense-only checkpoint, nothing to do
    for name in _NAMES:
        original = getattr(gi, name, None)
        if original is None or getattr(original, "__wrapped__", None) is not None:
            continue
        setattr(gi, name, _force_untuned(original))
    try:
        import sonicmoe.functional as smf
    except Exception:
        smf = None
    if smf is not None:
        for name in _NAMES:
            if hasattr(smf, name):
                setattr(smf, name, getattr(gi, name))
    _applied = True
    return True
