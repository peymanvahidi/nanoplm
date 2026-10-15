"""Feature detection and global flags for ModernBERT.

Centralises all CUDA-dependent imports and capability checks so the core
model files don't need to repeat them.
"""

from __future__ import annotations

import torch

# -- Canon / mHC Triton op registration (side-effect imports) ----------------

if torch.cuda.is_available():
    # Registers torch.ops.nanoplm_mhc::* used by MHCLiteBlock's Triton path.
    from . import mhc_triton_ops as _mhc_triton_ops  # noqa: F401
    # Registers torch.ops.nanoplm_canon::* — Triton varlen depthwise conv.
    from . import canon_ops as _canon_ops
    from .canon_ops import varlen_canon_conv as _varlen_canon_conv
    from .canon_ops import varlen_ln_canon_conv as _varlen_ln_canon_conv
    from .canon_ops import varlen_rms_canon_conv as _varlen_rms_canon_conv

    # CuTeDSL backend for the RMS+Canon and plain canon paths (sm90, bf16,
    # C in 512-3072 with C%256==0, K=7).
    # NANOPLM_CANON_BACKEND=triton disables it; =cute requires it; default
    # "auto" uses it whenever `canon_cute_ops.is_compatible()` holds.
    import os as _os

    _CANON_BACKEND = _os.getenv("NANOPLM_CANON_BACKEND", "auto").strip().lower()
    _canon_cute_ops = None
    if _CANON_BACKEND in ("auto", "cute"):
        try:
            if torch.cuda.get_device_capability() == (9, 0):
                from . import canon_cute_ops as _canon_cute_ops
        except Exception:
            if _CANON_BACKEND == "cute":
                raise
            _canon_cute_ops = None
else:
    _mhc_triton_ops = None  # type: ignore[assignment]
    _canon_ops = None  # type: ignore[assignment]
    _canon_cute_ops = None  # type: ignore[assignment]
    _CANON_BACKEND = "triton"
    _varlen_canon_conv = None  # type: ignore[assignment]
    _varlen_ln_canon_conv = None  # type: ignore[assignment]
    _varlen_rms_canon_conv = None  # type: ignore[assignment]

# -- Flash Attention detection -------------------------------------------------

_HAS_FLASH_VARLEN = False
_flash_varlen_fn = None
_FLASH_HAS_DROPOUT = False

if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
    try:
        # FA3 (H100 / sm90)
        import flash_attn_interface as _fa3

        _flash_varlen_fn = _fa3.flash_attn_varlen_func
        _HAS_FLASH_VARLEN = True
        _FLASH_HAS_DROPOUT = False  # FA3 removed dropout_p
    except ImportError:
        pass

if not _HAS_FLASH_VARLEN:
    try:
        # FA2 (Ampere+, RTX 30xx/40xx/50xx)
        from flash_attn import flash_attn_varlen_func as _flash_varlen_fn

        _HAS_FLASH_VARLEN = True
        _FLASH_HAS_DROPOUT = True  # FA2 supports dropout_p
    except ImportError:
        pass

# -- Triton sReLU availability -------------------------------------------------

import os as _os_srelu  # noqa: E402

USE_TRITON_SRELU = False

if torch.cuda.is_available() and (
    torch.cuda.get_device_capability() == (9, 0)
    or torch.cuda.get_device_capability() == (12, 0)
):
    USE_TRITON_SRELU = True

# The fused kernel is a raw autograd.Function -> dynamo graph break mid-layer
# (~1 ms GPU hole per bf16 layer at 35 W CPU). Setting
# NANOPLM_DISABLE_TRITON_SRELU=1 routes ModernBertSReluMLP through plain ops so
# inductor compiles the whole layer (max_autotune may fuse the epilogue itself).
if _os_srelu.environ.get("NANOPLM_DISABLE_TRITON_SRELU", "0") == "1":
    USE_TRITON_SRELU = False

# -- Activation checkpointing toggles ------------------------------------------

# Checkpoint the Canon sub-layer (varlen depthwise conv).
USE_ACTIVATION_CHECKPOINTING_CANON = True

# mHC-lite selective recompute (paper-inspired):
# checkpoint only mHC pre/post kernels, keep heavy layer function outside.
USE_ACTIVATION_CHECKPOINTING_MHC = False
