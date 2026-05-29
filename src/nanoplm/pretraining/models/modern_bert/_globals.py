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
else:
    _mhc_triton_ops = None  # type: ignore[assignment]
    _canon_ops = None  # type: ignore[assignment]
    _varlen_canon_conv = None  # type: ignore[assignment]
    _varlen_ln_canon_conv = None  # type: ignore[assignment]
    _varlen_rms_canon_conv = None  # type: ignore[assignment]

# -- Flash Attention detection -------------------------------------------------

_HAS_FLASH_VARLEN = False
_flash_varlen_fn = None
_FLASH_HAS_DROPOUT = False
_OFFLINE_MODE = False

if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
    try:
        
        # FA3 (H100 / sm90)
        if _OFFLINE_MODE:
            import flash_attn_interface as _fa3
        else: 
            from kernels import get_kernel
            _fa3 = get_kernel("varunneal/flash-attention-3")
            _fa3 = getattr(_fa3, "flash_attn_interface", _fa3)
        
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

USE_TRITON_SRELU = False

if torch.cuda.is_available() and (
    torch.cuda.get_device_capability() == (9, 0)
    or torch.cuda.get_device_capability() == (12, 0)
):
    USE_TRITON_SRELU = True

# -- Activation checkpointing toggles ------------------------------------------

# Checkpoint the Canon sub-layer (varlen depthwise conv).
USE_ACTIVATION_CHECKPOINTING_CANON = True

# mHC-lite selective recompute (paper-inspired):
# checkpoint only mHC pre/post kernels, keep heavy layer function outside.
USE_ACTIVATION_CHECKPOINTING_MHC = False
