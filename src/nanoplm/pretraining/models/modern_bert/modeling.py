"""Pure PyTorch ModernBERT for masked language modeling.

The model is intentionally small and readable:
- pre-norm transformer blocks
- RoPE attention (full + sliding-window layers)
- GLU MLP (or SwiGLU replacement)
- explicit, centralized initialization
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from dataclasses import dataclass, field, fields
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

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

_HAS_FLASH_VARLEN = False
_flash_varlen_fn = None
_FLASH_HAS_DROPOUT = False
USE_ACTIVATION_CHECKPOINTING_CANON = True
# mHC-lite selective recompute (paper-inspired):
# checkpoint only mHC pre/post kernels, keep heavy layer function outside.
USE_ACTIVATION_CHECKPOINTING_MHC = False

if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
    try:
        # FA3 (H100 / sm90)
        from kernels import get_kernel

        _fa3 = get_kernel("varunneal/flash-attention-3")
        _fa3 = getattr(_fa3, "flash_attn_interface", _fa3)
        _flash_varlen_fn = _fa3.flash_attn_varlen_func
        _HAS_FLASH_VARLEN = True
        _FLASH_HAS_DROPOUT = False  # FA3 removed dropout_p
    except ImportError:
        pass

USE_TRITON_SRELU = False

if torch.cuda.is_available() and (
    torch.cuda.get_device_capability() == (9, 0)
    or torch.cuda.get_device_capability() == (12, 0)
):
    USE_TRITON_SRELU = True


if not _HAS_FLASH_VARLEN:
    try:
        # FA2 (Ampere+, RTX 30xx/40xx/50xx)
        from flash_attn import flash_attn_varlen_func as _flash_varlen_fn

        _HAS_FLASH_VARLEN = True
        _FLASH_HAS_DROPOUT = True  # FA2 supports dropout_p
    except ImportError:
        pass


def _parse_canon_layers_mode(mode: str) -> frozenset[str]:
    if not isinstance(mode, str):
        raise ValueError(
            f"canon_layers_mode must be a string, got {type(mode).__name__}"
        )
    normalized = mode.strip().lower()
    if normalized in {"", "none", "off"}:
        return frozenset()

    allowed = {"a", "b", "c", "d"}
    separators = {" ", "+", "-", "_", "/", "|", ","}
    selected: set[str] = set()
    for char in normalized:
        if char in separators:
            continue
        if char not in allowed:
            raise ValueError(
                f"Invalid canon_layers_mode={mode!r}. "
                "Use a subset of ABCD (e.g., 'abcd', 'ac', 'bcd')."
            )
        selected.add(char)
    return frozenset(selected)


def _resolve_canon_kernel_size(
    canon_layers_kernel_size: Optional[int],
) -> int:
    allowed = frozenset({3, 5, 7})
    if canon_layers_kernel_size is None:
        return 5
    if isinstance(canon_layers_kernel_size, bool) or not isinstance(
        canon_layers_kernel_size, int
    ):
        raise ValueError(
            "canon_layers_kernel_size must be an integer or null/None "
            f"(auto default). Got {canon_layers_kernel_size!r}."
        )
    if canon_layers_kernel_size not in allowed:
        allowed_str = ", ".join(str(v) for v in sorted(allowed))
        raise ValueError(
            "Invalid canon_layers_kernel_size="
            f"{canon_layers_kernel_size}. Allowed values: {allowed_str}."
        )
    return canon_layers_kernel_size


_NOBLE_TARGET_ROLES: dict[str, frozenset[str]] = {
    "all":    frozenset({"attn_qkv", "attn_out", "ffn_wi", "ffn_wo"}),
    "attn":   frozenset({"attn_qkv", "attn_out"}),
    "ffn":    frozenset({"ffn_wi", "ffn_wo"}),
    "qkv":    frozenset({"attn_qkv"}),
    "out":    frozenset({"attn_out", "ffn_wo"}),
}


@dataclass
class ModernBertConfig:
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    num_kv_heads: Optional[int] = (
        None  # GQA: None means MHA (num_kv_heads = num_attention_heads)
    )
    mlp_activation: str = "swiglu"
    hidden_activation: str = "gelu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0
    norm_eps: float = 1e-5
    norm_bias: bool = False
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: Optional[int] = None
    unk_token_id: int = 2
    mask_token_id: int = 3
    attention_bias: bool = False
    attention_dropout: float = 0.0
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    embedding_dropout: float = 0.0
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    no_mlp_on_first_layer: bool = True
    decoder_bias: bool = True
    classifier_bias: bool = False
    classifier_activation: str = "gelu"
    sparse_prediction: bool = False
    sparse_pred_ignore_index: int = -100
    tie_word_embeddings: bool = True
    global_rope_theta: float = 160_000.0
    local_rope_theta: float = 10_000.0
    use_resid_lambdas: bool = False
    use_x0_lambdas: bool = False
    use_qk_norm: bool = False
    use_canon_layers: bool = False
    canon_layers_mode: str = "abcd"
    canon_layers_kernel_size: Optional[int] = None
    resid_lambda_init: float = 1.0
    x0_lambda_init: float = 0.1
    use_repo: bool = False
    repo_after_n_layers: int = 3
    use_prores: bool = False
    prores_T: int = 1000
    activation_checkpointing: bool = False
    activation_checkpointing_mode: Literal["layer", "attn", "attn+mlp"] = "layer"
    use_mhc_lite: bool = False
    mhc_n_streams: int = 4
    mhc_triton_fused: bool = False
    mhc_lite_wrapping_level: Literal["layer", "sublayers"] = "layer"
    use_diff_attn_v2: bool = False
    use_paired_head_attention: bool = False
    attn_layer_pattern: Optional[str] = None
    # When False, Q/K/V projections use separate nn.Linear modules instead of a
    # single fused Wqkv.  Separate projections give Muon's Newton-Schulz
    # orthogonalization independent subspaces for each head group, which is the
    # mathematically correct formulation.  Fused (True) is faster due to wider
    # matmuls but couples the orthogonalization across Q, K, and V.
    fused_qkv: bool = True
    # When False, SwiGLU/GLU up and gate projections use separate nn.Linear
    # modules instead of a single fused Wi.  Same Muon rationale as fused_qkv:
    # split projections keep the orthogonalization independent per subspace.
    fused_up_gate: bool = True
    # NOBLE: Nonlinear low-rank branches for linear enhancement
    use_noble: bool = False
    noble_rank: int = 64
    noble_alpha: float = 0.01               # W_up init scale
    noble_omega_range: tuple[float, float] = (0.8, 1.2)  # freq init range
    noble_phi_std: float = 0.1              # phase init std
    noble_half_kaiming: bool = True          # halve main weight init scale
    noble_targets: str = "all"              # which projections get NOBLE: all|attn|ffn|qkv|out
    wo_init: Literal["kaiming", "zero"] = "zero"
    # MoE (Mixture-of-Experts)
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_use_bias_correction: bool = True
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 5e-5  # prevents sigmoid saturation in router (DeepSeek-V3 style)
    moe_routed_scaling_factor: float = 1.0
    moe_n_group: int = 1
    moe_topk_group: int = 1
    moe_bias_update_rate: float = 1e-3
    moe_leading_dense_layers: int = 0  # first N layers are always dense (DeepSeek uses 1)
    # Huginn-style recycling: split the L-layer stack into prelude/recurrent/coda
    # regions. The recurrent block (R = L - P - C layers) is weight-tied and
    # looped recycling_loop_count times per forward pass. See docs/superpowers/
    # specs/2026-05-24-huggin-style-recycling-design.md.
    recycling_num_prelude_layers: int = 0     # P; first P layers, run once
    recycling_num_coda_layers: int = 0        # C; last C layers, run once
    recycling_loop_count: int = 1             # inference loop count; target mean at training. 1 = recycling off
    recycling_adapter_bias: bool = False      # Linear(2h, h) adapter bias (Huginn: no bias)
    recycling_s0_noise_sigma: float = 0.6324555320336759  # sqrt(2/5), Huginn σ²_s = 2/5
    recycling_use_takase_init: bool = True    # Overrides wo_init for recurrent block; rescales Wo by sqrt(1/(5·h·l))

    head_dim: int = field(init=False)
    sliding_window: int = field(init=False)
    layer_types: list[str] = field(init=False)
    canon_layer_set: frozenset[str] = field(init=False)
    moe_layer_flags: list[bool] = field(init=False)
    moe_dense_intermediate_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.mhc_lite_wrapping_level = str(self.mhc_lite_wrapping_level).strip().lower()  # type: ignore[assignment]
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"{self.hidden_size} vs {self.num_attention_heads}"
            )
        # Recycling validation
        if not isinstance(self.recycling_loop_count, int) or self.recycling_loop_count < 1:
            raise ValueError(
                f"recycling_loop_count must be int >= 1, got {self.recycling_loop_count!r}"
            )
        if not isinstance(self.recycling_num_prelude_layers, int) or self.recycling_num_prelude_layers < 0:
            raise ValueError(
                f"recycling_num_prelude_layers must be int >= 0, got {self.recycling_num_prelude_layers!r}"
            )
        if not isinstance(self.recycling_num_coda_layers, int) or self.recycling_num_coda_layers < 0:
            raise ValueError(
                f"recycling_num_coda_layers must be int >= 0, got {self.recycling_num_coda_layers!r}"
            )
        if self.recycling_loop_count == 1:
            if self.recycling_num_prelude_layers != 0 or self.recycling_num_coda_layers != 0:
                raise ValueError(
                    "recycling_loop_count=1 disables recycling; "
                    "recycling_num_prelude_layers and recycling_num_coda_layers must both be 0."
                )
        else:
            R = self.num_hidden_layers - self.recycling_num_prelude_layers - self.recycling_num_coda_layers
            if R < 1:
                raise ValueError(
                    f"recycling_num_prelude_layers ({self.recycling_num_prelude_layers}) + "
                    f"recycling_num_coda_layers ({self.recycling_num_coda_layers}) must be < "
                    f"num_hidden_layers ({self.num_hidden_layers}); got recurrent block size {R}."
                )
            # Delegate flag-compat checks to the YAML-driven validator.
            from nanoplm.config.recycling_compatibility import validate_recycling_compatibility
            validate_recycling_compatibility(self)
        # GQA: resolve num_kv_heads (None = MHA, i.e. same as num_attention_heads).
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_kv_heads for GQA: "
                f"{self.num_attention_heads} % {self.num_kv_heads} != 0"
            )
        if (
            self.use_diff_attn_v2
            and (2 * self.num_attention_heads) % self.num_kv_heads != 0
        ):
            raise ValueError(
                "With DiffV2, 2*num_attention_heads must be divisible by num_kv_heads: "
                f"2*{self.num_attention_heads} % {self.num_kv_heads} != 0"
            )
        if self.use_paired_head_attention and (self.num_attention_heads % 2) != 0:
            raise ValueError(
                "Paired head attention requires an even number of attention heads. "
                f"Got num_attention_heads={self.num_attention_heads}."
            )
        if self.use_paired_head_attention and (self.num_kv_heads % 2) != 0:
            raise ValueError(
                "Paired head attention requires an even num_kv_heads (KV heads "
                "are paired alongside Q heads). "
                f"Got num_kv_heads={self.num_kv_heads}."
            )
        if self.mhc_lite_wrapping_level not in {"layer", "sublayers"}:
            raise ValueError(
                "mhc_lite_wrapping_level must be one of {'layer', 'sublayers'}, "
                f"got {self.mhc_lite_wrapping_level!r}"
            )
        if not self.use_mhc_lite and self.mhc_lite_wrapping_level != "layer":
            raise ValueError(
                "mhc_lite_wrapping_level != 'layer' requires use_mhc_lite=true "
                "(to avoid a silent no-op configuration)."
            )
        if self.use_mhc_lite and self.use_resid_lambdas:
            raise ValueError(
                "use_mhc_lite=true is not compatible with use_resid_lambdas=true. "
                "resid_lambdas scales the hidden state before each layer, which breaks "
                "mHC-lite's stability guarantees (doubly-stochastic mixing)."
            )
        attn_stride = max(1, int(self.global_attn_every_n_layers))
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.use_paired_head_attention and (self.head_dim % 2) != 0:
            raise ValueError(
                "Paired head attention requires an even head_dim. "
                f"Got head_dim={self.head_dim}."
            )
        self.sliding_window = self.local_attention // 2
        self.mlp_activation = self.mlp_activation.lower()
        if self.mlp_activation not in {"swiglu", "glu", "srelu"}:
            raise ValueError(
                f"Unsupported mlp_activation: {self.mlp_activation}. Supported: ['swiglu', 'glu', 'srelu']"
            )
        self.canon_layers_kernel_size = _resolve_canon_kernel_size(
            self.canon_layers_kernel_size,
        )
        self.canon_layer_set = _parse_canon_layers_mode(self.canon_layers_mode)
        if not self.use_canon_layers:
            self.canon_layer_set = frozenset()
        elif not self.canon_layer_set:
            raise ValueError(
                "use_canon_layers=True requires non-empty canon_layers_mode "
                "(for example: 'abcd' or 'ac')."
            )
        # Build layer_types from explicit pattern or stride.
        if self.attn_layer_pattern is not None:
            pattern = self.attn_layer_pattern.upper().strip()
            if not pattern:
                raise ValueError("attn_layer_pattern must not be empty when provided.")
            _map = {"F": "full_attention", "S": "sliding_attention"}
            for ch in pattern:
                if ch not in _map:
                    raise ValueError(
                        f"Invalid character '{ch}' in attn_layer_pattern. "
                        "Use 'F' for full attention and 'S' for sliding attention."
                    )
            # Tile pattern to cover all layers.
            self.layer_types = [
                _map[pattern[i % len(pattern)]] for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                "full_attention" if i % attn_stride == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.use_diff_attn_v2 and self.use_repo:
            raise ValueError(
                "use_diff_attn_v2 is not compatible with use_repo. "
                "Differential attention V2 changes Q/K head counts which is "
                "incompatible with RePO's per-head position prediction."
            )
        if self.use_paired_head_attention and self.use_diff_attn_v2:
            raise ValueError(
                "use_paired_head_attention is not compatible with use_diff_attn_v2."
            )
        if self.use_repo and self.num_kv_heads != self.num_attention_heads:
            raise ValueError(
                "GQA (num_kv_heads != num_attention_heads) is not compatible with use_repo. "
                "RePO predicts per-head positions for Q and K jointly, which requires "
                "equal Q/K head counts."
            )
        if self.use_paired_head_attention and self.use_repo:
            raise ValueError(
                "use_paired_head_attention is not compatible with use_repo."
            )
        if self.use_paired_head_attention and "b" in self.canon_layer_set:
            raise ValueError(
                "use_paired_head_attention is not compatible with Canon-B. "
                "Remove 'b' from canon_layers_mode."
            )
        if self.use_noble and self.mlp_activation == "srelu":
            import warnings
            warnings.warn(
                "use_noble=True with mlp_activation='srelu': NOBLE is applied to attention "
                "layers (Wqkv, Wo) but NOT to SReluMLP layers (Wi uses a Triton kernel that "
                "extracts raw weights, Wo is a raw nn.Parameter). Consider using 'swiglu' "
                "for full NOBLE coverage.",
                stacklevel=2,
            )
        if self.use_noble and self.noble_rank >= self.hidden_size:
            import warnings
            warnings.warn(
                f"noble_rank ({self.noble_rank}) >= hidden_size ({self.hidden_size}). "
                "Low-rank factorization provides no parameter savings at this rank. "
                "Typical values are 64–256.",
                stacklevel=2,
            )
        if self.use_noble and self.noble_targets not in _NOBLE_TARGET_ROLES:
            raise ValueError(
                f"noble_targets={self.noble_targets!r} is not supported. "
                f"Choose from: {', '.join(sorted(_NOBLE_TARGET_ROLES))}."
            )
        self.wo_init = str(self.wo_init).strip().lower()  # type: ignore[assignment]
        if self.wo_init not in {"kaiming", "zero"}:
            raise ValueError(
                f"wo_init must be 'kaiming' or 'zero', got {self.wo_init!r}"
            )
        # Normalize norm_type and validate
        self.norm_type = str(self.norm_type).strip().lower()
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError(
                f"norm_type must be 'layernorm' or 'rmsnorm', got {self.norm_type!r}"
            )
        if self.norm_type == "rmsnorm" and self.norm_bias:
            raise ValueError(
                "norm_bias=True is not compatible with norm_type='rmsnorm'. "
                "RMSNorm does not support bias. Set norm_bias=False."
            )
        # MoE layer flags.
        if self.use_moe:
            if self.mlp_activation != "swiglu":
                raise ValueError(
                    "use_moe=True currently requires mlp_activation='swiglu'."
                )
            if self.moe_num_experts < 1:
                raise ValueError(
                    f"moe_num_experts must be >= 1, got {self.moe_num_experts}"
                )
            if self.moe_top_k < 1 or self.moe_top_k > self.moe_num_experts:
                raise ValueError(
                    "moe_top_k must satisfy 1 <= moe_top_k <= moe_num_experts: "
                    f"{self.moe_top_k} vs {self.moe_num_experts}"
                )
            if self.moe_n_group < 1:
                raise ValueError(f"moe_n_group must be >= 1, got {self.moe_n_group}")
            if self.moe_topk_group < 1:
                raise ValueError(f"moe_topk_group must be >= 1, got {self.moe_topk_group}")
            if self.moe_n_group > 1:
                if self.moe_num_experts % self.moe_n_group != 0:
                    raise ValueError(
                        "moe_num_experts must be divisible by moe_n_group: "
                        f"{self.moe_num_experts} % {self.moe_n_group} != 0"
                    )
                if self.moe_topk_group > self.moe_n_group:
                    raise ValueError(
                        "moe_topk_group must be <= moe_n_group: "
                        f"{self.moe_topk_group} > {self.moe_n_group}"
                    )
            else:
                self.moe_topk_group = 1
            self.moe_layer_flags = [True] * self.num_hidden_layers
            # Force first N layers to dense.
            for i in range(min(self.moe_leading_dense_layers, self.num_hidden_layers)):
                self.moe_layer_flags[i] = False
            if (
                self.activation_checkpointing
                and self.activation_checkpointing_mode == "attn+mlp"
                and any(self.moe_layer_flags)
            ):
                raise ValueError(
                    "MoE is not compatible with activation_checkpointing_mode='attn+mlp'. "
                    "Use 'attn' or 'layer' mode instead."
                )
            # Dense layers in MoE mode use the full active intermediate size
            # so they match the per-token compute of MoE layers.
            self.moe_dense_intermediate_size = (
                (self.moe_top_k + 1) * self.intermediate_size
            )
        else:
            self.moe_layer_flags = [False] * self.num_hidden_layers
            self.moe_dense_intermediate_size = self.intermediate_size

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (required by Transformers/W&B callbacks)."""
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, frozenset):
                val = sorted(val)
            result[f.name] = val
        return result


def _create_norm(hidden_size: int, eps: float, bias: bool, norm_type: str) -> nn.Module:
    """Create a normalization layer based on norm_type."""
    if norm_type == "rmsnorm":
        return nn.RMSNorm(hidden_size, eps=eps)
    return nn.LayerNorm(hidden_size, eps=eps, bias=bias)



def _get_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "srelu":
        return lambda x: F.relu(x).square()
    if name == "silu":
        return F.silu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: (B, H, S, D) or (T, H, D), cos/sin: (1, S, D) or (T, D)
    # Head dim broadcasts — works for different Q/KV head counts.
    q_dtype = q.dtype
    k_dtype = k.dtype
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


def _apply_paired_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply staggered RoPE to paired heads.

    q/k last dim is 2*D, where the first D channels belong to the first head in
    the pair and the second D channels belong to the second head in the pair.
    Each half is rotated independently with positions (2p) and (2p+1).
    """

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    cos1, cos2 = cos.chunk(2, dim=-1)
    sin1, sin2 = sin.chunk(2, dim=-1)

    q1_dtype, q2_dtype = q1.dtype, q2.dtype
    k1_dtype, k2_dtype = k1.dtype, k2.dtype
    q1f, q2f = q1.float(), q2.float()
    k1f, k2f = k1.float(), k2.float()

    q1 = (q1f * cos1 + _rotate_half(q1f) * sin1).to(dtype=q1_dtype)
    q2 = (q2f * cos2 + _rotate_half(q2f) * sin2).to(dtype=q2_dtype)
    k1 = (k1f * cos1 + _rotate_half(k1f) * sin1).to(dtype=k1_dtype)
    k2 = (k2f * cos2 + _rotate_half(k2f) * sin2).to(dtype=k2_dtype)
    return torch.cat((q1, q2), dim=-1), torch.cat((k1, k2), dim=-1)


def _pair_sdpa_attention_mask(
    attn_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attn_mask is None:
        return None
    if attn_mask.shape[-2] == 1:
        return attn_mask.repeat_interleave(2, dim=-1)
    return attn_mask.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)


def _pair_flash_window_size(window_size: tuple[int, int]) -> tuple[int, int]:
    return tuple(-1 if w < 0 else 2 * w for w in window_size)


def _diff_attn_v2(
    attn1: torch.Tensor,
    attn2: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """Differential Attention V2 subtraction.

    attn1, attn2: attention outputs from paired heads (same shape).
    lam: per-token, per-head scalar (broadcastable to attn shape).
    Returns attn1 - sigmoid(lam) * attn2.
    """
    return attn1 - torch.sigmoid(lam).unsqueeze(-1) * attn2


class RePOModule(nn.Module):
    """RePO (Re-Positioning): predicts continuous per-head positions from hidden states.

    Replaces fixed integer RoPE positions with learned content-dependent positions.
    Architecture: SwiGLU position representation + linear per-head position assignment.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, d_p: Optional[int] = None
    ):
        super().__init__()
        d_p = d_p or hidden_size // 8
        self.W_g = nn.Linear(hidden_size, d_p, bias=False)
        self.W_c = nn.Linear(hidden_size, d_p, bias=False)
        self.W_z = nn.Linear(d_p, num_heads, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., hidden_size) -> (..., num_heads) scalar positions per head."""
        r = F.silu(self.W_g(h)) * self.W_c(h)
        return self.W_z(r)


def _apply_rope_repo(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using learned per-head positions (varlen path).

    q, k:      (T, H, D)
    positions: (T, H)     – learned scalar position per head
    inv_freq:  (D//2,)
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    # (T, H, 1) * (1, 1, D//2) -> (T, H, D//2)
    freqs = positions.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, H, D)
    cos = emb.cos()
    sin = emb.sin()
    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


def _apply_rope_repo_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using learned per-head positions (SDPA/padded path).

    q, k:      (B, H, S, D)
    positions: (B, S, H)    – learned scalar position per head
    inv_freq:  (D//2,)
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    # (B, S, H) -> (B, H, S, 1)
    pos = positions.permute(0, 2, 1).unsqueeze(-1).float()
    freqs = pos * inv_freq.view(1, 1, 1, -1)  # (B, H, S, D//2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (B, H, S, D)
    cos = emb.cos()
    sin = emb.sin()
    qf, kf = q.float(), k.float()
    q = qf * cos + _rotate_half(qf) * sin
    k = kf * cos + _rotate_half(kf) * sin
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


def _full_attention_mask(
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.bool().all():
        return None
    return attention_mask[:, None, None, :].bool()


def _sliding_attention_mask(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    sliding_window: int,
    device: torch.device,
) -> torch.Tensor:
    q = torch.arange(seq_len, device=device)[:, None]
    kv = torch.arange(seq_len, device=device)[None, :]
    mask = (q - kv).abs() <= sliding_window
    mask = mask[None, None, :, :]

    if attention_mask is not None and not attention_mask.bool().all():
        mask = mask & attention_mask[:, None, None, :].bool()

    return mask


# ---------------------------------------------------------------------------
# Unpadding helpers for flash_attn_varlen_func
# ---------------------------------------------------------------------------


def _unpad_input(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive unpadding metadata from a (batch, seq_len) attention mask.

    Returns:
        indices:    (total_tokens,) – flat indices of non-padding positions.
        cu_seqlens: (batch + 1,)    – cumulative sequence lengths (int32).
        max_seqlen: 0-D int32 tensor – longest sequence in the batch.
    """
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens.max()  # keep as tensor to avoid torch.compile graph break
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def _pad_output(
    hidden: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Scatter a flat (total_tokens, …) tensor back to (batch, seqlen, …)."""
    out = torch.zeros(
        (batch * seqlen, *hidden.shape[1:]),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    out[indices] = hidden
    return out.view(batch, seqlen, *hidden.shape[1:])


def _position_ids_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    total: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert cu_seqlens to per-token position IDs (reset to 0 per sequence).

    Example: cu_seqlens=[0,3,5,9] → [0,1,2, 0,1, 0,1,2,3]
    """
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    offsets = cu_seqlens[:-1].repeat_interleave(seq_lens)
    return torch.arange(total, device=device, dtype=torch.int32) - offsets


# ---------------------------------------------------------------------------
# NOBLE: Nonlinear lOw-rank Branch for Linear Enhancement
# Paper: "NOBLE: Accelerating Transformers with Nonlinear Low-Rank Branches"
# (Smith, 2026, arXiv:2603.06492)
# ---------------------------------------------------------------------------


class CosNet(nn.Module):
    """Two-layer cosine nonlinearity with learnable frequency, phase, and mixing.

    CosNet(h) = cos(ω₂ ⊙ (M · cos(ω₁ ⊙ h + φ₁)) + φ₂)

    where h ∈ ℝʳ is the bottleneck representation, M ∈ ℝʳˣʳ is a learned
    mixing matrix, and each dimension i has learnable ωᵢ and φᵢ parameters.
    """

    def __init__(
        self,
        rank: int,
        omega_range: tuple[float, float] = (0.8, 1.2),
        phi_std: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        # First cosine layer: learnable frequency and phase
        self.omega1 = nn.Parameter(torch.empty(rank))
        self.phi1 = nn.Parameter(torch.empty(rank))
        # Mixing matrix (r × r)
        self.M = nn.Parameter(torch.empty(rank, rank))
        # Second cosine layer: learnable frequency and phase
        self.omega2 = nn.Parameter(torch.empty(rank))
        self.phi2 = nn.Parameter(torch.empty(rank))

        # --- Initialization (paper Appendix A) ---
        # Frequencies: Uniform[ω_min, ω_max]
        nn.init.uniform_(self.omega1, omega_range[0], omega_range[1])
        nn.init.uniform_(self.omega2, omega_range[0], omega_range[1])
        # Phases: Normal(0, σ_φ)
        nn.init.normal_(self.phi1, mean=0.0, std=phi_std)
        nn.init.normal_(self.phi2, mean=0.0, std=phi_std)
        # Mixing matrix: Xavier uniform
        nn.init.xavier_uniform_(self.M)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., rank)
        h = torch.cos(self.omega1 * h + self.phi1)   # eq (5)
        h = F.linear(h, self.M)                       # eq (6): M @ h (F.linear computes h @ M^T)
        h = torch.cos(self.omega2 * h + self.phi2)   # eq (7)
        return h


class NOBLELinear(nn.Module):
    """Drop-in replacement for nn.Linear with a nonlinear low-rank branch.

    f_NOBLE(x) = xW + b + CosNet(x @ W_down^T) @ W_up^T

    The branch starts near-silent (W_up ≈ 0) and gradually learns
    complementary features that the main linear pathway cannot represent.

    Exposes .weight and .bias properties so existing init code that does
    ``nn.init.uniform_(module.weight, ...)`` works unchanged.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
        alpha: float = 0.01,
        omega_range: tuple[float, float] = (0.8, 1.2),
        phi_std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Main linear pathway
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Low-rank nonlinear branch
        self.W_down = nn.Linear(in_features, rank, bias=False)
        self.cosnet = CosNet(rank, omega_range=omega_range, phi_std=phi_std)
        self.noble_w_up = nn.Linear(rank, out_features, bias=False)

        # noble_w_up: near-zero init
        nn.init.normal_(self.noble_w_up.weight, mean=0.0, std=alpha / math.sqrt(rank))

    @property
    def weight(self) -> nn.Parameter:
        """Delegate to main linear weight for compatibility with init_weights()."""
        return self.linear.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        """Delegate to main linear bias for compatibility with init_weights()."""
        return self.linear.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.linear.bias is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.linear(x)
        branch = self.noble_w_up(self.cosnet(self.W_down(x)))
        return main + branch


def _build_linear(
    in_features: int,
    out_features: int,
    config: ModernBertConfig,
    bias: bool = False,
    role: str = "attn_qkv",
) -> nn.Module:
    """Factory: returns NOBLELinear if NOBLE is enabled for this role, else plain nn.Linear."""
    if config.use_noble:
        targets = _NOBLE_TARGET_ROLES.get(config.noble_targets, _NOBLE_TARGET_ROLES["all"])
        if role in targets:
            return NOBLELinear(
                in_features,
                out_features,
                rank=config.noble_rank,
                bias=bias,
                alpha=config.noble_alpha,
                omega_range=config.noble_omega_range,
                phi_std=config.noble_phi_std,
            )
    return nn.Linear(in_features, out_features, bias=bias)


def _make_canon_layer(channels: int, config: ModernBertConfig) -> nn.Module:
    """Factory: returns a symmetric (bidirectional) Canon layer."""
    if config.canon_layers_kernel_size is None:
        raise ValueError(
            "canon_layers_kernel_size was not resolved in ModernBertConfig.__post_init__"
        )
    return ModernBertCanonLayer(channels, kernel_size=config.canon_layers_kernel_size)


def _canon_accum_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return x.dtype


class ModernBertCanonLayer(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self.radius = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.radius,
            groups=channels,
            bias=True,
        )

    def _forward_varlen(self, x, cu_seqlens, position_ids=None):
        T, C = x.shape
        n_seqs = cu_seqlens.shape[0] - 1

        if n_seqs <= 1:
            acc_dtype = _canon_accum_dtype(x)
            x_acc = x.to(dtype=acc_dtype)
            mixed = (
                F.conv1d(
                    x_acc.T.unsqueeze(0),
                    self.conv.weight.to(dtype=acc_dtype),
                    bias=(
                        self.conv.bias.to(dtype=acc_dtype)
                        if self.conv.bias is not None
                        else None
                    ),
                    stride=1,
                    padding=self.radius,
                    groups=self.conv.groups,
                )
                .squeeze(0)
                .T.to(dtype=x.dtype)
            )
            return x + mixed

        if position_ids is not None and position_ids.shape[0] == T:
            seq_start = (position_ids == 0).to(dtype=cu_seqlens.dtype)
            seq_start = seq_start.clone()
            seq_start[0] = 1
            seq_id = torch.cumsum(seq_start, dim=0) - 1
        else:
            positions = torch.arange(T, device=x.device, dtype=cu_seqlens.dtype)
            seq_id = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        weight = self.conv.weight[:, 0, :]
        bias = self.conv.bias
        mixed = _varlen_canon_conv(x, seq_id, weight, bias, self.radius)
        return x + mixed

    def _forward_padded(self, x, attention_mask=None):
        acc_dtype = _canon_accum_dtype(x)
        x_acc = x.to(dtype=acc_dtype)
        token_mask = None
        if attention_mask is not None:
            # Treat any nonzero as "valid token". This avoids accidental
            # NaNs if a caller passes an additive mask (e.g. 0 / -inf).
            token_mask = attention_mask.ne(0).unsqueeze(-1).to(dtype=acc_dtype)
            x_acc = x_acc * token_mask
        mixed = F.conv1d(
            x_acc.transpose(1, 2),
            self.conv.weight.to(dtype=acc_dtype),
            bias=(
                self.conv.bias.to(dtype=acc_dtype)
                if self.conv.bias is not None
                else None
            ),
            stride=1,
            padding=self.radius,
            groups=self.conv.groups,
        ).transpose(1, 2)
        out = x_acc + mixed
        if token_mask is not None:
            out = out * token_mask
        return out.to(dtype=x.dtype)

    def _forward_varlen_fused_ln(
        self, x, cu_seqlens, ln_weight, ln_eps, position_ids=None
    ):
        """Fused LN + Canon varlen path: computes LN(x) + conv(LN(x)) without
        materializing the intermediate LN output."""
        T, C = x.shape
        n_seqs = cu_seqlens.shape[0] - 1

        # Single-sequence: fall back to separate LN + F.conv1d (no Triton path)
        if n_seqs <= 1:
            acc_dtype = _canon_accum_dtype(x)
            # Compute LN manually to get normalized x
            x_fp32 = x.float()
            mean = x_fp32.mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(x_fp32.var(dim=-1, keepdim=True, correction=0) + ln_eps)
            ln_out = ((x_fp32 - mean) * rstd * ln_weight.float()).to(dtype=acc_dtype)
            mixed = (
                F.conv1d(
                    ln_out.T.unsqueeze(0),
                    self.conv.weight.to(dtype=acc_dtype),
                    bias=(
                        self.conv.bias.to(dtype=acc_dtype)
                        if self.conv.bias is not None
                        else None
                    ),
                    stride=1,
                    padding=self.radius,
                    groups=self.conv.groups,
                )
                .squeeze(0)
                .T.to(dtype=x.dtype)
            )
            return (ln_out + mixed).to(dtype=x.dtype)

        if position_ids is not None and position_ids.shape[0] == T:
            seq_start = (position_ids == 0).to(dtype=cu_seqlens.dtype)
            seq_start = seq_start.clone()
            seq_start[0] = 1
            seq_id = torch.cumsum(seq_start, dim=0) - 1
        else:
            positions = torch.arange(T, device=x.device, dtype=cu_seqlens.dtype)
            seq_id = torch.searchsorted(cu_seqlens[1:], positions, right=True)

        weight = self.conv.weight[:, 0, :]
        bias = self.conv.bias
        return _varlen_ln_canon_conv(
            x, seq_id, ln_weight, ln_eps, weight, bias, self.radius
        )

    def _forward_varlen_fused_rms(
        self, x, cu_seqlens, rms_weight, rms_eps, position_ids=None
    ):
        """Fused RMS + Canon varlen path: computes RMS(x) + conv(RMS(x)) without
        materializing the intermediate RMS output."""
        T, C = x.shape
        n_seqs = cu_seqlens.shape[0] - 1

        # Single-sequence: fall back to separate RMS + F.conv1d (no Triton path)
        if n_seqs <= 1:
            acc_dtype = _canon_accum_dtype(x)
            # Compute RMS manually to get normalized x
            x_fp32 = x.float()
            inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
            rms_out = (x_fp32 * inv_rms * rms_weight.float()).to(dtype=acc_dtype)
            mixed = (
                F.conv1d(
                    rms_out.T.unsqueeze(0),
                    self.conv.weight.to(dtype=acc_dtype),
                    bias=(
                        self.conv.bias.to(dtype=acc_dtype)
                        if self.conv.bias is not None
                        else None
                    ),
                    stride=1,
                    padding=self.radius,
                    groups=self.conv.groups,
                )
                .squeeze(0)
                .T.to(dtype=x.dtype)
            )
            return (rms_out + mixed).to(dtype=x.dtype)

        if position_ids is not None and position_ids.shape[0] == T:
            seq_start = (position_ids == 0).to(dtype=cu_seqlens.dtype)
            seq_start = seq_start.clone()
            seq_start[0] = 1
            seq_id = torch.cumsum(seq_start, dim=0) - 1
        else:
            positions = torch.arange(T, device=x.device, dtype=cu_seqlens.dtype)
            seq_id = torch.searchsorted(cu_seqlens[1:], positions, right=True)

        weight = self.conv.weight[:, 0, :]
        bias = self.conv.bias
        return _varlen_rms_canon_conv(
            x, seq_id, rms_weight, rms_eps, weight, bias, self.radius
        )

    def forward(
        self,
        x,
        position_ids=None,
        cu_seqlens=None,
        attention_mask=None,
        fuse_ln_weight=None,
        fuse_ln_eps=None,
        fuse_norm_type="layernorm",
    ):
        # Checkpoint at the module boundary so all Canon insertion sites (A/B/C/D)
        # and both varlen/padded paths are covered by the same flag.
        use_ckpt = (
            USE_ACTIVATION_CHECKPOINTING_CANON
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
        )
        # Eval/inference should avoid autotune warmup latency.
        # _canon_ops is None on CPU/MPS (no CUDA) — skip the Triton autotune guard.
        autotune_ctx = (
            nullcontext()
            if self.training or _canon_ops is None
            else _canon_ops.disable_autotune_temporarily()
        )

        with autotune_ctx:
            return self._forward_dispatch(
                x,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                fuse_ln_weight=fuse_ln_weight,
                fuse_ln_eps=fuse_ln_eps,
                fuse_norm_type=fuse_norm_type,
                use_ckpt=use_ckpt,
            )

    def _forward_dispatch(
        self,
        x,
        *,
        position_ids,
        cu_seqlens,
        attention_mask,
        fuse_ln_weight,
        fuse_ln_eps,
        fuse_norm_type,
        use_ckpt,
    ):
        # Fused Norm+Conv path (varlen multi-sequence only)
        if fuse_ln_weight is not None and cu_seqlens is not None:
            if fuse_norm_type == "rmsnorm":
                # RMSNorm fusion path
                if use_ckpt:
                    if position_ids is None:
                        return _checkpoint(
                            lambda x_, cu_: self._forward_varlen_fused_rms(
                                x_,
                                cu_seqlens=cu_,
                                rms_weight=fuse_ln_weight,
                                rms_eps=fuse_ln_eps,
                                position_ids=None,
                            ),
                            x,
                            cu_seqlens,
                            use_reentrant=False,
                        )
                    return _checkpoint(
                        lambda x_, cu_, pos_: self._forward_varlen_fused_rms(
                            x_,
                            cu_seqlens=cu_,
                            rms_weight=fuse_ln_weight,
                            rms_eps=fuse_ln_eps,
                            position_ids=pos_,
                        ),
                        x,
                        cu_seqlens,
                        position_ids,
                        use_reentrant=False,
                    )
                return self._forward_varlen_fused_rms(
                    x,
                    cu_seqlens=cu_seqlens,
                    rms_weight=fuse_ln_weight,
                    rms_eps=fuse_ln_eps,
                    position_ids=position_ids,
                )
            else:
                # LayerNorm fusion path
                if use_ckpt:
                    if position_ids is None:
                        return _checkpoint(
                            lambda x_, cu_: self._forward_varlen_fused_ln(
                                x_,
                                cu_seqlens=cu_,
                                ln_weight=fuse_ln_weight,
                                ln_eps=fuse_ln_eps,
                                position_ids=None,
                            ),
                            x,
                            cu_seqlens,
                            use_reentrant=False,
                        )
                    return _checkpoint(
                        lambda x_, cu_, pos_: self._forward_varlen_fused_ln(
                            x_,
                            cu_seqlens=cu_,
                            ln_weight=fuse_ln_weight,
                            ln_eps=fuse_ln_eps,
                            position_ids=pos_,
                        ),
                        x,
                        cu_seqlens,
                        position_ids,
                        use_reentrant=False,
                    )
                return self._forward_varlen_fused_ln(
                    x,
                    cu_seqlens=cu_seqlens,
                    ln_weight=fuse_ln_weight,
                    ln_eps=fuse_ln_eps,
                    position_ids=position_ids,
                )

        if cu_seqlens is not None:
            if use_ckpt:
                if position_ids is None:
                    return _checkpoint(
                        lambda x_, cu_: self._forward_varlen(
                            x_, cu_seqlens=cu_, position_ids=None
                        ),
                        x,
                        cu_seqlens,
                        use_reentrant=False,
                    )
                return _checkpoint(
                    lambda x_, cu_, pos_: self._forward_varlen(
                        x_, cu_seqlens=cu_, position_ids=pos_
                    ),
                    x,
                    cu_seqlens,
                    position_ids,
                    use_reentrant=False,
                )
            return self._forward_varlen(
                x, cu_seqlens=cu_seqlens, position_ids=position_ids
            )
        if x.dim() == 3:
            if use_ckpt:
                if attention_mask is None:
                    return _checkpoint(
                        lambda x_: self._forward_padded(x_, attention_mask=None),
                        x,
                        use_reentrant=False,
                    )
                return _checkpoint(
                    lambda x_, mask_: self._forward_padded(x_, attention_mask=mask_),
                    x,
                    attention_mask,
                    use_reentrant=False,
                )
            return self._forward_padded(x, attention_mask=attention_mask)
        raise ValueError(f"Expected padded input [B, S, C], got shape={tuple(x.shape)}")


def _varlen_canon_inner(x, seq_id, weight, bias, radius):
    """Depthwise conv with boundary masking for varlen Canon mixing."""
    acc_dtype = _canon_accum_dtype(x)
    x_acc = x.to(dtype=acc_dtype)
    weight_acc = weight.to(dtype=acc_dtype)
    bias_acc = bias.to(dtype=acc_dtype) if bias is not None else None
    out = torch.zeros_like(x_acc)
    for k, offset in enumerate(range(-radius, radius + 1)):
        rolled_x = torch.roll(x_acc, shifts=-offset, dims=0)
        rolled_id = torch.roll(seq_id, shifts=-offset, dims=0)
        # IMPORTANT: avoid `0 * NaN = NaN` propagation across sequence boundaries.
        valid = (rolled_id == seq_id).unsqueeze(-1)
        rolled_x = rolled_x.masked_fill(~valid, 0)
        out = out + rolled_x * weight_acc[:, k]
    if bias_acc is not None:
        out = out + bias_acc
    return out.to(dtype=x.dtype)


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.norm = _create_norm(
            config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
        )
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            "inv_freq_full",
            self._build_inv_freq(config.global_rope_theta),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_sliding",
            self._build_inv_freq(config.local_rope_theta),
            persistent=False,
        )

    def _build_inv_freq(self, theta: float) -> torch.Tensor:
        channel = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32)
        return 1.0 / (theta ** (channel / self.config.head_dim))

    def _select_inv_freq(self, layer_type: str, device: torch.device) -> torch.Tensor:
        inv_freq = (
            self.inv_freq_full
            if layer_type == "full_attention"
            else self.inv_freq_sliding
        )
        return inv_freq.to(device=device)

    def paired_from_positions(
        self,
        positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._select_inv_freq(layer_type, device)
        pos = positions.to(device=device, dtype=torch.float32)
        even_freqs = torch.outer(2.0 * pos, inv_freq)
        odd_freqs = torch.outer(2.0 * pos + 1.0, inv_freq)
        even = torch.cat((even_freqs, even_freqs), dim=-1)
        odd = torch.cat((odd_freqs, odd_freqs), dim=-1)
        emb = torch.cat((even, odd), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def paired(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        cos, sin = self.paired_from_positions(positions, device, dtype, layer_type)
        return cos[None], sin[None]

    def forward(
        self,
        seq_len: int | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_type == "full_attention":
            inv_freq = self.inv_freq_full
        else:
            inv_freq = self.inv_freq_sliding

        inv_freq = inv_freq.to(device=device)
        if isinstance(seq_len, torch.Tensor):
            # Dynamic tensor lengths cannot be passed to torch.arange without
            # materializing a Python int. Slice from the model cap instead so
            # compiled varlen paths stay graph-friendly.
            positions = torch.arange(
                self.config.max_position_embeddings,
                device=device,
                dtype=torch.float32,
            )[:seq_len]
        else:
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None].to(dtype=dtype), emb.sin()[None].to(dtype=dtype)


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.fused_up_gate = config.fused_up_gate
        if self.fused_up_gate:
            self.Wi = _build_linear(
                config.hidden_size,
                2 * config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
            self.W_up = None
            self.W_gate = None
        else:
            self.Wi = None
            self.W_up = _build_linear(
                config.hidden_size,
                config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
            self.W_gate = _build_linear(
                config.hidden_size,
                config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
        self.Wo = _build_linear(
            config.intermediate_size,
            config.hidden_size,
            config,
            bias=config.mlp_bias,
            role="ffn_wo",
        )
        self.drop = nn.Dropout(config.mlp_dropout)
        self.act = _get_activation(config.hidden_activation)
        self.canon_d = (
            _make_canon_layer(2 * config.intermediate_size, config)
            if "d" in config.canon_layer_set
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.fused_up_gate:
            wi = self.Wi(x)
        else:
            wi = torch.cat([self.W_up(x), self.W_gate(x)], dim=-1)
        if self.canon_d is not None:
            wi = self.canon_d(
                wi,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )
        x_proj, gate = wi.chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(x_proj) * gate))


class ModernBertSReluMLP(nn.Module):
    """MLP using relu(x)^2 activation (no gating).

    When USE_TRITON_SRELU is True, uses a fused Triton kernel for
    relu(x @ Wi.T)^2 in a single pass.  Wo is stored as (intermediate, hidden)
    to match the kernel layout (post @ Wo).

    When USE_TRITON_SRELU is False, uses plain PyTorch ops for benchmarking.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.Wo_weight = nn.Parameter(
            torch.empty(config.intermediate_size, config.hidden_size)
        )
        self.Wo_bias = (
            nn.Parameter(torch.zeros(config.hidden_size)) if config.mlp_bias else None
        )
        self.drop = nn.Dropout(config.mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if USE_TRITON_SRELU:
            from nanoplm.pretraining.models.modern_bert.triton_kernels import (
                FusedLinearReLUSquare,
            )

            out = FusedLinearReLUSquare.apply(x, self.Wi.weight, self.Wo_weight)
        else:
            h = F.relu(self.Wi(x))
            h = h * h
            out = h @ self.Wo_weight
        if self.Wo_bias is not None:
            out = out + self.Wo_bias
        return self.drop(out)


class ModernBertSwiGLUMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.fused_up_gate = config.fused_up_gate
        if self.fused_up_gate:
            self.Wi = _build_linear(
                config.hidden_size,
                2 * config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
            self.W_up = None
            self.W_gate = None
        else:
            self.Wi = None
            self.W_up = _build_linear(
                config.hidden_size,
                config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
            self.W_gate = _build_linear(
                config.hidden_size,
                config.intermediate_size,
                config,
                bias=config.mlp_bias,
                role="ffn_wi",
            )
        self.Wo = _build_linear(
            config.intermediate_size,
            config.hidden_size,
            config,
            bias=config.mlp_bias,
            role="ffn_wo",
        )
        self.drop = nn.Dropout(config.mlp_dropout)
        self.canon_d = (
            _make_canon_layer(2 * config.intermediate_size, config)
            if "d" in config.canon_layer_set
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.fused_up_gate:
            wi = self.Wi(x)
        else:
            wi = torch.cat([self.W_up(x), self.W_gate(x)], dim=-1)
        if self.canon_d is not None:
            wi = self.canon_d(
                wi,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )
        x_proj, gate = wi.chunk(2, dim=-1)
        return self.Wo(self.drop(F.silu(gate) * x_proj))


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int = 0):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.use_qk_norm = config.use_qk_norm
        self.dropout = config.attention_dropout
        self.scale = self.head_dim**-0.5
        self.use_diff_attn_v2 = config.use_diff_attn_v2
        self.use_paired_head_attention = config.use_paired_head_attention

        # GQA: num_kv_heads <= num_heads. MHA when num_kv_heads == num_heads.
        self.num_kv_heads = config.num_kv_heads

        if self.use_diff_attn_v2:
            # DiffV2 doubles Q heads on top of whatever GQA config is set.
            # Each original head splits into a pair for differential subtraction.
            self.num_q_heads = 2 * self.num_heads
            # Lambda: per-token, per-head scalar controlling subtraction weight.
            self.lambda_proj = nn.Linear(
                config.hidden_size,
                self.num_heads,
                bias=False,
            )
        else:
            self.num_q_heads = self.num_heads
            self.lambda_proj = None

        self.fused_qkv = config.fused_qkv
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        if self.fused_qkv:
            qkv_dim = q_dim + 2 * kv_dim
            self.Wqkv = _build_linear(
                config.hidden_size,
                qkv_dim,
                config,
                bias=config.attention_bias,
                role="attn_qkv",
            )
            self.Wq = None
            self.Wq2 = None
            self.Wk = None
            self.Wv = None
        else:
            self.Wqkv = None
            if self.use_diff_attn_v2:
                # DiffV2 doubles Q heads into two groups for differential
                # subtraction.  Split them into Wq (group 1) and Wq2 (group 2)
                # so Muon orthogonalizes each group independently.
                half_q_dim = self.num_heads * self.head_dim
                self.Wq = _build_linear(
                    config.hidden_size, half_q_dim, config,
                    bias=config.attention_bias, role="attn_qkv",
                )
                self.Wq2 = _build_linear(
                    config.hidden_size, half_q_dim, config,
                    bias=config.attention_bias, role="attn_qkv",
                )
            else:
                self.Wq = _build_linear(
                    config.hidden_size, q_dim, config,
                    bias=config.attention_bias, role="attn_qkv",
                )
                self.Wq2 = None
            self.Wk = _build_linear(
                config.hidden_size, kv_dim, config,
                bias=config.attention_bias, role="attn_qkv",
            )
            self.Wv = _build_linear(
                config.hidden_size, kv_dim, config,
                bias=config.attention_bias, role="attn_qkv",
            )

        self.Wo = _build_linear(
            config.hidden_size,
            config.hidden_size,
            config,
            bias=config.attention_bias,
            role="attn_out",
        )
        self.out_drop = (
            nn.Dropout(config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )
        qkv_out_dim = q_dim + 2 * kv_dim
        self.canon_b = (
            _make_canon_layer(qkv_out_dim, config)
            if "b" in config.canon_layer_set
            else None
        )

        # RePO: learned per-head positions replacing fixed RoPE indices
        self.repo = None
        if config.use_repo and layer_idx >= config.repo_after_n_layers:
            self.repo = RePOModule(
                config.hidden_size,
                config.num_attention_heads,
                config.head_dim,
            )
            theta = (
                config.global_rope_theta
                if config.layer_types[layer_idx] == "full_attention"
                else config.local_rope_theta
            )
            channel = torch.arange(0, config.head_dim, 2, dtype=torch.float32)
            self.register_buffer(
                "repo_inv_freq",
                1.0 / (theta ** (channel / config.head_dim)),
                persistent=False,
            )

    def _pair_varlen_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int | torch.Tensor
    ]:
        cos, sin = cos_sin
        q = q.reshape(q.shape[0], self.num_heads // 2, 2 * self.head_dim)
        k = k.reshape(k.shape[0], self.num_kv_heads // 2, 2 * self.head_dim)
        q, k = _apply_paired_rope(q, k, cos, sin)
        q = q.reshape(q.shape[0] * 2, self.num_heads // 2, self.head_dim)
        k = k.reshape(k.shape[0] * 2, self.num_kv_heads // 2, self.head_dim)
        v = v.reshape(v.shape[0] * 2, self.num_kv_heads // 2, self.head_dim)
        paired_cu = cu_seqlens * 2
        paired_max = max_seqlen * 2
        return q, k, v, paired_cu, paired_max

    def _pair_sdpa_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _, seq_len, _ = q.shape
        cos, sin = cos_sin
        q = q.transpose(1, 2).reshape(
            bsz, seq_len, self.num_heads // 2, 2 * self.head_dim
        )
        k = k.transpose(1, 2).reshape(
            bsz, seq_len, self.num_kv_heads // 2, 2 * self.head_dim
        )
        q, k = _apply_paired_rope(q, k, cos[0], sin[0])
        q = q.reshape(bsz, 2 * seq_len, self.num_heads // 2, self.head_dim).transpose(
            1, 2
        )
        k = k.reshape(
            bsz, 2 * seq_len, self.num_kv_heads // 2, self.head_dim
        ).transpose(1, 2)
        v = (
            v.transpose(1, 2)
            .reshape(bsz, 2 * seq_len, self.num_kv_heads // 2, self.head_dim)
            .transpose(1, 2)
        )
        return q, k, v

    # -- varlen (flash-attention) path -----------------------------------------

    def _forward_varlen(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        window_size: tuple[int, int],
        position_ids: Optional[torch.Tensor] = None,
        repo_active: bool = False,
    ) -> torch.Tensor:
        total = x.shape[0]  # (total_tokens, hidden)
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        if self.fused_qkv:
            qkv = self.Wqkv(x)
            if self.canon_b is not None:
                qkv = self.canon_b(
                    qkv, position_ids=position_ids, cu_seqlens=cu_seqlens
                )
            q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        else:
            if self.Wq2 is not None:
                # DiffV2 split: interleave the two Q groups so paired heads
                # (0::2, 1::2) are contiguous as the rest of the code expects.
                q1 = self.Wq(x).view(total, self.num_heads, self.head_dim)
                q2 = self.Wq2(x).view(total, self.num_heads, self.head_dim)
                q = torch.stack([q1, q2], dim=2).view(total, -1)
            else:
                q = self.Wq(x)
            k = self.Wk(x)
            v = self.Wv(x)
            if self.canon_b is not None:
                qkv = torch.cat([q, k, v], dim=-1)
                qkv = self.canon_b(
                    qkv, position_ids=position_ids, cu_seqlens=cu_seqlens
                )
                q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(total, self.num_q_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)

        if self.repo is not None and repo_active:
            positions = self.repo(x)  # (T, num_heads)
            q, k = _apply_rope_repo(q, k, positions, self.repo_inv_freq)
        elif not self.use_paired_head_attention:
            cos, sin = cos_sin
            q, k = _apply_rope(q, k, cos, sin)
        if self.use_qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        if self.use_paired_head_attention:
            q, k, v, cu_seqlens, max_seqlen = self._pair_varlen_qkv(
                q,
                k,
                v,
                cos_sin,
                cu_seqlens,
                max_seqlen,
            )
            window_size = _pair_flash_window_size(window_size)

        # When max_seqlen is already a plain int (static-shape mode) skip .item()
        # to avoid a graph break.  For tensor values (dynamic mode) .item() is
        # fine — flash_attn is an opaque C extension at a graph boundary.
        max_s = max_seqlen if isinstance(max_seqlen, int) else max_seqlen.item()
        kwargs = dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            softmax_scale=self.scale,
            window_size=window_size,
        )
        if _FLASH_HAS_DROPOUT:
            kwargs["dropout_p"] = self.dropout if self.training else 0.0

        y = _flash_varlen_fn(q, k, v, **kwargs)
        if isinstance(y, tuple):
            y = y[0]

        if self.use_paired_head_attention:
            y = y.reshape(total, self.num_heads, self.head_dim)

        if self.use_diff_attn_v2:
            # y: (total, 2*num_heads, head_dim) — paired heads in same GQA group
            # are contiguous, so 0::2 and 1::2 select the two halves.
            lam = self.lambda_proj(x)  # (total, num_heads)
            y = _diff_attn_v2(y[:, 0::2], y[:, 1::2], lam)
            # y: (total, num_heads, head_dim)

        y = y.contiguous().view(total, -1)  # (total, hidden)
        return self.out_drop(self.Wo(y))

    # -- SDPA (fallback) path --------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        repo_active: bool = False,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            return self._forward_varlen(
                x,
                cos_sin,
                cu_seqlens,
                max_seqlen,
                window_size,
                position_ids=position_ids,
                repo_active=repo_active,
            )

        bsz, seq_len, _ = x.shape
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        if self.fused_qkv:
            qkv = self.Wqkv(x)
            if self.canon_b is not None:
                qkv = self.canon_b(qkv, attention_mask=token_mask)
            q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        else:
            if self.Wq2 is not None:
                q1 = self.Wq(x).view(bsz, seq_len, self.num_heads, self.head_dim)
                q2 = self.Wq2(x).view(bsz, seq_len, self.num_heads, self.head_dim)
                q = torch.stack([q1, q2], dim=3).view(bsz, seq_len, -1)
            else:
                q = self.Wq(x)
            k = self.Wk(x)
            v = self.Wv(x)
            if self.canon_b is not None:
                qkv = torch.cat([q, k, v], dim=-1)
                qkv = self.canon_b(qkv, attention_mask=token_mask)
                q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(bsz, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.repo is not None and repo_active:
            positions = self.repo(x)  # (B, S, num_heads)
            q, k = _apply_rope_repo_sdpa(q, k, positions, self.repo_inv_freq)
        elif not self.use_paired_head_attention:
            cos, sin = cos_sin
            q, k = _apply_rope(q, k, cos, sin)
        if self.use_qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        if self.use_paired_head_attention:
            q, k, v = self._pair_sdpa_qkv(q, k, v, cos_sin)
            attn_mask = _pair_sdpa_attention_mask(attn_mask)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=(self.dropout if self.training else 0.0),
            scale=self.scale,
            enable_gqa=(self.num_q_heads != self.num_kv_heads),
        )

        if self.use_paired_head_attention:
            y = (
                y.transpose(1, 2)
                .reshape(bsz, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        if self.use_diff_attn_v2:
            # y: (B, 2H, S, D) -> differential subtraction
            lam = self.lambda_proj(x)  # (B, S, H)
            lam = lam.transpose(1, 2)  # (B, H, S)
            y = _diff_attn_v2(y[:, 0::2], y[:, 1::2], lam)
            # y: (B, H, S, D)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_drop(self.Wo(y))


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.has_mlp = (layer_idx != 0) or (not config.no_mlp_on_first_layer)
        self.norm_type = config.norm_type  # Store for fused Canon path
        self.activation_checkpointing = bool(
            getattr(config, "activation_checkpointing", False)
        )
        self.activation_checkpointing_mode = (
            str(getattr(config, "activation_checkpointing_mode", "layer"))
            .strip()
            .lower()
        )
        self.attn_norm = (
            nn.Identity()
            if layer_idx == 0
            else _create_norm(
                config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
            )
        )
        self.canon_a = (
            _make_canon_layer(config.hidden_size, config)
            if "a" in config.canon_layer_set
            else None
        )
        self.attn = ModernBertAttention(config, layer_idx=layer_idx)
        if self.has_mlp:
            self.mlp_norm = _create_norm(
                config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
            )
            self.canon_c = (
                _make_canon_layer(config.hidden_size, config)
                if "c" in config.canon_layer_set
                else None
            )
            if config.use_moe and config.moe_layer_flags[layer_idx]:
                from .moe import MoELayer
                self.mlp = MoELayer(config)
            else:
                # For dense layers in MoE mode, use the full active intermediate
                # size so per-token capacity matches the MoE layers.
                mlp_config = config
                if config.use_moe and not config.moe_layer_flags[layer_idx]:
                    import copy
                    mlp_config = copy.copy(config)
                    mlp_config.intermediate_size = config.moe_dense_intermediate_size
                if mlp_config.mlp_activation == "srelu":
                    self.mlp = ModernBertSReluMLP(mlp_config)
                elif mlp_config.mlp_activation == "swiglu":
                    self.mlp = ModernBertSwiGLUMLP(mlp_config)
                else:
                    self.mlp = ModernBertMLP(mlp_config)
        else:
            self.mlp_norm = nn.Identity()
            self.canon_c = None
            self.mlp = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        repo_active: bool = False,
        prores_alpha: "torch.Tensor | float" = 1.0,
    ) -> torch.Tensor:
        do_ckpt_attn = (
            self.activation_checkpointing
            and self.training
            and self.activation_checkpointing_mode in {"attn", "attn+mlp"}
        )
        if do_ckpt_attn:
            _attn_mask = attn_mask
            _cos_sin = cos_sin
            _cu_seqlens = cu_seqlens
            _max_seqlen = max_seqlen
            _window_size = window_size
            _position_ids = position_ids
            _token_mask = token_mask
            _repo_active = repo_active

            _fuse_ln_a = (
                self.canon_a is not None
                and isinstance(self.attn_norm, (nn.LayerNorm, nn.RMSNorm))
                and _cu_seqlens is not None
            )

            def _attn_branch(
                x_in: torch.Tensor,
                *,
                _layer: "ModernBertEncoderLayer" = self,
            ) -> torch.Tensor:
                if _fuse_ln_a:
                    attn_in = _layer.canon_a(
                        x_in,
                        position_ids=_position_ids,
                        cu_seqlens=_cu_seqlens,
                        fuse_ln_weight=_layer.attn_norm.weight,
                        fuse_ln_eps=_layer.attn_norm.eps,
                        fuse_norm_type=_layer.norm_type,
                    )
                else:
                    attn_in = _layer.attn_norm(x_in)
                    if _layer.canon_a is not None:
                        attn_in = _layer.canon_a(
                            attn_in,
                            position_ids=_position_ids,
                            cu_seqlens=_cu_seqlens,
                            attention_mask=_token_mask,
                        )
                return _layer.attn(
                    attn_in,
                    cos_sin=_cos_sin,
                    attn_mask=_attn_mask,
                    cu_seqlens=_cu_seqlens,
                    max_seqlen=_max_seqlen,
                    window_size=_window_size,
                    position_ids=_position_ids,
                    token_mask=_token_mask,
                    repo_active=_repo_active,
                )

            try:
                attn_out = _checkpoint(_attn_branch, x, use_reentrant=False)
            except TypeError:  # older torch checkpoint API
                attn_out = _checkpoint(_attn_branch, x)
            x = x + prores_alpha * attn_out
        else:
            fuse_ln_a = (
                self.canon_a is not None
                and isinstance(self.attn_norm, (nn.LayerNorm, nn.RMSNorm))
                and cu_seqlens is not None
            )
            if fuse_ln_a:
                attn_in = self.canon_a(
                    x,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    fuse_ln_weight=self.attn_norm.weight,
                    fuse_ln_eps=self.attn_norm.eps,
                    fuse_norm_type=self.norm_type,
                )
            else:
                attn_in = self.attn_norm(x)
                if self.canon_a is not None:
                    attn_in = self.canon_a(
                        attn_in,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                        attention_mask=token_mask,
                    )
            attn_out = self.attn(
                attn_in,
                cos_sin=cos_sin,
                attn_mask=attn_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                window_size=window_size,
                position_ids=position_ids,
                token_mask=token_mask,
                repo_active=repo_active,
            )
            x = x + prores_alpha * attn_out
        if self.mlp is not None:
            do_ckpt_mlp = (
                self.activation_checkpointing
                and self.training
                and self.activation_checkpointing_mode == "attn+mlp"
            )
            if do_ckpt_mlp:
                _position_ids = position_ids
                _cu_seqlens = cu_seqlens
                _token_mask = token_mask
                _fuse_ln_c = (
                    self.canon_c is not None
                    and isinstance(self.mlp_norm, (nn.LayerNorm, nn.RMSNorm))
                    and _cu_seqlens is not None
                )

                def _mlp_branch(
                    x_in: torch.Tensor,
                    *,
                    _layer: "ModernBertEncoderLayer" = self,
                ) -> torch.Tensor:
                    if _fuse_ln_c:
                        mlp_in = _layer.canon_c(
                            x_in,
                            position_ids=_position_ids,
                            cu_seqlens=_cu_seqlens,
                            fuse_ln_weight=_layer.mlp_norm.weight,
                            fuse_ln_eps=_layer.mlp_norm.eps,
                            fuse_norm_type=_layer.norm_type,
                        )
                    else:
                        mlp_in = _layer.mlp_norm(x_in)
                        if _layer.canon_c is not None:
                            mlp_in = _layer.canon_c(
                                mlp_in,
                                position_ids=_position_ids,
                                cu_seqlens=_cu_seqlens,
                                attention_mask=_token_mask,
                            )
                    return _layer.mlp(
                        mlp_in,
                        position_ids=_position_ids,
                        cu_seqlens=_cu_seqlens,
                        attention_mask=_token_mask,
                    )

                try:
                    mlp_out = _checkpoint(_mlp_branch, x, use_reentrant=False)
                except TypeError:  # older torch checkpoint API
                    mlp_out = _checkpoint(_mlp_branch, x)
                x = x + prores_alpha * mlp_out
            else:
                fuse_ln_c = (
                    self.canon_c is not None
                    and isinstance(self.mlp_norm, (nn.LayerNorm, nn.RMSNorm))
                    and cu_seqlens is not None
                )
                if fuse_ln_c:
                    mlp_in = self.canon_c(
                        x,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                        fuse_ln_weight=self.mlp_norm.weight,
                        fuse_ln_eps=self.mlp_norm.eps,
                        fuse_norm_type=self.norm_type,
                    )
                else:
                    mlp_in = self.mlp_norm(x)
                    if self.canon_c is not None:
                        mlp_in = self.canon_c(
                            mlp_in,
                            position_ids=position_ids,
                            cu_seqlens=cu_seqlens,
                            attention_mask=token_mask,
                        )
                mlp_out = self.mlp(
                    mlp_in,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    attention_mask=token_mask,
                )
                x = x + prores_alpha * mlp_out
        return x


class ModernBertAttnResidual(nn.Module):
    """Attention residual branch: x -> x + attn(attn_norm(x))."""

    def __init__(self, encoder: ModernBertEncoderLayer):
        super().__init__()
        # Store encoder without registering as a submodule (MHCLiteSublayersLayer owns it).
        self.__dict__["_encoder"] = encoder
        self.attention_type = encoder.attention_type

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        repo_active: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        enc: ModernBertEncoderLayer = self.__dict__["_encoder"]
        attn_in = enc.attn_norm(x)
        if enc.canon_a is not None:
            attn_in = enc.canon_a(
                attn_in,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=token_mask,
            )
        return x + enc.attn(
            attn_in,
            cos_sin=cos_sin,
            attn_mask=attn_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            window_size=window_size,
            position_ids=position_ids,
            token_mask=token_mask,
            repo_active=repo_active,
        )


class ModernBertMLPResidual(nn.Module):
    """MLP residual branch: x -> x + mlp(mlp_norm(x))."""

    def __init__(self, encoder: ModernBertEncoderLayer):
        super().__init__()
        # Store encoder without registering as a submodule (MHCLiteSublayersLayer owns it).
        self.__dict__["_encoder"] = encoder
        self.attention_type = encoder.attention_type

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        repo_active: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        enc: ModernBertEncoderLayer = self.__dict__["_encoder"]
        if enc.mlp is None:
            return x
        mlp_in = enc.mlp_norm(x)
        if enc.canon_c is not None:
            mlp_in = enc.canon_c(
                mlp_in,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                attention_mask=token_mask,
            )
        return x + enc.mlp(
            mlp_in,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            attention_mask=token_mask,
        )


def _build_permutation_matrices(n: int) -> tuple[torch.Tensor, int]:
    """Pre-compute all n! permutation matrices (flattened) and identity index."""
    from itertools import permutations

    perms = list(permutations(range(n)))
    identity_idx = 0  # (0,1,...,n-1) is first in lexicographic order
    P = torch.zeros(len(perms), n * n)
    for i, perm in enumerate(perms):
        for row, col in enumerate(perm):
            P[i, row * n + col] = 1.0
    return P, identity_idx


class MHCLiteBlock(nn.Module):
    """mHC-lite: wraps a transformer layer with n residual streams
    and doubly stochastic mixing via convex combination of permutation matrices.

    Forward: x_streams (..., n, C) -> (..., n, C)
    x_{l+1} = H^res_l @ x_l + H^post_l * f(H^pre_l @ x_l)
    where f is the wrapped transformer layer (without residual).

    Optimized I/O: fused projection, merged H_res-h_post application.
    Optional Triton kernels for further fusion (set triton_fused=True).
    """

    def __init__(
        self,
        n_streams: int,
        hidden_size: int,
        layer: nn.Module,
        triton_fused: bool = False,
    ):
        super().__init__()
        self.n = n_streams
        self.C = hidden_size
        self.nC = n_streams * hidden_size
        self.layer = layer
        n_fact = math.factorial(n_streams)
        self.n_fact = n_fact
        self.triton_fused = triton_fused

        self.alpha_pre = nn.Parameter(torch.tensor([0.01]))
        self.alpha_post = nn.Parameter(torch.tensor([0.01]))
        self.alpha_res = nn.Parameter(torch.tensor([0.01]))

        # Fused single projection: pre(n) + post(n) + res(n!) outputs
        total_out = n_streams + n_streams + n_fact
        self.W_all = nn.Linear(self.nC, total_out, bias=True)

        perm_flat, self._identity_idx = _build_permutation_matrices(n_streams)
        self.register_buffer("perm_mat", perm_flat)  # (n!, n*n)

    @property
    def attention_type(self):
        return self.layer.attention_type

    def _mhc_coeffs_pytorch(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute h_pre/h_post/H_merged (PyTorch path)."""
        n = self.n
        dt = x_streams.dtype

        x_flat = x_streams.reshape(*x_streams.shape[:-2], self.nC)
        x_norm = F.rms_norm(x_flat, (self.nC,))

        all_proj = F.linear(x_norm, self.W_all.weight.to(dt), None)
        pre_proj, post_proj, res_proj = all_proj.split([n, n, self.n_fact], dim=-1)

        bias = self.W_all.bias.to(dt)
        pre_bias = bias[:n]
        post_bias = bias[n : 2 * n]
        res_bias = bias[2 * n :]

        h_pre = torch.sigmoid(self.alpha_pre.to(dt) * pre_proj + pre_bias)
        h_post = 2.0 * torch.sigmoid(self.alpha_post.to(dt) * post_proj + post_bias)
        a_res = F.softmax(self.alpha_res.to(dt) * res_proj + res_bias, dim=-1)

        H_res = torch.matmul(a_res, self.perm_mat.to(dt)).unflatten(-1, (n, n))
        H_merged = H_res - h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)
        return h_pre, h_post, H_merged

    def _mhc_pre_map_pytorch(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-map bundle: layer_input + coefficients for post-res."""
        h_pre, h_post, H_merged = self._mhc_coeffs_pytorch(x_streams)
        layer_input = torch.matmul(h_pre.unsqueeze(-2), x_streams).squeeze(-2)
        return layer_input, H_merged, h_post

    def _mhc_post_res_pytorch(
        self,
        x_streams: torch.Tensor,
        layer_output: torch.Tensor,
        H_merged: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(H_merged, x_streams) + h_post.unsqueeze(
            -1
        ) * layer_output.unsqueeze(-2)

    def _forward_pytorch(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        """Pure PyTorch forward path (always correct, works everywhere)."""
        pre_map = (
            self._compiled_mhc_pre_map_pytorch
            if self.training and hasattr(self, "_compiled_mhc_pre_map_pytorch")
            else self._mhc_pre_map_pytorch
        )
        post_res = (
            self._compiled_mhc_post_res_pytorch
            if self.training and hasattr(self, "_compiled_mhc_post_res_pytorch")
            else self._mhc_post_res_pytorch
        )
        use_ckpt = (
            USE_ACTIVATION_CHECKPOINTING_MHC
            and self.training
            and torch.is_grad_enabled()
            and x_streams.requires_grad
        )

        if use_ckpt:
            layer_input, H_merged, h_post = _checkpoint(
                lambda x_: pre_map(x_),
                x_streams,
                use_reentrant=False,
            )
        else:
            layer_input, H_merged, h_post = pre_map(x_streams)
        layer_output = self.layer(layer_input, **kwargs)

        if use_ckpt:
            return _checkpoint(
                lambda x_, lo_, H_, hp_: post_res(x_, lo_, H_, hp_),
                x_streams,
                layer_output,
                H_merged,
                h_post,
                use_reentrant=False,
            )
        return post_res(x_streams, layer_output, H_merged, h_post)

    def _mhc_coeffs_triton(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute h_pre/h_post/H_merged using fused K1 + PyTorch tiny ops."""
        n = self.n
        T = x_streams.shape[0]
        dt = x_streams.dtype

        x_flat = x_streams.reshape(T, self.nC)
        all_proj, _inv_rms = torch.ops.nanoplm_mhc.fused_rmsnorm_project(
            x_flat, self.W_all.weight.to(dt)
        )

        pre_proj, post_proj, res_proj = all_proj.split([n, n, self.n_fact], dim=-1)

        bias = self.W_all.bias.to(dt)
        h_pre = torch.sigmoid(self.alpha_pre.to(dt) * pre_proj + bias[:n])
        h_post = 2.0 * torch.sigmoid(
            self.alpha_post.to(dt) * post_proj + bias[n : 2 * n]
        )
        a_res = F.softmax(self.alpha_res.to(dt) * res_proj + bias[2 * n :], dim=-1)

        H_res = torch.matmul(a_res, self.perm_mat.to(dt)).unflatten(-1, (n, n))
        H_merged = H_res - h_post.unsqueeze(-1) * h_pre.unsqueeze(-2)
        return h_pre, h_post, H_merged

    def _mhc_pre_map_triton(
        self, x_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-map bundle for Triton path: layer_input + post-res coefficients."""
        h_pre, h_post, H_merged = self._mhc_coeffs_triton(x_streams)
        layer_input = torch.ops.nanoplm_mhc.fused_pre_map(x_streams, h_pre.float())
        return layer_input, H_merged.float(), h_post.float()

    def _mhc_post_res_triton(
        self,
        x_streams: torch.Tensor,
        layer_output: torch.Tensor,
        H_merged: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.nanoplm_mhc.fused_post_res(
            x_streams, layer_output, H_merged, h_post
        )

    def _forward_triton(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        """Triton-fused forward path. Requires (T, n, C) bf16 on CUDA.

        Uses Triton for the memory-heavy stream operations (K3: pre-map,
        K4: post-res) and PyTorch for the coefficient computation (tiny ops).
        """
        pre_map = (
            self._compiled_mhc_pre_map_triton
            if self.training and hasattr(self, "_compiled_mhc_pre_map_triton")
            else self._mhc_pre_map_triton
        )
        post_res = (
            self._compiled_mhc_post_res_triton
            if self.training and hasattr(self, "_compiled_mhc_post_res_triton")
            else self._mhc_post_res_triton
        )
        use_ckpt = (
            USE_ACTIVATION_CHECKPOINTING_MHC
            and self.training
            and torch.is_grad_enabled()
            and x_streams.requires_grad
        )
        # Eval/inference should avoid autotune warmup latency.
        autotune_ctx = (
            nullcontext()
            if self.training
            else _mhc_triton_ops.disable_autotune_temporarily()
        )

        with autotune_ctx:
            if use_ckpt:
                layer_input, H_merged, h_post = _checkpoint(
                    lambda x_: pre_map(x_),
                    x_streams,
                    use_reentrant=False,
                )
            else:
                layer_input, H_merged, h_post = pre_map(x_streams)

            # Transformer layer
            layer_output = self.layer(layer_input, **kwargs)

            # K4: Triton fused post-res (H_merged @ x + h_post * layer_output)
            if use_ckpt:
                return _checkpoint(
                    lambda x_, lo_, H_, hp_: post_res(x_, lo_, H_, hp_),
                    x_streams,
                    layer_output,
                    H_merged,
                    h_post,
                    use_reentrant=False,
                )
            return post_res(x_streams, layer_output, H_merged, h_post)

    def forward(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        """x_streams: (..., n, C).  Returns (..., n, C)."""
        # Use Triton path when: triton_fused=True, CUDA, bf16, 2D token dim
        use_triton = (
            self.triton_fused
            and x_streams.is_cuda
            and x_streams.dtype == torch.bfloat16
            and x_streams.dim() == 3  # (T, n, C) — no batch dim
        )

        if use_triton:
            return self._forward_triton(x_streams, **kwargs)
        return self._forward_pytorch(x_streams, **kwargs)


class MHCLiteSublayersLayer(nn.Module):
    """Transformer layer with mHC-lite applied to attention and MLP sublayers separately."""

    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.enc = ModernBertEncoderLayer(config, layer_idx)
        self.mhc_attn = MHCLiteBlock(
            config.mhc_n_streams,
            config.hidden_size,
            ModernBertAttnResidual(self.enc),
            triton_fused=config.mhc_triton_fused,
        )
        self.mhc_mlp = (
            MHCLiteBlock(
                config.mhc_n_streams,
                config.hidden_size,
                ModernBertMLPResidual(self.enc),
                triton_fused=config.mhc_triton_fused,
            )
            if self.enc.mlp is not None
            else None
        )

    @property
    def attention_type(self):
        return self.enc.attention_type

    def forward(self, x_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        x_streams = self.mhc_attn(x_streams, **kwargs)
        if self.mhc_mlp is not None:
            x_streams = self.mhc_mlp(x_streams, **kwargs)
        return x_streams


class ModernBertModel(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        if config.use_mhc_lite:
            if config.mhc_lite_wrapping_level == "layer":
                self.layers = nn.ModuleList(
                    [
                        MHCLiteBlock(
                            config.mhc_n_streams,
                            config.hidden_size,
                            ModernBertEncoderLayer(config, i),
                            triton_fused=config.mhc_triton_fused,
                        )
                        for i in range(config.num_hidden_layers)
                    ]
                )
            else:
                self.layers = nn.ModuleList(
                    [
                        MHCLiteSublayersLayer(config, i)
                        for i in range(config.num_hidden_layers)
                    ]
                )
        else:
            self.layers = nn.ModuleList(
                [
                    ModernBertEncoderLayer(config, i)
                    for i in range(config.num_hidden_layers)
                ]
            )
        # ---- Huginn-style recycling: split into prelude/recurrent/coda ----
        P = config.recycling_num_prelude_layers
        C = config.recycling_num_coda_layers
        L = config.num_hidden_layers
        # Index-slice into self.layers so the three lists SHARE the same Module
        # instances — preserves init flow + checkpoint compat. Mutating a layer
        # via any of the three lists is reflected in self.layers and vice versa.
        self.prelude_layers = nn.ModuleList(list(self.layers[:P]))
        self.recurrent_layers = nn.ModuleList(list(self.layers[P:L - C])) if (L - P - C) > 0 else nn.ModuleList()
        self.coda_layers = nn.ModuleList(list(self.layers[L - C:])) if C > 0 else nn.ModuleList()

        # Adapter only when recycling is active.
        if config.recycling_loop_count > 1:
            self.adapter = nn.Linear(
                2 * config.hidden_size,
                config.hidden_size,
                bias=config.recycling_adapter_bias,
            )
        else:
            self.adapter = None

        if config.use_resid_lambdas:
            self.resid_lambdas = nn.Parameter(torch.ones(config.num_hidden_layers))
        else:
            self.register_parameter("resid_lambdas", None)
        if config.use_x0_lambdas:
            self.x0_lambdas = nn.Parameter(torch.zeros(config.num_hidden_layers))
        else:
            self.register_parameter("x0_lambdas", None)
        self.final_norm = _create_norm(
            config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
        )
        self.rotary_emb = ModernBertRotaryEmbedding(config)
        # RePO: disabled at init; enabled by the training loop after
        # repo_rope_warmup_steps (or warmup_steps fallback).
        self.repo_active = False
        # ProRes: progressive residual warmup. The training loop calls
        # update_prores_alphas(step) once per optimizer step.
        # Stored as a non-persistent buffer so torch.compile treats values as
        # dynamic (no graph-break / recompilation per unique float).
        self.use_prores = config.use_prores
        self._prores_T = config.prores_T
        _init_val = 0.0 if config.use_prores else 1.0
        self.register_buffer(
            "_prores_alphas",
            torch.full((config.num_hidden_layers,), _init_val),
            persistent=False,
        )

    def update_prores_alphas(self, step: int) -> None:
        """Recompute per-layer ProRes alphas. Call once per optimizer step."""
        T = self._prores_T
        vals = [
            min(step / (T * (l + 1)), 1.0) for l in range(self.config.num_hidden_layers)
        ]
        self._prores_alphas.copy_(torch.tensor(vals, dtype=self._prores_alphas.dtype))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        _cu_seqlens: Optional[torch.Tensor] = None,
        _max_seqlen: Optional[int] = None,
        _position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ---- varlen (flash-attention) path --------------------------------
        if _cu_seqlens is not None:
            device = input_ids.device
            x = self.embeddings(input_ids)  # (total_tokens, hidden)
            if self.config.use_mhc_lite:
                n = self.config.mhc_n_streams
                x = F.pad(x.unsqueeze(-2), (0, 0, 0, n - 1))  # (T, n, C)
            x0 = x if self.x0_lambdas is not None else None
            if _position_ids is None:
                _position_ids = _position_ids_from_cu_seqlens(
                    _cu_seqlens, x.shape[0], x.device
                )

            # Build RoPE tables only up to the current batch's max sequence
            # length when available. This trims needless work for short-sequence
            # packed batches while preserving a safe fallback for manual callers.
            rope_len = (
                _max_seqlen
                if _max_seqlen is not None
                else self.config.max_position_embeddings
            )
            cos_f, sin_f = self.rotary_emb(rope_len, device, x.dtype, "full_attention")
            cos_s, sin_s = self.rotary_emb(
                rope_len, device, x.dtype, "sliding_attention"
            )
            rope = {
                "full_attention": (
                    self.rotary_emb.paired_from_positions(
                        _position_ids,
                        device,
                        x.dtype,
                        "full_attention",
                    )
                    if self.config.use_paired_head_attention
                    else (cos_f[0, _position_ids], sin_f[0, _position_ids])
                ),
                "sliding_attention": (
                    self.rotary_emb.paired_from_positions(
                        _position_ids,
                        device,
                        x.dtype,
                        "sliding_attention",
                    )
                    if self.config.use_paired_head_attention
                    else (cos_s[0, _position_ids], sin_s[0, _position_ids])
                ),
            }
            windows = {
                "full_attention": (-1, -1),
                "sliding_attention": (
                    self.config.sliding_window,
                    self.config.sliding_window,
                ),
            }

            repo_active = self.repo_active
            prores_alphas = self._prores_alphas  # (num_layers,) tensor buffer

            for i, layer in enumerate(self.layers):
                if self.resid_lambdas is not None:
                    x = self.resid_lambdas[i] * x
                if self.x0_lambdas is not None:
                    x = x + self.x0_lambdas[i] * x0
                alpha = prores_alphas[i]
                lt = layer.attention_type
                if (
                    self.config.activation_checkpointing
                    and self.training
                    and str(self.config.activation_checkpointing_mode).strip().lower()
                    == "layer"
                ):
                    cos_sin = rope[lt]
                    cu_seqlens = _cu_seqlens
                    max_seqlen = _max_seqlen
                    window_size = windows[lt]
                    position_ids = _position_ids

                    def _layer_forward(
                        x_in: torch.Tensor,
                        *,
                        _layer: nn.Module = layer,
                        _cos_sin=cos_sin,
                        _cu_seqlens=cu_seqlens,
                        _max_seqlen=max_seqlen,
                        _window_size=window_size,
                        _position_ids=position_ids,
                        _repo_active: bool = repo_active,
                        _prores_alpha=alpha,
                    ) -> torch.Tensor:
                        return _layer(
                            x_in,
                            cos_sin=_cos_sin,
                            cu_seqlens=_cu_seqlens,
                            max_seqlen=_max_seqlen,
                            window_size=_window_size,
                            position_ids=_position_ids,
                            repo_active=_repo_active,
                            prores_alpha=_prores_alpha,
                        )

                    try:
                        x = _checkpoint(_layer_forward, x, use_reentrant=False)
                    except TypeError:  # older torch checkpoint API
                        x = _checkpoint(_layer_forward, x)
                else:
                    x = layer(
                        x,
                        cos_sin=rope[lt],
                        cu_seqlens=_cu_seqlens,
                        max_seqlen=_max_seqlen,
                        window_size=windows[lt],
                        position_ids=_position_ids,
                        repo_active=repo_active,
                        prores_alpha=alpha,
                    )

            if self.config.use_mhc_lite:
                x = x[..., 0, :]  # compress: take stream 0
            return self.final_norm(x)

        # ---- SDPA (fallback) path -----------------------------------------
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}"
            )
        device = input_ids.device

        x = self.embeddings(input_ids)
        if self.config.use_mhc_lite:
            n = self.config.mhc_n_streams
            x = F.pad(x.unsqueeze(-2), (0, 0, 0, n - 1))  # (B, S, n, C)
        x0 = x if self.x0_lambdas is not None else None

        attn_masks = {
            "full_attention": _full_attention_mask(attention_mask),
            "sliding_attention": _sliding_attention_mask(
                attention_mask,
                seq_len=seq_len,
                sliding_window=self.config.sliding_window,
                device=device,
            ),
        }

        rope = {
            "full_attention": (
                self.rotary_emb.paired(
                    seq_len=seq_len,
                    device=device,
                    dtype=x.dtype,
                    layer_type="full_attention",
                )
                if self.config.use_paired_head_attention
                else self.rotary_emb(
                    seq_len=seq_len,
                    device=device,
                    dtype=x.dtype,
                    layer_type="full_attention",
                )
            ),
            "sliding_attention": (
                self.rotary_emb.paired(
                    seq_len=seq_len,
                    device=device,
                    dtype=x.dtype,
                    layer_type="sliding_attention",
                )
                if self.config.use_paired_head_attention
                else self.rotary_emb(
                    seq_len=seq_len,
                    device=device,
                    dtype=x.dtype,
                    layer_type="sliding_attention",
                )
            ),
        }

        repo_active = self.repo_active
        prores_alphas = self._prores_alphas  # (num_layers,) tensor buffer

        for i, layer in enumerate(self.layers):
            if self.resid_lambdas is not None:
                x = self.resid_lambdas[i] * x
            if self.x0_lambdas is not None:
                x = x + self.x0_lambdas[i] * x0
            alpha = prores_alphas[i]
            layer_type = layer.attention_type
            if (
                self.config.activation_checkpointing
                and self.training
                and str(self.config.activation_checkpointing_mode).strip().lower()
                == "layer"
            ):
                attn_mask = attn_masks[layer_type]
                cos_sin = rope[layer_type]
                token_mask = attention_mask

                def _layer_forward(
                    x_in: torch.Tensor,
                    *,
                    _layer: nn.Module = layer,
                    _attn_mask=attn_mask,
                    _cos_sin=cos_sin,
                    _token_mask=token_mask,
                    _repo_active: bool = repo_active,
                    _prores_alpha=alpha,
                ) -> torch.Tensor:
                    return _layer(
                        x_in,
                        attn_mask=_attn_mask,
                        cos_sin=_cos_sin,
                        position_ids=None,
                        token_mask=_token_mask,
                        repo_active=_repo_active,
                        prores_alpha=_prores_alpha,
                    )

                try:
                    x = _checkpoint(_layer_forward, x, use_reentrant=False)
                except TypeError:  # older torch checkpoint API
                    x = _checkpoint(_layer_forward, x)
            else:
                x = layer(
                    x,
                    attn_mask=attn_masks[layer_type],
                    cos_sin=rope[layer_type],
                    position_ids=None,
                    token_mask=attention_mask,
                    repo_active=repo_active,
                    prores_alpha=alpha,
                )

        if self.config.use_mhc_lite:
            x = x[..., 0, :]  # compress: take stream 0
        return self.final_norm(x)


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.classifier_bias,
        )
        self.norm = _create_norm(
            config.hidden_size, config.norm_eps, config.norm_bias, config.norm_type
        )
        self.act = _get_activation(config.classifier_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(x)))


class ModernBertForMaskedLM(nn.Module):
    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

        self.sparse_prediction = config.sparse_prediction
        self.sparse_pred_ignore_index = config.sparse_pred_ignore_index

        self.init_weights()

        if config.tie_word_embeddings:
            self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    @torch.no_grad()
    def init_weights(self) -> None:

        width = self.config.hidden_size
        bound = math.sqrt(3.0 / width)
        # Residual-branch out-projections (attn.Wo, mlp.Wo) are configurable.
        # For MuonH, zero-init is invalid because ||W_init||_F defines a hard
        # hyperball radius and a zero-initialized matrix would be locked at zero.
        wo_init = self.config.wo_init
        mlp_out_bound = math.sqrt(3.0 / self.config.intermediate_size)
        # NOBLE half-Kaiming: halve init scale for layers that use Kaiming
        # (Wqkv, Wi).
        noble_kaiming = (
            self.config.use_noble and self.config.noble_half_kaiming
        )
        kaiming_bound = bound / 2 if noble_kaiming else bound

        def _noble_bound(mod: nn.Module) -> float:
            return kaiming_bound if isinstance(mod, NOBLELinear) else bound

        def _init_wo(tensor: torch.Tensor, init_bound: float) -> None:
            if wo_init == "zero":
                nn.init.zeros_(tensor)
            else:
                nn.init.uniform_(tensor, -init_bound, init_bound)

        embedding_std = 0.02 if self.config.tie_word_embeddings else 1.0

        nn.init.normal_(
            self.model.embeddings.tok_embeddings.weight,
            mean=0.0,
            std=embedding_std,
        )

        for module in self.modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        for layer in self.model.layers:
            if isinstance(layer, MHCLiteBlock):
                enc = layer.layer
                mhc_blocks = [layer]
            elif isinstance(layer, MHCLiteSublayersLayer):
                enc = layer.enc
                mhc_blocks = [
                    b for b in (layer.mhc_attn, layer.mhc_mlp) if b is not None
                ]
            else:
                enc = layer
                mhc_blocks = []
            # QKV weight init (NOBLE modules get half-Kaiming bound)
            if enc.attn.Wqkv is not None:
                b = _noble_bound(enc.attn.Wqkv)
                nn.init.uniform_(enc.attn.Wqkv.weight, -b, b)
            else:
                bq = _noble_bound(enc.attn.Wq)
                nn.init.uniform_(enc.attn.Wq.weight, -bq, bq)
                if enc.attn.Wq2 is not None:
                    bq2 = _noble_bound(enc.attn.Wq2)
                    nn.init.uniform_(enc.attn.Wq2.weight, -bq2, bq2)
                bk = _noble_bound(enc.attn.Wk)
                nn.init.uniform_(enc.attn.Wk.weight, -bk, bk)
                bv = _noble_bound(enc.attn.Wv)
                nn.init.uniform_(enc.attn.Wv.weight, -bv, bv)
            _init_wo(enc.attn.Wo.weight, bound)

            # MLP weight init (NOBLE modules get half-Kaiming bound)
            if enc.mlp is not None:
                is_moe_mlp = False
                if getattr(self.config, "use_moe", False):
                    from .moe import MoELayer as _MoELayer
                    is_moe_mlp = isinstance(enc.mlp, _MoELayer)

                if is_moe_mlp:
                    # Routed expert stacked weights (nn.Parameter, not nn.Linear).
                    nn.init.uniform_(enc.mlp.Wi, -bound, bound)
                    _init_wo(enc.mlp.Wo, mlp_out_bound)
                    nn.init.uniform_(enc.mlp.router.gate.weight, -bound, bound)
                    shared = enc.mlp.shared_expert
                    if shared.Wi is not None:
                        bi = _noble_bound(shared.Wi)
                        nn.init.uniform_(shared.Wi.weight, -bi, bi)
                    elif hasattr(shared, "W_up") and shared.W_up is not None:
                        bu = _noble_bound(shared.W_up)
                        nn.init.uniform_(shared.W_up.weight, -bu, bu)
                        bg = _noble_bound(shared.W_gate)
                        nn.init.uniform_(shared.W_gate.weight, -bg, bg)
                    _init_wo(shared.Wo.weight, mlp_out_bound)
                elif enc.mlp.Wi is not None:
                    bi = _noble_bound(enc.mlp.Wi)
                    nn.init.uniform_(enc.mlp.Wi.weight, -bi, bi)
                elif hasattr(enc.mlp, "W_up") and enc.mlp.W_up is not None:
                    bu = _noble_bound(enc.mlp.W_up)
                    nn.init.uniform_(enc.mlp.W_up.weight, -bu, bu)
                    bg = _noble_bound(enc.mlp.W_gate)
                    nn.init.uniform_(enc.mlp.W_gate.weight, -bg, bg)
                if not is_moe_mlp:
                    if hasattr(enc.mlp, "Wo"):
                        _init_wo(enc.mlp.Wo.weight, mlp_out_bound)
                    else:
                        _init_wo(enc.mlp.Wo_weight, mlp_out_bound)

            # QKV bias init
            if enc.attn.Wqkv is not None:
                if enc.attn.Wqkv.bias is not None:
                    nn.init.zeros_(enc.attn.Wqkv.bias)
            else:
                if enc.attn.Wq.bias is not None:
                    nn.init.zeros_(enc.attn.Wq.bias)
                if enc.attn.Wq2 is not None and enc.attn.Wq2.bias is not None:
                    nn.init.zeros_(enc.attn.Wq2.bias)
                if enc.attn.Wk.bias is not None:
                    nn.init.zeros_(enc.attn.Wk.bias)
                if enc.attn.Wv.bias is not None:
                    nn.init.zeros_(enc.attn.Wv.bias)
            if enc.attn.Wo.bias is not None:
                nn.init.zeros_(enc.attn.Wo.bias)

            # MLP bias init
            if enc.mlp is not None:
                is_moe_mlp = False
                if getattr(self.config, "use_moe", False):
                    from .moe import MoELayer as _MoELayer
                    is_moe_mlp = isinstance(enc.mlp, _MoELayer)

                if is_moe_mlp:
                    shared = enc.mlp.shared_expert
                    if shared.Wi is not None:
                        if shared.Wi.bias is not None:
                            nn.init.zeros_(shared.Wi.bias)
                    elif hasattr(shared, "W_up") and shared.W_up is not None:
                        if shared.W_up.bias is not None:
                            nn.init.zeros_(shared.W_up.bias)
                        if shared.W_gate.bias is not None:
                            nn.init.zeros_(shared.W_gate.bias)
                    if shared.Wo.bias is not None:
                        nn.init.zeros_(shared.Wo.bias)
                elif enc.mlp.Wi is not None:
                    if enc.mlp.Wi.bias is not None:
                        nn.init.zeros_(enc.mlp.Wi.bias)
                elif hasattr(enc.mlp, "W_up") and enc.mlp.W_up is not None:
                    if enc.mlp.W_up.bias is not None:
                        nn.init.zeros_(enc.mlp.W_up.bias)
                    if enc.mlp.W_gate.bias is not None:
                        nn.init.zeros_(enc.mlp.W_gate.bias)
                if not is_moe_mlp:
                    if hasattr(enc.mlp, "Wo"):
                        if enc.mlp.Wo.bias is not None:
                            nn.init.zeros_(enc.mlp.Wo.bias)
                    elif enc.mlp.Wo_bias is not None:
                        nn.init.zeros_(enc.mlp.Wo_bias)

            # DiffV2: zero-init lambda_proj so sigmoid(0)=0.5 at start.
            if enc.attn.lambda_proj is not None:
                nn.init.zeros_(enc.attn.lambda_proj.weight)

            # RePO: zero-init W_z so positions start at zero (NoPE-like).
            # W_g and W_c keep default Kaiming uniform init.
            if enc.attn.repo is not None:
                nn.init.zeros_(enc.attn.repo.W_z.weight)

            # mHC-lite: zero-init fused projection, set biases for identity behavior
            def _init_mhc_block(block: MHCLiteBlock) -> None:
                nn.init.zeros_(block.W_all.weight)
                n_s = block.n
                bias = block.W_all.bias.data
                # pre bias: first n values
                bias[:n_s].fill_(-1.0)
                bias[0] = 1.0
                # post bias: next n values
                bias[n_s : 2 * n_s].fill_(-1.0)
                bias[n_s] = 1.0
                # res bias: last n! values
                bias[2 * n_s :].fill_(-8.0)
                bias[2 * n_s + block._identity_idx] = 0.0
                block.alpha_pre.fill_(0.01)
                block.alpha_post.fill_(0.01)
                block.alpha_res.fill_(0.01)

            for block in mhc_blocks:
                _init_mhc_block(block)

        nn.init.uniform_(self.head.dense.weight, -bound, bound)
        if self.head.dense.bias is not None:
            nn.init.zeros_(self.head.dense.bias)

        decoder_std = embedding_std if self.config.tie_word_embeddings else 0.001
        nn.init.normal_(self.decoder.weight, mean=0.0, std=decoder_std)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        if self.model.resid_lambdas is not None:
            self.model.resid_lambdas.fill_(self.config.resid_lambda_init)
        if self.model.x0_lambdas is not None:
            self.model.x0_lambdas.fill_(self.config.x0_lambda_init)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embeddings.tok_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        # ---- static packed path (pre-flattened by collator) ---------------
        # When both cu_seqlens and position_ids are provided and input_ids is
        # already 1-D (flat), the collator has done all unpadding / position-id
        # computation.  No data-dependent ops here → dynamic=False safe.
        if cu_seqlens is not None and position_ids is not None and input_ids.dim() == 1:
            if not _HAS_FLASH_VARLEN:
                raise RuntimeError(
                    "Sequence packing requires flash attention (flash_attn or "
                    "flash_attn_interface)."
                )
            x = self.model(
                input_ids,
                _cu_seqlens=cu_seqlens,
                _max_seqlen=max_seqlen,  # pass int directly — no tensor
                _position_ids=position_ids,
            )
            # x: (F, hidden) where F is fixed flat length

            if labels is not None:
                logits = self.decoder(self.head(x))
                loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    ignore_index=self.sparse_pred_ignore_index,
                )
                # MoE auxiliary losses (balance + z-loss, coefficients baked in).
                if self.config.use_moe:
                    from .moe import MoELayer as _MoELayer
                    aux_losses = [
                        module.last_aux_loss
                        for module in self.modules()
                        if isinstance(module, _MoELayer)
                        and module.last_aux_loss is not None
                    ]
                    if aux_losses:
                        loss = loss + sum(aux_losses) / len(aux_losses)
            else:
                logits = self.decoder(self.head(x))
                loss = None

            return {"loss": loss, "logits": logits}

        use_varlen = (
            _HAS_FLASH_VARLEN and input_ids.is_cuda and attention_mask is not None
        )

        # ---- varlen (flash-attention) path --------------------------------
        if use_varlen:
            batch, seq_len = input_ids.shape
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

            if cu_seqlens is not None:
                # Packed path: cu_seqlens provided by the packing collator.
                # max_seqlen may be an int or a tensor; keep consistent.
                if isinstance(max_seqlen, int):
                    max_seqlen_t = torch.tensor(max_seqlen, dtype=torch.int32)
                else:
                    max_seqlen_t = max_seqlen
            else:
                # Unpacked path: derive cu_seqlens from attention_mask.
                _indices, cu_seqlens, max_seqlen_t = _unpad_input(attention_mask)
                indices = _indices

            flat_ids = input_ids.view(-1)[indices]  # (total_tokens,)
            position_ids = _position_ids_from_cu_seqlens(
                cu_seqlens, flat_ids.shape[0], flat_ids.device
            )

            x = self.model(
                flat_ids,
                _cu_seqlens=cu_seqlens,
                _max_seqlen=max_seqlen_t,
                _position_ids=position_ids,
            )
            # x: (total_tokens, hidden) — flat, no padding

            if self.sparse_prediction and labels is not None:
                flat_labels = labels.view(-1)[indices]
                keep = flat_labels != self.sparse_pred_ignore_index
                logits = self.decoder(self.head(x[keep]))
                loss = F.cross_entropy(logits.float(), flat_labels[keep])
            elif labels is not None:
                flat_labels = labels.view(-1)[indices]
                logits = self.decoder(self.head(x))
                loss = F.cross_entropy(
                    logits.float(),
                    flat_labels,
                    ignore_index=self.sparse_pred_ignore_index,
                )
            else:
                logits = self.decoder(self.head(x))
                logits = _pad_output(logits, indices, batch, seq_len)
                loss = None

            return {"loss": loss, "logits": logits}

        # ---- SDPA (fallback) path -----------------------------------------
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.sparse_prediction and labels is not None:
            flat_labels = labels.view(-1)
            x = x.view(flat_labels.shape[0], -1)
            keep = flat_labels != self.sparse_pred_ignore_index
            x = x[keep]
            flat_labels = flat_labels[keep]
        else:
            flat_labels = labels

        logits = self.decoder(self.head(x))

        loss = None
        if labels is not None:
            if self.sparse_prediction:
                loss = F.cross_entropy(logits.float(), flat_labels)
            else:
                loss = F.cross_entropy(
                    logits.float().view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=self.sparse_pred_ignore_index,
                )

        return {"loss": loss, "logits": logits}

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not only_trainable)
        )


def map_hf_state_dict_to_pure(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {k: v for k, v in hf_state_dict.items() if not k.startswith("_")}


def map_pure_state_dict_to_hf(
    pure_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return dict(pure_state_dict)
