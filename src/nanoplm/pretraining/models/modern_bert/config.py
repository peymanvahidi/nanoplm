"""ModernBERT model configuration and helper factories."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    rope_dim_fraction: float = 1.0  # fraction of head_dim to apply RoPE; remainder are NoPE content dims
    use_resid_lambdas: bool = False
    use_x0_lambdas: bool = False
    use_qk_norm: bool = False
    reorder_RoPE_QKNorm: bool = False  # True = QK norm BEFORE RoPE (LLaMA-style); False = RoPE first (original)
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
    # When False, Q/K/V use separate nn.Linear modules instead of a single Wqkv.
    # This keeps Muon orthogonalization independent per subspace.
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
    # Routed-expert compute backend (pure_torch MoE only). 'cutlass' = the custom
    # device-side grouped-GEMM kernel (default, sync-free). 'sonicmoe' = the
    # SonicMoE/quack fused expert path (sync-free fwd+bwd, fused dispatch gather);
    # requires mlp_activation in {srelu, swiglu}.
    moe_kernel_backend: Literal["cutlass", "sonicmoe"] = "cutlass"

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
    rope_dim: int = field(init=False)
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
        # Partial RoPE: compute rope_dim from fraction.
        self.rope_dim_fraction = float(self.rope_dim_fraction)
        if not (0.0 < self.rope_dim_fraction <= 1.0):
            raise ValueError(
                f"rope_dim_fraction must be in (0.0, 1.0], got {self.rope_dim_fraction}"
            )
        raw_rope_dim = int(self.head_dim * self.rope_dim_fraction)
        # Round down to nearest even number (RoPE needs pairs).
        self.rope_dim = max(2, raw_rope_dim - raw_rope_dim % 2)
        if self.rope_dim > self.head_dim:
            self.rope_dim = self.head_dim
        self.sliding_window = self.local_attention // 2
        self.mlp_activation = self.mlp_activation.lower()
        if self.mlp_activation not in {"swiglu", "glu", "srelu"}:
            raise ValueError(
                f"Unsupported mlp_activation: {self.mlp_activation}. "
                "Supported: ['swiglu', 'glu', 'srelu']"
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
            if self.mlp_activation not in {"swiglu", "srelu"}:
                raise ValueError(
                    "use_moe=True requires mlp_activation in {'swiglu', 'srelu'}."
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
            if self.moe_kernel_backend not in {"cutlass", "sonicmoe"}:
                raise ValueError(
                    "moe_kernel_backend must be 'cutlass' or 'sonicmoe', got "
                    f"{self.moe_kernel_backend!r}"
                )
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
