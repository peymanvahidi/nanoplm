"""Configuration dataclass for the pure-torch decoder-only (causal LM) pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DecoderConfig:
    """Configuration for the pure-torch decoder-only (causal LM) model.

    Mirrors ``ProtModernBertMLMConfig`` but removes MLM-specific fields
    (``num_register_tokens``, ``classifier_activation``) and adds
    decoder-specific ones (``max_position_embeddings``, ``local_attention``,
    token IDs, ``norm_eps``, ``norm_bias``, ``decoder_bias``).
    """

    # ------------------------------------------------------------------ #
    # Core architecture                                                    #
    # ------------------------------------------------------------------ #
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int

    vocab_size: int = 32
    num_kv_heads: Optional[int] = None  # GQA: None = MHA (same as num_attention_heads)

    norm_type: str = "layernorm"   # layernorm | rmsnorm
    norm_eps: float = 1e-5
    norm_bias: bool = False

    mlp_activation: str = "swiglu"
    mlp_dropout: float = 0.0
    mlp_bias: bool = False
    no_mlp_on_first_layer: bool = True

    attention_bias: bool = False
    attention_dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Positional / rotary                                                  #
    # ------------------------------------------------------------------ #
    max_position_embeddings: int = 8192
    global_rope_theta: float = 160_000.0  # RoPE base for full-attention layers
    local_rope_theta: float = 10_000.0    # RoPE base for sliding-window layers
    rope_dim_fraction: float = 1.0        # fraction of head_dim that gets RoPE

    # Sliding-window span in tokens.  Decoder causal sliding layers attend to
    # at most ``local_attention // 2`` past tokens on each side (right side is
    # always 0 for causal).
    local_attention: int = 128

    # ------------------------------------------------------------------ #
    # Attention pattern                                                    #
    # ------------------------------------------------------------------ #
    # None → PureDecoderLM forces all-full causal ("F" pattern).
    # Pass e.g. "FSS" for (1 full + 2 sliding) tiled across layers.
    attn_layer_pattern: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Token IDs                                                            #
    # ------------------------------------------------------------------ #
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Output head                                                          #
    # ------------------------------------------------------------------ #
    tie_word_embeddings: bool = True
    decoder_bias: bool = True

    # ------------------------------------------------------------------ #
    # Residual scaling                                                     #
    # ------------------------------------------------------------------ #
    use_resid_lambdas: bool = False
    resid_lambda_init: float = 1.0
    use_x0_lambdas: bool = False
    x0_lambda_init: float = 0.1

    # ------------------------------------------------------------------ #
    # Q/K normalisation                                                    #
    # ------------------------------------------------------------------ #
    use_qk_norm: bool = False
    reorder_RoPE_QKNorm: bool = False  # True = QK norm BEFORE RoPE (LLaMA-style)

    # ------------------------------------------------------------------ #
    # Canon layers                                                         #
    # ------------------------------------------------------------------ #
    use_canon_layers: bool = False
    canon_layers_mode: str = "abcd"
    canon_layers_kernel_size: Optional[int] = None

    # ------------------------------------------------------------------ #
    # ProRes                                                               #
    # ------------------------------------------------------------------ #
    use_prores: bool = False
    prores_T: int = 1000

    # ------------------------------------------------------------------ #
    # Activation checkpointing                                             #
    # ------------------------------------------------------------------ #
    activation_checkpointing: bool = False
    activation_checkpointing_mode: str = "layer"  # layer | attn | attn+mlp

    # ------------------------------------------------------------------ #
    # Projection layout                                                    #
    # ------------------------------------------------------------------ #
    fused_qkv: bool = True
    fused_up_gate: bool = True
    wo_init: str = "zero"   # zero | kaiming

    # ------------------------------------------------------------------ #
    # NOBLE                                                                #
    # ------------------------------------------------------------------ #
    use_noble: bool = False
    noble_rank: int = 64
    noble_alpha: float = 0.01
    noble_omega_range: tuple = (0.8, 1.2)
    noble_phi_std: float = 0.1
    noble_half_kaiming: bool = True
    noble_targets: str = "all"  # all | attn | ffn | qkv | out

    # ------------------------------------------------------------------ #
    # MoE                                                                  #
    # ------------------------------------------------------------------ #
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_use_bias_correction: bool = True
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 5e-5
    moe_routed_scaling_factor: float = 1.0
    moe_n_group: int = 1
    moe_topk_group: int = 1
    moe_bias_update_rate: float = 1e-3
    moe_leading_dense_layers: int = 0
    moe_kernel_backend: str = "cutlass"  # cutlass | sonicmoe
