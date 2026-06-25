from __future__ import annotations

from nanoplm.pretraining.models.modern_bert.config import ModernBertConfig
from nanoplm.pretraining.models.decoder.config import DecoderConfig
from nanoplm.pretraining.models.decoder.modeling import DecoderForCausalLM


class PureDecoderLM(DecoderForCausalLM):
    """Pure-torch decoder-only language model.

    Takes a ``DecoderConfig`` and builds a ``DecoderForCausalLM`` backed by
    the same ``ModernBertEncoderLayer`` / ``ModernBertAttention`` building
    blocks used by the encoder, but with causal attention.

    Default attention pattern
    -------------------------
    When ``config.attn_layer_pattern is None`` the model forces all layers to
    use full (global) causal attention by setting ``attn_layer_pattern="F"``.
    Pass an explicit pattern such as ``"FSS"`` to enable causal sliding-window
    layers (every 3rd layer full, others local).

    Example::

        cfg = DecoderConfig(
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            vocab_size=32,
        )
        model = PureDecoderLM(cfg)
    """

    def __init__(self, config: DecoderConfig):
        # Keep the user-facing config for checkpoint serialization.
        self.model_config = config

        # Default to all-full-attention when no pattern is given.
        attn_pattern = config.attn_layer_pattern if config.attn_layer_pattern is not None else "F"

        mb_config = ModernBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            norm_type=config.norm_type,
            norm_eps=config.norm_eps,
            norm_bias=config.norm_bias,
            mlp_activation=config.mlp_activation,
            mlp_bias=config.mlp_bias,
            mlp_dropout=config.mlp_dropout,
            no_mlp_on_first_layer=config.no_mlp_on_first_layer,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            max_position_embeddings=config.max_position_embeddings,
            global_rope_theta=config.global_rope_theta,
            local_rope_theta=config.local_rope_theta,
            rope_dim_fraction=config.rope_dim_fraction,
            local_attention=config.local_attention,
            attn_layer_pattern=attn_pattern,
            # Disable stride-based pattern; attn_layer_pattern takes precedence.
            global_attn_every_n_layers=1,
            tie_word_embeddings=config.tie_word_embeddings,
            decoder_bias=config.decoder_bias,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            bos_token_id=config.bos_token_id,
            use_qk_norm=config.use_qk_norm,
            reorder_RoPE_QKNorm=config.reorder_RoPE_QKNorm,
            use_resid_lambdas=config.use_resid_lambdas,
            resid_lambda_init=config.resid_lambda_init,
            use_x0_lambdas=config.use_x0_lambdas,
            x0_lambda_init=config.x0_lambda_init,
            fused_qkv=config.fused_qkv,
            fused_up_gate=config.fused_up_gate,
            wo_init=config.wo_init,
            use_prores=config.use_prores,
            prores_T=config.prores_T,
            activation_checkpointing=config.activation_checkpointing,
            activation_checkpointing_mode=str(
                getattr(config, "activation_checkpointing_mode", "layer")
            ).strip().lower(),
            use_canon_layers=config.use_canon_layers,
            canon_layers_mode=config.canon_layers_mode,
            canon_layers_kernel_size=config.canon_layers_kernel_size,
            use_noble=config.use_noble,
            noble_rank=config.noble_rank,
            noble_alpha=config.noble_alpha,
            noble_omega_range=config.noble_omega_range,
            noble_phi_std=config.noble_phi_std,
            noble_half_kaiming=config.noble_half_kaiming,
            noble_targets=config.noble_targets,
            use_moe=config.use_moe,
            moe_num_experts=config.moe_num_experts,
            moe_top_k=config.moe_top_k,
            moe_use_bias_correction=config.moe_use_bias_correction,
            moe_aux_loss_coef=config.moe_aux_loss_coef,
            moe_z_loss_coef=config.moe_z_loss_coef,
            moe_routed_scaling_factor=config.moe_routed_scaling_factor,
            moe_n_group=config.moe_n_group,
            moe_topk_group=config.moe_topk_group,
            moe_bias_update_rate=config.moe_bias_update_rate,
            moe_leading_dense_layers=config.moe_leading_dense_layers,
            moe_kernel_backend=config.moe_kernel_backend,
            # Decoder does not support Huginn recycling (always disabled).
            recycling_num_prelude_layers=0,
            recycling_num_coda_layers=0,
            recycling_loop_count=1,
            # Features not exposed in DecoderConfig (encoder-specific or
            # unsupported in the decoder yet).
            use_repo=False,
            use_mhc_lite=False,
            use_diff_attn_v2=False,
            use_paired_head_attention=False,
        )

        super().__init__(mb_config)
