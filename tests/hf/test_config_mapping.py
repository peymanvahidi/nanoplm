"""One mapping from ProtModernBertMLMConfig to ModernBertConfig, used everywhere."""
import ast
from dataclasses import fields
from pathlib import Path

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert import pure_model
from nanoplm.pretraining.models.modern_bert.pure_model import (
    modernbert_config_from,
    PureProtModernBertMLM,
)

# Fields intentionally not forwarded: they are consumed outside ModernBertConfig
# or renamed. Keep this list short and justified — a growing list means the
# mapping is drifting.
NOT_FORWARDED = {
    "num_register_tokens",  # goes to the tokenizer, not the model config
}

# ModernBertConfig(...) keywords that are legitimately NOT `config.<same_name>`:
# they don't correspond to a ProtModernBertMLMConfig field at all. Keep this list
# as short as the code requires — anything else showing up as a hardcoded value
# in the call is exactly the "silently dropped forwarding" bug this module guards
# against.
_ALLOWED_NON_FORWARDED_KEYWORDS = {
    "pad_token_id": "sourced from the tokenizer, not the model config",
    "eos_token_id": "sourced from the tokenizer, not the model config",
    "bos_token_id": "hardcoded None; ModernBERT has no BOS token",
    "unk_token_id": "sourced from the tokenizer, not the model config",
    "mask_token_id": "sourced from the tokenizer, not the model config",
    "max_position_embeddings": (
        "deliberate constant (RoPE-frequency cap), documented by a comment "
        "at the call site"
    ),
}


def test_mapping_is_invariant_to_num_register_tokens():
    """`num_register_tokens` is NOT_FORWARDED by design: register tokens occupy
    fixed vocab slots and never move pad/eos/unk/mask, so two configs that
    differ ONLY in `num_register_tokens` must map to an identical
    `ModernBertConfig`. This pins that invariance in code rather than losing
    the reason as tribal knowledge."""
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    mapped_0 = modernbert_config_from(
        ProtModernBertMLMConfig(num_register_tokens=0, **base)
    )
    mapped_2 = modernbert_config_from(
        ProtModernBertMLMConfig(num_register_tokens=2, **base)
    )
    assert mapped_0 == mapped_2


def test_mapped_config_exposes_every_field():
    mapped = modernbert_config_from(
        ProtModernBertMLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
    )
    missing = [
        f.name
        for f in fields(ProtModernBertMLMConfig)
        if f.name not in NOT_FORWARDED and not hasattr(mapped, f.name)
    ]
    assert missing == [], f"fields dropped by the mapping: {missing}"


# --- AST-level structural guard -------------------------------------------
#
# `test_mapped_config_exposes_every_field` above only checks `hasattr` on
# the *result*. `ModernBertConfig` is a dataclass where essentially every field
# already has a class-level default, so if a future edit silently drops a
# `x=config.x` line from the call, the resulting object still has attribute
# `x` — just holding ModernBertConfig's own default instead of the forwarded
# value. `hasattr` cannot see that. Parse the source instead: this catches a
# dropped forwarding line regardless of dataclass defaults, and also catches a
# keyword that's present but wired to a hardcoded constant instead of
# `config.<same_name>` — the same bug wearing a hat.
#
# This mirrors tools/schema_gen/dataclass_ast.py + tests/docs_schema/test_parity.py,
# which already parse this repo's config dataclasses statically via `ast` for the
# same reason: don't trust attribute presence, check the source that produced it.


def _modernbert_config_call() -> ast.Call:
    """Locate the `ModernBertConfig(...)` call inside `modernbert_config_from`."""
    source = Path(pure_model.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    func = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "modernbert_config_from"
    )
    return next(
        n
        for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "ModernBertConfig"
    )


def _forwarded_field_name(value_node: ast.expr):
    """If `value_node` forwards a single `config.<name>` field — optionally
    wrapped in str()/.strip()/.lower()/getattr(config, "name", default) — return
    that field's name. Otherwise return None (a hardcoded constant or a
    non-`config` source, e.g. the tokenizer)."""
    node = value_node
    while True:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("strip", "lower")
            and not node.args
            and not node.keywords
        ):
            node = node.func.value
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and len(node.args) == 1
            and not node.keywords
        ):
            node = node.args[0]
            continue
        break
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
    ):
        return node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "config"
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[1].value
    return None


def test_every_config_field_forwards_from_source():
    """Every ProtModernBertMLMConfig field (except NOT_FORWARDED) must appear as
    a keyword in the ModernBertConfig(...) call, wired to `config.<same_name>`
    (allowing the str()/.strip()/.lower() normalizations used for two fields)."""
    call = _modernbert_config_call()
    keyword_values = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}

    real_fields = {f.name for f in fields(ProtModernBertMLMConfig)} - NOT_FORWARDED

    missing = sorted(real_fields - keyword_values.keys())
    assert missing == [], f"fields absent from the ModernBertConfig(...) call: {missing}"

    not_forwarded_from_config = sorted(
        f"{name}={ast.unparse(keyword_values[name])}"
        for name in real_fields
        if _forwarded_field_name(keyword_values[name]) != name
    )
    assert not_forwarded_from_config == [], (
        "keywords present but not wired to config.<same_name> "
        f"(hardcoded or renamed source): {not_forwarded_from_config}"
    )


def test_no_unexplained_hardcoded_keywords():
    """Every keyword in the call that is NOT a ProtModernBertMLMConfig field name
    must be on the explicit, commented allowlist above. Anything else is a
    keyword wired to a hardcoded value with no justification on record."""
    call = _modernbert_config_call()
    keyword_names = {kw.arg for kw in call.keywords if kw.arg is not None}
    real_fields = {f.name for f in fields(ProtModernBertMLMConfig)}

    unexplained = sorted(
        keyword_names - real_fields - _ALLOWED_NON_FORWARDED_KEYWORDS.keys()
    )
    assert unexplained == [], f"unexplained non-config keywords: {unexplained}"


# --- Value-level guards ----------------------------------------------------
#
# The AST test above proves every field is wired to `config.<same_name>` (or is
# on the short allowlist). These tests additionally prove the *values* actually
# flow through end to end for a broad sample of fields, using non-default,
# valid values (a different enum member where a field is constrained, rather
# than an invalid string that would just exercise ModernBertConfig's own
# __post_init__ validation instead of the mapping).
#
# Fields are split across several configs because many of them are mutually
# exclusive per ModernBertConfig.__post_init__ / recycling_compatibility.yaml
# (e.g. use_moe and recycling_loop_count > 1 reject each other; use_repo
# requires no GQA; use_diff_attn_v2 is incompatible with use_repo and with
# Canon-B). Every one of the 68 forwarded fields (all ProtModernBertMLMConfig
# fields except num_register_tokens) is exercised by exactly one group below —
# none were skipped as "unsafe to vary".


def test_values_survive_the_mapping_structural_and_attention_group():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        vocab_size=99,
        norm_type="rmsnorm",
        mlp_activation="glu",
        mlp_dropout=0.15,
        mlp_bias=True,
        no_mlp_on_first_layer=False,
        attention_bias=True,
        attention_dropout=0.1,
        classifier_activation="relu",
        use_resid_lambdas=True,
        use_x0_lambdas=True,
        use_qk_norm=True,
        reorder_RoPE_QKNorm=True,
        global_rope_theta=5000.0,
        local_rope_theta=333.0,
        rope_dim_fraction=0.5,
        resid_lambda_init=0.7,
        x0_lambda_init=0.35,
        activation_checkpointing=True,
        activation_checkpointing_mode=" Attn ",  # exercises .strip().lower()
        tie_word_embeddings=False,
        attn_layer_pattern="FS",
        fused_qkv=False,
        fused_up_gate=False,
        wo_init="kaiming",
    )
    mapped = modernbert_config_from(src)
    assert mapped.hidden_size == 64
    assert mapped.intermediate_size == 128
    assert mapped.num_hidden_layers == 4
    assert mapped.num_attention_heads == 4
    assert mapped.num_kv_heads == 2
    assert mapped.vocab_size == 99
    assert mapped.norm_type == "rmsnorm"
    assert mapped.mlp_activation == "glu"
    assert mapped.mlp_dropout == 0.15
    assert mapped.mlp_bias is True
    assert mapped.no_mlp_on_first_layer is False
    assert mapped.attention_bias is True
    assert mapped.attention_dropout == 0.1
    assert mapped.classifier_activation == "relu"
    assert mapped.use_resid_lambdas is True
    assert mapped.use_x0_lambdas is True
    assert mapped.use_qk_norm is True
    assert mapped.reorder_RoPE_QKNorm is True
    assert mapped.global_rope_theta == 5000.0
    assert mapped.local_rope_theta == 333.0
    assert mapped.rope_dim_fraction == 0.5
    assert mapped.resid_lambda_init == 0.7
    assert mapped.x0_lambda_init == 0.35
    assert mapped.activation_checkpointing is True
    assert mapped.activation_checkpointing_mode == "attn"
    assert mapped.tie_word_embeddings is False
    assert mapped.attn_layer_pattern == "FS"
    assert mapped.fused_qkv is False
    assert mapped.fused_up_gate is False
    assert mapped.wo_init == "kaiming"


def test_values_survive_the_mapping_canon_layers_group():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_canon_layers=True,
        canon_layers_mode="ac",
        canon_layers_kernel_size=7,
    )
    mapped = modernbert_config_from(src)
    assert mapped.use_canon_layers is True
    assert mapped.canon_layers_mode == "ac"
    assert mapped.canon_layers_kernel_size == 7


def test_values_survive_the_mapping_repo_and_prores_group():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_repo=True,
        repo_after_n_layers=1,
        use_prores=True,
        prores_T=500,
    )
    mapped = modernbert_config_from(src)
    assert mapped.use_repo is True
    assert mapped.repo_after_n_layers == 1
    assert mapped.use_prores is True
    assert mapped.prores_T == 500


def test_values_survive_the_mapping_mhc_lite_group():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_mhc_lite=True,
        mhc_n_streams=6,
        mhc_triton_fused=True,
        mhc_lite_wrapping_level=" Sublayers ",  # exercises .strip().lower()
    )
    mapped = modernbert_config_from(src)
    assert mapped.use_mhc_lite is True
    assert mapped.mhc_n_streams == 6
    assert mapped.mhc_triton_fused is True
    assert mapped.mhc_lite_wrapping_level == "sublayers"


def test_values_survive_the_mapping_diff_attn_v2_group():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_diff_attn_v2=True,
    )
    mapped = modernbert_config_from(src)
    assert mapped.use_diff_attn_v2 is True


def test_values_survive_the_mapping_paired_head_attention_group():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_paired_head_attention=True,
    )
    mapped = modernbert_config_from(src)
    assert mapped.use_paired_head_attention is True


def test_values_survive_the_mapping_noble_group():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_noble=True,
        noble_rank=10,
        noble_alpha=0.05,
        noble_omega_range=(0.3, 0.9),
        noble_phi_std=0.2,
        noble_half_kaiming=False,
        noble_targets="attn",
    )
    mapped = modernbert_config_from(src)
    assert mapped.use_noble is True
    assert mapped.noble_rank == 10
    assert mapped.noble_alpha == 0.05
    assert tuple(mapped.noble_omega_range) == (0.3, 0.9)
    assert mapped.noble_phi_std == 0.2
    assert mapped.noble_half_kaiming is False
    assert mapped.noble_targets == "attn"


def test_values_survive_the_mapping_moe_group():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        use_moe=True,
        moe_num_experts=6,
        moe_top_k=3,
        moe_use_bias_correction=False,
        moe_aux_loss_coef=0.02,
        moe_z_loss_coef=1e-4,
        moe_routed_scaling_factor=1.5,
        moe_n_group=2,
        moe_topk_group=2,
        moe_bias_update_rate=2e-3,
        moe_leading_dense_layers=1,
        moe_kernel_backend="sonicmoe",
    )
    mapped = modernbert_config_from(src)
    assert mapped.use_moe is True
    assert mapped.moe_num_experts == 6
    assert mapped.moe_top_k == 3
    assert mapped.moe_use_bias_correction is False
    assert mapped.moe_aux_loss_coef == 0.02
    assert mapped.moe_z_loss_coef == 1e-4
    assert mapped.moe_routed_scaling_factor == 1.5
    assert mapped.moe_n_group == 2
    assert mapped.moe_topk_group == 2
    assert mapped.moe_bias_update_rate == 2e-3
    assert mapped.moe_leading_dense_layers == 1
    assert mapped.moe_kernel_backend == "sonicmoe"


def test_values_survive_the_mapping_recycling_group():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
        recycling_loop_count=2,
        recycling_adapter_bias=True,
        recycling_s0_noise_sigma=0.5,
        recycling_use_takase_init=False,
    )
    mapped = modernbert_config_from(src)
    assert mapped.recycling_num_prelude_layers == 1
    assert mapped.recycling_num_coda_layers == 1
    assert mapped.recycling_loop_count == 2
    assert mapped.recycling_adapter_bias is True
    assert mapped.recycling_s0_noise_sigma == 0.5
    assert mapped.recycling_use_takase_init is False


def test_pure_model_uses_the_shared_mapping():
    model = PureProtModernBertMLM(
        ProtModernBertMLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            moe_num_experts=5,
        )
    )
    assert model.config.moe_num_experts == 5
