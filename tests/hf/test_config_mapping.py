"""One mapping from ProtModernBertMLMConfig to ModernBertConfig, used everywhere."""
from dataclasses import fields

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
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


def test_every_model_config_field_is_forwarded():
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


def test_values_survive_the_mapping():
    src = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_moe=True,
        moe_num_experts=7,
        use_paired_head_attention=True,
        recycling_loop_count=1,
        noble_omega_range=(0.5, 1.5),
    )
    mapped = modernbert_config_from(src)
    assert mapped.moe_num_experts == 7
    assert mapped.use_moe is True
    assert mapped.use_paired_head_attention is True
    assert tuple(mapped.noble_omega_range) == (0.5, 1.5)


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
