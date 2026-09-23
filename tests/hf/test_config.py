import json
from dataclasses import fields

import pytest

from nanoplm.hf import NanoPLMConfig
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig


def test_model_type_is_stable():
    # Baked into every published config.json; changing it breaks every loader.
    assert NanoPLMConfig.model_type == "nanoplm"


def test_constructible_with_no_arguments():
    # transformers instantiates configs bare (AutoConfig.for_model).
    cfg = NanoPLMConfig()
    assert cfg.hidden_size > 0
    assert cfg.num_hidden_layers > 0


def test_carries_every_model_config_field():
    cfg = NanoPLMConfig()
    missing = [f.name for f in fields(ProtModernBertMLMConfig) if not hasattr(cfg, f.name)]
    assert missing == []


def test_json_roundtrip_restores_omega_range_as_tuple():
    # JSON has no tuples; noble_omega_range must come back as a tuple or the
    # NOBLE init path receives a list and silently misbehaves.
    cfg = NanoPLMConfig(use_noble=True, noble_omega_range=(0.5, 1.5))
    restored = NanoPLMConfig.from_dict(json.loads(cfg.to_json_string()))
    assert isinstance(restored.noble_omega_range, tuple)
    assert restored.noble_omega_range == (0.5, 1.5)


def test_frozen_runtime_state_defaults_and_roundtrips():
    cfg = NanoPLMConfig(use_repo=True, repo_active=True, prores_step=1234)
    restored = NanoPLMConfig.from_dict(json.loads(cfg.to_json_string()))
    assert restored.repo_active is True
    assert restored.prores_step == 1234
    assert NanoPLMConfig().repo_active is False


def test_to_model_config_returns_a_dataclass():
    cfg = NanoPLMConfig(hidden_size=64, moe_num_experts=9)
    model_cfg = cfg.to_model_config()
    assert isinstance(model_cfg, ProtModernBertMLMConfig)
    assert model_cfg.hidden_size == 64
    assert model_cfg.moe_num_experts == 9


def test_from_model_config_preserves_values():
    src = ProtModernBertMLMConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, use_moe=True, moe_num_experts=3,
    )
    cfg = NanoPLMConfig.from_model_config(src, repo_active=True, source_global_step=500)
    assert cfg.moe_num_experts == 3
    assert cfg.repo_active is True
    assert cfg.source_global_step == 500
