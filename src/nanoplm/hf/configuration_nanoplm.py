"""Hugging Face config for nanoPLM models.

Fields are derived from ProtModernBertMLMConfig at import time rather than
retyped, so a new model flag needs no edit here. The four fields that have no
dataclass default get explicit ones because transformers instantiates configs
with no arguments.
"""
from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

from transformers import PretrainedConfig

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig

# ProtModernBertMLMConfig's required fields, defaulted to the stock
# pretrain.yaml template values so NanoPLMConfig() works bare.
_REQUIRED_DEFAULTS: dict[str, Any] = {
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
}

# Fields that must be tuples in Python but are lists in JSON.
_TUPLE_FIELDS = ("noble_omega_range",)

# Runtime state resolved at export time; see the spec's "frozen runtime state".
_EXPORT_EXTRAS: dict[str, Any] = {
    "repo_active": False,
    "prores_step": 0,
    "nanoplm_version": "",
    "source_global_step": 0,
}


def _model_field_defaults() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(ProtModernBertMLMConfig):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            out[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            out[f.name] = _REQUIRED_DEFAULTS[f.name]
    return out


class NanoPLMConfig(PretrainedConfig):
    model_type = "nanoplm"

    def __init__(self, **kwargs: Any):
        defaults = _model_field_defaults()
        defaults.update(_EXPORT_EXTRAS)
        for name, default in defaults.items():
            setattr(self, name, kwargs.pop(name, default))
        for name in _TUPLE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, list):
                setattr(self, name, tuple(value))
        super().__init__(**kwargs)

    def to_model_config(self) -> ProtModernBertMLMConfig:
        return ProtModernBertMLMConfig(
            **{f.name: getattr(self, f.name) for f in fields(ProtModernBertMLMConfig)}
        )

    @classmethod
    def from_model_config(
        cls, config: ProtModernBertMLMConfig, **extra: Any
    ) -> "NanoPLMConfig":
        payload = {f.name: getattr(config, f.name) for f in fields(ProtModernBertMLMConfig)}
        payload.update(extra)
        return cls(**payload)
