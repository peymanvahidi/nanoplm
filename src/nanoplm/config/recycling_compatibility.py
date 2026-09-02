"""Validator for the Huginn-style recycling compatibility matrix.

The matrix lives in recycling_compatibility.yaml; this module loads it
and exposes validate_recycling_compatibility() for use at:
  - ModernBertConfig.__post_init__()  (model_flags subset only)
  - pure_pipeline.py startup           (full check, both subsets)

The validator no-ops when model_config.recycling_loop_count <= 1.

This validator covers Huginn-style recycling ONLY. As future paper recipes
(Think-Deeper, retrofitting, MoR) land, a multi-feature compatibility
framework will be needed; tracked as a separate GitHub issue.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from nanoplm.utils.logger import logger as _logger


_MATRIX_PATH = Path(__file__).parent / "recycling_compatibility.yaml"


@lru_cache(maxsize=1)
def load_compatibility_matrix() -> dict[str, Any]:
    """Load the YAML matrix once and cache it."""
    with open(_MATRIX_PATH) as f:
        return yaml.safe_load(f)


def _check_model_flags(model_config, matrix: dict[str, Any]) -> None:
    model_section = matrix["model_flags"]

    for entry in model_section.get("reject", []):
        flag = entry["flag"]
        if getattr(model_config, flag, False):
            raise ValueError(
                f"recycling: incompatible model flag '{flag}=True'. {entry['reason']}"
            )

    for entry in model_section.get("allow_with_warning", []):
        flag = entry["flag"]
        if not getattr(model_config, flag, False):
            continue
        if flag == "use_repo":
            P = model_config.recycling_num_prelude_layers
            C = model_config.recycling_num_coda_layers
            L = model_config.num_hidden_layers
            R_end = P + (L - P - C) - 1
            if P <= model_config.repo_after_n_layers <= R_end:
                _logger.warning(
                    "recycling: %s — %s", entry["condition"], entry["message"]
                )
        else:
            # Unconditional warn: the flag being set is the whole condition.
            _logger.warning(
                "recycling: %s — %s",
                entry.get("condition", flag),
                entry["message"],
            )


def _check_pipeline_flags(pipeline_config, matrix: dict[str, Any]) -> None:
    pipeline_section = matrix["pipeline_flags"]

    for entry in pipeline_section.get("reject", []):
        flag = entry["flag"]
        value = getattr(pipeline_config, flag, None)
        # Explicit per-flag predicate matches the model-side pattern.
        # The yaml `condition:` string stays as human-readable doc only.
        if flag == "pure_te" and value is True:
            raise ValueError(f"recycling: incompatible pipeline flag '{flag}'. {entry['reason']}")
        if flag == "pure_torch" and value is False:
            raise ValueError(f"recycling: incompatible pipeline flag '{flag}'. {entry['reason']}")

    for entry in pipeline_section.get("allow_with_caveat", []):
        flag = entry["flag"]
        value = getattr(pipeline_config, flag, None)
        if flag == "fp8" and value is True:
            _logger.warning("recycling: %s — %s", flag, entry["message"])
        elif flag == "use_compile_max_autotune" and value is True:
            _logger.warning("recycling: %s — %s", flag, entry["message"])


def validate_recycling_compatibility(
    model_config,
    pipeline_config=None,
) -> None:
    """Raise ValueError on any reject hit; log warnings on any allow_with_warning hit.

    No-ops when model_config.recycling_loop_count <= 1.
    """
    if getattr(model_config, "recycling_loop_count", 1) <= 1:
        return
    matrix = load_compatibility_matrix()
    _check_model_flags(model_config, matrix)
    if pipeline_config is not None:
        _check_pipeline_flags(pipeline_config, matrix)
