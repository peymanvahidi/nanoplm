"""Tests for the Huginn-style recycling compatibility validator.

The validator is the single source of truth for which existing nanoPLM flags
can be combined with recycling. It's called from ModernBertConfig.__post_init__
(model_flags only) and from pure_pipeline.py at startup (full check).
"""

import logging
import pytest

from nanoplm.config.recycling_compatibility import (
    validate_recycling_compatibility,
    load_compatibility_matrix,
)


def _model_cfg(**kwargs):
    """Minimal stand-in for ModernBertConfig that exposes the flags the validator inspects."""
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=32,
        recycling_loop_count=2,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
    )
    base.update(kwargs)
    return ModernBertConfig(**base)


def test_matrix_loads_with_expected_top_level_keys():
    matrix = load_compatibility_matrix()
    assert matrix["schema_version"] == 1
    assert matrix["feature"] == "huggin_style_recycling"
    assert "model_flags" in matrix
    assert "pipeline_flags" in matrix


def test_validator_no_op_when_recycling_off():
    cfg = _model_cfg(recycling_loop_count=1, recycling_num_prelude_layers=0, recycling_num_coda_layers=0)
    validate_recycling_compatibility(cfg)


def test_reject_use_moe():
    with pytest.raises(ValueError, match="use_moe"):
        _model_cfg(use_moe=True)


def test_reject_use_x0_lambdas():
    with pytest.raises(ValueError, match="use_x0_lambdas"):
        _model_cfg(use_x0_lambdas=True)


def test_reject_use_resid_lambdas():
    with pytest.raises(ValueError, match="use_resid_lambdas"):
        _model_cfg(use_resid_lambdas=True)


def test_reject_use_mhc_lite():
    with pytest.raises(ValueError, match="use_mhc_lite"):
        _model_cfg(use_mhc_lite=True)


def test_allow_use_qk_norm_silently():
    cfg = _model_cfg(use_qk_norm=True)
    validate_recycling_compatibility(cfg)


def test_repo_inside_recurrent_block_emits_warning(caplog):
    # repo_after_n_layers=1 falls in [P=1, P+R-1=2] for a 4-layer model
    # with (P, R, C) = (1, 2, 1), so RePO is inside the recurrent block.
    cfg = _model_cfg(use_repo=True, repo_after_n_layers=1)
    with caplog.at_level(logging.WARNING, logger="nanoplm"):
        validate_recycling_compatibility(cfg)
    assert any("RePO" in record.message for record in caplog.records)


def test_repo_outside_recurrent_block_no_warning(caplog):
    cfg = _model_cfg(use_repo=True, repo_after_n_layers=3)
    with caplog.at_level(logging.WARNING, logger="nanoplm"):
        validate_recycling_compatibility(cfg)
    assert not any("RePO" in record.message for record in caplog.records)


def test_pipeline_flag_rejects_pure_te():
    cfg = _model_cfg()
    from types import SimpleNamespace
    pipeline_cfg = SimpleNamespace(pure_te=True, pure_torch=True, fp8=False, use_compile_max_autotune=False)
    with pytest.raises(ValueError, match="te_pipeline"):
        validate_recycling_compatibility(cfg, pipeline_cfg)


def test_pipeline_flag_rejects_hf_path():
    cfg = _model_cfg()
    from types import SimpleNamespace
    pipeline_cfg = SimpleNamespace(pure_te=False, pure_torch=False, fp8=False, use_compile_max_autotune=False)
    with pytest.raises(ValueError, match="HF path"):
        validate_recycling_compatibility(cfg, pipeline_cfg)


def test_pipeline_flag_warns_on_fp8(caplog):
    cfg = _model_cfg()
    from types import SimpleNamespace
    pipeline_cfg = SimpleNamespace(pure_te=False, pure_torch=True, fp8=True, use_compile_max_autotune=False)
    with caplog.at_level(logging.WARNING, logger="nanoplm"):
        validate_recycling_compatibility(cfg, pipeline_cfg)
    assert any("fp8" in record.message for record in caplog.records)
