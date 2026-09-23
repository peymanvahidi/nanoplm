"""Tests for the Huginn-style recycling compatibility validator.

The validator is the single source of truth for which existing nanoPLM flags
can be combined with recycling. It's called from ModernBertConfig.__post_init__
(model_flags only) and from pure_pipeline.py at startup (full check).
"""

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
        use_huginn_looping=True,
        recycling_loop_count=2,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
    )
    base.update(kwargs)
    return ModernBertConfig(**base)


def _pipeline_model_cfg(**kwargs):
    """SimpleNamespace stand-in for pipeline tests.

    Bypasses ModernBertConfig so pipeline-flag tests pass before Batch B
    adds the recycling_* fields to ModernBertConfig.
    """
    from types import SimpleNamespace
    base = dict(
        use_huginn_looping=True,
        recycling_loop_count=2,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
        num_hidden_layers=4,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_matrix_loads_with_expected_top_level_keys():
    matrix = load_compatibility_matrix()
    assert matrix["schema_version"] == 1
    assert matrix["feature"] == "huggin_style_recycling"
    assert "model_flags" in matrix
    assert "pipeline_flags" in matrix


def test_validator_no_op_when_recycling_off():
    cfg = _model_cfg(use_huginn_looping=False, recycling_loop_count=1,
                     recycling_num_prelude_layers=0, recycling_num_coda_layers=0)
    validate_recycling_compatibility(cfg)


def test_warn_use_moe(monkeypatch):
    """MoE is allowed under recycling now that per-layer aux/count state
    accumulates across iterations instead of being overwritten; the matrix
    downgrades it from reject to allow_with_warning."""
    warnings_seen = []
    from nanoplm.utils.logger import logger as _nplm_logger
    monkeypatch.setattr(_nplm_logger, "warning",
                        lambda msg, *a, **k: warnings_seen.append(msg % a if a else msg))
    cfg = _model_cfg(
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
        mlp_activation="swiglu",
    )
    validate_recycling_compatibility(cfg)
    assert any("MoE" in w for w in warnings_seen), warnings_seen


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


def test_repo_inside_recurrent_block_emits_warning(monkeypatch):
    # repo_after_n_layers=1 falls in [P=1, P+R-1=2] for a 4-layer model
    # with (P, R, C) = (1, 2, 1), so RePO is inside the recurrent block.
    cfg = _model_cfg(use_repo=True, repo_after_n_layers=1)
    # Spy on the nanoplm logger (caplog doesn't work because propagate=False)
    warnings_seen = []
    from nanoplm.utils.logger import logger as _nplm_logger
    monkeypatch.setattr(_nplm_logger, "warning",
                        lambda msg, *a, **k: warnings_seen.append(msg % a if a else msg))
    validate_recycling_compatibility(cfg)
    assert any("RePO" in w for w in warnings_seen)


def test_repo_outside_recurrent_block_no_warning(monkeypatch):
    cfg = _model_cfg(use_repo=True, repo_after_n_layers=3)
    # Spy on the nanoplm logger (caplog doesn't work because propagate=False)
    warnings_seen = []
    from nanoplm.utils.logger import logger as _nplm_logger
    monkeypatch.setattr(_nplm_logger, "warning",
                        lambda msg, *a, **k: warnings_seen.append(msg % a if a else msg))
    validate_recycling_compatibility(cfg)
    assert not any("RePO" in w for w in warnings_seen)


def test_pipeline_flag_rejects_hf_path():
    cfg = _pipeline_model_cfg()
    from types import SimpleNamespace
    pipeline_cfg = SimpleNamespace(pure_torch=False, fp8=False, use_compile_max_autotune=False)
    with pytest.raises(ValueError, match="HF path"):
        validate_recycling_compatibility(cfg, pipeline_cfg)


def test_pipeline_flag_warns_on_fp8(monkeypatch):
    cfg = _pipeline_model_cfg()
    from types import SimpleNamespace
    pipeline_cfg = SimpleNamespace(pure_torch=True, fp8=True, use_compile_max_autotune=False)
    # Spy on the nanoplm logger (caplog doesn't work because propagate=False)
    warnings_seen = []
    from nanoplm.utils.logger import logger as _nplm_logger
    monkeypatch.setattr(_nplm_logger, "warning",
                        lambda msg, *a, **k: warnings_seen.append(msg % a if a else msg))
    validate_recycling_compatibility(cfg, pipeline_cfg)
    assert any("fp8" in w for w in warnings_seen)


# ---------------------------------------------------------------------------
# Explicit looping switches: use_huginn_looping / use_loopie
# ---------------------------------------------------------------------------

def test_huginn_flag_requires_loop_count():
    with pytest.raises(ValueError, match="use_huginn_looping"):
        _model_cfg(use_huginn_looping=True, recycling_loop_count=1,
                   recycling_num_prelude_layers=0, recycling_num_coda_layers=0)


def test_loop_count_requires_huginn_flag():
    with pytest.raises(ValueError, match="use_huginn_looping"):
        _model_cfg(use_huginn_looping=False)


def test_loopie_and_huginn_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _model_cfg(use_loopie=True)


def test_loopie_loop_count_must_be_at_least_two():
    with pytest.raises(ValueError, match="loopie_loop_count"):
        _model_cfg(use_huginn_looping=False, recycling_loop_count=1,
                   recycling_num_prelude_layers=0, recycling_num_coda_layers=0,
                   use_loopie=True, loopie_loop_count=1)


def test_loopie_loop_count_without_flag_is_rejected():
    with pytest.raises(ValueError, match="no effect"):
        _model_cfg(use_huginn_looping=False, recycling_loop_count=1,
                   recycling_num_prelude_layers=0, recycling_num_coda_layers=0,
                   loopie_loop_count=3)


@pytest.mark.parametrize("flag", ["use_resid_lambdas", "use_x0_lambdas", "use_mhc_lite"])
def test_loopie_rejects_per_layer_state_flags(flag):
    with pytest.raises(ValueError, match=flag):
        _model_cfg(use_huginn_looping=False, recycling_loop_count=1,
                   recycling_num_prelude_layers=0, recycling_num_coda_layers=0,
                   use_loopie=True, **{flag: True})


def test_loopie_valid_config_builds():
    cfg = _model_cfg(use_huginn_looping=False, recycling_loop_count=1,
                     recycling_num_prelude_layers=0, recycling_num_coda_layers=0,
                     use_loopie=True, loopie_loop_count=2)
    assert cfg.use_loopie and cfg.loopie_loop_count == 2


def _prot_cfg(**kwargs):
    from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
    base = dict(vocab_size=32, hidden_size=64, intermediate_size=64,
                num_hidden_layers=4, num_attention_heads=4, num_kv_heads=4)
    base.update(kwargs)
    return ProtModernBertMLMConfig(**base)


@pytest.mark.parametrize("kwargs", [
    dict(use_huginn_looping=True, recycling_loop_count=2,
         recycling_num_prelude_layers=1, recycling_num_coda_layers=1),
    dict(use_loopie=True, loopie_loop_count=2),
])
def test_hf_path_rejects_looping(kwargs):
    from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM
    with pytest.raises(ValueError, match="pure-torch"):
        ProtModernBertMLM(_prot_cfg(**kwargs))


def test_eval_checkpoint_backfills_huginn_flag():
    """Checkpoints written before use_huginn_looping existed encode recycling
    only via recycling_loop_count > 1; the loader must infer the flag."""
    from nanoplm.eval.checkpoint import _model_config_from_dict
    cfg = _model_config_from_dict(dict(
        vocab_size=32, hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_kv_heads=4,
        recycling_loop_count=2, recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
    ))
    assert cfg.use_huginn_looping is True
    assert _model_config_from_dict(dict(
        vocab_size=32, hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_kv_heads=4,
    )).use_huginn_looping is False
