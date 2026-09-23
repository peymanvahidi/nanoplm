"""Regression test: recycling MFU must charge FLOPs at the ACTUAL loop count.

Under the loop-count curriculum (e.g. ``1_sqrt``), early training steps run the
recurrent block far fewer times than the target ``recycling_loop_count``. MFU was
computed from a constant flops/token baked in at the *target* loop count, which
overstated utilization by ~2.5x early on (e.g. ~80% reported vs ~32% real for a
4x loop count over an 8-layer recurrent block). FLOPs/token must therefore be a
function of the actual per-step loop count, not a startup constant.
"""

import pytest

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.pure_pipeline import (
    _estimate_model_flops_per_token,
    _recycling_flops_components,
    _recycling_flops_per_token,
)

SEQ = 1024


def _cfg(**overrides):
    """Return the inner ModernBertConfig the pipeline feeds the FLOPs estimator.

    The estimator reads derived fields (local_attention, layer_types) that only
    exist on the inner config built during model construction — exactly what the
    pipeline passes via ``orig_model.config``. Small dims keep instantiation fast.
    P=2, C=2, n=8 -> R=4 recurrent layers (same ratio shape as the production
    16/4/4 config: stack scale 2.5 at r=4 vs 1.0 at r=1).
    """
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        vocab_size=32,
        recycling_num_prelude_layers=2,
        recycling_num_coda_layers=2,
        use_huginn_looping=True,
        recycling_loop_count=4,
        recycling_use_takase_init=False,
        wo_init="kaiming",
    )
    kwargs.update(overrides)
    return PureProtModernBertMLM(ProtModernBertMLMConfig(**kwargs)).config


def test_flops_scale_linearly_with_loop_count():
    """Each extra loop adds a constant recurrent-block increment; the tail
    (lm-head + prediction head) does not scale, so the deltas are equal."""
    comp = _recycling_flops_components(_cfg(), SEQ)
    f1 = _recycling_flops_per_token(comp, 1)
    f2 = _recycling_flops_per_token(comp, 2)
    f3 = _recycling_flops_per_token(comp, 3)
    assert f1 < f2 < f3
    assert (f2 - f1) == pytest.approx(f3 - f2, rel=1e-9)


def test_estimate_matches_target_loop_count():
    """The startup estimate must equal flops/token at the configured target r."""
    cfg = _cfg(recycling_loop_count=4)
    comp = _recycling_flops_components(cfg, SEQ)
    assert _estimate_model_flops_per_token(cfg, SEQ) == pytest.approx(
        _recycling_flops_per_token(comp, 4)
    )


def test_target_loop_overstates_flops_when_curriculum_runs_r1():
    """The exact MFU bug: charging the target r=4 while the curriculum runs r=1
    overstates FLOPs (and thus MFU) by >2x for this P/C/R layout."""
    cfg = _cfg(recycling_loop_count=4)
    comp = _recycling_flops_components(cfg, SEQ)
    nominal = _estimate_model_flops_per_token(cfg, SEQ)   # target r=4
    actual_r1 = _recycling_flops_per_token(comp, 1)       # early curriculum: r=1
    # P=4,C=4,R=8,n=16: stack scale = (4+4*8+4)/16 = 2.5 at r=4 vs 1.0 at r=1.
    assert nominal > actual_r1
    assert nominal / actual_r1 > 2.0


def test_no_recycling_collapses_to_dense_cost():
    """loop_count == 1 (recycling off) must equal the dense n_layers cost."""
    cfg = _cfg(use_huginn_looping=False,
               recycling_loop_count=1,
               recycling_num_prelude_layers=0,
               recycling_num_coda_layers=0)
    comp = _recycling_flops_components(cfg, SEQ)
    assert _estimate_model_flops_per_token(cfg, SEQ) == pytest.approx(
        _recycling_flops_per_token(comp, 1)
    )


def test_truncated_bptt_charges_less_backward_than_full():
    """Truncated BPTT (K < r) backprops only the last K recurrent iters, so the
    hardware does LESS backward than full BPTT. Charging full backward (the old
    3x-forward) overcounts FLOPs and inflates MFU. The corrected estimate must be
    strictly smaller than the full-BPTT estimate whenever K < r, and must equal it
    when K >= r."""
    comp = _recycling_flops_components(_cfg(recycling_loop_count=4), SEQ)
    full = _recycling_flops_per_token(comp, 4)            # bptt=None -> full BPTT == 3x fwd
    trunc_k1 = _recycling_flops_per_token(comp, 4, bptt=1)
    trunc_k2 = _recycling_flops_per_token(comp, 4, bptt=2)
    assert trunc_k1 < trunc_k2 < full
    # K >= loop_count recovers full BPTT exactly.
    assert _recycling_flops_per_token(comp, 4, bptt=4) == pytest.approx(full)
    assert _recycling_flops_per_token(comp, 4, bptt=99) == pytest.approx(full)
    # At r == 1 there is nothing to truncate: K=1 equals full.
    assert _recycling_flops_per_token(comp, 1, bptt=1) == pytest.approx(
        _recycling_flops_per_token(comp, 1)
    )


def test_truncated_bptt_matches_fwd_plus_2x_grad_subgraph():
    """Corrected FLOPs == forward(all r iters) + 2 * forward(P + K*R + C tail).

    Verified against the analytic decomposition: forward runs all r recurrent
    iters; backward (~2x forward) is retained only for the differentiated subgraph
    of P + K*R + C layers (prelude/coda + last K recurrent iters)."""
    stack_flops, tail_flops, P, R, C, n_layers = _recycling_flops_components(
        _cfg(recycling_loop_count=4), SEQ
    )

    def fwd(loops):
        return (P + loops * R + C) / n_layers * stack_flops + tail_flops

    comp = (stack_flops, tail_flops, P, R, C, n_layers)
    for r, k in [(4, 1), (4, 2), (3, 1), (2, 1)]:
        expected = fwd(r) + 2.0 * fwd(min(k, r))
        assert _recycling_flops_per_token(comp, r, bptt=k) == pytest.approx(expected)
