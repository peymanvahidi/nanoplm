"""Compile-stability and FSDP2 invariants for Loopie stochastic depth.

Two constraints pull against each other here. The trip count must stay out of
every traced graph, or Dynamo specializes on it and each distinct R costs a
fresh trace (the failure ``ModernBertModel._drive_recurrence`` documents). And
the repeats must stay inside ``ModernBertEncoderLayer.forward``, or FSDP2 —
which shards per encoder layer — all-gathers once per repeat instead of once
per layer. Both live in this file because a fix for either can break the other.

``dynamic=False`` matches production: ``use_packing`` + ``use_static_inp_size``
resolve ``compile_dynamic`` to False in ``pure_pipeline``.
"""

import warnings

import torch
import torch._dynamo
from torch._dynamo.utils import counters

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

# 10 stored layers is the production loop-imp geometry and exceeds Dynamo's
# recompile_limit, so per-layer frame specialization would show up as an eager
# fallback rather than passing quietly.
_NUM_LAYERS = 10


def _cfg(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        num_kv_heads=4,
        attn_layer_pattern="F",
        wo_init="kaiming",
        no_mlp_on_first_layer=False,
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


def _loopie_model(**overrides):
    return PureProtModernBertMLM(
        _cfg(use_loopie=True, loopie_loop_count=2, **overrides)
    ).eval()


def _batch(cfg):
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    return input_ids, torch.ones_like(input_ids)


def test_compile_bounded_for_varying_loopie_loop_count(make_counting_backend):
    """Distinct per-micro-step R values must not each trigger a fresh trace."""
    torch.manual_seed(0)
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

    backend, n_compiles = make_counting_backend()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, dynamic=False)

    with torch.no_grad():
        compiled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loop_count=torch.tensor(2, dtype=torch.int64),
        )
    warmup_compiles = n_compiles[0]
    assert warmup_compiles > 0, "expected at least one compile on first call"

    rs_after = (1, 3, 1, 3, 2, 4, 5, 6)
    for r in rs_after:
        with torch.no_grad():
            compiled(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loop_count=torch.tensor(r, dtype=torch.int64),
            )

    recompiles = n_compiles[0] - warmup_compiles
    assert recompiles == 0, (
        f"Expected 0 recompiles across {len(rs_after)} R values after warmup; got "
        f"{recompiles}. Either the _forward_repeats disable is bypassed, or R is "
        f"reaching a compiled segment as a Python int and installing a value guard."
    )


def test_no_per_layer_cache_thrash_at_production_depth(make_counting_backend):
    """The per-layer disable boundaries must not exhaust the Dynamo cache.

    Structurally identical layers share a cache entry, so the graph count tracks
    the number of distinct layer shapes, not the layer count. If it tracked the
    layer count instead, Dynamo would hit recompile_limit, fall back to eager,
    and make the zero-recompile assertion above pass vacuously.
    """
    torch.manual_seed(0)
    assert _NUM_LAYERS > torch._dynamo.config.recompile_limit, (
        f"test precondition: {_NUM_LAYERS} layers must exceed recompile_limit "
        f"({torch._dynamo.config.recompile_limit}) to exercise the failure"
    )
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

    backend, n_compiles = make_counting_backend()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, dynamic=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for r in (2, 3, 4):
            with torch.no_grad():
                compiled(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loop_count=torch.tensor(r, dtype=torch.int64),
                )

    cache_warnings = [
        str(w.message) for w in caught
        if "cache_size_limit" in str(w.message) or "recompile_limit" in str(w.message)
    ]
    assert not cache_warnings, (
        "Dynamo hit its recompile limit (eager fallback). Warnings:\n"
        + "\n".join(cache_warnings)
    )
    assert n_compiles[0] <= 4 * _NUM_LAYERS, (
        f"Expected the graph count to track distinct layer shapes; got "
        f"{n_compiles[0]} for {_NUM_LAYERS} layers."
    )


def test_fixed_path_compiles_exactly_as_an_unlooped_model(make_counting_backend):
    """Without the kwarg, loopie must trace to the graph it traced before.

    ``loop_count is None`` is a constant test, so Dynamo folds the branch and
    never reaches ``_forward_repeats``. Two consequences are asserted: no graph
    break names that method, and the graph count equals an unlooped model's.
    The count is not 1 — this model breaks on data-dependent branching and on
    ``run_flush`` for reasons that predate looping — so the unlooped model is
    the reference rather than a hard-coded number.
    """
    torch.manual_seed(0)
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

    backend, n_compiles = make_counting_backend()
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(model, backend=backend, dynamic=False)
    with torch.no_grad():
        compiled(input_ids=input_ids, attention_mask=attention_mask)
    looped_compiles = n_compiles[0]

    breaks = [reason for reason in counters["graph_break"] if "_forward_repeats" in reason]
    assert not breaks, (
        "The fixed path traced the Dynamo-disabled driver, so `loop_count is None` "
        f"was not constant-folded:\n{breaks}"
    )

    with torch.no_grad():
        compiled(input_ids=input_ids, attention_mask=attention_mask)
    assert n_compiles[0] == looped_compiles, "fixed path recompiled on a repeat call"

    plain = PureProtModernBertMLM(_cfg()).eval()
    plain_backend, n_plain = make_counting_backend()
    torch._dynamo.reset()
    compiled_plain = torch.compile(plain, backend=plain_backend, dynamic=False)
    with torch.no_grad():
        compiled_plain(input_ids=input_ids, attention_mask=attention_mask)

    assert looped_compiles == n_plain[0], (
        f"Fixed-depth loopie compiled into {looped_compiles} graphs vs "
        f"{n_plain[0]} for the same model without looping; the new branch added "
        f"a graph break."
    )


def test_one_layer_call_per_step_regardless_of_depth():
    """FSDP2 all-gathers on __call__, so every repeat must stay inside forward.

    Counting forward pre-hooks against ``_forward_once`` executions is exactly
    the invariant fully_shard keys on, and it needs no GPUs to check.
    """
    torch.manual_seed(0)
    model = _loopie_model()
    input_ids, attention_mask = _batch(model.model_config)

    calls: dict[int, int] = {}
    bodies: dict[int, int] = {}
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(
            lambda module, args, _i=i: calls.__setitem__(_i, calls.get(_i, 0) + 1)
        )
        original = layer._forward_once

        def counted(*args, _original=original, _i=i, **kwargs):
            bodies[_i] = bodies.get(_i, 0) + 1
            return _original(*args, **kwargs)

        layer._forward_once = counted

    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loop_count=torch.tensor(4, dtype=torch.int64),
        )

    assert calls == {i: 1 for i in range(_NUM_LAYERS)}, (
        f"Each stored layer must be entered once per step (one all-gather, one "
        f"reshard); got {calls}."
    )
    assert bodies == {i: 4 for i in range(_NUM_LAYERS)}, (
        f"Each stored layer must run its body loop_count times; got {bodies}."
    )
