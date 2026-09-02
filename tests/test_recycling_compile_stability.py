"""Compile-stability regression test for Huginn-style recycling.

Background
----------
When ``recycling_loop_count`` varies across forwards (e.g. under
``lognormal_poisson`` sampling), torch.compile must NOT re-trace the model
for each distinct r value. The recurrent block has a fixed graph shape;
only the trip count of the outer Python loop varies, and that loop must
live in eager Python (via a ``@torch._dynamo.disable``-decorated driver)
so it never enters a compiled graph.

The compile-cache thrashing this guards against was the root cause of
training stalls observed with ``recycling_loop_count_sampling: lognormal_poisson``.
"""

import pytest
import torch
import torch._dynamo

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM


def _recycling_cfg(tokenizer, **overrides):
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=tokenizer.vocab_size,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
        use_huginn_looping=True,
        recycling_loop_count=4,
        recycling_use_takase_init=False,
        wo_init="kaiming",
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


def _make_counting_backend():
    """Return (backend, counter_list) where counter_list[0] is the compile count."""
    n = [0]

    def backend(gm, example_inputs):
        n[0] += 1
        return gm.forward

    return backend, n


def test_compile_bounded_for_varying_loop_count(tokenizer):
    """Five distinct r values must NOT each trigger a fresh Dynamo trace.

    On the broken code (Python-int loop bounds inside the compiled graph), each
    unique r causes a guard miss and a fresh trace -> 5+ compile events for 5
    distinct r values. With the fix (eager driver, shape-stable inner step),
    the compile count is a small constant independent of r.
    """
    torch.manual_seed(0)
    cfg = _recycling_cfg(tokenizer)
    model = PureProtModernBertMLM(cfg)
    model.eval()  # dropout off; deterministic forward

    backend, n_compiles = _make_counting_backend()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, dynamic=False)

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    # Warm up the compile cache with the first r. Whatever number of
    # graph segments the model decomposes into, they all compile here.
    K_t = torch.tensor(1, dtype=torch.int64)
    with torch.no_grad():
        compiled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            recycling_loop_count=torch.tensor(2, dtype=torch.int64),
            recycling_bptt=K_t,
        )
    warmup_compiles = n_compiles[0]
    assert warmup_compiles > 0, "expected at least one compile on first call"

    # New r values (as 0-d int tensors — the contract for compiled
    # callers, see ModernBertModel._drive_recurrence docstring) must
    # reuse the warm cache. Python ints would re-introduce an int-value
    # guard at the disable boundary and trigger a recompile per r.
    rs_after = (3, 4, 5, 6, 7, 8, 9, 10)
    for r in rs_after:
        with torch.no_grad():
            compiled(
                input_ids=input_ids,
                attention_mask=attention_mask,
                recycling_loop_count=torch.tensor(r, dtype=torch.int64),
                recycling_bptt=K_t,
            )

    recompiles = n_compiles[0] - warmup_compiles
    assert recompiles == 0, (
        f"Expected 0 recompiles across {len(rs_after)} new r values after warmup; "
        f"got {recompiles}. The recurrent loop is being re-traced for each r — "
        f"either the eager-driver disable is bypassed, or r is reaching a "
        f"compiled segment as a Python int (re-introducing a value guard)."
    )


def test_no_layer_idx_cache_thrash_at_production_R(tokenizer):
    """Regression: with R recurrent layers > cache_size_limit, the per-layer
    dispatch in the recurrent step must NOT specialize per-layer-index.

    Production config has R = num_hidden_layers - P - C = 16 - 4 - 4 = 8,
    which equals Dynamo's default cache_size_limit. An earlier version of
    the eager-driver pattern dispatched per-layer through a closure that
    captured ``layer_idx`` as a Python-int kwarg; Dynamo specialized on
    that int once per layer, exhausted the cache, and silently fell back
    to eager — masking what should have been a clean compile.

    Asserting the layer count > cache_size_limit forces the test to catch
    this. With the no-closure design (``_recurrent_step_varlen`` inlines
    the per-layer call inside one frame), the unrolled for-loop produces a
    single graph regardless of R.
    """
    torch.manual_seed(0)
    cfg = ProtModernBertMLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=12,            # P=1 + R=10 + C=1
        num_attention_heads=4,
        vocab_size=tokenizer.vocab_size,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
        use_huginn_looping=True,
        recycling_loop_count=4,
        recycling_use_takase_init=False,
        wo_init="kaiming",
    )
    R = cfg.num_hidden_layers - cfg.recycling_num_prelude_layers - cfg.recycling_num_coda_layers
    assert R > torch._dynamo.config.recompile_limit, (
        f"test precondition: R={R} must exceed cache_size_limit "
        f"({torch._dynamo.config.recompile_limit}) to exercise the bug"
    )

    model = PureProtModernBertMLM(cfg)
    model.eval()
    backend, n_compiles = _make_counting_backend()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, dynamic=False)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    attention_mask = torch.ones_like(input_ids)
    K_t = torch.tensor(1, dtype=torch.int64)

    # Single warmup forward, then a few more with different r.
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for r in (2, 3, 4):
            with torch.no_grad():
                compiled(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    recycling_loop_count=torch.tensor(r, dtype=torch.int64),
                    recycling_bptt=K_t,
                )

    # No Dynamo "hit cache_size_limit" warnings should fire.
    cache_warnings = [
        str(w.message) for w in caught
        if "cache_size_limit" in str(w.message) or "recompile_limit" in str(w.message)
    ]
    assert not cache_warnings, (
        f"Dynamo hit cache_size_limit (eager fallback). Warnings:\n"
        + "\n".join(cache_warnings)
    )
    # And compile count should stay bounded — well under R for the
    # recurrent block (the unrolled loop is one graph, not R graphs).
    assert n_compiles[0] <= R, (
        f"Expected <= R={R} compiles total; got {n_compiles[0]}. "
        f"Indicates per-layer-index specialization is still happening."
    )


def test_drive_recurrence_exists(tokenizer):
    """The fix introduces a Dynamo-disabled method ``_drive_recurrence`` on
    the inner ModernBertModel. This test pins down the API so future
    refactors don't silently revert it.
    """
    cfg = _recycling_cfg(tokenizer)
    inner = PureProtModernBertMLM(cfg).model
    assert hasattr(inner, "_drive_recurrence"), (
        "ModernBertModel must expose _drive_recurrence so the variable-trip "
        "recurrent loop runs in eager Python, outside any torch.compile region."
    )
    # @torch._dynamo.disable marks the function with _torchdynamo_disable.
    fn = inner._drive_recurrence
    underlying = getattr(fn, "__func__", fn)
    assert getattr(underlying, "_torchdynamo_disable", False), (
        "_drive_recurrence must be decorated with @torch._dynamo.disable; "
        "otherwise the variable trip count flows into a compiled graph and "
        "every unique r triggers a recompile."
    )
