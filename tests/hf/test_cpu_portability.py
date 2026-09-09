"""The exported model must run with no Triton, no TE, no flash-attn."""
import ast
from pathlib import Path

import pytest
import torch

from nanoplm.pretraining.models.modern_bert import moe_triton_ops
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _tiny(**overrides):
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        vocab_size=32,
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_moe_model_runs_without_triton(device):
    model = PureProtModernBertMLM(
        _tiny(use_moe=True, moe_num_experts=4, moe_top_k=2)
    ).eval().to(device)
    ids = torch.tensor([[4, 5, 6, 7, 1, 0]], device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    assert out.logits.shape == (1, 6, 32)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_grouped_gemm_dispatches_on_mps():
    import nanoplm.pretraining.models.modern_bert.moe_grouped_gemm_ops  # noqa: F401

    a = torch.randn(8, 16)
    b = torch.randn(2, 16, 32)
    counts = torch.tensor([4, 4], dtype=torch.int64)
    expected = torch.ops.nanoplm_moe.grouped_gemm(a, b, counts)
    got = torch.ops.nanoplm_moe.grouped_gemm(
        a.to("mps"), b.to("mps"), counts.to("mps")
    )
    assert got.shape == expected.shape
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_scatter_dispatch_matches_cpu_on_mps():
    import nanoplm.pretraining.models.modern_bert.moe_triton_ops  # noqa: F401

    torch.manual_seed(0)
    T, H = 6, 16
    top_k = 2
    x = torch.randn(T, H)
    # token_idx: for each of the T*top_k dispatch slots, which token feeds it.
    # Mirrors MoELayer.forward's `token_idx = (sorted_idx // top_k)` — an
    # int32 tensor of length T*top_k with values in [0, T).
    token_idx = torch.arange(T, dtype=torch.int32).repeat_interleave(top_k)
    perm = torch.randperm(T * top_k)
    token_idx = token_idx[perm]

    expected = torch.ops.nanoplm_moe.scatter_dispatch(x, token_idx)
    got = torch.ops.nanoplm_moe.scatter_dispatch(x.to("mps"), token_idx.to("mps"))
    assert got.shape == expected.shape
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_gather_combine_matches_cpu_on_mps():
    from nanoplm.pretraining.models.modern_bert.moe_triton_ops import (
        build_inverse_map,
    )

    torch.manual_seed(0)
    T, C = 6, 16
    top_k = 2
    M = T * top_k

    # Build a real inv_map the way MoELayer.forward does: token_idx/slot_idx
    # cover every (token, slot) pair exactly once, in a permuted order, and
    # build_inverse_map produces the actual int32 index dtype this op sees
    # in production — the dtype the finding is specifically about.
    tok_grid = torch.arange(T, dtype=torch.int32).repeat_interleave(top_k)
    slot_grid = torch.arange(top_k, dtype=torch.int32).repeat(T)
    perm = torch.randperm(M)
    token_idx = tok_grid[perm]
    slot_idx = slot_grid[perm]
    inv_map = build_inverse_map(token_idx, slot_idx, T, top_k)
    assert inv_map.dtype == torch.int32

    expert_out_sorted = torch.randn(M, C)
    weights = torch.rand(T, top_k)
    scale = 0.7

    expected = torch.ops.nanoplm_moe.gather_combine(
        expert_out_sorted, inv_map, weights, scale,
    )
    got = torch.ops.nanoplm_moe.gather_combine(
        expert_out_sorted.to("mps"), inv_map.to("mps"), weights.to("mps"), scale,
    )
    assert got.shape == expected.shape
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_moe_model_logits_match_between_cpu_and_mps():
    """End-to-end parity check covering scatter_dispatch, grouped_gemm, and
    gather_combine together: same weights, same inputs, CPU vs MPS forward.
    """
    torch.manual_seed(0)
    model = PureProtModernBertMLM(
        _tiny(use_moe=True, moe_num_experts=4, moe_top_k=2)
    ).eval()
    ids = torch.tensor([[4, 5, 6, 7, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])

    with torch.no_grad():
        cpu_logits = model(input_ids=ids, attention_mask=mask).logits.clone()

    model = model.to("mps")
    with torch.no_grad():
        mps_logits = model(
            input_ids=ids.to("mps"), attention_mask=mask.to("mps"),
        ).logits

    torch.testing.assert_close(mps_logits.cpu(), cpu_logits, rtol=1e-3, atol=1e-3)


# --- AST-level structural guard --------------------------------------------
#
# `test_moe_model_runs_without_triton[cpu]` above only fails on a machine that
# actually lacks Triton installed; re-add a module-scope `import triton` to
# moe_triton_ops.py on a Linux/Hopper box and every test here still passes.
# CI (`.github/workflows/docs.yml`) runs only `tests/docs_schema`, so it
# catches nothing either. The invariant the module split exists to establish
# — that importing `moe_triton_ops` never requires Triton, because the only
# Triton-dependent code paths (the CUDA launch functions) import it lazily
# inside a function body via `from . import moe_triton_kernels as k` — is
# therefore unguarded exactly where it would be broken. Parse the source
# instead of importing it, mirroring `test_config_mapping.py`'s
# `_modernbert_config_call`/AST idiom (itself following
# `tools/schema_gen/dataclass_ast.py` + `tests/docs_schema/test_parity.py`).


def test_moe_triton_ops_has_no_module_scope_triton_import():
    """A guarded import nested inside a function or an `if` is fine and must
    NOT be flagged — only `tree.body` (module scope) is checked."""
    source = Path(moe_triton_ops.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    offending = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(alias.name.split(".")[0] == "triton" for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.split(".")[0] == "triton"
        )
    ]
    assert offending == [], (
        "module-scope `import triton` in moe_triton_ops.py reintroduces the "
        "hard Triton dependency the CPU/MPS portability task removed"
    )
