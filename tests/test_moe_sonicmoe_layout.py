"""SonicMoE routed-expert glue: weight layout + top-k contract.

Regression tests for two defects in ``moe_sonicmoe.py``:

D1  SwiGLU layout. nanoplm's routed-expert ``Wi`` stores its output columns
    **interleaved, gate first** (``moe.WI_GATED_LAYOUT_VERSION == 2``: column
    ``2j`` is gate unit ``j``, ``2j+1`` is up unit ``j``), which is exactly the
    split QuACK's gated GEMM applies. Historically ``Wi`` was block-concat
    ``[up | gate]`` (version 1) while the glue handed QuACK the tensor
    unchanged, so the sonicmoe backend computed a *different function* from the
    same weights (relative L2 error ~= sqrt(2) vs the reference, i.e.
    uncorrelated) and disagreed with the cutlass backend. The first repair
    gathered ``Wi``'s columns on every forward; the packing convention was then
    changed so no reorder is needed at all. Both the pairing and the
    no-reorder property are pinned below.

D7  Fixed top-k. The glue synthesised ``token_indices`` and went through
    SonicMoE's *general* routing entry point, which hard-codes variable-K
    routing; both combine kernels then specialise on ``MAX_K = num_experts``
    instead of ``top_k``.

``srelu`` is not gated and is unaffected by D1; it is checked as a
non-regression guard.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

T, H, I, E, TOP_K = 1024, 256, 128, 8, 2

# A kernel computing the same function in bf16 lands around 5e-3; one computing
# a different gate/up pairing lands at ~1.41 (= sqrt(2)). The gate is not delicate.
TOL = 3e-2


def _sonicmoe():
    pytest.importorskip("sonicmoe")
    from nanoplm.pretraining.models.modern_bert import moe_sonicmoe

    return moe_sonicmoe


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    g, r = got.detach().float(), ref.detach().float()
    return float((g - r).norm() / r.norm().clamp_min(1e-12))


def _problem(srelu: bool, seed: int = 0, device: str = "cuda"):
    """bf16 tensors + realistic top-k routing (distinct experts per token)."""
    g = torch.Generator(device=device).manual_seed(seed)
    i_out = I if srelu else 2 * I

    x = torch.randn(T, H, generator=g, device=device, dtype=torch.float32)
    x = x.to(torch.bfloat16).requires_grad_(True)

    bi, bo = (1.0 / H) ** 0.5, (1.0 / I) ** 0.5
    Wi = torch.empty(E, H, i_out, device=device, dtype=torch.float32)
    Wi.uniform_(-bi, bi, generator=g)
    Wi = Wi.to(torch.bfloat16).requires_grad_(True)
    Wo = torch.empty(E, I, H, device=device, dtype=torch.float32)
    Wo.uniform_(-bo, bo, generator=g)
    Wo = Wo.to(torch.bfloat16).requires_grad_(True)

    logits = torch.randn(T, E, generator=g, device=device, dtype=torch.float32)
    scores = torch.sigmoid(logits)
    weights, indices = torch.topk(scores, TOP_K, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).detach()

    grad_out = torch.randn(T, H, generator=g, device=device,
                           dtype=torch.float32).to(torch.bfloat16)
    return x, Wi, Wo, weights, indices, grad_out


def _activate(h: torch.Tensor, srelu: bool, pairing: str) -> torch.Tensor:
    if srelu:
        r = F.relu(h)
        return r * r
    if pairing == "nanoplm":          # layout v2: interleaved gate-first
        gate, up = h[..., 0::2], h[..., 1::2]
    elif pairing == "legacy_concat":  # layout v1: block-concat [up|gate]
        up, gate = h.chunk(2, dim=-1)
    else:  # pragma: no cover
        raise ValueError(pairing)
    return F.silu(gate) * up


def _reference(x, Wi, Wo, weights, indices, srelu, pairing="nanoplm"):
    """Obviously-correct per-expert loop, nanoplm's activation convention."""
    flat_expert = indices.reshape(-1)
    tok = torch.arange(T, device=x.device).repeat_interleave(TOP_K)
    out = torch.zeros(T, H, device=x.device, dtype=torch.bfloat16)
    for e in range(E):
        sel = (flat_expert == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        rows = tok.index_select(0, sel)
        y = _activate(x.index_select(0, rows) @ Wi[e], srelu, pairing) @ Wo[e]
        y = y * weights.reshape(-1).index_select(0, sel).unsqueeze(-1).to(y.dtype)
        out = out.index_add(0, rows, y)
    return out


def _run(srelu: bool, seed: int = 0):
    mod = _sonicmoe()
    p = _problem(srelu, seed)
    x, Wi, Wo, weights, indices, grad_out = p
    out = mod.sonicmoe_routed_experts(x, indices, weights, Wi, Wo, E, TOP_K,
                                      srelu, 1.0)
    grads = torch.autograd.grad(out, [x, Wi, Wo], grad_out)
    return p, out, grads


# ── D1: SwiGLU weight layout ─────────────────────────────────────────────────


def test_sonicmoe_swiglu_matches_nanoplm_reference():
    """fwd + dx/dWi/dWo must match nanoplm's interleaved gate-first convention."""
    (x, Wi, Wo, weights, indices, grad_out), out, grads = _run(srelu=False)

    ref = _reference(x, Wi, Wo, weights, indices, srelu=False)
    ref_grads = torch.autograd.grad(ref, [x, Wi, Wo], grad_out)

    assert _rel_l2(out, ref) < TOL, (
        f"sonicmoe swiglu forward does not implement nanoplm's gate/up pairing "
        f"(rel_l2={_rel_l2(out, ref):.4g})"
    )
    for name, got, exp in zip(("dx", "dWi", "dWo"), grads, ref_grads):
        assert _rel_l2(got, exp) < TOL, f"{name} rel_l2={_rel_l2(got, exp):.4g}"


def test_sonicmoe_swiglu_is_not_the_legacy_concat_pairing():
    """The layout-v1 block-concat pairing must not be what the kernel computes.

    Anti-regression for a *partial* revert: if either the packing convention in
    ``moe.py`` or the glue in ``moe_sonicmoe.py`` moved back to ``[up | gate]``
    without the other, this is the pairing the kernel would land on.
    """
    (x, Wi, Wo, weights, indices, _grad_out), out, _ = _run(srelu=False)

    nanoplm_ref = _reference(x, Wi, Wo, weights, indices, srelu=False,
                             pairing="nanoplm")
    legacy_ref = _reference(x, Wi, Wo, weights, indices, srelu=False,
                            pairing="legacy_concat")
    err_nanoplm = _rel_l2(out, nanoplm_ref)
    err_legacy = _rel_l2(out, legacy_ref)
    assert err_nanoplm < err_legacy, (
        f"sonicmoe matches the legacy [up|gate] gate/up pairing "
        f"(rel_l2={err_legacy:.4g}) better than nanoplm's current one "
        f"(rel_l2={err_nanoplm:.4g}) -- the D1 layout bug is back"
    )
    assert err_legacy > 0.5


def test_sonicmoe_srelu_matches_reference():
    """relu_sq is ungated: unaffected by the layout fix, must still be exact."""
    (x, Wi, Wo, weights, indices, grad_out), out, grads = _run(srelu=True)

    ref = _reference(x, Wi, Wo, weights, indices, srelu=True)
    ref_grads = torch.autograd.grad(ref, [x, Wi, Wo], grad_out)

    assert _rel_l2(out, ref) < TOL
    for name, got, exp in zip(("dx", "dWi", "dWo"), grads, ref_grads):
        assert _rel_l2(got, exp) < TOL, f"{name} rel_l2={_rel_l2(got, exp):.4g}"


@pytest.mark.parametrize("srelu", [False, True])
def test_glue_never_copies_wi(srelu):
    """``w1`` must be a pure view of ``Wi`` — no gather, no copy, either way.

    This is the whole point of storing ``Wi`` interleaved: the previous fix
    reordered the columns on every forward, which cost a full ``(E, H, 2I)``
    copy each way (0.362 ms/layer/step at the production shape).
    """
    mod = _sonicmoe()
    i_out = I if srelu else 2 * I
    Wi = torch.randn(E, H, i_out, device="cuda", dtype=torch.bfloat16)
    w1 = mod._sonicmoe_w1(Wi)
    assert w1.data_ptr() == Wi.data_ptr(), "w1 is not a view of Wi"
    assert w1.shape == (i_out, H, E)
    assert w1._base is Wi or w1._base is None
    # And no permutation machinery is left behind.
    assert not hasattr(mod, "_GatedWeightInterleave")
    assert not hasattr(mod, "_interleave_perm")


# ── D7: the combine kernels must see the real top_k ──────────────────────────


@pytest.mark.parametrize("srelu", [False, True])
def test_combine_kernels_specialise_on_true_top_k(monkeypatch, srelu):
    """``MAX_K`` must be ``top_k``, not ``num_experts``.

    SonicMoE's ``token_gather_sum_kernel`` takes ``MAX_K`` as a ``tl.constexpr``
    and ``is_varlen_K`` decides between a static ``Ms = MAX_K * t`` and a
    dynamic per-token segment lookup. Going through the general-routing entry
    point forces ``MAX_K = E`` and ``is_varlen_K = True`` for both the forward
    combine and the backward token reduction.
    """
    mod = _sonicmoe()
    import sonicmoe.functional.backward as sm_bwd
    import sonicmoe.functional.forward as sm_fwd
    from sonicmoe.functional.reduction_over_k_gather import (
        token_gather_and_sum_varlen_K_triton as real_fn,
    )

    seen = []

    def spy(x, w, out, M_perm, M_offset, T_, MAX_K, H_, is_varlen_K):
        seen.append({"MAX_K": MAX_K, "is_varlen_K": is_varlen_K,
                     "w_is_None": w is None})
        return real_fn(x, w, out, M_perm, M_offset, T_, MAX_K, H_, is_varlen_K)

    monkeypatch.setattr(sm_fwd, "token_gather_and_sum_varlen_K_triton", spy)
    monkeypatch.setattr(sm_bwd, "token_gather_and_sum_varlen_K_triton", spy)

    x, Wi, Wo, weights, indices, grad_out = _problem(srelu)
    out = mod.sonicmoe_routed_experts(x, indices, weights, Wi, Wo, E, TOP_K,
                                      srelu, 1.0)
    torch.autograd.grad(out, [x, Wi, Wo], grad_out)

    assert len(seen) == 2, f"expected fwd combine + bwd reduction, got {seen}"
    for call in seen:
        assert call["MAX_K"] == TOP_K, (
            f"combine kernel specialised on MAX_K={call['MAX_K']} instead of "
            f"top_k={TOP_K} (num_experts={E})"
        )
        assert call["is_varlen_K"] is False


def test_no_token_index_materialisation():
    """The fixed-top-k path must not build the ``(T*top_k,)`` token index."""
    mod = _sonicmoe()
    assert mod._HAVE_FIXED_TOPK_PATH, "sonicmoe fixed-top-k entry points missing"

    calls = []
    real_repeat_interleave = torch.Tensor.repeat_interleave

    def spy(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return real_repeat_interleave(self, *args, **kwargs)

    x, Wi, Wo, weights, indices, _ = _problem(srelu=True)
    try:
        torch.Tensor.repeat_interleave = spy
        mod.sonicmoe_routed_experts(x, indices, weights, Wi, Wo, E, TOP_K,
                                    True, 1.0)
    finally:
        torch.Tensor.repeat_interleave = real_repeat_interleave
    assert calls == [], f"unexpected repeat_interleave calls: {calls}"


# ── cross-backend agreement ──────────────────────────────────────────────────


def test_sonicmoe_agrees_with_cutlass_backend():
    """The two production backends must compute the same function (swiglu)."""
    mod = _sonicmoe()
    try:
        from nanoplm.pretraining.models.modern_bert.moe_grouped_gemm_ops import (
            moe_grouped_gemm,
        )
        from nanoplm.pretraining.models.modern_bert.moe_triton_ops import (
            build_inverse_map, moe_gather_combine, moe_scatter_dispatch,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cutlass backend unavailable: {exc}")

    x, Wi, Wo, weights, indices, grad_out = _problem(srelu=False)
    out_s = mod.sonicmoe_routed_experts(x, indices, weights, Wi, Wo, E, TOP_K,
                                        False, 1.0)
    g_s = torch.autograd.grad(out_s, [x, Wi, Wo], grad_out)

    # replicated from MoELayer.forward's cutlass branch
    try:
        expert_flat = indices.reshape(-1)
        sorted_idx = expert_flat.argsort(stable=True)
        counts = expert_flat.bincount(minlength=E).to(torch.int64)
        token_idx = (sorted_idx // TOP_K).to(torch.int32)
        slot_idx = (sorted_idx % TOP_K).to(torch.int32)
        x_sorted = moe_scatter_dispatch(x, token_idx)
        wi = moe_grouped_gemm(x_sorted, Wi, counts)
        gate, up = wi[..., 0::2], wi[..., 1::2]   # layout v2, gate-first
        eo = moe_grouped_gemm(F.silu(gate) * up, Wo, counts)
        inv_map = build_inverse_map(token_idx, slot_idx, T, TOP_K)
        out_c = moe_gather_combine(eo, inv_map, weights, 1.0)
        g_c = torch.autograd.grad(out_c, [x, Wi, Wo], grad_out.to(out_c.dtype))
    except Exception as exc:  # pragma: no cover - JIT build needs ninja/nvcc
        pytest.skip(f"cutlass grouped-gemm extension unavailable: {exc}")

    assert _rel_l2(out_s, out_c) < TOL
    for name, a, b in zip(("dx", "dWi", "dWo"), g_s, g_c):
        assert _rel_l2(a, b) < TOL, f"{name} rel_l2={_rel_l2(a, b):.4g}"
