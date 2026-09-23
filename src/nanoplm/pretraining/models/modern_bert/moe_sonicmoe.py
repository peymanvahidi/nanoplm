"""SonicMoE/quack-backed routed-expert compute for ``MoELayer``.

Drives SonicMoE's fused expert path with nanoplm's own routing decisions, so
nanoplm keeps its sigmoid router, bias-correction, group-limited routing,
aux/z-loss and shared expert while the routed-expert GEMMs run on the sync-free
(fwd+bwd) SonicMoE kernels.

The dispatch gather is fused into the up-projection GEMM (``A_idx=x_gather_idx``)
and the down-projection scatters/combines with the router weights — no
materialized sorted-token buffer.

Weight layout: nanoplm stores ``Wi=(E, H, I_out)`` and ``Wo=(E, I, H)``; SonicMoE
expects ``w1=(I_out, H, E)`` and ``w2=(H, I, E)``. Both are passed as permuted
views (the double permute is a no-op round trip — SonicMoE permutes back before
the GEMM), and nothing else is done to them: no copy, no gather, no reorder.

SwiGLU gate/up convention
-------------------------
QuACK's gated GEMM splits the preactivation **interleaved, gate-first**
(``quack/gemm_interface.py``: ``gate = preact[..., ::2]; up = preact[..., 1::2]``).
nanoplm's routed-expert ``Wi`` is stored in exactly that order
(``moe.WI_GATED_LAYOUT_VERSION == 2``: column ``2j`` is gate unit ``j``, column
``2j+1`` is up unit ``j``), so the two agree natively and this glue is
layout-free.

History: ``Wi`` used to be block-concat ``[up | gate]`` (layout version 1). The
glue then had to gather ``Wi``'s columns into QuACK's order on every forward and
un-permute ``dWi`` in the backward — a full ``(E, H, 2I)`` copy each way,
measured at **0.362 ms/layer/step** (~1.3 % of a training step) at the
production shape, plus one extra ``Wi``-sized activation buffer per MoE layer.
Changing the packing convention deleted all of it. A *strided view* was never an
option: an interleave of two block-concat halves needs two alternating strides
on one dim. QuACK's ``concat_layout=True`` is not a substitute either — it
interleaves along a tensor's *non-contiguous* dim, which for
``(E, H, I_out)``-contiguous ``Wi`` is ``H``, not ``I_out`` (measured: it matches
none of the four possible gate/up pairings).

**Checkpoints.** ``moe_kernel_backend`` may be switched freely between
``cutlass`` and ``sonicmoe`` across a save/resume — both read the same layout.
A checkpoint written under layout version 1 must be converted before it can be
resumed; ``MoELayer`` carries a persistent version marker and refuses to load a
mismatched one. See ``moe.convert_wi_state_dict_v1_to_v2``. ``srelu`` is ungated
and unaffected by any of this.
"""
from __future__ import annotations

import torch

# Imported at module scope on purpose: a function-local import inside the traced
# forward is a Dynamo graph-break risk, and this module is itself only imported
# lazily by ``MoELayer.forward`` when ``moe_kernel_backend='sonicmoe'``.
try:  # pragma: no cover - exercised only when sonicmoe is installed
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import moe_general_routing_inputs

    _SONICMOE_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - optional dependency
    ActivationType = None  # type: ignore[assignment]
    moe_general_routing_inputs = None  # type: ignore[assignment]
    _SONICMOE_IMPORT_ERROR = _exc

# Fixed-top-k fast path. ``moe_general_routing_inputs`` hard-codes
# ``is_each_token_has_variable_activated_expert=True`` and ``K=None``, which
# makes SonicMoE's two combine kernels specialise on ``MAX_K = num_experts``
# (47) instead of ``top_k`` (3) and take the dynamic ``is_varlen_K`` branch
# (``reduction_over_k_gather.py``: ``Ms``/``Me`` loads + non-constexpr trip
# count). nanoplm's routing is always exactly top_k-per-token, so we drive
# SonicMoE's fixed-top-k entry points directly instead. Verified bit-identical
# metadata (expert_frequency / offsets / s_scatter / s_reverse / x_gather) vs
# the general-routing kernels for token-major assignments.
try:  # pragma: no cover - exercised only when sonicmoe is installed
    from sonicmoe.functional import _DownProjection, _UpProjection
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton

    _HAVE_FIXED_TOPK_PATH = True
except Exception:  # pragma: no cover - future sonicmoe may rename these
    _HAVE_FIXED_TOPK_PATH = False


def _sonicmoe_w1(Wi: torch.Tensor) -> torch.Tensor:
    """nanoplm ``Wi (E, H, I_out)`` -> SonicMoE ``w1 (I_out, H, E)`` view.

    A pure view for both activations: ``srelu``/``relu_sq`` is ungated, and the
    gated layout already matches QuACK's interleaved gate-first split (see the
    module docstring).
    """
    return Wi.permute(2, 1, 0)


def _routed_experts_fixed_topk(
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    top_k: int,
    activation,
) -> torch.Tensor:
    """SonicMoE with ``MAX_K = top_k`` and static (non-varlen) combines."""
    T = x.size(0)
    TK = T * top_k
    device = x.device
    i32 = torch.int32

    # (T, top_k) int32 expert ids — the TC metadata kernel wants the 2-D form,
    # so no ``arange(T).repeat_interleave(top_k)`` token index is materialized.
    topk_indices = indices.to(i32).contiguous()

    s_scatter_idx = torch.empty(TK, dtype=i32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=i32, device=device)
    expert_frequency = torch.empty(num_experts, dtype=i32, device=device)
    expert_frequency_offset = torch.empty(num_experts + 1, dtype=i32, device=device)
    x_gather_idx = torch.empty(TK, dtype=i32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices,
        num_experts,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
    )

    a, h = _UpProjection.apply(
        x,
        w1,
        None,                      # b1
        expert_frequency_offset,
        TK,
        top_k,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,                      # num_activated_expert_per_token_offset
        False,                     # is_each_token_has_variable_activated_experts
        activation,
        False,                     # is_inference_mode_enabled
        False,                     # concat_layout
    )
    return _DownProjection.apply(
        a,
        h,
        w2,
        None,                      # b2
        scores,                    # (T, top_k) fp32
        expert_frequency_offset,
        T,
        top_k,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,                      # num_activated_expert_per_token_offset
        False,                     # is_varlen_K
        activation,
    )


def _routed_experts_general(
    x: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    top_k: int,
    activation,
) -> torch.Tensor:
    """Fallback: SonicMoE's public general-routing entry point.

    Same math, but ``MAX_K`` degrades to ``num_experts``. Only used if a future
    SonicMoE renames the fixed-top-k internals.
    """
    T = x.size(0)
    token_indices = torch.arange(
        T, device=x.device, dtype=torch.int32
    ).repeat_interleave(top_k)                                  # [0,0,1,1,...]
    expert_indices = indices.reshape(-1).to(torch.int32).contiguous()  # token-major

    out, _ = moe_general_routing_inputs(
        x,
        scores.reshape(-1),
        token_indices,
        expert_indices,
        w1,
        None,            # b1
        w2,
        None,            # b2
        num_experts,
        torch.cuda.current_stream().cuda_stream,   # unused by sonicmoe 0.1.2
        activation,
    )
    return out


def sonicmoe_routed_experts(
    x: torch.Tensor,
    indices: torch.Tensor,        # (T, top_k) selected expert ids
    weights: torch.Tensor,        # (T, top_k) combine weights (renormalized sigmoid)
    Wi: torch.Tensor,             # (E, H, I_out)
    Wo: torch.Tensor,             # (E, I, H)
    num_experts: int,
    top_k: int,
    srelu: bool,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Return the combined routed-expert output ``(T, H)``.

    Routing is token-major (token id ascending): assignment ``t * top_k + k`` is
    token ``t``'s ``k``-th expert, which is what both of SonicMoE's metadata
    kernels expect.
    """
    if moe_general_routing_inputs is None:  # pragma: no cover - optional dep
        raise RuntimeError(
            "moe_kernel_backend='sonicmoe' requires the sonic-moe package "
            f"(install it with `nanoplm deps`). Import failed: {_SONICMOE_IMPORT_ERROR!r}"
        )

    # fp32 combine weights. ``routed_scaling_factor == 1.0`` is the default and
    # the multiply is then a no-op whose backward is pure overhead, so skip it.
    scores = weights.float()
    if routed_scaling_factor != 1.0:
        scores = scores * routed_scaling_factor
    scores = scores.contiguous()

    w1 = _sonicmoe_w1(Wi)                   # (I_out, H, E)
    w2 = Wo.permute(2, 1, 0)                # (H,     I, E)
    activation = ActivationType.RELU_SQ if srelu else ActivationType.SWIGLU

    runner = _routed_experts_fixed_topk if _HAVE_FIXED_TOPK_PATH else _routed_experts_general
    return runner(x, indices, scores, w1, w2, num_experts, top_k, activation)
