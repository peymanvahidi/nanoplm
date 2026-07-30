"""SonicMoE/quack-backed routed-expert compute for ``MoELayer``.

Drives SonicMoE's fused expert path (``moe_general_routing_inputs``) with
nanoplm's own routing decisions, so nanoplm keeps its sigmoid router,
bias-correction, group-limited routing, aux/z-loss and shared expert while the
routed-expert GEMMs run on the sync-free (fwd+bwd) SonicMoE kernels.

The dispatch gather is fused into the up-projection GEMM (``A_idx=x_gather_idx``)
and the down-projection scatters/combines with the router weights — no
materialized sorted-token buffer.

Weight layout: nanoplm stores ``Wi=(E, H, I_out)`` and ``Wo=(E, I, H)``; SonicMoE
expects ``w1=(I_out, H, E)`` and ``w2=(H, I, E)`` (used via permuted views).
"""
from __future__ import annotations

import torch


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

    SonicMoE's ``general_routing`` builds the expert-grouped dispatch itself, so
    it wants the **token-major** assignments (token id ascending), NOT an
    expert-sorted order: ``token_indices = [0,0,1,1,...]`` and
    ``expert_indices = indices.reshape(-1)``.
    """
    from sonicmoe.functional import moe_general_routing_inputs
    from sonicmoe.enums import ActivationType

    T = x.size(0)
    token_indices = torch.arange(
        T, device=x.device, dtype=torch.int32
    ).repeat_interleave(top_k)                                  # [0,0,1,1,...]
    expert_indices = indices.reshape(-1).to(torch.int32).contiguous()  # token-major
    scores = (weights.reshape(-1).float() * routed_scaling_factor).contiguous()

    # nanoplm (E, in, out) -> sonicmoe (out, in, E) view.
    if srelu:
        activation = ActivationType.RELU_SQ
        w1_src = Wi
    else:
        activation = ActivationType.SWIGLU
        # nanoplm stores the gated Wi as CHUNKED [up(I) | gate(I)] on the last
        # axis (see MoELayer / the cutlass path's `x_proj, gate = wi.chunk(2)`),
        # but sonicmoe's gated kernels expect the halves INTERLEAVED as
        # [g0,u0,g1,u1,...] with even lanes = gate, odd = up (sonicmoe.moe._swiglu:
        # `u = x[...,1::2]; g = x[...,::2]; u*silu(g)`). Reinterleave here so the
        # fused SwiGLU silu()s the correct half; without this the output is wrong
        # (silu applied to the wrong columns). Grads flow back to the chunked Wi.
        # NB: this allocates a fresh (E,H,2I) tensor per forward; store Wi
        # interleaved natively if this shows up in profiles.
        I = Wi.shape[-1] // 2
        up, gate = Wi[..., :I], Wi[..., I:]
        w1_src = torch.stack([gate, up], dim=-1).reshape(*Wi.shape[:-1], 2 * I)
    w1 = w1_src.permute(2, 1, 0)  # (I_out, H, E)
    w2 = Wo.permute(2, 1, 0)      # (H,     I, E)
    stream_id = torch.cuda.current_stream().cuda_stream

    out, _ = moe_general_routing_inputs(
        x,
        scores,
        token_indices,
        expert_indices,
        w1,
        None,            # b1
        w2,
        None,            # b2
        num_experts,
        stream_id,
        activation,
    )
    return out
