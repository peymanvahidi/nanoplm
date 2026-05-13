"""Mixture-of-Experts layer for nanoPLM.

Uses ``grouped_gemm`` for routed expert compute so per-expert token counts stay
on GPU and expert GEMMs run as a single grouped kernel instead of a host loop.

Only supports the packed flat-token static-shape forward path
(``use_packing=True``, ``use_static_inp_size=True``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_grouped_gemm_ops import moe_grouped_gemm
from .moe_triton_ops import (
    build_inverse_map,
    moe_gather_combine,
    moe_scatter_dispatch,
)


class Router(nn.Module):
    """Sigmoid top-k router with optional routing-bias correction."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        bias_correction: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        if n_group > 1:
            assert num_experts % n_group == 0, (
                f"num_experts ({num_experts}) must be divisible by n_group ({n_group})"
            )
        if bias_correction:
            # Correction bias: added to logits for expert *selection* only,
            # NOT included in the combine weights.  Updated externally based
            # on global expert-load statistics (DeepSeek-V3 style).
            self.register_buffer(
                "correction_bias", torch.zeros(num_experts), persistent=True,
            )
        else:
            self.correction_bias = None

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: ``(total_tokens, hidden_size)``

        Returns:
            weights: ``(total_tokens, top_k)`` — combine weights (from raw
                     sigmoid scores, bias-free, renormalized).
            indices: ``(total_tokens, top_k)`` — selected expert indices.
            z_loss:  scalar — mean squared pre-sigmoid logit (prevents
                     sigmoid saturation / vanishing router gradients).
        """
        # Router runs in fp32 for stability (Ling-V2 / DeepSeek-V3 style).
        raw_logits = self.gate(x).float()  # (T, E) — cast to fp32 after projection
        logits = raw_logits.sigmoid()  # (T, E)

        # Z-loss: penalise large pre-sigmoid logits to keep sigmoid in its
        # sensitive region (DeepSeek-V3 style).
        z_loss = raw_logits.float().square().mean()

        if self.correction_bias is not None:
            # Use biased logits for selection, raw logits for combine weights.
            selection_logits = logits + self.correction_bias
        else:
            selection_logits = logits

        # Group-limited routing (Ling-V2 / DeepSeek-V3 style):
        # partition experts into groups, score groups by top-2 expert scores,
        # keep only topk_group groups, mask out the rest before final top-k.
        if self.n_group > 1:
            T = x.shape[0]
            epg = self.gate.out_features // self.n_group  # experts per group
            grouped = selection_logits.view(T, self.n_group, epg)
            # Score each group by sum of its top-2 expert scores.
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)  # (T, n_group)
            # Keep top topk_group groups.
            top_groups = group_scores.topk(self.topk_group, dim=-1).indices  # (T, topk_group)
            # Build per-expert mask: 1 for experts in selected groups, 0 otherwise.
            group_mask = torch.zeros(
                T, self.n_group, device=x.device, dtype=selection_logits.dtype,
            )
            group_mask.scatter_(1, top_groups, 1.0)
            expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, epg).reshape(T, -1)
            # Mask out non-selected experts so top-k ignores them.
            selection_logits = selection_logits.masked_fill(
                expert_mask == 0, torch.finfo(selection_logits.dtype).min,
            )

        weights, indices = torch.topk(selection_logits, self.top_k, dim=-1)

        # Combine weights come from the *unbiased* sigmoid scores.
        if self.correction_bias is not None:
            weights = logits.gather(dim=-1, index=indices)

        # Renormalize so combine weights sum to 1 per token. Clamp the
        # denominator to keep backward stable when selected sigmoid scores are
        # extremely small early in training.
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights / denom

        return weights, indices, z_loss, logits


class MoELayer(nn.Module):
    """Drop-in MLP replacement with Mixture-of-Experts.

    Forward signature matches ``ModernBertSwiGLUMLP`` so the encoder layer
    can swap it in without any call-site changes.
    """

    def __init__(self, config) -> None:
        super().__init__()
        from .modeling import ModernBertSwiGLUMLP

        self.num_experts: int = config.moe_num_experts
        self.top_k: int = config.moe_top_k
        self.hidden_size: int = config.hidden_size
        self.intermediate_size: int = config.intermediate_size

        # Router
        self.router = Router(
            config.hidden_size,
            self.num_experts,
            self.top_k,
            bias_correction=config.moe_use_bias_correction,
            n_group=config.moe_n_group,
            topk_group=config.moe_topk_group,
        )

        # Routed expert weights — stacked for grouped_gemm.
        # Layout: (num_experts, in_features, out_features).
        # This is transposed relative to nn.Linear's (out, in).
        self.Wi = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size, 2 * config.intermediate_size),
        )
        self.Wo = nn.Parameter(
            torch.empty(self.num_experts, config.intermediate_size, config.hidden_size),
        )
        self.drop = nn.Dropout(config.mlp_dropout)

        # Shared expert — a normal MLP that processes all tokens.
        self.shared_expert = ModernBertSwiGLUMLP(config)

        self.aux_loss_coef: float = float(config.moe_aux_loss_coef)
        self.z_loss_coef: float = float(config.moe_z_loss_coef)
        self.routed_scaling_factor: float = float(config.moe_routed_scaling_factor)
        ckpt_enabled = bool(getattr(config, "activation_checkpointing", False))
        ckpt_mode = str(
            getattr(config, "activation_checkpointing_mode", "layer")
        ).strip().lower()
        # Recompute can revisit the MoE branch in full-layer checkpointing or
        # when the MLP branch itself is checkpointed. Attention-only
        # checkpointing does not re-enter MoE, so keep workspace reuse enabled
        # there to avoid the slower bincount fallback.
        self._reuse_dispatch_workspaces = not (
            ckpt_enabled and ckpt_mode in {"layer", "attn+mlp"}
        )

        # Reuse small dispatch metadata buffers across steps to reduce allocator
        # churn in the static packed MoE path. Keep this off when checkpoint
        # recompute can re-enter the MoE branch and clobber buffers still needed
        # later.
        self.register_buffer(
            "_dispatch_sort_values",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_dispatch_sorted_idx",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_dispatch_count_ones",
            torch.empty(0, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_dispatch_counts",
            torch.empty(self.num_experts, dtype=torch.int64),
            persistent=False,
        )

        # Bookkeeping set during forward for external consumption.
        self.last_expert_counts: Optional[torch.Tensor] = None
        self.last_aux_loss: Optional[torch.Tensor] = None

    def _ensure_dispatch_buffers(
        self,
        num_assignments: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._dispatch_sort_values.device != device:
            self._dispatch_sort_values = self._dispatch_sort_values.to(device=device)
            self._dispatch_sorted_idx = self._dispatch_sorted_idx.to(device=device)
            self._dispatch_count_ones = self._dispatch_count_ones.to(device=device)
            self._dispatch_counts = self._dispatch_counts.to(device=device)
        if self._dispatch_sort_values.numel() < num_assignments:
            self._dispatch_sort_values.resize_(num_assignments)
            self._dispatch_sorted_idx.resize_(num_assignments)
            self._dispatch_count_ones.resize_(num_assignments)
        self._dispatch_counts.resize_(self.num_experts)
        return (
            self._dispatch_sort_values[:num_assignments],
            self._dispatch_sorted_idx[:num_assignments],
            self._dispatch_count_ones[:num_assignments],
            self._dispatch_counts,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Handle both packed 2D (T, H) and padded 3D (B, S, H) input.
        orig_shape = x.shape
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        T, H = x.shape

        # --- shared expert (processes all tokens) ---
        shared_out = self.shared_expert(
            x,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
        )

        # --- routing ---
        weights, indices, z_loss, router_scores = self.router(x)  # (T, top_k), (T, top_k), scalar, (T, E)

        # --- dispatch: sort by expert, compute per-expert counts, fused gather ---
        expert_flat = indices.reshape(-1)  # (T*top_k,)
        if self._reuse_dispatch_workspaces:
            num_assignments = expert_flat.numel()
            sort_vals, sorted_idx, count_ones, counts = self._ensure_dispatch_buffers(
                num_assignments,
                expert_flat.device,
            )
            torch.sort(expert_flat, stable=True, out=(sort_vals, sorted_idx))
            count_ones.fill_(1)
            counts.zero_()
            counts.scatter_add_(0, expert_flat, count_ones)
        else:
            sorted_idx = expert_flat.argsort(stable=True)
            counts = expert_flat.bincount(minlength=self.num_experts).to(torch.int64)

        token_idx = (sorted_idx // self.top_k).to(torch.int32)
        slot_idx = (sorted_idx % self.top_k).to(torch.int32)
        x_sorted = moe_scatter_dispatch(x, token_idx)  # (T*top_k, H)

        # Store for external bias-update logic.
        self.last_expert_counts = counts.detach().clone()

        # --- expert SwiGLU via grouped_gemm ---
        wi = moe_grouped_gemm(x_sorted, self.Wi, counts)  # (T*top_k, 2*inter)
        x_proj, gate = wi.chunk(2, dim=-1)
        activated = F.silu(gate) * x_proj  # (T*top_k, inter)
        expert_out_sorted = moe_grouped_gemm(
            self.drop(activated),
            self.Wo,
            counts,
        )  # (T*top_k, H)

        # --- combine: fused unsort + weighted sum ---
        inv_map = build_inverse_map(token_idx, slot_idx, T, self.top_k)
        expert_out = moe_gather_combine(
            expert_out_sorted, inv_map, weights, self.routed_scaling_factor,
        )  # (T, H)

        # --- auxiliary losses ---
        aux = torch.zeros(1, device=x.device, dtype=torch.float32)
        # Differentiable balance regularizer over normalized router mass.
        # The previous bincount-based aux loss used hard top-k indices and did
        # not produce useful router gradients.
        if self.aux_loss_coef > 0:
            router_mass = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            mean_mass = router_mass.mean(dim=0)
            target = torch.ones_like(mean_mass) / self.num_experts
            aux = aux + self.aux_loss_coef * ((mean_mass - target) ** 2).sum() * self.num_experts
        # Z-loss: prevents sigmoid saturation in the router.
        if self.z_loss_coef > 0:
            aux = aux + self.z_loss_coef * z_loss
        self.last_aux_loss = aux.squeeze()

        out = expert_out + shared_out
        if len(orig_shape) == 3:
            out = out.view(orig_shape)
        return out
