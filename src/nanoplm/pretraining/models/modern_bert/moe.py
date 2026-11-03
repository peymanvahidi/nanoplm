"""Mixture-of-Experts layer for nanoPLM.

Uses ``grouped_gemm`` for routed expert compute so per-expert token counts stay
on GPU and expert GEMMs run as a single grouped kernel instead of a host loop.

Only supports the packed flat-token static-shape forward path
(``use_packing=True``, ``use_static_inp_size=True``).

Routed-expert weight layout
---------------------------
Expert weights are stacked ``nn.Linear`` weights, ``(E, out_features,
in_features)``: ``Wi (E, 2I, H)`` (``(E, I, H)`` for sReLU) and ``Wo (E, H, I)``.
Matrix optimizers (Muon/NorMuon/Dion) read the last two dims as
``(fan_out, fan_in)``: the spectral-norm LR scale and NorMuon's per-neuron
normalization both depend on it, so storing ``(E, in, out)`` silently gives
expert matrices a different update than a dense layer of the same shape. It is
also SonicMoE's native layout. See ``EXPERT_LAYOUT_VERSION``.

For SwiGLU, ``Wi``'s output rows are stored **interleaved, gate first**: row
``2j`` is gate unit ``j``, row ``2j+1`` is up unit ``j``. This is
QuACK/SonicMoE's native gated-GEMM split, so the ``sonicmoe`` backend consumes
``Wi`` with no reordering; the ``cutlass``/eager branch reads the two strided
slices. ``Wo`` is indexed by unit ``j`` under both conventions and is
unaffected. See ``WI_GATED_LAYOUT_VERSION``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.library
import torch.nn as nn
import torch.nn.functional as F

from .moe_grouped_gemm_ops import moe_grouped_gemm
from .moe_triton_ops import (
    build_inverse_map,
    moe_gather_combine,
    moe_scatter_dispatch,
)

# Storage convention for the gated routed-expert input projection ``Wi``.
#
#   1  block-concat ``[up | gate]``  (nanoPLM <= 2026-08-06)
#   2  interleaved gate-first ``[gate_0, up_0, gate_1, up_1, ...]``  (current)
#
# Version 2 is QuACK/SonicMoE's native layout, which removes a full
# ``(E, H, 2I)`` gather in the forward and its inverse in the backward on the
# ``sonicmoe`` path (measured 0.362 ms/layer/step at the production shape).
#
# The two conventions are *different functions of the same weights*, so a v1
# checkpoint loaded by v2 code would silently re-pair gate/up and destroy the
# run. Every gated ``MoELayer`` therefore carries the version as a persistent
# buffer and refuses to load a state_dict that does not match; use
# ``convert_wi_state_dict_v1_to_v2`` to migrate.
WI_GATED_LAYOUT_VERSION = 2
_WI_LAYOUT_BUFFER = "_wi_gated_layout_version"


def _interleave_gate_first(wi: torch.Tensor) -> torch.Tensor:
    """``[up | gate]`` block-concat -> ``[gate_0, up_0, gate_1, up_1, ...]``."""
    half = wi.shape[-1] // 2
    out = torch.empty_like(wi)
    out[..., 0::2] = wi[..., half:]   # gate
    out[..., 1::2] = wi[..., :half]   # up
    return out


def convert_wi_state_dict_v1_to_v2(
    state_dict: dict,
    optimizer_state_dict: Optional[dict] = None,
) -> list[str]:
    """Migrate a pre-2026-08-06 checkpoint's gated routed-expert ``Wi`` in place.

    ``[up | gate]`` block-concat  ->  ``[gate_0, up_0, gate_1, up_1, ...]``::

        I = Wi.shape[-1] // 2
        Wi_v2[..., 0::2] = Wi_v1[..., I:]    # gate
        Wi_v2[..., 1::2] = Wi_v1[..., :I]    # up

    ``Wo`` is indexed by intermediate unit under both conventions and is left
    alone. Gate/up version 1 predates ``EXPERT_LAYOUT_VERSION`` 2, so this works
    on the ``(E, H, 2I)`` expert layout; follow it with
    ``convert_expert_layout_v1_to_v2``.  A ``Wi`` is treated as gated iff its sibling ``Wo`` says so
    (``Wi.shape[-1] == 2 * Wo.shape[-2]``), so an sReLU checkpoint is a no-op
    and this is safe to run over anything.

    **Pass ``optimizer_state_dict`` too when resuming training.**  Muon/NorMuon
    momentum and second moments are per-element, so they carry the old column
    order; every optimizer state tensor shaped like a converted ``Wi`` is
    permuted the same way.  (Newton-Schulz itself is exactly equivariant under a
    column permutation, so no other optimizer surgery is needed.)  Omitting it
    is only correct if the optimizer state is being discarded.

    Usage::

        from nanoplm.pretraining.models.modern_bert.moe import (
            convert_wi_state_dict_v1_to_v2,
        )
        msd = torch.load(ckpt / "pytorch_model.bin", weights_only=True)
        osd = torch.load(ckpt / "optimizer.pt", weights_only=False)
        convert_wi_state_dict_v1_to_v2(msd, osd)
        torch.save(msd, ckpt / "pytorch_model.bin")
        torch.save(osd, ckpt / "optimizer.pt")

    Returns the list of converted ``Wi`` keys.
    """
    converted: list[str] = []
    shapes: dict[str, torch.Size] = {}
    for key in [k for k in state_dict if k.endswith("Wi")]:
        wi = state_dict[key]
        if not torch.is_tensor(wi) or wi.ndim != 3:
            continue  # dense nn.Linear Wi (2-D) — different convention, skip
        wo = state_dict.get(key[: -len("Wi")] + "Wo")
        if not torch.is_tensor(wo) or wo.ndim != 3:
            continue
        if wi.shape[0] != wo.shape[0] or wi.shape[-1] != 2 * wo.shape[-2]:
            continue  # ungated (sReLU) or not a routed-expert pair
        state_dict[key] = _interleave_gate_first(wi)
        state_dict[key[: -len("Wi")] + _WI_LAYOUT_BUFFER] = torch.tensor(
            [WI_GATED_LAYOUT_VERSION], dtype=torch.int64
        )
        converted.append(key)
        shapes[key] = wi.shape

    if not converted:
        raise ValueError(
            "convert_wi_state_dict_v1_to_v2 found no gated MoE routed-expert "
            "Wi/Wo pair in this state_dict — it is either already converted, "
            "an sReLU model, or not a MoE checkpoint."
        )

    if optimizer_state_dict is not None:
        opt_state = optimizer_state_dict.get("state", optimizer_state_dict)
        for key in converted:
            entry = opt_state.get(key)
            if not isinstance(entry, dict):
                continue
            for name, val in entry.items():
                if torch.is_tensor(val) and val.shape == shapes[key]:
                    entry[name] = _interleave_gate_first(val)
    return converted

# Storage convention for the routed-expert weights ``Wi`` / ``Wo``.
#
#   1  ``(E, in_features, out_features)``, transposed vs nn.Linear  (nanoPLM <= 2026-10-29)
#   2  ``(E, out_features, in_features)``, stacked nn.Linear weights  (current)
#
# Same function either way; what changes is how a matrix optimizer reads the
# tensor. Dion's Muon/NorMuon take ``(fan_out, fan_in) = shape[-2:]``, so under
# version 1 the spectral-norm LR scale was inverted for every expert matrix
# (``Wo`` at g7 shapes got 0.375x the update a dense layer of its shape gets,
# ``Wi`` 1.33x) and NorMuon normalized per *input* unit instead of per output
# neuron. Input units of ``Wo`` (intermediate units) go dormant in sparse
# experts; output neurons do not. Every ``MoELayer`` carries the version as a
# persistent buffer and refuses a mismatched state_dict; migrate with
# ``convert_expert_layout_v1_to_v2``.
EXPERT_LAYOUT_VERSION = 2
_EXPERT_LAYOUT_BUFFER = "_expert_layout_version"


def _is_routed_expert_pair(state_dict: dict, wi_key: str) -> bool:
    wi = state_dict.get(wi_key)
    wo = state_dict.get(wi_key[: -len("Wi")] + "Wo")
    return (
        torch.is_tensor(wi) and torch.is_tensor(wo)
        and wi.ndim == 3 and wo.ndim == 3 and wi.shape[0] == wo.shape[0]
    )


def convert_expert_layout_v1_to_v2(
    state_dict: dict,
    optimizer_state_dict: Optional[dict] = None,
) -> list[str]:
    """Migrate a version-1 ``(E, in, out)`` checkpoint to ``(E, out, in)`` in place.

    Transposes the last two dims of every routed-expert ``Wi`` and ``Wo`` and
    writes the version marker. Run ``convert_wi_state_dict_v1_to_v2`` *first*
    on checkpoints older than 2026-08-06 (it works on the version-1 expert
    layout).

    **Optimizer state** (pass ``optimizer_state_dict`` when resuming): every
    per-element state tensor shaped like the old weight (Muon/NorMuon
    ``momentum``, AdamW ``exp_avg``/``exp_avg_sq``) is transposed too. NorMuon's
    ``variance_neuron`` is a per-row statistic of the *old* rows (input units),
    which has no counterpart on the new rows, so it is reset to zeros of the new
    shape; NorMuon then rebuilds it from the next updates (its EMA makes the
    first update per row at most ``1/sqrt(1 - muon_beta2)`` larger than steady
    state, and the Frobenius rescale bounds the whole update). Omitting the
    optimizer state is only correct if it is being discarded.

    Returns the list of converted parameter keys.
    """
    converted: list[str] = []
    old_shapes: dict[str, torch.Size] = {}
    for wi_key in [k for k in state_dict if k.endswith("Wi")]:
        if not _is_routed_expert_pair(state_dict, wi_key):
            continue  # dense nn.Linear Wi (2-D) or not a routed-expert pair
        prefix = wi_key[: -len("Wi")]
        for key in (wi_key, prefix + "Wo"):
            old_shapes[key] = state_dict[key].shape
            state_dict[key] = state_dict[key].transpose(-1, -2).contiguous()
            converted.append(key)
        state_dict[prefix + _EXPERT_LAYOUT_BUFFER] = torch.tensor(
            [EXPERT_LAYOUT_VERSION], dtype=torch.int64
        )

    if not converted:
        raise ValueError(
            "convert_expert_layout_v1_to_v2 found no MoE routed-expert Wi/Wo pair "
            "in this state_dict; it is not a MoE checkpoint."
        )

    if optimizer_state_dict is not None:
        # nanoplm's MuonAdamWGroup saves {"muon": {...}, "adamw": {...}}; a plain
        # optimizer saves {"state": ..., "param_groups": ...}.
        subs = (
            [optimizer_state_dict[k] for k in ("muon", "adamw") if k in optimizer_state_dict]
            or [optimizer_state_dict]
        )
        for sub in subs:
            opt_state = sub.get("state", sub)
            for key in converted:
                entry = opt_state.get(key)
                if not isinstance(entry, dict):
                    continue
                old = old_shapes[key]
                for name, val in list(entry.items()):
                    if not torch.is_tensor(val):
                        continue
                    if val.shape == old:
                        entry[name] = val.transpose(-1, -2).contiguous()
                    elif name == "variance_neuron":
                        entry[name] = torch.zeros(
                            *old[:-2], old[-1], 1, dtype=val.dtype, device=val.device
                        )
    return converted


# ── Fused router top-k + expert histogram ────────────────────────────────────
#
# ``torch.topk`` on the router's (T, E) score matrix is pathologically slow at
# MoE shapes: picking 3 of 47 over 131,072 rows measures **470 us** on a GH200
# (24.6 MB read => 6.7 us at 3,668 GB/s, i.e. 1.4% of HBM roofline), split
# across ``sbtopk::gatherTopK`` and a ``bitonicSortKVInPlace`` that only exists
# because ``sorted=True`` is the default.  A single-pass Triton kernel that
# keeps the whole row in registers does the same job in **63 us** (7.5x), and
# gets the per-expert assignment histogram almost for free: the one-hot mask
# used to knock the winner out for the next round is exactly the histogram
# contribution, so it costs one extra column reduction and one vectorised
# atomic per tile instead of ``T*top_k`` colliding atomics on ``E`` addresses
# (measured 137 us for the ``ones_like``/``zeros``/``scatter_add_`` triple).
#
# The kernel returns *indices only*; the caller gathers the values from the
# source tensor, which is bit-identical to ``torch.topk``'s values and keeps
# the differentiable path in plain ATen ops.
try:  # pragma: no cover - exercised on GPU builds
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _router_topk_kernel(
        X, IDX, CNT,
        M,
        E_: tl.constexpr,
        K_: tl.constexpr,
        COUNT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Top-K over the last dim of a (M, E_) row-major tensor.

        One program owns BLOCK_M rows and holds the full row in registers.
        Round ``i`` takes the row max, records the *lowest* column index that
        attains it (deterministic tie-break), then masks that column to -inf.
        Values are recomputed by the caller via ``gather``.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_E)
        rmask = rows < M
        cmask = cols < E_
        x = tl.load(
            X + rows[:, None] * E_ + cols[None, :],
            mask=rmask[:, None] & cmask[None, :],
            other=float("-inf"),
        )
        acc = tl.zeros((BLOCK_E,), dtype=tl.int32)
        for i in tl.static_range(K_):
            m = tl.max(x, axis=1)
            eq = x == m[:, None]
            idx = tl.min(tl.where(eq, cols[None, :], E_), axis=1)
            tl.store(IDX + rows * K_ + i, idx, mask=rmask)
            hit = cols[None, :] == idx[:, None]
            if COUNT:
                acc += tl.sum((hit & rmask[:, None]).to(tl.int32), axis=0)
            x = tl.where(hit, float("-inf"), x)
        if COUNT:
            tl.atomic_add(CNT + cols, acc, mask=cmask)


_lib = torch.library.Library("nanoplm_router", "DEF")
_lib.define("topk_counts(Tensor scores, int k, bool want_counts) -> (Tensor, Tensor)")


@torch.library.register_fake("nanoplm_router::topk_counts")
def _topk_counts_fake(scores, k, want_counts):
    m, e = scores.shape
    return (
        scores.new_empty((m, k), dtype=torch.int64),
        scores.new_empty((e,), dtype=torch.int64),
    )


@torch.library.impl(_lib, "topk_counts", "CUDA")
def _topk_counts_cuda(scores, k, want_counts):
    scores = scores.contiguous()
    m, e = scores.shape
    idx = torch.empty((m, k), device=scores.device, dtype=torch.int64)
    cnt32 = torch.zeros((e,), device=scores.device, dtype=torch.int32)
    if m == 0:
        return idx, cnt32.to(torch.int64)
    block_e = triton.next_power_of_2(e)
    block_m = max(16, min(128, 8192 // block_e))
    num_warps = 8 if block_m >= 128 else 4
    _router_topk_kernel[(triton.cdiv(m, block_m),)](
        scores, idx, cnt32, m, e, k, bool(want_counts),
        BLOCK_M=block_m, BLOCK_E=block_e,
        num_warps=num_warps, num_stages=1,
    )
    return idx, cnt32.to(torch.int64)


@torch.library.impl(_lib, "topk_counts", "CPU")
def _topk_counts_fallback(scores, k, want_counts):
    """Reference path (CPU). Mirrors the kernel, including its tie-break.

    ``argmax`` does not promise a tie-break, so the winning column is picked
    explicitly as the lowest index attaining the row max — the same rule the
    Triton kernel uses.
    """
    e = scores.shape[-1]
    work = scores.clone()
    cols_ar = torch.arange(e, device=scores.device)
    picks = []
    for _ in range(k):
        m = work.max(dim=-1, keepdim=True).values
        i = torch.where(work == m, cols_ar, e).min(dim=-1, keepdim=True).values
        picks.append(i)
        work.scatter_(-1, i, float("-inf"))
    idx = torch.cat(picks, dim=-1)
    flat = idx.reshape(-1)
    cnt = torch.zeros(e, device=scores.device, dtype=torch.int64)
    if want_counts:
        cnt.scatter_add_(0, flat, torch.ones_like(flat))
    return idx, cnt


def _router_topk_indices(
    scores: torch.Tensor, k: int, want_counts: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Top-k expert indices (and optionally the assignment histogram).

    Falls back to ``torch.topk`` for shapes the fused kernel cannot hold in
    registers, keeping the exact same tie-break and output layout.
    """
    e = scores.shape[-1]
    supported = (
        0 < k <= e <= 1024
        and scores.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )
    use_fused = supported and (_HAVE_TRITON if scores.is_cuda else True)
    if use_fused:
        idx, cnt = torch.ops.nanoplm_router.topk_counts(scores, k, want_counts)
        return idx, (cnt if want_counts else None)
    idx = torch.topk(scores, k, dim=-1).indices
    if not want_counts:
        return idx, None
    flat = idx.reshape(-1)
    cnt = torch.zeros(e, device=scores.device, dtype=torch.int64)
    cnt.scatter_add_(0, flat, torch.ones_like(flat))
    return idx, cnt


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
        nan_guard: bool = False,
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        # Same value on every layer, so dynamo guards once and keeps one graph.
        self.nan_guard = bool(nan_guard)
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        # cuBLAS has no sm90 kernel for a bf16 GEMM whose N (fwd) / K (dgrad) /
        # M (wgrad) is 47: the odd extent forces an `align1` tile and it falls
        # back to an Ampere-class `cutlass_75_..._s1688` kernel. Inductor's
        # pad_mm already pads the *forward* to 48 (visible in the trace as
        # `triton_poi_fused_mm_t` -> a `nvjet_sm90_tst_48x512` GEMM) but not
        # the two backward GEMMs. Padding the weight here forces all three onto
        # the sm90 path: measured 348->110 us (fwd), 556->394 us (dgrad),
        # 230->111 us (wgrad) at T=131072, H=1024.
        self._gate_pad = (-num_experts) % 8
        # Only the routed experts consume the counts; skip the histogram
        # entirely when nothing reads it (see MoELayer.last_expert_counts).
        self.track_expert_counts = bool(bias_correction)
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
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        """Route tokens to experts.

        Args:
            x: ``(total_tokens, hidden_size)``

        Returns:
            weights: ``(total_tokens, top_k)`` — combine weights (from raw
                     sigmoid scores, bias-free, renormalized).
            indices: ``(total_tokens, top_k)`` — selected expert indices.
            z_loss:  scalar — mean squared pre-sigmoid logit (prevents
                     sigmoid saturation / vanishing router gradients).
            logits:  ``(total_tokens, num_experts)`` — raw sigmoid scores.
            counts:  ``(num_experts,)`` int64 assignment histogram, or ``None``
                     when ``track_expert_counts`` is off. Produced by the fused
                     top-k kernel, so it costs one reduction instead of
                     ``total_tokens * top_k`` colliding atomics.
        """
        # Router runs in fp32 for stability (Ling-V2 / DeepSeek-V3 style).
        if self._gate_pad:
            w = F.pad(self.gate.weight, (0, 0, 0, self._gate_pad))
            raw_logits = F.linear(x, w)[:, : self.num_experts].float()
        else:
            raw_logits = self.gate(x).float()  # (T, E)
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

        # Fused single-pass top-k. Values are gathered rather than returned by
        # the kernel: `selection_logits.gather(-1, indices)` is bit-identical to
        # `torch.topk(...).values` and keeps autograd on plain ATen ops.
        indices, counts = _router_topk_indices(
            selection_logits, self.top_k, self.track_expert_counts,
        )
        if self.nan_guard and self.training:
            # An all-NaN row comes back from the fused top-k as an out-of-bounds
            # sentinel, and the gather below would abort the job on it. Keep the
            # index in range so the forward finishes with a non-finite loss that
            # the pipeline's skip path discards. Training only: in eval there is
            # no skip path, so a NaN still aborts loudly. Out-of-place, because
            # the gather saves `indices` for backward.
            indices = indices.clamp(0, self.num_experts - 1)

        # Combine weights come from the *unbiased* sigmoid scores.
        if self.correction_bias is not None:
            weights = logits.gather(dim=-1, index=indices)
        else:
            weights = selection_logits.gather(dim=-1, index=indices)

        # Renormalize so combine weights sum to 1 per token. Clamp the
        # denominator to keep backward stable when selected sigmoid scores are
        # extremely small early in training.
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights / denom

        return weights, indices, z_loss, logits, counts


class MoELayer(nn.Module):
    """Drop-in MLP replacement with Mixture-of-Experts.

    Forward signature matches ``ModernBertSwiGLUMLP`` so the encoder layer
    can swap it in without any call-site changes.
    """

    def __init__(self, config) -> None:
        super().__init__()
        from .modeling import ModernBertSReluMLP, ModernBertSwiGLUMLP

        self.num_experts: int = config.moe_num_experts
        self.top_k: int = config.moe_top_k
        self.hidden_size: int = config.hidden_size
        self.intermediate_size: int = config.intermediate_size
        self.srelu: bool = config.mlp_activation == "srelu"

        # Router
        self.router = Router(
            config.hidden_size,
            self.num_experts,
            self.top_k,
            bias_correction=config.moe_use_bias_correction,
            n_group=config.moe_n_group,
            topk_group=config.moe_topk_group,
            nan_guard=config.moe_nan_guard,
        )

        # Routed expert weights — stacked for grouped_gemm.
        # Layout: (num_experts, out_features, in_features), i.e. stacked
        # nn.Linear weights (``EXPERT_LAYOUT_VERSION``). Matrix optimizers read
        # the last two dims as (fan_out, fan_in), so this must match nn.Linear.
        #
        # SwiGLU's Wi emits gate+up (2*inter); sReLU is non-gated (just inter).
        # ``Wi``'s output rows are stored **interleaved, gate first**:
        # row ``2j`` is gate unit ``j``, row ``2j+1`` is up unit ``j``
        # (``WI_GATED_LAYOUT``).  That is exactly the split QuACK/SonicMoE's
        # gated GEMM applies (``gate = preact[..., ::2]; up = preact[..., 1::2]``),
        # so the ``sonicmoe`` backend can consume ``Wi`` with no reordering at
        # all.  ``Wo`` is indexed by *unit* ``j``, which both backends agree on,
        # so it is unaffected by this convention.
        #
        # NOTE: this differs from the *dense* SwiGLU MLPs (including this
        # layer's own ``shared_expert``), which keep nn.Linear's block-concat
        # ``[up | gate]``.  Do not copy weights between the two without
        # permuting — see ``convert_wi_concat_to_interleaved``.
        inter_out = config.intermediate_size if self.srelu else 2 * config.intermediate_size
        self.Wi = nn.Parameter(
            torch.empty(self.num_experts, inter_out, config.hidden_size),
        )
        self.Wo = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size, config.intermediate_size),
        )
        self.drop = nn.Dropout(config.mlp_dropout)

        # Shared expert — a normal MLP that processes all tokens.
        self.shared_expert = (
            ModernBertSReluMLP if self.srelu else ModernBertSwiGLUMLP
        )(config)

        self.aux_loss_coef: float = float(config.moe_aux_loss_coef)
        self.z_loss_coef: float = float(config.moe_z_loss_coef)
        self.routed_scaling_factor: float = float(config.moe_routed_scaling_factor)
        # Routed-expert compute backend: 'cutlass' (device-side grouped GEMM,
        # default) or 'sonicmoe' (SonicMoE/quack fused expert path). The sonicmoe
        # path fuses dispatch+up+act+down+combine and cannot insert dropout
        # between the projections.
        self.moe_kernel_backend: str = str(getattr(config, "moe_kernel_backend", "cutlass"))
        if self.moe_kernel_backend == "sonicmoe" and float(config.mlp_dropout) != 0.0:
            raise ValueError("moe_kernel_backend='sonicmoe' does not support mlp_dropout > 0.")
        ckpt_enabled = bool(getattr(config, "activation_checkpointing", False))
        ckpt_mode = str(
            getattr(config, "activation_checkpointing_mode", "layer")
        ).strip().lower()
        # Recompute can revisit the MoE branch in full-layer checkpointing or
        # when the MLP branch itself is checkpointed. Attention-only
        # checkpointing does not re-enter MoE, so keep workspace reuse enabled
        # there to avoid the slower bincount fallback.
        #
        # Looping (Huginn recycling / Loopie layer-loop) also executes the same
        # MoELayer more than once per forward. `moe_grouped_gemm` stashes the
        # dispatch histogram via `ctx.save_for_backward(a, b, batch_sizes)`, and
        # with reuse enabled `batch_sizes` *is* the persistent `_dispatch_counts`
        # buffer — a second execution would mutate it in place and invalidate
        # the first execution's saved tensor.
        _looping = (
            int(getattr(config, "recycling_loop_count", 1) or 1) > 1
            or bool(getattr(config, "use_loopie", False))
        )
        self._reuse_dispatch_workspaces = not (
            (ckpt_enabled and ckpt_mode in {"layer", "attn+mlp"}) or _looping
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
        # `last_expert_counts` only exists for the expert-bias load-balancing
        # update in pure_pipeline.py, which is gated on moe_use_bias_correction.
        self._track_expert_counts: bool = bool(config.moe_use_bias_correction)
        # The cutlass path has to build `counts` for the grouped GEMM anyway, so
        # only the sonicmoe path needs the router to emit the histogram.
        self.router.track_expert_counts = (
            self._track_expert_counts and self.moe_kernel_backend == "sonicmoe"
        )
        # These accumulate across every execution of this layer within a single
        # ModernBertForMaskedLM.forward (one execution normally; `r` under
        # Huginn recycling; `loopie_loop_count` under Loopie), and are reset to
        # None at the top of that forward. `last_aux_count` is a 0-d *tensor*,
        # not a Python int: an int attribute read inside a compiled region gets
        # an EQUALS_MATCH value guard, which would force a recompile per
        # iteration index and per distinct loop count.
        self.last_expert_counts: Optional[torch.Tensor] = None
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_z_loss: Optional[torch.Tensor] = None
        self.last_aux_count: Optional[torch.Tensor] = None
        # Number of real (non-padding) rows in the packed batch, set once per
        # forward by ModernBertForMaskedLM. A 0-d tensor, not an int, so
        # torch.compile does not guard on its value and recompile per batch.
        self.valid_token_count: Optional[torch.Tensor] = None

        # Gate/up storage convention marker (persistent, so it lands in the
        # checkpoint). Only gated activations have a convention to get wrong;
        # sReLU's Wi is ungated, so it stays byte-compatible with every older
        # checkpoint and gets no buffer at all. 1-D on purpose: DCP's
        # `_distribute_state_dict` special-cases 0-dim tensors onto the CPU.
        # Expert storage convention marker, for every activation (the
        # transpose matters for sReLU too).
        self.register_buffer(
            _EXPERT_LAYOUT_BUFFER,
            torch.tensor([EXPERT_LAYOUT_VERSION], dtype=torch.int64),
            persistent=True,
        )
        if not self.srelu:
            self.register_buffer(
                _WI_LAYOUT_BUFFER,
                torch.tensor([WI_GATED_LAYOUT_VERSION], dtype=torch.int64),
                persistent=True,
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Refuse a state_dict whose ``Wi`` uses a different gate/up layout.

        A v1 (``[up | gate]`` block-concat) checkpoint is numerically valid but
        means something *different* under v2 code: gate and up get re-paired, so
        the model would silently resume as a different function and the loss
        would blow up (or, worse, slowly recover into a different optimum).
        Missing marker == v1, because v1 never wrote one.

        The message goes into ``error_msgs``, which ``nn.Module.load_state_dict``
        raises regardless of ``strict`` — this is a correctness mismatch, not a
        tolerable partial load. ``srelu`` is ungated and never checked.

        The expert *storage* convention (``EXPERT_LAYOUT_VERSION``) is checked
        the same way for every activation; a missing marker means version 1,
        ``(E, in, out)``.
        """
        key = prefix + _EXPERT_LAYOUT_BUFFER
        found = state_dict.get(key)
        if found is None:
            version = 1          # version 1 never wrote a marker
        else:
            try:
                version = int(found.reshape(-1)[0])
            except Exception:
                version = EXPERT_LAYOUT_VERSION
        if version != EXPERT_LAYOUT_VERSION:
            error_msgs.append(
                f"MoE routed-expert layout mismatch at {prefix}Wi/Wo: the checkpoint "
                f"was written with expert layout version {version}, this build expects "
                f"{EXPERT_LAYOUT_VERSION}. Version 1 stored experts as (E, in, out), "
                f"transposed against nn.Linear, which made Muon/NorMuon treat input "
                f"units as output neurons; version 2 stores (E, out, in). Migrate "
                f"the checkpoint (and its optimizer state when resuming) with "
                f"nanoplm.pretraining.models.modern_bert.moe."
                f"convert_expert_layout_v1_to_v2(state_dict, optimizer_state_dict)."
            )
        if found is None:
            state_dict[key] = getattr(self, _EXPERT_LAYOUT_BUFFER).clone()
        if not self.srelu:
            key = prefix + _WI_LAYOUT_BUFFER
            found = state_dict.get(key)
            if found is None:
                version = 1          # v1 never wrote a marker
            else:
                try:
                    version = int(found.reshape(-1)[0])
                except Exception:
                    # Present but not a plain readable tensor (some exotic
                    # loader). Presence alone proves it is not v1, which is the
                    # case that matters; do not block the load on the value.
                    version = WI_GATED_LAYOUT_VERSION
            if version != WI_GATED_LAYOUT_VERSION:
                error_msgs.append(
                    f"MoE routed-expert Wi layout mismatch at {prefix}Wi: the "
                    f"checkpoint was written with layout version {version}, this "
                    f"build expects {WI_GATED_LAYOUT_VERSION}. Version 1 stored "
                    f"Wi's last dim as block-concat [up | gate]; version 2 stores "
                    f"it interleaved gate-first [gate_0, up_0, gate_1, up_1, ...]. "
                    f"Loading it as-is would silently re-pair gate/up and train a "
                    f"different function. Migrate the checkpoint with "
                    f"nanoplm.pretraining.models.modern_bert.moe."
                    f"convert_wi_state_dict_v1_to_v2(state_dict)."
                )
            if found is None:
                # Supply the expected value so the generic machinery does not
                # *also* report a bare "Missing key(s)" and bury the message
                # above. If we reached here the load is already failing.
                state_dict[key] = getattr(self, _WI_LAYOUT_BUFFER).clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

        # The collator pads the packed batch to a fixed token budget with a
        # trailing dummy sequence, so the real tokens are always the prefix
        # [0, valid_token_count). Comparing against arange keeps the shape
        # static, which the MoE path requires.
        valid_mask = None
        if self.valid_token_count is not None:
            valid_mask = (
                torch.arange(T, device=x.device)
                < self.valid_token_count.to(x.device)
            ).unsqueeze(1)

        # --- shared expert (processes all tokens) ---
        shared_out = self.shared_expert(
            x,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
        )

        # --- routing ---
        # counts is the (E,) assignment histogram, produced for free by the
        # fused top-k kernel; None when nothing consumes it.
        weights, indices, z_loss, router_scores, router_counts = self.router(x)

        # --- dispatch metadata ---
        expert_flat = indices.reshape(-1)  # (T*top_k,)
        if self.moe_kernel_backend == "sonicmoe":
            # SonicMoE builds its own expert-grouped dispatch, so the only
            # consumer of `counts` on this path is `last_expert_counts`.
            counts = router_counts
        elif self._reuse_dispatch_workspaces:
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
            # zeros + scatter_add_, not `bincount`: bincount's output shape
            # depends on its data, which Dynamo can only handle with a graph
            # break, and under layer-mode activation checkpointing a break
            # inside the checkpoint HOP is fatal — Dynamo cannot resume from it
            # and the whole layer falls back to eager, which is what the
            # FSDP(AC(layer)) reorder is there to prevent. Same histogram as the
            # workspace-reuse branch above, with a static [num_experts] shape.
            # The src is an expanded scalar, so the ones cost no memory (the
            # reuse branch keeps a persistent `count_ones` buffer for the same
            # reason).
            _ones = torch.ones((), device=expert_flat.device, dtype=torch.int64)
            counts = torch.zeros(
                self.num_experts, device=expert_flat.device, dtype=torch.int64
            ).scatter_add_(0, expert_flat, _ones.expand_as(expert_flat))

        # Store for external bias-update logic. The pure-torch loop accumulates
        # this every micro-step (pure_pipeline.py `_moe_count_accum`), so it
        # cannot be gated on the bias-update cadence without changing the
        # load-balancing semantics — but it is pure overhead when bias
        # correction is off, in which case nothing reads it at all.
        if self._track_expert_counts and counts is not None:
            # Sum over executions within this forward (see __init__). The clone
            # stays mandatory: on the cutlass path `counts` may be a reused
            # workspace buffer.
            _new_counts = counts.detach().clone()
            if valid_mask is not None:
                # `counts` itself must keep the padding rows: it doubles as the
                # group sizes for moe_grouped_gemm below. Only the copy the
                # bias-correction loop and the telemetry read gets corrected.
                _pad_w = (~valid_mask).to(torch.int64).expand(-1, self.top_k)
                _pad_counts = torch.zeros_like(_new_counts).scatter_add_(
                    0, expert_flat, _pad_w.reshape(-1),
                )
                _new_counts = _new_counts - _pad_counts
            self.last_expert_counts = (
                _new_counts
                if self.last_expert_counts is None
                else self.last_expert_counts + _new_counts
            )

        if self.moe_kernel_backend == "sonicmoe":
            # SonicMoE/quack fused expert path: dispatch gather is fused into the
            # up-projection GEMM, and the down-projection scatters/combines with
            # the router weights — all driven by nanoplm's routing decisions.
            from .moe_sonicmoe import sonicmoe_routed_experts
            # quack's SM90 GEMM takes 16- or 8-bit operands of a single dtype.
            # Training's bf16 autocast already hands the residual stream over
            # that way, but a custom op is invisible to autocast, so an fp32
            # caller reaches the kernel unconverted and trips its dtype check.
            # Both casts are no-ops when the dtypes already agree.
            expert_out = sonicmoe_routed_experts(
                x.to(self.Wi.dtype), indices, weights,
                self.Wi, self.Wo, self.num_experts, self.top_k,
                self.srelu, self.routed_scaling_factor,
            ).to(x.dtype)  # (T, H)
        else:
            # --- expert MLP via cutlass grouped_gemm ---
            token_idx = (sorted_idx // self.top_k).to(torch.int32)
            slot_idx = (sorted_idx % self.top_k).to(torch.int32)
            x_sorted = moe_scatter_dispatch(x, token_idx)  # (T*top_k, H)
            # Wi is (E, out, in): trans_b computes x @ Wi[e].T per expert.
            wi = moe_grouped_gemm(x_sorted, self.Wi, counts, trans_b=True)  # (T*top_k, inter or 2*inter)
            if self.srelu:
                h = F.relu(wi)
                activated = h * h  # (T*top_k, inter)
            else:
                # Wi's columns are interleaved gate-first (WI_GATED_LAYOUT),
                # so unit j is the pair (2j, 2j+1) = (gate_j, up_j). Same
                # function as the old `[up | gate]` chunk, different storage.
                gate, x_proj = wi[..., 0::2], wi[..., 1::2]
                activated = F.silu(gate) * x_proj  # (T*top_k, inter)
            expert_out_sorted = moe_grouped_gemm(
                self.drop(activated),
                self.Wo,
                counts,
                trans_b=True,
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
            if valid_mask is None:
                mean_mass = router_mass.mean(dim=0)
            else:
                _m = valid_mask.to(router_mass.dtype)
                mean_mass = (router_mass * _m).sum(dim=0) / _m.sum().clamp_min(1.0)
            target = torch.ones_like(mean_mass) / self.num_experts
            aux = aux + self.aux_loss_coef * ((mean_mass - target) ** 2).sum() * self.num_experts
        # Z-loss: prevents sigmoid saturation in the router.
        if self.z_loss_coef > 0:
            aux = aux + self.z_loss_coef * z_loss
        # Sum aux (and count executions) across every execution of this layer
        # within one forward; the reader takes the per-execution mean so the
        # loss contribution stays on the same scale as a single-execution run.
        # Kept graph-connected -- this is the only gradient channel into the
        # router gate, and summing across recurrent iterations is what lets
        # every iteration's routing decision train the gate.
        aux = aux.squeeze()
        z_detached = z_loss.detach()
        if self.last_aux_loss is None:
            self.last_aux_loss = aux
            self.last_z_loss = z_detached
            self.last_aux_count = torch.ones((), device=aux.device, dtype=aux.dtype)
        else:
            self.last_aux_loss = self.last_aux_loss + aux
            self.last_z_loss = self.last_z_loss + z_detached
            self.last_aux_count = self.last_aux_count + 1

        out = expert_out + shared_out
        if len(orig_shape) == 3:
            out = out.view(orig_shape)
        return out
