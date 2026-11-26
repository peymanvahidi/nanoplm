"""Identity modules that accumulate debug statistics inside the graph.

``nanoplm.pretraining.debug_metrics.LayerwiseDebugTracker`` installs an fp32
accumulator on each tap with ``enable_stats`` and drops it again with
``disable_stats``; until then a tap is a plain identity. The accumulators are
plain attributes — not parameters, not registered buffers — so they add no
state_dict keys and no FSDP2 sharding surface, and the arithmetic runs inside
the compiled graph as an in-place mutation of a lifted input (the forward hooks
this replaced had to be ``torch.compiler.disable``d, which made Dynamo skip the
whole compiled frame).

Stats cover real tokens only. The packed collator fills each micro-batch to
its token budget with a trailing dummy pad sequence, and padded batches carry
pad positions; neither is trained, and the pad segment's residual grows far
larger than the real tokens' (hundreds of times by mid-training), so a plain
mean over every row reports the pad, not the model. A tap therefore takes a
row mask from one of two places: ``token_mask`` passed at the call (the
padded path, where the layers already receive the attention mask), or
``valid_token_count``, which ``ModernBertForMaskedLM.forward`` sets on the
full-stream taps from ``num_valid_tokens``, the same way ``MoELayer`` gets it
(the packed path, where the real tokens are the prefix ``[0, n)``).

This module exists so that ``canon_layer`` and ``modeling`` can both use the
taps: ``modeling`` imports ``canon_layer``, so the taps cannot live there.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_mask(
    x: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    valid_token_count: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """fp32 0/1 weight per token row of ``x``, or ``None`` when every row counts.

    ``token_mask`` (nonzero = real token) must match ``x``'s leading dims, e.g.
    ``(B, S)`` for ``(B, S, C)`` or ``(B, S, n, C)``. Otherwise
    ``valid_token_count`` marks rows ``[0, n)`` along dim 0 of a packed ``x``
    (``(T, C)`` or ``(T, n, C)``). A row is everything after the token dims.
    """
    if token_mask is not None:
        if tuple(x.shape[: token_mask.dim()]) != tuple(token_mask.shape):
            raise ValueError(
                f"tap token_mask {tuple(token_mask.shape)} does not match the "
                f"leading dims of x {tuple(x.shape)}"
            )
        return token_mask.reshape(-1).ne(0).to(torch.float32)
    if valid_token_count is not None:
        rows = torch.arange(x.shape[0], device=x.device)
        return (rows < valid_token_count.to(x.device)).to(torch.float32)
    return None


class RepeatTap(nn.Module):
    """Identity that can accumulate per-loop-application residual-stream stats.

    Pure identity until ``nanoplm.pretraining.debug_metrics`` installs a
    ``[2, max_rep]`` fp32 accumulator via :meth:`enable_stats`: row 0 sums
    ``mean(x*x)`` over the real tokens per application, row 1 counts the
    applications (see the module docstring for the mask).
    """

    stats: Optional[torch.Tensor] = None
    # Set per forward by ModernBertForMaskedLM on the packed path; see _row_mask.
    valid_token_count: Optional[torch.Tensor] = None

    def enable_stats(self, max_rep: int, device: torch.device) -> torch.Tensor:
        """Install (and return) the accumulator this tap adds into."""
        self.stats = torch.zeros(
            2, max(1, int(max_rep)), device=device, dtype=torch.float32
        )
        # Inductor's cudagraph path skips any graph that mutates a non-static
        # input; the address is stable because `flush()` zeroes in place.
        torch._dynamo.mark_static_address(self.stats)
        return self.stats

    def disable_stats(self) -> None:
        self.stats = None

    def forward(
        self, x: torch.Tensor, rep: int, token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        stats = self.stats
        # `rep` is a Python int and `training`/`stats is None` are Dynamo
        # guards, so this whole branch is a trace-time constant per graph.
        if stats is not None and self.training and rep < stats.shape[1]:
            with torch.no_grad():
                xd = x.detach()
                # fp32 accumulation without an fp32 copy of x: vector_norm's
                # `dtype` upcasts inside the reduction. norm^2 / numel is
                # mean(x*x); numel() may be a SymInt under dynamic shapes.
                m = _row_mask(xd, token_mask, self.valid_token_count)
                if m is None:
                    sq = torch.linalg.vector_norm(xd, dtype=torch.float32).square()
                    stats[0, rep].add_(sq / xd.numel())
                    stats[1, rep].add_(1.0)
                else:
                    # Per-row squared norms (fp32 inside the reduction, no fp32
                    # copy of x), weighted by the mask. An application with no
                    # real token adds nothing and is not counted.
                    xf = xd.reshape(m.numel(), -1)
                    row_sq = torch.linalg.vector_norm(
                        xf, dim=-1, dtype=torch.float32
                    ).square()
                    n_real = m.sum()
                    stats[0, rep].add_(
                        (row_sq * m).sum() / (n_real * xf.shape[-1]).clamp_min(1.0)
                    )
                    stats[1, rep].add_((n_real > 0).to(torch.float32))
        return x


class StatsTap(nn.Module):
    """Identity that accumulates activation stats in place, inside the graph.

    ``stats`` is ``None`` until ``nanoplm.pretraining.debug_metrics`` installs a
    6-element fp32 accumulator via :meth:`enable_stats`, laid out as
    ``[sq_sum, sum, abs_max, numel, entropy_sum, token_count]`` (slot 1 stays
    zero unless ``with_sum``, the last two unless ``with_entropy``). Only real
    tokens count (see the module docstring for the mask).
    """

    # Set per forward by ModernBertForMaskedLM on the packed path; see _row_mask.
    valid_token_count: Optional[torch.Tensor] = None

    def __init__(self, with_entropy: bool = False, with_sum: bool = True) -> None:
        super().__init__()
        self.with_entropy = bool(with_entropy)
        # The logit tap's mean is never reported, and summing [tokens, vocab]
        # is not free; skip the reduction when nobody reads slot 1.
        self.with_sum = bool(with_sum)
        self.stats: Optional[torch.Tensor] = None

    def enable_stats(self, device: torch.device) -> torch.Tensor:
        """Install (and return) the accumulator this tap adds into."""
        self.stats = torch.zeros(6, device=device, dtype=torch.float32)
        # See RepeatTap.enable_stats: keeps the mutation cudagraph-capturable.
        torch._dynamo.mark_static_address(self.stats)
        return self.stats

    def disable_stats(self) -> None:
        self.stats = None

    def forward(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        stats = self.stats
        if stats is not None and self.training:
            with torch.no_grad():
                xd = x.detach()
                m = _row_mask(xd, token_mask, self.valid_token_count)
                if m is not None:
                    self._accumulate_masked(stats, xd, m)
                    return x
                stats[0].add_(
                    torch.linalg.vector_norm(xd, dtype=torch.float32).square()
                )
                if self.with_sum:
                    stats[1].add_(xd.sum(dtype=torch.float32))
                stats[2].copy_(torch.maximum(stats[2], xd.abs().amax().float()))
                stats[3].add_(xd.numel())
                if self.with_entropy:
                    # Entropy via the logsumexp identity: H = lse - Σ p * x.
                    # lse and softmax run in x's dtype (bf16 is fine over this
                    # range); softmax and p*x are each as large as x, the two
                    # per-token reductions are not.
                    lse = torch.logsumexp(xd, dim=-1)
                    probs = F.softmax(xd, dim=-1)
                    entropy_per_token = lse - (probs * xd).sum(dim=-1)
                    stats[4].add_(entropy_per_token.sum(dtype=torch.float32))
                    stats[5].add_(xd.numel() // xd.shape[-1])
        return x

    def _accumulate_masked(
        self, stats: torch.Tensor, xd: torch.Tensor, m: torch.Tensor
    ) -> None:
        """The unmasked arithmetic above, restricted to rows where ``m`` is 1."""
        xf = xd.reshape(m.numel(), -1)
        stats[0].add_(
            (torch.linalg.vector_norm(xf, dim=-1, dtype=torch.float32).square() * m).sum()
        )
        if self.with_sum:
            stats[1].add_((xf.sum(dim=-1, dtype=torch.float32) * m).sum())
        # |x| >= 0, so zeroing the pad rows cannot raise the max.
        row_max = xf.abs().amax(dim=-1).float()
        stats[2].copy_(torch.maximum(stats[2], (row_max * m).amax()))
        n_real = m.sum()
        stats[3].add_(n_real * xf.shape[-1])
        if self.with_entropy:
            # Rows are tokens and the last dim is the vocab (logits).
            lse = torch.logsumexp(xf, dim=-1)
            probs = F.softmax(xf, dim=-1)
            entropy_per_token = lse - (probs * xf).sum(dim=-1)
            stats[4].add_((entropy_per_token.float() * m).sum())
            stats[5].add_(n_real)
