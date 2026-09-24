"""Identity modules that accumulate debug statistics inside the graph.

``nanoplm.pretraining.debug_metrics.LayerwiseDebugTracker`` installs an fp32
accumulator on each tap with ``enable_stats`` and drops it again with
``disable_stats``; until then a tap is a plain identity. The accumulators are
plain attributes — not parameters, not registered buffers — so they add no
state_dict keys and no FSDP2 sharding surface, and the arithmetic runs inside
the compiled graph as an in-place mutation of a lifted input (the forward hooks
this replaced had to be ``torch.compiler.disable``d, which made Dynamo skip the
whole compiled frame).

This module exists so that ``canon_layer`` and ``modeling`` can both use the
taps: ``modeling`` imports ``canon_layer``, so the taps cannot live there.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepeatTap(nn.Module):
    """Identity that can accumulate per-loop-application residual-stream stats.

    Pure identity until ``nanoplm.pretraining.debug_metrics`` installs a
    ``[2, max_rep]`` fp32 accumulator via :meth:`enable_stats`: row 0 sums
    ``mean(x*x)`` per application, row 1 counts the applications.
    """

    stats: Optional[torch.Tensor] = None

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

    def forward(self, x: torch.Tensor, rep: int) -> torch.Tensor:
        stats = self.stats
        # `rep` is a Python int and `training`/`stats is None` are Dynamo
        # guards, so this whole branch is a trace-time constant per graph.
        if stats is not None and self.training and rep < stats.shape[1]:
            with torch.no_grad():
                xd = x.detach()
                # fp32 accumulation without an fp32 copy of x: vector_norm's
                # `dtype` upcasts inside the reduction. norm^2 / numel is
                # mean(x*x); numel() may be a SymInt under dynamic shapes.
                sq = torch.linalg.vector_norm(xd, dtype=torch.float32).square()
                stats[0, rep].add_(sq / xd.numel())
                stats[1, rep].add_(1.0)
        return x


class StatsTap(nn.Module):
    """Identity that accumulates activation stats in place, inside the graph.

    ``stats`` is ``None`` until ``nanoplm.pretraining.debug_metrics`` installs a
    6-element fp32 accumulator via :meth:`enable_stats`, laid out as
    ``[sq_sum, sum, abs_max, numel, entropy_sum, token_count]`` (slot 1 stays
    zero unless ``with_sum``, the last two unless ``with_entropy``).
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.stats
        if stats is not None and self.training:
            with torch.no_grad():
                xd = x.detach()
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
