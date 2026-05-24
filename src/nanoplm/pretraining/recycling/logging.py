"""wandb logging helpers for Huginn-style recycling.

Per spec §6:
  - recycling/training:    sampled_r (per step), curriculum_k_t (per step),
                           r_distribution (Histogram, per logging_steps)
  - recycling/diagnostics: adapter_w_rms, adapter_w_grad_rms (per logging_steps),
                           s_convergence_at_r, s_norm_final (per eval_steps)
  - recycling/eval_inloop: per-r in-loop eval losses (per eval_steps)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore


class RecyclingMetricsBuffer:
    """Buffers per-step `sampled_r` values for the periodic histogram."""

    def __init__(self, window: int = 100):
        self.r_buffer: deque[int] = deque(maxlen=window)

    def record_step(self, sampled_r: int) -> None:
        self.r_buffer.append(int(sampled_r))

    def histogram(self):
        if wandb is None or len(self.r_buffer) == 0:
            return None
        return wandb.Histogram(list(self.r_buffer))


def per_step_payload(sampled_r: int, curriculum_k_t: int) -> dict:
    """Per-training-step recycling scalars."""
    return {
        "recycling/training/sampled_r": int(sampled_r),
        "recycling/training/curriculum_k_t": int(curriculum_k_t),
    }


def logging_steps_payload(adapter: Optional[torch.nn.Module],
                          buffer: RecyclingMetricsBuffer) -> dict:
    """Per-logging-steps payload: adapter RMS + grad RMS + r histogram."""
    payload: dict = {}
    if adapter is not None:
        w_rms = adapter.weight.detach().float().pow(2).mean().sqrt().item()
        payload["recycling/diagnostics/adapter_w_rms"] = w_rms
        if adapter.weight.grad is not None:
            g_rms = adapter.weight.grad.detach().float().pow(2).mean().sqrt().item()
            payload["recycling/diagnostics/adapter_w_grad_rms"] = g_rms
    hist = buffer.histogram()
    if hist is not None:
        payload["recycling/training/r_distribution"] = hist
    return payload


def inloop_eval_payload(losses_by_r: dict) -> dict:
    """In-loop eval sweep payload: per-r scalar + line_series."""
    if wandb is None:
        return {}
    rs = sorted(losses_by_r.keys())
    losses = [losses_by_r[r] for r in rs]
    payload: dict = {
        "recycling/eval_inloop/inloop_eval_loss_vs_r": wandb.plot.line_series(
            xs=rs, ys=[losses],
            keys=["eval_loss"], title="In-loop eval loss vs r",
            xname="loop_count r",
        )
    }
    for r, loss in losses_by_r.items():
        payload[f"recycling/eval_inloop/inloop_eval_loss_at_r{r}"] = loss
    return payload
