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


def _local_shard(tensor: torch.Tensor) -> torch.Tensor:
    """Return a parameter's *local* shard without communication.

    Under FSDP2 the adapter weight is a sharded ``DTensor``. Reducing over the
    full DTensor (``.mean()``) issues a collective. This payload is built on
    rank 0 only (the caller gates it behind ``if should_log and is_main:``), so
    that collective has no peers on the other ranks: it shifts rank 0's NCCL
    sequence by one and every later collective is then matched to the wrong op
    (ALLREDUCE vs BROADCAST), ending in a watchdog timeout one step after the
    first logged step. Reading ``to_local()`` keeps the diagnostic rank-local
    and collective-free (plain tensors — DDP / single-GPU — have no ``to_local``
    and pass through unchanged).
    """
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def logging_steps_payload(adapter: Optional[torch.nn.Module],
                          buffer: RecyclingMetricsBuffer) -> dict:
    """Per-logging-steps payload: adapter RMS + grad RMS + r histogram.

    The adapter weight is an FSDP2-sharded ``DTensor``; we read its local shard
    (see ``_local_shard``) so this stays collective-free — computing the RMS
    over the full DTensor here would desync NCCL across ranks. The local-shard
    RMS tracks the same scale as the global RMS for this diagnostic.
    """
    payload: dict = {}
    if adapter is not None:
        w = _local_shard(adapter.weight.detach())
        payload["recycling/diagnostics/adapter_w_rms"] = w.float().pow(2).mean().sqrt().item()
        grad = adapter.weight.grad
        if grad is not None:
            g = _local_shard(grad.detach())
            payload["recycling/diagnostics/adapter_w_grad_rms"] = g.float().pow(2).mean().sqrt().item()
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
