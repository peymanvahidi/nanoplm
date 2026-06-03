"""Regression test: recycling logging must not issue rank-asymmetric collectives.

Under FSDP2 the recycling adapter weight is a sharded ``DTensor``. The
per-logging-steps payload is built on rank 0 only (the caller gates it behind
``if should_log and is_main:``), so any reduction over the *full* DTensor
(e.g. ``.mean()``) enqueues a collective that the other ranks never join. NCCL
matches collectives by per-rank sequence number, so that single extra op shifts
rank 0's stream by one; every later collective is then matched to the wrong op
(ALLREDUCE on rank 0 vs BROADCAST on the others), ending in a watchdog timeout
one step after the first logged step.

``logging_steps_payload`` must therefore read the *local shard* of the adapter
weight (collective-free) rather than operate on the full DTensor.
"""

import pytest
import torch

from nanoplm.pretraining.recycling.logging import (
    RecyclingMetricsBuffer,
    logging_steps_payload,
)


class _FakeShardedTensor:
    """Stand-in for a sharded ``DTensor``.

    Implements only the contract the logging code is allowed to use:
    ``.detach()`` returns another sharded view and ``.to_local()`` returns the
    local shard without communication. It deliberately does NOT implement the
    elementwise / reduction ops (``.float()``, ``.pow()``, ``.mean()``), so any
    code path that operates on the *full* tensor instead of the local shard
    raises ``AttributeError`` — mirroring the illegal rank-0-only collective on
    real hardware.
    """

    def __init__(self, local: torch.Tensor, grad_local: torch.Tensor | None = None):
        self._local = local
        self._grad = _FakeShardedTensor(grad_local) if grad_local is not None else None

    def detach(self):
        return self

    def to_local(self) -> torch.Tensor:
        return self._local

    @property
    def grad(self):
        return self._grad


class _FakeAdapter:
    def __init__(self, weight):
        self.weight = weight


def test_logging_payload_reads_local_shard_of_sharded_weight():
    """A sharded (DTensor-like) weight must be reduced on its local shard only.

    On the broken code (``adapter.weight.detach().float().pow(2).mean()``) the
    reduction runs over the full DTensor — a rank-0-only collective. Here the
    fake raises on any full-tensor op, so the broken code errors; the fix
    reads ``to_local()`` first and computes the RMS on the plain local shard.
    """
    torch.manual_seed(0)
    local = torch.randn(256, 2048)  # one rank's shard of a [1024, 2048] adapter
    adapter = _FakeAdapter(_FakeShardedTensor(local))

    payload = logging_steps_payload(adapter, RecyclingMetricsBuffer())

    expected = local.float().pow(2).mean().sqrt().item()
    assert payload["recycling/diagnostics/adapter_w_rms"] == pytest.approx(expected)
    # grad is None (zero_grad ran before logging) -> no grad key.
    assert "recycling/diagnostics/adapter_w_grad_rms" not in payload


def test_logging_payload_reads_local_shard_of_sharded_grad():
    """When a grad is present it must also be reduced on the local shard."""
    torch.manual_seed(0)
    local = torch.randn(256, 2048)
    grad_local = torch.randn(256, 2048)
    adapter = _FakeAdapter(_FakeShardedTensor(local, grad_local=grad_local))

    payload = logging_steps_payload(adapter, RecyclingMetricsBuffer())

    assert payload["recycling/diagnostics/adapter_w_grad_rms"] == pytest.approx(
        grad_local.float().pow(2).mean().sqrt().item()
    )


def test_logging_payload_plain_tensor_unchanged():
    """DDP / single-GPU path: a plain replicated tensor has no ``to_local`` and
    must pass through unchanged (full, exact RMS, still collective-free)."""
    torch.manual_seed(0)
    w = torch.randn(64, 128, requires_grad=True)
    w.sum().backward()  # populate w.grad
    adapter = _FakeAdapter(w)

    payload = logging_steps_payload(adapter, RecyclingMetricsBuffer())

    assert payload["recycling/diagnostics/adapter_w_rms"] == pytest.approx(
        w.detach().float().pow(2).mean().sqrt().item()
    )
    assert payload["recycling/diagnostics/adapter_w_grad_rms"] == pytest.approx(
        w.grad.detach().float().pow(2).mean().sqrt().item()
    )


def test_logging_payload_no_adapter():
    """No adapter (recycling off / adapter missing) -> no adapter keys, no error."""
    payload = logging_steps_payload(None, RecyclingMetricsBuffer())
    assert "recycling/diagnostics/adapter_w_rms" not in payload
