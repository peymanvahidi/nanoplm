"""Transformer Engine CPU activation offload for the pure-torch FP8 pipeline.

We offload a chosen subset of each selected TE module's backward-saved
activations (see ``_OFFLOAD_KEEP_INDICES``) to pinned CPU memory during forward,
and stream them back during backward -- trading copy bandwidth for GPU memory.

The heavy lifting lives in TE's manual offload controller
(``get_cpu_offload_context(manual_synchronization=True)``). This module only
adds two small things on top:

1. A filter so each module offloads only the chosen saved tensors instead of
   the full activation set -- see ``_install_offload_filter``.
2. A 1-step-ahead schedule so the device<->host copies overlap compute instead
   of stalling it -- see ``_OffloadController``.

Wiring (see ``install``):

* every target module gets a ``_cpu_offload`` handle and routes its forward
  through ``run_target``;
* the encoder gets a ``_cpu_offload_flush`` callable that runs after its final
  activation to release/prefetch the last target.

``run_target`` and ``run_flush`` are wrapped with ``torch.compiler.disable`` so
TorchInductor never traces TE's Python offload scheduler (which fails with
``NameError: default_stream``); the rest of the model stays compiled.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch

# TE module files whose ``mark_activation_offload`` we replace with the filter.
_TE_MODULE_NAMES = (
    "transformer_engine.pytorch.module.layernorm_linear",
    "transformer_engine.pytorch.module.layernorm_mlp",
)


def _require_te():
    try:
        import transformer_engine.pytorch as te
    except Exception as exc:  # pragma: no cover - depends on runtime image
        raise ImportError(
            "FP8 CPU activation offload requires Transformer Engine."
        ) from exc
    return te


# ---------------------------------------------------------------------------
# Saved-tensor offload filter
# ---------------------------------------------------------------------------

# Indices (into TE's ``mark_activation_offload`` arg list) of the saved tensors
# we offload. Both LayerNormLinear and LayerNormMLP pass ``inputmat`` at 0 and
# the fp8 norm output (``ln_out`` / ``ln_out_to_save``) at 3; mu/rsigma (1, 2)
# are below TE's 256K-element offload threshold regardless. The larger MLP
# intermediates (fc1_out / act_out at 4-6) are intentionally kept resident:
# they would blow the D2H copy budget on the 4xGH200 SKU (170 GB/s D2H).
_OFFLOAD_KEEP_INDICES = frozenset({0, 3})

_filter_installed = False


@torch.compiler.disable
def _offload_filter(*tensors) -> None:
    """Drop-in replacement for TE's ``mark_activation_offload``.

    TE calls this inside each module forward with that module's backward-saved
    tensors in a fixed order. In TE's v2 offload path the original is a no-op and
    *every* eligible saved tensor is offloaded, which saturates the copy engines.
    We instead tag everything outside ``_OFFLOAD_KEEP_INDICES`` as do-not-offload
    so only the chosen tensors leave the GPU.
    """
    blocked = tuple(
        t for i, t in enumerate(tensors) if i not in _OFFLOAD_KEEP_INDICES and t is not None
    )
    if blocked:
        _require_te().mark_not_offload(*blocked)


def _install_offload_filter() -> None:
    """Patch TE once. Safe because TE only calls it inside an active context."""
    global _filter_installed
    if _filter_installed:
        return
    for name in _TE_MODULE_NAMES:
        importlib.import_module(name).mark_activation_offload = _offload_filter
    _filter_installed = True


# ---------------------------------------------------------------------------
# Manual offload scheduling (1 step ahead)
# ---------------------------------------------------------------------------


class _OffloadController:
    """Drives TE's manual controller for an ordered list of target modules.

    Forward: after target ``i`` runs we start its D2H copy and free target
    ``i-1``'s forward memory (its copy has had a full step to finish). Backward:
    when target ``i+1``'s grad arrives we start reloading target ``i``. The
    encoder ``flush`` plays the role of "target n" for the last real target.
    """

    def __init__(self, context, manual_controller, num_targets: int):
        self._ctx = context
        self._ctrl = manual_controller
        self._n = num_targets

    def run_target(self, layer_id: int, fn, x):
        self._ctx.__enter__()
        try:
            out = fn(x)
        finally:
            self._ctx.__exit__(None, None, None)
        self._ctrl.start_offload_layer(layer_id)
        if layer_id > 0:
            self._ctrl.release_activation_forward_gpu_memory(layer_id - 1)
        return self._arm_reload(out, bwd_layer=layer_id)

    def flush(self, x):
        # Trailing dummy step at the encoder output: enter/exit once so TE's
        # forward counter advances to n and its per-step reset fires; then
        # release the last target and arm its reload for the start of backward.
        self._ctx.__enter__()
        self._ctx.__exit__(None, None, None)
        if self._n > 0:
            self._ctrl.release_activation_forward_gpu_memory(self._n - 1)
        return self._arm_reload(x, bwd_layer=self._n)

    def _arm_reload(self, tensor, *, bwd_layer: int):
        if not tensor.requires_grad or tensor.grad_fn is None:
            return tensor
        reload_layer = bwd_layer - 1

        def prehook(_):
            # Release this layer's backward memory once its grad is computed.
            engine = torch.autograd.variable.Variable._execution_engine
            engine.queue_callback(self._ctrl.finish_part_of_bwd)
            self._ctrl.bwd_step(bwd_layer)
            if reload_layer >= 0:
                self._ctrl.start_reload_layer(reload_layer)

        tensor.grad_fn.register_prehook(prehook)
        return tensor


@dataclass
class _TargetHandle:
    """Stored on each target module as ``_cpu_offload``."""

    controller: _OffloadController
    layer_id: int

    def run(self, fn, x):
        return self.controller.run_target(self.layer_id, fn, x)


# ---------------------------------------------------------------------------
# Entry points called from module forwards (kept out of the compiled graph)
# ---------------------------------------------------------------------------


@torch.compiler.disable
def run_target(handle, fn, x):
    """Run ``fn(x)`` with offload scheduling, or plainly when offload is off."""
    if handle is None:
        return fn(x)
    return handle.run(fn, x)


@torch.compiler.disable
def run_flush(flush, x):
    """Release/prefetch the final target after the encoder's last activation."""
    return x if flush is None else flush(x)


# ---------------------------------------------------------------------------
# Install / disable
# ---------------------------------------------------------------------------


def install(model, targets, *, retain_pinned_buffers: bool = False) -> int:
    """Wire CPU activation offload onto ``targets``.

    ``targets`` are the TE wrapper modules whose ``inputmat`` is offloaded, in
    forward-execution order. ``model.model`` is the encoder that emits the final
    activation used to flush the last target. Returns the number of targets.
    """
    te = _require_te()
    n = len(targets)
    # model_layers == n + 1 reserves a trailing scheduler slot for ``flush``.
    context, _sync, controller = te.get_cpu_offload_context(
        enabled=True,
        num_layers=n,
        model_layers=n + 1,
        offload_activations=True,
        offload_weights=False,
        retain_pinned_cpu_buffers=retain_pinned_buffers,
        manual_synchronization=True,
    )
    if controller is None:
        raise RuntimeError("Transformer Engine did not return a manual offload controller.")

    _install_offload_filter()
    ctrl = _OffloadController(context, controller, n)
    for layer_id, target in enumerate(targets):
        target._cpu_offload = _TargetHandle(ctrl, layer_id)
    model.model._cpu_offload_flush = ctrl.flush
    return n


def disable(model) -> None:
    """Detach any offload handles (no-op on a fresh model)."""
    for module in model.modules():
        if hasattr(module, "_cpu_offload"):
            module._cpu_offload = None
        if hasattr(module, "_cpu_offload_flush"):
            module._cpu_offload_flush = None


__all__ = ["disable", "install", "run_flush", "run_target"]
