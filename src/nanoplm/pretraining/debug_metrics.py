"""Layerwise debug metrics tracker.

Gathers per-layer grad/weight norms, per-application residual-stream RMS, and
lm_head logit stats with a single host-sync per logging window.

Design invariants
-----------------
* Activation stats are computed by the model's own tap modules
  (``modern_bert.taps.RepeatTap`` / ``StatsTap``), not by forward hooks: this
  tracker only allocates the fp32 accumulators and installs them on the taps
  with ``enable_stats``. Hooks used to be ``@torch.compiler.disable``d, which
  made Dynamo skip the whole compiled frame and ran the model largely eager;
  the taps mutate a lifted graph input instead, so nothing breaks the graph.
* All accumulation uses in-place ops on preallocated GPU tensors. No `.item()`,
  `.cpu()`, or `.tolist()` outside `flush()`.
* `flush()` is the only host sync point: one `all_reduce` per reduction op, one
  `.cpu()`, one Python fan-out into a payload dict.
* The taps accumulate on every training step, not only on log steps — the cost
  is a handful of fused reductions per block, negligible against the block
  itself, and `flush()` divides by the counts it collected.
* One tensor per tap (never views into one big buffer), so AOTAutograd never
  sees several mutated inputs aliasing the same storage.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from nanoplm.pretraining.models.modern_bert.taps import RepeatTap, StatsTap
from nanoplm.utils.logger import logger


def _is_sharded(t: torch.Tensor) -> bool:
    """True if ``t`` is a DTensor with at least one Shard placement.

    Replicated DTensors and plain Tensors return False — every rank already
    holds the full data, so SUM-reducing local squared sums would overcount
    by world_size.
    """
    placements = getattr(t, "placements", None)
    if placements is None:
        return False
    for p in placements:
        if getattr(p, "is_shard", None) is not None and p.is_shard():
            return True
    return False


_BLOCK_CLASS_NAMES = (
    "ModernBertCanonLayer",
    "ModernBertEncoderLayer",
    "MHCLiteBlock",
    "MHCLiteSublayersLayer",
)


def _role_from_name(name: str) -> str:
    """Coarse role bucket used for W&B namespacing.  Falls back to raw name."""
    lower = name.lower()
    parts = lower.split(".")
    if "embedding" in lower or "tok_embed" in lower:
        return "embed"
    if parts[0] == "decoder" or lower.startswith("decoder."):
        return "decoder"
    if parts[0] == "head" or lower.startswith("head."):
        return "head"

    # Path-aware: check whether the parent module is .attn. or .mlp.
    in_attn = ".attn." in lower
    in_mlp = ".mlp." in lower

    if in_attn:
        if "qkv" in lower:
            return "attn.qkv"
        if ".wq" in lower or "q_proj" in lower:
            return "attn.q"
        if ".wk" in lower or "k_proj" in lower:
            return "attn.k"
        if ".wv" in lower or "v_proj" in lower:
            return "attn.v"
        if ".wo" in lower or "o_proj" in lower or "out_proj" in lower:
            return "attn.o"
        return "attn.other"

    if in_mlp:
        # Fused SwiGLU gate+up (single Wi) vs. separate up/gate vs. down.
        if ".wi" in lower or "up_gate" in lower:
            return "mlp.up_gate"
        if "up_proj" in lower or ".up." in lower or "_up" in lower:
            return "mlp.up"
        if "gate_proj" in lower or ".gate." in lower:
            return "mlp.gate"
        if ".wo" in lower or "down_proj" in lower or ".down." in lower or "_down" in lower:
            return "mlp.down"
        return "mlp.other"

    if "norm" in lower:
        return "norm"
    return "other"


class LayerwiseDebugTracker:
    """Owns the debug accumulators and the single per-window reduction.

    Construction installs fp32 accumulators on the model's taps: one
    ``[2, max_rep]`` tensor per enumerated block (``RepeatTap``), one
    6-element tensor for the logits and one for the final-norm output
    (``StatsTap``). Install after ``model.to(device)``, ``fully_shard`` and
    ``torch.compile`` but before the first forward — the tensors are plain
    module attributes, so they are lifted as graph inputs on the first trace
    and must already live on the training device.

    ``record_step_boundary`` fills the grad/weight slots once per optimizer
    step; ``flush`` reduces everything, resets the accumulators in place and
    returns the payload on rank 0. ``close`` removes the accumulators again,
    which turns every tap back into a plain identity.
    """

    def __init__(
        self,
        model: nn.Module,
        log_every: int,
        device: torch.device,
        output_dir: str,
        is_main: bool,
        distributed: bool,
        max_rep: int = 1,
    ) -> None:
        self.log_every = int(log_every)
        self.device = device
        self.is_main = is_main
        self.distributed = distributed
        self.max_rep = max(1, int(max_rep))

        # Taps whose accumulators this tracker installed; cleared by `close`.
        self._taps: list[nn.Module] = []

        # ---- param list (grad + weight norms) -------------------------------
        params: list[tuple[str, nn.Parameter]] = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]
        self.param_names: list[str] = [n for n, _ in params]
        self._params: list[nn.Parameter] = [p for _, p in params]
        n_params = len(self._params)
        # These hold per-param LOCAL-SHARD squared sums after
        # record_step_boundary; the global squared norm is recovered by the
        # SUM all_reduce in flush(). Stored squared so flush can pack with the
        # other reduction stats and do a single all_reduce.
        self.grad_norm_buf = torch.zeros(n_params, device=device, dtype=torch.float32)
        self.weight_norm_buf = torch.zeros(n_params, device=device, dtype=torch.float32)
        self._step_recorded = False  # set when record_step_boundary ran this window

        # ---- residual-stream RMS taps ---------------------------------------
        # FSDP2's composable `fully_shard` swaps `mod.__class__` to a dynamic
        # subclass, so match via MRO rather than immediate class name.
        def _mro_named(mod: nn.Module, names: tuple[str, ...]) -> bool:
            for cls in type(mod).__mro__:
                if cls.__name__ in names:
                    return True
            return False

        def _is_block(mod: nn.Module) -> bool:
            return _mro_named(mod, _BLOCK_CLASS_NAMES)

        self.block_names: list[str] = []
        blocks: list[nn.Module] = []
        for mod_name, mod in model.named_modules():
            if _is_block(mod):
                self.block_names.append(mod_name)
                blocks.append(mod)
        n_blocks = len(blocks)
        # One `[2, max_rep]` tensor per block: [0]=Σ mean(x*x) per application,
        # [1]=application count. Owned by the block's RepeatTap from here on.
        self._res_stats: list[torch.Tensor] = []
        n_missing_taps = 0
        for block in blocks:
            tap = getattr(block, "repeat_tap", None)
            # isinstance, not hasattr: the two taps take different arguments, so
            # a misplaced module should warn here rather than raise on install.
            if isinstance(tap, RepeatTap):
                self._res_stats.append(tap.enable_stats(self.max_rep, device))
                self._taps.append(tap)
            else:
                # Placeholder so the packing layout in flush() is unconditional;
                # it stays zero, so no key is emitted for this block.
                n_missing_taps += 1
                self._res_stats.append(
                    torch.zeros(2, self.max_rep, device=device, dtype=torch.float32)
                )
        if n_missing_taps:
            logger.warning(
                "LayerwiseDebugTracker: %d of %d blocks have no `repeat_tap` "
                "(modern_bert.taps.RepeatTap); their residual RMS stays unreported.",
                n_missing_taps,
                n_blocks,
            )

        # ---- logit stats ----------------------------------------------------
        # StatsTap layout: [sq_sum, sum, abs_max, numel, entropy_sum, tokens].
        # `sum` stays zero for logits (StatsTap(with_sum=False)); nothing reads
        # it — `debug/logits/norm` is the raw window sum of squares, not a mean.
        self._logit_stats = self._enable_scalar_tap(
            getattr(model, "logit_tap", None),
            device,
            "`logit_tap` (modeling.ModernBertForMaskedLM tags every scoring "
            "path with it); logit stats disabled.",
        )

        # ---- final_norm output stats (the "embedding" fed into head/decoder)
        # `final_norm_tap` lives on `model.model` (the encoder) in ModernBert;
        # also accept it directly on `model` for robustness.
        final_norm_tap = getattr(model, "final_norm_tap", None)
        if final_norm_tap is None:
            inner = getattr(model, "model", None)
            final_norm_tap = (
                getattr(inner, "final_norm_tap", None) if inner is not None else None
            )
        self._final_stats = self._enable_scalar_tap(
            final_norm_tap,
            device,
            "`final_norm_tap` (modeling.ModernBertModel); final-norm output "
            "stats disabled.",
        )

        # ---- JSONL sink -----------------------------------------------------
        self._jsonl_fh = None
        if self.is_main:
            os.makedirs(output_dir, exist_ok=True)
            self._jsonl_fh = open(
                os.path.join(output_dir, "debug_layerwise.jsonl"), "a", buffering=1
            )

        logger.info(
            "LayerwiseDebugTracker enabled: %d params, %d blocks, log_every=%d",
            n_params,
            n_blocks,
            self.log_every,
        )

    # -------------------------------------------------------------------- taps
    def _enable_scalar_tap(
        self, tap: Optional[object], device: torch.device, missing: str
    ) -> torch.Tensor:
        """Install a 6-element accumulator on ``tap``; zeros if it is absent."""
        if isinstance(tap, StatsTap):
            self._taps.append(tap)
            return tap.enable_stats(device)
        logger.warning("LayerwiseDebugTracker: model has no %s", missing)
        return torch.zeros(6, device=device, dtype=torch.float32)

    def _stacked_res(self) -> torch.Tensor:
        """The per-block residual accumulators as one `[n_blocks, 2, max_rep]`.

        A stack (a copy) rather than a preallocated buffer the taps write into:
        separate storages keep AOTAutograd from seeing aliased mutated inputs.
        """
        if not self._res_stats:
            return torch.zeros(
                0, 2, self.max_rep, device=self.device, dtype=torch.float32
            )
        return torch.stack(self._res_stats)

    @property
    def residual_sq_sum(self) -> torch.Tensor:
        """`[n_blocks, max_rep]` sum of per-application `mean(x*x)`."""
        return self._stacked_res()[:, 0, :]

    @property
    def residual_count(self) -> torch.Tensor:
        """`[n_blocks, max_rep]` application counts in this window."""
        return self._stacked_res()[:, 1, :]

    # --------------------------------------------------------- step boundary
    def record_step_boundary(self) -> None:
        """Called once per optimizer step, post-bwd, pre-clip.

        Stores the local-shard squared sum for each grad/weight in
        ``grad_norm_buf`` / ``weight_norm_buf``. The cross-rank reduction is
        deferred to the single packed ``all_reduce`` in ``flush()``, so this
        method issues NO collectives.

        Earlier versions called ``DTensor.full_tensor()`` on every grad and
        weight here, which materialized via 2*N eager all-gathers per flush
        window. Racing with FSDP's prefetched all-gather schedule could leave
        per-rank collective queues mismatched and trigger NCCL deadlocks
        several steps later. The local-shard form avoids that entirely.
        """
        # For replicated DTensors and plain Tensors, every rank already holds
        # the same data; pre-dividing by world_size cancels the SUM-reduce
        # overcount in flush().
        world = (
            dist.get_world_size()
            if self.distributed and dist.is_initialized()
            else 1
        )

        with torch.no_grad():
            for i, p in enumerate(self._params):
                w = p.data
                w_local = w.to_local() if hasattr(w, "to_local") else w
                w_sq = w_local.float().pow(2).sum()
                if world > 1 and not _is_sharded(w):
                    w_sq = w_sq / world
                self.weight_norm_buf[i].copy_(w_sq)

                g = p.grad
                if g is None:
                    self.grad_norm_buf[i].zero_()
                    continue
                g_local = g.to_local() if hasattr(g, "to_local") else g
                g_sq = g_local.float().pow(2).sum()
                if world > 1 and not _is_sharded(g):
                    g_sq = g_sq / world
                self.grad_norm_buf[i].copy_(g_sq)
        self._step_recorded = True

    # ---------------------------------------------------------------- flush
    def should_flush(self, global_step: int) -> bool:
        return global_step > 0 and (global_step % self.log_every == 0)

    def flush(self, global_step: int) -> Optional[dict]:
        """Single host-sync point.  Returns payload dict on rank 0, else None.

        Key semantics are unchanged from the forward-hook era, including
        ``debug/logits/norm``: it is the raw sqrt of the squares summed over the
        whole window (not per token and not per step), so it scales with the
        window length and the batch, while ``debug/logits/entropy`` and the
        residual/final-norm keys are means.
        """
        # Pack everything into two contiguous tensors: one SUM-reduced, one MAX-reduced.
        # Layout (SUM): [grad_norm_sq... weight_norm_sq... residual_sq... residual_cnt...
        #                logit_sq_sum, logit_entropy_sum, logit_elem_count, logit_token_count]
        # Layout (MAX): [logit_abs_max]
        # grad/weight slots hold per-rank LOCAL-shard sums of squares (set in
        # record_step_boundary). After SUM-all-reduce they become global squared
        # norms directly — no per-rank averaging. Residual/logit slots are
        # per-rank stats and are averaged across ranks below.

        n_p = len(self._params)
        n_b = len(self.block_names)
        n_br = n_b * self.max_rep
        res = self._stacked_res()  # [n_b, 2, max_rep]
        sum_buf = torch.cat(
            [
                self.grad_norm_buf,
                self.weight_norm_buf,
                res[:, 0, :].reshape(-1),   # residual Σ mean(x*x)
                res[:, 1, :].reshape(-1),   # residual application counts
                self._logit_stats[0:1],     # sq_sum
                self._logit_stats[4:5],     # entropy_sum
                self._logit_stats[3:4],     # elem count
                self._logit_stats[5:6],     # token count
                self._final_stats[0:1],     # sq_sum
                self._final_stats[1:2],     # sum
                self._final_stats[3:4],     # elem count
            ]
        )
        max_buf = torch.stack([self._logit_stats[2], self._final_stats[2]])

        # Boundary between "summed-squared-norm" slots and "averaged-stat" slots.
        gw_end = 2 * n_p

        if self.distributed and dist.is_initialized():
            world = dist.get_world_size()
            dist.all_reduce(sum_buf, op=dist.ReduceOp.SUM)
            # Only residual/logit stats want a mean across ranks. Grad/weight
            # squared sums are SUM-correct as-is.
            sum_buf[gw_end:].div_(world)
            dist.all_reduce(max_buf, op=dist.ReduceOp.MAX)

        sum_cpu = sum_buf.cpu()
        max_cpu = max_buf.cpu()

        off = 0
        grad_sq_cpu = sum_cpu[off : off + n_p]; off += n_p
        weight_sq_cpu = sum_cpu[off : off + n_p]; off += n_p
        residual_sq_cpu = sum_cpu[off : off + n_br].view(n_b, self.max_rep); off += n_br
        residual_cnt_cpu = sum_cpu[off : off + n_br].view(n_b, self.max_rep); off += n_br
        logit_sq_sum = float(sum_cpu[off]); off += 1
        logit_entropy_sum = float(sum_cpu[off]); off += 1
        logit_elem_count = float(sum_cpu[off]); off += 1
        logit_token_count = float(sum_cpu[off]); off += 1
        final_norm_sq_sum = float(sum_cpu[off]); off += 1
        final_norm_sum = float(sum_cpu[off]); off += 1
        final_norm_elem_count = float(sum_cpu[off]); off += 1
        logit_abs_max = float(max_cpu[0])
        final_norm_abs_max = float(max_cpu[1])

        # Reset GPU state in-place for next window. In place and never
        # reassigned: the taps (and, after the first trace, the compiled graph)
        # hold references to exactly these tensors.
        with torch.no_grad():
            for buf in self._res_stats:
                buf.zero_()
            self._logit_stats.zero_()
            self._final_stats.zero_()
        self._step_recorded = False

        if not self.is_main:
            return None

        payload: dict = {}
        grad_sq_list = grad_sq_cpu.tolist()
        weight_sq_list = weight_sq_cpu.tolist()
        for i, name in enumerate(self.param_names):
            role = _role_from_name(name)
            payload[f"debug/grad_norm/{role}/{name}"] = grad_sq_list[i] ** 0.5
            payload[f"debug/weight_norm/{role}/{name}"] = weight_sq_list[i] ** 0.5

        residual_sq_list = residual_sq_cpu.tolist()
        residual_cnt_list = residual_cnt_cpu.tolist()
        for i, name in enumerate(self.block_names):
            for r in range(self.max_rep):
                cnt = residual_cnt_list[i][r]
                if cnt > 0:
                    mean_sq = residual_sq_list[i][r] / cnt
                    payload[f"debug/residual_rms/block_{i:03d}_rep{r}"] = mean_sq ** 0.5

        if logit_elem_count > 0:
            payload["debug/logits/norm"] = (logit_sq_sum) ** 0.5
            payload["debug/logits/abs_max"] = logit_abs_max
        if logit_token_count > 0:
            payload["debug/logits/entropy"] = logit_entropy_sum / logit_token_count

        if final_norm_elem_count > 0:
            mean = final_norm_sum / final_norm_elem_count
            mean_sq = final_norm_sq_sum / final_norm_elem_count
            payload["debug/final_norm/rms"] = mean_sq ** 0.5
            payload["debug/final_norm/mean"] = mean
            payload["debug/final_norm/abs_max"] = final_norm_abs_max

        if self._jsonl_fh is not None:
            self._jsonl_fh.write(
                json.dumps({"step": global_step, "metrics": payload}) + "\n"
            )

        return payload

    # ---------------------------------------------------------------- close
    def close(self) -> None:
        # Drops the accumulators, so every tap is a plain identity again.
        for tap in self._taps:
            try:
                tap.disable_stats()
            except Exception:
                pass
        self._taps.clear()
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.close()
            except Exception:
                pass
            self._jsonl_fh = None
