"""Layerwise debug metrics tracker.

Gathers per-layer grad/weight norms, per-block residual-stream RMS, and
lm_head logit stats with a single host-sync per logging window.

Design invariants
-----------------
* All accumulation uses in-place ops on preallocated GPU tensors. No `.item()`,
  `.cpu()`, or `.tolist()` outside `flush()`.
* `flush()` is the only host sync point: one `all_reduce` per reduction op, one
  `.cpu()`, one Python fan-out into a payload dict.
* Forward hooks remain registered for the lifetime of the tracker. On non-log
  steps they still fire and accumulate — the cost is one fused scalar add per
  hook, which is negligible compared to a transformer block.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

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
    def __init__(
        self,
        model: nn.Module,
        log_every: int,
        device: torch.device,
        output_dir: str,
        is_main: bool,
        distributed: bool,
    ) -> None:
        self.log_every = int(log_every)
        self.device = device
        self.is_main = is_main
        self.distributed = distributed

        self._hook_handles: list = []

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

        # ---- residual-stream RMS hooks --------------------------------------
        # FSDP2's composable `fully_shard` swaps `mod.__class__` to a dynamic
        # subclass, so match via MRO rather than immediate class name.
        def _is_block(mod: nn.Module) -> bool:
            for cls in type(mod).__mro__:
                if cls.__name__ in _BLOCK_CLASS_NAMES:
                    return True
            return False

        self.block_names: list[str] = []
        blocks: list[nn.Module] = []
        for mod_name, mod in model.named_modules():
            if _is_block(mod):
                self.block_names.append(mod_name)
                blocks.append(mod)
        n_blocks = len(blocks)
        self.residual_sq_sum = torch.zeros(n_blocks, device=device, dtype=torch.float32)
        self.residual_count = torch.zeros(n_blocks, device=device, dtype=torch.float32)
        for idx, block in enumerate(blocks):
            self._hook_handles.append(
                block.register_forward_hook(self._make_residual_hook(idx))
            )

        # ---- logit stats ----------------------------------------------------
        # [0]=sq_sum over all scored elements, [1]=abs_max, [2]=entropy_sum
        self.logit_buf = torch.zeros(3, device=device, dtype=torch.float32)
        self.logit_elem_count = torch.zeros(1, device=device, dtype=torch.float32)
        self.logit_token_count = torch.zeros(1, device=device, dtype=torch.float32)
        decoder = getattr(model, "decoder", None)
        if isinstance(decoder, nn.Linear):
            self._hook_handles.append(decoder.register_forward_hook(self._logit_hook))
        else:
            logger.warning(
                "LayerwiseDebugTracker: model has no nn.Linear `decoder` attribute; "
                "logit stats disabled."
            )

        # ---- final_norm output stats (the "embedding" fed into head/decoder)
        # [0]=sq_sum, [1]=sum (for mean), [2]=abs_max
        self.final_norm_buf = torch.zeros(3, device=device, dtype=torch.float32)
        self.final_norm_elem_count = torch.zeros(1, device=device, dtype=torch.float32)
        # `final_norm` lives on `model.model` (the encoder) in ModernBert; also
        # accept it directly on `model` for robustness.
        final_norm = getattr(model, "final_norm", None)
        if final_norm is None:
            inner = getattr(model, "model", None)
            final_norm = getattr(inner, "final_norm", None) if inner is not None else None
        if isinstance(final_norm, nn.Module):
            self._hook_handles.append(
                final_norm.register_forward_hook(self._final_norm_hook)
            )
        else:
            logger.warning(
                "LayerwiseDebugTracker: model has no `final_norm` module; "
                "final-norm output stats disabled."
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

    # ------------------------------------------------------------------ hooks
    def _make_residual_hook(self, idx: int):
        buf_sq = self.residual_sq_sum
        buf_cnt = self.residual_count

        # `torch.compiler.disable` keeps the hook body OUT of the compiled
        # graph. The model is wrapped by torch.compile, so without this Dynamo
        # would inline the hook's data-dependent control flow and in-place
        # mutation of these external tracker tensors, churning guards and
        # forcing repeated recompiles. Running it eager trades a graph break at
        # each block boundary for compile stability.
        @torch.compiler.disable
        def hook(_module, _inputs, output):
            if not _module.training:
                return
            x = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(x):
                return
            with torch.no_grad():
                # Compute in native dtype (residual std is O(1), bf16 safe),
                # cast only the scalar to fp32 for the fp32 accumulator.
                xd = x.detach()
                sq = (xd * xd).mean()
                buf_sq[idx].add_(sq.float())
                buf_cnt[idx].add_(1.0)

        return hook

    # See `_make_residual_hook` for why these hooks are compile-disabled.
    @torch.compiler.disable
    def _final_norm_hook(self, _module, _inputs, output):
        if not _module.training:
            return
        x = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(x):
            return
        with torch.no_grad():
            xd = x.detach()
            sq_sum = (xd * xd).sum()
            sum_ = xd.sum()
            abs_max = xd.abs().amax()
            self.final_norm_buf[0].add_(sq_sum.float())
            self.final_norm_buf[1].add_(sum_.float())
            self.final_norm_buf[2].copy_(
                torch.maximum(self.final_norm_buf[2], abs_max.float())
            )
            self.final_norm_elem_count.add_(float(xd.numel()))

    @torch.compiler.disable
    def _logit_hook(self, _module, _inputs, output):
        if not _module.training:
            return
        if not torch.is_tensor(output):
            return
        logits = output.detach()
        with torch.no_grad():
            # Scalars cast to fp32 at the end; avoids a full [B,T,V] fp32 alloc.
            sq_sum = (logits * logits).sum()
            abs_max = logits.abs().amax()
            # Entropy via logsumexp identity: H = lse - Σ p * logits.
            # logsumexp and softmax run in logits' dtype (bf16 is fine for this
            # range since lse is stable); only two [B, T] tensors allocated.
            lse = torch.logsumexp(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            entropy_per_token = lse - (probs * logits).sum(dim=-1)
            self.logit_buf[0].add_(sq_sum.float())
            self.logit_buf[1].copy_(torch.maximum(self.logit_buf[1], abs_max.float()))
            self.logit_buf[2].add_(entropy_per_token.sum().float())
            n_elem = logits.numel()
            n_tokens = n_elem // logits.shape[-1]
            self.logit_elem_count.add_(float(n_elem))
            self.logit_token_count.add_(float(n_tokens))

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
        """Single host-sync point.  Returns payload dict on rank 0, else None."""
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
        sum_buf = torch.cat(
            [
                self.grad_norm_buf,
                self.weight_norm_buf,
                self.residual_sq_sum,
                self.residual_count,
                self.logit_buf[0:1],
                self.logit_buf[2:3],
                self.logit_elem_count,
                self.logit_token_count,
                self.final_norm_buf[0:1],   # sq_sum
                self.final_norm_buf[1:2],   # sum
                self.final_norm_elem_count,
            ]
        )
        max_buf = torch.stack([self.logit_buf[1], self.final_norm_buf[2]])

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
        residual_sq_cpu = sum_cpu[off : off + n_b]; off += n_b
        residual_cnt_cpu = sum_cpu[off : off + n_b]; off += n_b
        logit_sq_sum = float(sum_cpu[off]); off += 1
        logit_entropy_sum = float(sum_cpu[off]); off += 1
        logit_elem_count = float(sum_cpu[off]); off += 1
        logit_token_count = float(sum_cpu[off]); off += 1
        final_norm_sq_sum = float(sum_cpu[off]); off += 1
        final_norm_sum = float(sum_cpu[off]); off += 1
        final_norm_elem_count = float(sum_cpu[off]); off += 1
        logit_abs_max = float(max_cpu[0])
        final_norm_abs_max = float(max_cpu[1])

        # Reset GPU state in-place for next window.
        with torch.no_grad():
            self.residual_sq_sum.zero_()
            self.residual_count.zero_()
            self.logit_buf.zero_()
            self.logit_elem_count.zero_()
            self.logit_token_count.zero_()
            self.final_norm_buf.zero_()
            self.final_norm_elem_count.zero_()
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
            cnt = residual_cnt_list[i]
            if cnt > 0:
                mean_sq = residual_sq_list[i] / cnt
                payload[f"debug/residual_rms/block_{i:03d}"] = mean_sq ** 0.5

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
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.close()
            except Exception:
                pass
            self._jsonl_fh = None
