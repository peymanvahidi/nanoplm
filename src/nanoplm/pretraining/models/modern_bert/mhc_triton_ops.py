"""Torch custom ops for mHC-lite Triton kernels.

Goal: make the fused Triton kernels usable under torch.compile without Dynamo
graph breaks by exposing them as dispatcher ops with FakeTensor support and a
registered autograd formula.

The CUDA implementations launch the existing Triton kernels in
mhc_triton_kernels.py. Fake implementations only allocate outputs with correct
shape/dtype/device.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any

import torch
import torch.library


_lib = torch.library.Library("nanoplm_mhc", "DEF")

_lib.define(
    "fused_rmsnorm_project(Tensor x_flat, Tensor W) -> (Tensor out, Tensor inv_rms)"
)
_lib.define(
    "fused_rmsnorm_project_bwd_dx(Tensor grad_out, Tensor x_flat, Tensor W, Tensor proj_out, Tensor inv_rms) -> Tensor grad_x"
)

_lib.define("fused_pre_map(Tensor x_streams, Tensor h_pre) -> Tensor out")
_lib.define(
    "fused_pre_map_backward(Tensor grad_out, Tensor x_streams, Tensor h_pre) -> (Tensor grad_x, Tensor grad_h_pre)"
)

_lib.define(
    "fused_post_res(Tensor x_streams, Tensor layer_output, Tensor H_merged, Tensor h_post) -> Tensor out"
)
_lib.define(
    "fused_post_res_backward(Tensor grad_out, Tensor x_streams, Tensor layer_output, Tensor H_merged, Tensor h_post) -> (Tensor grad_x, Tensor grad_layer_output, Tensor grad_H_merged, Tensor grad_h_post)"
)


_AUTOTUNE_CONTROL = threading.local()


def _autotune_disable_depth() -> int:
    return int(getattr(_AUTOTUNE_CONTROL, "disable_depth", 0))


@contextmanager
def disable_autotune_temporarily():
    """Temporarily force the legacy non-autotune launch path."""
    _AUTOTUNE_CONTROL.disable_depth = _autotune_disable_depth() + 1
    try:
        yield
    finally:
        _AUTOTUNE_CONTROL.disable_depth = max(0, _autotune_disable_depth() - 1)


def _autotune_enabled() -> bool:
    v = os.getenv("NANOPLM_MHC_TRITON_AUTOTUNE", "1").strip().lower()
    return v not in {"0", "false", "off", "no"} and _autotune_disable_depth() == 0


def _autotune_status_enabled() -> bool:
    v = os.getenv("NANOPLM_MHC_TRITON_AUTOTUNE_STATUS", "1").strip().lower()
    return v not in {"0", "false", "off", "no"}


_AUTOTUNE_STATUS_SEEN: set[tuple[str, tuple[Any, ...]]] = set()


def _autotune_key(autotuner: Any, args_by_name: dict[str, Any]) -> tuple[Any, ...]:
    key: list[Any] = []
    for name in getattr(autotuner, "keys", ()):
        if name in args_by_name:
            key.append(args_by_name[name])
    for name in getattr(autotuner, "arg_names", ()):
        if name in args_by_name:
            arg = args_by_name[name]
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
    return tuple(key)


def _autotune_status_begin(
    *,
    kernel_name: str,
    autotuner: Any,
    args_by_name: dict[str, Any],
) -> tuple[tuple[Any, ...], bool, Any] | None:
    if not _autotune_status_enabled():
        return None
    key = _autotune_key(autotuner, args_by_name)
    tag = (kernel_name, key)
    if tag in _AUTOTUNE_STATUS_SEEN:
        return None
    _AUTOTUNE_STATUS_SEEN.add(tag)

    in_mem_hit = key in getattr(autotuner, "cache", {})
    bench_before = getattr(autotuner, "bench_time", None)
    if in_mem_hit:
        cfg = autotuner.cache[key]
        print(
            f"[nanoplm][triton] autotune cache hit: {kernel_name} key={key} cfg={cfg}",
            flush=True,
        )
    else:
        num_cfgs = len(getattr(autotuner, "configs", ()))
        print(
            f"[nanoplm][triton] resolving autotune config: {kernel_name} key={key} "
            f"(may benchmark up to {num_cfgs} configs on first run)",
            flush=True,
        )
    return key, in_mem_hit, bench_before


def _autotune_status_end(
    *,
    kernel_name: str,
    autotuner: Any,
    state: tuple[tuple[Any, ...], bool, Any] | None,
) -> None:
    if not _autotune_status_enabled() or state is None:
        return
    key, in_mem_hit, bench_before = state
    if in_mem_hit:
        return
    bench_after = getattr(autotuner, "bench_time", None)
    ran_benchmark = bench_after is not None and bench_after != bench_before
    cfg = getattr(autotuner, "cache", {}).get(key, getattr(autotuner, "best_config", None))
    if ran_benchmark:
        print(
            f"[nanoplm][triton] autotune finished: {kernel_name} selected={cfg}",
            flush=True,
        )
    else:
        print(
            f"[nanoplm][triton] autotune loaded from disk cache: {kernel_name} selected={cfg}",
            flush=True,
        )


@torch.library.register_fake("nanoplm_mhc::fused_rmsnorm_project")
def _fused_rmsnorm_project_fake(x_flat: torch.Tensor, W: torch.Tensor):
    T = x_flat.shape[0]
    D_out = W.shape[0]
    out = torch.empty((T, D_out), device=x_flat.device, dtype=x_flat.dtype)
    inv_rms = torch.empty((T,), device=x_flat.device, dtype=torch.float32)
    return out, inv_rms


@torch.library.register_fake("nanoplm_mhc::fused_rmsnorm_project_bwd_dx")
def _fused_rmsnorm_project_bwd_dx_fake(
    grad_out: torch.Tensor,
    x_flat: torch.Tensor,
    W: torch.Tensor,
    proj_out: torch.Tensor,
    inv_rms: torch.Tensor,
):
    return torch.empty_like(x_flat)


@torch.library.register_fake("nanoplm_mhc::fused_pre_map")
def _fused_pre_map_fake(x_streams: torch.Tensor, h_pre: torch.Tensor):
    T, _, C = x_streams.shape
    return torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)


@torch.library.register_fake("nanoplm_mhc::fused_pre_map_backward")
def _fused_pre_map_backward_fake(
    grad_out: torch.Tensor, x_streams: torch.Tensor, h_pre: torch.Tensor
):
    grad_x = torch.empty_like(x_streams)
    grad_h_pre = torch.empty_like(h_pre, dtype=torch.float32)
    return grad_x, grad_h_pre


@torch.library.register_fake("nanoplm_mhc::fused_post_res")
def _fused_post_res_fake(
    x_streams: torch.Tensor,
    layer_output: torch.Tensor,
    H_merged: torch.Tensor,
    h_post: torch.Tensor,
):
    return torch.empty_like(x_streams)


@torch.library.register_fake("nanoplm_mhc::fused_post_res_backward")
def _fused_post_res_backward_fake(
    grad_out: torch.Tensor,
    x_streams: torch.Tensor,
    layer_output: torch.Tensor,
    H_merged: torch.Tensor,
    h_post: torch.Tensor,
):
    grad_x = torch.empty_like(x_streams)
    grad_layer_output = torch.empty_like(layer_output)
    grad_H_merged = torch.empty_like(H_merged, dtype=torch.float32)
    grad_h_post = torch.empty_like(h_post, dtype=torch.float32)
    return grad_x, grad_layer_output, grad_H_merged, grad_h_post


@torch.library.impl(_lib, "fused_rmsnorm_project", "CUDA")
def _fused_rmsnorm_project_cuda(x_flat: torch.Tensor, W: torch.Tensor):
    from . import mhc_triton_kernels as k

    T, nC = x_flat.shape
    D_out = W.shape[0]
    out = torch.empty((T, D_out), device=x_flat.device, dtype=x_flat.dtype)
    inv_rms = torch.empty((T,), device=x_flat.device, dtype=torch.float32)

    if _autotune_enabled():
        autotuner = k._fused_rmsnorm_project_fwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="k1_fwd",
            autotuner=autotuner,
            args_by_name={
                "x_ptr": x_flat,
                "W_ptr": W,
                "out_ptr": out,
                "inv_rms_ptr": inv_rms,
                "T": T,
                "nC": nC,
                "D_out": D_out,
            },
        )
        grid = lambda META: (k.triton.cdiv(T, META["BLOCK_T"]),)
        autotuner[grid](
            x_flat,
            W,
            out,
            inv_rms,
            T,
            nC,
            D_out,
        )
        _autotune_status_end(kernel_name="k1_fwd", autotuner=autotuner, state=status)
    else:
        cc_major, _ = torch.cuda.get_device_capability()
        _, nw, ns = k._get_hw_config()
        BLOCK_K = min(128, k.triton.next_power_of_2(nC))
        BLOCK_T = 128
        if cc_major >= 12:
            BLOCK_T = 128
            BLOCK_K = 64
            nw = 4
            ns = 3
        grid = (k.triton.cdiv(T, BLOCK_T),)
        k._fused_rmsnorm_project_fwd_kernel[grid](
            x_flat,
            W,
            out,
            inv_rms,
            T,
            nC,
            D_out,
            BLOCK_T=BLOCK_T,
            BLOCK_K=BLOCK_K,
            num_warps=nw,
            num_stages=ns,
        )
    return out, inv_rms


@torch.library.impl(_lib, "fused_rmsnorm_project_bwd_dx", "CUDA")
def _fused_rmsnorm_project_bwd_dx_cuda(
    grad_out: torch.Tensor,
    x_flat: torch.Tensor,
    W: torch.Tensor,
    proj_out: torch.Tensor,
    inv_rms: torch.Tensor,
):
    from . import mhc_triton_kernels as k

    T, nC = x_flat.shape
    D_out = W.shape[0]
    grad_out = grad_out.contiguous()

    grad_x = torch.empty_like(x_flat)
    if _autotune_enabled():
        autotuner = k._fused_rmsnorm_project_bwd_dx_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="k1_bwd_dx",
            autotuner=autotuner,
            args_by_name={
                "x_ptr": x_flat,
                "W_ptr": W,
                "grad_out_ptr": grad_out,
                "proj_out_ptr": proj_out,
                "inv_rms_ptr": inv_rms,
                "grad_x_ptr": grad_x,
                "T": T,
                "nC": nC,
                "D_out": D_out,
            },
        )
        grid = lambda META: (k.triton.cdiv(T, META["BLOCK_T"]),)
        autotuner[grid](
            x_flat,
            W,
            grad_out,
            proj_out,
            inv_rms,
            grad_x,
            T,
            nC,
            D_out,
        )
        _autotune_status_end(kernel_name="k1_bwd_dx", autotuner=autotuner, state=status)
    else:
        _, nw, ns = k._get_hw_config()
        cc_major, _ = torch.cuda.get_device_capability()
        if cc_major >= 12:
            BLOCK_T = 64
            BLOCK_K = 128
            nw = 8
            ns_bwd = 3
        elif cc_major == 9:
            BLOCK_T = 64
            BLOCK_K = min(128, k.triton.next_power_of_2(nC))
            ns_bwd = ns
        else:
            BLOCK_T = 128
            BLOCK_K = min(128, k.triton.next_power_of_2(nC))
            ns_bwd = ns
        grid = (k.triton.cdiv(T, BLOCK_T),)
        k._fused_rmsnorm_project_bwd_dx_kernel[grid](
            x_flat,
            W,
            grad_out,
            proj_out,
            inv_rms,
            grad_x,
            T,
            nC,
            D_out,
            BLOCK_T=BLOCK_T,
            BLOCK_K=BLOCK_K,
            num_warps=nw,
            num_stages=ns_bwd,
        )
    return grad_x


@torch.library.impl(_lib, "fused_pre_map", "CUDA")
def _fused_pre_map_cuda(x_streams: torch.Tensor, h_pre: torch.Tensor):
    from . import mhc_triton_kernels as k

    T, n, C = x_streams.shape
    if n != 4:
        raise ValueError("nanoplm_mhc::fused_pre_map currently supports n=4 only")
    out = torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)
    NUM_SMS, nw, ns = k._get_hw_config()
    if _autotune_enabled():
        autotuner = k._fused_pre_map_fwd_kernel_autotuned
        status = _autotune_status_begin(
            kernel_name="k3_fwd",
            autotuner=autotuner,
            args_by_name={
                "x_ptr": x_streams,
                "h_pre_ptr": h_pre,
                "out_ptr": out,
                "T": T,
                "C": C,
                "n": n,
            },
        )
        grid = (NUM_SMS,)
        autotuner[grid](
            x_streams,
            h_pre,
            out,
            T,
            C,
            n,
            NUM_SMS=NUM_SMS,
        )
        _autotune_status_end(kernel_name="k3_fwd", autotuner=autotuner, state=status)
    else:
        cc_major, _ = torch.cuda.get_device_capability()
        if cc_major >= 12:
            # Tuned on RTX 5090 (SM120), T=65536/C=1024/n=4.
            BLOCK_T = 128
            BLOCK_C = 64
            nw = 4
            ns_pre = 2
        elif cc_major == 9:
            # Tuned on H100 for shape T=65536, C=1024, n=4.
            BLOCK_T = 128
            BLOCK_C = 128
            nw = 8
            ns_pre = 4
        else:
            # Original heuristic path for non-SM90 devices.
            BLOCK_T = 64
            BLOCK_C = min(256, k.triton.next_power_of_2(C))
            ns_pre = ns
        grid = (NUM_SMS,)
        k._fused_pre_map_fwd_kernel[grid](
            x_streams,
            h_pre,
            out,
            T,
            C,
            n,
            BLOCK_T=BLOCK_T,
            BLOCK_C=BLOCK_C,
            NUM_SMS=NUM_SMS,
            num_warps=nw,
            num_stages=ns_pre,
        )
    return out


@torch.library.impl(_lib, "fused_pre_map_backward", "CUDA")
def _fused_pre_map_backward_cuda(
    grad_out: torch.Tensor, x_streams: torch.Tensor, h_pre: torch.Tensor
):
    from . import mhc_triton_kernels as k

    T, n, C = x_streams.shape
    if n != 4:
        raise ValueError("nanoplm_mhc::fused_pre_map_backward supports n=4 only")
    grad_out = grad_out.contiguous()
    NUM_SMS, nw, ns = k._get_hw_config()
    grid = (NUM_SMS,)

    grad_x = torch.empty_like(x_streams)
    grad_h_pre = torch.empty((T, n), device=x_streams.device, dtype=torch.float32)
    if _autotune_enabled():
        autotuner = k._fused_pre_map_bwd_fused_kernel_n4_autotuned
        status = _autotune_status_begin(
            kernel_name="k3_bwd_fused",
            autotuner=autotuner,
            args_by_name={
                "x_ptr": x_streams,
                "h_pre_ptr": h_pre,
                "grad_out_ptr": grad_out,
                "grad_x_ptr": grad_x,
                "grad_hp_ptr": grad_h_pre,
                "T": T,
                "C": C,
                "n": n,
            },
        )
        autotuner[grid](
            x_streams,
            h_pre,
            grad_out,
            grad_x,
            grad_h_pre,
            T,
            C,
            n,
            NUM_SMS=NUM_SMS,
        )
        _autotune_status_end(kernel_name="k3_bwd_fused", autotuner=autotuner, state=status)
    else:
        cc_major, _ = torch.cuda.get_device_capability()
        if cc_major >= 12:
            # Tuned on RTX 5090 (SM120), T=65536/C=1024/n=4.
            BLOCK_T = 32
            BLOCK_C = 256
            nw_f = 8
            ns_f = 4
        elif cc_major == 9:
            BLOCK_T = 64
            BLOCK_C = 128
            nw_f = 8
            ns_f = 3
        else:
            BLOCK_T = 64
            BLOCK_C = min(256, k.triton.next_power_of_2(C))
            nw_f = nw
            ns_f = ns
        k._fused_pre_map_bwd_fused_kernel_n4[grid](
            x_streams,
            h_pre,
            grad_out,
            grad_x,
            grad_h_pre,
            T,
            C,
            n,
            BLOCK_T=BLOCK_T,
            BLOCK_C=BLOCK_C,
            NUM_SMS=NUM_SMS,
            num_warps=nw_f,
            num_stages=ns_f,
        )
    return grad_x, grad_h_pre


@torch.library.impl(_lib, "fused_post_res", "CUDA")
def _fused_post_res_cuda(
    x_streams: torch.Tensor,
    layer_output: torch.Tensor,
    H_merged: torch.Tensor,
    h_post: torch.Tensor,
):
    from . import mhc_triton_kernels as k

    T, n, C = x_streams.shape
    if n != 4:
        raise ValueError("nanoplm_mhc::fused_post_res currently supports n=4 only")
    out = torch.empty_like(x_streams)
    NUM_SMS, nw_default, ns = k._get_hw_config()
    if _autotune_enabled():
        autotuner = k._fused_post_res_fwd_kernel_n4_autotuned
        status = _autotune_status_begin(
            kernel_name="k4_fwd",
            autotuner=autotuner,
            args_by_name={
                "x_ptr": x_streams,
                "lo_ptr": layer_output,
                "H_ptr": H_merged,
                "hp_ptr": h_post,
                "out_ptr": out,
                "T": T,
                "C": C,
                "n": n,
            },
        )
        grid = (NUM_SMS,)
        autotuner[grid](
            x_streams,
            layer_output,
            H_merged,
            h_post,
            out,
            T,
            C,
            n,
            NUM_SMS=NUM_SMS,
        )
        _autotune_status_end(kernel_name="k4_fwd", autotuner=autotuner, state=status)
    else:
        cc_major, _ = torch.cuda.get_device_capability()
        if cc_major >= 12:
            # Tuned on RTX 5090 (SM120), T=65536/C=1024/n=4.
            BLOCK_T = 32
            BLOCK_C = 128
            nw = 8
            ns = 3
        else:
            BLOCK_T = 64 if cc_major >= 9 else 32
            BLOCK_C = 128 if C >= 128 else k.triton.next_power_of_2(C)
            nw = 8 if cc_major >= 9 else nw_default
        grid = (NUM_SMS,)
        k._fused_post_res_fwd_kernel_n4[grid](
            x_streams,
            layer_output,
            H_merged,
            h_post,
            out,
            T,
            C,
            n,
            BLOCK_T=BLOCK_T,
            BLOCK_C=BLOCK_C,
            NUM_SMS=NUM_SMS,
            num_warps=nw,
            num_stages=ns,
        )
    return out


@torch.library.impl(_lib, "fused_post_res_backward", "CUDA")
def _fused_post_res_backward_cuda(
    grad_out: torch.Tensor,
    x_streams: torch.Tensor,
    layer_output: torch.Tensor,
    H_merged: torch.Tensor,
    h_post: torch.Tensor,
):
    from . import mhc_triton_kernels as k

    T, n, C = x_streams.shape
    if n != 4:
        raise ValueError("nanoplm_mhc::fused_post_res_backward supports n=4 only")
    grad_out = grad_out.contiguous()
    NUM_SMS, nw_default, ns = k._get_hw_config()
    grid = (NUM_SMS,)

    grad_x = torch.empty_like(x_streams)
    grad_layer_output = torch.empty((T, C), device=x_streams.device, dtype=x_streams.dtype)
    grad_H = torch.empty((T, n, n), device=x_streams.device, dtype=torch.float32)
    grad_hp = torch.empty((T, n), device=x_streams.device, dtype=torch.float32)

    if _autotune_enabled():
        autotuner = k._fused_post_res_bwd_fused_kernel_n4_autotuned
        status = _autotune_status_begin(
            kernel_name="k4_bwd_fused",
            autotuner=autotuner,
            args_by_name={
                "x_ptr": x_streams,
                "lo_ptr": layer_output,
                "H_ptr": H_merged,
                "hp_ptr": h_post,
                "grad_out_ptr": grad_out,
                "grad_x_ptr": grad_x,
                "grad_lo_ptr": grad_layer_output,
                "grad_H_ptr": grad_H,
                "grad_hp_ptr": grad_hp,
                "T": T,
                "C": C,
                "n": n,
            },
        )
        autotuner[grid](
            x_streams,
            layer_output,
            H_merged,
            h_post,
            grad_out,
            grad_x,
            grad_layer_output,
            grad_H,
            grad_hp,
            T,
            C,
            n,
            NUM_SMS=NUM_SMS,
        )
        _autotune_status_end(kernel_name="k4_bwd_fused", autotuner=autotuner, state=status)
    else:
        cc_major, _ = torch.cuda.get_device_capability()
        if cc_major >= 12:
            # Tuned on RTX 5090 (SM120), T=65536/C=1024/n=4.
            BLOCK_T_F = 16
            BLOCK_C_F = 256
            nw_fused = 8
            ns_fused = 2
        elif cc_major == 9:
            BLOCK_T_F = 32
            BLOCK_C_F = 128
            nw_fused = nw_default
            ns_fused = 2
        else:
            BLOCK_T_F = 32
            BLOCK_C_F = min(256, k.triton.next_power_of_2(C))
            nw_fused = nw_default
            ns_fused = ns
        k._fused_post_res_bwd_fused_kernel_n4[grid](
            x_streams,
            layer_output,
            H_merged,
            h_post,
            grad_out,
            grad_x,
            grad_layer_output,
            grad_H,
            grad_hp,
            T,
            C,
            n,
            BLOCK_T=BLOCK_T_F,
            BLOCK_C=BLOCK_C_F,
            NUM_SMS=NUM_SMS,
            num_warps=nw_fused,
            num_stages=ns_fused,
        )

    return grad_x, grad_layer_output, grad_H, grad_hp


def _setup_save_inputs_outputs(ctx, inputs, output):
    # output can be Tensor or tuple(Tensor,...)
    if isinstance(output, tuple):
        ctx.save_for_backward(*inputs, *output)
    else:
        ctx.save_for_backward(*inputs, output)


def _fused_rmsnorm_project_backward(ctx, grad_out, grad_inv_rms):
    x_flat, W, proj_out, inv_rms = ctx.saved_tensors
    grad_x = torch.ops.nanoplm_mhc.fused_rmsnorm_project_bwd_dx(
        grad_out, x_flat, W, proj_out, inv_rms
    )
    grad_out_scaled = (grad_out * inv_rms[:, None]).to(x_flat.dtype)
    grad_W = torch.matmul(grad_out_scaled.transpose(0, 1), x_flat)
    return grad_x, grad_W


def _fused_pre_map_backward(ctx, grad_out):
    x_streams, h_pre, _out = ctx.saved_tensors
    return torch.ops.nanoplm_mhc.fused_pre_map_backward(grad_out, x_streams, h_pre)


def _fused_post_res_backward(ctx, grad_out):
    x_streams, layer_output, H_merged, h_post, _out = ctx.saved_tensors
    return torch.ops.nanoplm_mhc.fused_post_res_backward(
        grad_out, x_streams, layer_output, H_merged, h_post
    )


torch.library.register_autograd(
    "nanoplm_mhc::fused_rmsnorm_project",
    _fused_rmsnorm_project_backward,
    setup_context=_setup_save_inputs_outputs,
)
torch.library.register_autograd(
    "nanoplm_mhc::fused_pre_map",
    _fused_pre_map_backward,
    setup_context=_setup_save_inputs_outputs,
)
torch.library.register_autograd(
    "nanoplm_mhc::fused_post_res",
    _fused_post_res_backward,
    setup_context=_setup_save_inputs_outputs,
)
