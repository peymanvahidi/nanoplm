"""Fused Triton kernel for MLP: relu(x @ W1.T)^2.

Adapted from modded-nanogpt by @andrewbriand, @jrauvola.
Auto-selects block sizes for H100 (SM90) and RTX 5090 (SM120).
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

NUM_SMS = None
# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages_fwd, num_stages_bwd, num_warps)
_HW_CONFIG = None

_CONFIGS = {
    # H100: 228 KB shared memory — use modded-nanogpt's original tuning
    (9, 0): (128, 256, 64, 4, 3, 8),
    # RTX 5090: 100 KB shared memory
    (12, 0): (128, 128, 64, 3, 3, 4),
}


def _init_hw():
    global NUM_SMS, _HW_CONFIG
    props = torch.cuda.get_device_properties("cuda")
    NUM_SMS = props.multi_processor_count
    cc = (props.major, props.minor)
    _HW_CONFIG = _CONFIGS.get(cc)
    if _HW_CONFIG is None:
        raise RuntimeError(
            f"Fused SReLU kernel not tuned for SM{props.major}{props.minor} "
            f"({props.name}). Supported: H100 (SM90), RTX 5090 (SM120)."
        )


@triton.jit
def linear_relu_square_kernel(
    a_desc, b_desc, c_desc, aux_desc,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FORWARD: tl.constexpr,
):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am, offs_bn])
            c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)

        c_desc.store([offs_am, offs_bn], c0)

        if FORWARD:
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            aux_desc.store([offs_am, offs_bn], c0_post)

        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am, offs_bn + BLOCK_SIZE_N // 2])
            c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)

        c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)

        if FORWARD:
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            aux_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1_post)


def linear_relu_square(a, b, aux=None):
    global NUM_SMS, _HW_CONFIG
    if _HW_CONFIG is None:
        _init_hw()

    M, K = a.shape
    N = b.shape[0]
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    FORWARD = aux is None
    if FORWARD:
        aux = torch.empty((M, N), device=a.device, dtype=dtype)

    BM, BN, BK, stages_fwd, stages_bwd, nw = _HW_CONFIG
    num_stages = stages_fwd if FORWARD else stages_bwd

    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(b, [BN, BK])
    c_desc = TensorDescriptor.from_tensor(c, [BM, BN // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BM, BN // 2])

    grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
    linear_relu_square_kernel[grid](
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        NUM_SMS=NUM_SMS, FORWARD=FORWARD,
        num_stages=num_stages, num_warps=nw,
    )

    if FORWARD:
        return c, aux
    return c


class FusedLinearReLUSquare(torch.autograd.Function):
    """Legacy eager-only wrapper. A raw autograd.Function is a dynamo graph
    break (the model shattered into two compiled fragments per layer with a
    ~1 ms GPU hole at the seam); prefer fused_linear_relu_square below, which
    routes through registered custom ops that torch.compile keeps in-graph.
    """

    @staticmethod
    def forward(ctx, x, W1, W2):
        flat = x.view(-1, x.shape[-1])
        pre, post = linear_relu_square(flat, W1)
        out = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        flat_x = x.view(-1, x.shape[-1])
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        dW2 = post.T @ flat_grad
        dpre = linear_relu_square(flat_grad, W2, aux=pre)
        dW1 = dpre.T @ flat_x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2


# -- torch.compile-friendly custom ops ----------------------------------------
#
# Same kernels and op order as FusedLinearReLUSquare (bit-identical results),
# but registered through torch.library so dynamo records a single graph node
# instead of breaking the layer graph in half.  pre/post are returned from the
# forward op so register_autograd can save them for the fused backward.


@torch.library.custom_op("nanoplm_srelu::mlp_fwd", mutates_args=(), device_types="cuda")
def _srelu_mlp_fwd(
    x2d: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre, post = linear_relu_square(x2d.contiguous(), w1)
    out = post @ w2
    return out, pre, post


@_srelu_mlp_fwd.register_fake
def _srelu_mlp_fwd_fake(x2d, w1, w2):
    M = x2d.shape[0]
    N = w1.shape[0]
    H = w2.shape[1]
    return (
        x2d.new_empty((M, H)),
        x2d.new_empty((M, N)),
        x2d.new_empty((M, N)),
    )


@torch.library.custom_op("nanoplm_srelu::lrs_bwd", mutates_args=(), device_types="cuda")
def _srelu_lrs_bwd(g2d: torch.Tensor, w2: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    # (g2d @ w2.T) * 2*relu(pre): gemm with the activation derivative fused in.
    return linear_relu_square(g2d.contiguous(), w2, aux=pre)


@_srelu_lrs_bwd.register_fake
def _srelu_lrs_bwd_fake(g2d, w2, pre):
    return g2d.new_empty((g2d.shape[0], w2.shape[0]))


def _srelu_mlp_setup_context(ctx, inputs, output):
    x2d, w1, w2 = inputs
    _, pre, post = output
    ctx.save_for_backward(x2d, w1, w2, pre, post)


def _srelu_mlp_backward(ctx, grad_out, grad_pre, grad_post):
    x2d, w1, w2, pre, post = ctx.saved_tensors
    g = grad_out.contiguous()
    dw2 = post.T @ g
    dpre = torch.ops.nanoplm_srelu.lrs_bwd(g, w2, pre)
    dw1 = dpre.T @ x2d
    dx = dpre @ w1
    return dx, dw1, dw2


_srelu_mlp_fwd.register_autograd(_srelu_mlp_backward, setup_context=_srelu_mlp_setup_context)


def fused_linear_relu_square(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """relu(x @ w1.T)^2 @ w2 via the fused kernels; traceable by torch.compile."""
    flat = x.view(-1, x.shape[-1])
    out, _, _ = torch.ops.nanoplm_srelu.mlp_fwd(flat, w1, w2)
    return out.view(x.shape)
