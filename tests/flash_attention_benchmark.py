"""
Flash Attention benchmark
Just checks how much benefit FA2 is providing on your GPUs
"""

import torch, time
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention

device = "cuda"
dtype = torch.float16  # FlashAttention requires float16 or bfloat16

# Large enough to trigger efficient kernels
B, H, L, D = 1, 16, 1024, 128  
q = torch.randn(B, H, L, D, device=device, dtype=dtype)
k = torch.randn(B, H, L, D, device=device, dtype=dtype)
v = torch.randn(B, H, L, D, device=device, dtype=dtype)

def benchmark_attention(name, backends):
    # Warmup
    for _ in range(5):
        with sdpa_kernel(backends=backends):
            scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(10000):
        with sdpa_kernel(backends=backends):
            scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()
    end = time.time()

    print(f"{name:<20}: {end - start:.4f} seconds")

# Benchmark each backend
benchmark_attention("FlashAttention", [SDPBackend.FLASH_ATTENTION])
benchmark_attention("Mem-efficient", [SDPBackend.EFFICIENT_ATTENTION])
benchmark_attention("Math (fallback)", [SDPBackend.MATH])
