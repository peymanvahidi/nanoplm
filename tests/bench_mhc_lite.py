"""Microbenchmark: compare old vs new MHCLiteBlock performance.

Usage: .venv/bin/python tests/bench_mhc_lite.py
"""
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Old implementation (for comparison) ─────────────────────────────────────
def _build_permutation_matrices(n):
    from itertools import permutations
    perms = list(permutations(range(n)))
    identity_idx = 0
    P = torch.zeros(len(perms), n * n)
    for i, perm in enumerate(perms):
        for row, col in enumerate(perm):
            P[i, row * n + col] = 1.0
    return P, identity_idx


class MHCLiteBlockOld(nn.Module):
    """Original MHCLiteBlock (3 separate linears, full rms_norm, delta pattern)."""

    def __init__(self, n_streams, hidden_size, layer):
        super().__init__()
        self.n = n_streams
        self.nC = n_streams * hidden_size
        self.layer = layer
        n_fact = math.factorial(n_streams)
        self.alpha_pre = nn.Parameter(torch.tensor([0.01]))
        self.alpha_post = nn.Parameter(torch.tensor([0.01]))
        self.alpha_res = nn.Parameter(torch.tensor([0.01]))
        self.W_pre = nn.Linear(self.nC, n_streams, bias=True)
        self.W_post = nn.Linear(self.nC, n_streams, bias=True)
        self.W_res = nn.Linear(self.nC, n_fact, bias=True)
        perm_flat, self._identity_idx = _build_permutation_matrices(n_streams)
        self.register_buffer("perm_mat", perm_flat)

    def forward(self, x_streams, **kwargs):
        leading = x_streams.shape[:-2]
        n = self.n
        dt = x_streams.dtype
        x_flat = x_streams.reshape(*leading, self.nC)
        x_norm = F.rms_norm(x_flat, (self.nC,))
        pre_out = F.linear(x_norm, self.W_pre.weight.to(dt), self.W_pre.bias.to(dt))
        post_out = F.linear(x_norm, self.W_post.weight.to(dt), self.W_post.bias.to(dt))
        res_out = F.linear(x_norm, self.W_res.weight.to(dt), self.W_res.bias.to(dt))
        h_pre = torch.sigmoid(self.alpha_pre.to(dt) * pre_out)
        h_post = 2.0 * torch.sigmoid(self.alpha_post.to(dt) * post_out)
        a_res = F.softmax(self.alpha_res.to(dt) * res_out, dim=-1)
        H_res = torch.matmul(a_res, self.perm_mat.to(dt)).unflatten(-1, (n, n))
        layer_input = torch.matmul(h_pre.unsqueeze(-2), x_streams).squeeze(-2)
        layer_output = self.layer(layer_input, **kwargs)
        delta = layer_output - layer_input
        expanded = h_post.unsqueeze(-1) * delta.unsqueeze(-2)
        mixed = torch.matmul(H_res, x_streams)
        return mixed + expanded


# ─── Import new implementation ───────────────────────────────────────────────
sys.path.insert(0, "/workspace/nanoplm/src")
from nanoplm.pretraining.models.modern_bert.modeling import MHCLiteBlock


# ─── Dummy layer that mimics a transformer layer ─────────────────────────────
class DummyLayer(nn.Module):
    """Simple MLP stand-in for a transformer layer with residual."""
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_type = "full_attention"

    def forward(self, x, **kwargs):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # --- Config ---
    T = 65536    # tokens (packed sequence)
    C = 768      # hidden size
    n = 4        # streams
    warmup = 10
    iters = 50

    print(f"Device: {device}, dtype: {dtype}")
    print(f"T={T}, C={C}, n={n}, warmup={warmup}, iters={iters}")
    print()

    # --- Build old & new blocks ---
    layer_old = DummyLayer(C)
    layer_new = DummyLayer(C)
    # Copy weights so both start identical
    layer_new.load_state_dict(layer_old.state_dict())

    old_block = MHCLiteBlockOld(n, C, layer_old).to(device=device, dtype=dtype)
    new_block = MHCLiteBlock(n, C, layer_new, triton_fused=True).to(device=device, dtype=dtype)

    # Sync weights: map old W_pre/W_post/W_res -> new W_all
    with torch.no_grad():
        W_all_weight = torch.cat([
            old_block.W_pre.weight,
            old_block.W_post.weight,
            old_block.W_res.weight,
        ], dim=0)
        W_all_bias = torch.cat([
            old_block.W_pre.bias,
            old_block.W_post.bias,
            old_block.W_res.bias,
        ], dim=0)
        new_block.W_all.weight.copy_(W_all_weight)
        new_block.W_all.bias.copy_(W_all_bias)
        new_block.alpha_pre.copy_(old_block.alpha_pre)
        new_block.alpha_post.copy_(old_block.alpha_post)
        new_block.alpha_res.copy_(old_block.alpha_res)

    # --- Numerical equivalence check ---
    print("=" * 60)
    print("NUMERICAL EQUIVALENCE CHECK")
    print("=" * 60)
    x = torch.randn(T, n, C, device=device, dtype=dtype)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        out_old = old_block(x.clone())
        out_new = new_block(x.clone())
    diff = (out_old - out_new).abs()
    print(f"  Max absolute diff:  {diff.max().item():.6e}")
    print(f"  Mean absolute diff: {diff.mean().item():.6e}")
    rel = diff / (out_old.abs() + 1e-8)
    print(f"  Max relative diff:  {rel.max().item():.6e}")
    print()

    # --- Benchmark function ---
    def bench(block, x_in, name, warmup_iters, bench_iters):
        torch.cuda.synchronize()
        # Warmup
        for _ in range(warmup_iters):
            out = block(x_in)
            out.sum().backward()
            block.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # Timed
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
        for i in range(bench_iters):
            start_events[i].record()
            out = block(x_in)
            out.sum().backward()
            block.zero_grad(set_to_none=True)
            end_events[i].record()
        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_ms = sum(times) / len(times)
        std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
        print(f"  {name}: {avg_ms:.2f} ± {std_ms:.2f} ms  (fwd+bwd per step)")
        return avg_ms

    # --- Benchmark ---
    print("=" * 60)
    print("THROUGHPUT BENCHMARK (forward + backward)")
    print("=" * 60)
    x_bench = torch.randn(T, n, C, device=device, dtype=dtype, requires_grad=False)
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        t_old = bench(old_block, x_bench, "Old MHCLiteBlock", warmup, iters)
        t_new = bench(new_block, x_bench, "New MHCLiteBlock", warmup, iters)

    print()
    speedup = t_old / t_new
    print(f"  SPEEDUP: {speedup:.2f}x")
    print()

    # --- Memory usage ---
    print("=" * 60)
    print("PEAK MEMORY COMPARISON")
    print("=" * 60)
    for name, block in [("Old", old_block), ("New", new_block)]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            out = block(x_bench)
            out.sum().backward()
            block.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  {name}: {peak_mb:.1f} MB peak")

    print("\nDone.")


if __name__ == "__main__":
    main()
