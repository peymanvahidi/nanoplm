# mHC-lite Triton kernels (current state)

This document tracks the **current** mHC-lite Triton setup in this repo:
- active kernels
- where each kernel is launched
- current launch-config logic (hardcoded vs heuristic path)
- one-shot gridsearch workflow for finding hardcoded params

## End-to-end mHC-lite path

Main model path:
- `src/nanoplm/pretraining/models/modern_bert/modeling.py` (`MHCLiteBlock._forward_triton`)

Per-layer flow:
1. **K1 (Triton):** fused RMSNorm + projection  
   `torch.ops.nanoplm_mhc.fused_rmsnorm_project(...)`
2. **K2 (PyTorch):** coefficient math (`sigmoid/softmax`, `a_res @ perm_mat`)
3. **K3 (Triton):** pre-map  
   `torch.ops.nanoplm_mhc.fused_pre_map(...)`
4. Wrapped transformer layer
5. **K4 (Triton):** post-res  
   `torch.ops.nanoplm_mhc.fused_post_res(...)`

Custom op registration + autograd lives in:
- `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py`

Kernel implementations live in:
- `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py`

## Active Triton kernels

The mHC-lite Triton stack currently uses **6 kernels**:

1. `_fused_rmsnorm_project_fwd_kernel`
2. `_fused_rmsnorm_project_bwd_dx_kernel`
3. `_fused_pre_map_fwd_kernel`
4. `_fused_pre_map_bwd_fused_kernel_n4`
5. `_fused_post_res_fwd_kernel_n4`
6. `_fused_post_res_bwd_fused_kernel_n4`

Important update:
- K4 backward is now a **single fused kernel** (`_fused_post_res_bwd_fused_kernel_n4`).
- The old split K4 backward kernels were removed:
  - `_fused_post_res_bwd_xlo_kernel_n4`
  - `_fused_post_res_bwd_Hhp_kernel`

Important update:
- K3 backward is now a **single fused kernel**
  (`_fused_pre_map_bwd_fused_kernel_n4`) to avoid rereading `grad_out`.

## Launch config logic (what is hardcoded vs heuristic)

Source of truth: launcher code in `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py`.

### K1
- `fused_rmsnorm_project` fwd:
  - **SM120 (`cc_major >= 12`) hardcoded:** `BLOCK_T=128`, `BLOCK_K=64`, `num_warps=4`, `num_stages=3`
  - **other arch:** `BLOCK_T=128`, `BLOCK_K=min(128, next_power_of_2(nC))`, warps/stages from `_get_hw_config()`
- `fused_rmsnorm_project_bwd_dx`:
  - **SM120 (`cc_major >= 12`) hardcoded:** `BLOCK_T=64`, `BLOCK_K=128`, `num_warps=8`, `num_stages=3`
  - **SM90:** `BLOCK_T=64`, `BLOCK_K=min(128, next_power_of_2(nC))`, `num_stages` from `_get_hw_config()`
  - **other arch:** `BLOCK_T=128`, `BLOCK_K=min(128, next_power_of_2(nC))`, warps/stages from `_get_hw_config()`

### K3
- `fused_pre_map` fwd:
  - **SM120 (`cc_major >= 12`) hardcoded:** `BLOCK_T=128`, `BLOCK_C=64`, `num_warps=4`, `num_stages=2`
  - **SM90 hardcoded path:** `BLOCK_T=128`, `BLOCK_C=128`, `num_warps=8`, `num_stages=4`
  - **heuristic path (other arch):** `BLOCK_T=64`, `BLOCK_C=min(256, next_power_of_2(C))`, `num_stages` from `_get_hw_config()`
- `fused_pre_map_backward`:
  - single fused kernel `_fused_pre_map_bwd_fused_kernel_n4` (saves a full reread of `grad_out`)
  - **SM120 (`cc_major >= 12`) hardcoded (fused):** `BLOCK_T=32`, `BLOCK_C=256`, `num_warps=8`, `num_stages=4`
  - **SM90 (`cc_major == 9`) hardcoded (fused):** `BLOCK_T=64`, `BLOCK_C=128`, `num_warps=8`, `num_stages=3`
  - **other arch (fused):** `BLOCK_T=64`, `BLOCK_C=min(256, next_power_of_2(C))`, warps/stages from `_get_hw_config()`

### K4
- `fused_post_res` fwd:
  - **SM120 (`cc_major >= 12`) hardcoded:** `BLOCK_T=32`, `BLOCK_C=128`, `num_warps=8`, `num_stages=3`
  - **other arch:** `BLOCK_T=64` on `cc_major >= 9`, else `32`; `BLOCK_C=128` when `C >= 128`, else `next_power_of_2(C)`; warps `8` on `cc_major >= 9`, else default
- `fused_post_res_backward` (fused K4 bwd):
  - **SM120 (`cc_major >= 12`) hardcoded:** `BLOCK_T=16`, `BLOCK_C=256`, `num_warps=8`, `num_stages=4`
  - **SM90:** `BLOCK_T=32`, `BLOCK_C=128`, `num_stages=2`
  - **other arch:** `BLOCK_T=32`, `BLOCK_C=min(256, next_power_of_2(C))`, `num_stages` default

## One-shot gridsearch workflow

Script:
- `tests/mhc_triton_gridsearch.py`

What it searches:
- launch configs for the 7 active kernels above
- candidate dimensions include `BLOCK_T`, `BLOCK_K`/`BLOCK_C`, `num_warps`, `num_stages`

Example:
```bash
python3 tests/mhc_triton_gridsearch.py \
  --T 65536 --C 1024 --n 4 \
  --warmup 8 --iters 25 \
  --json-out output/logs/mhc_gridsearch_h100_65536_1024.json
```

Behavior:
- runs a one-time sweep
- prints best config per kernel + top-N trials
- writes JSON payload (device, shape, timing, per-kernel best/top/failed counts)

Intended usage:
1. Run gridsearch once on target hardware/shape.
2. Copy best configs into launcher logic in `mhc_triton_ops.py`.
3. Keep runtime path simple (no online autotune during training).

## Runtime Triton autotune (capped search space)

The runtime mHC custom-op path now supports Triton autotune launch wrappers for:
- K1 fwd / K1 bwd_dx
- K3 fwd / K3 bwd_fused
- K4 fwd / K4 bwd_fused

Each kernel's autotune candidate set is capped at **32 configs max** to keep first-run tuning latency bounded.

Behavior:
- Default: autotune enabled (`NANOPLM_MHC_TRITON_AUTOTUNE=1`)
- Fallback to legacy hardcoded launch params: `NANOPLM_MHC_TRITON_AUTOTUNE=0`
- Runtime autotune wrappers enable Triton disk caching (`cache_results=True`), so first-run tuning is reused on later runs for matching keys.
- Cache location follows Triton defaults (`TRITON_CACHE_DIR`, usually `~/.triton/cache`).
- Runtime prints one-time status lines for new shape keys so first-run autotune does not look like a hang (`NANOPLM_MHC_TRITON_AUTOTUNE_STATUS=1`, set to `0` to silence).
- During `model.eval()`, mHC Triton path temporarily disables autotune to avoid eval-time warmup/tuning latency.

Implementation:
- Autotuned wrappers: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py`
- Runtime dispatch toggle: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py`

## Benchmarking before/after hardcoding

Microbenchmark script:
- `tests/mhc_triton_kernels_benchmark.py`

Use it to compare:
1. current hardcoded launcher values
2. new hardcoded values derived from gridsearch output

Example:
```bash
python3 tests/mhc_triton_kernels_benchmark.py --T 65536 --C 1024 --n 4 --iters 50 --warmup 10
```

The benchmark prints per-kernel:
- avg/median/min/max latency
- achieved bandwidth
- optional roofline efficiency (if `--peak-gbps` is provided)

## Constraints

- mHC Triton kernels are specialized for `n=4`.
- Custom ops enforce `n=4` and raise on other values.
Per layer:
1) **K1 (Triton)** fused RMSNorm + projection  
   Call site: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1007`  
   Op: `torch.ops.nanoplm_mhc.fused_rmsnorm_project(x_flat, W_all.weight)`
2) **K2 (PyTorch)** coefficients + permutation mixing  
   - `sigmoid/softmax` on the small projected head (size `2n + n!`)  
   - `H_res = a_res @ perm_mat`  
   Code: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1020`
3) **K3 (Triton)** pre-map (weighted stream aggregation)  
   Call site: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1024`  
   Op: `torch.ops.nanoplm_mhc.fused_pre_map(x_streams, h_pre)`
4) **Transformer layer** (wrapped encoder layer, no residual inside the wrapper)
5) **K4 (Triton)** post-res (merged stream mixing + post scaling)  
   Call site: `src/nanoplm/pretraining/models/modern_bert/modeling.py:1030`  
   Op: `torch.ops.nanoplm_mhc.fused_post_res(x_streams, layer_output, H_merged, h_post)`

## Triton custom-op stack (responsibility chain)

To keep `torch.compile` happy (no graph breaks), the fused kernels are exposed as dispatcher ops:
- Op definitions + FakeTensor support + autograd registration: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:1`
- Triton kernel implementations: `src/nanoplm/pretraining/models/modern_bert/mhc_triton_kernels.py:1`
- `modeling.py` imports ops for registration: `src/nanoplm/pretraining/models/modern_bert/modeling.py:22`

## Chrome traces (main GPU compute stream = 7, legacy snapshots)

These traces are historical snapshots from an earlier implementation period.
They are useful for context/perf archaeology, but do **not** represent the
current active K4 backward path (which is now fused).

The repo contains exactly two profiler traces:
- `output/pretraining_checkpoints/run-26021148-2/profiler_traces/chrome_trace.json`
- `output/pretraining_checkpoints/run-26021149-2/profiler_traces/chrome_trace.json`

### `run-26021148-2` (stream 7): legacy mHC-lite path (pre-K4-bwd fusion)

On stream 7, the trace contains the mHC-lite Triton kernels below (total: **142 calls**, ~**218.4 ms** on stream 7):
- `_fused_rmsnorm_project_fwd_kernel` (17 calls, ~14.4 ms)
- `_fused_rmsnorm_project_bwd_dx_kernel` (18 calls, ~32.9 ms)
- `_fused_pre_map_fwd_kernel` (17 calls, ~17.6 ms)
- `_fused_post_res_fwd_kernel_n4` (18 calls, ~36.2 ms)
- `_fused_post_res_bwd_xlo_kernel_n4` (18 calls, ~37.4 ms)
- `_fused_post_res_bwd_Hhp_kernel` (18 calls, ~39.6 ms)

The trace also contains stream-7 kernels caused by the remaining **PyTorch** parts of mHC-lite (identified by matching `aten::mm` shapes to mHC-lite math):
- `H_res = a_res @ perm_mat` (`[T,24] x [24,16]`): CUTLASS WMMA bf16 kernel (16 calls, ~0.048 ms total)
- Backward of that matmul (`[T,16] x [16,24]`): CUTLASS WMMA bf16 kernel (16 calls, ~0.048 ms total)
- `grad_W_all = (grad_out * inv_rms)^T @ x_flat` (from K1 backward): CUTLASS bf16 GEMM kernel (24 calls, ~15.0 ms total) + a small `cublasLt::splitKreduce_kernel` (~0.04 ms total)

### `run-26021149-2` (stream 7): mHC-lite fused path absent

No `_fused_{rmsnorm,pre_map,post_res}_*` kernel names appear on stream 7 in this trace, so it does **not** include the fused mHC-lite path.

## RTX 5090 theoretical minimum time + efficiency (legacy split-kernel baseline, T=65536, C=2560, n=4)

This section estimates a *roofline-style* lower bound and compares it to the measured kernel durations from:
- `output/pretraining_checkpoints/run-26021148-2/profiler_traces/chrome_trace.json` (GPU stream 7)

Assumptions:
- Shapes: `T=65536`, `C=2560`, `n=4` (`nC=10240`), `D_out=32` (because `2n + n! = 8 + 24`)
- Data types as used by the fused path: `x_streams/x_flat` bf16, `h_pre/h_post/H_merged` float32 (passed as `.float()`), outputs bf16
- Peak DRAM bandwidth for RTX 5090: **~1792 GB/s**
- “Theoretical mathematical minimum time” = `(mandatory bytes read+written) / peak_bandwidth`
  - Ignores launch overhead, instruction/latency limits, and cache effects (e.g. `W_all.weight` can be L2-resident), so treat this as an approximation.

| kernel | avg time (ms) | min time @1792 GB/s (ms) | efficiency (=min/avg) | achieved BW (GB/s) |
|---|---:|---:|---:|---:|
| `_fused_rmsnorm_project_fwd_kernel` | 0.847 | 0.752 | 88.8% | 1590.5 |
| `_fused_rmsnorm_project_bwd_dx_kernel` | 1.825 | 1.503 | 82.3% | 1475.6 |
| `_fused_pre_map_fwd_kernel` | 1.037 | 0.937 | 90.3% | 1619.0 |
| `_fused_post_res_fwd_kernel_n4` | 2.009 | 1.688 | 84.0% | 1506.1 |
| `_fused_post_res_bwd_xlo_kernel_n4` | 2.075 | 1.688 | 81.3% | 1457.7 |
| `_fused_post_res_bwd_Hhp_kernel` | 2.199 | 1.688 | 76.8% | 1375.6 |

## SM90 (H100) optimizations + trace validation (legacy split-kernel context, T=65536, C=2560, n=4)

### Optimizations applied

The original kernel configs were tuned for SM120 (RTX 5090). On SM90 (H100 80GB HBM3, 132 SMs, 3350 GB/s peak BW), three kernels had poor efficiency due to register pressure and shared memory limits. All fixes are gated behind `cc_major == 9` so the SM120 path is unchanged.

**1. `_fused_post_res_bwd_Hhp_kernel` — restructured inner C-loop (18% → 83%)**

Root cause: the original inner loop loaded 9 2D tiles simultaneously (lo + x0..x3 + go0..go3), each `(BLOCK_T=64, BLOCK_C=256)` bf16→f32. With 8 warps (256 threads): 9 tiles × 64 × 256 / 256 threads = 576 f32 regs/thread. H100 limit is 255 → massive register spilling.

Fix: phased tile streaming. Load 4 `go` tiles first (stay live), then stream `lo` once (compute `ghp`, `lo` dies), then stream `x_j` one at a time (max live: 4 go + 1 x = 5 tiles). SM90 launch config: `BLOCK_C=128`, `num_stages=2`. The restructured kernel body applies to all architectures (it's a strict improvement in load ordering) but the reduced `BLOCK_C` is SM90-only.

**2. `_fused_rmsnorm_project_bwd_dx_kernel` — reduced BLOCK_T (57% → 77%)**

Root cause: `BLOCK_T=128` creates a `(128, D_out=32)` f32 dot output = 128×32/256 = 16 regs, but the full pipeline (x tile, W tile, grad_out, proj_out, accumulators) hits ~170 regs/thread.

Fix: `BLOCK_T=64` for SM90 only. Halves register pressure, enables 2-block occupancy per SM.

**3. `_fused_pre_map_fwd_kernel` — reduced BLOCK_C + pipeline tuning (61% → 69%)**

Root cause: `BLOCK_C=256` with `num_stages=4` needs 4 × (n=4 x tiles) × 64×256×2B = 512KB pipeline buffers. H100 has 228KB shared mem → pipelining silently disabled by Triton.

Fix: `BLOCK_C=128`, `num_stages=3` for SM90 only → 4 × 3 × 16KB = 192KB, fits in 228KB.

### Trace validation (`run-26021500-2`, H100 80GB)

Trace: `output/pretraining_checkpoints/run-26021500-2/profiler_traces/chrome_trace.json`
Config: `hidden_size=2560`, `mhc_n_streams=4`, `num_hidden_layers=4`, `micro_batch_size=128`, packing enabled → `T=65536` tokens/microbatch.
Profiler captured 9 training steps (steps 10-15 per config). 132 SMs, 8 warps, stream 7.

| kernel | bytes (MB) | LB (us) | trace avg (us) | eff % | clean* eff % | achieved BW (GB/s) |
|---|---:|---:|---:|---:|---:|---:|
| `rmsnorm_project_fwd` | 1347 | 402 | 459 | 82% | **88%** | 2937 |
| `rmsnorm_project_bwd_dx` | 2694 | 804 | 1040 | 77% | **77%** | 2591 |
| `pre_map_fwd` | 1679 | 501 | 723 | 69% | **69%** | 2322 |
| `post_res_fwd_n4` | 3025 | 903 | 1063 | 85% | **85%** | 2846 |
| `post_res_bwd_xlo_n4` | 3361 | 1003 | 1065 | 94% | **94%** | 3156 |
| `post_res_bwd_Hhp` | 3025 | 903 | 1089† | 67%→ | **83%** | 2779 |
| **TOTAL** | **18489** | **5519** | **6648** | | **83%** | **2781** |

\* "Clean" = excluding invocations that overlap with NCCL ReduceScatter on stream 26.

† The Hhp kernel shows bimodal latency: 21 fast invocations at 1089 us (83% eff) and 15 slow invocations at ~1690 us (53% eff). Every slow invocation overlaps with `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` on a separate stream, confirming the slowdown is memory bandwidth contention from communication overlap — not a kernel issue.

**Per-layer total: 6.65 ms (clean) / 5.52 ms (lower bound) = 83% overall bandwidth efficiency.**

### Benchmark vs trace comparison

The microbenchmark (`tests/mhc_triton_kernels_benchmark.py`, T=2048, C=256) accurately predicts real-training efficiency:

| kernel | benchmark eff % | trace clean eff % | delta |
|---|---:|---:|---:|
| `rmsnorm_project_fwd` | 89% | 88% | -1% |
| `rmsnorm_project_bwd_dx` | 78% | 77% | -1% |
| `pre_map_fwd` | 69% | 69% | 0% |
| `post_res_fwd_n4` | 84% | 85% | +1% |
| `post_res_bwd_xlo_n4` | 94% | 94% | 0% |
| `post_res_bwd_Hhp` | 84% | 83% | -1% |

All within ±2%. The benchmark is a reliable proxy for real training performance.

### Architecture-specific launch configs

Current code has explicit hardcoded paths for both SM90 and SM120 in `mhc_triton_ops.py`.

| kernel | param | SM90 (H100) | SM120 (RTX 5090) |
|---|---|---|---|
| K1 fwd (`rmsnorm_project_fwd`) | `BLOCK_K`, warps, stages | heuristic (`BLOCK_K=min(128,nC_pow2)`) | `BLOCK_K=64`, `warps=4`, `stages=3` |
| K1 bwd (`rmsnorm_project_bwd_dx`) | `BLOCK_T` | 64 | 64 |
| K1 bwd (`rmsnorm_project_bwd_dx`) | `BLOCK_K`, warps, stages | heuristic (`BLOCK_K=min(128,nC_pow2)`) | `BLOCK_K=128`, `warps=8`, `stages=3` |
| K3 fwd (`pre_map_fwd`) | `BLOCK_C`, warps, stages | `BLOCK_C=128`, `warps=8`, `stages=4` | `BLOCK_C=64`, `warps=4`, `stages=2` |
| K3 bwd (`pre_map_bwd_fused_n4`) | `BLOCK_T/BLOCK_C`, warps, stages | `64/128`, `warps=8`, `stages=3` | `32/256`, `warps=8`, `stages=4` |
| K4 fwd (`post_res_fwd_n4`) | `BLOCK_T/BLOCK_C`, warps, stages | heuristic | `32/128`, `warps=8`, `stages=3` |
| K4 bwd (`post_res_bwd_fused_n4`) | `BLOCK_T/BLOCK_C`, warps, stages | `32/128`, `stages=2` | `16/256`, `warps=8`, `stages=4` |

The Hhp kernel body restructuring (phased 4+1 tile streaming) applies to all architectures — it reduces max live 2D tiles from 9 to 5 without changing the computation, which is neutral-to-beneficial everywhere.

## SM120 retuning summary (RTX 5090, T=65536, C=1024, n=4)

This is a quick summary of the follow-up tuning pass done on SM120 (RTX 5090):

1. Ran baseline microbenchmark:
   - `python tests/mhc_triton_kernels_benchmark.py --T 65536 --C 1024 --n 4 --peak-gbps 1792`
2. Ran one-shot grid search:
   - `python tests/mhc_triton_gridsearch.py --T 65536 --C 1024 --n 4 --json-out output/mhc_gridsearch_T65536_C1024_n4.json`
3. Re-ran benchmark with best params from grid search and compared before vs after.
4. Hardcoded SM120 launch configs in:
   - `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py`

### Best SM120 launch configs selected

- `k1_fwd`: `BLOCK_T=128`, `BLOCK_K=64`, `warps=4`, `stages=3`
- `k1_bwd_dx`: `BLOCK_T=64`, `BLOCK_K=128`, `warps=8`, `stages=3`
- `k3_fwd`: `BLOCK_T=128`, `BLOCK_C=64`, `warps=4`, `stages=2`
- `k3_bwd_fused`: `BLOCK_T=32`, `BLOCK_C=256`, `warps=8`, `stages=4`
- `k4_fwd`: `BLOCK_T=32`, `BLOCK_C=128`, `warps=8`, `stages=3`
- `k4_bwd_fused`: `BLOCK_T=16`, `BLOCK_C=256`, `warps=8`, `stages=4`

### Before vs after (peak BW assumption = 1792 GB/s)

| kernel | before avg (ms) | after avg (ms) | speedup | before eff % | after eff % |
|---|---:|---:|---:|---:|---:|
| `_fused_rmsnorm_project_fwd_kernel` | 0.342 | 0.337 | 1.5% | 88.4 | 89.7 |
| `_fused_rmsnorm_project_bwd_dx_kernel` | 0.783 | 0.747 | 4.6% | 77.2 | 80.9 |
| `_fused_pre_map_fwd_kernel` | 0.466 | 0.432 | 7.3% | 80.5 | 86.8 |
| `_fused_pre_map_bwd_fused_kernel_n4` | 0.911 | 0.889 | 2.5% | 82.4 | 84.5 |
| `_fused_post_res_fwd_kernel_n4` | 0.817 | 0.820 | -0.4% | 82.9 | 82.6 |
| `_fused_post_res_bwd_fused_kernel_n4` | 4.486 | 1.251 | 72.1% | 23.5 | 84.3 |
| **TOTAL** | **7.804** | **4.484** | **42.5%** | **48.2** | **83.9** |

Key result: the SM120 bottleneck was `k4_bwd_fused`; after retuning it moved from ~23.5% to ~84.3% efficiency, and total kernel time dropped by ~42.5% for this shape.

## Kernel → responsible code

### mHC-lite Triton kernels (directly “ownable”)

| Trace kernel name | Purpose | Kernel definition | Launcher (torch op impl) | Model callsite |
|---|---|---:|---:|---:|
| `_fused_rmsnorm_project_fwd_kernel` | K1 forward: RMSNorm(x_flat) + dot with `W_all.weight` | `mhc_triton_kernels.py::_fused_rmsnorm_project_fwd_kernel` | `mhc_triton_ops.py::_fused_rmsnorm_project_cuda` | `modeling.py` (`_forward_triton`) |
| `_fused_rmsnorm_project_bwd_dx_kernel` | K1 backward: dX for fused RMSNorm+proj | `mhc_triton_kernels.py::_fused_rmsnorm_project_bwd_dx_kernel` | `mhc_triton_ops.py::_fused_rmsnorm_project_bwd_dx_cuda` | autograd via `torch.library.register_autograd` |
| `_fused_pre_map_fwd_kernel` | K3 forward: `layer_input[t,c] = Σ_j h_pre[t,j] * x[t,j,c]` | `mhc_triton_kernels.py::_fused_pre_map_fwd_kernel` | `mhc_triton_ops.py::_fused_pre_map_cuda` | `modeling.py` (`_forward_triton`) |
| `_fused_pre_map_bwd_fused_kernel_n4` | K3 backward: fused dX_streams + d(h_pre) | `mhc_triton_kernels.py::_fused_pre_map_bwd_fused_kernel_n4` | `mhc_triton_ops.py::_fused_pre_map_backward_cuda` | autograd via `torch.library.register_autograd` |
| `_fused_post_res_fwd_kernel_n4` | K4 forward (n=4): `H_merged @ x + h_post * layer_output` | `mhc_triton_kernels.py::_fused_post_res_fwd_kernel_n4` | `mhc_triton_ops.py::_fused_post_res_cuda` | `modeling.py` (`_forward_triton`) |
| `_fused_post_res_bwd_fused_kernel_n4` | K4 fused backward: dX + d(layer_output) + d(H_merged) + d(h_post) | `mhc_triton_kernels.py::_fused_post_res_bwd_fused_kernel_n4` | `mhc_triton_ops.py::_fused_post_res_backward_cuda` | autograd via `torch.library.register_autograd` |

### mHC-lite-specific (non-Triton) GPU kernels seen on stream 7

These kernels are not defined in this repo (they come from CUTLASS/cuBLAS), but they are still attributable to specific mHC-lite code paths:
- `H_res = a_res @ perm_mat` (and its backward matmul): `src/nanoplm/pretraining/models/modern_bert/modeling.py:1020`
- `grad_W_all = (grad_out * inv_rms)^T @ x_flat` (K1 backward weight grad): `src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py:365`
