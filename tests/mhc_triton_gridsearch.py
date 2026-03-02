"""One-shot grid search for mHC-lite Triton launch parameters.

Finds best launch configs for the kernels currently used by
`src/nanoplm/pretraining/models/modern_bert/mhc_triton_ops.py`:
  - K1 fwd: _fused_rmsnorm_project_fwd_kernel
  - K1 bwd: _fused_rmsnorm_project_bwd_dx_kernel
  - K3 fwd: _fused_pre_map_fwd_kernel
  - K3 bwd_fused: _fused_pre_map_bwd_fused_kernel_n4
  - K4 fwd: _fused_post_res_fwd_kernel_n4
  - K4 bwd_fused: _fused_post_res_bwd_fused_kernel_n4

Usage:
  python3 tests/mhc_triton_gridsearch.py --T 65536 --C 1024 --n 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


def _add_repo_to_path() -> None:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


@dataclass(frozen=True)
class LaunchConfig:
    block_t: int
    num_warps: int
    num_stages: int
    block_k: int | None = None
    block_c: int | None = None

    def summary(self) -> str:
        parts = [f"BLOCK_T={self.block_t}"]
        if self.block_k is not None:
            parts.append(f"BLOCK_K={self.block_k}")
        if self.block_c is not None:
            parts.append(f"BLOCK_C={self.block_c}")
        parts.append(f"warps={self.num_warps}")
        parts.append(f"stages={self.num_stages}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Trial:
    cfg: LaunchConfig
    ms: float


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _time_cuda(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    acc = 0.0
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        acc += float(start.elapsed_time(end))
    return acc / max(1, iters)


def _search(
    *,
    name: str,
    candidates: list[LaunchConfig],
    launcher_for: Callable[[LaunchConfig], Callable[[], None]],
    warmup: int,
    iters: int,
    topk: int,
) -> tuple[LaunchConfig, list[Trial], int]:
    trials: list[Trial] = []
    failed = 0
    total = len(candidates)
    print(f"\n[{name}] searching {total} configs")
    for idx, cfg in enumerate(candidates, start=1):
        try:
            fn = launcher_for(cfg)
            ms = _time_cuda(fn, warmup=warmup, iters=iters)
            trials.append(Trial(cfg=cfg, ms=ms))
        except Exception:
            failed += 1
        if idx == total or idx % 20 == 0:
            print(f"[{name}] progress {idx}/{total}")
    if not trials:
        raise RuntimeError(f"No valid candidate ran successfully for {name}")
    trials.sort(key=lambda t: t.ms)
    best = trials[0]
    print(f"[{name}] best {best.ms:.4f} ms :: {best.cfg.summary()}")
    return best.cfg, trials[: max(1, topk)], failed


def _space(args: argparse.Namespace, cc_major: int) -> dict[str, list[int]]:
    if args.full:
        warps = [2, 4, 8]
        stages = [1, 2, 3, 4]
    else:
        warps = [4, 8] if cc_major >= 9 else [2, 4]
        stages = [2, 3, 4] if cc_major >= 9 else [1, 2, 3]
    return {
        "warps": warps,
        "stages": stages,
        "k1_fwd_bt": [64, 128, 256],
        "k1_bwd_bt": [64, 128],
        "k1_bk": [32, 64, 128],
        "k3_bt": [32, 64, 128],
        "k3_bc": [64, 128, 256],
        "k4_fwd_bt": [32, 64],
        "k4_bwd_bt": [16, 32, 64],
        "k4_bc": [64, 128, 256],
    }


def _combos(
    *,
    block_ts: list[int],
    block_ks: list[int] | None,
    block_cs: list[int] | None,
    warps: list[int],
    stages: list[int],
    max_block_k: int | None = None,
    max_block_c: int | None = None,
) -> list[LaunchConfig]:
    out: list[LaunchConfig] = []
    for block_t, warp, stage in itertools.product(block_ts, warps, stages):
        if block_ks is not None:
            for block_k in block_ks:
                if max_block_k is not None and block_k > max_block_k:
                    continue
                out.append(
                    LaunchConfig(
                        block_t=block_t,
                        block_k=block_k,
                        num_warps=warp,
                        num_stages=stage,
                    )
                )
        elif block_cs is not None:
            for block_c in block_cs:
                if max_block_c is not None and block_c > max_block_c:
                    continue
                out.append(
                    LaunchConfig(
                        block_t=block_t,
                        block_c=block_c,
                        num_warps=warp,
                        num_stages=stage,
                    )
                )
        else:
            out.append(
                LaunchConfig(
                    block_t=block_t,
                    num_warps=warp,
                    num_stages=stage,
                )
            )
    return out


def main() -> None:
    _add_repo_to_path()
    import triton
    from nanoplm.pretraining.models.modern_bert import mhc_triton_kernels as k

    parser = argparse.ArgumentParser(description="One-shot grid search for mHC Triton launch params")
    parser.add_argument("--T", type=int, default=65536)
    parser.add_argument("--C", type=int, default=1024)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="Expand search space (slower).")
    parser.add_argument("--json-out", type=str, default="", help="Optional JSON output path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if args.n != 4:
        raise SystemExit("Current fused kernels are specialized for n=4.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    num_sms, _nw_default, _ns_default = k._get_hw_config()

    T = int(args.T)
    C = int(args.C)
    n = int(args.n)
    nC = n * C
    D_out = 2 * n + math.factorial(n)
    max_block_k = min(128, _next_pow2(nC))
    max_block_c = min(256, _next_pow2(C))

    print("mHC Triton grid search")
    print(f"device={torch.cuda.get_device_name(0)} cc={cc_major}.{cc_minor} sms={num_sms}")
    print(f"shape: T={T} C={C} n={n} nC={nC} D_out={D_out}")
    print(f"timing: warmup={args.warmup} iters={args.iters}")
    print(f"search mode: {'full' if args.full else 'default'}")

    # Allocate once and reuse for all candidates.
    x_streams = torch.randn((T, n, C), device=device, dtype=torch.bfloat16)
    x_flat = x_streams.reshape(T, nC).contiguous()
    W = torch.randn((D_out, nC), device=device, dtype=torch.bfloat16)
    out_proj = torch.empty((T, D_out), device=device, dtype=torch.bfloat16)
    inv_rms = torch.empty((T,), device=device, dtype=torch.float32)
    grad_out_proj = torch.randn((T, D_out), device=device, dtype=torch.bfloat16).contiguous()
    grad_x_flat = torch.empty_like(x_flat)

    h_pre = torch.randn((T, n), device=device, dtype=torch.float32)
    grad_out_pre = torch.randn((T, C), device=device, dtype=torch.bfloat16).contiguous()
    out_pre = torch.empty((T, C), device=device, dtype=torch.bfloat16)
    grad_x_streams = torch.empty_like(x_streams)
    grad_hpre = torch.empty((T, n), device=device, dtype=torch.float32)

    h_post = torch.randn((T, n), device=device, dtype=torch.float32)
    H = torch.randn((T, n, n), device=device, dtype=torch.float32)
    layer_output = torch.randn((T, C), device=device, dtype=torch.bfloat16)
    out_post = torch.empty_like(x_streams)
    grad_out_post = torch.randn((T, n, C), device=device, dtype=torch.bfloat16).contiguous()
    grad_x_fused = torch.empty_like(x_streams)
    grad_lo_fused = torch.empty((T, C), device=device, dtype=torch.bfloat16)
    grad_H_fused = torch.empty((T, n, n), device=device, dtype=torch.float32)
    grad_hp_fused = torch.empty((T, n), device=device, dtype=torch.float32)

    grid_sms = (num_sms,)
    space = _space(args, cc_major=cc_major)

    results: dict[str, dict[str, object]] = {}

    k1_fwd_candidates = _combos(
        block_ts=space["k1_fwd_bt"],
        block_ks=space["k1_bk"],
        block_cs=None,
        warps=space["warps"],
        stages=space["stages"],
        max_block_k=max_block_k,
    )

    best_cfg, top_trials, failed = _search(
        name="k1_fwd",
        candidates=k1_fwd_candidates,
        launcher_for=lambda cfg: (
            lambda: k._fused_rmsnorm_project_fwd_kernel[(triton.cdiv(T, cfg.block_t),)](
                x_flat,
                W,
                out_proj,
                inv_rms,
                T,
                nC,
                D_out,
                BLOCK_T=cfg.block_t,
                BLOCK_K=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
        ),
        warmup=args.warmup,
        iters=args.iters,
        topk=args.topk,
    )
    results["k1_fwd"] = {
        "best": asdict(best_cfg),
        "top": [{"ms": t.ms, "cfg": asdict(t.cfg)} for t in top_trials],
        "failed": failed,
        "total": len(k1_fwd_candidates),
    }

    k1_bwd_candidates = _combos(
        block_ts=space["k1_bwd_bt"],
        block_ks=space["k1_bk"],
        block_cs=None,
        warps=space["warps"],
        stages=space["stages"],
        max_block_k=max_block_k,
    )
    best_cfg, top_trials, failed = _search(
        name="k1_bwd_dx",
        candidates=k1_bwd_candidates,
        launcher_for=lambda cfg: (
            lambda: k._fused_rmsnorm_project_bwd_dx_kernel[(triton.cdiv(T, cfg.block_t),)](
                x_flat,
                W,
                grad_out_proj,
                out_proj,
                inv_rms,
                grad_x_flat,
                T,
                nC,
                D_out,
                BLOCK_T=cfg.block_t,
                BLOCK_K=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
        ),
        warmup=args.warmup,
        iters=args.iters,
        topk=args.topk,
    )
    results["k1_bwd_dx"] = {
        "best": asdict(best_cfg),
        "top": [{"ms": t.ms, "cfg": asdict(t.cfg)} for t in top_trials],
        "failed": failed,
        "total": len(k1_bwd_candidates),
    }

    k3_candidates = _combos(
        block_ts=space["k3_bt"],
        block_ks=None,
        block_cs=space["k3_bc"],
        warps=space["warps"],
        stages=space["stages"],
        max_block_c=max_block_c,
    )
    best_cfg, top_trials, failed = _search(
        name="k3_fwd",
        candidates=k3_candidates,
        launcher_for=lambda cfg: (
            lambda: k._fused_pre_map_fwd_kernel[grid_sms](
                x_streams,
                h_pre,
                out_pre,
                T,
                C,
                n,
                BLOCK_T=cfg.block_t,
                BLOCK_C=cfg.block_c,
                NUM_SMS=num_sms,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
        ),
        warmup=args.warmup,
        iters=args.iters,
        topk=args.topk,
    )
    results["k3_fwd"] = {
        "best": asdict(best_cfg),
        "top": [{"ms": t.ms, "cfg": asdict(t.cfg)} for t in top_trials],
        "failed": failed,
        "total": len(k3_candidates),
    }

    best_cfg, top_trials, failed = _search(
        name="k3_bwd_fused",
        candidates=k3_candidates,
        launcher_for=lambda cfg: (
            lambda: k._fused_pre_map_bwd_fused_kernel_n4[grid_sms](
                x_streams,
                h_pre,
                grad_out_pre,
                grad_x_streams,
                grad_hpre,
                T,
                C,
                n,
                BLOCK_T=cfg.block_t,
                BLOCK_C=cfg.block_c,
                NUM_SMS=num_sms,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
        ),
        warmup=args.warmup,
        iters=args.iters,
        topk=args.topk,
    )
    results["k3_bwd_fused"] = {
        "best": asdict(best_cfg),
        "top": [{"ms": t.ms, "cfg": asdict(t.cfg)} for t in top_trials],
        "failed": failed,
        "total": len(k3_candidates),
    }

    k4_fwd_candidates = _combos(
        block_ts=space["k4_fwd_bt"],
        block_ks=None,
        block_cs=space["k4_bc"],
        warps=space["warps"],
        stages=space["stages"],
        max_block_c=max_block_c,
    )
    best_cfg, top_trials, failed = _search(
        name="k4_fwd",
        candidates=k4_fwd_candidates,
        launcher_for=lambda cfg: (
            lambda: k._fused_post_res_fwd_kernel_n4[grid_sms](
                x_streams,
                layer_output,
                H,
                h_post,
                out_post,
                T,
                C,
                n,
                BLOCK_T=cfg.block_t,
                BLOCK_C=cfg.block_c,
                NUM_SMS=num_sms,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
        ),
        warmup=args.warmup,
        iters=args.iters,
        topk=args.topk,
    )
    results["k4_fwd"] = {
        "best": asdict(best_cfg),
        "top": [{"ms": t.ms, "cfg": asdict(t.cfg)} for t in top_trials],
        "failed": failed,
        "total": len(k4_fwd_candidates),
    }

    k4_bwd_candidates = _combos(
        block_ts=space["k4_bwd_bt"],
        block_ks=None,
        block_cs=space["k4_bc"],
        warps=space["warps"],
        stages=space["stages"],
        max_block_c=max_block_c,
    )
    best_cfg, top_trials, failed = _search(
        name="k4_bwd_fused",
        candidates=k4_bwd_candidates,
        launcher_for=lambda cfg: (
            lambda: k._fused_post_res_bwd_fused_kernel_n4[grid_sms](
                x_streams,
                layer_output,
                H,
                h_post,
                grad_out_post,
                grad_x_fused,
                grad_lo_fused,
                grad_H_fused,
                grad_hp_fused,
                T,
                C,
                n,
                BLOCK_T=cfg.block_t,
                BLOCK_C=cfg.block_c,
                NUM_SMS=num_sms,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )
        ),
        warmup=args.warmup,
        iters=args.iters,
        topk=args.topk,
    )
    results["k4_bwd_fused"] = {
        "best": asdict(best_cfg),
        "top": [{"ms": t.ms, "cfg": asdict(t.cfg)} for t in top_trials],
        "failed": failed,
        "total": len(k4_bwd_candidates),
    }

    print("\n=== Recommended hardcode block ===")
    print(f"# Device: {torch.cuda.get_device_name(0)} (cc {cc_major}.{cc_minor})")
    print(f"# Shape: T={T}, C={C}, n={n}")
    for key in [
        "k1_fwd",
        "k1_bwd_dx",
        "k3_fwd",
        "k3_bwd_fused",
        "k4_fwd",
        "k4_bwd_fused",
    ]:
        cfg = results[key]["best"]
        print(f"{key} = {cfg}")

    payload = {
        "device_name": torch.cuda.get_device_name(0),
        "cc": [cc_major, cc_minor],
        "num_sms": num_sms,
        "shape": {"T": T, "C": C, "n": n, "nC": nC, "D_out": D_out},
        "timing": {"warmup": args.warmup, "iters": args.iters},
        "results": results,
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON results: {out_path}")


if __name__ == "__main__":
    main()
