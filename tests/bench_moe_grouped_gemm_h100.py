from __future__ import annotations

import argparse
import os
import time

import torch

from nanoplm.pretraining.models.modern_bert.moe_grouped_gemm_ops import moe_grouped_gemm


_VARIANTS = (
    "wide_coop",
    "wide_pingpong",
    "tall_coop",
    "tall_pingpong",
    "square_coop",
    "square_pingpong",
    "smallm_pingpong",
)


def _balanced_counts(total_rows: int, experts: int, device: torch.device) -> torch.Tensor:
    base = total_rows // experts
    rem = total_rows % experts
    counts = torch.full((experts,), base, device=device, dtype=torch.int64)
    if rem:
        counts[:rem] += 1
    return counts


def _reference_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    counts: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    sizes = [int(x) for x in counts.detach().cpu().tolist()]
    if trans_a:
        out = torch.empty((len(sizes), a.shape[1], b.shape[1]), device=a.device, dtype=a.dtype)
        offset = 0
        for idx, rows in enumerate(sizes):
            if rows == 0:
                out[idx].zero_()
            else:
                out[idx] = a[offset : offset + rows].transpose(0, 1) @ b[offset : offset + rows]
            offset += rows
        return out

    out_cols = b.shape[1] if trans_b else b.shape[2]
    out = torch.empty((a.shape[0], out_cols), device=a.device, dtype=a.dtype)
    offset = 0
    for idx, rows in enumerate(sizes):
        rhs = b[idx].transpose(0, 1) if trans_b else b[idx]
        out[offset : offset + rows] = a[offset : offset + rows] @ rhs
        offset += rows
    return out


def _check_case(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    counts: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> None:
    out = moe_grouped_gemm(a, b, counts, trans_a=trans_a, trans_b=trans_b)
    ref = _reference_grouped_gemm(a, b, counts, trans_a, trans_b)
    torch.cuda.synchronize()
    max_abs = (out.float() - ref.float()).abs().max().item()
    max_ref = ref.float().abs().max().item()
    rel = max_abs / max(max_ref, 1e-6)
    ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    print(f"check {name:12s} ok={ok} max_abs={max_abs:.4e} rel={rel:.4e}")
    if not ok:
        raise SystemExit(f"{name} failed correctness check")


def _grad_check_case(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    counts: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> None:
    a_fast = a.detach().clone().requires_grad_(True)
    b_fast = b.detach().clone().requires_grad_(True)
    y_fast = moe_grouped_gemm(a_fast, b_fast, counts, trans_a=trans_a, trans_b=trans_b)
    y_fast.float().square().mean().backward()

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    y_ref = _reference_grouped_gemm(a_ref, b_ref, counts, trans_a, trans_b)
    y_ref.float().square().mean().backward()
    torch.cuda.synchronize()

    checks = {
        "out": (y_fast, y_ref),
        "grad_a": (a_fast.grad, a_ref.grad),
        "grad_b": (b_fast.grad, b_ref.grad),
    }
    for label, (got, expected) in checks.items():
        max_abs = (got.float() - expected.float()).abs().max().item()
        max_ref = expected.float().abs().max().item()
        rel = max_abs / max(max_ref, 1e-6)
        ok = torch.allclose(got.float(), expected.float(), atol=2e-2, rtol=2e-2)
        print(f"grad_check {name:12s} {label:6s} ok={ok} max_abs={max_abs:.4e} rel={rel:.4e}")
        if not ok:
            raise SystemExit(f"{name} failed {label} gradient check")


def _bench_case(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    counts: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    warmup: int,
    iters: int,
) -> None:
    for _ in range(warmup):
        moe_grouped_gemm(a, b, counts, trans_a=trans_a, trans_b=trans_b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        moe_grouped_gemm(a, b, counts, trans_a=trans_a, trans_b=trans_b)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    print(f"{name:18s} {ms:8.3f} ms")


def _set_optional_env(name: str, value: str | None) -> None:
    if value is not None:
        os.environ[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="H100 MoE grouped GEMM exact-shape benchmark")
    parser.add_argument("--backend", default=None, choices=["auto", "sm80", "sm90"])
    parser.add_argument("--tokens", type=int, default=65_536)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--wi-variant", choices=_VARIANTS)
    parser.add_argument("--wo-variant", choices=_VARIANTS)
    parser.add_argument("--trans-a-variant", choices=_VARIANTS)
    parser.add_argument("--trans-b-variant", choices=_VARIANTS)
    parser.add_argument("--sweep-variants", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--grad-check", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    backend = args.backend or os.environ.get("NANOPLM_MOE_GROUPED_GEMM_BACKEND", "auto")
    os.environ["NANOPLM_MOE_GROUPED_GEMM_BACKEND"] = backend
    _set_optional_env("NANOPLM_MOE_GROUPED_GEMM_WI_VARIANT", args.wi_variant)
    _set_optional_env("NANOPLM_MOE_GROUPED_GEMM_WO_VARIANT", args.wo_variant)
    _set_optional_env("NANOPLM_MOE_GROUPED_GEMM_TRANS_A_VARIANT", args.trans_a_variant)
    _set_optional_env("NANOPLM_MOE_GROUPED_GEMM_TRANS_B_VARIANT", args.trans_b_variant)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows = args.tokens * args.top_k
    counts = _balanced_counts(rows, args.experts, device)
    h = args.hidden
    i = args.intermediate

    print(
        f"backend={backend} device={torch.cuda.get_device_name()} "
        f"rows={rows} experts={args.experts} dtype={dtype}"
    )
    print(
        "variants "
        f"wi={os.environ.get('NANOPLM_MOE_GROUPED_GEMM_WI_VARIANT', 'tall_coop')} "
        f"wo={os.environ.get('NANOPLM_MOE_GROUPED_GEMM_WO_VARIANT', 'tall_coop')} "
        f"trans_a={os.environ.get('NANOPLM_MOE_GROUPED_GEMM_TRANS_A_VARIANT', 'wide_coop')} "
        f"trans_b={os.environ.get('NANOPLM_MOE_GROUPED_GEMM_TRANS_B_VARIANT', 'auto')}"
    )

    x_h = torch.randn(rows, h, device=device, dtype=dtype)
    x_i = torch.randn(rows, i, device=device, dtype=dtype)
    go_2i = torch.randn(rows, 2 * i, device=device, dtype=dtype)
    go_h = torch.randn(rows, h, device=device, dtype=dtype)
    wi = torch.randn(args.experts, h, 2 * i, device=device, dtype=dtype)
    wo = torch.randn(args.experts, i, h, device=device, dtype=dtype)

    # Give the JIT extension a chance to build before timing.
    t0 = time.time()
    moe_grouped_gemm(x_h, wi, counts)
    torch.cuda.synchronize()
    print(f"first_call_build_and_run {time.time() - t0:.2f} s")

    if args.check:
        _check_case("Wi fwd", x_h, wi, counts, False, False)
        _check_case("Wo fwd", x_i, wo, counts, False, False)
        _check_case("dA Wi", go_2i, wi, counts, False, True)
        _check_case("dW Wi", x_h, go_2i, counts, True, False)
        _check_case("dA Wo", go_h, wo, counts, False, True)
        _check_case("dW Wo", x_i, go_h, counts, True, False)

    if args.grad_check:
        _grad_check_case("Wi fwd", x_h, wi, counts, False, False)
        _grad_check_case("Wo fwd", x_i, wo, counts, False, False)
        _grad_check_case("dA Wi", go_2i, wi, counts, False, True)
        _grad_check_case("dW Wi", x_h, go_2i, counts, True, False)
        _grad_check_case("dA Wo", go_h, wo, counts, False, True)
        _grad_check_case("dW Wo", x_i, go_h, counts, True, False)

    _bench_case("Wi fwd", x_h, wi, counts, False, False, args.warmup, args.iters)
    _bench_case("Wo fwd", x_i, wo, counts, False, False, args.warmup, args.iters)
    _bench_case("dA Wi", go_2i, wi, counts, False, True, args.warmup, args.iters)
    _bench_case("dW Wi", x_h, go_2i, counts, True, False, args.warmup, args.iters)
    _bench_case("dA Wo", go_h, wo, counts, False, True, args.warmup, args.iters)
    _bench_case("dW Wo", x_i, go_h, counts, True, False, args.warmup, args.iters)

    if args.sweep_variants:
        sweep_cases = [
            ("Wi fwd", "NANOPLM_MOE_GROUPED_GEMM_WI_VARIANT", x_h, wi, False, False),
            ("Wo fwd", "NANOPLM_MOE_GROUPED_GEMM_WO_VARIANT", x_i, wo, False, False),
            ("dA Wi", "NANOPLM_MOE_GROUPED_GEMM_TRANS_B_VARIANT", go_2i, wi, False, True),
            ("dA Wo", "NANOPLM_MOE_GROUPED_GEMM_TRANS_B_VARIANT", go_h, wo, False, True),
            ("dW Wi", "NANOPLM_MOE_GROUPED_GEMM_TRANS_A_VARIANT", x_h, go_2i, True, False),
            ("dW Wo", "NANOPLM_MOE_GROUPED_GEMM_TRANS_A_VARIANT", x_i, go_h, True, False),
        ]
        print("variant sweep")
        for case_name, env_name, a_case, b_case, trans_a, trans_b in sweep_cases:
            for variant in _VARIANTS:
                os.environ[env_name] = variant
                _bench_case(
                    f"{case_name} {variant}",
                    a_case,
                    b_case,
                    counts,
                    trans_a,
                    trans_b,
                    args.warmup,
                    args.iters,
                )


if __name__ == "__main__":
    main()
