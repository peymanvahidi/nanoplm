"""Portable mode imports no custom kernel and runs the eager/SDPA paths.

`PORTABLE` is read once, when `_globals` is imported, so every portable-mode
check runs in one fresh interpreter (`_CHILD`) started with NANOPLM_PORTABLE=1.
Triton, flash-attn and friends ARE installed on the GPU nodes, so the child
blocks them before importing nanoplm instead of relying on their absence.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import nanoplm
from nanoplm.pretraining.models.modern_bert import _globals

MODERN_BERT_DIR = Path(_globals.__file__).resolve().parent
KERNEL_MODULES = (
    "canon_ops",
    "mhc_triton_ops",
    "canon_cute_ops",
    "triton_kernels",
    "moe_triton_kernels",
    "mhc_triton_kernels",
)
WITH_CUDA = torch.cuda.is_available()

_CHILD = r"""
import json
import sys
import traceback

from import_blocker import block_imports

BLOCKED = (
    "triton", "flash_attn", "flash_attn_interface", "kernels",
    "transformer_engine", "quack", "sonicmoe",
)
MODERN_BERT_DIR, KERNEL_MODULES = sys.argv[1], sys.argv[2].split(",")
WITH_CUDA = sys.argv[3] == "1"
attempts = block_imports(BLOCKED)  # (blocked name, file of the frame that imported it)

import torch  # noqa: E402

import nanoplm.hf as nhf  # noqa: E402
from nanoplm.pretraining.models.modern_bert import _globals  # noqa: E402
from nanoplm.pretraining.models.modern_bert import cutlass_grouped_gemm  # noqa: E402
from nanoplm.pretraining.models.modern_bert import moe_grouped_gemm_ops  # noqa: E402
from nanoplm.pretraining.models.modern_bert.model import (  # noqa: E402
    ProtModernBertMLMConfig,
)
from nanoplm.pretraining.models.modern_bert.modeling import MHCLiteBlock  # noqa: E402
from nanoplm.pretraining.models.modern_bert.pure_model import (  # noqa: E402
    PureProtModernBertMLM,
)

# nanoplm.pretraining.pure_pipeline switches TF32 on at import. Its 10-bit
# mantissa would swamp the fp32 CPU-vs-CUDA comparison below, and an exported
# model's consumer never imports the training pipeline.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=32,
)
CASES = {
    "dense": {},
    "srelu": {"mlp_activation": "srelu"},
    "moe": {
        "use_moe": True,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_kernel_backend": "cutlass",
    },
    "mhc_lite": {"use_mhc_lite": True},
    "canon": {"use_canon_layers": True},
}
IDS = torch.tensor([[4, 5, 6, 7, 8, 1], [9, 10, 11, 1, 0, 0]])
MASK = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
LABELS = torch.full_like(IDS, -100)
LABELS[0, 2] = 6
LABELS[1, 1] = 10


def _mismatch(actual, expected):
    try:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    except AssertionError as exc:
        return str(exc)
    return None


def _hf_model(overrides):
    torch.manual_seed(0)
    return nhf.NanoPLMForMaskedLM(nhf.NanoPLMConfig(**TINY, **overrides)).eval()


def _forward_case(overrides):
    model = _hf_model(overrides)
    with torch.no_grad():
        cpu = model(input_ids=IDS, attention_mask=MASK).logits
    out = {"cpu_finite": bool(torch.isfinite(cpu).all())}
    if WITH_CUDA:
        model.to("cuda")
        with torch.no_grad():
            cuda = model(input_ids=IDS.cuda(), attention_mask=MASK.cuda()).logits
        out["cuda_finite"] = bool(torch.isfinite(cuda).all())
        out["mismatch"] = _mismatch(cuda.cpu(), cpu)
    return out


def _moe_backward():
    # Backward runs the scatter_add_bwd / gather_combine_bwd CUDA impls.
    model = _hf_model(CASES["moe"])
    model(input_ids=IDS, attention_mask=MASK, labels=LABELS).loss.backward()
    cpu = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.to("cuda")
    model(
        input_ids=IDS.cuda(), attention_mask=MASK.cuda(), labels=LABELS.cuda()
    ).loss.backward()
    cuda = {n: p.grad.cpu() for n, p in model.named_parameters() if p.grad is not None}
    if cuda.keys() != cpu.keys():
        return {"mismatch": f"grad keys differ: {sorted(cuda.keys() ^ cpu.keys())}"}
    for name in cpu:
        msg = _mismatch(cuda[name], cpu[name])
        if msg is not None:
            return {"mismatch": f"{name}: {msg}"}
    return {"mismatch": None, "n_grads": len(cpu)}


def _grouped_gemm_bf16():
    # bf16 is the dtype the CUDA impl would hand to CUTLASS; the parent points
    # NANOPLM_CUTLASS_ROOT at a missing tree, so any build attempt raises.
    torch.manual_seed(0)
    a = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([3, 5], device="cuda")
    got = moe_grouped_gemm_ops.moe_grouped_gemm(a, b, counts)
    ref = moe_grouped_gemm_ops._reference_grouped_gemm(a, b, counts, False, False)
    return {
        "equal": bool(torch.equal(got, ref)),
        "extension_loaded": cutlass_grouped_gemm._EXTENSION is not None,
    }


def _native_cuda():
    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**TINY)).eval().cuda()
    with torch.no_grad():
        logits = model(input_ids=IDS.cuda(), attention_mask=MASK.cuda()).logits
    return {"finite": bool(torch.isfinite(logits).all())}


def _mhc_block_bf16():
    # (T, n, C) bf16 on CUDA with triton_fused=True is the one input
    # MHCLiteBlock routes to its Triton ops, which portable mode never loads.
    torch.manual_seed(0)
    block = MHCLiteBlock(4, 16, torch.nn.Linear(16, 16), triton_fused=True)
    block = block.to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        out = block(torch.randn(8, 4, 16, device="cuda", dtype=torch.bfloat16))
    return {"finite": bool(torch.isfinite(out).all()), "shape": list(out.shape)}


def _router_ties():
    # Only three distinct scores over eight experts, so nearly every row has
    # exact ties. The native Triton kernel (and the CPU reference) pick the
    # LOWEST index among tied experts; torch.topk promises no tie order, and on
    # CUDA it does not follow that rule. Portable mode must route identically
    # on every device, or CPU and CUDA consumers of one export would disagree.
    from nanoplm.pretraining.models.modern_bert import moe

    k = 3
    gen = torch.Generator().manual_seed(0)
    scores = torch.randint(0, 3, (256, 8), generator=gen).float() / 4
    rule = [
        sorted(range(len(row)), key=lambda j: (-row[j], j))[:k]
        for row in scores.tolist()
    ]
    cpu_idx, cpu_cnt = moe._router_topk_indices(scores, k, True)
    out = {"cpu_matches_rule": cpu_idx.tolist() == rule}
    if WITH_CUDA:
        cuda_idx, cuda_cnt = moe._router_topk_indices(scores.cuda(), k, True)
        out["cuda_matches_rule"] = cuda_idx.cpu().tolist() == rule
        out["counts_equal"] = bool(torch.equal(cuda_cnt.cpu(), cpu_cnt))
    return out


def _guarded(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return {"error": traceback.format_exc()}


report = {
    "cases": {name: _guarded(_forward_case, o) for name, o in CASES.items()},
    "router_ties": _guarded(_router_ties),
}
if WITH_CUDA:
    report["moe_backward"] = _guarded(_moe_backward)
    report["grouped_gemm_bf16"] = _guarded(_grouped_gemm_bf16)
    report["native_cuda"] = _guarded(_native_cuda)
    report["mhc_block_bf16"] = _guarded(_mhc_block_bf16)
report["flags"] = {
    name: repr(getattr(_globals, name, "<missing>"))
    for name in (
        "PORTABLE", "_HAS_FLASH_VARLEN", "_flash_varlen_fn", "_FLASH_HAS_DROPOUT",
        "USE_TRITON_SRELU", "_CANON_BACKEND", "_mhc_triton_ops", "_canon_ops",
        "_canon_cute_ops", "_varlen_canon_conv", "_varlen_ln_canon_conv",
        "_varlen_rms_canon_conv",
    )
}
report["loaded"] = sorted(
    m for m in sys.modules
    if m.split(".")[0] in BLOCKED
    or m in {f"nanoplm.pretraining.models.modern_bert.{k}" for k in KERNEL_MODULES}
)
# torch.library ops are process-global; portable mode must register none.
report["nanoplm_ops"] = sorted(
    op for op in torch._C._dispatch_get_all_op_names() if op.startswith("nanoplm")
)
report["modeling_tree_attempts"] = sorted(
    {f"{name} <- {path}" for name, path in attempts if path.startswith(MODERN_BERT_DIR)}
)
print("PORTABLE_REPORT " + json.dumps(report))
"""


@pytest.fixture(scope="module")
def child():
    env = dict(os.environ)
    env["NANOPLM_PORTABLE"] = "1"
    env["NANOPLM_CUTLASS_ROOT"] = str(MODERN_BERT_DIR / "no-cutlass-here")
    src = str(Path(nanoplm.__file__).resolve().parents[1])
    helpers = str(Path(__file__).resolve().parent)  # import_blocker
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [src, helpers, env.get("PYTHONPATH")])
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(MODERN_BERT_DIR),
            ",".join(KERNEL_MODULES),
            "1" if WITH_CUDA else "0",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def _report(child) -> dict:
    lines = [
        ln for ln in child.stdout.splitlines() if ln.startswith("PORTABLE_REPORT ")
    ]
    assert (
        child.returncode == 0 and lines
    ), f"portable-mode child failed (exit {child.returncode}):\n{child.stderr[-6000:]}"
    return json.loads(lines[-1].removeprefix("PORTABLE_REPORT "))


def test_portable_child_imports_no_kernel(child):
    report = _report(child)
    flags = report["flags"]
    assert flags["PORTABLE"] == "True"
    assert report["loaded"] == []
    assert report["modeling_tree_attempts"] == []
    assert report["nanoplm_ops"] == []
    assert flags["_HAS_FLASH_VARLEN"] == "False"
    assert flags["_FLASH_HAS_DROPOUT"] == "False"
    assert flags["USE_TRITON_SRELU"] == "False"
    assert flags["_CANON_BACKEND"] == "'triton'"
    for name in (
        "_flash_varlen_fn",
        "_mhc_triton_ops",
        "_canon_ops",
        "_canon_cute_ops",
        "_varlen_canon_conv",
        "_varlen_ln_canon_conv",
        "_varlen_rms_canon_conv",
    ):
        assert flags[name] == "None", name


@pytest.mark.parametrize("case", ["dense", "srelu", "moe", "mhc_lite", "canon"])
def test_portable_forward_matches_between_cpu_and_cuda(child, case):
    result = _report(child)["cases"][case]
    assert "error" not in result, result.get("error")
    assert result["cpu_finite"]
    if not WITH_CUDA:
        pytest.skip("CUDA half needs a GPU")
    assert result["cuda_finite"]
    assert result["mismatch"] is None, result["mismatch"]


def test_portable_router_breaks_ties_by_lowest_index_on_every_device(child):
    result = _report(child)["router_ties"]
    assert "error" not in result, result.get("error")
    assert result["cpu_matches_rule"]
    if not WITH_CUDA:
        pytest.skip("CUDA half needs a GPU")
    assert result["cuda_matches_rule"], "CUDA routing differs from CPU on ties"
    assert result["counts_equal"]


@pytest.mark.skipif(not WITH_CUDA, reason="needs CUDA")
def test_portable_moe_backward_on_cuda_matches_cpu(child):
    result = _report(child)["moe_backward"]
    assert "error" not in result, result.get("error")
    assert result["mismatch"] is None, result["mismatch"]


@pytest.mark.skipif(not WITH_CUDA, reason="needs CUDA")
def test_portable_grouped_gemm_never_builds_cutlass(child):
    result = _report(child)["grouped_gemm_bf16"]
    assert "error" not in result, result.get("error")
    assert result == {"equal": True, "extension_loaded": False}


@pytest.mark.skipif(not WITH_CUDA, reason="needs CUDA")
def test_portable_native_fp32_masked_forward_on_cuda(child):
    result = _report(child)["native_cuda"]
    assert "error" not in result, result.get("error")
    assert result["finite"]


@pytest.mark.skipif(not WITH_CUDA, reason="needs CUDA")
def test_portable_mhc_block_skips_triton_ops_on_cuda(child):
    result = _report(child)["mhc_block_bf16"]
    assert "error" not in result, result.get("error")
    assert result == {"finite": True, "shape": [8, 4, 16]}


def test_portable_is_off_without_env_var():
    if os.environ.get("NANOPLM_PORTABLE") is not None:
        pytest.skip("NANOPLM_PORTABLE is set in this environment")
    assert _globals.PORTABLE is False
    if torch.cuda.is_available():
        # The kernel branch still runs, and binds the same op objects.
        assert _globals._mhc_triton_ops is not None
        assert _globals._varlen_canon_conv is _globals._canon_ops.varlen_canon_conv
        assert (
            _globals._varlen_rms_canon_conv is _globals._canon_ops.varlen_rms_canon_conv
        )
        assert (
            _globals._varlen_ln_canon_conv is _globals._canon_ops.varlen_ln_canon_conv
        )


def test_no_kernel_module_import_is_visible_to_hf():
    """transformers copies, and scans for requirements, every sibling named by
    `from .x import y`; an absolute `nanoplm.` import makes it require nanoplm.
    Kernel modules must be reached only through `from . import x as _x`."""
    names = "|".join(KERNEL_MODULES)
    relative = re.compile(rf"^[ \t]*from\s+\.({names})\s+import", re.MULTILINE)
    absolute = re.compile(
        rf"^[ \t]*(from|import)\s+nanoplm\.pretraining\.models\.modern_bert\.({names})\b",
        re.MULTILINE,
    )
    hits = []
    for path in sorted(MODERN_BERT_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in (relative, absolute):
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.name}:{line}: {match.group(0).strip()}")
    assert hits == []
