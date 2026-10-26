"""moe_nan_guard: a NaN router input must not crash a training step.

Without the guard, an all-NaN row of router scores comes back from the fused
top-k as an out-of-bounds index, and the expert gather after it aborts the job.
With the guard on, the router clamps the index in training mode so the forward
finishes with a non-finite loss, which the pipeline's skip path then discards.
"""
from __future__ import annotations

import pytest
import torch

from nanoplm.pretraining.models.modern_bert.moe import Router

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available")),
]

_H, _E, _K = 16, 8, 2


def _routers(device):
    torch.manual_seed(0)
    plain = Router(_H, _E, _K).to(device)
    guarded = Router(_H, _E, _K, nan_guard=True).to(device)
    guarded.load_state_dict(plain.state_dict())
    return plain, guarded


@pytest.mark.parametrize("device", DEVICES)
def test_guard_is_inert_on_finite_input(device):
    plain, guarded = _routers(device)
    x = torch.randn(64, _H, device=device)
    for out_p, out_g in zip(plain(x), guarded(x)):
        if out_p is None:
            assert out_g is None
        else:
            assert torch.equal(out_p, out_g)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# NaN rows are CUDA-only: the fused Triton top-k returns an out-of-bounds
# sentinel for them, which is what the guard clamps. The CPU reference raises
# inside top-k instead, before the guard runs.
@CUDA
def test_guard_keeps_nan_rows_in_range_while_training():
    _, guarded = _routers("cuda")
    guarded.train()
    x = torch.randn(64, _H, device="cuda")
    x[5] = float("nan")
    x[40:48] = float("nan")
    _, indices, z_loss, _, _ = guarded(x)
    assert int(indices.min()) >= 0 and int(indices.max()) < _E
    # The NaN still reaches the loss, which is what the pipeline's skip path keys on.
    assert not torch.isfinite(z_loss)


_EVAL_SCRIPT = """
import torch
from nanoplm.pretraining.models.modern_bert.moe import Router
r = Router(16, 8, 2, nan_guard=True).cuda().eval()
x = torch.randn(64, 16, device="cuda")
x[5] = float("nan")
r(x)
torch.cuda.synchronize()
"""


@CUDA
def test_guard_does_not_clamp_in_eval():
    # Eval has no skip path, so a NaN must keep failing loudly rather than be
    # turned into a plausible-looking routing decision. The failure is a device
    # assert, which would poison this process's CUDA context, so it runs in a
    # child process.
    import os
    import subprocess
    import sys
    from pathlib import Path

    import nanoplm

    # Import the same nanoplm this process tests, not whatever is installed.
    src = str(Path(nanoplm.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (src, os.environ.get("PYTHONPATH")) if p
    ))
    proc = subprocess.run(
        [sys.executable, "-c", _EVAL_SCRIPT],
        capture_output=True, text=True, timeout=600, env=env,
    )
    assert proc.returncode != 0
    # The kernel's own "index out of bounds" printf does not always reach
    # stderr; the device-side assert it raises does.
    assert "device-side assert" in proc.stderr or "out of bounds" in proc.stderr


def test_config_reaches_every_router():
    from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
    from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
    from nanoplm.pretraining.models.modern_bert.moe import MoELayer

    def routers(guard):
        model = PureProtModernBertMLM(ProtModernBertMLMConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=2,
            vocab_size=32,
            use_moe=True,
            moe_num_experts=4,
            moe_top_k=2,
            moe_leading_dense_layers=1,
            moe_kernel_backend="cutlass",
            moe_nan_guard=guard,
        ))
        return [m.router for m in model.modules() if isinstance(m, MoELayer)]

    on, off = routers(True), routers(False)
    assert len(on) == 2
    assert all(r.nan_guard for r in on)
    assert not any(r.nan_guard for r in off)


@pytest.mark.parametrize("bias_correction", [True, False])
@pytest.mark.parametrize("guard", [True, False])
def test_pipeline_sees_the_guard(guard, bias_correction):
    # The pipeline arms its loss and gradient-norm skip checks off this flag.
    # It must not depend on moe_use_bias_correction: the routers clamp either
    # way, and a clamp with no skip behind it lets a NaN step reach the weights.
    from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
    from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
    from nanoplm.pretraining.pure_pipeline import _moe_nan_guard_active

    model = PureProtModernBertMLM(ProtModernBertMLMConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        vocab_size=32,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_leading_dense_layers=1,
        moe_kernel_backend="cutlass",
        moe_nan_guard=guard,
        moe_use_bias_correction=bias_correction,
    ))
    assert _moe_nan_guard_active(model) is guard


# ── end to end: a poisoned gradient is skipped, not stepped ─────────────────


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("bias_correction", [True, False])
def test_nonfinite_grad_step_is_skipped(tmp_path, monkeypatch, bias_correction):
    """One NaN gradient element mid-run: that step is skipped and logged, the
    run finishes its budget, and the final weights are finite."""
    if not torch.cuda.is_available():
        pytest.skip("run_pure_pretraining needs CUDA")
    import logging

    from nanoplm.pretraining.models.modern_bert import modeling
    from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
    from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
    from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
    from nanoplm.pretraining.pure_pipeline import run_pure_pretraining
    from nanoplm.cli.pretrain import _populate_batch_setup
    from nanoplm.pretraining.config import PretrainingConfig
    from tests.test_wsd_equivalence import (
        _GBT, _MICRO, _build_dataset, _find_checkpoint, _load_final_weights,
    )

    if not modeling._HAS_FLASH_VARLEN:
        pytest.skip("MoE runs need packing, which needs flash-attention varlen")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)

    tokenizer = ProtModernBertTokenizer()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    manifest = _build_dataset(dataset_dir, tokenizer)

    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=tokenizer.vocab_size,
        mlp_dropout=0.0,
        attention_dropout=0.0,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_leading_dense_layers=1,
        moe_kernel_backend="cutlass",
        moe_nan_guard=True,
        moe_use_bias_correction=bias_correction,
    ))

    # Poison one gradient element right before the second optimizer step's
    # clip. The loss stays finite, so only the gradient-norm check can catch it.
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def _poisoning_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        calls["n"] += 1
        if calls["n"] == 2:
            next(p for p in parameters if p.grad is not None).grad.view(-1)[0] = float("nan")
        return real_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _poisoning_clip)

    steps = 4
    cfg = PretrainingConfig(
        dataset_dir=str(dataset_dir),
        ckp_dir=str(tmp_path / "run"),
        micro_batch_seqs=_MICRO,
        global_batch_tokens=_GBT,
        warmup_steps=1,
        lr_schedule="warmup_stable",
        train_tokens=_GBT * steps,
        bf16=True,
        fp8=False,
        use_packing=True,
        use_static_inp_size=True,
        multi_gpu=False,
        world_size=1,
        num_workers=0,
        eval_steps=steps,
        save_steps=10_000_000,
        logging_steps=10_000_000,
        seed=0,
        optimizer="adamw",
    )
    _populate_batch_setup(cfg)
    # The nanoplm logger does not propagate, so caplog cannot see it.
    records: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.ERROR)
    handler.emit = records.append
    nanoplm_logger = logging.getLogger("nanoplm")
    nanoplm_logger.addHandler(handler)
    try:
        run_pure_pretraining(model, cfg, manifest)
    finally:
        nanoplm_logger.removeHandler(handler)

    skips = [r for r in records if "non-finite gradient norm" in r.getMessage()]
    assert len(skips) == 1, [r.getMessage() for r in records]
    final = _load_final_weights(_find_checkpoint(tmp_path / "run", "stable-end"))
    assert all(torch.isfinite(v).all() for v in final.values() if v.is_floating_point())
