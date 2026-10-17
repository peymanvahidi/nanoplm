"""FSDP2 must not carry a discarded trailing grad-accum window into the next pass.

A data pass of 5 micro-batches with grad_accum 2 takes 2 optimizer steps and
leaves 1 micro-batch that never reaches a step. Under FSDP2 with gradient sync
off, that micro-batch's gradient lives on the unsharded parameters, out of
reach of optimizer.zero_grad(), and used to be reduced into the first step of
the next pass. DDP accumulates into param.grad, which zero_grad() clears, so
it is the fresh-window reference: FSDP2 and DDP must end with the same weights.

Runs resume_harness.py under torchrun, 2 ranks, gloo on CPU.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_resume_terminal_data_position import _ckpt, _dataset, _pass_line, _run  # noqa: E402


def _weights(tmp: Path, ds: Path, mode: str, steps: int, ga: int, bwarm: int, over: int):
    name = f"{mode}{steps}"
    out = _run(tmp, name, dict(dataset_dir=str(ds), micro=2, ga=ga, bwarm=bwarm, mode="dist",
                               dist_mode=mode, ckp_dir=str(tmp / name), steps=steps,
                               len_overreport=over), 2)
    return torch.load(_ckpt(tmp / name) / "pytorch_model.bin"), out[0]["pass_lines"]


# n samples per rank, 2 per micro-batch. pass_end ends on the last step of
# pass 1 (control); after_tail is the first step of pass 2, after the trailing
# micro-batch. warmup_tail: 11 micro-batches, grad_accum 1/2/4 over steps 0-5
# (bwarm=6), so steps end at micro 0, 1, 3, 5, 9 and micro 10 can't fill a
# 4-window. early_exhaustion: the loader claims 6 micro-batches but yields 5,
# so micro 4 starts a window that micro 5 never completes.
@pytest.mark.slow
@pytest.mark.parametrize("n,ga,bwarm,steps,over,lines", [
    pytest.param(10, 2, 0, 2, 0, [], id="pass_end"),
    pytest.param(10, 2, 0, 3, 0, [_pass_line(1, 2, 3, 4, 2, 1)], id="after_tail"),
    pytest.param(22, 8, 6, 6, 0, [_pass_line(1, 5, 6, 10, 5, 1)], id="warmup_tail"),
    pytest.param(10, 2, 0, 3, 1, [_pass_line(1, 2, 3, 5, 2, 1)], id="early_exhaustion"),
])
def test_fsdp_first_step_after_pass_boundary_matches_ddp(tmp_path, n, ga, bwarm, steps, over,
                                                         lines):
    ds = _dataset(tmp_path, n * 2)
    ddp, ddp_lines = _weights(tmp_path, ds, "ddp", steps, ga, bwarm, over)
    fsdp, fsdp_lines = _weights(tmp_path, ds, "fsdp", steps, ga, bwarm, over)
    assert ddp_lines == fsdp_lines == lines
    assert ddp.keys() == fsdp.keys()
    diff = {k: (ddp[k].float() - fsdp[k].float()).abs().max().item() for k in ddp}
    worst = max(diff, key=diff.get)
    # Exact (0.0) on 2-rank gloo today; 1e-6 is ~8 fp32 ulps at 1.0 and far
    # below the ~1e-4 leak.
    assert diff[worst] <= 1e-6, f"{worst}: max |ddp - fsdp| = {diff[worst]:.3e}"


@pytest.fixture
def one_rank_mesh():
    """A one-rank gloo group, so ``fully_shard`` runs on CPU."""
    import socket

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    owns_group = not dist.is_initialized()
    if owns_group:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
    yield init_device_mesh("cpu", (1,))
    if owns_group:
        dist.destroy_process_group()


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16_params_fp32_reduce"])
def test_drop_fsdp_unsynced_grads_clears_private_state(one_rank_mesh, bf16):
    # bf16 params with an fp32 reduce keep the grad in unsharded_accumulated_grad
    # instead of on the unsharded param.
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from nanoplm.pretraining.pure_pipeline import _drop_fsdp_unsynced_grads

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if bf16 else None
    fully_shard(model, mesh=one_rank_mesh, **({"mp_policy": mp} if mp else {}))
    model.set_requires_gradient_sync(False)
    model(torch.randn(2, 4)).sum().backward()

    params = [p for g in model._get_fsdp_state()._fsdp_param_groups for p in g.fsdp_params]

    held = [p.unsharded_accumulated_grad if bf16 else p._unsharded_param.grad for p in params]
    assert any(g is not None for g in held)
    _drop_fsdp_unsynced_grads(model)
    # Booleans only: pytest's repr of a resharded (storage-freed) param segfaults.
    left = [p.unsharded_accumulated_grad is not None or p._unsharded_param.grad is not None
            for p in params]
    assert not any(left), left
