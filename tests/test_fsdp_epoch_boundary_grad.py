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
from test_resume_terminal_data_position import _ckpt, _dataset, _run  # noqa: E402

pytestmark = pytest.mark.slow


def _weights(tmp: Path, ds: Path, mode: str, steps: int) -> dict:
    name = f"{mode}{steps}"
    _run(tmp, name, dict(dataset_dir=str(ds), micro=2, ga=2, bwarm=0, mode="dist",
                         dist_mode=mode, ckp_dir=str(tmp / name), steps=steps), 2)
    return torch.load(_ckpt(tmp / name) / "pytorch_model.bin")


@pytest.mark.parametrize("steps", [2, 3])
def test_fsdp_first_step_after_pass_boundary_matches_ddp(tmp_path, steps):
    # steps=2 ends on the last step of pass 1 (control); steps=3 is the first
    # step of pass 2, after the trailing micro-batch.
    ds = _dataset(tmp_path, 10 * 2)
    ddp = _weights(tmp_path, ds, "ddp", steps)
    fsdp = _weights(tmp_path, ds, "fsdp", steps)
    assert ddp.keys() == fsdp.keys()
    diff = {k: (ddp[k].float() - fsdp[k].float()).abs().max().item() for k in ddp}
    worst = max(diff, key=diff.get)
    assert diff[worst] == 0.0, f"{worst}: max |ddp - fsdp| = {diff[worst]:.3e}"
