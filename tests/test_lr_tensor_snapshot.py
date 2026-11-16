"""dion >= 7692479 keeps param_group["lr"] as a persistent 0-d tensor that LambdaLR
fills in place. The training loop must log the current step's LR, not the next one."""
import torch
from torch.optim.lr_scheduler import LambdaLR

import nanoplm.pretraining.pure_pipeline as pp
from nanoplm.pretraining.optim import MuonAdamWGroup
from nanoplm.pretraining.pure_pipeline import _create_scheduler, _group_lr


def _sgd_with_tensor_lr(lr=0.02):
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p], lr=lr)
    opt.param_groups[0]["lr"] = torch.tensor(lr)  # what dion's _sync_hyperparam_tensors does
    return opt


def test_bare_reference_aliases_the_next_step_lr():
    # The bug pattern: a reference read before scheduler.step() changes under it.
    opt = _sgd_with_tensor_lr()
    sched = LambdaLR(opt, lambda s: (s + 1) / 10)
    ref = opt.param_groups[0]["lr"]
    before = float(ref)
    sched.step()
    assert float(ref) != before


def test_group_lr_snapshot_keeps_the_current_step_lr():
    opt = _sgd_with_tensor_lr()
    sched = LambdaLR(opt, lambda s: (s + 1) / 10)
    pg = opt.param_groups[0]
    for _ in range(3):
        expected = float(pg["lr"])
        snap = _group_lr(pg)
        sched.step()
        assert float(snap) == expected
        assert float(pg["lr"]) != expected  # the live tensor did move
    assert isinstance(pg["lr"], torch.Tensor)  # scheduler kept filling the same tensor


def test_group_lr_float_passthrough():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.5)
    assert _group_lr(opt.param_groups[0]) == 0.5
    assert isinstance(_group_lr(opt.param_groups[0]), float)


def test_resume_scheduler_stores_float_initial_lr():
    # last_epoch >= 0 path: initial_lr must not alias the live lr tensor.
    opt = _sgd_with_tensor_lr(lr=1.0)
    sched = _create_scheduler(opt, 0, 100, 1.0, 0.1, "Linear", last_epoch=4)
    pg = opt.param_groups[0]
    assert isinstance(pg["initial_lr"], float) and pg["initial_lr"] == 1.0
    for _ in range(5):
        sched.step()
    assert pg["initial_lr"] == 1.0


# ---- _snapshot_lrs: the loop's step -> snapshot -> scheduler.step() order ----

def _run_steps(opt, sched, n):
    """Mimic the training loop; return (snapshot, lr the step actually applied) pairs."""
    out = []
    for _ in range(n):
        muon_p = _muon_param(opt)
        before = muon_p.detach().clone()
        muon_p.grad = torch.ones_like(muon_p)
        for p in _other_params(opt):
            p.grad = torch.ones_like(p)
        opt.step()
        applied = float((before - muon_p.detach())[0])  # plain SGD, grad 1: delta == lr
        snap = pp._snapshot_lrs(opt)
        sched.step()
        opt.zero_grad(set_to_none=True)
        out.append((snap, applied))
    return out


def _muon_param(opt):
    groups = opt.muon.param_groups if isinstance(opt, MuonAdamWGroup) else opt.param_groups
    return groups[0]["params"][0]


def _other_params(opt):
    if isinstance(opt, MuonAdamWGroup):
        return [p for g in opt.adamw.param_groups for p in g["params"]]
    return [p for g in opt.param_groups[1:] for p in g["params"]]


def _close(a, b):
    # float32 lr tensors and float32 param updates; adjacent steps differ by >5%.
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b)) + 1e-12


def _lambda(step, warmup=3, total=10, floor=0.1):
    # Linear schedule in _create_scheduler: warmup step/warmup, then linear decay.
    if step < warmup:
        return step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max(floor, 1.0 - (1.0 - floor) * progress)


def test_snapshot_lrs_muon_adamw_group_reads_the_step_lr():
    muon = _sgd_with_tensor_lr(lr=0.02)  # dion side: tensor lr filled in place
    adamw = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)  # float lr
    opt = MuonAdamWGroup(muon, adamw)
    sched = _create_scheduler(opt, 3, 10, 0.02, 0.1, "Linear")
    muon_lr_obj = muon.param_groups[0]["lr"]

    for step, ((lr, muon_lr), applied) in enumerate(_run_steps(opt, sched, 8)):
        # tensor snapshot (no host sync), detached from the live tensor
        assert isinstance(muon_lr, torch.Tensor) and muon_lr is not muon_lr_obj
        assert isinstance(lr, float)
        assert _close(float(muon_lr), applied)
        assert _close(float(muon_lr), 0.02 * _lambda(step))
        assert _close(lr, 1e-3 * _lambda(step))
    assert muon.param_groups[0]["lr"] is muon_lr_obj  # the clone never rebinds group["lr"]


def test_snapshot_lrs_dion_only_branch(monkeypatch):
    class FakeDion(torch.optim.SGD):
        pass

    monkeypatch.setattr(pp, "_DION_OPTIMIZER_TYPES", (FakeDion,))
    opt = FakeDion(
        [
            {"params": [torch.nn.Parameter(torch.zeros(2))], "lr": 0.02},  # muon
            {"params": [torch.nn.Parameter(torch.zeros(2))], "lr": 1e-3},  # adamw
        ],
        lr=0.02,
    )
    for pg in opt.param_groups:
        pg["lr"] = torch.tensor(pg["lr"])
    sched = _create_scheduler(opt, 3, 10, 0.02, 0.1, "Linear")

    for step, ((lr, muon_lr), applied) in enumerate(_run_steps(opt, sched, 6)):
        assert isinstance(lr, torch.Tensor) and isinstance(muon_lr, torch.Tensor)
        assert _close(float(muon_lr), applied)
        assert _close(float(muon_lr), 0.02 * _lambda(step))
        assert _close(float(lr), 1e-3 * _lambda(step))


def test_snapshot_lrs_plain_torch_branch():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.5)
    sched = _create_scheduler(opt, 3, 10, 0.5, 0.1, "Linear")

    for step, ((lr, muon_lr), applied) in enumerate(_run_steps(opt, sched, 6)):
        assert muon_lr is None
        assert isinstance(lr, float)
        assert _close(lr, applied)
        assert _close(lr, 0.5 * _lambda(step))
