import math
import pytest
import torch
from nanoplm.pretraining.pure_pipeline import _create_scheduler


def _factors(lr_schedule, warmup, total, floor, **kw):
    """LR multipliers (base_lr=1.0) for steps 0..total-1."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = _create_scheduler(opt, warmup, total, 1.0, floor, lr_schedule, **kw)
    out = []
    for _ in range(total):
        out.append(opt.param_groups[0]["lr"])
        sched.step()
    return out


class TestWSDScheduler:
    def test_warmup_stable_ramps_then_holds(self):
        f = _factors("warmup_stable", warmup=4, total=10, floor=0.1)
        assert f[0] == pytest.approx(0.0)
        assert f[2] == pytest.approx(0.5)        # 2/4
        assert f[4] == pytest.approx(1.0)        # reached peak
        assert all(x == pytest.approx(1.0) for x in f[4:])  # constant after

    def test_wsd_plateau_then_decays_to_floor(self):
        f = _factors("wsd", warmup=2, total=10, floor=0.1,
                     stable_steps=6, decay_steps=4, decay_shape="linear")
        assert f[2] == pytest.approx(1.0)        # plateau start (post-warmup)
        assert f[5] == pytest.approx(1.0)        # still plateau (step < S=6)
        assert f[6] == pytest.approx(1.0)        # decay start = peak
        assert f[9] == pytest.approx(0.1 + 0.9 * 0.25)

    def test_wsd_1sqrt_drops_faster_than_linear_early(self):
        lin = _factors("wsd", warmup=0, total=8, floor=0.0,
                       stable_steps=4, decay_steps=4, decay_shape="linear")
        sq = _factors("wsd", warmup=0, total=8, floor=0.0,
                      stable_steps=4, decay_steps=4, decay_shape="1-sqrt")
        assert sq[5] < lin[5]

    def test_decay_only_schedule_for_resume(self):
        f = _factors("decay", warmup=0, total=5, floor=0.2, decay_shape="linear")
        assert f[0] == pytest.approx(1.0)
        assert f[4] == pytest.approx(0.2 + 0.8 * (1 - 4 / 5))

    def test_cosine_unchanged(self):
        f = _factors("cosine", warmup=2, total=12, floor=0.1)
        assert f[2] == pytest.approx(1.0)        # peak right after warmup
        assert f[11] < 0.2                       # near floor at the end
