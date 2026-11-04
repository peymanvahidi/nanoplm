"""Expert-bias update rule: proportional (default) vs sign (DeepSeek-V3 / Ling-V2).

CPU-only. Checks the per-layer step that pure_pipeline applies to
`router.correction_bias` after each optimizer step.
"""

import pytest
import torch

from nanoplm.pretraining.models.modern_bert.config import ModernBertConfig
from nanoplm.pretraining.pure_pipeline import _moe_bias_delta

RATE = 1e-3


def _counts():
    return torch.tensor([40.0, 10.0, 25.0, 25.0, 0.0, 100.0])


def test_proportional_matches_the_previous_inline_rule():
    c = _counts()
    target = c.sum() / c.numel()
    old = RATE * (target - c) / target
    old = old - old.mean()
    torch.testing.assert_close(_moe_bias_delta(c, RATE, "proportional"), old)


def test_sign_steps_by_a_fixed_rate_and_stays_centered():
    c = _counts()
    d = _moe_bias_delta(c, RATE, "sign")
    sign = torch.sign(c.sum() / c.numel() - c)
    torch.testing.assert_close(d, RATE * (sign - sign.mean()))
    assert abs(d.sum().item()) < 1e-9
    # underused experts move up, overused down, balanced ones only by the recentering
    assert d[4] > 0 and d[5] < 0 and d[1] > d[0]
    # the step does not depend on how far an expert is off target
    assert d[1] == d[4] and d[0] == d[5]


@pytest.mark.parametrize("mode", ["proportional", "sign"])
def test_no_tokens_means_no_step(mode):
    assert torch.count_nonzero(_moe_bias_delta(torch.zeros(6), RATE, mode)) == 0


@pytest.mark.parametrize("mode", ["proportional", "sign"])
def test_balanced_load_means_no_step(mode):
    assert torch.count_nonzero(_moe_bias_delta(torch.full((6,), 7.0), RATE, mode)) == 0


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="moe_bias_update_mode"):
        ModernBertConfig(use_moe=True, moe_num_experts=8, moe_top_k=2, moe_bias_update_mode="signum")
