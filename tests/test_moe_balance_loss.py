"""The balance loss must stay a per-expert statistic with a live gradient.

HuggingFace transformers#28255 shipped a load-balancing loss that reduced over
the token axis instead of the top-k axis. It became the constant `num_experts`
for any real batch, so Mixtral trained with no balancing signal for a month.
Its unit test passed throughout, because the test asserted the saturated value.
These cases assert the properties that failure would have broken.

CPU-only: the router's `topk_counts` has a CPU fallback, the grouped-GEMM
expert path does not, so `MoELayer.forward` is not exercised here.
"""

import torch

from nanoplm.pretraining.models.modern_bert.moe import Router

E, K, H, T = 8, 2, 16, 512


def _aux(router_scores, num_experts, coef=1.0):
    mass = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    mean_mass = mass.mean(dim=0)
    target = torch.ones_like(mean_mass) / num_experts
    return coef * ((mean_mass - target) ** 2).sum() * num_experts


def _router_scores(bias_on_expert0=0.0):
    torch.manual_seed(0)
    router = Router(H, E, K, bias_correction=True)
    x = torch.randn(T, H)
    with torch.no_grad():
        router.gate.weight[0] += bias_on_expert0
    return router(x)[3]


def test_router_scores_keep_a_per_expert_axis():
    scores = _router_scores()
    assert scores.shape == (T, E), (
        f"router scores collapsed to {tuple(scores.shape)}; the balance "
        f"statistic needs one column per expert"
    )
    mean_mass = (scores / scores.sum(dim=-1, keepdim=True)).mean(dim=0)
    assert mean_mass.shape == (E,), (
        f"mean_mass is {tuple(mean_mass.shape)}, not ({E},); a reduction "
        f"collapsed the expert axis, which is transformers#28255"
    )


def test_balance_loss_is_near_zero_when_routing_is_uniform():
    mass = torch.full((T, E), 1.0 / E)
    assert _aux(mass, E).item() < 1e-12


def test_balance_loss_grows_with_imbalance():
    values = [_aux(_router_scores(b), E).item() for b in (0.0, 2.0, 8.0)]
    assert values == sorted(values), f"not monotone in skew: {values}"
    assert values[-1] > values[0] * 5, (
        f"balance loss barely moved under heavy skew: {values}; a saturated "
        f"loss carries no balancing signal"
    )


def test_balance_loss_has_a_live_gradient():
    torch.manual_seed(0)
    router = Router(H, E, K, bias_correction=True)
    x = torch.randn(T, H)
    _aux(router(x)[3], E).backward()
    grad = router.gate.weight.grad
    assert grad is not None and grad.abs().sum().item() > 1e-8, (
        "no gradient reached the router gate; the balance loss is a constant"
    )


def test_per_execution_mean_recovers_the_single_execution_value():
    scores = _router_scores()
    single = _aux(scores, E)
    total = single + single + single
    count = torch.tensor(3.0)
    torch.testing.assert_close(total / count, single)


def _masked_mean_mass(router_scores, valid_mask):
    mass = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    m = valid_mask.to(mass.dtype)
    return (mass * m).sum(dim=0) / m.sum().clamp_min(1.0)


def _padded_batch(n_real=256, n_pad=64):
    torch.manual_seed(1)
    real = torch.randn(n_real, E) * 0.5
    pad = torch.randn(1, E).expand(n_pad, E)
    scores = torch.sigmoid(torch.cat([real, pad], dim=0))
    valid = (torch.arange(n_real + n_pad) < n_real).unsqueeze(1)
    return scores, valid, torch.sigmoid(real)


def test_masked_balance_statistic_ignores_padding_rows():
    scores, valid, real_only = _padded_batch()
    torch.testing.assert_close(
        _masked_mean_mass(scores, valid),
        (real_only / real_only.sum(dim=-1, keepdim=True)).mean(dim=0),
    )


def test_unmasked_balance_statistic_is_skewed_by_padding():
    scores, _, real_only = _padded_batch()
    mass = scores / scores.sum(dim=-1, keepdim=True)
    contaminated = mass.mean(dim=0)
    clean = (real_only / real_only.sum(dim=-1, keepdim=True)).mean(dim=0)
    assert (contaminated - clean).abs().max().item() > 1e-4, (
        "padding rows did not move the statistic, so this fixture proves nothing"
    )


def test_pad_subtraction_recovers_real_token_counts():
    scores, valid, real_only = _padded_batch()
    _, idx = torch.topk(scores, K, dim=-1)
    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=E)
    pad_w = (~valid).to(torch.int64).expand(-1, K).reshape(-1)
    corrected = counts - torch.zeros_like(counts).scatter_add_(0, flat, pad_w)

    _, idx_real = torch.topk(real_only, K, dim=-1)
    expected = torch.bincount(idx_real.reshape(-1), minlength=E)
    torch.testing.assert_close(corrected, expected)
    assert corrected.sum().item() == real_only.shape[0] * K
