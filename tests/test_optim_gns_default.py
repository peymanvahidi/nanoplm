"""Gram Newton-Schulz is the default Muon-family orthogonalizer."""

import pytest
import torch

from nanoplm.pretraining import optim
from nanoplm.pretraining.config import PretrainingConfig


def test_config_defaults_to_gns():
    assert PretrainingConfig.__dataclass_fields__["muon_use_gram_newton_schulz"].default is True


def _build(**overrides):
    w = torch.nn.Parameter(torch.randn(3, 256, 512, device="cuda") * 1e-2)
    b = torch.nn.Parameter(torch.zeros(8, device="cuda"))
    kwargs = dict(
        muon_params=[w], adamw_params=[b], muon_learning_rate=0.02, muon_weight_decay=0.0,
        muon_cautious_weight_decay=False, muon_use_polar_express=True, muon_momentum=0.95,
        muon_nesterov=False, muon_eps=1e-8, adamw_learning_rate=1e-3, adamw_weight_decay=0.0,
        adamw_betas=(0.9, 0.98), adamw_epsilon=1e-8, muon_variant="normuon",
    )
    kwargs.update(overrides)
    return w, b, optim.build_optimizer(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gns_overrides_polar_express_and_steps():
    pytest.importorskip("gram_newton_schulz")
    w, b, opt = _build()
    assert opt.muon._newton_schulz_func is not optim._polar_express_paper
    w0 = w.detach().clone()
    w.grad = torch.randn_like(w) * 1e-3
    b.grad = torch.randn_like(b)
    opt.step()
    assert torch.isfinite(w).all() and not torch.equal(w, w0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_polar_express_still_selectable_with_gns_off():
    _, _, opt = _build(muon_use_gram_newton_schulz=False)
    assert opt.muon._newton_schulz_func is optim._polar_express_paper
