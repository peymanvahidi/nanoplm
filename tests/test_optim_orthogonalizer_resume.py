"""The Muon orthogonalizer is named, logged, checkpointed, and checked on resume."""

import json
import logging

import pytest
import torch
import yaml

from nanoplm.pretraining import optim
from nanoplm.pretraining.optim import (
    CUSTOM_PE,
    DION_PE,
    GNS,
    QUINTIC,
    legacy_muon_orthogonalizer,
    resolve_muon_orthogonalizer,
)
from nanoplm.pretraining.pure_pipeline import (
    _checkpoint_muon_orthogonalizer,
    _save_checkpoint,
    _warn_on_orthogonalizer_change,
)


@pytest.fixture
def nanoplm_records():
    # The nanoplm logger does not propagate, so caplog cannot see it.
    records: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.INFO)
    handler.emit = records.append
    lg = logging.getLogger("nanoplm")
    lg.addHandler(handler)
    yield records
    lg.removeHandler(handler)


def _warnings(records):
    return [r for r in records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("pe,gns,has_kwarg,want", [
    (True, True, True, GNS),
    (True, False, True, CUSTOM_PE),
    (False, True, True, GNS),
    (False, False, True, QUINTIC),
    (True, True, False, DION_PE),
    (True, False, False, CUSTOM_PE),
    (False, True, False, QUINTIC),
    (False, False, False, QUINTIC),
])
def test_resolve(pe, gns, has_kwarg, want):
    assert resolve_muon_orthogonalizer(pe, gns, has_kwarg) == want


@pytest.mark.parametrize("pe,gns,want", [
    (True, True, CUSTOM_PE),  # our PE won over GNS before #171
    (True, False, CUSTOM_PE),
    (False, True, GNS),
    (False, False, QUINTIC),
])
def test_legacy(pe, gns, want):
    assert legacy_muon_orthogonalizer(pe, gns) == want


@pytest.mark.parametrize("pe,gns", [(True, True), (True, False), (False, True), (False, False)])
def test_build_optimizer_label_matches_dion_choice(pe, gns, nanoplm_records):
    pytest.importorskip("dion")
    if gns:
        pytest.importorskip("gram_newton_schulz")
    w = torch.nn.Parameter(torch.randn(16, 32))
    b = torch.nn.Parameter(torch.zeros(4))
    opt = optim.build_optimizer(
        muon_params=[w], adamw_params=[b], muon_learning_rate=0.02, muon_weight_decay=0.0,
        muon_cautious_weight_decay=False, muon_use_polar_express=pe, muon_momentum=0.95,
        muon_nesterov=False, muon_eps=1e-8, adamw_learning_rate=1e-3, adamw_weight_decay=0.0,
        adamw_betas=(0.9, 0.98), adamw_epsilon=1e-8, muon_variant="normuon",
        muon_use_gram_newton_schulz=gns,
    )
    fn = opt.muon._newton_schulz_func
    want = {
        GNS: lambda: fn is not optim._polar_express_paper and fn.__name__ == "<lambda>",
        CUSTOM_PE: lambda: fn is optim._polar_express_paper,
        QUINTIC: lambda: fn.__name__ in {"newton_schulz_triton", "zeropower_via_newtonschulz5"},
    }
    assert want[opt.muon_orthogonalizer](), (opt.muon_orthogonalizer, fn)
    msgs = [r.getMessage() for r in nanoplm_records]
    assert (
        f"Muon orthogonalizer: {opt.muon_orthogonalizer} (muon_use_polar_express={pe}, "
        f"muon_use_gram_newton_schulz={gns})"
    ) in msgs


def _ckpt(path, pc=None, ts=None):
    if pc is not None:
        (path / "pretrain_config.yaml").write_text(yaml.safe_dump(pc))
    if ts is not None:
        (path / "training_state.json").write_text(json.dumps(ts))
    return path


def test_pe_checkpoint_resumed_under_gns_warns(tmp_path, nanoplm_records):
    d = _ckpt(tmp_path, {"optimizer": "normuon", "muon_use_polar_express": True,
                         "muon_use_gram_newton_schulz": False}, {"global_step": 5})
    assert _warn_on_orthogonalizer_change(d, GNS)
    (rec,) = _warnings(nanoplm_records)
    msg = rec.getMessage()
    assert str(d) in msg and CUSTOM_PE in msg and GNS in msg
    assert "now defaults to true and takes precedence" in msg
    assert "To keep the checkpoint's orthogonalizer, set under pretraining:\n" in msg
    assert "muon_use_polar_express: true\n  muon_use_gram_newton_schulz: false" in msg


_PE_CFG = {"optimizer": "normuon", "muon_use_polar_express": True,
           "muon_use_gram_newton_schulz": False}
_GNS_ONLY_CFG = {"optimizer": "muon", "muon_use_polar_express": False,
                 "muon_use_gram_newton_schulz": True}


@pytest.mark.parametrize("pc,ts,current,pin", [
    # GNS checkpoint resumed with PE pinned back: the pin restores GNS.
    (None, {"muon_orthogonalizer": GNS}, CUSTOM_PE, "muon_use_gram_newton_schulz: true"),
    # PE checkpoint on a dion without the GNS kwarg (dion's own PE runs).
    (_PE_CFG, None, DION_PE,
     "muon_use_polar_express: true\n  muon_use_gram_newton_schulz: false"),
    # GNS checkpoint (PE off) resumed with both flags off.
    (_GNS_ONLY_CFG, None, QUINTIC, "muon_use_gram_newton_schulz: true"),
    # Quintic checkpoint resumed with PE.
    ({"optimizer": "normuon", "muon_use_polar_express": False,
      "muon_use_gram_newton_schulz": False}, None, CUSTOM_PE,
     "muon_use_polar_express: false\n  muon_use_gram_newton_schulz: false"),
])
def test_change_to_non_gns_does_not_blame_gns_default(tmp_path, nanoplm_records, pc, ts,
                                                      current, pin):
    d = _ckpt(tmp_path, pc, ts)
    (previous,) = _checkpoint_muon_orthogonalizer(d)
    assert _warn_on_orthogonalizer_change(d, current)
    (rec,) = _warnings(nanoplm_records)
    msg = rec.getMessage()
    assert msg.startswith(
        f"Muon orthogonalizer changed on resume: checkpoint {d} was trained with "
        f"{previous}, this run uses {current}. The effective update changes from this step on."
    )
    assert "defaults to true" not in msg and "precedence" not in msg
    assert "optimizer," not in msg and "old optimizer" not in msg
    assert msg.endswith(f"To keep the checkpoint's orthogonalizer, set under pretraining:\n  {pin}")


def test_quintic_to_gns_names_default_not_precedence(tmp_path, nanoplm_records):
    d = _ckpt(tmp_path, {"optimizer": "normuon", "muon_use_polar_express": False,
                         "muon_use_gram_newton_schulz": False})
    assert _warn_on_orthogonalizer_change(d, GNS)
    (rec,) = _warnings(nanoplm_records)
    msg = rec.getMessage()
    assert "muon_use_gram_newton_schulz now defaults to true, so the effective update" in msg
    assert "precedence" not in msg
    assert msg.endswith("muon_use_polar_express: false\n  muon_use_gram_newton_schulz: false")


def test_recorded_dion_pe_has_no_pin(tmp_path, nanoplm_records):
    # No config selects dion's own PE on a dion with the GNS kwarg: no pin line.
    d = _ckpt(tmp_path, None, {"muon_orthogonalizer": DION_PE})
    assert _warn_on_orthogonalizer_change(d, GNS)
    (rec,) = _warnings(nanoplm_records)
    assert "set under pretraining" not in rec.getMessage()


@pytest.mark.parametrize("current", [GNS, CUSTOM_PE, QUINTIC])
def test_unlabelled_pe_and_gns_is_ambiguous(tmp_path, nanoplm_records, current):
    # pe: true + gns: true ran our PE before #171 (gns was a no-op) but GNS on
    # builds that already had the GNS default. Without a label we cannot tell.
    d = _ckpt(tmp_path, {"optimizer": "normuon", "muon_use_polar_express": True,
                         "muon_use_gram_newton_schulz": True}, {"global_step": 5})
    assert _checkpoint_muon_orthogonalizer(d) == (CUSTOM_PE, GNS)
    assert _warn_on_orthogonalizer_change(d, current)
    (rec,) = _warnings(nanoplm_records)
    msg = rec.getMessage()
    assert msg.startswith(f"Muon orthogonalizer of checkpoint {d} is not recorded.")
    assert "changed on resume" not in msg
    assert f"ran {CUSTOM_PE} if it was trained before PR #171 and {GNS} if" in msg
    assert f"This run uses {current}." in msg
    assert (f"To pin {CUSTOM_PE}, set under pretraining:\n"
            "  muon_use_polar_express: true\n  muon_use_gram_newton_schulz: false\n") in msg
    assert msg.endswith(f"To pin {GNS}, set under pretraining:\n  muon_use_gram_newton_schulz: true")


@pytest.mark.parametrize("pe,gns,want", [
    (True, False, CUSTOM_PE), (False, True, GNS), (False, False, QUINTIC),
])
def test_unlabelled_unambiguous_flags(tmp_path, pe, gns, want):
    d = _ckpt(tmp_path, {"optimizer": "nordion2", "muon_use_polar_express": pe,
                         "muon_use_gram_newton_schulz": gns})
    assert _checkpoint_muon_orthogonalizer(d) == (want,)


def test_recorded_label_wins_over_inference(tmp_path, nanoplm_records):
    d = _ckpt(tmp_path, {"optimizer": "normuon", "muon_use_polar_express": True,
                         "muon_use_gram_newton_schulz": True},
              {"global_step": 5, "muon_orthogonalizer": GNS})
    assert _checkpoint_muon_orthogonalizer(d) == (GNS,)
    assert not _warn_on_orthogonalizer_change(d, GNS)
    assert _warnings(nanoplm_records) == []


def test_same_orthogonalizer_is_silent(tmp_path, nanoplm_records):
    d = _ckpt(tmp_path, {"optimizer": "normuon", "muon_use_polar_express": True,
                         "muon_use_gram_newton_schulz": False})
    assert not _warn_on_orthogonalizer_change(d, CUSTOM_PE)
    assert _warnings(nanoplm_records) == []


def test_missing_files_adamw_and_no_label_are_silent(tmp_path, nanoplm_records):
    assert _checkpoint_muon_orthogonalizer(tmp_path) == ()
    assert not _warn_on_orthogonalizer_change(tmp_path, GNS)
    (tmp_path / "training_state.json").write_text("{trunc")
    assert not _warn_on_orthogonalizer_change(tmp_path, GNS)
    (tmp_path / "training_state.json").write_text("[1, 2]")  # not a dict
    assert _checkpoint_muon_orthogonalizer(tmp_path) == ()
    d = _ckpt(tmp_path, {"optimizer": "adamw"}, {"global_step": 5})
    assert _checkpoint_muon_orthogonalizer(d) == ()
    assert not _warn_on_orthogonalizer_change(d, GNS)
    d = _ckpt(tmp_path, {"optimizer": "normuon"})
    assert not _warn_on_orthogonalizer_change(d, None)  # current run is not Muon
    assert _warnings(nanoplm_records) == []


class _StubOptimizer:
    def __init__(self, label=None):
        if label is not None:
            self.muon_orthogonalizer = label

    def state_dict(self):
        return {}


class _StubScheduler:
    def state_dict(self):
        return {}


@pytest.mark.parametrize("label", [GNS, None])
def test_training_state_records_orthogonalizer(tmp_path, label):
    _save_checkpoint(
        torch.nn.Linear(2, 2), _StubOptimizer(label), _StubScheduler(), 7, 0,
        tmp_path, 1, 1, 1,
    )
    (ckpt,) = tmp_path.iterdir()
    state = json.loads((ckpt / "training_state.json").read_text())
    if label is None:
        assert "muon_orthogonalizer" not in state
    else:
        assert state["muon_orthogonalizer"] == label
        assert _checkpoint_muon_orthogonalizer(ckpt) == (label,)
