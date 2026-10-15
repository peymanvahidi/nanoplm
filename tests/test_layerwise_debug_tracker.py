import torch

from nanoplm.pretraining.debug_metrics import LayerwiseDebugTracker
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM


def _cfg(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        attn_layer_pattern="F",
        wo_init="kaiming",
        no_mlp_on_first_layer=False,
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


def _perturb(model):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)


def _tracker(model, tmp_path, max_rep):
    return LayerwiseDebugTracker(
        model=model,
        log_every=1,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        is_main=True,
        distributed=False,
        max_rep=max_rep,
    )


def _run(model, n, **kwargs):
    input_ids = torch.randint(0, 32, (2, 8))
    for _ in range(n):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            **kwargs,
        )


def test_plain_model_accumulates_into_rep0(tmp_path):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg())
    model.train()
    tracker = _tracker(model, tmp_path, max_rep=1)
    try:
        _run(model, 3)
        assert tracker.residual_count.shape == (2, 1)
        assert torch.equal(
            tracker.residual_count[:, 0], torch.full((2,), 3.0)
        )
        payload = tracker.flush(global_step=1)
    finally:
        tracker.close()

    rms = {k for k in payload if k.startswith("debug/residual_rms/")}
    assert rms == {
        "debug/residual_rms/block_000_rep0",
        "debug/residual_rms/block_001_rep0",
    }


def test_loopie_separates_the_two_applications(tmp_path):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2))
    _perturb(model)
    model.train()
    tracker = _tracker(model, tmp_path, max_rep=2)
    try:
        _run(model, 3)
        assert torch.equal(tracker.residual_count[:, 0], torch.full((2,), 3.0))
        assert torch.equal(tracker.residual_count[:, 1], torch.full((2,), 3.0))
        payload = tracker.flush(global_step=1)
    finally:
        tracker.close()

    differ = [
        i
        for i in range(2)
        if payload[f"debug/residual_rms/block_{i:03d}_rep0"]
        != payload[f"debug/residual_rms/block_{i:03d}_rep1"]
    ]
    assert differ, "rep0 and rep1 are different points in the residual stream"


def test_forward_time_depth_lands_in_the_matching_rep(tmp_path):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2))
    _perturb(model)
    model.train()
    tracker = _tracker(model, tmp_path, max_rep=3)
    try:
        _run(model, 2, loop_count=3)
        assert torch.equal(tracker.residual_count[:, 2], torch.full((2,), 2.0))
        payload = tracker.flush(global_step=1)
    finally:
        tracker.close()
    assert "debug/residual_rms/block_000_rep2" in payload


def test_depth_beyond_max_rep_is_dropped(tmp_path):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2))
    _perturb(model)
    model.train()
    tracker = _tracker(model, tmp_path, max_rep=2)
    try:
        _run(model, 2, loop_count=3)
        assert tracker.residual_count.shape == (2, 2)
        payload = tracker.flush(global_step=1)
    finally:
        tracker.close()

    rms = {k for k in payload if k.startswith("debug/residual_rms/")}
    assert rms == {
        "debug/residual_rms/block_000_rep0",
        "debug/residual_rms/block_000_rep1",
        "debug/residual_rms/block_001_rep0",
        "debug/residual_rms/block_001_rep1",
    }


def test_eval_forwards_accumulate_nothing(tmp_path):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(_cfg(use_loopie=True, loopie_loop_count=2))
    model.eval()
    tracker = _tracker(model, tmp_path, max_rep=2)
    try:
        with torch.no_grad():
            _run(model, 3)
        assert float(tracker.residual_count.sum()) == 0.0
        payload = tracker.flush(global_step=1)
    finally:
        tracker.close()
    assert not [k for k in payload if k.startswith("debug/residual_rms/")]


def test_repeat_tap_adds_no_state_dict_keys():
    for cfg in (_cfg(), _cfg(use_loopie=True, loopie_loop_count=2)):
        model = PureProtModernBertMLM(cfg)
        assert not [k for k in model.state_dict() if "repeat_tap" in k]
