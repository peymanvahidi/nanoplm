import torch

from nanoplm.pretraining import optim
from nanoplm.pretraining.config import PretrainingConfig


class _DummyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 4
        self.num_q_heads = 2
        self.num_kv_heads = 1
        self.lambda_proj = None
        self.Wqkv = torch.nn.Linear(8, 16, bias=False)


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _DummyAttention()
        self.norm = torch.nn.LayerNorm(8)


def _capture_optimizer_kwargs(monkeypatch):
    captured = {}

    def fake_build_optimizer(**kwargs):
        captured.update(kwargs)
        return captured

    monkeypatch.setattr(optim, "build_optimizer", fake_build_optimizer)
    return captured


def test_fused_qkv_uses_muon_split_sizes_by_default(monkeypatch):
    monkeypatch.delenv("NANOPLM_MUON_SPLIT_FUSED_QKV", raising=False)
    captured = _capture_optimizer_kwargs(monkeypatch)
    model = _DummyModel()

    optim.build_muon_optimizer(model, PretrainingConfig(dataset_dir="unused"))

    assert captured["muon_params"] == []
    assert list(captured["muon_split_groups"].keys()) == [(8, 4, 4)]
    assert captured["muon_split_groups"][(8, 4, 4)] == [model.attn.Wqkv.weight]


def test_fused_qkv_split_sizes_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("NANOPLM_MUON_SPLIT_FUSED_QKV", "0")
    captured = _capture_optimizer_kwargs(monkeypatch)
    model = _DummyModel()

    optim.build_muon_optimizer(model, PretrainingConfig(dataset_dir="unused"))

    assert captured["muon_split_groups"] == {}
    assert captured["muon_params"] == [model.attn.Wqkv.weight]
