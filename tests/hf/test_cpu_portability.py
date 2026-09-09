"""The exported model must run with no Triton, no TE, no flash-attn."""
import pytest
import torch

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _tiny(**overrides):
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        vocab_size=32,
    )
    kwargs.update(overrides)
    return ProtModernBertMLMConfig(**kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_moe_model_runs_without_triton(device):
    model = PureProtModernBertMLM(
        _tiny(use_moe=True, moe_num_experts=4, moe_top_k=2)
    ).eval().to(device)
    ids = torch.tensor([[4, 5, 6, 7, 1, 0]], device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    assert out.logits.shape == (1, 6, 32)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_grouped_gemm_dispatches_on_mps():
    import nanoplm.pretraining.models.modern_bert.moe_grouped_gemm_ops  # noqa: F401

    a = torch.randn(8, 16)
    b = torch.randn(2, 16, 32)
    counts = torch.tensor([4, 4], dtype=torch.int64)
    expected = torch.ops.nanoplm_moe.grouped_gemm(a, b, counts)
    got = torch.ops.nanoplm_moe.grouped_gemm(
        a.to("mps"), b.to("mps"), counts.to("mps")
    )
    assert got.shape == expected.shape
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-4, atol=1e-4)
