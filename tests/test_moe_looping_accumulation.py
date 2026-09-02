"""MoE per-layer state must accumulate across executions within one forward.

Before this, `last_aux_loss` / `last_expert_counts` were overwritten on every
execution, so under Huginn recycling (or Loopie) only the final iteration's
routing reached the loss and the pipeline's expert-bias correction. That is why
the recycling compatibility matrix used to reject `use_moe`.

sonicmoe kernels are CUDA-only, hence the skip.
"""

import pytest
import torch

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.moe import MoELayer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MoE sonicmoe kernels require CUDA"
)

BATCH, SEQ, TOP_K = 2, 8, 2
TOKENS = BATCH * SEQ


def _model(**kwargs):
    base = dict(
        vocab_size=32, hidden_size=64, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_kv_heads=4, attn_layer_pattern="F",
        wo_init="kaiming", no_mlp_on_first_layer=False,
        use_moe=True, moe_num_experts=4, moe_top_k=TOP_K, moe_leading_dense_layers=0,
        moe_kernel_backend="sonicmoe", mlp_activation="swiglu",
        moe_use_bias_correction=True,
    )
    base.update(kwargs)
    # bf16: the flash-attention varlen path rejects fp32.
    return PureProtModernBertMLM(ProtModernBertMLMConfig(**base)).to("cuda", torch.bfloat16)


def _moe_of(layer):
    found = [m for m in layer.modules() if isinstance(m, MoELayer)]
    return found[0] if found else None


def _forward(model):
    ids = torch.randint(0, 32, (BATCH, SEQ), device="cuda")
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())


def test_recycling_accumulates_aux_and_counts_per_execution():
    torch.manual_seed(0)
    model = _model(use_huginn_looping=True, recycling_loop_count=2,
                   recycling_num_prelude_layers=1, recycling_num_coda_layers=1)
    enc = model.model
    recurrent_ids = {id(l) for l in enc.recurrent_layers}
    _forward(model)

    for layer in enc.layers:
        moe = _moe_of(layer)
        if moe is None:
            continue
        expected = 2 if id(layer) in recurrent_ids else 1
        assert moe.last_aux_count.item() == expected
        # top_k experts selected per token, summed over executions
        assert moe.last_expert_counts.sum().item() == expected * TOKENS * TOP_K


def test_workspace_reuse_disabled_under_looping():
    """Cutlass reuses `_dispatch_counts` as the tensor saved for backward, so a
    second execution of the same layer would invalidate the first one's graph."""
    looped = _model(use_huginn_looping=True, recycling_loop_count=2,
                    recycling_num_prelude_layers=1, recycling_num_coda_layers=1)
    assert all(not m._reuse_dispatch_workspaces
               for m in looped.modules() if isinstance(m, MoELayer))
    plain = _model()
    assert all(m._reuse_dispatch_workspaces
               for m in plain.modules() if isinstance(m, MoELayer))


def test_aux_gradient_reaches_router_gate_across_iterations():
    torch.manual_seed(0)
    model = _model(use_huginn_looping=True, recycling_loop_count=2,
                   recycling_num_prelude_layers=1, recycling_num_coda_layers=1)
    out = _forward(model)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    moe = _moe_of(model.model.recurrent_layers[0])
    grad = moe.router.gate.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_state_resets_between_forwards():
    torch.manual_seed(0)
    model = _model(use_huginn_looping=True, recycling_loop_count=2,
                   recycling_num_prelude_layers=1, recycling_num_coda_layers=1)
    moe = _moe_of(model.model.recurrent_layers[0])
    _forward(model)
    first = (moe.last_aux_count.item(), moe.last_expert_counts.sum().item())
    _forward(model)
    assert (moe.last_aux_count.item(), moe.last_expert_counts.sum().item()) == first


def test_loopie_accumulates_per_repeat():
    torch.manual_seed(0)
    model = _model(num_hidden_layers=2, use_loopie=True, loopie_loop_count=2)
    out = _forward(model)
    assert torch.isfinite(out.loss)
    for layer in model.model.layers:
        moe = _moe_of(layer)
        if moe is None:
            continue
        assert moe.last_aux_count.item() == 2
        assert moe.last_expert_counts.sum().item() == 2 * TOKENS * TOP_K
