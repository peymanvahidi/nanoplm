import pytest
import torch

from nanoplm.hf import NanoPLMConfig, NanoPLMForMaskedLM, NanoPLMModel

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=32,
)
IDS = torch.tensor([[4, 5, 6, 7, 1]])
MASK = torch.ones_like(IDS)


def test_hidden_states_absent_by_default():
    model = NanoPLMModel(NanoPLMConfig(**TINY)).eval()
    with torch.no_grad():
        out = model(input_ids=IDS, attention_mask=MASK)
    assert out.hidden_states is None


def test_hidden_states_one_per_layer_plus_embeddings():
    cfg = NanoPLMConfig(**TINY)
    model = NanoPLMModel(cfg).eval()
    with torch.no_grad():
        out = model(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    # embeddings output + one per transformer layer
    assert len(out.hidden_states) == cfg.num_hidden_layers + 1
    for h in out.hidden_states:
        assert h.shape[:2] == IDS.shape
        assert h.shape[-1] == cfg.hidden_size


def test_last_hidden_state_is_final_norm_of_last_entry():
    model = NanoPLMModel(NanoPLMConfig(**TINY)).eval()
    with torch.no_grad():
        out = model(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    # The final captured entry is the last layer's raw output (pre-final-norm);
    # last_hidden_state is that same tensor after final_norm. Bit-exact for the
    # default config (verified at bfe6083).
    assert torch.equal(
        model.model.final_norm(out.hidden_states[-1]), out.last_hidden_state
    )


def test_hooks_are_removed_after_forward():
    model = NanoPLMModel(NanoPLMConfig(**TINY)).eval()
    with torch.no_grad():
        model(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    assert model.model.embeddings._forward_hooks == {}
    for layer in model.model.layers:
        assert layer._forward_hooks == {}


def test_hooks_are_removed_after_forward_raises():
    # An input longer than max_position_embeddings makes ModernBertModel.forward
    # raise ValueError AFTER hooks are registered (the SDPA-path length check
    # runs before any layer executes). Hooks must still be cleaned up.
    cfg = NanoPLMConfig(**TINY)
    model = NanoPLMModel(cfg).eval()
    max_pos = model._inner_config.max_position_embeddings
    long_ids = torch.randint(0, cfg.vocab_size, (1, max_pos + 1))
    long_mask = torch.ones_like(long_ids)
    with torch.no_grad():
        with pytest.raises(ValueError):
            model(
                input_ids=long_ids, attention_mask=long_mask, output_hidden_states=True
            )
    assert model.model.embeddings._forward_hooks == {}
    for layer in model.model.layers:
        assert layer._forward_hooks == {}


def test_masked_lm_also_reports_hidden_states():
    cfg = NanoPLMConfig(**TINY)
    model = NanoPLMForMaskedLM(cfg).eval()
    with torch.no_grad():
        out = model(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    assert len(out.hidden_states) == cfg.num_hidden_layers + 1
    assert out.logits.shape == (1, 5, cfg.vocab_size)


def test_hidden_states_gradient_flows_from_intermediate_layer():
    cfg = NanoPLMConfig(**TINY)
    model = NanoPLMModel(cfg)
    out = model(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    out.hidden_states[1].sum().backward()
    grad = model.model.embeddings.tok_embeddings.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad) > 0


def test_output_hidden_states_falls_back_to_config():
    cfg = NanoPLMConfig(output_hidden_states=True, **TINY)
    model = NanoPLMModel(cfg).eval()
    with torch.no_grad():
        out = model(input_ids=IDS, attention_mask=MASK)
    assert out.hidden_states is not None
    assert len(out.hidden_states) == cfg.num_hidden_layers + 1


def test_explicit_false_overrides_config_true():
    cfg = NanoPLMConfig(output_hidden_states=True, **TINY)
    model = NanoPLMModel(cfg).eval()
    with torch.no_grad():
        out = model(input_ids=IDS, attention_mask=MASK, output_hidden_states=False)
    assert out.hidden_states is None


def test_recycling_reports_one_entry_per_iteration():
    # R7-1: one entry per recurrent *iteration*, not per layer firing.
    # 1 embeddings + 1 prelude + 2 iterations (recurrent_layers[-1] only) + 1 coda.
    cfg = NanoPLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        vocab_size=32,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
        recycling_loop_count=2,
        use_huginn_looping=True,
    )
    model = NanoPLMModel(cfg).eval()
    with torch.no_grad():
        out = model(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    # Recycling forwards are non-deterministic even in eval (random initial
    # state) -- only the count is asserted, never values.
    assert len(out.hidden_states) == 5


def test_loopie_reports_one_entry_per_layer():
    cfg = NanoPLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_kv_heads=2,
        vocab_size=32,
        use_loopie=True,
        loopie_loop_count=2,
    )
    model = NanoPLMModel(cfg).eval()
    with torch.no_grad():
        out = model(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    # Loopie repeats happen inside each layer's forward, so a layer hook fires
    # once per layer regardless of loopie_loop_count.
    assert len(out.hidden_states) == cfg.num_hidden_layers + 1


def test_mhc_lite_layer_entries_keep_stream_dimension():
    cfg = NanoPLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_kv_heads=2,
        vocab_size=32,
        use_mhc_lite=True,
    )
    model = NanoPLMModel(cfg).eval()
    with torch.no_grad():
        out = model(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    # Embeddings output is (B, S, C); every layer output under mHC-lite carries
    # an extra stream dimension (B, S, n, C) with n = mhc_n_streams. This is
    # documented behaviour, not reshaped away -- see the helper's docstring.
    embeddings_entry, *layer_entries = out.hidden_states
    assert embeddings_entry.shape == (*IDS.shape, cfg.hidden_size)
    for h in layer_entries:
        assert h.shape == (*IDS.shape, cfg.mhc_n_streams, cfg.hidden_size)
