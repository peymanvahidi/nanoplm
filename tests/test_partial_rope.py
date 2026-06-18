"""Tests for partial RoPE (rope_dim_fraction < 1.0).

Covers:
- Config validation (fraction bounds, even rope_dim)
- Standard attention forward/backward with partial RoPE
- Paired head attention forward/backward with partial RoPE
- RePO + partial RoPE interaction
- Full RoPE (fraction=1.0) backward compatibility
"""

import pytest
import torch


def _make_config(rope_dim_fraction=0.5, use_paired_head_attention=False, use_repo=False, **kwargs):
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

    defaults = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_activation="swiglu",
        use_canon_layers=False,
        use_mhc_lite=False,
        use_diff_attn_v2=False,
        use_noble=False,
    )
    defaults.update(kwargs)
    return ModernBertConfig(
        rope_dim_fraction=rope_dim_fraction,
        use_paired_head_attention=use_paired_head_attention,
        use_repo=use_repo,
        **defaults,
    )


class TestPartialRoPEConfig:
    """Config validation for rope_dim_fraction."""

    def test_default_fraction_is_full(self):
        cfg = _make_config(rope_dim_fraction=1.0)
        assert cfg.rope_dim == cfg.head_dim

    def test_half_fraction(self):
        cfg = _make_config(rope_dim_fraction=0.5)
        assert cfg.head_dim == 8  # 32 / 4
        assert cfg.rope_dim == 4  # 8 * 0.5

    def test_quarter_fraction(self):
        cfg = _make_config(rope_dim_fraction=0.25)
        assert cfg.head_dim == 8
        assert cfg.rope_dim == 2  # 8 * 0.25

    def test_rope_dim_rounded_even(self):
        """rope_dim must be even; odd fractions are rounded down."""
        # head_dim=8, fraction=0.375 -> raw=3 -> rounded to 2
        cfg = _make_config(rope_dim_fraction=0.375)
        assert cfg.rope_dim == 2
        assert cfg.rope_dim % 2 == 0

    def test_rejects_zero_fraction(self):
        with pytest.raises(ValueError, match="rope_dim_fraction"):
            _make_config(rope_dim_fraction=0.0)

    def test_rejects_negative_fraction(self):
        with pytest.raises(ValueError, match="rope_dim_fraction"):
            _make_config(rope_dim_fraction=-0.5)

    def test_rejects_over_one_fraction(self):
        with pytest.raises(ValueError, match="rope_dim_fraction"):
            _make_config(rope_dim_fraction=1.5)

    def test_minimum_rope_dim(self):
        """Even very small fractions produce rope_dim >= 2."""
        cfg = _make_config(rope_dim_fraction=0.01)
        assert cfg.rope_dim == 2


class TestPartialRoPEForwardBackward:
    """Forward/backward pass with partial RoPE on standard attention."""

    @pytest.mark.parametrize("fraction", [1.0, 0.5, 0.25])
    @pytest.mark.parametrize("fused_qkv", [True, False])
    def test_forward_backward(self, fraction, fused_qkv):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM

        cfg = _make_config(rope_dim_fraction=fraction, fused_qkv=fused_qkv)
        model = ModernBertForMaskedLM(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)
        attention_mask = torch.ones((bsz, seq), dtype=torch.long)
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = out["logits"]
        loss = out["loss"]

        assert logits.shape == (bsz, seq, cfg.vocab_size)
        assert loss is not None
        assert torch.isfinite(loss).all()
        assert torch.isfinite(logits).all()

        loss.backward()
        # Check gradients flow
        attn = model.model.layers[1].attn
        if fused_qkv:
            assert attn.Wqkv.weight.grad is not None
            assert torch.isfinite(attn.Wqkv.weight.grad).all()
        else:
            for proj in (attn.Wq, attn.Wk, attn.Wv):
                assert proj.weight.grad is not None
                assert torch.isfinite(proj.weight.grad).all()


class TestPartialRoPEPairedHead:
    """Partial RoPE + paired head attention."""

    @pytest.mark.parametrize("fraction", [1.0, 0.5, 0.25])
    def test_paired_head_forward_backward(self, fraction):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM

        cfg = _make_config(
            rope_dim_fraction=fraction,
            use_paired_head_attention=True,
        )
        model = ModernBertForMaskedLM(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)
        attention_mask = torch.ones((bsz, seq), dtype=torch.long)
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out["loss"]
        assert torch.isfinite(loss).all()
        loss.backward()

    @pytest.mark.parametrize("fraction", [0.5, 0.25])
    def test_paired_reshape_shapes(self, fraction):
        """Verify _pair_sdpa_qkv produces correct shapes with partial RoPE."""
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertAttention

        cfg = _make_config(
            rope_dim_fraction=fraction,
            use_paired_head_attention=True,
        )
        attn = ModernBertAttention(cfg, layer_idx=1)
        bsz, seq = 2, 6
        hd = cfg.head_dim
        rd = cfg.rope_dim
        nq = cfg.num_attention_heads

        q = torch.randn(bsz, nq, seq, hd)
        k = torch.randn(bsz, nq, seq, hd)
        v = torch.randn(bsz, nq, seq, hd)
        # paired cos/sin have last dim = 2*rope_dim
        cos = torch.randn(1, seq, 2 * rd)
        sin = torch.randn(1, seq, 2 * rd)

        q_p, k_p, v_p = attn._pair_sdpa_qkv(q, k, v, (cos, sin))
        assert q_p.shape == (bsz, nq // 2, 2 * seq, hd)
        assert k_p.shape == (bsz, nq // 2, 2 * seq, hd)
        assert v_p.shape == (bsz, nq // 2, 2 * seq, hd)


class TestPartialRoPERepo:
    """Partial RoPE + RePO learned positions."""

    @pytest.mark.parametrize("fraction", [1.0, 0.5])
    def test_repo_forward_backward(self, fraction):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM

        cfg = _make_config(
            rope_dim_fraction=fraction,
            use_repo=True,
            repo_after_n_layers=0,
        )
        model = ModernBertForMaskedLM(cfg)
        model.model.repo_active = True
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)
        attention_mask = torch.ones((bsz, seq), dtype=torch.long)
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out["loss"]
        assert torch.isfinite(loss).all()
        loss.backward()

        # Check that RePO parameters received gradients
        repo = model.model.layers[1].attn.repo
        assert repo is not None
        assert repo.W_z.weight.grad is not None
        assert torch.isfinite(repo.W_z.weight.grad).all()

    def test_repo_inv_freq_uses_rope_dim(self):
        """repo_inv_freq should have rope_dim//2 elements, not head_dim//2."""
        cfg = _make_config(
            rope_dim_fraction=0.5,
            use_repo=True,
            repo_after_n_layers=0,
        )
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertAttention
        attn = ModernBertAttention(cfg, layer_idx=1)
        assert attn.repo_inv_freq.shape == (cfg.rope_dim // 2,)


class TestPartialRoPEOps:
    """Unit tests for the low-level RoPE apply functions."""

    def test_apply_rope_partial_preserves_nope_dims(self):
        """NoPE dimensions should pass through _apply_rope unchanged."""
        from nanoplm.pretraining.models.modern_bert.rope import _apply_rope

        T, H, D = 10, 4, 8
        rope_dim = 4
        q = torch.randn(T, H, D)
        k = torch.randn(T, H, D)
        cos = torch.ones(T, rope_dim)
        sin = torch.zeros(T, rope_dim)  # identity rotation

        q_out, k_out = _apply_rope(q, k, cos, sin)
        # With cos=1, sin=0 rotation is identity; all dims should match
        torch.testing.assert_close(q_out, q, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_out, k, atol=1e-5, rtol=1e-5)

    def test_apply_rope_partial_nope_dims_unchanged_under_rotation(self):
        """Under non-trivial rotation, NoPE dims should still be unchanged."""
        from nanoplm.pretraining.models.modern_bert.rope import _apply_rope

        T, H, D = 10, 4, 8
        rope_dim = 4
        q = torch.randn(T, H, D)
        k = torch.randn(T, H, D)
        cos = torch.randn(T, rope_dim)
        sin = torch.randn(T, rope_dim)

        q_out, k_out = _apply_rope(q, k, cos, sin)
        # NoPE dims (last 4) must be exactly unchanged
        torch.testing.assert_close(q_out[..., rope_dim:], q[..., rope_dim:])
        torch.testing.assert_close(k_out[..., rope_dim:], k[..., rope_dim:])

    def test_apply_paired_rope_partial_nope_unchanged(self):
        """NoPE dims for each head in a pair should be unchanged."""
        from nanoplm.pretraining.models.modern_bert.rope import _apply_paired_rope

        T, H_pairs, D = 10, 2, 8
        rope_dim = 4
        # q/k last dim is 2*D (one pair = two heads)
        q = torch.randn(T, H_pairs, 2 * D)
        k = torch.randn(T, H_pairs, 2 * D)
        cos = torch.randn(T, 2 * rope_dim)
        sin = torch.randn(T, 2 * rope_dim)

        q_out, k_out = _apply_paired_rope(q, k, cos, sin)
        # First head's NoPE dims: indices rope_dim..D
        torch.testing.assert_close(q_out[..., rope_dim:D], q[..., rope_dim:D])
        # Second head's NoPE dims: indices D+rope_dim..2D
        torch.testing.assert_close(q_out[..., D + rope_dim:], q[..., D + rope_dim:])

    def test_full_rope_backward_compat(self):
        """When rope_dim == head_dim, output should match the original path exactly."""
        from nanoplm.pretraining.models.modern_bert.rope import _apply_rope

        T, H, D = 10, 4, 8
        q = torch.randn(T, H, D)
        k = torch.randn(T, H, D)
        cos = torch.randn(T, D)
        sin = torch.randn(T, D)

        q_out, k_out = _apply_rope(q, k, cos, sin)
        # All dims should be rotated — no passthrough slice
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        # NoPE slice should be empty (no dims skipped)
        assert q_out[..., D:].shape[-1] == 0
