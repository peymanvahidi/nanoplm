"""Tests for Paired Head Attention (PHA) across fused/unfused QKV and MHA/GQA."""

import pytest
import torch


def _pha_config(*, fused_qkv: bool, num_attention_heads: int, num_kv_heads: int):
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

    return ModernBertConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        use_paired_head_attention=True,
        fused_qkv=fused_qkv,
        fused_up_gate=True,
        mlp_activation="swiglu",
        use_canon_layers=False,
        use_mhc_lite=False,
        use_repo=False,
        use_diff_attn_v2=False,
        use_noble=False,
    )


# num_attention_heads, num_kv_heads
_HEAD_CONFIGS = [
    pytest.param(4, 4, id="mha"),
    pytest.param(4, 2, id="gqa_2x"),
    pytest.param(8, 4, id="gqa_2x_wider"),
    pytest.param(8, 2, id="gqa_4x"),
]


class TestPHAForwardBackward:
    @pytest.mark.parametrize("num_attention_heads,num_kv_heads", _HEAD_CONFIGS)
    @pytest.mark.parametrize("fused_qkv", [True, False])
    def test_forward_backward(self, fused_qkv, num_attention_heads, num_kv_heads):
        from nanoplm.pretraining.models.modern_bert.modeling import (
            ModernBertForMaskedLM,
        )

        cfg = _pha_config(
            fused_qkv=fused_qkv,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
        )
        model = ModernBertForMaskedLM(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)
        attention_mask = torch.ones((bsz, seq), dtype=torch.long)
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)

        out = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        logits = out["logits"]
        loss = out["loss"]

        assert logits.shape == (bsz, seq, cfg.vocab_size)
        assert loss is not None
        assert torch.isfinite(loss).all()
        assert torch.isfinite(logits).all()

        loss.backward()

        # Sanity: attention projections received gradient signal.
        attn = model.model.layers[1].attn
        if fused_qkv:
            assert attn.Wqkv is not None
            assert attn.Wqkv.weight.grad is not None
            assert torch.isfinite(attn.Wqkv.weight.grad).all()
        else:
            assert attn.Wq is not None and attn.Wk is not None and attn.Wv is not None
            for proj in (attn.Wq, attn.Wk, attn.Wv):
                assert proj.weight.grad is not None
                assert torch.isfinite(proj.weight.grad).all()


class TestPHAValidators:
    def test_rejects_odd_num_kv_heads(self):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

        with pytest.raises(ValueError, match="even num_kv_heads"):
            ModernBertConfig(
                vocab_size=32,
                hidden_size=48,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=6,
                num_kv_heads=3,  # odd
                use_paired_head_attention=True,
                use_canon_layers=False,
            )

    def test_rejects_odd_num_attention_heads(self):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

        with pytest.raises(ValueError, match="even number of attention heads"):
            ModernBertConfig(
                vocab_size=32,
                hidden_size=30,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=3,  # odd
                num_kv_heads=3,
                use_paired_head_attention=True,
                use_canon_layers=False,
            )

    def test_accepts_gqa_with_even_kv_heads(self):
        """Previously rejected as 'MHA only'; now accepted when both head counts are even."""
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

        cfg = ModernBertConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_kv_heads=4,
            use_paired_head_attention=True,
            use_canon_layers=False,
        )
        assert cfg.use_paired_head_attention is True
        assert cfg.num_kv_heads == 4


class TestPHAPairReshapeShapes:
    """Verify _pair_varlen_qkv / _pair_sdpa_qkv use the correct head-count constants."""

    @pytest.mark.parametrize("num_attention_heads,num_kv_heads", _HEAD_CONFIGS)
    def test_pair_sdpa_qkv_shapes(self, num_attention_heads, num_kv_heads):
        from nanoplm.pretraining.models.modern_bert.modeling import (
            ModernBertAttention,
        )

        cfg = _pha_config(
            fused_qkv=True,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
        )
        attn = ModernBertAttention(cfg, layer_idx=1)

        bsz, seq, hd = 2, 6, cfg.head_dim
        q = torch.randn(bsz, num_attention_heads, seq, hd)
        k = torch.randn(bsz, num_kv_heads, seq, hd)
        v = torch.randn(bsz, num_kv_heads, seq, hd)
        cos = torch.randn(1, seq, 2 * hd)
        sin = torch.randn(1, seq, 2 * hd)

        q_p, k_p, v_p = attn._pair_sdpa_qkv(q, k, v, (cos, sin))

        assert q_p.shape == (bsz, num_attention_heads // 2, 2 * seq, hd)
        assert k_p.shape == (bsz, num_kv_heads // 2, 2 * seq, hd)
        assert v_p.shape == (bsz, num_kv_heads // 2, 2 * seq, hd)

    @pytest.mark.parametrize("num_attention_heads,num_kv_heads", _HEAD_CONFIGS)
    def test_pair_varlen_qkv_shapes(self, num_attention_heads, num_kv_heads):
        from nanoplm.pretraining.models.modern_bert.modeling import (
            ModernBertAttention,
        )

        cfg = _pha_config(
            fused_qkv=True,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
        )
        attn = ModernBertAttention(cfg, layer_idx=1)

        total, hd = 10, cfg.head_dim
        q = torch.randn(total, num_attention_heads, hd)
        k = torch.randn(total, num_kv_heads, hd)
        v = torch.randn(total, num_kv_heads, hd)
        cos = torch.randn(total, 2 * hd)
        sin = torch.randn(total, 2 * hd)
        cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)
        max_seqlen = 6

        q_p, k_p, v_p, cu_p, max_p = attn._pair_varlen_qkv(
            q, k, v, (cos, sin), cu_seqlens, max_seqlen
        )

        assert q_p.shape == (total * 2, num_attention_heads // 2, hd)
        assert k_p.shape == (total * 2, num_kv_heads // 2, hd)
        assert v_p.shape == (total * 2, num_kv_heads // 2, hd)
        assert torch.equal(cu_p, cu_seqlens * 2)
        assert max_p == max_seqlen * 2
