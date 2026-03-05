import pytest
import torch


def _tiny_config(*, use_mhc_lite: bool, wrapping_level: str):
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

    return ModernBertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_mhc_lite=use_mhc_lite,
        mhc_n_streams=3,
        mhc_triton_fused=False,
        mhc_lite_wrapping_level=wrapping_level,
        use_canon_layers=False,
        use_repo=False,
    )


class TestMHCLiteWrappingLevel:
    def test_construction_layer(self):
        from nanoplm.pretraining.models.modern_bert.modeling import (
            MHCLiteBlock,
            ModernBertForMaskedLM,
        )

        cfg = _tiny_config(use_mhc_lite=True, wrapping_level="layer")
        model = ModernBertForMaskedLM(cfg)
        assert isinstance(model.model.layers[0], MHCLiteBlock)

    def test_construction_sublayers(self):
        from nanoplm.pretraining.models.modern_bert.modeling import (
            MHCLiteBlock,
            MHCLiteSublayersLayer,
            ModernBertForMaskedLM,
        )

        cfg = _tiny_config(use_mhc_lite=True, wrapping_level="sublayers")
        model = ModernBertForMaskedLM(cfg)
        layer0 = model.model.layers[0]
        assert isinstance(layer0, MHCLiteSublayersLayer)
        assert isinstance(layer0.mhc_attn, MHCLiteBlock)
        assert isinstance(layer0.mhc_mlp, MHCLiteBlock)

    @pytest.mark.parametrize("wrapping_level", ["layer", "sublayers"])
    def test_forward_backward(self, wrapping_level):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM

        cfg = _tiny_config(use_mhc_lite=True, wrapping_level=wrapping_level)
        model = ModernBertForMaskedLM(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)
        attention_mask = torch.ones((bsz, seq), dtype=torch.long)
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert out["logits"].shape == (bsz, seq, cfg.vocab_size)
        assert out["loss"] is not None
        out["loss"].backward()

    def test_validation_requires_use_mhc_lite(self):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

        with pytest.raises(ValueError, match="requires use_mhc_lite=true"):
            ModernBertConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                use_mhc_lite=False,
                mhc_lite_wrapping_level="sublayers",
                use_canon_layers=False,
            )

    def test_validation_rejects_invalid_value(self):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

        with pytest.raises(ValueError, match="mhc_lite_wrapping_level must be one of"):
            ModernBertConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                use_mhc_lite=True,
                mhc_lite_wrapping_level="bogus",
                use_canon_layers=False,
            )

