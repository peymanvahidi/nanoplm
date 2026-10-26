"""Pre-#167 mHC checkpoints (no mhc_head.* readout) are refused with a clear error
by both the resume loader and the eval loader, instead of a missing-keys error."""
import pytest
import torch

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM


def _cfg(use_mhc_lite=True):
    return ProtModernBertMLMConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=4, use_mhc_lite=use_mhc_lite, mhc_n_streams=4,
                                   mhc_triton_fused=False, use_canon_layers=False, use_repo=False)


def _save(tmp_path, drop_head):
    torch.manual_seed(0)
    sd = PureProtModernBertMLM(_cfg()).state_dict()
    assert any(k.startswith("model.mhc_head.") for k in sd)
    if drop_head:
        sd = {k: v for k, v in sd.items() if "mhc_head." not in k}
    torch.save(sd, tmp_path / "pytorch_model.bin")
    return tmp_path


def _resume(ckpt):
    from nanoplm.pretraining.pure_pipeline import _load_checkpoint

    return _load_checkpoint(PureProtModernBertMLM(_cfg()), None, None, ckpt, "cpu")


def _eval(ckpt):
    from nanoplm.eval.checkpoint import load_checkpoint

    return load_checkpoint(ckpt, _cfg(), torch.device("cpu"))


@pytest.mark.parametrize("load", [_resume, _eval], ids=["resume", "eval"])
def test_pre_rework_mhc_checkpoint_refused(tmp_path, load):
    with pytest.raises(RuntimeError, match="predates the mHC rework in #167"):
        load(_save(tmp_path, drop_head=True))


def test_eval_loads_current_mhc_checkpoint(tmp_path):
    _eval(_save(tmp_path, drop_head=False))


def test_non_mhc_model_unaffected():
    from nanoplm.pretraining.models.modern_bert.modeling import check_mhc_checkpoint

    m = PureProtModernBertMLM(_cfg(use_mhc_lite=False))
    check_mhc_checkpoint(m, m.state_dict())
