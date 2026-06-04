import pytest
from nanoplm.cli.pretrain import _check_pipeline_supported


class TestPipelineGating:
    def test_hf_pretraining_rejected(self):
        with pytest.raises(ValueError, match="pure_torch"):
            _check_pipeline_supported(pure_torch=False, pure_te=False, lr_schedule="cosine",
                                      resume_mode=None)

    def test_te_allows_warmup_stable(self):
        _check_pipeline_supported(pure_torch=False, pure_te=True, lr_schedule="warmup_stable",
                                  resume_mode=None)

    def test_te_rejects_wsd(self):
        with pytest.raises(ValueError, match="pure_torch"):
            _check_pipeline_supported(pure_torch=False, pure_te=True, lr_schedule="wsd",
                                      resume_mode=None)

    def test_te_rejects_decay_resume(self):
        with pytest.raises(ValueError, match="pure_torch"):
            _check_pipeline_supported(pure_torch=False, pure_te=True, lr_schedule="warmup_stable",
                                      resume_mode="decay")

    def test_pure_torch_allows_everything(self):
        _check_pipeline_supported(pure_torch=True, pure_te=False, lr_schedule="wsd",
                                  resume_mode="decay")
