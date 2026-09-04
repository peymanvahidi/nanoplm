import pytest
from nanoplm.cli.pretrain import _check_pipeline_supported


class TestPipelineGating:
    def test_hf_pretraining_rejected(self):
        with pytest.raises(ValueError, match="pure_torch"):
            _check_pipeline_supported(pure_torch=False)

    def test_pure_torch_allowed(self):
        _check_pipeline_supported(pure_torch=True)
