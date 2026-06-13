import pytest
from click.testing import CliRunner

from nanoplm.cli.pretrain import _load_pretrain_config, pretrain


def test_pretrain_config_accepts_span_mlm_fields():
    cfg = _load_pretrain_config(
        {
            "dataset_dir": "/tmp/dataset",
            # wsd is the default schedule and requires a duration budget.
            "stable_tokens": 9000000,
            "decay_tokens": 1000000,
            "mlm_probability": "0.3",
            "mask_replace_prob": "0.8",
            "random_token_prob": "0.1",
            "keep_probability": "0.1",
            "mlm_masking_strategy": "SPAN",
            "mlm_span_mean_length": "4.5",
            "mlm_span_max_length": "12",
        }
    )

    assert cfg.mlm_probability == 0.3
    assert cfg.mask_replace_prob == 0.8
    assert cfg.random_token_prob == 0.1
    assert cfg.keep_probability == 0.1
    assert cfg.mlm_masking_strategy == "span"
    assert cfg.mlm_span_mean_length == 4.5
    assert cfg.mlm_span_max_length == 12


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("mlm_masking_strategy", "bad", "mlm_masking_strategy"),
        ("mlm_span_mean_length", 0, "mlm_span_mean_length"),
        ("mlm_span_max_length", 0, "mlm_span_max_length"),
    ],
)
def test_pretrain_config_rejects_invalid_span_mlm_fields(field, value, match):
    raw = {"dataset_dir": "/tmp/dataset", field: value}

    with pytest.raises(ValueError, match=match):
        _load_pretrain_config(raw)


def test_pretrain_run_help_documents_span_mlm_flags():
    result = CliRunner().invoke(pretrain, ["run", "--help"])

    assert result.exit_code == 0
    assert "--mlm-masking-strategy" in result.output
    assert "--mlm-span-mean-length" in result.output
    assert "--mlm-span-max-length" in result.output
