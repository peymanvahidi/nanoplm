import pytest
from nanoplm.cli.pretrain import _check_pipeline_supported


class TestPipelineGating:
    def test_hf_pretraining_rejected(self):
        with pytest.raises(ValueError, match="pure_torch"):
            _check_pipeline_supported(pure_torch=False)

    def test_pure_torch_allowed(self):
        _check_pipeline_supported(pure_torch=True)


def _mixture_dir(root):
    from nanoplm.data.infra.mixture import write_mixture
    from tests.data.datasets.mixture_sources import plain_source
    a = plain_source(root / "a", 30, tag=5)
    b = plain_source(root / "b", 30, tag=6)
    write_mixture(root, [("a", a, 0.5, "a.fasta"), ("b", b, 0.5, "b.fasta")])
    return root


def test_a_mixture_without_pure_torch_names_the_flag(tmp_path):
    from click.testing import CliRunner
    from nanoplm.cli.pretrain import pretrain
    _mixture_dir(tmp_path / "mix")
    (tmp_path / "pretrain.yaml").write_text(
        "model: {hidden_size: 16, intermediate_size: 32, num_hidden_layers: 1, "
        "num_attention_heads: 2}\n"
        f"pretraining: {{dataset_dir: {tmp_path / 'mix'}, ckp_dir: {tmp_path / 'ckp'}, "
        "micro_batch_seqs: 4, global_batch_tokens: 512, train_tokens: 5120, "
        "lr_schedule: warmup_stable, warmup_steps: 1}\n")
    r = CliRunner().invoke(pretrain, ["from-yaml", str(tmp_path / "pretrain.yaml")])
    assert r.exit_code != 0
    assert "data mixtures need --pure-torch" in r.output
    assert "Validating mixture source" not in r.output    # refused before the shard checks


def test_the_hf_pipeline_refuses_a_mixture(tmp_path):
    from types import SimpleNamespace
    from nanoplm.data.infra.manifest import read_manifest
    from nanoplm.pretraining.pipeline import run_pretraining
    m = read_manifest(_mixture_dir(tmp_path))
    with pytest.raises(ValueError, match="data mixtures need --pure-torch"):
        run_pretraining(None, SimpleNamespace(dataset_dir=str(tmp_path)), m)
