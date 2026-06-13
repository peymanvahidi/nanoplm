import yaml
from click.testing import CliRunner
from nanoplm.cli.pretrain import get_yaml


def _emit(tmp_path):
    out = tmp_path / "pretrain.yaml"
    res = CliRunner().invoke(get_yaml, [str(out)])
    assert res.exit_code == 0, res.output
    return yaml.safe_load(out.read_text())


class TestGetYamlTemplate:
    def test_pretraining_block_uses_new_fields(self, tmp_path):
        doc = _emit(tmp_path)
        p = doc["pretraining"]
        assert "num_epochs" not in p
        assert "global_batch_size" not in p and "micro_batch_size" not in p
        assert p["lr_schedule"] == "wsd"
        assert p["warmup_steps"] == 1000
        assert p["stable_tokens"] == 9_000_000
        assert p["decay_tokens"] == 1_000_000
        assert p["decay_shape"] == "1-sqrt"
        assert p["lr_decay_to_fraction"] == 0.1
        assert p["micro_batch_seqs"] == 64
        assert p["global_batch_tokens"] == 256000
        assert p["global_batch_warmup_steps"] == 0
        # advanced/alternative fields must stay COMMENTED (absent as active keys)
        for k in ("train_tokens", "stable_steps", "decay_steps", "max_steps"):
            assert k not in p, f"{k} should remain a commented alternative, not active"

    def test_resume_block_uses_modes(self, tmp_path):
        doc = _emit(tmp_path)
        r = doc["resume"]
        assert r["mode"] == "decay"
        assert "resume_epochs" not in r and "reset_scheduler" not in r
        assert r["decay_tokens"] == 1_000_000
        assert r["decay_shape"] == "1-sqrt"
        assert "decay_steps" not in r, "decay_steps should remain a commented alternative"
