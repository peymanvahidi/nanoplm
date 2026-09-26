"""Config derivation from a checkpoint: frozen runtime state + backend remap."""

import json

import pytest
import torch
import yaml

from nanoplm.eval.checkpoint import load_checkpoint
from nanoplm.hf import NanoPLMForMaskedLM, NanoPLMModel
from nanoplm.hf.checkpoint_config import nanoplm_config_from_checkpoint
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

BASE_MODEL_CFG = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=32,
)


def _make_checkpoint(
    tmp_path,
    model_cfg=None,
    pretrain_cfg=None,
    global_step=1000,
    training_state=None,
    write_pretrain_config=True,
):
    ckpt = tmp_path / "run-1" / "checkpoint-1000"
    ckpt.mkdir(parents=True)
    (ckpt / "model_config.yaml").write_text(
        yaml.safe_dump({**BASE_MODEL_CFG, **(model_cfg or {})})
    )
    if write_pretrain_config:
        (ckpt / "pretrain_config.yaml").write_text(yaml.safe_dump(pretrain_cfg or {}))
    state = {"global_step": global_step} if training_state is None else training_state
    (ckpt / "training_state.json").write_text(json.dumps(state))
    return ckpt


def test_sonicmoe_is_remapped_to_cutlass(tmp_path):
    # Both backends read the same Wi/Wo Parameters, so this is config-only.
    # sonicmoe imports a GPU-only package; cutlass has a CPU reference kernel.
    ckpt = _make_checkpoint(
        tmp_path, model_cfg={"use_moe": True, "moe_kernel_backend": "sonicmoe"}
    )
    cfg = nanoplm_config_from_checkpoint(ckpt)
    assert cfg.moe_kernel_backend == "cutlass"


def test_cutlass_backend_is_left_alone(tmp_path):
    ckpt = _make_checkpoint(
        tmp_path, model_cfg={"use_moe": True, "moe_kernel_backend": "cutlass"}
    )
    assert nanoplm_config_from_checkpoint(ckpt).moe_kernel_backend == "cutlass"


def test_repo_active_frozen_true_after_warmup(tmp_path):
    ckpt = _make_checkpoint(
        tmp_path,
        model_cfg={"use_repo": True},
        pretrain_cfg={"repo_rope_warmup_steps": 500},
        global_step=1000,
    )
    assert nanoplm_config_from_checkpoint(ckpt).repo_active is True


def test_repo_active_frozen_false_before_warmup(tmp_path):
    ckpt = _make_checkpoint(
        tmp_path,
        model_cfg={"use_repo": True},
        pretrain_cfg={"repo_rope_warmup_steps": 5000},
        global_step=1000,
    )
    assert nanoplm_config_from_checkpoint(ckpt).repo_active is False


def test_repo_warmup_falls_back_to_warmup_steps(tmp_path):
    # load_checkpoint uses repo_rope_warmup_steps, else warmup_steps.
    ckpt = _make_checkpoint(
        tmp_path,
        model_cfg={"use_repo": True},
        pretrain_cfg={"warmup_steps": 5000},
        global_step=1000,
    )
    assert nanoplm_config_from_checkpoint(ckpt).repo_active is False


def test_repo_active_fallback_to_training_state_warmup_when_no_pretrain_config(
    tmp_path,
):
    # load_checkpoint's precedence: no pretrain_config.yaml at all falls back
    # to training_state.json's own "warmup_steps" (default 0).
    ckpt = _make_checkpoint(
        tmp_path,
        model_cfg={"use_repo": True},
        write_pretrain_config=False,
        training_state={"global_step": 1000, "warmup_steps": 5000},
    )
    assert nanoplm_config_from_checkpoint(ckpt).repo_active is False


def test_prores_step_is_recorded(tmp_path):
    ckpt = _make_checkpoint(tmp_path, model_cfg={"use_prores": True}, global_step=777)
    cfg = nanoplm_config_from_checkpoint(ckpt)
    assert cfg.prores_step == 777
    assert cfg.source_global_step == 777


def test_accepts_a_run_directory(tmp_path):
    _make_checkpoint(tmp_path)
    cfg = nanoplm_config_from_checkpoint(tmp_path / "run-1")
    assert cfg.hidden_size == 64


def test_missing_model_config_is_a_clear_error(tmp_path):
    ckpt = tmp_path / "run-1" / "checkpoint-1"
    ckpt.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="model_config.yaml"):
        nanoplm_config_from_checkpoint(ckpt)


# --- CPU parity against the native loader -----------------------------------
#
# `load_checkpoint` (the native eval path) and `nanoplm_config_from_checkpoint`
# (the export path) resolve RePO/ProRes runtime state from the same on-disk
# facts through one shared helper (`resolve_checkpoint_runtime_state`). These
# cases build a real checkpoint on disk, load it both ways, and check the
# resolved runtime state agrees -- the thing Task 10's brief calls "the least
# visible correctness issue in the whole design".

_PARITY_CASES = {
    "repo_before_warmup": (
        # RePO requires equal Q/K head counts (no GQA), unlike BASE_MODEL_CFG's
        # default num_kv_heads=2.
        {"use_repo": True, "repo_after_n_layers": 0, "num_kv_heads": 4},
        {"repo_rope_warmup_steps": 5000},
        1000,
    ),
    "repo_after_warmup": (
        {"use_repo": True, "repo_after_n_layers": 0, "num_kv_heads": 4},
        {"repo_rope_warmup_steps": 500},
        1000,
    ),
    "prores_mid_warmup": (
        # alpha_l = min(step / (prores_T * (l+1)), 1): with prores_T=1000,
        # step=500 gives alpha0=0.5, alpha1=0.25 -- both strictly in (0, 1).
        {"use_prores": True, "prores_T": 1000},
        {},
        500,
    ),
}


@pytest.mark.parametrize(
    "model_cfg, pretrain_cfg, global_step",
    _PARITY_CASES.values(),
    ids=_PARITY_CASES.keys(),
)
def test_native_and_exported_runtime_state_agree(
    tmp_path, model_cfg, pretrain_cfg, global_step
):
    ckpt = _make_checkpoint(
        tmp_path,
        model_cfg=model_cfg,
        pretrain_cfg=pretrain_cfg,
        global_step=global_step,
    )
    native_model_cfg = ProtModernBertMLMConfig(**{**BASE_MODEL_CFG, **model_cfg})
    seed = PureProtModernBertMLM(native_model_cfg).eval()
    torch.save(seed.state_dict(), ckpt / "pytorch_model.bin")

    native = load_checkpoint(ckpt, native_model_cfg, torch.device("cpu"))
    hf = NanoPLMModel(nanoplm_config_from_checkpoint(ckpt)).eval()

    assert native.model.repo_active == hf.model.repo_active
    assert torch.equal(native.model._prores_alphas, hf.model._prores_alphas)
    if model_cfg.get("use_prores"):
        assert torch.all(hf.model._prores_alphas > 0)
        assert torch.all(hf.model._prores_alphas < 1)


def test_sonicmoe_checkpoint_produces_a_working_cpu_model(tmp_path):
    moe_cfg = {
        "use_moe": True,
        "moe_kernel_backend": "sonicmoe",
        "moe_num_experts": 4,
        "moe_top_k": 2,
    }
    ckpt = _make_checkpoint(tmp_path, model_cfg=moe_cfg)
    cfg = nanoplm_config_from_checkpoint(ckpt)
    assert cfg.moe_kernel_backend == "cutlass"

    # MoELayer.Wi/Wo are bare nn.Parameters (not nn.Linear), so a freshly
    # constructed NanoPLMForMaskedLM never initializes them -- only a real
    # weight load does, exactly like every real export/inference use of this
    # config. A native model (cutlass backend, same shapes) supplies real
    # values the same way a checkpoint's state_dict would.
    native_cfg = ProtModernBertMLMConfig(
        **{**BASE_MODEL_CFG, **moe_cfg, "moe_kernel_backend": "cutlass"}
    )
    torch.manual_seed(0)
    native = PureProtModernBertMLM(native_cfg).eval()

    model = NanoPLMForMaskedLM(cfg).eval()
    model.load_state_dict(native.state_dict(), strict=True)

    ids = torch.tensor([[4, 5, 6, 7, 1]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    assert torch.isfinite(logits).all()
