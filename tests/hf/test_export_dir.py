"""End-to-end export of a checkpoint -> a Hugging Face repo.

The dense-checkpoint fixture (`checkpoint`) is CPU-portable and exercises
most of the suite; a handful of tests are GPU-gated (FP8/TE, sonicmoe,
sReLU) because those source layouts only exist with real kernels.

Every model built here uses `wo_init="kaiming"` (R14-8): the default
`wo_init="zero"` zeroes every attention/MLP output projection at init, which
makes every residual branch a no-op and every native-vs-export forward
comparison trivially agree regardless of whether the branch under test is
correct -- see `nanoplm.hf.verify`'s module docstring.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open

import nanoplm.hf.export as export_module
from nanoplm.hf.export import export_checkpoint
from nanoplm.hf.verify import ExportVerificationError
from nanoplm.pretraining.fp8 import convert_fp8_targets
from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import PureProtModernBertMLM

WITH_CUDA = torch.cuda.is_available()
try:
    import transformer_engine.pytorch as te

    WITH_TE = True
except Exception:  # pragma: no cover - depends on the runtime image
    te = None
    WITH_TE = False
try:
    import sonicmoe  # noqa: F401

    WITH_SONICMOE = True
except Exception:  # pragma: no cover - depends on the runtime image
    WITH_SONICMOE = False

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=32,
    wo_init="kaiming",
)


def _write_checkpoint(
    root,
    model,
    model_cfg_kwargs,
    *,
    pretrain_cfg=None,
    manifest=None,
    training_state=None,
    run_name="run-1",
    step=100,
    write_weights=True,
):
    ckpt = root / run_name / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    if write_weights:
        torch.save(model.state_dict(), ckpt / "pytorch_model.bin")
    (ckpt / "model_config.yaml").write_text(yaml.safe_dump(model_cfg_kwargs))
    (ckpt / "pretrain_config.yaml").write_text(
        yaml.safe_dump(
            pretrain_cfg
            if pretrain_cfg is not None
            else {"optimizer": "normuon", "lr_schedule": "wsd"}
        )
    )
    (ckpt / "data_manifest.yaml").write_text(
        yaml.safe_dump(manifest or {"max_seq_len": 512, "num_register_tokens": 0})
    )
    (ckpt / "training_state.json").write_text(
        json.dumps(training_state or {"global_step": step})
    )
    return ckpt


def _spy(monkeypatch):
    """Wrap `nanoplm.hf.export.verify_exported`, recording every call's result."""
    calls = []
    real = export_module.verify_exported

    def wrapper(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(export_module, "verify_exported", wrapper)
    return calls


@pytest.fixture
def checkpoint(tmp_path):
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**TINY)).eval()
    return _write_checkpoint(tmp_path, model, TINY)


# --- the plan's seven tests (adapted per R14-8) ------------------------------


def test_export_writes_a_loadable_repo(checkpoint, tmp_path):
    out = export_checkpoint(checkpoint, tmp_path / "hf")
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
        "configuration_nanoplm.py",
        "modeling_nanoplm.py",
    ):
        assert (out / name).exists(), f"missing {name}"


def test_config_json_declares_model_type_and_auto_map(checkpoint, tmp_path):
    out = export_checkpoint(checkpoint, tmp_path / "hf")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "nanoplm"
    assert cfg["auto_map"]["AutoModel"] == "modeling_nanoplm.NanoPLMModel"
    assert (
        cfg["auto_map"]["AutoModelForMaskedLM"] == "modeling_nanoplm.NanoPLMForMaskedLM"
    )


def test_exported_model_reproduces_native_logits(checkpoint, tmp_path):
    from nanoplm.hf import NanoPLMForMaskedLM

    out = export_checkpoint(checkpoint, tmp_path / "hf")
    native = PureProtModernBertMLM(ProtModernBertMLMConfig(**TINY)).eval()
    native.load_state_dict(
        torch.load(
            checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
    )
    exported = NanoPLMForMaskedLM.from_pretrained(out, local_files_only=True).eval()
    ids = torch.tensor([[4, 5, 6, 7, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    with torch.no_grad():
        want = native(input_ids=ids, attention_mask=mask).logits
        got = exported(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_dtype_flag_is_honored(checkpoint, tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    out = export_checkpoint(checkpoint, tmp_path / "hf", dtype="bf16")
    with safe_open(out / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        assert keys
        for key in keys:
            assert f.get_tensor(key).dtype == torch.bfloat16, key
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["dtype"] == "bfloat16"
    assert len(calls) == 1
    assert calls[0].passed


def test_verify_runs_by_default_and_can_be_disabled(checkpoint, tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    assert (export_checkpoint(checkpoint, tmp_path / "a") / "config.json").exists()
    assert len(calls) == 1

    assert (
        export_checkpoint(checkpoint, tmp_path / "b", verify=False) / "config.json"
    ).exists()
    assert len(calls) == 1  # unchanged: verify=False must not call verify_exported


def test_default_out_dir_is_inside_the_checkpoint(checkpoint):
    out = export_checkpoint(checkpoint)
    assert out.resolve() == (checkpoint / "hf_export").resolve()
    assert (out / "config.json").exists()


def test_tokenizer_reloads_as_a_plain_fast_tokenizer(checkpoint, tmp_path):
    from transformers import PreTrainedTokenizerFast

    out = export_checkpoint(checkpoint, tmp_path / "hf")
    tok = PreTrainedTokenizerFast(tokenizer_file=str(out / "tokenizer.json"))
    # normalizer must survive: lowercase upcased, UZOB -> X
    assert tok("mkalc")["input_ids"] == tok("MKALC")["input_ids"]


# --- (a) fp32 export is bit-identical to the checkpoint file -----------------


def test_fp32_export_is_bit_identical_to_the_checkpoint_file(checkpoint, tmp_path):
    out = export_checkpoint(checkpoint, tmp_path / "hf")
    file_state = torch.load(
        checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    with safe_open(out / "model.safetensors", framework="pt") as f:
        saved_keys = set(f.keys())
        for key in saved_keys:
            assert key in file_state, f"unexpected key {key!r} in export"
            assert torch.equal(
                f.get_tensor(key), file_state[key]
            ), f"mismatch at {key!r}"
    for key in set(file_state) - saved_keys:
        # A key dropped from the safetensors file must be a tied duplicate:
        # exactly equal to some saved tensor (save_pretrained dedupes ties).
        assert any(
            torch.equal(file_state[key], file_state[other]) for other in saved_keys
        ), f"{key!r} is missing from the export and is not a tied duplicate"


# --- (b) out_dir must not exist or must be empty -----------------------------


def test_nonempty_out_dir_is_refused(checkpoint, tmp_path):
    out = tmp_path / "hf"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_text("do not touch")
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, out)
    assert sentinel.read_text() == "do not touch"
    assert list(out.iterdir()) == [sentinel]


def test_empty_out_dir_is_accepted(checkpoint, tmp_path):
    out = tmp_path / "hf"
    out.mkdir()
    result = export_checkpoint(checkpoint, out)
    assert result == out
    assert (out / "config.json").exists()


# --- (I1) the published dir respects the umask and inherits setgid ----------
#
# Regression: the staging dir used to be created with tempfile.mkdtemp (mode
# 0o700, always) and then chmod'd to 0o777 & ~umask. chmod cannot preserve a
# setgid bit the directory inherited from its (setgid) parent -- it clears it
# -- so every file written into the export afterwards (weights, tokenizer,
# bundle, README) got the exporting user's primary group instead of the
# project's shared group. Fixed by creating the staging dir with plain
# os.mkdir, which applies the umask and inherits setgid from the parent
# natively, with no chmod at all.


def test_published_dir_permissions_and_setgid_inheritance(checkpoint, tmp_path):
    out = tmp_path / "hf"
    result = export_checkpoint(checkpoint, out)

    # Read the umask without permanently changing it (briefly toggled here in
    # the TEST only -- export.py itself no longer does this at all).
    current_umask = os.umask(0)
    os.umask(current_umask)
    expected_rwx = 0o777 & ~current_umask
    # Mask off the setgid/setuid/sticky bits before comparing: on a setgid
    # parent, the kernel ORs setgid into a freshly `mkdir`ed subdirectory's
    # mode independently of the umask, which is the whole point of this test
    # -- it must not be mistaken for a umask mismatch.
    actual_mode = stat.S_IMODE(result.stat().st_mode)
    assert actual_mode & 0o777 == expected_rwx

    parent_mode = tmp_path.stat().st_mode
    if not (parent_mode & stat.S_ISGID):
        pytest.skip(f"{tmp_path} is not setgid on this filesystem")

    assert actual_mode & stat.S_ISGID, "published dir lost setgid"
    config_json = result / "config.json"
    assert (
        config_json.stat().st_gid == tmp_path.stat().st_gid
    ), "a file inside the published dir did not inherit the parent's group"


# --- (c) a verify failure leaves nothing written -----------------------------


def test_verify_failure_leaves_nothing_written(checkpoint, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise ExportVerificationError("boom")

    monkeypatch.setattr(export_module, "verify_exported", boom)

    push_calls = []

    class _FakeHfApi:
        def __init__(self, *a, **k):
            pass

        def create_repo(self, *a, **k):
            push_calls.append("create_repo")

        def upload_folder(self, *a, **k):
            push_calls.append("upload_folder")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)

    out = tmp_path / "hf"
    before = set(tmp_path.iterdir())
    with pytest.raises(ExportVerificationError):
        export_checkpoint(checkpoint, out, push_to_hub="someuser/some-repo")
    assert not out.exists()
    assert set(tmp_path.iterdir()) == before, "a staging dir was left behind"
    assert push_calls == []


# --- (d) FP8 end-to-end on CUDA + TE -----------------------------------------


@pytest.mark.skipif(
    not (WITH_CUDA and WITH_TE), reason="needs CUDA + Transformer Engine"
)
def test_fp8_export_end_to_end(tmp_path):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**TINY)).to("cuda").eval()
    n = convert_fp8_targets(model, target="mlp,attn_wqkv,attn_wo")
    assert n > 0
    ckpt = _write_checkpoint(
        tmp_path,
        model,
        TINY,
        pretrain_cfg={"fp8": True, "fp8_target": "mlp,attn_wqkv,attn_wo"},
    )

    out = export_checkpoint(ckpt, tmp_path / "hf")
    assert (out / "config.json").exists()


@pytest.mark.skipif(
    not (WITH_CUDA and WITH_TE), reason="needs CUDA + Transformer Engine"
)
def test_fp8_export_with_broken_gate_up_restore_is_rejected_and_writes_nothing(
    tmp_path, monkeypatch
):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**TINY)).to("cuda").eval()
    convert_fp8_targets(model, target="mlp,attn_wqkv,attn_wo")
    ckpt = _write_checkpoint(
        tmp_path,
        model,
        TINY,
        pretrain_cfg={"fp8": True, "fp8_target": "mlp,attn_wqkv,attn_wo"},
    )

    import nanoplm.pretraining.fp8 as fp8_module

    # A gate/up restore bug: te_fc1_to_native_wi becomes identity, looked up
    # by restore_dense_from_te as a bare module-global name.
    monkeypatch.setattr(fp8_module, "te_fc1_to_native_wi", lambda t, activation: t)

    out = tmp_path / "hf"
    before = set(tmp_path.iterdir())
    with pytest.raises(ExportVerificationError):
        export_checkpoint(ckpt, out)
    assert not out.exists()
    assert set(tmp_path.iterdir()) == before, "a staging dir was left behind"


@pytest.mark.skipif(
    not (WITH_CUDA and WITH_TE), reason="needs CUDA + Transformer Engine"
)
def test_fp8_layernorm_checkpoint_with_a_trained_te_norm_bias_is_refused(tmp_path):
    """The native LayerNorm is bias-free, but TE's LayerNormMLP owns a trainable
    `layer_norm_bias` that FP8 training moves off zero. The export used to drop
    it and still pass verify (the drift stayed under the loose FP8 tolerance);
    the restore now refuses, and nothing is written."""
    cfg = {**TINY, "norm_type": "layernorm"}
    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**cfg)).to("cuda").eval()
    convert_fp8_targets(model, target="mlp,attn_wqkv,attn_wo")
    with torch.no_grad():  # what a few FP8 training steps do to it
        model.model.layers[1].mlp.layernorm_mlp.layer_norm_bias.fill_(0.05)
    ckpt = _write_checkpoint(
        tmp_path,
        model,
        cfg,
        pretrain_cfg={"fp8": True, "fp8_target": "mlp,attn_wqkv,attn_wo"},
    )

    out = tmp_path / "hf"
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match=r"model\.layers\.1\.mlp"):
        export_checkpoint(ckpt, out)
    assert not out.exists()
    assert set(tmp_path.iterdir()) == before, "a staging dir was left behind"


# --- (e) sonicmoe end-to-end on CUDA -----------------------------------------


@pytest.mark.skipif(not (WITH_CUDA and WITH_SONICMOE), reason="needs CUDA + sonicmoe")
def test_sonicmoe_export_end_to_end(tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    cfg = {
        **TINY,
        "use_moe": True,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_kernel_backend": "sonicmoe",
    }
    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**cfg)).to("cuda").eval()
    ckpt = _write_checkpoint(tmp_path, model, cfg)

    out = export_checkpoint(ckpt, tmp_path / "hf")
    exported_cfg = json.loads((out / "config.json").read_text())
    assert exported_cfg["moe_kernel_backend"] == "cutlass"

    file_state = torch.load(
        ckpt / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    with safe_open(out / "model.safetensors", framework="pt") as f:
        expert_keys = [
            k for k in f.keys() if k.endswith("mlp.Wi") or k.endswith("mlp.Wo")
        ]
        assert expert_keys
        for key in expert_keys:
            assert torch.equal(f.get_tensor(key), file_state[key]), key

    assert len(calls) == 1
    assert calls[0].passed
    assert calls[0].tolerance > 1e-4  # class approx: native experts are bf16


# --- (f) srelu end-to-end on CUDA --------------------------------------------


@pytest.mark.skipif(not WITH_CUDA, reason="needs CUDA")
def test_srelu_export_end_to_end(tmp_path):
    cfg = {**TINY, "mlp_activation": "srelu"}
    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**cfg)).to("cuda").eval()
    ckpt = _write_checkpoint(tmp_path, model, cfg)

    out = export_checkpoint(ckpt, tmp_path / "hf")
    assert (out / "config.json").exists()


# --- (g) fp16 export with verify passes --------------------------------------


def test_fp16_export_passes_verify(checkpoint, tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    out = export_checkpoint(checkpoint, tmp_path / "hf", dtype="fp16")
    assert (out / "config.json").exists()
    assert len(calls) == 1
    assert calls[0].passed


# --- (h) RePO frozen state ----------------------------------------------------

_REPO_CFG = {**TINY, "num_kv_heads": 4, "use_repo": True, "repo_after_n_layers": 0}


def test_repo_export_before_warmup(tmp_path):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**_REPO_CFG)).eval()
    ckpt = _write_checkpoint(
        tmp_path,
        model,
        _REPO_CFG,
        pretrain_cfg={"repo_rope_warmup_steps": 5000},
        training_state={"global_step": 1000},
    )
    out = export_checkpoint(ckpt, tmp_path / "hf")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["repo_active"] is False


def test_repo_export_at_or_after_warmup(tmp_path):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**_REPO_CFG)).eval()
    ckpt = _write_checkpoint(
        tmp_path,
        model,
        _REPO_CFG,
        pretrain_cfg={"repo_rope_warmup_steps": 500},
        training_state={"global_step": 1000},
    )
    out = export_checkpoint(ckpt, tmp_path / "hf")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["repo_active"] is True


def test_repo_export_with_flipped_repo_active_is_rejected(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = PureProtModernBertMLM(ProtModernBertMLMConfig(**_REPO_CFG)).eval()
    ckpt = _write_checkpoint(
        tmp_path,
        model,
        _REPO_CFG,
        pretrain_cfg={"repo_rope_warmup_steps": 500},
        training_state={"global_step": 1000},
    )

    real = export_module.nanoplm_config_from_checkpoint

    def flipped(checkpoint_dir):
        cfg = real(checkpoint_dir)
        cfg.repo_active = not cfg.repo_active
        return cfg

    monkeypatch.setattr(export_module, "nanoplm_config_from_checkpoint", flipped)
    with pytest.raises(ExportVerificationError):
        export_checkpoint(ckpt, tmp_path / "hf")


# --- (i) push_to_hub with a fake HfApi ---------------------------------------


def test_push_to_hub_calls_the_fake_api_with_the_published_dir(
    checkpoint, tmp_path, monkeypatch
):
    calls = {"create_repo": [], "upload_folder": []}

    class _FakeHfApi:
        # No defaults on create_repo's kwargs (I2): the real HfApi.create_repo
        # defaults `exist_ok` to False, so a fake that supplies its own
        # default here would pass even if export.py stopped passing
        # `exist_ok` explicitly -- silently letting a real export 409 on an
        # existing repo.
        def __init__(self, *a, **k):
            pass

        def create_repo(self, repo_id, **kwargs):
            calls["create_repo"].append({"repo_id": repo_id, "kwargs": kwargs})

        def upload_folder(self, folder_path, repo_id, **kw):
            calls["upload_folder"].append(
                {"folder_path": folder_path, "repo_id": repo_id}
            )

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)

    out = export_checkpoint(
        checkpoint, tmp_path / "hf", push_to_hub="someuser/some-repo", private=True
    )

    assert len(calls["create_repo"]) == 1
    create_call = calls["create_repo"][0]
    assert create_call["repo_id"] == "someuser/some-repo"
    assert create_call["kwargs"]["exist_ok"] is True
    assert create_call["kwargs"]["private"] is True

    assert calls["upload_folder"] == [
        {"folder_path": str(out), "repo_id": "someuser/some-repo"}
    ]
    readme = (out / "README.md").read_text()
    assert "someuser/some-repo" in readme


# --- (j) off-CUDA refusals -----------------------------------------------------


def test_off_cuda_fp8_source_is_refused_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ckpt = _write_checkpoint(
        tmp_path,
        None,
        TINY,
        pretrain_cfg={"fp8": True},
        write_weights=False,  # never loaded -- proven by no pytorch_model.bin
    )
    out = tmp_path / "hf"
    with pytest.raises(RuntimeError):
        export_checkpoint(ckpt, out)
    assert not out.exists()


def test_off_cuda_sonicmoe_source_is_refused_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = {
        **TINY,
        "use_moe": True,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_kernel_backend": "sonicmoe",
    }
    ckpt = _write_checkpoint(tmp_path, None, cfg, write_weights=False)
    out = tmp_path / "hf"
    with pytest.raises(RuntimeError):
        export_checkpoint(ckpt, out)
    assert not out.exists()


def test_fp8_source_without_transformer_engine_is_refused(tmp_path, monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **k):
        if name == "transformer_engine":
            return None
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    ckpt = _write_checkpoint(
        tmp_path, None, TINY, pretrain_cfg={"fp8": True}, write_weights=False
    )
    out = tmp_path / "hf"
    with pytest.raises(RuntimeError, match="[Tt]ransformer Engine"):
        export_checkpoint(ckpt, out)
    assert not out.exists()


# --- (k) a forced-CPU export succeeds ----------------------------------------


def test_forced_cpu_export_succeeds_with_verify(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = _spy(monkeypatch)
    out = export_checkpoint(checkpoint, tmp_path / "hf")
    assert (out / "config.json").exists()
    assert len(calls) == 1
    assert calls[0].passed
