"""The generated model card. Static assertions on the rendered markdown, plus
one end-to-end check that the quickstart block it emits actually runs (R13-3):
extracted from the RENDERED card (not the QUICKSTART constant), in a fresh
subprocess with nanoplm/triton/flash-attn/TE blocked -- the same shape of
check tests/hf/test_vendor.py and tests/hf/test_verify.py use for the export
bundle itself.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from nanoplm.hf import NanoPLMConfig, NanoPLMForMaskedLM
from nanoplm.hf.model_card import render_model_card
from nanoplm.hf.vendor import AUTO_MAP, vendor_modeling_code
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

HERE = Path(__file__).resolve().parent

CFG = NanoPLMConfig(hidden_size=1024, num_hidden_layers=16, num_attention_heads=16)
PRETRAIN = {"optimizer": "normuon", "lr_schedule": "wsd", "train_tokens": 10_000_000}
MANIFEST = {
    "max_seq_len": 512,
    "train_sequences": 18000,
    "total_train_tokens": 4823117,
    "num_register_tokens": 0,
}


def _card(**overrides):
    kwargs = dict(
        config=CFG,
        pretrain_cfg=PRETRAIN,
        manifest=MANIFEST,
        param_count=62_893_760,
        repo_id="acme/nanoplm-base",
    )
    kwargs.update(overrides)
    return render_model_card(**kwargs)


# --- the plan's six tests (adapted per R13-1) -----------------------------------


def test_starts_with_hf_front_matter():
    card = _card()
    assert card.startswith("---\n")
    head = card.split("---")[1]
    assert "library_name: transformers" in head
    assert "pipeline_tag: fill-mask" in head
    assert "license: mit" in head
    for tag in ("protein-language-model", "biology", "protein", "nanoplm"):
        assert tag in head


def test_reports_architecture_and_parameters():
    card = _card()
    assert "1024" in card
    assert "62,893,760" in card or "62.9M" in card


def test_reports_training_provenance():
    card = _card()
    assert "normuon" in card
    assert "wsd" in card
    assert "4,823,117" in card


def test_quickstart_uses_trust_remote_code():
    card = _card()
    # R13-1: both from_pretrained calls carry trust_remote_code=True.
    assert card.count("trust_remote_code=True") >= 2
    assert "AutoTokenizer.from_pretrained" in card
    assert "AutoModel.from_pretrained" in card
    assert "acme/nanoplm-base" in card


def test_documents_the_device_matrix():
    card = _card().lower()
    for token in ("cpu", "mps", "cuda"):
        assert token in card


def test_survives_missing_provenance():
    card = _card(pretrain_cfg={}, manifest={}, repo_id=None)
    assert "trust_remote_code=True" in card
    assert "unknown" in card.lower()


# --- (a) the quickstart actually runs (R13-3) -----------------------------------


def _extract_quickstart(card: str) -> str:
    """The fenced ```python block from the RENDERED card, not the constant."""
    match = re.search(r"```python\n(.*?)```", card, re.DOTALL)
    assert match, "no fenced python block in the rendered card"
    return match.group(1)


def _build_export(tmp_path, *, num_register_tokens=0):
    """A tiny vendored export directory, reusing test_vendor.py's shape."""
    cfg = NanoPLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        vocab_size=32,
        num_register_tokens=num_register_tokens,
    )
    model = NanoPLMForMaskedLM(cfg).eval()
    model.config.auto_map = AUTO_MAP
    model.save_pretrained(tmp_path)
    ProtModernBertTokenizer(num_register_tokens=num_register_tokens).save_pretrained(
        tmp_path
    )
    vendor_modeling_code(tmp_path)
    return cfg


_RUNNER = r"""
import sys
from import_blocker import block_imports

block_imports(
    ("nanoplm", "triton", "flash_attn", "flash_attn_interface", "kernels",
     "transformer_engine", "quack", "sonicmoe")
)

exec(compile(open(sys.argv[1]).read(), sys.argv[1], "exec"))
"""


def _run_quickstart(tmp_path, snippet):
    script = tmp_path / "quickstart.py"
    script.write_text(snippet, encoding="utf-8")
    env = dict(os.environ)
    env.pop("NANOPLM_PORTABLE", None)
    env["PYTHONPATH"] = str(HERE)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_MODULES_CACHE"] = str(tmp_path / "hf_modules")
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )


@pytest.mark.parametrize("num_register_tokens", [0, 2], ids=["default", "registers"])
def test_quickstart_runs_and_prints_expected_shapes(tmp_path, num_register_tokens):
    cfg = _build_export(tmp_path, num_register_tokens=num_register_tokens)
    card = render_model_card(
        config=cfg,
        pretrain_cfg=PRETRAIN,
        manifest=MANIFEST,
        param_count=123,
        repo_id=str(tmp_path),
    )
    snippet = _extract_quickstart(card)
    result = _run_quickstart(tmp_path, snippet)
    assert result.returncode == 0, result.stderr[-6000:]

    sequences = ["MKALCLLLLPVLG", "MVLSPADKTNVKAAWGKV"]
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) == len(sequences)
    for seq, line in zip(sequences, lines):
        assert line.startswith(seq)
        assert f"torch.Size([{len(seq)}, {cfg.hidden_size}])" in line
        assert f"torch.Size([{cfg.hidden_size}])" in line


# --- (b) the device table marks MPS / sm70+ as not verified ---------------------


def _row_containing(card: str, token: str) -> str:
    (line,) = [l for l in card.splitlines() if token in l]
    return line


def test_device_table_distinguishes_verified_from_expected():
    card = _card().lower()
    assert "not verified" in card
    # MPS and non-Hopper CUDA rows carry the caveat, unlike CPU/sm90's.
    assert "not verified" in _row_containing(card, "mps")
    assert "not verified" in _row_containing(card, "sm70")
    cpu_row = _row_containing(card, "| cpu |")
    assert "not verified" not in cpu_row and "verified" in cpu_row
    sm90_row = _row_containing(card, "sm90")
    assert "not verified" not in sm90_row and "verified" in sm90_row


# --- (c) recycling / mHC-lite / MoE notes are conditional -----------------------


def test_no_feature_notes_for_a_default_config():
    card = _card().lower()
    assert "recurrent iteration" not in card
    assert "mhc" not in card
    assert "reference kernel" not in card


def test_recycling_note_appears_only_when_the_flag_is_set():
    cfg = NanoPLMConfig(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        use_huginn_looping=True,
        recycling_loop_count=2,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
        recycling_s0_noise_sigma=0.5,
    )
    card = _card(config=cfg).lower()
    assert "recurrent iteration" in card
    assert "seed" in card  # s0 noise > 0 => outputs vary unless seeded


def test_recycling_note_omits_seeding_caveat_when_noise_is_zero():
    cfg = NanoPLMConfig(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        use_huginn_looping=True,
        recycling_loop_count=2,
        recycling_num_prelude_layers=1,
        recycling_num_coda_layers=1,
        recycling_s0_noise_sigma=0.0,
    )
    card = _card(config=cfg).lower()
    assert "recurrent iteration" in card
    assert "unless you seed" not in card


def test_mhc_lite_note_appears_only_when_the_flag_is_set():
    cfg = NanoPLMConfig(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        use_mhc_lite=True,
        mhc_n_streams=4,
    )
    card = _card(config=cfg)
    assert "mhc" in card.lower()
    assert "(batch, seq, 4, hidden)" in card


def test_moe_note_appears_only_when_the_flag_is_set():
    cfg = NanoPLMConfig(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
    )
    card = _card(config=cfg).lower()
    assert "reference kernel" in card
    assert "slower" in card


# --- (d) training steps / nanoplm version, with "unknown" when absent -----------


def test_training_steps_and_version_rendered_when_present():
    cfg = NanoPLMConfig.from_model_config(
        CFG.to_model_config(), source_global_step=12_345, nanoplm_version="0.9.3"
    )
    card = _card(config=cfg)
    assert "12,345" in card
    assert "0.9.3" in card


def test_training_steps_and_version_are_unknown_when_absent():
    # NanoPLMConfig() defaults source_global_step=0, nanoplm_version="" --
    # never having been derived from a real checkpoint export.
    card = _card(config=CFG)
    assert "unknown" in card.lower()
    assert "12,345" not in card
