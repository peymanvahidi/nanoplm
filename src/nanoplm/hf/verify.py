"""Compare a freshly exported model against the native checkpoint.

Export runs on the machine where BOTH models execute -- the FP8/sonicmoe native
model and the dense portable one -- so this check is nearly free and it covers
the failure modes that are otherwise invisible: the TE gate/up ordering, the
sReLU transpose, and cutlass-vs-sonicmoe numerical agreement.

This loads the artifact a downstream user actually gets -- the vendored,
portable bundle behind `config.json`'s `auto_map` -- through transformers'
own remote-code loader, not `nanoplm.hf`'s in-process model classes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from nanoplm.utils.logger import logger

# Short, long, and non-standard-residue cases. The normalizer maps UZOB -> X,
# so the last sequence also exercises that path.
VERIFY_SEQUENCES: tuple[str, ...] = (
    "MKALCLL",
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKL",
    "MAIGTXBOUZAIGTMKALC",
)

# Measured (R12-6): both branches below replay literally the same eager
# forward on native and exported alike (see verify_exported's comment on
# native_model.model(...)), so nothing here has an algorithmic reason to
# differ. Probed on this node's H100 plus CPU, torch 2.12.1+cu130,
# 2026-09-28, hidden_size in {64, 256, 768, 1024}, 2-16 layers, fp32 and
# bf16: dense, MoE (cutlass), recycling (default noise sigma), sRelu (bf16 --
# the native Triton kernel does not run in fp32), and an FP8 source (mlp +
# attn_wqkv + attn_wo converted, restored copy re-exported) all measured
# max|dlogits| = max|dhidden| = 0.0 exactly. Both tolerances below are kept
# well above that measured 0.0 for whatever a real, much larger checkpoint
# the probe didn't cover might still do.
_TOL_DENSE = 1e-5
_TOL_FP8 = 1e-2

# Shared by both forwards below so a recycling model's `torch.randn_like`
# noise draw (modeling.py's recurrent step) comes out identical on both
# sides. Arbitrary; not tuned.
_VERIFY_SEED = 20260909


class ExportVerificationError(RuntimeError):
    """Raised when an exported model disagrees with the native checkpoint."""


@dataclass(frozen=True)
class VerificationResult:
    max_logit_diff: float
    max_hidden_diff: float
    tolerance: float
    passed: bool


def _tokenizer_special_ids(tokenizer) -> dict:
    """The special-token ids compared between the two tokenizers.

    `cls_token_id` falls back to `bos_token_id` so this covers both a
    CLS-style and a BOS-style scheme; ProtModernBertTokenizer defines neither
    when built with `use_bos_token=False` (the default), in which case both
    sides read None and the comparison passes trivially.
    """
    return {
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "cls_token_id": (
            tokenizer.cls_token_id
            if tokenizer.cls_token_id is not None
            else tokenizer.bos_token_id
        ),
    }


def _check_tokenizer(native_tokenizer, exported_tokenizer) -> dict:
    """Tokenizer round-trip (spec Section 5.7 item 4). Returns the native encoding."""
    encode = dict(
        return_tensors="pt", padding=True, truncation=False, add_special_tokens=True
    )
    encoded_native = native_tokenizer(list(VERIFY_SEQUENCES), **encode)
    encoded_exported = exported_tokenizer(list(VERIFY_SEQUENCES), **encode)

    mismatches = []
    if not torch.equal(encoded_native["input_ids"], encoded_exported["input_ids"]):
        mismatches.append("input_ids")
    if not torch.equal(
        encoded_native["attention_mask"], encoded_exported["attention_mask"]
    ):
        mismatches.append("attention_mask")
    native_ids = _tokenizer_special_ids(native_tokenizer)
    exported_ids = _tokenizer_special_ids(exported_tokenizer)
    mismatches += [
        name for name in native_ids if native_ids[name] != exported_ids[name]
    ]
    if mismatches:
        raise ExportVerificationError(
            "export verification failed: tokenizer differs from the native "
            f"tokenizer in {', '.join(mismatches)}"
        )
    return encoded_native


@torch.no_grad()
def verify_exported(
    native_model: torch.nn.Module,
    exported_dir: Union[str, Path],
    *,
    tokenizer,
    device: torch.device,
    fp8_source: bool,
) -> VerificationResult:
    exported_dir = Path(exported_dir)
    config_path = exported_dir / "config.json"
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportVerificationError(
            f"export verification failed: no config.json in {exported_dir}"
        ) from exc
    if "auto_map" not in raw_config:
        raise ExportVerificationError(
            f"export verification failed: {config_path} has no auto_map; "
            "AutoModelForMaskedLM would not resolve to the vendored bundle"
        )

    # The artifact a downstream user gets, loaded exactly the way they would
    # load it -- never nanoplm.hf's own NanoPLMForMaskedLM.
    exported = (
        AutoModelForMaskedLM.from_pretrained(
            exported_dir, trust_remote_code=True, local_files_only=True
        )
        .to(device)
        .eval()
    )
    exported_tokenizer = AutoTokenizer.from_pretrained(
        exported_dir, local_files_only=True
    )

    encoded = _check_tokenizer(tokenizer, exported_tokenizer)

    native_model = native_model.to(device).eval()
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    keep = mask.bool()

    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.manual_seed() reseeds every CUDA device's global generator, not
    # just `device` -- so every visible CUDA device (not only the target
    # one) must be in fork_rng's list, or a CPU-only verify would still leak
    # a GPU RNG-state change to the caller.
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        # Every native path computes logits as decoder(head(model_output)),
        # and native_model.model(...) below -- the SDPA path -- is exactly
        # what the export's own model(...) forward computes, so this
        # reproduces the native forward exactly. It also sidesteps a CUDA
        # trap: ModernBertForMaskedLM.forward (i.e. calling native_model(...)
        # directly) takes the flash-attention varlen branch whenever a mask
        # is given on a CUDA tensor, and flash-attn only accepts
        # fp16/bf16/fp8 -- an fp32 checkpoint would crash there.
        # native_model.model(...) never takes that branch, on any device.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(_VERIFY_SEED)
            native_hidden = native_model.model(input_ids=ids, attention_mask=mask)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(_VERIFY_SEED)
            exported_hidden = exported.model(input_ids=ids, attention_mask=mask)

        native_logits = native_model.decoder(native_model.head(native_hidden))
        exported_logits = exported.decoder(exported.head(exported_hidden))

        # Only real (unpadded) positions: a pad position's hidden state is
        # not meaningful and need not match.
        logit_diff = (
            (native_logits.float() - exported_logits.float())[keep].abs().max().item()
        )
        hidden_diff = (
            (native_hidden.float() - exported_hidden.float())[keep].abs().max().item()
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    tolerance = _TOL_FP8 if fp8_source else _TOL_DENSE
    failing = []
    if logit_diff > tolerance:
        failing.append(f"logits (max|delta|={logit_diff:.3e})")
    if hidden_diff > tolerance:
        failing.append(f"last_hidden_state (max|delta|={hidden_diff:.3e})")
    passed = not failing

    logger.info(
        "export verify: max|dlogits|=%.3e max|dhidden|=%.3e tol=%.1e -> %s",
        logit_diff,
        hidden_diff,
        tolerance,
        "PASS" if passed else "FAIL",
    )
    if failing:
        raise ExportVerificationError(
            "export verification failed: "
            + "; ".join(failing)
            + f", tolerance={tolerance:.1e}. "
            "For an FP8 source check the TE gate/up ordering in "
            "fp8.te_fc1_to_native_wi / te_fc2_to_native_wo; for MoE check the "
            "sonicmoe remap."
        )
    return VerificationResult(logit_diff, hidden_diff, tolerance, passed)
