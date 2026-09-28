"""Checkpoint -> Hugging Face repo.

Runs on the training machine: FP8/TE dequantization needs Transformer Engine,
and verification needs both models to execute. What it writes is portable.

The flow (spec Section 5.2, narrowed by the Task 14 controller amendments):

1. Validate `dtype`, resolve `checkpoint_dir`/`out_dir`, and refuse early if
   `out_dir` already holds files (R14-4) -- before anything is loaded.
2. Read `model_config.yaml`/`pretrain_config.yaml` and apply the device
   policy (R14-7): refuse off-CUDA for an FP8 or sonicmoe source, and refuse
   an FP8 source when Transformer Engine is not importable, regardless of
   device.
3. Load the checkpoint the way `nanoplm eval` does (`load_checkpoint`) --
   this is also the model `verify_exported` compares the export against, and
   it is NEVER mutated in place (R14-1): a bug in the restore path must show
   up as a native/export mismatch, not agree with itself.
4. Build the exported model's weights from a SEPARATE copy of the
   checkpoint file's own values (R14-2): for a dense source this is just the
   file's state dict; for an FP8 source it is a fresh TE-layout copy loaded
   from the file and then run through `restore_dense_from_te`. This is what
   keeps a "fp32 export" honest when `load_checkpoint` has, for example,
   cast sonicmoe's experts to bf16 in place for its own (native) purposes.
5. Write everything into a fresh staging directory next to `out_dir`, run
   `verify_exported` against the untouched native model from step 3, and
   only then publish by renaming staging -> `out_dir` (R14-4). Any failure
   between staging and publish removes the staging directory and re-raises
   unchanged; nothing is ever left partially written at `out_dir`.
6. Push to the Hub only after a successful publish, and only the published
   directory.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Optional, Union

import torch
import yaml

from nanoplm.eval.checkpoint import (
    load_checkpoint,
    load_model_config_from_checkpoint,
    resolve_checkpoint_dir,
)
from nanoplm.hf.checkpoint_config import nanoplm_config_from_checkpoint
from nanoplm.hf.model_card import render_model_card
from nanoplm.hf.modeling_nanoplm import NanoPLMForMaskedLM
from nanoplm.hf.vendor import (
    AUTO_MAP,
    assert_no_unguarded_optional_imports,
    vendor_modeling_code,
)
from nanoplm.hf.verify import verify_exported
from nanoplm.utils.logger import logger

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

_STAGING_PREFIX = ".nanoplm-export-staging-"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _ensure_out_dir_is_writable(out_dir: Path) -> None:
    """R14-4 step 1: `out_dir` must not exist, or be an empty directory.

    Checked before anything is loaded or written, and never deletes or
    overwrites a user's files.
    """
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise FileExistsError(
            f"{out_dir} already exists and is not a directory. Remove it or "
            "pass a different out_dir."
        )
    if any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already exists and is not empty. Remove it or pass "
            "a different out_dir."
        )


def _sonicmoe_source(model_config) -> bool:
    return bool(getattr(model_config, "use_moe", False)) and (
        str(getattr(model_config, "moe_kernel_backend", "")).strip().lower()
        == "sonicmoe"
    )


def _apply_device_policy(
    *, device: torch.device, fp8_source: bool, sonicmoe_source: bool
) -> None:
    """R14-7: refuse early, before writing (or loading) anything."""
    if fp8_source and device.type != "cuda":
        raise RuntimeError(
            "This checkpoint was trained with fp8=true, so its weights are in "
            "Transformer Engine module layout. Export it on the training "
            "machine (CUDA + Transformer Engine required)."
        )
    if sonicmoe_source and device.type != "cuda":
        raise RuntimeError(
            "This checkpoint's model_config.yaml sets "
            "moe_kernel_backend='sonicmoe', a CUDA-only kernel. Export it on "
            "the training machine (CUDA required)."
        )
    if fp8_source and importlib.util.find_spec("transformer_engine") is None:
        raise RuntimeError(
            "This checkpoint was trained with fp8=true, which requires "
            "Transformer Engine to load and restore its weights, but "
            "Transformer Engine is not importable in this environment. "
            "Install Transformer Engine, or export on a machine that has it."
        )
    if device.type != "cuda":
        # Deliberate narrowing of the spec's "refuse off-CUDA": only FP8/TE
        # and sonicmoe checkpoints actually need a GPU to load. Refusing
        # every non-CUDA export would also make this whole module untestable
        # on a laptop.
        logger.warning(
            "export: no CUDA device. This works for a dense or "
            "cutlass-MoE checkpoint, but kernel-backed paths cannot be "
            "exercised here, so --verify proves less than it would on the "
            "training machine."
        )


def _build_export_state_dict(
    *,
    checkpoint_dir: Path,
    native: torch.nn.Module,
    fp8_source: bool,
) -> dict:
    """R14-2: the exported tensors carry the checkpoint FILE's values.

    `native` (loaded by `load_checkpoint`) may already have been mutated for
    its own purposes -- e.g. sonicmoe experts cast to bf16 in place -- so the
    export is built from a separate read of the file, never from
    `native.state_dict()`.
    """
    file_state_dict = torch.load(
        checkpoint_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    if not fp8_source:
        return file_state_dict

    from nanoplm.pretraining.fp8 import restore_dense_from_te

    # A fresh TE-layout copy of `native`, holding the FILE's own values (not
    # whatever `native` currently holds), then restored to dense modules.
    restore_source = copy.deepcopy(native).float()
    restore_source.load_state_dict(file_state_dict, strict=True)
    del file_state_dict
    restored = restore_dense_from_te(restore_source, restore_source.config)
    logger.info("export: restored %d FP8 block(s) to dense modules", restored)
    if restored == 0:
        raise RuntimeError(
            "pretrain_config.yaml has fp8=true but no TE blocks were found "
            "to restore. Refusing to export a possibly mislabeled checkpoint."
        )
    export_state_dict = restore_source.state_dict()
    del restore_source
    return export_state_dict


def _mkstagingdir(parent: Path) -> Path:
    """A fresh staging directory under `parent`, made with plain `os.mkdir`.

    Not `tempfile.mkdtemp`: it hardcodes mode 0o700, so a later `os.chmod`
    would be needed to fix permissions -- and `os.chmod` cannot restore a
    parent's setgid bit once cleared (I1: this project's directories on the
    cluster are setgid, so `chmod(0o777 & ~umask)` silently reparents every
    file later written into staging to the exporting user's primary group
    instead of the project group). `os.mkdir` lets the kernel apply the
    umask and inherit setgid from `parent` natively, with no chmod at all.
    """
    for _ in range(10):
        candidate = parent / f"{_STAGING_PREFIX}{secrets.token_hex(8)}"
        try:
            os.mkdir(candidate)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not create a unique staging directory under {parent}")


def export_checkpoint(
    checkpoint_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    dtype: str = "fp32",
    verify: bool = True,
    push_to_hub: Optional[str] = None,
    private: bool = False,
) -> Path:
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")

    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    out_dir = Path(out_dir) if out_dir else checkpoint_dir / "hf_export"
    _ensure_out_dir_is_writable(out_dir)

    model_config = load_model_config_from_checkpoint(checkpoint_dir)
    if model_config is None:
        raise FileNotFoundError(
            f"model_config.yaml not found in {checkpoint_dir}. Export needs "
            "a checkpoint written by a current nanoPLM pretraining run."
        )
    pretrain_cfg = _read_yaml(checkpoint_dir / "pretrain_config.yaml")
    manifest = _read_yaml(checkpoint_dir / "data_manifest.yaml")
    fp8_source = bool(pretrain_cfg.get("fp8", False))
    sonicmoe_source = _sonicmoe_source(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _apply_device_policy(
        device=device, fp8_source=fp8_source, sonicmoe_source=sonicmoe_source
    )

    # R14-3: the native reference is loaded from model_config.yaml (the raw,
    # un-remapped backend), never from hf_config.to_model_config() -- the
    # export's config is already sonicmoe->cutlass remapped, and comparing a
    # remapped model against itself would never exercise the backend
    # equivalence this check exists for.
    hf_config = nanoplm_config_from_checkpoint(checkpoint_dir)
    native = load_checkpoint(checkpoint_dir, model_config, device)

    export_state_dict = _build_export_state_dict(
        checkpoint_dir=checkpoint_dir, native=native, fp8_source=fp8_source
    )

    # R14-6: sReLU's sm90 Triton kernel is bf16-only and USE_TRITON_SRELU is
    # process-wide, so an fp32 native forward on CUDA crashes. Cast only
    # AFTER the export source above is taken, so the written weights stay
    # exact; the verify reference becomes native-in-bf16.
    mlp_activation = str(getattr(model_config, "mlp_activation", "")).lower()
    if device.type == "cuda" and mlp_activation == "srelu":
        native = native.to(torch.bfloat16)

    exported = NanoPLMForMaskedLM(hf_config)
    exported.load_state_dict(export_state_dict, strict=True)
    del export_state_dict
    exported = exported.to(dtype=DTYPES[dtype])

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = _mkstagingdir(out_dir.parent)
    try:
        hf_config.auto_map = dict(AUTO_MAP)
        exported.config.auto_map = dict(AUTO_MAP)
        exported.save_pretrained(staging, safe_serialization=True)
        native.tokenizer.save_pretrained(staging)

        vendor_modeling_code(staging)
        assert_no_unguarded_optional_imports(staging)

        param_count = sum(p.numel() for p in exported.parameters())
        (staging / "README.md").write_text(
            render_model_card(
                config=hf_config,
                pretrain_cfg=pretrain_cfg,
                manifest=manifest,
                param_count=param_count,
                repo_id=push_to_hub,
            )
        )

        if verify:
            verify_exported(
                native,
                staging,
                tokenizer=native.tokenizer,
                device=device,
                fp8_source=fp8_source,
            )

        # os.replace onto an existing EMPTY directory works on POSIX; onto a
        # nonexistent path it is a plain rename. `staging` was created in
        # `out_dir.parent`, so this is a same-filesystem, atomic publish.
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(push_to_hub, private=private, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=push_to_hub)
        logger.info("export: pushed to https://huggingface.co/%s", push_to_hub)

    logger.info("export: wrote %s", out_dir)
    return out_dir
