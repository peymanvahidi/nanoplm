"""Derive an exported NanoPLMConfig from a nanoPLM checkpoint directory."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Union

from nanoplm.eval.checkpoint import (
    load_model_config_from_checkpoint,
    resolve_checkpoint_dir,
    resolve_checkpoint_runtime_state,
)
from nanoplm.hf.configuration_nanoplm import NanoPLMConfig
from nanoplm.utils.logger import logger


def nanoplm_config_from_checkpoint(checkpoint_dir: Union[str, Path]) -> NanoPLMConfig:
    """Build the config the exported model will ship with.

    Two things happen here that the raw model_config.yaml cannot express:

    1. Step-dependent runtime state is frozen. RePO activation and the ProRes
       alphas are reconstructed from training_state.json at load time in the
       native path; Hugging Face has no notion of a training step, so the
       resolved values are written into the config instead.
    2. moe_kernel_backend 'sonicmoe' is remapped to 'cutlass'. Both backends
       read the same Wi/Wo Parameters, so no weights change -- but sonicmoe
       imports a GPU-only package while cutlass has a CPU reference kernel.
    """
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    model_config = load_model_config_from_checkpoint(checkpoint_dir)
    if model_config is None:
        raise FileNotFoundError(
            f"model_config.yaml not found in {checkpoint_dir}. Export needs a "
            "checkpoint written by a current nanoPLM pretraining run."
        )

    runtime_state = resolve_checkpoint_runtime_state(checkpoint_dir, model_config)
    if model_config.use_repo:
        logger.info(
            "export: freezing repo_active=%s (step=%d, warmup=%d)",
            runtime_state.repo_active,
            runtime_state.global_step,
            runtime_state.repo_warmup_steps,
        )

    if (
        model_config.use_moe
        and str(model_config.moe_kernel_backend).strip().lower() == "sonicmoe"
    ):
        logger.info(
            "export: remapping moe_kernel_backend sonicmoe -> cutlass "
            "(same weights; cutlass has a portable CPU kernel)"
        )
        model_config.moe_kernel_backend = "cutlass"

    try:
        nanoplm_version = version("nanoplm")
    except PackageNotFoundError:
        nanoplm_version = ""

    return NanoPLMConfig.from_model_config(
        model_config,
        repo_active=runtime_state.repo_active,
        prores_step=runtime_state.global_step,
        source_global_step=runtime_state.global_step,
        nanoplm_version=nanoplm_version,
    )
