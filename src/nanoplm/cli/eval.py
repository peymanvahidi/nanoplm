#!/usr/bin/env python3
"""nanoPLM CLI — eval subcommands for downstream benchmark evaluation."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from nanoplm.cli._common import (
    YAML_FIRST_SECTIONS,
    ConfigErrorGroup,
    log_level_option,
    unknown_keys_message,
)
from nanoplm.eval.config import EvalConfig
from nanoplm.utils.common import create_dirs, read_yaml
from nanoplm.utils.logger import logger

_VALID_FRAMEWORKS = {
    "pgym",
    "pbc_zeroshot_contact",
    "pbc_supervised",
    "pbc_supervised_contact",
}
_VALID_ZERO_SHOT_METHODS = {"masked_marginals", "wt_marginals", "pseudoperplexity"}

# Frameworks dropped in the biotrainer 2.0.0 migration.
_REMOVED_FRAMEWORKS = {"pbc", "flip"}


def _check_biotrainer_available() -> None:
    """Abort early when biotrainer is requested but not installed."""
    try:
        import biotrainer.autoeval  # noqa: F401
        import biotrainer_core.data_classes  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "biotrainer 2.1.1 is required for nanoplm eval but is not installed.\n"
            "Reinstall nanoplm to pull it in."
        )


def _validate_frameworks(frameworks: List[str]) -> None:
    if not frameworks:
        raise click.ClickException("At least one framework must be specified.")
    for fw in frameworks:
        if fw in _REMOVED_FRAMEWORKS:
            raise click.ClickException(
                f"Framework {fw!r} is no longer supported. nanoplm eval runs on "
                "biotrainer 2.1.1 and covers: "
                f"{sorted(_VALID_FRAMEWORKS)}."
            )
        if fw not in _VALID_FRAMEWORKS:
            raise click.ClickException(
                f"Unsupported framework: {fw!r}. "
                f"Expected one of: {sorted(_VALID_FRAMEWORKS)}"
            )


def _coerce_list_of_str(value: Any, field_name: str) -> List[str]:
    """Accept a string (wrap to [s]) or a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ValueError(
        f"{field_name} must be a string or list of strings; got {type(value).__name__}"
    )


def _load_eval_config(config: Optional[Dict[str, Any]]) -> EvalConfig:
    """Build an EvalConfig from the ``eval:`` block of a YAML file.

    Unknown keys raise ``ValueError``. Required keys missing raise ``ValueError``.
    """
    if config is None:
        raise ValueError("Eval configuration is required but not found in YAML")

    config = dict(config)
    expected_keys = set(EvalConfig.__annotations__.keys())
    present_keys = set(config.keys())

    extra = present_keys - expected_keys
    if extra:
        raise ValueError(unknown_keys_message("eval", extra, expected_keys))

    if "checkpoint_dir" not in config or not config["checkpoint_dir"]:
        raise ValueError("checkpoint_dir is required in eval configuration")

    kwargs: Dict[str, Any] = {}
    for key in present_keys:
        value = config.get(key)
        if value is not None:
            kwargs[key] = value

    # Integer fields.
    for key in ("zero_shot_batch_size", "loop_count"):
        if key in kwargs:
            try:
                kwargs[key] = int(kwargs[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {key} value: {kwargs[key]}. Must be an integer."
                ) from exc

    # Scalar string fields.
    str_fields = ("device", "embedder_name", "zero_shot_method", "dataset_dir")
    for key in str_fields:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = str(kwargs[key])

    # List-of-string fields.
    if "framework" in kwargs:
        kwargs["framework"] = _coerce_list_of_str(kwargs["framework"], "framework")

    # Boolean flags.
    for key in ("force_download", "development_mode", "torch_compile"):
        if key in kwargs:
            kwargs[key] = bool(kwargs[key])

    # Validate zero_shot_method against the allowed set.
    if "zero_shot_method" in kwargs and kwargs["zero_shot_method"] not in _VALID_ZERO_SHOT_METHODS:
        raise ValueError(
            f"Invalid zero_shot_method: {kwargs['zero_shot_method']!r}. "
            f"Expected one of: {sorted(_VALID_ZERO_SHOT_METHODS)}"
        )

    if kwargs.get("force_download") and kwargs.get("dataset_dir"):
        raise ValueError(
            "force_download and dataset_dir are mutually exclusive: a pre-populated "
            "dataset_dir does not use the download cache that force_download clears."
        )

    return EvalConfig(**kwargs)


@click.group(name="eval", cls=ConfigErrorGroup, sections=YAML_FIRST_SECTIONS)
@click.help_option("--help", "-h")
def eval():
    """Group of commands for downstream evaluation."""
    pass


@eval.command("run")
@log_level_option
@click.help_option("--help", "-h")
@click.option(
    "--checkpoint-dir",
    type=str,
    required=True,
    help=(
        "Checkpoint path. Accepts a checkpoint-* directory or a run-* directory "
        "(the latest checkpoint inside the run-* is used)."
    ),
)
@click.option(
    "--framework",
    type=click.Choice(sorted(_VALID_FRAMEWORKS)),
    multiple=True,
    default=("pgym",),
    help=(
        "Benchmark framework(s) to evaluate on. Pass multiple times to run "
        "several, e.g. --framework pgym --framework pbc_supervised."
    ),
    show_default=True,
)
@click.option(
    "--pbc-supervised-dataset",
    multiple=True,
    default=("secondary_structure", "scl"),
    help=(
        "PBC_SUPERVISED dataset(s) to run when --framework includes "
        "'pbc_supervised'. Restricting them also restricts the embedding, which "
        "is the expensive part. 'secondary_structure' carries the newPISCES364 "
        "test split. Pass 'all' to run every dataset."
    ),
    show_default=True,
)
@click.option(
    "--zero-shot-method",
    type=click.Choice(sorted(_VALID_ZERO_SHOT_METHODS)),
    default="masked_marginals",
    help="Zero-shot scoring method when --framework includes 'pgym'.",
    show_default=True,
)
@click.option(
    "--output-dir",
    type=str,
    default="autoeval_output",
    help="Directory to save autoeval results.",
    show_default=True,
)
@click.option(
    "--zero-shot-batch-size",
    type=int,
    default=60,
    help=(
        "Sequences per forward pass. Batches masked positions (masked-marginals) "
        "and mutated sequences (categorical Jacobian) that biotrainer would "
        "otherwise score one at a time."
    ),
    show_default=True,
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device override (auto-detected if not specified).",
)
@click.option(
    "--embedder-name",
    type=str,
    default=None,
    help="Name for the embedder in the report. Defaults to checkpoint dir name.",
)
@click.option(
    "--development-mode/--full-eval",
    default=True,
    help=(
        "Evaluate on subsampled task sets (fast) or the full benchmark "
        "(final numbers)."
    ),
    show_default=True,
)
@click.option(
    "--dataset-dir",
    type=str,
    default=None,
    help=(
        "Pre-downloaded benchmark dataset directory (biotrainer's "
        "custom_storage_path). Use on machines without internet access. "
        "Mutually exclusive with --force-download."
    ),
)
@click.option(
    "--force-download",
    is_flag=True,
    default=False,
    help="Force re-download of framework datasets (use when new tasks land upstream).",
)
@click.option(
    "--loop-count",
    type=int,
    default=None,
    help=(
        "Refinement depth to run the checkpoint at, overriding the depth it was "
        "trained with. Requires a looped (loopie or huginn) checkpoint."
    ),
)
@click.option(
    "--torch-compile/--no-torch-compile",
    default=True,
    help=(
        "Run the model through torch.compile with dynamic shapes. Roughly 2x "
        "faster; eager spends most of its time in unfused elementwise kernels."
    ),
    show_default=True,
)
def run(
    checkpoint_dir: str,
    framework: tuple,
    pbc_supervised_dataset: tuple,
    zero_shot_method: str,
    output_dir: str,
    zero_shot_batch_size: int,
    device: Optional[str],
    embedder_name: Optional[str],
    development_mode: bool,
    dataset_dir: Optional[str],
    force_download: bool,
    loop_count: Optional[int],
    torch_compile: bool,
):
    """Run downstream evaluation directly from CLI flags."""
    frameworks_list = list(framework)
    _validate_frameworks(frameworks_list)
    if force_download and dataset_dir:
        raise click.ClickException(
            "--force-download and --dataset-dir are mutually exclusive."
        )
    _check_biotrainer_available()

    eval_config = EvalConfig(
        checkpoint_dir=checkpoint_dir,
        framework=frameworks_list,
        output_dir=output_dir,
        device=device,
        embedder_name=embedder_name,
        zero_shot_method=zero_shot_method,
        zero_shot_batch_size=zero_shot_batch_size,
        development_mode=development_mode,
        dataset_dir=dataset_dir,
        force_download=force_download,
        loop_count=loop_count,
        torch_compile=torch_compile,
        pbc_supervised_datasets=list(pbc_supervised_dataset),
    )

    from nanoplm.eval.pipeline import run_eval
    run_eval(eval_config)


@eval.command("from-yaml")
@log_level_option
@click.help_option("--help", "-h")
@click.argument(
    "config",
    default="eval.yaml",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
def from_yaml(config: str):
    """Run downstream evaluation from a YAML file.

    Expected YAML shape:

        eval:
            checkpoint_dir: ...
            framework: [pgym]
            ...
    """
    eval_yaml = Path(config).resolve()

    raw = read_yaml(eval_yaml) or {}
    if not isinstance(raw, dict):
        raise click.ClickException(
            f"Expected top-level YAML mapping in {eval_yaml}; got {type(raw).__name__}."
        )
    eval_dict = raw.get("eval")

    eval_config = _load_eval_config(eval_dict)

    _validate_frameworks(eval_config.framework)
    _check_biotrainer_available()

    logger.info(f"Loaded eval config from {eval_yaml}")
    from nanoplm.eval.pipeline import run_eval
    run_eval(eval_config)


@eval.command("get-yaml")
@click.help_option("--help", "-h")
@click.argument(
    "output",
    required=False,
    type=click.Path(dir_okay=True, writable=True, resolve_path=True),
)
@click.option("--force", is_flag=True, default=False, help="Overwrite if file exists")
def get_yaml(output: Optional[str], force: bool):
    """Generate an evaluation YAML template.

    If OUTPUT is omitted, the file is saved as ./eval.yaml in the current directory.
    If OUTPUT is a directory, the file will be saved as eval.yaml inside it.
    """
    if output is None:
        output_path = Path.cwd() / "eval.yaml"
    else:
        output_path = Path(output)
        if output_path.is_dir():
            output_path = output_path / "eval.yaml"

    create_dirs(output_path)

    from nanoplm.cli.templates import load_template

    if output_path.exists() and not force:
        raise click.ClickException(
            f"File already exists: {output_path}. Use --force to overwrite."
        )

    output_path.write_text(load_template("eval"), encoding="utf-8")
    click.echo(f"Template written to: {output_path}")
