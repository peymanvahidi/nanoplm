#!/usr/bin/env python3
"""nanoPLM CLI — eval subcommands for downstream benchmark evaluation."""

from pathlib import Path
from typing import Any, Dict, Optional

import click

from nanoplm.eval.config import EvalConfig
from nanoplm.utils.common import create_dirs, read_yaml
from nanoplm.utils.logger import logger

_VALID_FRAMEWORKS = {"pbc", "flip"}


def _check_biotrainer_available() -> None:
    """Abort early when biotrainer is requested but the extra is not installed."""
    try:
        import biotrainer.autoeval  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "biotrainer is required for nanoplm eval but is not installed.\n"
            "Install it with:  pip install nanoplm[eval]"
        )


def _validate_framework(framework: str) -> None:
    if framework not in _VALID_FRAMEWORKS:
        raise click.ClickException(
            f"Unsupported framework: {framework!r}. "
            f"Expected one of: {sorted(_VALID_FRAMEWORKS)}"
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

    extra = sorted(present_keys - expected_keys)
    if extra:
        raise ValueError(
            f"Unexpected eval configuration keys: {', '.join(extra)}"
        )

    if "checkpoint_dir" not in config or not config["checkpoint_dir"]:
        raise ValueError("checkpoint_dir is required in eval configuration")

    kwargs: Dict[str, Any] = {}
    for key in present_keys:
        value = config.get(key)
        if value is not None:
            kwargs[key] = value

    # Type coercion for fields that may arrive as strings from YAML.
    int_fields = ("batch_size", "min_seq_length", "max_seq_length", "parallel_tasks")
    for key in int_fields:
        if key in kwargs:
            try:
                kwargs[key] = int(kwargs[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {key} value: {kwargs[key]}. Must be an integer."
                ) from exc

    str_fields = ("framework", "device", "embedder_name")
    for key in str_fields:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = str(kwargs[key])

    return EvalConfig(**kwargs)


@click.group(name="eval")
@click.help_option("--help", "-h")
def eval():
    """Group of commands for downstream evaluation."""
    pass


@eval.command("run")
@click.help_option("--help", "-h")
@click.option(
    "--checkpoint-dir",
    type=str,
    required=True,
    help=(
        "Checkpoint path. Accepts a checkpoint-* directory, a run-* directory, "
        "or the keyword 'latest'."
    ),
)
@click.option(
    "--checkpoint-root",
    type=str,
    default="output/pretraining_checkpoints",
    help="Root searched when --checkpoint-dir=latest.",
    show_default=True,
)
@click.option(
    "--framework",
    type=click.Choice(sorted(_VALID_FRAMEWORKS)),
    default="pbc",
    help="Benchmark framework to evaluate on.",
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
    "--config-yaml",
    type=str,
    default="pretrain.yaml",
    help="Fallback pretrain.yaml for model config (when checkpoint has no model_config.yaml).",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for embedding computation.",
    show_default=True,
)
@click.option(
    "--min-seq-length",
    type=int,
    default=0,
    help="Minimum sequence length filter for benchmark datasets.",
    show_default=True,
)
@click.option(
    "--max-seq-length",
    type=int,
    default=2000,
    help="Maximum sequence length filter for benchmark datasets.",
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
    "--parallel-tasks",
    type=int,
    default=0,
    help=(
        "Number of parallel workers for downstream task training. "
        "0 = sequential, >0 = parallel mode."
    ),
    show_default=True,
)
def run(
    checkpoint_dir: str,
    checkpoint_root: str,
    framework: str,
    output_dir: str,
    config_yaml: str,
    batch_size: int,
    min_seq_length: int,
    max_seq_length: int,
    device: Optional[str],
    embedder_name: Optional[str],
    parallel_tasks: int,
):
    """Run downstream evaluation directly from CLI flags."""
    _validate_framework(framework)
    _check_biotrainer_available()

    eval_config = EvalConfig(
        checkpoint_dir=checkpoint_dir,
        checkpoint_root=checkpoint_root,
        framework=framework,
        output_dir=output_dir,
        config_yaml=config_yaml,
        batch_size=batch_size,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        device=device,
        embedder_name=embedder_name,
        parallel_tasks=parallel_tasks,
    )

    from nanoplm.eval.pipeline import run_eval

    run_eval(eval_config)


@eval.command("from-yaml")
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
            framework: pbc
            ...
    """
    config_path = Path(config)
    if config_path.is_absolute():
        eval_yaml = config_path
    else:
        eval_yaml = Path.cwd() / config_path

    raw = read_yaml(eval_yaml)
    eval_dict = raw.get("eval")

    eval_config = _load_eval_config(eval_dict)
    _validate_framework(eval_config.framework)
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

    template = (
        "# Evaluation configuration for nanoPLM\n"
        "#\n"
        "# Runs biotrainer's autoeval pipeline on a pure-torch nanoPLM checkpoint.\n"
        "# Requires:  pip install nanoplm[eval]\n"
        "#\n"
        "# The model config is loaded automatically from <checkpoint_dir>/model_config.yaml\n"
        "# when present, with config_yaml below as a fallback for older checkpoints.\n"
        "\n"
        "eval:\n"
        "  # Checkpoint path. Accepts a checkpoint-* directory, a run-* directory,\n"
        "  # or the keyword 'latest'.\n"
        '  checkpoint_dir: "output/pretraining_checkpoints/run-1/checkpoint-1"\n'
        '  checkpoint_root: "output/pretraining_checkpoints"\n'
        "\n"
        "  framework: pbc                     # one of: pbc, flip\n"
        '  output_dir: "autoeval_output"\n'
        "\n"
        "  # Fallback model config when the checkpoint has no model_config.yaml.\n"
        '  config_yaml: "pretrain.yaml"\n'
        "\n"
        "  batch_size: 16\n"
        "  min_seq_length: 0\n"
        "  max_seq_length: 2000\n"
        "\n"
        "  # device: cuda                     # auto-detected when omitted\n"
        "  # embedder_name: my-run            # defaults to checkpoint dir stem\n"
        "  parallel_tasks: 0                  # 0 = sequential, >0 = parallel workers\n"
    )

    if output_path.exists() and not force:
        raise click.ClickException(
            f"File already exists: {output_path}. Use --force to overwrite."
        )

    output_path.write_text(template, encoding="utf-8")
    click.echo(f"Template written to: {output_path}")
