"""nanoplm export — checkpoint to a Hugging Face repo."""

import click

from nanoplm.hf.export import DTYPES, export_checkpoint


@click.command(name="export")
@click.help_option("--help", "-h")
@click.argument(
    "checkpoint",
    type=click.Path(exists=True, file_okay=False, readable=True),
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory (default: <checkpoint>/hf_export).",
)
@click.option(
    "--push-to-hub",
    "push_to_hub",
    default=None,
    help="Repo id to upload to, e.g. 'acme/nanoplm-base'.",
)
@click.option(
    "--private", is_flag=True, default=False, help="Create the Hub repo as private."
)
@click.option(
    "--dtype",
    type=click.Choice(sorted(DTYPES)),
    default="fp32",
    show_default=True,
    help="Exported weight dtype. fp32 is the portable default.",
)
@click.option(
    "--verify/--no-verify",
    default=True,
    show_default=True,
    help="Compare exported vs native outputs and refuse to write on mismatch.",
)
def export(checkpoint, out_dir, push_to_hub, private, dtype, verify):
    """Convert a nanoPLM checkpoint into a Hugging Face model repo.

    Run this on the training machine: FP8 checkpoints need Transformer Engine
    to dequantize, and verification runs both models. The result is
    portable, needing only torch and transformers (trust_remote_code=True),
    and has been verified on CPU and H100; other devices (MPS, other CUDA
    GPUs) are expected to work but are unverified.
    """
    try:
        path = export_checkpoint(
            checkpoint,
            out_dir,
            dtype=dtype,
            verify=verify,
            push_to_hub=push_to_hub,
            private=private,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote {path}")
    if push_to_hub:
        click.echo(f"Pushed to https://huggingface.co/{push_to_hub}")
