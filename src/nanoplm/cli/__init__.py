import click

from nanoplm.cli.data import data
from nanoplm.cli.distill import distill
from nanoplm.cli.pretrain import pretrain


@click.group(invoke_without_command=True)
@click.version_option()
@click.help_option("--help", "-h")
@click.pass_context
def cli(ctx):
    """Make your own protein language model"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


# Attach grouped subcommands
cli.add_command(data)
cli.add_command(distill)
cli.add_command(pretrain)
