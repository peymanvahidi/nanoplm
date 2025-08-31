import click

from .data import data
from .train import train


@click.group()
@click.version_option()
@click.help_option('--help', '-h')
def cli():
    """ProtX - Knowledge distillation of ProtT5"""
    pass


# Attach grouped subcommands
cli.add_command(data)
cli.add_command(train)


