import click

from myplm.cli.data import data
from myplm.cli.train import train


@click.group()
@click.version_option()
@click.help_option('--help', '-h')
def cli():
    """Make your own protein language model"""
    pass


# Attach grouped subcommands
cli.add_command(data)
cli.add_command(train)


