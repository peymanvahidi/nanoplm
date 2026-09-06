"""Plumbing shared by the nanoPLM CLI command groups."""

import os
from difflib import get_close_matches
from typing import Iterable

import click


class ConfigErrorGroup(click.Group):
    """Turns a config-validation failure into one clean ``Error: ...`` line.

    Every subcommand under this group builds a config from user YAML or user
    flags, and every validator on that path reports a bad value by raising
    ``ValueError``. Catching it once here keeps the validators free of any
    click import and gives a typo the same presentation wherever it is caught.

    ``ValueError`` is the whole caught set on purpose. ``click.exceptions.Exit``
    and ``click.Abort`` both subclass ``RuntimeError``, so a wider net swallows
    the clean exit that ``--help`` raises and reports it as a failure.

    Set ``NANOPLM_DEBUG=1`` to get the traceback back.
    """

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except ValueError as e:
            if os.environ.get("NANOPLM_DEBUG"):
                raise
            raise click.ClickException(str(e)) from e


def unknown_keys_message(
    label: str, unknown: Iterable[str], valid: Iterable[str]
) -> str:
    """Build the error text for config keys that no dataclass field matches.

    The 0.7 cutoff is the lowest one that suggests nothing for ``adam_lr``,
    which a looser cutoff matches to the unrelated ``adam_epsilon``.
    """
    known = sorted(valid)
    described = []
    for key in sorted(unknown):
        closest = get_close_matches(key, known, n=1, cutoff=0.7)
        if closest:
            described.append(f"{key} (did you mean {closest[0]}?)")
        else:
            described.append(key)
    return f"Unexpected {label} configuration keys: {', '.join(described)}"
