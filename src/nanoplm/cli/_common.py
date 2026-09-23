"""Plumbing shared by the nanoPLM CLI command groups."""

import os
from difflib import get_close_matches
from typing import Iterable, Sequence, Tuple

import click

# click.Group.format_commands truncates each subcommand's help to
# `formatter.width - 6 - <longest name>`, which is 64 columns on an 80-column
# terminal, and max_content_width cannot raise it because the terminal width
# binds first. Passing a limit nothing reaches leaves the full sentence for
# write_dl to wrap.
_NO_TRUNCATION = 10_000

Sections = Sequence[Tuple[str, Sequence[str]]]

# pretrain and eval expose the same three commands in the same roles, so they
# share one grouping rather than each declaring its own.
YAML_FIRST_SECTIONS: Sections = (
    ("Main commands (scaffold a config, then run it)", ("get-yaml", "from-yaml")),
    ("Direct invocation (every setting as a flag)", ("run",)),
)

LOG_LEVELS = ("debug", "info", "warning", "error")


class SectionedGroup(click.Group):
    """Lists subcommands under named sections, with their help untruncated.

    Pass ``sections`` to the ``click.group`` decorator to order and label the
    listing. A group with no sections still gets the untruncated listing.
    """

    def __init__(self, *args, sections: Sections = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.sections = sections

    def format_commands(self, ctx, formatter):
        visible = {}
        for name in self.list_commands(ctx):
            command = self.get_command(ctx, name)
            if command is not None and not command.hidden:
                visible[name] = command

        def rows(names):
            return [
                (name, visible[name].get_short_help_str(_NO_TRUNCATION))
                for name in names
                if name in visible
            ]

        placed = set()
        for header, names in self.sections:
            section = rows(names)
            if section:
                placed.update(name for name, _ in section)
                with formatter.section(header):
                    formatter.write_dl(section)

        rest = rows(name for name in visible if name not in placed)
        if rest:
            with formatter.section("Other commands" if placed else "Commands"):
                formatter.write_dl(rest)


def log_level_option(command):
    """Add ``--log-level`` to a command without widening its signature.

    ``click.Choice`` rejects an unknown level at the boundary, so the
    ``ValueError`` that ``set_log_level`` raises for one is unreachable here.
    """

    def apply(ctx, param, value):
        from nanoplm.utils import set_log_level

        set_log_level(value)
        return value

    return click.option(
        "--log-level",
        type=click.Choice(LOG_LEVELS, case_sensitive=False),
        default="info",
        show_default=True,
        is_eager=True,
        expose_value=False,
        callback=apply,
        help="Log level for nanoplm output.",
    )(command)


class ConfigErrorGroup(SectionedGroup):
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
