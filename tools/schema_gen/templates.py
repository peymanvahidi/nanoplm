# tools/schema_gen/templates.py
"""Statically extract a get-yaml template string and parse its fields + inline comments.
No imports of nanoplm; uses ast to read string literals and a line parser for the YAML body."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from tools.schema_gen.sources import TEMPLATE_SOURCES


@dataclass
class ExtractedField:
    section: str          # top-level key, e.g. "model" / "pretraining" / "data_params"
    key: str              # field name
    raw_default: str      # exact text of the value, e.g. '64', '"adamw"', '.inf', 'null'
    eol_comment: str      # end-of-line comment text (no leading '#')
    group: str = ""       # nearest `# ── Title ───` banner above the key ("" if none)
    preceding: list[str] = field(default_factory=list)  # full-line comments above the key

    @property
    def description(self) -> str:
        parts = [*self.preceding]
        if self.eol_comment:
            parts.append(self.eol_comment)
        return " ".join(p.strip() for p in parts if p.strip())


def _concat_str_parts(node: ast.AST) -> str | None:
    """Return the concatenated string value if node is a (possibly parenthesized) run of str literals."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # implicit concatenation -> a single Constant in 3.8+, but explicit '+' or tuple-in-parens:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _concat_str_parts(node.left)
        right = _concat_str_parts(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def load_template_text(surface: str) -> str:
    """Return the raw YAML template text for a docs surface.

    Sources may be:
    - a packaged ``.yaml`` file (read as UTF-8 text), or
    - a ``.py`` file with a module-level string constant / ``get_yaml`` body
      (static AST extract; no import of nanoplm).
    """
    spec = TEMPLATE_SOURCES[surface]
    path = Path(spec.file)
    if path.suffix in {".yaml", ".yml"}:
        return path.read_text(encoding="utf-8")

    tree = ast.parse(path.read_text(encoding="utf-8"))

    # Case 1: module-level assignment `SYMBOL = "..."`
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == spec.symbol:
                    val = _concat_str_parts(n.value)
                    if val is not None:
                        return val

    # Case 2: a function named SYMBOL whose body assigns `template = ( "..." ... )`
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == spec.symbol:
            for stmt in ast.walk(n):
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Name) and t.id == "template":
                            val = _concat_str_parts(stmt.value)
                            if val is not None:
                                return val
    raise ValueError(f"Could not extract template for surface={surface} from {spec.file}")


_TOP = re.compile(r"^(?P<key>[A-Za-z_][\w]*):")  # column-0 lines in these templates are bare section headers
_FIELD = re.compile(r"^\s+(?P<key>[A-Za-z_][\w]*):\s?(?P<val_and_comment>.*)$")

# A group banner: `# ── Norms & MLP ──────────────`. The pretrain template uses
# these to break its long sections into labelled blocks; the docs reference
# renders the same blocks, so the grouping cannot drift from the template.
# Box-drawing U+2500 only, so an ordinary `# --- note ---` comment stays prose.
_GROUP = re.compile(r"^─+\s*(?P<title>\S.*?)\s*─+$")
# A titleless rule (`# =========`, `# ─────────`). Never a description.
_RULE = re.compile(r"^[─=_-]{3,}$")

# A buffered full-line comment shaped like `<identifier>: ...`.
_COMMENTED_FIELD = re.compile(r"^(?P<key>[A-Za-z_]\w*):")


def _describes(comment: str, field_key: str) -> bool:
    """Whether a buffered full-line comment documents ``field_key``.

    Most comments document the field directly below them and are kept. The one
    exception is a *commented-out other field*: an optional ``# num_kv_heads: 8``
    line, say, or a value-less placeholder like
    ``# max_steps:   # ↔ train_tokens``, sitting directly above an unrelated
    field. Such a line must NOT become that field's description. It is
    recognised as ``<identifier>:`` for a *different* key with either a single
    scalar value (``8``, ``"x"``, ``false``, ``null``) or no value at all.
    Human-written headings and prose that merely start with ``Word:`` (e.g.
    ``# Optimizer: adamw | muon | normuon``, ``# Note: paths are RELATIVE ...``,
    ``# resume_epochs: 0 (or unset) continues ...``) carry multi-word values and
    are kept. The key comparison is case-insensitive so a ``# Optimizer:`` heading
    is recognised as documenting the ``optimizer`` field.
    """
    m = _COMMENTED_FIELD.match(comment)
    if not m or m.group("key").lower() == field_key.lower():
        return True
    value, _ = _split_value_comment(comment[m.end():])
    # Keep only genuine prose: a multi-word value. A commented-out config field
    # carries a single scalar value (`# num_kv_heads: 8`) or none at all
    # (`# max_steps:   # ↔ train_tokens`). Both must be dropped.
    return len(value.split()) > 1


def _preceding_for(pending: list[str], field_key: str) -> list[str]:
    """Select, from the buffered full-line comments, those that document ``field_key``.

    A commented-out *other* field in the buffer documents itself, not the field
    below it, and so does the prose directly above it, which is typically the
    heading that introduces the commented-out block::

        # Advanced step overrides (each replaces its *_tokens counterpart):
        # stable_steps:  # <-> stable_tokens (wsd)
        # max_steps:     # <-> train_tokens
        optimizer: "adamw"   # <- must NOT inherit any of the three lines above

    So the description of a field is the comment run that follows the LAST
    commented-out field line in the buffer. (Those commented-out fields are real
    config keys; they are documented via overlay.yaml's ``x-extra-fields``.)
    """
    last_commented_out = -1
    for i, comment in enumerate(pending):
        if not _describes(comment, field_key):
            last_commented_out = i
    return pending[last_commented_out + 1:]


def _split_value_comment(s: str) -> tuple[str, str]:
    """Split 'value  # comment' honoring quotes/braces. Returns (raw_default, eol_comment)."""
    in_s = in_d = False
    depth = 0
    for i, ch in enumerate(s):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch in "{[" and not in_s and not in_d:
            depth += 1
        elif ch in "}]" and not in_s and not in_d:
            depth = max(0, depth - 1)
        elif ch == "#" and not in_s and not in_d and depth == 0:
            return s[:i].strip(), s[i + 1 :].strip()
    return s.strip(), ""


def parse_template(text: str) -> tuple[list[ExtractedField], dict[tuple[str, str], str]]:
    """Parse a template into its fields plus its per-group intro notes.

    A group note is the prose between a `# ── Title ───` banner and the first
    field under it. That prose describes the *block*: "Gradient accumulation is
    inferred from those two...", "Splits the L-layer stack into: prelude ...",
    "adam_* also drives the AdamW side...". The field parser has nowhere to put it
    except the first field's description, which misattributes a block-wide note to
    one arbitrary key. Captured here instead, keyed by (section, group).

    Prose that follows a field is unaffected: it still documents the next field,
    which is what the resume block's FOOTGUN note above `mode:` relies on.
    """
    fields: list[ExtractedField] = []
    notes: dict[tuple[str, str], str] = {}
    section = ""
    group = ""
    at_group_start = False  # no field seen yet under the current banner
    pending: list[str] = []  # buffered full-line comments awaiting the next field
    # The same comments with their indentation intact, for group notes only.
    pending_raw: list[str] = []

    def flush() -> None:
        pending.clear()
        pending_raw.clear()

    for raw in text.splitlines():
        if not raw.strip():
            flush()
            continue
        if raw.lstrip().startswith("#"):
            body = raw.lstrip()[1:]
            comment = body.strip()
            # Banners and rules label a block; they never describe the field
            # below them, so they are consumed rather than buffered. (The
            # templates also leave a blank line after each, which would clear
            # the buffer anyway, but this makes it independent of that.)
            m_group = _GROUP.match(comment)
            if m_group:
                group = m_group.group("title")
                at_group_start = True
                flush()
            elif _RULE.match(comment):
                flush()
            else:
                pending.append(comment)
                # Drop only the conventional single space after `#`, so the
                # comment's own indentation survives into the note.
                pending_raw.append(body[1:] if body.startswith(" ") else body)
            continue
        m_top = _TOP.match(raw)
        if m_top and not raw.startswith((" ", "\t")):
            section = m_top.group("key")
            group = ""
            at_group_start = False
            flush()
            continue
        m_field = _FIELD.match(raw)
        if m_field:
            key = m_field.group("key")
            val, comment = _split_value_comment(m_field.group("val_and_comment"))
            # Prose sitting between the banner and this, the block's first field,
            # belongs to the block, so take it whole rather than through
            # _preceding_for, which exists to strip commented-out *fields* from a
            # single field's description.
            preceding = _preceding_for(pending, key)
            if at_group_start and group:
                # Newlines kept, unlike a field description. Several of these
                # notes are laid out as ASCII in the template (an aligned
                # hardware table under Precision, an indented schedule tree under
                # Duration, the prelude/recurrent/coda split under Recycling) and
                # collapsing them to one line destroys the structure that makes
                # them readable. The docs render them preformatted.
                note = "\n".join(pending_raw).strip("\n")
                if note:
                    notes[(section, group)] = note
                    preceding = []
                at_group_start = False
            fields.append(
                ExtractedField(
                    section=section,
                    key=key,
                    raw_default=val,
                    eol_comment=comment,
                    group=group,
                    preceding=preceding,
                )
            )
            flush()
    return fields, notes


def parse_fields(text: str) -> list[ExtractedField]:
    return parse_template(text)[0]
