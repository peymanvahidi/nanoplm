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

# A buffered full-line comment shaped like `<identifier>: ...`.
_COMMENTED_FIELD = re.compile(r"^(?P<key>[A-Za-z_]\w*):")


def _describes(comment: str, field_key: str) -> bool:
    """Whether a buffered full-line comment documents ``field_key``.

    Most comments document the field directly below them and are kept. The one
    exception is a *commented-out other field* — e.g. an optional
    ``# num_kv_heads: 8`` line, or a value-less placeholder like
    ``# max_steps:   # ↔ train_tokens``, sitting directly above an unrelated
    field — which must NOT become that field's description. Such a line is
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
    # (`# max_steps:   # ↔ train_tokens`) — both must be dropped.
    return len(value.split()) > 1


def _preceding_for(pending: list[str], field_key: str) -> list[str]:
    """Select, from the buffered full-line comments, those that document ``field_key``.

    A commented-out *other* field in the buffer documents itself, not the field
    below it — and so does the prose directly above it, which is typically the
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


def parse_fields(text: str) -> list[ExtractedField]:
    fields: list[ExtractedField] = []
    section = ""
    pending: list[str] = []  # buffered full-line comments awaiting the next field
    for raw in text.splitlines():
        if not raw.strip():
            pending.clear()
            continue
        if raw.lstrip().startswith("#"):
            pending.append(raw.lstrip()[1:].strip())
            continue
        m_top = _TOP.match(raw)
        if m_top and not raw.startswith((" ", "\t")):
            section = m_top.group("key")
            pending.clear()
            continue
        m_field = _FIELD.match(raw)
        if m_field:
            key = m_field.group("key")
            val, comment = _split_value_comment(m_field.group("val_and_comment"))
            fields.append(
                ExtractedField(
                    section=section,
                    key=key,
                    raw_default=val,
                    eol_comment=comment,
                    preceding=_preceding_for(pending, key),
                )
            )
            pending.clear()
    return fields
