# tools/schema_gen/dataclass_ast.py
"""Extract @dataclass fields (name, annotation, default, leading-comment description)
statically via ast + tokenize. No import of the target module (avoids torch deps)."""
from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataField:
    key: str
    annotation: str
    raw_default: str
    description: str


def _comments_by_line(source: str) -> tuple[dict[int, str], dict[int, str]]:
    """Map line numbers to comment text, split into own-line vs trailing comments.

    An own-line comment is the first non-trivial token on its physical row (a
    documentation comment above a field); a trailing comment shares its row with
    code (the field's own end-of-line comment). Keeping them apart prevents a
    field's trailing comment from bleeding upward into the next field's
    leading-comment block.
    """
    own_line: dict[int, str] = {}
    trailing: dict[int, str] = {}
    last_code_row = -1
    skip = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            row = tok.start[0]
            text = tok.string.lstrip("#").strip()
            if last_code_row == row:
                trailing[row] = text
            else:
                own_line[row] = text
        elif tok.type not in skip:
            last_code_row = tok.start[0]
    return own_line, trailing


def _leading_block(comments: dict[int, str], field_line: int) -> list[str]:
    """Collect contiguous own-line comments immediately above field_line."""
    block: list[str] = []
    line = field_line - 1
    while line in comments:
        block.append(comments[line])
        line -= 1
    block.reverse()
    return block


def _class_node(source: str, classname: str) -> ast.ClassDef:
    tree = ast.parse(source)
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == classname:
            return n
    raise ValueError(f"class {classname} not found")


def extract_fields(file: str, classname: str) -> list[DataField]:
    source = Path(file).read_text(encoding="utf-8")
    own_line, trailing = _comments_by_line(source)
    cls = _class_node(source, classname)
    fields: list[DataField] = []
    for stmt in cls.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        key = stmt.target.id
        annotation = ast.unparse(stmt.annotation)
        default = ast.unparse(stmt.value) if stmt.value is not None else ""
        eol = trailing.get(stmt.lineno, "")
        block = _leading_block(own_line, stmt.lineno)
        desc = " ".join([*block, eol]).strip()
        fields.append(DataField(key=key, annotation=annotation, raw_default=default, description=desc))
    return fields


def field_names(file: str, classname: str) -> list[str]:
    return [f.key for f in extract_fields(file, classname)]
