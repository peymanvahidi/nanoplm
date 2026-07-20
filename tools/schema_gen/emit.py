# tools/schema_gen/emit.py
"""Assemble a JSON Schema (draft 2020-12) per surface, folding template/dataclass fields,
type inference, overlay annotations, visibility gates, and compat rules."""
from __future__ import annotations

import ast
import copy

from tools.schema_gen import overlay as _overlay
from tools.schema_gen import templates as _tpl
from tools.schema_gen import types as _types
from tools.schema_gen import dataclass_ast as _dc
from tools.schema_gen import gates as _gates
from tools.schema_gen.sources import DATACLASS_SOURCES, TEMPLATE_SOURCES

_DRAFT = "https://json-schema.org/draft/2020-12/schema"


# Annotation keys an overlay entry (or an x-extra-fields declaration) may carry.
_ANNOTATION_KEYS = ("enum", "minimum", "maximum", "backend", "advanced", "group")


def _apply(frag: dict, key: str, description: str, ov: dict, vis: dict) -> dict:
    """Fold one annotation dict, the description, and the visibility gate onto a fragment."""
    if "enum" in ov:
        frag["enum"] = ov["enum"]
    for bound in ("minimum", "maximum"):
        if bound in ov:
            frag[bound] = ov[bound]
    if description:
        frag["description"] = description
    if ov.get("backend"):
        frag["x-nanoplm-backend"] = ov["backend"]
    if ov.get("advanced"):
        frag["x-nanoplm-advanced"] = True
    # An overlay/x-extra-fields `group` wins over the banner the parser found,
    # which is how a commented-out field is filed under the block that documents
    # it (the parser cannot see it, so it has no banner of its own).
    if ov.get("group"):
        frag["x-nanoplm-group"] = ov["group"]
    if key in vis:
        frag["x-nanoplm-visible-when"] = vis[key]
    return frag


def _fold(frag: dict, section: str, key: str, description: str, o: dict, vis: dict) -> dict:
    """Fold the flat `<section>.<field>` overlay entry onto a fragment."""
    return _apply(frag, key, description, _overlay.for_field(o, f"{section}.{key}"), vis)


def _field_schema(section: str, key: str, raw_default: str, description: str,
                  o: dict, vis: dict, group: str = "") -> dict:
    # Template (YAML-literal) surfaces: infer type + default from the literal text.
    frag = _types.infer(raw_default)
    if group:
        frag["x-nanoplm-group"] = group
    return _fold(frag, section, key, description, o, vis)


def _extra_field_schema(section: str, key: str, spec: dict, o: dict, vis: dict) -> dict:
    """Build a fragment for an `x-extra-fields` declaration (a config field the
    get-yaml template ships commented out, so the parser never sees it).

    Type/default/description come from the declaration itself; annotations
    declared inline win over any flat `<section>.<field>` overlay entry.
    An absent `default` key means "no default" (required); `default: null`
    means an explicit JSON-null default, matching `Optional[...] = None`.
    """
    frag: dict = {"type": spec.get("type", "string")}
    if "default" in spec:
        frag["default"] = spec["default"]
    ov = _overlay.for_field(o, f"{section}.{key}")
    ov.update({k: v for k, v in spec.items() if k in _ANNOTATION_KEYS})
    return _apply(frag, key, spec.get("description", ""), ov, vis)


# ---- dataclass surface (eval) -------------------------------------------------
# The eval surface comes from a Python dataclass, so defaults are Python source
# (`None`, `field(default_factory=lambda: [...])`, literals) and the type is best
# taken from the annotation rather than guessed from the default text.

_ANNOTATION_JSON_TYPE = (  # first prefix match wins
    ("list", "array"), ("dict", "object"), ("bool", "boolean"),
    ("int", "integer"), ("float", "number"), ("str", "string"), ("path", "string"),
)

_EMPTY_FACTORY = {"list": [], "dict": {}, "tuple": [], "set": [],
                  "str": "", "int": 0, "float": 0.0, "bool": False}


def _json_type_from_annotation(annotation: str) -> str:
    a = annotation.replace(" ", "")
    if a.startswith("Optional[") and a.endswith("]"):
        return _json_type_from_annotation(a[len("Optional["):-1])
    if a.startswith("Union[") and a.endswith("]"):
        return _json_type_from_annotation(a[len("Union["):-1].split(",")[0])
    low = a.lower()
    for needle, jtype in _ANNOTATION_JSON_TYPE:
        if low.startswith(needle):
            return jtype
    return "string"


def _literal_value(node: ast.AST):
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return False, None


def _factory_value(node: ast.AST):
    if isinstance(node, ast.Lambda):
        return _literal_value(node.body)
    if isinstance(node, ast.Name) and node.id in _EMPTY_FACTORY:
        return True, _EMPTY_FACTORY[node.id]
    return False, None


def _resolve_default(raw: str):
    """Resolve a dataclass field's default *value* from its source text.
    Returns (has_value, value); has_value=False means no usable literal."""
    try:
        node = ast.parse(raw, mode="eval").body
    except SyntaxError:
        return False, None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "field":
        for kw in node.keywords:
            if kw.arg == "default":
                return _literal_value(kw.value)
            if kw.arg == "default_factory":
                return _factory_value(kw.value)
        return False, None
    return _literal_value(node)


def _dataclass_field_schema(field, o: dict, vis: dict):
    """Return (fragment, is_required) for one dataclass field. Type comes from the
    annotation; the default is the resolved real value (None, factory, or literal).
    A field with no default is required and carries no `default` key."""
    frag: dict = {"type": _json_type_from_annotation(field.annotation)}
    raw = field.raw_default.strip()
    required = raw == ""
    if not required:
        has_value, value = _resolve_default(raw)
        frag["default"] = value if has_value else raw
    _fold(frag, "eval", field.key, field.description, o, vis)
    return frag, required


def _regroup(properties: dict) -> dict:
    """Reorder one section's fields so each group's fields stay contiguous.

    The template parser emits fields in template order, which is already grouped,
    but `x-extra-fields` are folded in afterwards and land at the end of the
    section, away from the block that documents them. Bucketing by group (groups
    in first-appearance order, fields in template order within a group) puts them
    back. A section with no banners has one bucket and comes out unchanged.
    """
    order: list[str] = []
    for frag in properties.values():
        g = frag.get("x-nanoplm-group", "")
        if g not in order:
            order.append(g)
    buckets: dict[str, list[tuple[str, dict]]] = {g: [] for g in order}
    for key, frag in properties.items():
        buckets[frag.get("x-nanoplm-group", "")].append((key, frag))
    return {key: frag for g in order for key, frag in buckets[g]}


def build_surface(surface: str) -> dict:
    o = _overlay.load()
    vis = _gates.visibility_gates()
    props: dict = {}
    required: list[str] = []

    if surface in TEMPLATE_SOURCES:
        text = _tpl.load_template_text(surface)
        sections: dict[str, dict] = {}
        parsed, group_notes = _tpl.parse_template(text)
        for f in parsed:
            sec = sections.setdefault(f.section, {"type": "object", "properties": {}})
            sec["properties"][f.key] = _field_schema(
                f.section, f.key, f.raw_default, f.description, o, vis, f.group
            )
        # Fold in the fields the template ships commented out (see overlay.yaml's
        # `x-extra-fields`). They are user-settable and must be documented.
        for sec_name, extras in _overlay.extra_fields(o, surface).items():
            sec = sections.setdefault(sec_name, {"type": "object", "properties": {}})
            for key, spec in (extras or {}).items():
                sec["properties"][key] = _extra_field_schema(sec_name, key, spec or {}, o, vis)
        for sec_name, sec in sections.items():
            sec["properties"] = _regroup(sec["properties"])
            # The banner-intro prose for each block that has any, so the docs can
            # render it under the heading instead of on the block's first field.
            notes = {g: note for (s, g), note in group_notes.items() if s == sec_name}
            if notes:
                sec["x-nanoplm-group-notes"] = notes
        props = sections
    elif surface in DATACLASS_SOURCES:
        spec = DATACLASS_SOURCES[surface]
        for f in _dc.extract_fields(spec.file, spec.classname):
            frag, is_required = _dataclass_field_schema(f, o, vis)
            props[f.key] = frag
            if is_required:
                required.append(f.key)
    else:
        raise ValueError(f"unknown surface {surface}")

    schema = {
        "$schema": _DRAFT,
        "x-nanoplm-surface": surface,
        "type": "object",
        "properties": props,
    }
    if required:
        schema["required"] = required
    if surface in ("pretrain", "distill"):
        schema["x-nanoplm-compat"] = copy.deepcopy(_gates.compat_rules())
    return schema
