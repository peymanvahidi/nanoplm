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
_ANNOTATION_KEYS = ("type", "items", "enum", "minimum", "exclusiveMinimum", "maximum", "backend",
                    "advanced", "group", "default")


def _overlay_key(section: str, key: str) -> str:
    """The flat overlay key of a field: `<section>.<field>`, or the bare field
    name for a top-level scalar (data.yaml's `output_dir`)."""
    return f"{section}.{key}" if section else key


def _apply(frag: dict, key: str, description: str, ov: dict, vis: dict) -> dict:
    """Fold one annotation dict, the description, and the visibility gate onto a fragment."""
    # An overlay `type` replaces the one inferred from the default literal. It is
    # how a field whose template default is `null` says what else it accepts,
    # e.g. `type: [integer, "null"]`.
    if "type" in ov:
        frag["type"] = ov["type"]
    if "items" in ov:
        frag["items"] = ov["items"]
    if "enum" in ov:
        frag["enum"] = ov["enum"]
    for bound in ("minimum", "exclusiveMinimum", "maximum"):
        if bound in ov:
            frag[bound] = ov[bound]
    # An overlay `default` / `description` replaces what the template gave, for a
    # field whose template value is only an example or whose comment the parser
    # cannot attach (it sits above the section header, say).
    if "default" in ov:
        frag["default"] = ov["default"]
    if "description" in ov:
        description = ov["description"]
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
    return _apply(frag, key, description, _overlay.for_field(o, _overlay_key(section, key)), vis)


def _field_schema(section: str, key: str, raw_default: str, description: str,
                  o: dict, vis: dict, group: str = "") -> dict:
    # Template (YAML-literal) surfaces: infer type + default from the literal text.
    frag = _types.infer(raw_default)
    if group:
        frag["x-nanoplm-group"] = group
    frag = _fold(frag, section, key, description, o, vis)
    # A `null` default says nothing about the field's real type, and `"type":
    # "null"` would tell the docs the field can only ever be null.
    types = frag["type"] if isinstance(frag["type"], list) else [frag["type"]]
    if types == ["null"] or (frag["default"] is None and "null" not in types):
        raise ValueError(
            f"{section}.{key} defaults to null: give it a nullable type in "
            f'overlay.yaml, e.g. `type: [integer, "null"]` (got {frag["type"]!r})'
        )
    return frag


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
    # A nested block (a source's `cluster_subsampling`) declares its own fields.
    if "properties" in spec:
        frag["properties"] = {
            sub: _extra_field_schema(f"{section}.{key}", sub, sub_spec or {}, o, vis)
            for sub, sub_spec in spec["properties"].items()
        }
        required = [sub for sub, f in frag["properties"].items() if "default" not in f]
        if required:
            frag["required"] = required
    ov = _overlay.for_field(o, _overlay_key(section, key))
    ov.update({k: v for k, v in spec.items() if k in _ANNOTATION_KEYS})
    return _apply(frag, key, spec.get("description", ""), ov, vis)


def _template_description(f, ov: dict) -> str:
    """A field's description, minus a heading the overlay turned into its group.

    A plain `# Rarely changed` line above a field reads to the parser as that
    field's description; once overlay.yaml files the block under that title, the
    heading is the group's label and must not stay on the field.
    """
    group = ov.get("group")
    if group and f.preceding and f.preceding[0].strip() == group:
        parts = [*f.preceding[1:], f.eol_comment]
        return " ".join(p.strip() for p in parts if p.strip())
    return f.description


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
        # Top-level scalars and sections, in template order.
        sections: dict[str, dict] = {}
        parsed, group_notes = _tpl.parse_template(text)
        for f in parsed:
            ov = _overlay.for_field(o, _overlay_key(f.section, f.key))
            frag = _field_schema(f.section, f.key, f.raw_default, _template_description(f, ov),
                                 o, vis, f.group)
            if not f.section:
                # Only the top-level scalars overlay.yaml names: pretrain.yaml's
                # `pure_torch: true` has one legal value and is not a setting.
                if ov.get("top_level"):
                    sections[f.key] = frag
                continue
            if f.entry:
                # Inside a name -> spec mapping the template's value is an example
                # (`fasta: swissprot`), not a default, unless the overlay says so.
                if "default" not in ov:
                    frag["examples"] = [frag.pop("default")]
                if not _overlay.for_field(o, f.section).get("map"):
                    raise ValueError(
                        f"{f.section}.{f.entry}.{f.key} is nested under {f.entry!r}: mark "
                        f"`{f.section}: {{map: true}}` in overlay.yaml if {f.section} maps "
                        "names to specs"
                    )
            sec = sections.setdefault(f.section, {"type": "object", "properties": {}})
            sec["properties"][f.key] = frag
        # Fold in the fields the template ships commented out (see overlay.yaml's
        # `x-extra-fields`). They are user-settable and must be documented.
        for sec_name, extras in _overlay.extra_fields(o, surface).items():
            sec = sections.setdefault(sec_name, {"type": "object", "properties": {}})
            for key, spec in (extras or {}).items():
                sec["properties"][key] = _extra_field_schema(sec_name, key, spec or {}, o, vis)
        for sec_name, sec in sections.items():
            if "properties" not in sec:
                continue  # a top-level scalar
            sec["properties"] = _regroup(sec["properties"])
            # The banner-intro prose for each block that has any, so the docs can
            # render it under the heading instead of on the block's first field.
            notes = {g: note for (s, g), note in group_notes.items() if s == sec_name}
            if notes:
                sec["x-nanoplm-group-notes"] = notes
            sec_ov = _overlay.for_field(o, sec_name)
            if sec_ov.get("map"):
                # `sources: {<name>: <spec>}`: the fields describe every entry.
                spec = {"type": "object", **{k: v for k, v in sec.items() if k != "type"}}
                sections[sec_name] = sec = {"type": "object", "additionalProperties": spec}
            if sec_ov.get("description"):
                sec["description"] = sec_ov["description"]
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
