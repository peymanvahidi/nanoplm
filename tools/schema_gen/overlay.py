# tools/schema_gen/overlay.py
from __future__ import annotations
import copy
from functools import lru_cache
from pathlib import Path
import yaml

_PATH = Path(__file__).parent / "overlay.yaml"

# Reserved top-level key in overlay.yaml. Everything else in the file is a flat
# "<section>.<field>" annotation; this one holds whole *extra field* declarations
# for config fields the get-yaml templates ship commented out (see overlay.yaml).
# It deliberately contains no "." so it can never collide with the flat namespace.
EXTRA_FIELDS_KEY = "x-extra-fields"

@lru_cache(maxsize=1)
def load() -> dict:
    return yaml.safe_load(_PATH.read_text(encoding="utf-8")) or {}

def for_field(o: dict, key: str) -> dict:
    # The reserved key is not a field annotation — never let it be read as one.
    if key == EXTRA_FIELDS_KEY:
        return {}
    # Deep-copy so callers (e.g. emit.py merging `enum`/`backend` lists into a
    # schema fragment) can never mutate the lru_cache'd overlay in place.
    return copy.deepcopy(o.get(key, {}) or {})

def backends_for(o: dict, key: str) -> list[str]:
    return list(for_field(o, key).get("backend", []))

def extra_fields(o: dict, surface: str) -> dict:
    """Extra-field declarations for one surface: {section: {field: spec}}.

    Deep-copied for the same reason as ``for_field``: emit.py folds these
    straight into schema fragments.
    """
    return copy.deepcopy((o.get(EXTRA_FIELDS_KEY) or {}).get(surface, {}) or {})
