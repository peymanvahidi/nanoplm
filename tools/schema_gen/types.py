# tools/schema_gen/types.py
"""Infer a JSON-Schema fragment ({"type":..., "default":...}) from a raw YAML scalar literal."""
from __future__ import annotations

import math


def infer(raw: str) -> dict:
    s = raw.strip()
    low = s.lower()

    if low in ("null", "~", ""):
        return {"type": "null", "default": None}
    if low in ("true", "false"):
        return {"type": "boolean", "default": low == "true"}
    if low in (".inf", "+.inf"):
        return {"type": "number", "default": math.inf}
    if low == "-.inf":
        return {"type": "number", "default": -math.inf}
    if s in ("{}",):
        return {"type": "object", "default": {}}
    if s in ("[]",):
        return {"type": "array", "default": []}

    # quoted string
    if (s[0], s[-1]) in (('"', '"'), ("'", "'")) and len(s) >= 2:
        return {"type": "string", "default": s[1:-1]}

    # int
    try:
        return {"type": "integer", "default": int(s)}
    except ValueError:
        pass
    # float (covers 1e-4, 0.1, 3e-4, 5e-5)
    try:
        return {"type": "number", "default": float(s)}
    except ValueError:
        pass

    # bare (unquoted) string e.g. `all`, `auto`
    return {"type": "string", "default": s}
