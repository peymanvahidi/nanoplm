# tools/schema_gen/generate.py
"""Write the canonical per-surface JSON Schema files for the docs site."""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.schema_gen import emit
from tools.schema_gen.sources import REPO_ROOT

SURFACES = ("data", "pretrain", "distill", "eval")
DEFAULT_OUT = REPO_ROOT / "website" / "src" / "generated"


def _json_safe(obj):
    """Recursively coerce a schema into strictly-JSON-serializable form.

    JSON has no Infinity/NaN, and JS/Vite reject the non-standard ``Infinity``
    token Python's json would otherwise emit. Non-finite float defaults (e.g.
    ``max_grad_norm: .inf``) become the YAML-style string sentinels users
    actually type; NaN becomes null; tuples/sets become lists.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj):
            return ".inf" if obj > 0 else "-.inf"
        if math.isnan(obj):
            return None
    return obj


def _dump(obj) -> str:
    # allow_nan=False makes any non-finite that slips past _json_safe fail loudly
    # rather than silently emit invalid JSON.
    return json.dumps(_json_safe(obj), indent=2, allow_nan=False) + "\n"


def run(out_dir: Path = DEFAULT_OUT) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    index = {"surfaces": []}
    for surface in SURFACES:
        schema = emit.build_surface(surface)
        path = out_dir / f"nanoplm.{surface}.schema.json"
        path.write_text(_dump(schema), encoding="utf-8")
        written += 1
        index["surfaces"].append({"surface": surface, "file": path.name})
    (out_dir / "index.json").write_text(_dump(index), encoding="utf-8")
    return written + 1


if __name__ == "__main__":
    n = run()
    print(f"wrote {n} files to {DEFAULT_OUT}")
