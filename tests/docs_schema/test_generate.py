# tests/docs_schema/test_generate.py
import json
from pathlib import Path
from tools.schema_gen import generate


def _reject_constant(token):
    # json.loads calls parse_constant ONLY for the bare Infinity/-Infinity/NaN
    # constant tokens — never for those substrings inside a string literal (a
    # legitimate description like "checks params for NaN/Inf ..." is fine). This
    # is exactly the strict-JSON invariant JS/Vite's JSON parser enforces.
    raise AssertionError(f"non-finite JSON constant {token!r} in artifact (JS/Vite would reject it)")


def test_generate_writes_four_surfaces(tmp_path):
    out = generate.run(out_dir=tmp_path)
    for surface in ("data", "pretrain", "distill", "eval"):
        p = tmp_path / f"nanoplm.{surface}.schema.json"
        assert p.exists()
        json.loads(p.read_text())  # valid JSON
    idx = json.loads((tmp_path / "index.json").read_text())
    assert sorted(s["surface"] for s in idx["surfaces"]) == ["data", "distill", "eval", "pretrain"]
    assert out == 5  # 4 schemas + index

def test_artifacts_are_strict_json(tmp_path):
    generate.run(out_dir=tmp_path)
    for surface in ("data", "pretrain", "distill", "eval"):
        text = (tmp_path / f"nanoplm.{surface}.schema.json").read_text()
        # Raises via _reject_constant if a bare Infinity/-Infinity/NaN slipped in.
        json.loads(text, parse_constant=_reject_constant)
    pre = json.loads((tmp_path / "nanoplm.pretrain.schema.json").read_text())
    mgn = pre["properties"]["pretraining"]["properties"]["max_grad_norm"]
    assert mgn["default"] == ".inf"   # non-finite default serialized as a JSON-safe sentinel
