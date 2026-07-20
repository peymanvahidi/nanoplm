# tests/docs_schema/test_types.py
from tools.schema_gen.types import infer

def test_infer_int():
    assert infer("64") == {"type": "integer", "default": 64}

def test_infer_float():
    assert infer("1e-4") == {"type": "number", "default": 1e-4}

def test_infer_bool():
    assert infer("true") == {"type": "boolean", "default": True}

def test_infer_string_keeps_unquoted_value():
    assert infer('"adamw"') == {"type": "string", "default": "adamw"}

def test_infer_null_is_nullable_with_no_default():
    out = infer("null")
    assert out["default"] is None
    assert "null" in out["type"] if isinstance(out["type"], list) else out["type"] == "null"

def test_infer_inf_float():
    out = infer(".inf")
    assert out["type"] == "number"
    assert out["default"] == float("inf")

def test_infer_empty_mapping():
    assert infer("{}") == {"type": "object", "default": {}}

def test_infer_empty_sequence():
    assert infer("[]") == {"type": "array", "default": []}

def test_infer_bare_word_is_string():
    # The fallthrough branch — exercised in the real templates by `noble_targets: all`.
    assert infer("auto") == {"type": "string", "default": "auto"}
    assert infer("all") == {"type": "string", "default": "all"}
