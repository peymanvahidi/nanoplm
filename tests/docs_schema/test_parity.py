# tests/docs_schema/test_parity.py
import pytest
from tools.schema_gen import emit, dataclass_ast as dc
from tools.schema_gen.sources import PARITY_SOURCES, PARITY_INTERNAL_ALLOWLIST

def _documented(surface, section):
    s = emit.build_surface(surface)
    props = s["properties"]
    node = props[section]["properties"] if section else props
    return set(node)

CASES = [
    # (surface, section, parity_key)
    ("pretrain", "pretraining", "pretraining"),
    ("pretrain", "resume", "resume"),
    ("distill", "distillation", "distillation"),
    ("eval", None, "eval"),
]

@pytest.mark.parametrize("surface,section,pkey", CASES)
def test_every_real_field_is_documented(surface, section, pkey):
    spec = PARITY_SOURCES[pkey]
    real = set(dc.field_names(spec.file, spec.classname)) - PARITY_INTERNAL_ALLOWLIST[pkey]
    documented = _documented(surface, section)
    missing = real - documented
    assert not missing, f"{pkey}: undocumented config fields: {sorted(missing)}"

@pytest.mark.parametrize("surface,section,pkey", CASES)
def test_no_invented_fields(surface, section, pkey):
    spec = PARITY_SOURCES[pkey]
    real = set(dc.field_names(spec.file, spec.classname))
    documented = _documented(surface, section)
    invented = documented - real
    assert not invented, f"{pkey}: documented fields not on the dataclass: {sorted(invented)}"


# ---- data surface -----------------------------------------------------------
# data.yaml has no dataclass; its field rules are module-level tuples in
# data_config.py / cluster_config.py. Read them statically (no nanoplm import).

def _module_tuple(path, name):
    import ast
    from pathlib import Path
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found in {path}")


def test_data_surface_matches_data_config():
    from tools.schema_gen.sources import REPO_SRC
    prep = REPO_SRC / "data" / "preparation"
    props = emit.build_surface("data")["properties"]
    spec = props["sources"]["additionalProperties"]["properties"]

    assert set(props) == set(_module_tuple(prep / "data_config.py", "TOP_KEYS"))
    assert list(props["data_params"]["properties"]) == list(
        _module_tuple(prep / "data_config.py", "PARAM_FIELDS"))
    assert set(spec) == set(_module_tuple(prep / "data_config.py", "SOURCE_KEYS"))
    assert set(spec["cluster_subsampling"]["properties"]) == set(
        _module_tuple(prep / "data_config.py", "CLUSTER_KEYS"))
    # clusters also takes a .tsv path, so it has no enum: the description names every
    # word value
    words = _module_tuple(prep / "cluster_config.py", "CLUSTERS")
    text = spec["cluster_subsampling"]["properties"]["clusters"]["description"]
    assert "enum" not in spec["cluster_subsampling"]["properties"]["clusters"]
    assert all(w in text for w in words), (words, text)
    for suffix in _module_tuple(prep / "cluster_config.py", "TSV_SUFFIXES"):
        assert suffix in text
