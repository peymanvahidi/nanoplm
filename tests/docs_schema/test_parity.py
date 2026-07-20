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
