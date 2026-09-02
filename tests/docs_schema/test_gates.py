# tests/docs_schema/test_gates.py
from tools.schema_gen import gates

def test_reject_rules_loaded_from_matrix():
    rules = gates.compat_rules()
    rejected = {r["flag"] for r in rules["model_reject"]}
    assert {"use_x0_lambdas", "use_resid_lambdas", "use_mhc_lite"} <= rejected
    # each carries the human reason for the validator UI
    lam = next(r for r in rules["model_reject"] if r["flag"] == "use_x0_lambdas")
    assert lam["reason"]
    # MoE is allowed under recycling (per-layer aux/count state accumulates
    # across iterations) but still warns about the changed semantics.
    warned = {r["flag"] for r in rules["model_warn"]}
    assert "use_moe" in warned
    moe = next(r for r in rules["model_warn"] if r["flag"] == "use_moe")
    assert "MoE" in moe["message"]

def test_visibility_gate_for_recycling_fields():
    g = gates.visibility_gates()
    assert g["recycling_bptt"] == {"field": "recycling_loop_count", "op": ">", "value": 1}
    assert g["canon_layers_mode"] == {"field": "use_canon_layers", "op": "==", "value": True}

def test_trigger_is_loop_count_gt_1():
    assert gates.compat_rules()["trigger"] == "recycling_loop_count > 1"

def test_pipeline_reject_keeps_inverting_condition():
    # The condition string disambiguates the inverted polarity for the compat
    # matrix: pure_torch is rejected only when False (i.e. it is REQUIRED).
    rules = gates.compat_rules()
    pt = next(r for r in rules["pipeline_reject"] if r["flag"] == "pure_torch")
    assert pt["condition"] == "pure_torch is False"

def test_allow_lists_present_for_matrix():
    # The explicitly-compatible ("green") flags are first-class compat-matrix content.
    rules = gates.compat_rules()
    assert "use_qk_norm" in rules["model_allow"]
    assert "bf16" in rules["pipeline_allow"]
