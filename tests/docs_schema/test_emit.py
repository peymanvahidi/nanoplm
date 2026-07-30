# tests/docs_schema/test_emit.py
from tools.schema_gen import emit

def test_pretrain_schema_shape():
    s = emit.build_surface("pretrain")
    assert s["$schema"].startswith("http")
    assert s["x-nanoplm-surface"] == "pretrain"
    props = s["properties"]
    assert {"model", "pretraining", "resume"} <= set(props)
    mb = props["pretraining"]["properties"]["micro_batch_seqs"]
    assert mb["type"] == "integer" and mb["default"] == 128
    opt = props["pretraining"]["properties"]["optimizer"]
    assert opt["enum"] == ["adamw", "stable_adamw", "muon", "normuon", "nordion2"]
    sched = props["pretraining"]["properties"]["lr_schedule"]
    assert sched["enum"] == ["wsd", "warmup_stable", "cosine", "linear"]
    assert sched["default"] == "wsd"
    # visibility gate annotation present on a recycling field
    rb = props["pretraining"]["properties"]["recycling_bptt"]
    assert rb["x-nanoplm-visible-when"]["field"] == "recycling_loop_count"
    # backend tag folded onto a model field
    canon = props["model"]["properties"]["use_canon_layers"]
    assert "pure_torch" in canon["x-nanoplm-backend"]
    # compat rules attached at surface level
    assert any(r["flag"] == "use_moe" for r in s["x-nanoplm-compat"]["model_reject"])


def test_extra_fields_are_folded_into_their_section():
    # The duration budgets the get-yaml template ships COMMENTED OUT (they are
    # mutually exclusive with their *_tokens counterparts) are still real,
    # user-settable fields — overlay.yaml's `x-extra-fields` documents them.
    s = emit.build_surface("pretrain")
    pre = s["properties"]["pretraining"]["properties"]

    tt = pre["train_tokens"]
    assert tt["type"] == "integer"
    assert tt["default"] is None          # Optional[int] = None on the dataclass
    assert tt["x-nanoplm-advanced"] is True
    assert "max_steps" in tt["description"]      # names its mutually-exclusive twin
    assert "warmup_stable" in tt["description"]  # names the schedules it applies to

    for key in ("stable_steps", "decay_steps", "max_steps"):
        assert pre[key]["type"] == "integer"
        assert pre[key]["default"] is None
        assert pre[key]["x-nanoplm-advanced"] is True

    resume = s["properties"]["resume"]["properties"]
    assert resume["decay_steps"]["type"] == "integer"
    assert resume["decay_steps"]["x-nanoplm-advanced"] is True

    # ...and template-parsed fields in the same sections are untouched.
    assert pre["stable_tokens"]["default"] == 9000000
    assert resume["decay_tokens"]["default"] == 1000000


def test_extra_fields_cover_distill_adam_betas():
    dist = emit.build_surface("distill")["properties"]["distillation"]["properties"]
    assert dist["adam_beta1"]["default"] == 0.9
    assert dist["adam_beta2"]["default"] == 0.999
    assert dist["adam_epsilon"]["type"] == "number"
    assert dist["adam_beta1"]["x-nanoplm-advanced"] is True

def test_eval_schema_built_from_dataclass():
    s = emit.build_surface("eval")
    props = s["properties"]
    assert "framework" in props
    assert "pbc, flip, pgym" in props["framework"]["description"]


def test_eval_required_field_has_no_fabricated_default():
    s = emit.build_surface("eval")
    cd = s["properties"]["checkpoint_dir"]
    assert cd["type"] == "string"
    assert "default" not in cd            # required field: no fabricated null default
    assert "checkpoint_dir" in s.get("required", [])

def test_eval_factory_default_is_resolved():
    props = emit.build_surface("eval")["properties"]
    fr = props["framework"]
    assert fr["type"] == "array"
    assert fr["default"] == ["pbc"]       # the resolved value, not the lambda source string

def test_eval_none_default_is_json_null():
    props = emit.build_surface("eval")["properties"]
    assert props["device"]["default"] is None
    assert props["device"]["type"] == "string"

def test_eval_scalar_defaults_and_enum():
    props = emit.build_surface("eval")["properties"]
    assert props["batch_size"]["type"] == "integer" and props["batch_size"]["default"] == 16
    zsm = props["zero_shot_method"]
    assert zsm["default"] == "masked_marginals"
    assert zsm["enum"] == ["masked_marginals", "wt_marginals", "pseudoperplexity"]
    assert props["force_download"]["type"] == "boolean" and props["force_download"]["default"] is False

def test_data_and_distill_surfaces_build():
    for surf in ("data", "distill"):
        s = emit.build_surface(surf)
        assert s["x-nanoplm-surface"] == surf and s["properties"]
    assert "x-nanoplm-compat" in emit.build_surface("distill")
    assert "x-nanoplm-compat" not in emit.build_surface("data")
