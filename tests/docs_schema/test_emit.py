# tests/docs_schema/test_emit.py
import pytest

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
    assert any(r["flag"] == "use_moe" for r in s["x-nanoplm-compat"]["model_warn"])


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
    assert "pgym, pbc_zeroshot_contact" in props["framework"]["description"]


def test_eval_required_field_has_no_fabricated_default():
    s = emit.build_surface("eval")
    cd = s["properties"]["checkpoint_dir"]
    assert cd["type"] == "string"
    assert "default" not in cd            # required field: no fabricated null default
    assert "checkpoint_dir" in s.get("required", [])

def test_eval_factory_default_is_resolved():
    props = emit.build_surface("eval")["properties"]
    fr = props["framework"]
    assert emit._json_type_from_annotation("List[str]") == "array"
    assert fr["type"] == ["array", "string"]  # overlay: a bare string is accepted too
    assert fr["default"] == ["pgym"]      # the resolved value, not the lambda source string

def test_eval_none_default_is_json_null():
    props = emit.build_surface("eval")["properties"]
    assert props["device"]["default"] is None
    assert props["device"]["type"] == "string"

def test_eval_scalar_defaults_and_enum():
    props = emit.build_surface("eval")["properties"]
    assert (
        props["zero_shot_batch_size"]["type"] == "integer"
        and props["zero_shot_batch_size"]["default"] == 60
    )
    assert (
        props["development_mode"]["type"] == "boolean"
        and props["development_mode"]["default"] is True
    )
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


def _leaf_fields(schema: dict, prefix: str = ""):
    for key, frag in schema["properties"].items():
        name = f"{prefix}{key}"
        # A name -> spec map (data's `sources`) describes its entries here.
        frag = frag.get("additionalProperties", frag)
        if frag.get("type") == "object" and "properties" in frag:
            yield from _leaf_fields(frag, f"{name}.")
        else:
            yield name, frag


def test_no_field_is_typed_null_only():
    # `"type": "null"` is what type inference yields for a `null` default, and it
    # tells the docs the field can only ever be null.
    for surface in ("data", "pretrain", "distill", "eval"):
        for name, frag in _leaf_fields(emit.build_surface(surface)):
            assert frag.get("type") not in ("null", ["null"]), f"{surface}:{name}"


def test_null_default_fields_get_their_overlay_type():
    pre = emit.build_surface("pretrain")["properties"]
    bptt = pre["model"]["properties"]["loopie_bptt"]
    assert bptt["type"] == ["integer", "null"]
    assert bptt["default"] is None and bptt["minimum"] == 1
    counts = pre["pretraining"]["properties"]["recycling_pretraining_eval_loop_counts"]
    assert counts["type"] == ["array", "null"]
    assert counts["items"] == {"type": "integer"}
    source = emit.build_surface("data")["properties"]["sources"]["additionalProperties"]
    members = source["properties"]["cluster_subsampling"]["properties"]["members"]
    assert members["type"] == ["string", "null"]
    kv = emit.build_surface("distill")["properties"]["model"]["properties"]["num_kv_heads"]
    assert kv["type"] == ["integer", "null"] and kv["default"] is None


def test_null_default_without_overlay_type_fails_loudly():
    with pytest.raises(ValueError, match="sec.f defaults to null"):
        emit._field_schema("sec", "f", "null", "", {}, {})
    with pytest.raises(ValueError, match="nullable type"):
        emit._field_schema("sec", "f", "null", "", {"sec.f": {"type": "integer"}}, {})
    frag = emit._field_schema("sec", "f", "null", "", {"sec.f": {"type": ["integer", "null"]}}, {})
    assert frag["type"] == ["integer", "null"] and frag["default"] is None


def _source_spec() -> dict:
    return emit.build_surface("data")["properties"]["sources"]["additionalProperties"]


def test_cluster_fractions_exclude_zero():
    block = _source_spec()["properties"]["cluster_subsampling"]["properties"]
    for key in ("identity", "coverage"):
        assert block[key]["exclusiveMinimum"] == 0 and "minimum" not in block[key]
        assert block[key]["maximum"] == 1
    assert block["identity"]["default"] == 0.5 and block["coverage"]["default"] == 0.8
    assert "default" not in block["clusters"] and "enum" not in block["clusters"]
    cluster = _source_spec()["properties"]["cluster_subsampling"]
    assert cluster["required"] == ["clusters"]
    assert block["members"]["default"] is None   # left out: this source's fasta


def test_data_schema_nests_source_specs_under_sources():
    props = emit.build_surface("data")["properties"]
    assert list(props) == ["output_dir", "sources", "data_params"]
    assert props["output_dir"] == {
        "type": "string",
        "default": "output/data/pretrain_data",
        "description": "every file goes here; pretrain.yaml's dataset_dir points here",
    }
    sources = props["sources"]
    # A name -> spec map: no fixed keys, every entry is a source spec.
    assert sources["type"] == "object" and "properties" not in sources
    assert "override" in sources["description"]
    spec = sources["additionalProperties"]["properties"]
    # The template's `fasta: swissprot` is an example, not a default.
    assert "default" not in spec["fasta"] and spec["fasta"]["examples"] == ["swissprot"]
    assert "uniref90" in spec["fasta"]["description"]
    assert spec["weight"]["type"] == "number" and spec["weight"]["default"] == 1.0
    assert "not tokens" in spec["weight"]["description"]
    assert "prepared" not in spec          # removed: a source is built from its fasta
    assert spec["cluster_subsampling"]["type"] == ["boolean", "object"]
    assert spec["cluster_subsampling"]["default"] is False


def test_data_params_rarely_changed_group():
    params = emit.build_surface("data")["properties"]["data_params"]["properties"]
    assert "seqs_num" not in params and params["max_seqs"]["default"] == 20000
    assert "x-nanoplm-group" not in params["max_seq_len"]
    assert params["shuffle"]["x-nanoplm-group"] == "Rarely changed"
    # The heading labels the block; shuffle's description is its own comment.
    assert params["shuffle"]["description"] == "shuffle before the train / val split"


def test_nested_entry_outside_a_map_section_fails_loudly(monkeypatch):
    monkeypatch.setattr(emit._tpl, "load_template_text", lambda s: "sec:\n  a:\n    f: 1\n")
    with pytest.raises(ValueError, match="map: true"):
        emit.build_surface("data")


def test_only_marked_top_level_scalars_are_documented():
    """data.yaml's output_dir is a field; pretrain.yaml's `pure_torch: true` is not."""
    assert "output_dir" in emit.build_surface("data")["properties"]
    assert "pure_torch" not in emit.build_surface("pretrain")["properties"]


def test_eval_sources_is_a_name_or_a_list_of_names():
    """pretrain.yaml's eval_sources is "all" or a list of source names."""
    es = emit.build_surface("pretrain")["properties"]["pretraining"]["properties"]["eval_sources"]
    assert es["type"] == ["string", "array"] and es["items"] == {"type": "string"}
    assert es["default"] == "all"
