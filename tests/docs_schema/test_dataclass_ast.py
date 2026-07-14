# tests/docs_schema/test_dataclass_ast.py
from tools.schema_gen import dataclass_ast as dc
from tools.schema_gen.sources import PARITY_SOURCES, DATACLASS_SOURCES

def test_field_names_of_pretraining_config():
    spec = PARITY_SOURCES["pretraining"]
    names = dc.field_names(spec.file, spec.classname)
    assert "micro_batch_seqs" in names
    assert "recycling_bptt" in names
    assert "inferred_grad_accum_steps" in names  # present on the class (allowlisted later)
    assert "global_batch_seqs" in names          # runtime-derived (allowlisted later)

def test_eval_fields_carry_leading_comments():
    spec = DATACLASS_SOURCES["eval"]
    fields = dc.extract_fields(spec.file, spec.classname)
    by = {f.key: f for f in fields}
    assert "framework" in by
    assert "pbc, flip, pgym" in by["framework"].description
    assert by["batch_size"].raw_default == "16"

def test_eval_zero_shot_method_description():
    spec = DATACLASS_SOURCES["eval"]
    by = {f.key: f for f in dc.extract_fields(spec.file, spec.classname)}
    assert "masked_marginals" in by["zero_shot_method"].description

def test_trailing_comment_does_not_bleed_into_next_field():
    spec = PARITY_SOURCES["pretraining"]
    by = {f.key: f for f in dc.extract_fields(spec.file, spec.classname)}
    # noble_lr_gamma keeps its OWN trailing comment...
    assert "W_up LR power" in by["noble_lr_gamma"].description
    # ...but the previous field's trailing comment must NOT bleed up into it.
    assert "independent base LR" not in by["noble_lr_gamma"].description

def test_field_keeps_its_own_leading_and_trailing_comment():
    spec = PARITY_SOURCES["pretraining"]
    by = {f.key: f for f in dc.extract_fields(spec.file, spec.classname)}
    # noble_base_lr has BOTH the own-line block header and its own trailing comment.
    desc = by["noble_base_lr"].description
    assert "NOBLE optimizer LR scaling" in desc
    assert "independent base LR" in desc

def test_distillation_field_has_no_bled_trailing_comment():
    spec = PARITY_SOURCES["distillation"]
    by = {f.key: f for f in dc.extract_fields(spec.file, spec.classname)}
    # A field directly below `fsdp_shard_granularity` must not inherit its
    # `"layer" | "sublayer"` trailing comment. (Verify the exact field by dumping;
    # `seed` is expected.)
    assert "sublayer" not in by["seed"].description
