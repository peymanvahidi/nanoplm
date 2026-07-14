# tests/docs_schema/test_templates.py
from tools.schema_gen import templates

def test_extract_pretrain_template_text():
    text = templates.load_template_text("pretrain")
    assert "model:" in text and "pretraining:" in text and "resume:" in text
    assert "micro_batch_seqs: 64" in text

def test_parse_fields_captures_section_key_default_and_comments():
    text = templates.load_template_text("pretrain")
    fields = templates.parse_fields(text)
    by_key = {(f.section, f.key): f for f in fields}

    # bare int default, no end-of-line comment
    mb = by_key[("pretraining", "micro_batch_seqs")]
    assert mb.raw_default == "64"
    assert mb.eol_comment == ""

    # int default WITH an end-of-line comment
    gbt = by_key[("pretraining", "global_batch_tokens")]
    assert gbt.raw_default == "256000"

    st = by_key[("pretraining", "stable_tokens")]
    assert st.raw_default == "9000000"
    assert "pre-decay checkpoint" in st.eol_comment

    # quoted string default + its enum-listing end-of-line comment
    opt = by_key[("pretraining", "optimizer")]
    assert opt.raw_default == '"adamw"'
    assert "adamw | stable_adamw | muon | normuon | muonh | aurora" in opt.description

    # preceding full-line comment block becomes part of the description
    canon = by_key[("model", "use_canon_layers")]
    assert "Canon" in canon.description

def test_data_template_has_data_dirs_and_data_params_sections():
    fields = templates.parse_fields(templates.load_template_text("data"))
    sections = {f.section for f in fields}
    assert {"data_dirs", "data_params"} <= sections

def test_commented_out_field_not_attached_as_description():
    by_key = {(f.section, f.key): f for f in templates.parse_fields(templates.load_template_text("pretrain"))}
    # `# num_kv_heads: 8 ...` sits directly above `vocab_size:` but is a
    # commented-out *other* field, so it must not leak into vocab_size's docs.
    assert "num_kv_heads" not in by_key[("model", "vocab_size")].description
    assert by_key[("model", "vocab_size")].description == ""

def test_commented_out_step_override_is_not_attached_as_description():
    by_key = {(f.section, f.key): f for f in templates.parse_fields(templates.load_template_text("pretrain"))}
    # `# train_tokens: 10000000 ...` and the value-less `# stable_steps:` /
    # `# decay_steps:` / `# max_steps:` placeholders sit directly above
    # `optimizer:`. They are commented-out *other fields* (documented instead via
    # overlay.yaml's x-extra-fields), so they must not leak into optimizer's docs.
    opt = by_key[("pretraining", "optimizer")]
    for leaked in ("train_tokens", "stable_steps", "decay_steps", "max_steps"):
        assert leaked not in opt.description, opt.description
    # The heading that introduces that commented-out block ("Advanced step
    # overrides ...") documents the block, not `optimizer`, so it goes too.
    assert "Advanced step overrides" not in opt.description
    # ...while optimizer's own end-of-line comment survives.
    assert opt.description == "adamw | stable_adamw | muon | normuon | muonh | aurora"

def test_same_key_comment_prose_is_kept():
    by_key = {(f.section, f.key): f for f in templates.parse_fields(templates.load_template_text("pretrain"))}
    # Prose shaped like `Word: ...` (`# NOTE: decay restarts the dataloader ...`)
    # and plain headings (`# Two resume modes:`) are documentation for the field
    # below them and must be retained (no over-correction), together with the
    # field's own end-of-line comment.
    mode = by_key[("resume", "mode")]
    assert "Two resume modes" in mode.description
    assert "decay restarts the dataloader" in mode.description
    assert "continue | decay" in mode.description
    # A `# decay-mode fields (leave null / omit for mode: continue):` heading is
    # prose too, not a commented-out field.
    assert "decay-mode fields" in by_key[("resume", "decay_tokens")].description

def test_distill_descriptions_not_over_dropped():
    by_key = {(f.section, f.key): f for f in templates.parse_fields(templates.load_template_text("distill"))}
    # `# Optimizer: adamw | muon | normuon` is a human heading documenting the
    # optimizer field (case-insensitive key match) — must be retained.
    assert "adamw | muon | normuon" in by_key[("distillation", "optimizer")].description
    # `# Note: paths are RELATIVE ...` is prose, not a commented-out field.
    assert "RELATIVE" in by_key[("distillation", "dataset_dir")].description
    # cross-referenced `# resume_epochs: 0 (or unset) ...` prose above is_resume is kept.
    assert "continues to the original" in by_key[("resume", "is_resume")].description
