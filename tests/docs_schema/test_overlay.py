# tests/docs_schema/test_overlay.py
from tools.schema_gen import overlay

def test_optimizer_enum_present():
    o = overlay.load()
    entry = o["pretraining.optimizer"]
    # Must match click.Choice in cli/pretrain.py::run --optimizer.
    assert entry["enum"] == ["adamw", "stable_adamw", "muon", "normuon", "nordion2"]

def test_lr_schedule_enum_matches_validator():
    # click.Choice --lr-schedule and _validate_duration_config's `valid` set.
    assert overlay.load()["pretraining.lr_schedule"]["enum"] == [
        "wsd", "warmup_stable", "cosine", "linear"
    ]

def test_extra_fields_key_is_reserved_not_a_field_annotation():
    o = overlay.load()
    extras = o[overlay.EXTRA_FIELDS_KEY]
    assert "train_tokens" in extras["pretrain"]["pretraining"]
    assert "decay_steps" in extras["pretrain"]["resume"]
    assert "adam_beta1" in extras["distill"]["distillation"]
    # The reserved key must never be read as a "<section>.<field>" annotation...
    assert overlay.for_field(o, overlay.EXTRA_FIELDS_KEY) == {}
    # ...and the flat annotation namespace keeps working unchanged.
    assert overlay.for_field(o, "pretraining.lr_schedule")["enum"][0] == "wsd"

def test_extra_fields_for_unknown_surface_is_empty():
    assert overlay.extra_fields(overlay.load(), "data") == {}

def test_backend_tag_for_pure_torch_only_field():
    o = overlay.load()
    assert "pure_torch" in overlay.backends_for(o, "model.use_canon_layers")

def test_kernel_size_enum():
    o = overlay.load()
    assert overlay.load()["model.canon_layers_kernel_size"]["enum"] == [3, 5, 7]

def test_unknown_key_returns_empty_overlay():
    o = overlay.load()
    assert overlay.for_field(o, "pretraining.does_not_exist") == {}

def test_mlp_activation_enum_matches_validator():
    # Must match the authoritative validator (config.py: {swiglu, geglu, srelu}),
    # NOT gelu/relu which are rejected at config-build time.
    assert overlay.load()["model.mlp_activation"]["enum"] == ["swiglu", "geglu", "srelu"]

def test_for_field_does_not_alias_cached_overlay():
    # Mutating a for_field result must not corrupt the lru_cache'd overlay.
    got = overlay.for_field(overlay.load(), "pretraining.optimizer")
    got["enum"].append("BOGUS")
    assert "BOGUS" not in overlay.load()["pretraining.optimizer"]["enum"]
