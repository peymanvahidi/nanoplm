from pathlib import Path
from tools.schema_gen import dataclass_ast as dc, sources

def test_sources_resolve_independent_of_cwd(monkeypatch, tmp_path):
    # Paths are absolute (anchored to repo root), so they resolve no matter
    # what directory the generator/tests are invoked from.
    monkeypatch.chdir(tmp_path)
    for spec in sources.TEMPLATE_SOURCES.values():
        assert Path(spec.file).is_absolute() and Path(spec.file).exists()
    assert Path(sources.COMPAT_MATRIX).is_absolute() and Path(sources.COMPAT_MATRIX).exists()

def test_parity_allowlist_keys_match_parity_sources():
    # The two dicts are maintained in parallel; lock the invariant so a new
    # parity surface can't be added to one without the other.
    assert sources.PARITY_INTERNAL_ALLOWLIST.keys() == sources.PARITY_SOURCES.keys()

def test_template_sources_point_at_real_files():
    for surface in ("data", "pretrain", "distill"):
        spec = sources.TEMPLATE_SOURCES[surface]
        assert Path(spec.file).exists(), f"{surface}: {spec.file} missing"

def test_eval_dataclass_source_exists():
    spec = sources.DATACLASS_SOURCES["eval"]
    assert Path(spec.file).exists()
    assert spec.classname == "EvalConfig"

def test_parity_dataclass_sources_exist():
    for key in ("pretraining", "resume", "distillation", "eval"):
        spec = sources.PARITY_SOURCES[key]
        assert Path(spec.file).exists()

def test_parity_allowlist_names_only_fields_that_still_exist():
    # A renamed field leaves a dead allowlist entry that silently excludes
    # nothing (this is exactly how `global_batch_size_samples` outlived the
    # rename to `global_batch_seqs`). Lock the allowlist to the real dataclass.
    for key, allowed in sources.PARITY_INTERNAL_ALLOWLIST.items():
        spec = sources.PARITY_SOURCES[key]
        real = set(dc.field_names(spec.file, spec.classname))
        stale = allowed - real
        assert not stale, f"{key}: allowlisted fields no longer on the dataclass: {sorted(stale)}"
