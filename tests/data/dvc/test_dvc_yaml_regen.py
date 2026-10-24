# tests/data/dvc/test_dvc_yaml_regen.py
import pytest
import yaml
from nanoplm.cli._dvc_yaml_gen import generate_dvc_yaml
from nanoplm.data.preparation.data_config import is_compressed


PRETRAIN_URL_CONFIG = {
    "data_dirs": {
        "url": "https://example.com/x.fasta.gz",
        "local_fasta": None,
        "compressed_fasta": "out/raw/x.fasta.gz",
        "extracted_fasta": "out/raw/x.fasta",
        "validation_dir": "out/validation",
        "shuffled_fasta": "out/raw/x_shuffled.fasta",
        "filtered_fasta": "out/filter/x_filtered.fasta",
        "splitted_fasta_dir": "out/split",
        "output_dir": "out/pretrain",
    },
    "data_params": {
        "seqs_num": 100, "min_seq_len": 20, "max_seq_len": 512,
        "val_ratio": 0.1, "shuffle": True, "shuffle_seed": 24,
        "drop_invalid_sequences": False,
        "samples_per_shard": 10, "max_workers": 2, "register_tokens": 0,
    },
}


PRETRAIN_LOCAL_CONFIG = {
    **PRETRAIN_URL_CONFIG,
    "data_dirs": {**PRETRAIN_URL_CONFIG["data_dirs"], "url": None,
                  "local_fasta": "/local/path/x.fasta.gz"},
}


def test_pretrain_url_mode_dvc_yaml(tmp_path):
    out = generate_dvc_yaml(PRETRAIN_URL_CONFIG, kind="pretrain", params_filename="data.yaml")
    parsed = yaml.safe_load(out)
    stages = parsed["stages"]
    assert "get_file" in stages
    assert "extract" in stages
    assert "check" in stages
    assert "shuffle" in stages
    assert "filter" in stages
    assert "split" in stages
    assert "pre-tokenize" in stages
    assert "write_manifest" in stages
    # URL mode: get_file has no local_fasta dep
    deps = stages["get_file"].get("deps", [])
    assert "${data_dirs.local_fasta}" not in deps
    assert "/local/path" not in str(deps)


def test_pretrain_uses_foreach_for_tokenize():
    out = generate_dvc_yaml(PRETRAIN_URL_CONFIG, kind="pretrain", params_filename="data.yaml")
    parsed = yaml.safe_load(out)
    pre_tok = parsed["stages"]["pre-tokenize"]
    assert "foreach" in pre_tok
    assert "train" in pre_tok["foreach"]
    assert "val" in pre_tok["foreach"]
    assert "do" in pre_tok


def test_generated_deps_are_absolute_and_exist():
    """Source-file deps in dvc.yaml must be absolute on-disk paths so DVC works
    when invoked from a separate data directory (not the nanoplm repo root)."""
    from pathlib import Path
    out = generate_dvc_yaml(PRETRAIN_URL_CONFIG, kind="pretrain", params_filename="data.yaml")
    parsed = yaml.safe_load(out)
    for stage_name, stage in parsed["stages"].items():
        if "foreach" in stage:
            deps = stage["do"].get("deps", [])
        else:
            deps = stage.get("deps", [])
        for dep in deps:
            # Skip ${...} interpolation refs (DVC-resolved at run time)
            if "${" in dep:
                continue
            assert Path(dep).is_absolute(), (
                f"Stage {stage_name!r}: dep {dep!r} is relative; "
                f"generated dvc.yaml must use absolute source-file paths "
                f"so it works from any cwd."
            )
            assert Path(dep).exists(), (
                f"Stage {stage_name!r}: absolute dep {dep!r} does not exist on disk."
            )


def test_distillation_uses_distillation_yaml_filename():
    cfg = {
        "data_dirs": {**PRETRAIN_URL_CONFIG["data_dirs"]},
        "data_params": {
            "seqs_num": 100, "min_seq_len": 20, "max_seq_len": 512,
            "val_ratio": 0.1, "shuffle": True, "shuffle_seed": 24,
            "drop_invalid_sequences": False, "device": "auto",
            "teacher_model": "prott5", "samples_per_shard": 10,
            "embed_calc_batch_size": 4,
        },
    }
    out = generate_dvc_yaml(cfg, kind="distillation", params_filename="distillation_data.yaml")
    parsed = yaml.safe_load(out)
    assert "kd-embed" in parsed["stages"]
    assert "pre-tokenize" not in parsed["stages"]
    # params: references distillation_data.yaml not data.yaml
    pre_filter = parsed["stages"]["filter"]
    assert any("distillation_data.yaml" in str(p) for p in pre_filter["params"])


@pytest.mark.parametrize("source,expected", [
    # Local paths
    ("/path/to/file.fasta", False),
    ("/path/to/file.fasta.gz", True),
    ("relative/path.fasta", False),
    ("relative/path.fasta.gz", True),
    # URLs without query strings
    ("https://example.com/x.fasta", False),
    ("https://example.com/x.fasta.gz", True),
    ("http://example.com/x.fasta.gz", True),
    # URLs with query strings (key: strip ?query before suffix check)
    ("https://example.com/x.fasta.gz?token=abc", True),
    ("https://example.com/x.fasta?token=abc", False),
    # URLs with fragments
    ("https://example.com/x.fasta.gz#section", True),
    ("https://example.com/x.fasta#section", False),
])
def test_source_is_compressed(source, expected):
    assert is_compressed(source) is expected


PLAIN_LOCAL_CONFIG = {
    **PRETRAIN_URL_CONFIG,
    "data_dirs": {**PRETRAIN_URL_CONFIG["data_dirs"], "url": None,
                  "local_fasta": "/local/path/x.fasta"},
}

PLAIN_URL_CONFIG = {
    **PRETRAIN_URL_CONFIG,
    "data_dirs": {**PRETRAIN_URL_CONFIG["data_dirs"],
                  "url": "https://example.com/x.fasta"},
}


@pytest.mark.parametrize("config,is_compressed", [
    (PRETRAIN_URL_CONFIG, True),        # .fasta.gz URL
    (PRETRAIN_LOCAL_CONFIG, True),      # .fasta.gz local
    (PLAIN_URL_CONFIG, False),          # plain .fasta URL
    (PLAIN_LOCAL_CONFIG, False),        # plain .fasta local
])
def test_dvc_yaml_extract_stage_matches_compression(config, is_compressed):
    out = generate_dvc_yaml(config, kind="pretrain", params_filename="data.yaml")
    parsed = yaml.safe_load(out)
    stages = parsed["stages"]

    if is_compressed:
        assert "extract" in stages
        assert stages["get_file"]["outs"] == ["${data_dirs.compressed_fasta}"]
    else:
        assert "extract" not in stages
        # When source is plain, get_file writes the extracted file directly
        assert stages["get_file"]["outs"] == ["${data_dirs.extracted_fasta}"]
        # Downstream stages must still reference extracted_fasta unchanged
        assert "${data_dirs.extracted_fasta}" in stages["check"]["deps"]
        assert "${data_dirs.extracted_fasta}" in stages["shuffle"]["deps"]

    # Local-mode invariant: local_fasta appears in get_file.deps (was previously
    # covered by the removed test_pretrain_local_mode_dvc_yaml).
    if config["data_dirs"].get("local_fasta"):
        assert "${data_dirs.local_fasta}" in stages["get_file"]["deps"]


def test_dvc_yaml_plain_get_file_params_omit_compressed_fasta():
    """In plain mode, get_file's params: must not reference compressed_fasta
    (the field may be missing from data.yaml entirely)."""
    out = generate_dvc_yaml(PLAIN_LOCAL_CONFIG, kind="pretrain", params_filename="data.yaml")
    parsed = yaml.safe_load(out)
    params = parsed["stages"]["get_file"]["params"]
    flat = []
    for p in params:
        if isinstance(p, dict):
            for v in p.values():
                flat.extend(v)
    assert "data_dirs.compressed_fasta" not in flat
