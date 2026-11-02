"""One val split for a whole data.yaml: the YAML rules, the weight split and the counts."""
from __future__ import annotations

import copy

import pytest
import yaml

from nanoplm.cli.templates import load_template
from nanoplm.data.preparation import data_config as dc
from nanoplm.data.preparation import val_split as vs


def _template():
    return yaml.safe_load(load_template("data_pretrain"))


def _mix(**params):
    cfg = _template()
    cfg["sources"] = {
        "swissprot": {"fasta": "swissprot", "weight": 0.2},
        "uniref50": {"fasta": "uniref50", "weight": 0.8, "max_seqs": -1,
                     "cluster_subsampling": {"members": "uniref90", "clusters": "uniprot"}},
    }
    cfg["data_params"].update(params)
    return cfg


# ------------------------------------------------------------------ the YAML rules


def test_val_ratio_and_val_size_together_are_an_error():
    cfg = _template()
    cfg["data_params"]["val_size"] = 1000
    with pytest.raises(dc.DataConfigError, match="set data_params.val_ratio or "
                                                 "data_params.val_size, not both"):
        dc.parse_data_config(cfg)


def test_neither_is_an_error():
    cfg = _template()
    del cfg["data_params"]["val_ratio"]
    with pytest.raises(dc.DataConfigError, match="data_params.val_ratio is required"):
        dc.parse_data_config(cfg)


@pytest.mark.parametrize("field, value", [("val_ratio", 0.01), ("val_size", 100)])
@pytest.mark.parametrize("mixed", [False, True])
def test_a_val_field_on_a_source_says_move_it_to_data_params(field, value, mixed):
    cfg = _mix() if mixed else _template()
    cfg["sources"]["swissprot"][field] = value
    with pytest.raises(dc.DataConfigError,
                       match=rf"sources.swissprot.{field}: move it to data_params"):
        dc.parse_data_config(cfg)


@pytest.mark.parametrize("value", [0, -5, 1.5, True, "100"])
def test_val_size_is_a_positive_whole_number(value):
    cfg = _template()
    del cfg["data_params"]["val_ratio"]
    cfg["data_params"]["val_size"] = value
    with pytest.raises(dc.DataConfigError, match="data_params.val_size must be a whole number"):
        dc.parse_data_config(cfg)


def test_one_source_with_val_ratio_keeps_its_own_split():
    cfg = dc.parse_data_config(_template())
    assert cfg.val == ("val_ratio", 0.1) and not cfg.val_from_split
    assert cfg.source.params["val_ratio"] == 0.1


def test_a_mix_or_val_size_draws_one_split():
    cfg = dc.parse_data_config(_mix(val_ratio=0.0044))
    assert cfg.val == ("val_ratio", 0.0044)
    assert all(s.val_from_split for s in cfg.sources.values())
    one = _template()
    del one["data_params"]["val_ratio"]
    one["data_params"]["val_size"] = 500
    got = dc.parse_data_config(one)
    assert got.val == ("val_size", 500) and got.source.val_from_split
    assert got.source.params["val_ratio"] is None


def test_val_size_needs_the_sources_layout():
    old = {"data_dirs": {"url": "x.fasta", "output_dir": "out"},
           "data_params": {"val_size": 10}}
    with pytest.raises(dc.DataConfigError, match="val_size needs the sources layout"):
        dc.parse_data_config(old)


# ------------------------------------------------------------------ the counts


@pytest.mark.parametrize("total, weights", [
    (147_000, {"a": 0.2, "b": 0.8}),
    (10, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}),
    (7, {"a": 0.5, "b": 0.25, "c": 0.25}),
    (1001, {"a": 0.123, "b": 0.456, "c": 0.421}),
])
def test_the_weight_split_sums_to_the_total(total, weights):
    got = vs.split_by_weight(total, weights)
    assert sum(got.values()) == total
    for n, w in weights.items():
        assert abs(got[n] - total * w / sum(weights.values())) < 1


def test_weights_of_built_sources_are_scaled_to_sum_to_1():
    assert vs.split_by_weight(100, {"a": 0.1, "b": 0.3}) == {"a": 25, "b": 75}


def test_val_ratio_counts_sequences_and_clusters_alike():
    """0.0044 of 570k swissprot sequences + 32.9M uniref50 clusters, 0.2 / 0.8."""
    pool = {"swissprot": (570_000, vs.PLAIN, 0.2), "uniref50": (32_900_000, vs.CLUSTER, 0.8)}
    got = vs.val_counts(pool, val_ratio=0.0044)
    total = round(0.0044 * 33_470_000)
    assert sum(got.values()) == total == 147_268
    assert got == {"swissprot": 29_454, "uniref50": 117_814}


def test_val_size_is_the_total():
    pool = {"a": (1000, vs.PLAIN, 0.5), "b": (1000, vs.CLUSTER, 0.5)}
    assert vs.val_counts(pool, val_size=301) == {"a": 151, "b": 150}


def test_a_source_running_short_names_it_and_the_fix():
    pool = {"swissprot": (570_000, vs.PLAIN, 0.2), "uniref50": (32_900_000, vs.CLUSTER, 0.8)}
    with pytest.raises(vs.ValSplitError,
                       match="sources.swissprot would need 669,400 val sequences but has "
                             "570,000; lower data_params.val_ratio or use data_params.val_size"):
        vs.val_counts(pool, val_ratio=0.1)
    with pytest.raises(vs.ValSplitError, match="sources.uniref50 would need 40 val clusters "
                                               "but has 40; lower data_params.val_size"):
        vs.val_counts({"a": (1000, vs.PLAIN, 0.5), "uniref50": (40, vs.CLUSTER, 0.5)},
                      val_size=80)


def test_every_source_gets_at_least_one_val_unit():
    pool = {"a": (10**6, vs.PLAIN, 0.999), "tiny": (100, vs.CLUSTER, 0.001)}
    with pytest.raises(vs.ValSplitError, match="sources.tiny would get 0 val clusters"):
        vs.val_counts(pool, val_size=100)


# ------------------------------------------------------------------ counting units


def test_count_fasta_records_across_chunk_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "_CHUNK", 7)
    f = tmp_path / "x.fasta"
    f.write_bytes(b"".join(b">id%d some text\nACDEFG\nHIK\n" % i for i in range(23)))
    assert vs.count_fasta_records(f) == 23
    assert sum(len(ids) for ids in vs._header_ids(f)) == 23
    assert [i for ids in vs._header_ids(f) for i in ids][:2] == [b"id0", b"id1"]


def test_count_universe_clusters_counts_reps_in_the_pool(tmp_path, monkeypatch):
    """A cluster is in the universe when its representative (its first table row) is
    in the pool FASTA; max_clusters caps the count."""
    tsv = tmp_path / "members.tsv"
    tsv.write_text("c1\tA\nc1\tB\nc2\tC\nc3\tD\nc3\tE\nc4\tF\n")
    pool = tmp_path / "pool.fasta"
    pool.write_text("".join(f">{a}\nMK\n" for a in ("A", "B", "E", "F")))  # reps A, F
    monkeypatch.setattr(vs, "_CHUNK", 5)
    assert vs.count_universe_clusters(pool, tsv) == 2
    assert vs.count_universe_clusters(pool, tsv, max_clusters=1) == 1
    assert vs.count_universe_clusters(pool, tsv, max_clusters=5) == 2
