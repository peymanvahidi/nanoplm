"""A source whose fasta is a list of files: parsing, dvc.yaml, the table, from-yaml."""
from __future__ import annotations

import gzip
from pathlib import Path

import pytest
import yaml

from nanoplm.cli._dvc_yaml_gen import _NANOPLM_PKG_DIR, generate_dvc_yaml
from nanoplm.cli.templates import load_template
from nanoplm.data.preparation import data_config as dc
from nanoplm.data.preparation.cluster_config import ClusterConfigError, resolve_cluster_plan


def _cfg(*sources, **one):
    """The template with output_dir "out" and the given sources ({name: spec})."""
    cfg = yaml.safe_load(load_template("data_pretrain"))
    cfg["output_dir"] = "out"
    cfg["sources"] = one or dict(sources)
    return cfg


def _stages(cfg, base_dir=None):
    text = generate_dvc_yaml(cfg, "pretrain", "data.yaml", base_dir)
    return yaml.safe_load(text.replace(str(_NANOPLM_PKG_DIR), "<PKG>"))["stages"]


URL_A = "https://example.org/a/shard.fa.zst"
URL_B = "https://example.org/b/shard.fa.zst"


# ------------------------------------------------------------------ parsing


def test_a_list_is_kept_in_order_and_merged_into_the_work_dir():
    src = dc.parse_data_config(_cfg(logan={"fasta": [URL_A, "x/c.fa", "y/d.fasta.gz"]})).source
    assert src.is_list and src.fasta == (URL_A, "x/c.fa", "y/d.fasta.gz")
    assert not src.is_url and not src.is_compressed
    assert src.paths["extracted_fasta"] == "out/logan/merged.fasta"
    assert src.paths["compressed_fasta"] is None


def test_a_one_item_list_is_that_file():
    one = _cfg(s={"fasta": ["x/c.fa.gz"]})
    assert dc.parse_data_config(one).source.fasta == "x/c.fa.gz"
    assert generate_dvc_yaml(one, "pretrain", "data.yaml") == \
        generate_dvc_yaml(_cfg(s={"fasta": "x/c.fa.gz"}), "pretrain", "data.yaml")


@pytest.mark.parametrize("fasta, message", [
    ([], r"sources\.s\.fasta is an empty list"),
    (["x/a.fa", "swissprot"], r"sources\.s\.fasta\[1\]: swissprot is a UniProt shortcut"),
    (["x/a.fa", 3], r"sources\.s\.fasta\[1\] must be a URL or a file path, got 3"),
    (["x/a.fa", ""], r"sources\.s\.fasta\[1\] must be a URL or a file path"),
    (["x/a.fa", "x/b.txt"], r"sources\.s\.fasta\[1\] must be a URL or path that ends with "
                            r"\.fasta or \.fa"),
    (["x/a.fa", "x/b.fa", "x/a.fa"], r"sources\.s\.fasta\[2\] is sources\.s\.fasta\[0\] again"),
    (["a.fa", "./a.fa"], r"sources\.s\.fasta\[1\] is sources\.s\.fasta\[0\] again"),
    (["x/a.fa", "x/../x/a.fa"], r"sources\.s\.fasta\[1\] is sources\.s\.fasta\[0\] again"),
    ([URL_A, ["x/a.fa"]], r"sources\.s\.fasta\[1\] must be a URL or a file path"),
])
def test_bad_lists_name_the_item(fasta, message):
    with pytest.raises(dc.DataConfigError, match=message):
        dc.parse_data_config(_cfg(s={"fasta": fasta}))


def test_a_path_and_its_absolute_form_are_the_same_file(tmp_path):
    with pytest.raises(dc.DataConfigError, match=r"fasta\[1\] is sources\.s\.fasta\[0\] again"):
        dc.parse_data_config(_cfg(s={"fasta": ["a.fa", str(tmp_path / "a.fa")]}), tmp_path)


def test_a_directory_in_a_list_is_refused(tmp_path):
    (tmp_path / "d.fa").mkdir()
    with pytest.raises(dc.DataConfigError, match=r"fasta\[1\] is a directory"):
        dc.parse_data_config(_cfg(s={"fasta": ["a.fa", "d.fa"]}), tmp_path)


# ------------------------------------------------------------------ dvc.yaml


def test_two_local_files_have_one_merge_stage_and_no_download():
    st = _stages(_cfg(s={"fasta": ["x/b.fa", "x/a.fa.gz"]}))
    assert list(st)[:3] == ["merge", "check", "shuffle"]
    assert st["merge"]["cmd"] == "nanoplm data extract -i x/b.fa -i x/a.fa.gz -o out/s/merged.fasta"
    assert st["merge"]["deps"] == ["x/b.fa", "x/a.fa.gz",
                                   "<PKG>/data/preparation/extractor.py"]
    assert st["merge"]["outs"] == ["out/s/merged.fasta"]
    assert "params" not in st["merge"]
    assert st["check"]["cmd"].startswith("nanoplm data check -i out/s/merged.fasta ")
    assert st["shuffle"]["cmd"].startswith("nanoplm data shuffle -i out/s/merged.fasta ")


def test_urls_with_one_file_name_download_to_two_files():
    st = _stages(_cfg(s={"fasta": [URL_A, "x/c.fa", URL_B]}))
    a, b = (f"out/downloads/{dc.list_download_name(u)}" for u in (URL_A, URL_B))
    assert a != b and a.endswith("-shard.fa.zst")
    downloads = [n for n in st if n.startswith("download-")]
    assert [st[n]["outs"] for n in downloads] == [[a], [b]]
    assert st[downloads[0]]["cmd"] == f'nanoplm data get-file --url "{URL_A}" -o {a}'
    assert st["merge"]["cmd"] == f"nanoplm data extract -i {a} -i x/c.fa -i {b} " \
                                 "-o out/s/merged.fasta"
    assert st["merge"]["deps"][:3] == [a, "x/c.fa", b]


def test_string_configs_keep_their_dvc_yaml():
    """The template and plain sources are untouched (the golden tests cover data_dirs)."""
    st = _stages(_cfg(s={"fasta": "https://example.org/a.fa.zst"}))
    assert list(st)[:3] == ["get_file", "extract", "check"]
    assert not any(n.startswith(("download-", "merge")) for n in st)


def test_a_mix_shares_a_url_and_merges_per_source():
    st = _stages(_cfg(("l", {"fasta": [URL_A, "x/c.fa"], "weight": 0.5}),
                      ("m", {"fasta": [URL_A, "y/d.fa.gz"], "weight": 0.3}),
                      ("p", {"fasta": "swissprot", "weight": 0.2})))
    downloads = [n for n in st if n.startswith("download-")]
    assert len(downloads) == 2                      # URL_A once, swissprot once
    assert st["l-merge"]["outs"] == ["out/l/merged.fasta"]
    assert st["m-merge"]["outs"] == ["out/m/merged.fasta"]
    assert st["l-check"]["outs"] == ["out/l/validation"]
    mix = st["mixture"]["cmd"]
    assert f'"{URL_A}, x/c.fa"' in mix and f'"{URL_A}, y/d.fa.gz"' in mix


# ------------------------------------------------------------------ cluster subsampling


def _cluster_src(block, base_dir=None, fasta=("x/a.fa", "x/b.fa")):
    cfg = _cfg(s={"fasta": list(fasta), "cluster_subsampling": block})
    return dc.parse_data_config(cfg, base_dir).source, cfg


def test_mmseqs2_clusters_the_merged_list():
    src, cfg = _cluster_src({"clusters": "mmseqs2"})
    plan = resolve_cluster_plan(src)
    assert plan.members_from_source and str(plan.members_fasta) == "out/s/merged.fasta"
    st = _stages(cfg)
    assert list(st)[:3] == ["merge", "check", "filter"]
    assert st["filter"]["cmd"].startswith("nanoplm data filter -i out/s/merged.fasta ")


def test_headers_peeks_the_first_item(tmp_path):
    (tmp_path / "a.fa").write_text(">s1 CL:c1\nMK\n")
    (tmp_path / "b.fa").write_text(">s2\nMK\n")
    src, _ = _cluster_src({"clusters": "headers"}, tmp_path, ("a.fa", "b.fa"))
    assert resolve_cluster_plan(src, tmp_path).cluster_source == "headers"
    src, _ = _cluster_src({"clusters": "headers"}, tmp_path, ("b.fa", "a.fa"))
    with pytest.raises(ClusterConfigError, match=r"1 of the first 1 headers of b\.fa"):
        resolve_cluster_plan(src, tmp_path)


def test_members_and_uniprot_are_refused_with_a_list():
    src, _ = _cluster_src({"clusters": "mmseqs2", "members": "uniref90"})
    with pytest.raises(ClusterConfigError, match="not supported with a fasta list yet"):
        resolve_cluster_plan(src)
    src, _ = _cluster_src({"clusters": "uniprot"})
    with pytest.raises(ClusterConfigError, match=r"got fasta x/a\.fa and 1 more and members"):
        resolve_cluster_plan(src)


# ------------------------------------------------------------------ the table, from-yaml


def test_the_table_shows_the_first_item_and_how_many_more(tmp_path):
    from nanoplm.cli.data import plan_table
    (tmp_path / "a.fa").write_text(">a\nMK\n")
    cfg = dc.parse_data_config(_cfg(s={"fasta": [URL_A, "a.fa", URL_B]}), tmp_path)
    table = plan_table(cfg, {"s": None}, tmp_path, tmp_path / "data.yaml")
    assert f"{dc.list_download_name(URL_A)} (url) and 2 more" in table


def test_from_yaml_names_a_missing_list_item(tmp_path):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    (tmp_path / "a.fa").write_text(">a\nMK\n")
    cfg = _cfg(s={"fasta": ["a.fa", "missing.fa"]})
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg))
    r = CliRunner().invoke(data, ["from-yaml", str(tmp_path / "data.yaml"), "--dry-run"])
    assert r.exit_code != 0 and "sources.s.fasta[1]: missing.fa not found" in r.output


def test_from_yaml_builds_a_list_of_plain_gz_and_zst(tmp_path):
    import zstandard
    from nanoplm.data.infra.manifest import read_manifest
    from tests.data.dvc.test_cluster_stages import _from_yaml, _project
    texts = []
    for f in range(3):
        texts.append("".join(f">f{f}_{i}\n" + "ACDEFGHIKLMNPQRSTVWY"[i % 20] * (30 + i) + "\n"
                             for i in range(20)))
    (tmp_path / "a.fa").write_text(texts[0].rstrip("\n"))          # no newline at the end
    with gzip.open(tmp_path / "b.fasta.gz", "wt") as f:
        f.write(texts[1])
    (tmp_path / "c.fa.zst").write_bytes(zstandard.ZstdCompressor().compress(texts[2].encode()))
    env = _project(tmp_path, None, ["c.fa.zst", "a.fa", "b.fasta.gz"], max_seqs=-1,
                   min_seq_len=20)
    _from_yaml(tmp_path, env)
    out = tmp_path / "output/data/pretrain_data"
    assert (out / "src/merged.fasta").read_text() == texts[2] + texts[0] + texts[1]
    m = read_manifest(out)
    assert m.train_sequences + m.val_sequences == 60
