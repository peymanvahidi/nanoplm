"""uniref_hierarchy: the accession hash and the XML scan (ported from the experiment's
test_extract_hierarchy.py)."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from nanoplm.data.preparation import uniref_hierarchy as uh
from tests.data.preparation.uniref_fixture import make_release

ALPHANUM = np.array(list(b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"), dtype=np.uint8)


# ------------------------------------------------------------------- hash ---


def test_hash_matches_scalar_reference_at_every_length():
    accs = [b"P12345", b"A0ACN6ENM0", b"UPI00447321E6", b"Q8WZ42-10", b"A2ASS6-2", b"X"]
    keys = uh.hash_strings(accs)
    assert [int(k) for k in keys] == [uh.hash_one(a) for a in accs]
    assert keys.dtype == np.uint64


def test_hash_is_pinned():
    # the experiment's hierarchy arrays were keyed with these exact values
    assert uh.hash_one(b"P12345") == 0x786312A9CF0D4F21
    assert uh.hash_one(b"A0ACN6ENM0") == 0xD635338358E73FCA
    assert uh.hash_one(b"UPI00447321E6") == 0x6E6C0E9B95202396


def test_salt_changes_every_key():
    accs = [b"P12345", b"A0ACN6ENM0", b"UPI00447321E6"]
    plain, salted = uh.hash_strings(accs, 0), uh.hash_strings(accs, 0xDEADBEEF)
    assert not np.any(plain == salted)
    assert [int(k) for k in salted] == [uh.hash_one(a, 0xDEADBEEF) for a in accs]


def test_hash_collision_free_and_batch_independent():
    rng = np.random.default_rng(7)
    accs = sorted({bytes(r) for w in (6, 10, 13)
                   for r in ALPHANUM[rng.integers(0, 36, size=(60_000, w))]})
    keys = uh.hash_strings(accs)
    assert len(np.unique(keys)) == len(accs)
    lengths = np.array([len(a) for a in accs])
    blob = np.frombuffer(b"".join(accs), dtype=np.uint8)
    assert np.array_equal(keys, uh.hash_accessions(blob, lengths, batch=97))


# ------------------------------------------------------------------ parse ---


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return make_release(tmp_path_factory.mktemp("release"))


@pytest.mark.parametrize("level,truth_attr,prefix", [("uniref90", "u90", "u90"),
                                                      ("uniref100", "u100", "u100")])
def test_hierarchy_matches_ground_truth(release, tmp_path, level, truth_attr, prefix):
    out = tmp_path / "h"
    stats = uh.extract_hierarchy(release.xml, out, level=level)
    truth = getattr(release, truth_attr)
    assert stats["verify"]["ok"] and stats["counts"]["members"] == len(truth)
    assert stats["input"]["release_version"] == "2026_03"
    keys, cluster, is_rep, sizes = uh.load_hierarchy(out, level)
    accs = list(truth)
    probe = uh.hash_strings([a.encode() for a in accs])
    pos = np.searchsorted(keys, probe)
    assert np.array_equal(keys[pos], probe)
    assert np.array_equal(cluster[pos], [truth[a][0] for a in accs])
    assert np.array_equal(is_rep[pos], [truth[a][1] for a in accs])
    assert np.array_equal(sizes, np.bincount([truth[a][0] for a in accs]))
    assert (out / "cluster_ids.txt").read_text().split() == release.cluster_ids
    assert (out / f"{prefix}_keys.npy").exists()


def _xml(entries: str) -> bytes:
    return ('<?xml version="1.0" encoding="ISO-8859-1" ?>\n<UniRef50 version="2026_03">\n'
            + entries + "</UniRef50>\n").encode()


def _entry(u50, members, rep_blocks=1, member_first=False):
    lines = [f'<entry id="UniRef50_{u50}">']
    for i, m in enumerate(members):
        w = "representativeMember" if i < rep_blocks and not member_first else "member"
        lines += [f"<{w}>", f'<property type="UniRef90 ID" value="UniRef90_{m}"/>', f"</{w}>"]
    if member_first:
        lines += ["<representativeMember>",
                  f'<property type="UniRef90 ID" value="UniRef90_{u50}"/>',
                  "</representativeMember>"]
    return "\n".join(lines + ["</entry>"]) + "\n"


def _write(tmp_path, data: bytes, gz=True):
    p = tmp_path / ("x.xml.gz" if gz else "x.xml")
    p.write_bytes(gzip.compress(data) if gz else data)
    return p


def test_dedup_and_representative(tmp_path):
    src = _write(tmp_path, _xml(_entry("A1", ["A1", "A1", "B2", "A1", "UPI0001"])
                                + _entry("C3", ["C3"])
                                + _entry("D4", ["UPI0002", "D4", "Q8WZ42-10"])))
    uh.extract_hierarchy(src, tmp_path / "h")
    keys, cluster, is_rep, sizes = uh.load_hierarchy(tmp_path / "h", "uniref90")
    assert sizes.tolist() == [3, 1, 3]
    reps = {uh.hash_one(b"A1"), uh.hash_one(b"C3"), uh.hash_one(b"UPI0002")}
    assert set(keys[np.asarray(is_rep)].tolist()) == reps


def test_plain_xml_and_python_filter_give_the_same_arrays(release, tmp_path, monkeypatch):
    uh.extract_hierarchy(release.xml, tmp_path / "gz")
    plain = tmp_path / "u50.xml"
    plain.write_bytes(gzip.decompress(release.xml.read_bytes()))
    uh.extract_hierarchy(plain, tmp_path / "plain")
    real_which = uh.shutil.which
    monkeypatch.setattr(uh.shutil, "which", lambda n: None if n == "grep" else real_which(n))
    uh.extract_hierarchy(release.xml, tmp_path / "py")
    for d in ("plain", "py"):
        for name in ("u90_keys.npy", "u90_cluster.npy", "u90_is_rep.npy",
                     "cluster_raw_sizes.npy"):
            assert np.array_equal(np.load(tmp_path / "gz" / name), np.load(tmp_path / d / name))


def test_cross_entry_duplicate_is_refused(tmp_path):
    src = _write(tmp_path, _xml(_entry("A1", ["A1", "B2"]) + _entry("C3", ["C3", "B2"])))
    with pytest.raises(uh.HierarchyError, match="same id in two entries"):
        uh.extract_hierarchy(src, tmp_path / "h")


def test_truncated_gzip_is_refused(release, tmp_path):
    raw = release.xml.read_bytes()
    src = tmp_path / "cut.xml.gz"
    src.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(uh.HierarchyError, match="incomplete"):
        uh.extract_hierarchy(src, tmp_path / "h")


def test_missing_root_close_is_refused(tmp_path):
    src = _write(tmp_path, _xml(_entry("A1", ["A1"]))[: -len("</UniRef50>\n")])
    with pytest.raises(uh.HierarchyError, match="incomplete"):
        uh.extract_hierarchy(src, tmp_path / "h")


@pytest.mark.parametrize("entry,match", [
    (_entry("A1", ["A1", "B2"], rep_blocks=2), "2 <representativeMember>"),
    (_entry("A1", ["B2"], member_first=True), "before its <representativeMember>"),
    ('<entry id="UniRef50_A1">\n<representativeMember>\n</representativeMember>\n</entry>\n',
     "carries no uniref90 id"),
])
def test_malformed_entries_fail_loudly(tmp_path, entry, match):
    src = _write(tmp_path, _xml(entry))
    with pytest.raises(uh.HierarchyError, match=match):
        uh.extract_hierarchy(src, tmp_path / "h")


def test_verify_catches_a_tampered_array_and_a_wrong_hash(release, tmp_path):
    out = tmp_path / "h"
    uh.extract_hierarchy(release.xml, out)
    cl, rep = np.load(out / "u90_cluster.npy"), np.load(out / "u90_is_rep.npy")
    i = int(np.flatnonzero(rep)[0])
    j = int(np.flatnonzero(~rep & (cl != cl[i]))[0])
    rep[i], rep[j] = False, True          # one cluster loses its rep, another gains one
    np.save(out / "u90_is_rep.npy", rep)
    assert "one_representative_per_cluster" in uh.verify_hierarchy(out, "uniref90")["failures"]
    stats = json.loads((out / "hierarchy_stats.json").read_text())
    stats["hash"]["test_vectors"]["P12345"] = "0x0000000000000000"
    assert "hash_self_check" in uh.verify_hierarchy(out, "uniref90", stats=stats)["failures"]


def test_load_refuses_an_unverified_or_other_level_hierarchy(release, tmp_path):
    out = tmp_path / "h"
    uh.extract_hierarchy(release.xml, out)
    with pytest.raises(uh.HierarchyError, match="uniref90 hierarchy, not uniref100"):
        uh.load_hierarchy(out, "uniref100")
    stats = json.loads((out / "hierarchy_stats.json").read_text())
    stats["verify"]["ok"] = False
    (out / "hierarchy_stats.json").write_text(json.dumps(stats))
    with pytest.raises(uh.HierarchyError, match="verify.ok"):
        uh.load_hierarchy(out, "uniref90")
