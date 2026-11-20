"""uniref_hierarchy: the accession hash and the XML scan (ported from the experiment's
test_extract_hierarchy.py)."""
from __future__ import annotations

import gzip

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


def _rows(tsv):
    return [tuple(line.split("\t")) for line in tsv.read_text().splitlines()]


@pytest.mark.parametrize("level", ["uniref90", "uniref100"])
def test_members_table_matches_ground_truth(release, tmp_path, level):
    out = tmp_path / "members.tsv"
    stats = uh.xml_to_members_tsv(release.xml, out, level=level)
    truth = release.u90 if level == "uniref90" else release.u100
    prefix = "UniRef90_" if level == "uniref90" else "UniRef100_"
    rows = _rows(out)
    assert stats["members"] == len(rows) == len(truth)
    assert stats["clusters"] == len(release.cluster_ids)
    assert stats["release_version"] == "2026_03"
    # grouped in XML entry order, representative first, full FASTA ids
    clusters = [c for c, _ in rows]
    firsts = [i for i in range(len(rows)) if i == 0 or clusters[i] != clusters[i - 1]]
    assert [clusters[i] for i in firsts] == [f"UniRef50_{u}" for u in release.cluster_ids]
    for i, (c, m) in enumerate(rows):
        acc = m[len(prefix):]
        assert m.startswith(prefix)
        cl, is_rep, _ = truth[acc]
        assert c == f"UniRef50_{release.cluster_ids[cl]}"
        assert is_rep == (i in set(firsts))


def test_uniref100_uses_the_paired_rep(release, tmp_path):
    assert release.literal_differs > 0
    stats = uh.xml_to_members_tsv(release.xml, tmp_path / "m.tsv", level="uniref100")
    assert stats["paired_rep_differs_from_literal"] == release.literal_differs
    assert stats["paired_rep_missing"] == 0
    rows90 = _rows(tmp_path / "m.tsv")
    uh.xml_to_members_tsv(release.xml, tmp_path / "m90.tsv", level="uniref90")
    reps90 = [m[len("UniRef90_"):] for i, (c, m) in enumerate(_rows(tmp_path / "m90.tsv"))
              if i == 0 or c != _rows(tmp_path / "m90.tsv")[i - 1][0]]
    reps100 = [m[len("UniRef100_"):] for i, (c, m) in enumerate(rows90)
               if i == 0 or c != rows90[i - 1][0]]
    assert reps100 == reps90          # the same accession, hence the same sequence


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
    uh.xml_to_members_tsv(src, tmp_path / "m.tsv")
    assert (tmp_path / "m.tsv").read_text() == (
        "UniRef50_A1\tUniRef90_A1\nUniRef50_A1\tUniRef90_B2\nUniRef50_A1\tUniRef90_UPI0001\n"
        "UniRef50_C3\tUniRef90_C3\n"
        "UniRef50_D4\tUniRef90_UPI0002\nUniRef50_D4\tUniRef90_D4\n"
        "UniRef50_D4\tUniRef90_Q8WZ42-10\n")


def test_plain_xml_and_python_filter_give_the_same_table(release, tmp_path, monkeypatch):
    uh.xml_to_members_tsv(release.xml, tmp_path / "gz.tsv")
    plain = tmp_path / "u50.xml"
    plain.write_bytes(gzip.decompress(release.xml.read_bytes()))
    uh.xml_to_members_tsv(plain, tmp_path / "plain.tsv")
    real_which = uh.shutil.which
    monkeypatch.setattr(uh.shutil, "which", lambda n: None if n == "grep" else real_which(n))
    uh.xml_to_members_tsv(release.xml, tmp_path / "py.tsv")
    want = (tmp_path / "gz.tsv").read_bytes()
    assert (tmp_path / "plain.tsv").read_bytes() == want == (tmp_path / "py.tsv").read_bytes()


def test_truncated_gzip_is_refused(release, tmp_path):
    raw = release.xml.read_bytes()
    src = tmp_path / "cut.xml.gz"
    src.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(uh.HierarchyError, match="incomplete"):
        uh.xml_to_members_tsv(src, tmp_path / "m.tsv")
    assert not (tmp_path / "m.tsv").exists()


def test_missing_root_close_is_refused(tmp_path):
    src = _write(tmp_path, _xml(_entry("A1", ["A1"]))[: -len("</UniRef50>\n")])
    with pytest.raises(uh.HierarchyError, match="incomplete"):
        uh.xml_to_members_tsv(src, tmp_path / "m.tsv")


@pytest.mark.parametrize("entry,match", [
    (_entry("A1", ["A1", "B2"], rep_blocks=2), "2 <representativeMember>"),
    (_entry("A1", ["B2"], member_first=True), "before its <representativeMember>"),
    ('<entry id="UniRef50_A1">\n<representativeMember>\n</representativeMember>\n</entry>\n',
     "carries no uniref90 id"),
])
def test_malformed_entries_fail_loudly(tmp_path, entry, match):
    src = _write(tmp_path, _xml(entry))
    with pytest.raises(uh.HierarchyError, match=match):
        uh.xml_to_members_tsv(src, tmp_path / "m.tsv")


# ------------------------------------------------------------ sub-clusters ---


def test_submembers_match_ground_truth_and_members_unchanged(release, tmp_path):
    from nanoplm.data.preparation.cluster_table import read_members_table
    uh.xml_to_members_tsv(release.xml, tmp_path / "plain.tsv", level="uniref100")
    stats = uh.xml_to_members_tsv(release.xml, tmp_path / "m.tsv", level="uniref100",
                                  sub_out=tmp_path / "sub.tsv")
    assert (tmp_path / "m.tsv").read_bytes() == (tmp_path / "plain.tsv").read_bytes()
    rows = _rows(tmp_path / "sub.tsv")
    got = {m[len("UniRef100_"):]: s[len("UniRef90_"):] for s, m in rows}
    assert len(got) == len(rows) == len(release.sub90)
    assert got == release.sub90
    subs = [s for s, _ in rows]
    firsts = [s for i, s in enumerate(subs) if i == 0 or s != subs[i - 1]]
    assert len(firsts) == len(set(firsts)) == stats["subclusters"] == len(release.u90)
    # every sub-cluster lies inside one cluster of members.tsv
    cluster_of = {m: c for c, m in _rows(tmp_path / "m.tsv")}
    for s in set(subs):
        assert len({cluster_of[m] for t, m in rows if t == s}) == 1
    table = read_members_table(tmp_path / "sub.tsv")
    assert len(table.keys) == len(rows) and len(table.sizes) == len(firsts)


def test_submembers_same_with_the_python_filter(release, tmp_path, monkeypatch):
    uh.xml_to_members_tsv(release.xml, tmp_path / "m.tsv", "uniref100", sub_out=tmp_path / "a")
    real_which = uh.shutil.which
    monkeypatch.setattr(uh.shutil, "which", lambda n: None if n == "grep" else real_which(n))
    uh.xml_to_members_tsv(release.xml, tmp_path / "m.tsv", "uniref100", sub_out=tmp_path / "b")
    assert (tmp_path / "a").read_bytes() == (tmp_path / "b").read_bytes()


def _rec(w, u100, u90):
    lines = [f"<{w}>", f'<dbReference type="UniProtKB ID" id="{u100 or u90}_TEST">']
    if u100:
        lines.append(f'<property type="UniRef100 ID" value="UniRef100_{u100}"/>')
    if u90:
        lines.append(f'<property type="UniRef90 ID" value="UniRef90_{u90}"/>')
    return lines + ["</dbReference>", f"</{w}>"]


def _entry100(u50, recs):
    lines = [f'<entry id="UniRef50_{u50}">']
    for i, (u100, u90) in enumerate(recs):
        lines += _rec("representativeMember" if i == 0 else "member", u100, u90)
    return "\n".join(lines + ["</entry>"]) + "\n"


def test_submembers_small_table(tmp_path):
    src = _write(tmp_path, _xml(_entry100("A1", [("A1", "A1"), ("B2", "B2"), ("C3", "A1"),
                                                 ("C3", "A1"), ("D4", "B2")])
                                + _entry100("E5", [("E5", "E5")])))
    uh.xml_to_members_tsv(src, tmp_path / "m.tsv", "uniref100", sub_out=tmp_path / "s.tsv")
    assert (tmp_path / "s.tsv").read_text() == (
        "UniRef90_A1\tUniRef100_A1\nUniRef90_A1\tUniRef100_C3\n"
        "UniRef90_B2\tUniRef100_B2\nUniRef90_B2\tUniRef100_D4\n"
        "UniRef90_E5\tUniRef100_E5\n")


@pytest.mark.parametrize("entries,match", [
    (_entry100("A1", [("A1", "A1"), ("B2", None), ("C3", "A1")]), "1 member records .*B2"),
    (_entry100("A1", [("A1", "A1"), (None, "B2")]), "do not alternate.*UniRef90_B2 follows"),
    # one record without UniRef90, the next without UniRef100: never paired across records
    (_entry100("A1", [("A1", "A1"), ("B2", None), (None, "C3")]),
     "2 member records .*UniRef100_B2 has no UniRef90 line; .*UniRef90_C3 follows no UniRef100"),
    (_entry100("A1", [("A1", "A1"), ("B2", None)]), "UniRef100_B2 has no UniRef90"),
    (_entry100("A1", [("A1", "A1"), ("B2", "A1"), ("B2", "C3")]),
     "1 UniRef100 ids listed under two UniRef90 ids.*UniRef90_A1 and UniRef90_C3"),
    (_entry100("A1", [("A1", "A1")]) + _entry100("B2", [("B2", "B2"), ("C3", "A1")]),
     "1 UniRef90 ids appear in two UniRef50 entries .first: UniRef90_A1 in UniRef50_A1 and "
     "UniRef50_B2"),
])
def test_submembers_bad_pairing_fails(tmp_path, entries, match):
    src = _write(tmp_path, _xml(entries))
    uh.xml_to_members_tsv(src, tmp_path / "ok.tsv", "uniref100")      # members alone pass
    with pytest.raises(uh.HierarchyError, match=match):
        uh.xml_to_members_tsv(src, tmp_path / "m.tsv", "uniref100", sub_out=tmp_path / "s.tsv")
    assert not (tmp_path / "m.tsv").exists() and not (tmp_path / "s.tsv").exists()


def test_submembers_need_uniref100(release, tmp_path):
    with pytest.raises(ValueError, match="needs level uniref100"):
        uh.xml_to_members_tsv(release.xml, tmp_path / "m.tsv", "uniref90",
                              sub_out=tmp_path / "s.tsv")


def test_submembers_hash_collisions_are_confirmed_on_exact_ids(release, tmp_path, monkeypatch):
    uh.xml_to_members_tsv(release.xml, tmp_path / "m.tsv", "uniref100", sub_out=tmp_path / "a")
    monkeypatch.setattr(uh, "_id_hash", lambda b: len(b) % 3)   # nearly every id collides
    uh.xml_to_members_tsv(release.xml, tmp_path / "m.tsv", "uniref100", sub_out=tmp_path / "b")
    assert (tmp_path / "a").read_bytes() == (tmp_path / "b").read_bytes()
    src = _write(tmp_path, _xml(_entry100("A1", [("A1", "A1"), ("D4", "D4")])
                                + _entry100("B2", [("B2", "B2"), ("C3", "A1")])))
    with pytest.raises(uh.HierarchyError, match="UniRef90_A1 in UniRef50_A1 and UniRef50_B2"):
        uh.xml_to_members_tsv(src, tmp_path / "x.tsv", "uniref100", sub_out=tmp_path / "s.tsv")
