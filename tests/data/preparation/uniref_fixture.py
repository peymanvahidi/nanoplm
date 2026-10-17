"""A tiny UniRef release: uniref50.xml.gz plus uniref90 / uniref100 FASTA, with ground truth.

Shaped like the real files: each <entry> lists its <representativeMember> first, then
one <member> per UniProtKB/UniParc record, and every record carries both a UniRef90 and
a UniRef100 id property, so ids repeat inside an entry and must be deduped. The FASTA
records are in an order unrelated to the clusters, some sequences fall outside the
length window, some representatives do not survive it, and some records are wrapped
at 60 columns.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AA = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
MIN_LEN, MAX_LEN = 20, 128


@dataclass
class Release:
    xml: Path
    fasta90: Path
    fasta100: Path
    # ground truth per level: accession -> (cluster index, is_rep, sequence). The
    # UniRef100 rep is the paired one: the UniRef100 sharing the UniRef90 rep's accession.
    u90: dict = field(default_factory=dict)
    u100: dict = field(default_factory=dict)
    cluster_ids: list = field(default_factory=list)
    literal_differs: int = 0


def _seq(rng, n: int) -> str:
    return AA[rng.integers(0, len(AA), n)].tobytes().decode()


def _length(rng) -> int:
    r = rng.random()
    if r < 0.06:
        return int(rng.integers(5, MIN_LEN))           # too short
    if r < 0.12:
        return int(rng.integers(MAX_LEN + 1, 200))     # too long
    return int(rng.integers(MIN_LEN, MAX_LEN + 1))


def _fasta(path: Path, records, rng) -> None:
    out = []
    for acc, desc, seq in records:
        out.append(f">{acc} {desc}\n")
        if rng.random() < 0.3:                          # wrapped record
            out.extend(seq[i:i + 60] + "\n" for i in range(0, len(seq), 60))
        else:
            out.append(seq + "\n")
    path.write_text("".join(out))


def make_release(root: Path, n_clusters: int = 400, seed: int = 3) -> Release:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    rel = Release(root / "uniref50.xml.gz", root / "uniref90.fasta", root / "uniref100.fasta")
    entries = []
    u90_records, u100_records = [], []
    n90 = n100 = 0
    for c in range(n_clusters):
        # cluster size in UniRef90 members: ~half singletons, a few large ones
        r = rng.random()
        k90 = 1 if r < 0.5 else int(rng.integers(2, 6)) if r < 0.95 else int(rng.integers(10, 30))
        u90s = []
        for _ in range(k90):
            acc90 = f"Q{n90:07d}"
            n90 += 1
            # every UniRef90 has 1-3 UniRef100 records; the first shares the UniRef90's
            # accession and sequence (UniProt names a UniRef90 after its seed UniRef100)
            k100 = int(rng.integers(1, 4))
            u100s = [acc90]
            for _ in range(k100 - 1):
                u100s.append(f"P{n100:07d}")
                n100 += 1
            u90s.append((acc90, u100s))
        u50 = u90s[0][0] if rng.random() < 0.9 else f"R{c:07d}"
        rel.cluster_ids.append(u50)
        lines = [f'<entry id="UniRef50_{u50}" updated="2026-09-02">',
                 "<name>Cluster: Test</name>",
                 f'<property type="member count" value="{k90}"/>']
        for i, (acc90, u100s) in enumerate(u90s):
            # several UniProtKB records per UniRef90 -> repeated UniRef90 ids in the entry.
            # Sometimes the representativeMember is another UniRef100 of the rep UniRef90,
            # so the literal UniRef100 rep differs from the paired one.
            listed = list(u100s)
            if i == 0 and len(listed) > 1 and rng.random() < 0.4:
                listed = listed[1:] + listed[:1]
                rel.literal_differs += 1
            for j, acc100 in enumerate(listed):
                wrapper = "representativeMember" if (i == 0 and j == 0) else "member"
                lines += [f"<{wrapper}>",
                          f'<dbReference type="UniProtKB ID" id="{acc100}_TEST">',
                          f'<property type="UniProtKB accession" value="{acc100}"/>',
                          f'<property type="UniRef100 ID" value="UniRef100_{acc100}"/>',
                          f'<property type="UniRef90 ID" value="UniRef90_{acc90}"/>',
                          '<property type="length" value="3"/>',
                          "</dbReference>",
                          f"</{wrapper}>"]
            seq90 = _seq(rng, _length(rng))
            rel.u90[acc90] = (c, i == 0, seq90)
            u90_records.append((f"UniRef90_{acc90}", f"Test n={len(u100s)} Tax=X", seq90))
            for j, acc100 in enumerate(u100s):
                seq100 = seq90 if j == 0 else _seq(rng, _length(rng))
                rel.u100[acc100] = (c, i == 0 and j == 0, seq100)
                u100_records.append((f"UniRef100_{acc100}", "Test n=1 Tax=X", seq100))
        lines += ['<sequence length="3">MKV</sequence>', "</entry>"]
        entries.append("\n".join(lines) + "\n")
    xml = ('<?xml version="1.0" encoding="ISO-8859-1" ?>\n'
           '<UniRef50 xmlns="http://uniprot.org/uniref" releaseDate="2026-09-02" '
           'version="2026_03">\n' + "".join(entries) + "</UniRef50>\n")
    with gzip.open(rel.xml, "wb") as f:
        f.write(xml.encode())
    # FASTA order unrelated to cluster order
    _fasta(rel.fasta90, [u90_records[i] for i in rng.permutation(len(u90_records))], rng)
    _fasta(rel.fasta100, [u100_records[i] for i in rng.permutation(len(u100_records))], rng)
    return rel
