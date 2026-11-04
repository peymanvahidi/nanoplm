import gzip
import shutil
from pathlib import Path
import pytest
import zstandard
from nanoplm.data.preparation.extractor import Extractor, ExtractionError


def _make_gz(tmp_path, payload=b">seq1\nMAGV\n>seq2\nVVAGM\n"):
    src = tmp_path / "in.fasta.gz"
    with gzip.open(src, "wb") as f:
        f.write(payload)
    return src, payload


def test_extracts_with_isal_when_pigz_absent(tmp_path, monkeypatch):
    """When pigz isn't on PATH, falls back to isal and produces identical output."""
    monkeypatch.setenv("PATH", str(tmp_path))  # empty PATH → no pigz
    src, payload = _make_gz(tmp_path)
    dst = tmp_path / "out.fasta"
    Extractor(input_path=src, output_path=dst).extract()
    assert dst.read_bytes() == payload


def test_extracts_with_pigz_when_present(tmp_path, monkeypatch):
    """When pigz is on PATH, uses pigz (verify by smoke — both produce identical output)."""
    if not shutil.which("pigz"):
        pytest.skip("pigz not installed in test environment")
    src, payload = _make_gz(tmp_path)
    dst = tmp_path / "out.fasta"
    Extractor(input_path=src, output_path=dst).extract()
    assert dst.read_bytes() == payload


def test_raises_on_corrupt_gz(tmp_path):
    src = tmp_path / "corrupt.fasta.gz"
    src.write_bytes(b"not a real gzip file")
    dst = tmp_path / "out.fasta"
    with pytest.raises(ExtractionError):
        Extractor(input_path=src, output_path=dst).extract()


def test_no_partial_output_on_corrupt_gz(tmp_path):
    """ExtractionError must not leave a partial output file behind."""
    src = tmp_path / "corrupt.fasta.gz"
    src.write_bytes(b"not a real gzip file")
    dst = tmp_path / "out.fasta"
    with pytest.raises(ExtractionError):
        Extractor(input_path=src, output_path=dst).extract()
    assert not dst.exists()


# ------------------------------------------------------------------ zstd


def _make_zst(tmp_path, frames=(b">seq1\nMAGV*\n", b">seq2\nVVAGM*\n")):
    """A .zst of several frames (what pzstd or `cat a.zst b.zst` writes)."""
    cctx = zstandard.ZstdCompressor()
    src = tmp_path / "in.fa.zst"
    src.write_bytes(b"".join(cctx.compress(f) for f in frames))
    return src, b"".join(frames)


@pytest.mark.parametrize("chunk", [5, 1 << 20])
def test_extracts_zst_with_zstandard_when_zstd_absent(tmp_path, monkeypatch, chunk):
    """Without zstd on PATH the zstandard package reads every frame, also when a frame
    ends inside a read chunk."""
    from nanoplm.data.preparation import extractor
    monkeypatch.setenv("PATH", str(tmp_path))  # empty PATH → no zstd
    monkeypatch.setattr(extractor, "_CHUNK_BYTES", chunk)
    src, payload = _make_zst(tmp_path)
    dst = tmp_path / "out.fa"
    Extractor(input_path=src, output_path=dst).extract()
    assert dst.read_bytes() == payload


def test_extracts_zst_with_zstd_when_present(tmp_path):
    if not shutil.which("zstd"):
        pytest.skip("zstd not installed in test environment")
    src, payload = _make_zst(tmp_path)
    dst = tmp_path / "out.fa"
    Extractor(input_path=src, output_path=dst).extract()
    assert dst.read_bytes() == payload


@pytest.mark.parametrize("zstd_on_path", [False, True])
@pytest.mark.parametrize("damage", ["truncated", "corrupt", "empty"])
def test_a_broken_zst_raises_and_leaves_no_output(tmp_path, monkeypatch, zstd_on_path, damage):
    if zstd_on_path and not shutil.which("zstd"):
        pytest.skip("zstd not installed in test environment")
    if not zstd_on_path:
        monkeypatch.setenv("PATH", str(tmp_path))
    whole = zstandard.ZstdCompressor().compress(bytes(range(256)) * 400)
    src = tmp_path / "bad.fa.zst"
    src.write_bytes({"truncated": whole[:-3], "corrupt": b"not a real zstd file",
                     "empty": b""}[damage])
    dst = tmp_path / "out.fa"
    with pytest.raises(ExtractionError):
        Extractor(input_path=src, output_path=dst).extract()
    assert not dst.exists()


def test_other_suffixes_are_refused(tmp_path):
    src = tmp_path / "in.fa.bz2"
    src.write_bytes(b"x")
    with pytest.raises(ExtractionError, match=r"\.gz or \.zst"):
        Extractor(input_path=src, output_path=tmp_path / "out.fa").extract()


def test_an_empty_gz_raises_without_pigz_too(tmp_path, monkeypatch):
    """pigz refuses a 0-byte .gz; isal would write an empty FASTA."""
    monkeypatch.setenv("PATH", str(tmp_path))
    src = tmp_path / "empty.fasta.gz"
    src.write_bytes(b"")
    dst = tmp_path / "out.fasta"
    with pytest.raises(ExtractionError, match=r"is empty \(0 bytes\)"):
        Extractor(input_path=src, output_path=dst).extract()
    assert not dst.exists()


def test_a_zst_without_zstd_or_zstandard_says_what_to_install(tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv("PATH", str(tmp_path))               # no zstd tool
    monkeypatch.setitem(sys.modules, "zstandard", None)     # import zstandard fails
    src, _ = _make_zst(tmp_path)
    dst = tmp_path / "out.fa"
    with pytest.raises(ExtractionError, match="pip install zstandard") as e:
        Extractor(input_path=src, output_path=dst).extract()
    assert "zstd" in str(e.value) and str(src) in str(e.value)
    assert not dst.exists()


# ------------------------------------------------------------------ merge (a fasta list)


def test_merge_writes_plain_gz_and_zst_in_list_order(tmp_path):
    from nanoplm.data.preparation.extractor import merge_fasta
    plain = tmp_path / "a.fa"
    plain.write_bytes(b">a1\nMK\n>a2\nMKV")                 # no newline at the end
    gz, gz_text = _make_gz(tmp_path, b">g1\nAAAA\n")
    zst, zst_text = _make_zst(tmp_path)
    dst = tmp_path / "merged.fasta"
    merge_fasta([zst, plain, gz], dst)
    assert dst.read_bytes() == zst_text + b">a1\nMK\n>a2\nMKV\n" + gz_text


@pytest.mark.parametrize("damage", ["truncated", "empty"])
def test_merge_with_a_broken_input_leaves_no_output(tmp_path, damage):
    from nanoplm.data.preparation.extractor import merge_fasta
    plain = tmp_path / "a.fa"
    plain.write_bytes(b">a1\nMK\n")
    whole = zstandard.ZstdCompressor().compress(b">z\nMKV\n" * 1000)
    bad = tmp_path / "bad.fa.zst"
    bad.write_bytes(whole[:-3] if damage == "truncated" else b"")
    dst = tmp_path / "merged.fasta"
    with pytest.raises(ExtractionError):
        merge_fasta([plain, bad], dst)
    assert not dst.exists()


def test_extract_cli_takes_several_inputs(tmp_path):
    from click.testing import CliRunner
    from nanoplm.cli.data import data
    a, b = tmp_path / "a.fa", tmp_path / "b.fasta"
    a.write_text(">a\nMK\n")
    b.write_text(">b\nMV\n")
    gz, _ = _make_gz(tmp_path, b">g\nAA\n")
    out = tmp_path / "out" / "merged.fasta"
    r = CliRunner().invoke(data, ["extract", "-i", str(b), "-i", str(gz), "-i", str(a),
                                  "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.read_text() == ">b\nMV\n>g\nAA\n>a\nMK\n"
    r = CliRunner().invoke(data, ["extract", "-i", str(a), "-i", str(b), "-o", str(a)])
    assert r.exit_code != 0 and "is the input file" in r.output
    r = CliRunner().invoke(data, ["extract", "-i", str(a), "-o", str(tmp_path / "x.fa")])
    assert r.exit_code != 0 and ".gz or .zst" in r.output
