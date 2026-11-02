import io
import urllib.error

import pytest

from nanoplm.data.preparation import downloader as dl
from nanoplm.data.preparation.downloader import Downloader, DownloadError

URL = "https://example.org/file.fasta.gz"


class _Response(io.BytesIO):
    def __init__(self, body: bytes, length: int | None):
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}


def _patch_opener(monkeypatch, open_fn):
    class _Opener:
        def open(self, request):
            return open_fn(request)

    monkeypatch.setattr(dl.urllib.request, "build_opener", lambda *a: _Opener())


def test_full_body_is_written(tmp_path, monkeypatch):
    _patch_opener(monkeypatch, lambda r: _Response(b"abcdef", 6))
    out = tmp_path / "f.gz"
    Downloader(URL, out).download()
    assert out.read_bytes() == b"abcdef"


def test_no_content_length_is_accepted(tmp_path, monkeypatch):
    _patch_opener(monkeypatch, lambda r: _Response(b"abc", None))
    out = tmp_path / "f.gz"
    Downloader(URL, out).download()
    assert out.read_bytes() == b"abc"


def test_short_body_raises_and_leaves_nothing(tmp_path, monkeypatch):
    _patch_opener(monkeypatch, lambda r: _Response(b"abc", 10))
    out = tmp_path / "f.gz"
    with pytest.raises(DownloadError) as exc:
        Downloader(URL, out).download()
    msg = str(exc.value)
    assert URL in msg
    assert "got 3 of 10 bytes; the download stopped early, run again" in msg
    assert list(tmp_path.iterdir()) == []


def test_http_error_names_url(tmp_path, monkeypatch):
    def _raise(request):
        raise urllib.error.HTTPError(URL, 404, "Not Found", {}, None)

    _patch_opener(monkeypatch, _raise)
    with pytest.raises(DownloadError, match="example.org/file.fasta.gz.*404"):
        Downloader(URL, tmp_path / "f.gz").download()


def test_url_error_names_url(tmp_path, monkeypatch):
    def _raise(request):
        raise urllib.error.URLError("no route")

    _patch_opener(monkeypatch, _raise)
    with pytest.raises(DownloadError, match="example.org/file.fasta.gz.*no route"):
        Downloader(URL, tmp_path / "f.gz").download()
