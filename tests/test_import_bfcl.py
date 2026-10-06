import pytest

from bellwether.importers import pypi

WHEEL = "pkg-1.0-py3-none-any.whl"


class Response:
    def __init__(self, content: bytes = b"", payload: dict | None = None):
        self.content, self.payload = content, payload

    def raise_for_status(self):
        return self

    def json(self):
        return self.payload


def serve(data: bytes, calls: list[str]):
    def get(url, **kwargs):
        calls.append(url)
        if url.endswith("/json"):
            return Response(payload={"urls": [{"filename": WHEEL, "url": "https://files.example/pkg.whl"}]})
        return Response(content=data)

    return get


def test_fetch_uses_a_cached_wheel_whose_hash_matches_without_the_network(tmp_path, monkeypatch):
    (tmp_path / WHEEL).write_bytes(b"wheel bytes")
    monkeypatch.setattr(pypi.httpx, "get", lambda *a, **k: pytest.fail("no download expected"))
    path = pypi.fetch("pkg", "1.0", WHEEL, pypi.sha256_of(b"wheel bytes"), cache=tmp_path)
    assert path.read_bytes() == b"wheel bytes"


def test_fetch_downloads_the_named_file_and_caches_it(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(pypi.httpx, "get", serve(b"wheel bytes", calls))
    path = pypi.fetch("pkg", "1.0", WHEEL, pypi.sha256_of(b"wheel bytes"), cache=tmp_path)
    assert path == tmp_path / WHEEL and path.read_bytes() == b"wheel bytes"
    assert calls == ["https://pypi.org/pypi/pkg/1.0/json", "https://files.example/pkg.whl"]


def test_fetch_replaces_a_cached_wheel_whose_hash_does_not_match(tmp_path, monkeypatch):
    (tmp_path / WHEEL).write_bytes(b"stale")
    monkeypatch.setattr(pypi.httpx, "get", serve(b"wheel bytes", []))
    path = pypi.fetch("pkg", "1.0", WHEEL, pypi.sha256_of(b"wheel bytes"), cache=tmp_path)
    assert path.read_bytes() == b"wheel bytes"


def test_fetch_rejects_a_download_that_is_not_the_pinned_one_and_caches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(pypi.httpx, "get", serve(b"tampered", []))
    with pytest.raises(ValueError, match="is not the pinned"):
        pypi.fetch("pkg", "1.0", WHEEL, pypi.sha256_of(b"wheel bytes"), cache=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_names_a_file_pypi_does_not_have(tmp_path, monkeypatch):
    monkeypatch.setattr(pypi.httpx, "get", lambda url, **k: Response(payload={"urls": []}))
    with pytest.raises(ValueError, match="has no file pkg-1.0-py3-none-any.whl"):
        pypi.fetch("pkg", "1.0", WHEEL, "0" * 64, cache=tmp_path)
