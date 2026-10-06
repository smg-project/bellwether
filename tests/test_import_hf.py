import hashlib
import re

import httpx
import huggingface_hub
import pytest
from huggingface_hub import constants
from huggingface_hub.utils import _http

from bellwether.importers import hf

REPO = "org/data"
REVISION = "a" * 40
FILE = "data/rows.jsonl"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serve_file(path, calls: list):
    """A stand-in for ``hf_hub_download`` that returns ``path`` and records each call's arguments."""

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(path)

    return download


def test_fetch_asks_for_the_file_at_the_commit_in_its_cache_and_sends_no_token(tmp_path, monkeypatch):
    (tmp_path / "rows.jsonl").write_bytes(b"rows")
    calls: list = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", serve_file(tmp_path / "rows.jsonl", calls))
    assert hf.fetch(REPO, REVISION, FILE, sha(b"rows"), cache=tmp_path / "cache") == tmp_path / "rows.jsonl"
    asked = {"repo_type": "dataset", "revision": REVISION, "cache_dir": tmp_path / "cache" / "huggingface"}
    assert calls == [(REPO, FILE, {**asked, "force_download": False, "token": False})]


def test_a_file_that_is_not_the_pinned_one_is_downloaded_again_once_then_refused_by_its_path(tmp_path, monkeypatch):
    (tmp_path / "rows.jsonl").write_bytes(b"tampered")
    calls: list = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", serve_file(tmp_path / "rows.jsonl", calls))
    pinned = f"{tmp_path / 'rows.jsonl'}: sha256 {sha(b'tampered')} is not the pinned {sha(b'rows')}"
    with pytest.raises(ValueError, match=re.escape(pinned)):
        hf.fetch(REPO, REVISION, FILE, sha(b"rows"), cache=tmp_path / "cache")
    assert [kwargs["force_download"] for _, _, kwargs in calls] == [False, True]


class Hub:
    """The Hub's file endpoint for ``REPO`` at ``REVISION``, as an httpx stub transport: it serves ``files`` and
    records every request it gets, and no request leaves the process."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        prefix = f"/datasets/{REPO}/resolve/{REVISION}/"
        name = request.url.path.removeprefix(prefix)
        if not request.url.path.startswith(prefix) or name not in self.files:
            return httpx.Response(404)
        data = self.files[name]
        headers = {"X-Repo-Commit": REVISION, "ETag": f'"{sha(data)}"', "Content-Length": str(len(data))}
        return httpx.Response(200, headers=headers, content=b"" if request.method == "HEAD" else data)

    def downloads(self, name: str) -> int:
        return sum(1 for request in self.requests if request.method == "GET" and request.url.path.endswith(name))


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """The real ``hf_hub_download`` over the stub ``Hub``: huggingface_hub makes every request through the one client
    its factory builds, so the stub stands in for the network. A test that does not give ``fetch`` a cache in its own
    directory fails, rather than writing to the machine's."""
    stub = Hub()
    transport = httpx.MockTransport(stub.handle)
    monkeypatch.setattr(_http, "_GLOBAL_CLIENT_FACTORY", lambda: httpx.Client(transport=transport))
    monkeypatch.setattr(_http, "_GLOBAL_CLIENT", None)
    download = huggingface_hub.hf_hub_download

    def in_the_tests_cache(*args, cache_dir, **kwargs):
        assert cache_dir.is_relative_to(tmp_path), f"fetch was given no cache under {tmp_path}"
        return download(*args, cache_dir=cache_dir, **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", in_the_tests_cache)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "no-token-here"))
    return stub


@pytest.mark.parametrize("held", ["HF_TOKEN", "token file"])
def test_fetch_sends_no_token_where_the_machine_holds_one(tmp_path, monkeypatch, hub, held):
    if held == "HF_TOKEN":
        monkeypatch.setenv("HF_TOKEN", "hf_dummy_env_token")
    else:
        (tmp_path / "token").write_text("hf_dummy_file_token")
        monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "token"))
    hub.files[FILE] = b"rows"
    assert hf.fetch(REPO, REVISION, FILE, sha(b"rows"), cache=tmp_path / "cache").read_bytes() == b"rows"
    assert hub.downloads(FILE) == 1
    assert [request.headers.get("authorization") for request in hub.requests] == [None] * len(hub.requests)


def test_fetch_keeps_the_file_under_its_cache_and_reads_it_back_without_a_request_for_it(tmp_path, hub):
    hub.files[FILE] = b"rows"
    path = hf.fetch(REPO, REVISION, FILE, sha(b"rows"), cache=tmp_path / "cache")
    snapshot = (tmp_path / "cache").resolve() / "huggingface" / "datasets--org--data" / "snapshots" / REVISION
    assert path == snapshot / FILE and path.read_bytes() == b"rows"
    hub.requests.clear()
    assert hf.fetch(REPO, REVISION, FILE, sha(b"rows"), cache=tmp_path / "cache") == path
    assert [request for request in hub.requests if request.url.path.endswith(FILE)] == []


def test_a_cached_file_whose_bytes_are_not_the_pinned_ones_is_downloaded_again(tmp_path, hub):
    hub.files[FILE] = b"rows"
    path = hf.fetch(REPO, REVISION, FILE, sha(b"rows"), cache=tmp_path / "cache")
    path.write_bytes(b"damaged")
    assert hf.fetch(REPO, REVISION, FILE, sha(b"rows"), cache=tmp_path / "cache") == path
    assert path.read_bytes() == b"rows" and hub.downloads(FILE) == 2


def test_a_file_the_hub_serves_with_other_bytes_is_downloaded_once_more_and_refused_by_its_path(tmp_path, hub):
    hub.files[FILE] = b"tampered"
    snapshot = (tmp_path / "cache").resolve() / "huggingface" / "datasets--org--data" / "snapshots" / REVISION
    pinned = f"{snapshot / FILE}: sha256 {sha(b'tampered')} is not the pinned {sha(b'rows')}"
    with pytest.raises(ValueError, match=re.escape(pinned)):
        hf.fetch(REPO, REVISION, FILE, sha(b"rows"), cache=tmp_path / "cache")
    assert hub.downloads(FILE) == 2


def test_fetch_hashes_the_whole_file_not_its_first_mebibyte(tmp_path, hub):
    hub.files[FILE] = b"a" * 2**20 + b"tampered"
    with pytest.raises(ValueError, match="is not the pinned"):
        hf.fetch(REPO, REVISION, FILE, sha(b"a" * 2**20 + b"rows"), cache=tmp_path / "cache")


@pytest.mark.parametrize("revision", ["main", "v1.0", "a" * 39, "a" * 41, "A" * 40, "g" * 40])
def test_fetch_refuses_a_revision_that_is_not_a_commit_id_before_any_request(tmp_path, hub, revision):
    hub.files[FILE] = b"rows"
    with pytest.raises(ValueError, match=re.escape(f"{REPO}: {revision!r} is not a commit id (40 lowercase hex")):
        hf.fetch(REPO, revision, FILE, sha(b"rows"), cache=tmp_path / "cache")
    assert hub.requests == [] and not (tmp_path / "cache").exists()


CARD = "---\nconfigs:\n  - config_name: default\n    license: mit\nlicense: apache-2.0\n---\n\n# A dataset\n"


def test_card_license_is_the_top_level_license_of_the_front_matter():
    assert hf.card_license(CARD) == "apache-2.0"
    assert hf.card_license(CARD.replace("\n", "\r\n")) == "apache-2.0"


def test_a_license_line_outside_the_front_matter_or_nested_in_it_does_not_count():
    assert hf.card_license("---\nconfigs:\n  - license: mit\n---\nlicense: mit\n") is None
    assert hf.card_license("# A dataset\nlicense: mit\n") is None
    assert hf.card_license("") is None


def test_a_line_separator_inside_the_front_matter_does_not_start_a_line():
    assert hf.card_license("---\npretty_name: a license: mit\n---\n") is None


def test_check_card_license_passes_the_reviewed_license_and_refuses_any_other():
    hf.check_card_license("org/data", CARD, "apache-2.0")
    with pytest.raises(
        ValueError, match=r"org/data README.md: the card's license is 'apache-2.0', not the reviewed 'mit'"
    ):
        hf.check_card_license("org/data", CARD, "mit")


def test_check_card_license_with_none_reviewed_refuses_a_card_that_states_one():
    hf.check_card_license("org/data", "---\npretty_name: a\n---\n", None)
    with pytest.raises(ValueError, match=r"the card's license is 'apache-2.0', not the reviewed None"):
        hf.check_card_license("org/data", CARD, None)
