"""``github.fetch`` reads a pinned file again after a passing fault, and only then: a 5xx answer, a connection that
stalls or is reset, a body cut short; a 4xx answer is raised at once, and the pinned sha256 still admits the bytes.

A server on the loopback interface stands in for the raw-file host; the fetch reaches it through the real ``urlopen``
with the host rewritten, so the 503 is the ``HTTPError`` urllib builds and the short body the ``IncompleteRead`` it
raises. The cache, copy and pin checks of ``fetch`` are proven in ``test_import_gsm8k.py`` with a fake ``urlopen``.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import threading
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError

import pytest

from bellwether.importers import github

RAW = "https://raw.githubusercontent.com"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
PATH = "data/simulations/run.json"
URL = f"{RAW}/owner/repo/{COMMIT}/{PATH}"
BODY = b'{"simulations": []}\n'
SHA256 = hashlib.sha256(BODY).hexdigest()
SHORT = "short"  # an answer whose body ends before its Content-Length


class RawHost(http.server.ThreadingHTTPServer):
    """Answers each GET with the next of ``answers`` (an HTTP status, or ``SHORT``) and keeps every request's path;
    once ``answers`` is spent, every GET gets the file."""

    block_on_close = False

    def __init__(self, answers: tuple) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.answers = list(answers)
        self.requests: list[str] = []

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class _Handler(http.server.BaseHTTPRequestHandler):
    server: RawHost

    def do_GET(self) -> None:
        self.server.requests.append(self.path)
        answer = self.server.answers.pop(0) if self.server.answers else 200
        if answer == 200 or answer == SHORT:
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY) * (2 if answer == SHORT else 1)))
            self.end_headers()
            self.wfile.write(BODY)  # after a short answer the connection closes with half the body owed
        else:
            self.send_error(answer, "between bytes timeout" if answer == 503 else None)

    def log_message(self, *args: object) -> None:
        pass  # keep pytest's output to the tests


@contextmanager
def raw_host(monkeypatch: pytest.MonkeyPatch, *answers: int | str) -> Iterator[RawHost]:
    """A ``RawHost`` serving ``answers``, which ``github.fetch`` reaches in place of the raw-file host."""
    server = RawHost(answers)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no proxy of the environment on the way

    def urlopen(url: str, timeout: float):
        assert url.startswith(f"{RAW}/")
        return direct.open(server.url + url.removeprefix(RAW), timeout=timeout)

    monkeypatch.setattr(github, "urlopen", urlopen)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def fetch(cache, waits: list[float], sha256: str = SHA256):
    return github.fetch("owner", "repo", COMMIT, PATH, sha256, cache=cache, sleep=waits.append)


def retry_line(attempt: int, wait: int, cause: str = "HTTP 503 between bytes timeout") -> str:
    return f"{URL}: read {attempt} of {github.ATTEMPTS} failed ({cause}); reading again in {wait} s"


@pytest.mark.loopback
def test_a_file_whose_first_read_gets_a_503_arrives_at_the_second(tmp_path, monkeypatch, capsys):
    waits: list[float] = []
    with raw_host(monkeypatch, 503) as host:
        path = fetch(tmp_path, waits)
    assert path == tmp_path / "github" / "owner" / "repo" / COMMIT / PATH and path.read_bytes() == BODY
    assert host.requests == [f"/owner/repo/{COMMIT}/{PATH}"] * 2
    assert waits == [2]
    assert capsys.readouterr().err == retry_line(1, 2) + "\n"


@pytest.mark.loopback
def test_a_body_cut_short_is_read_again(tmp_path, monkeypatch):
    waits: list[float] = []
    with raw_host(monkeypatch, SHORT) as host:
        assert fetch(tmp_path, waits).read_bytes() == BODY
    assert len(host.requests) == 2 and waits == [2]


@pytest.mark.loopback
def test_a_404_is_raised_at_once_and_nothing_is_read_again(tmp_path, monkeypatch, capsys):
    waits: list[float] = []
    with raw_host(monkeypatch, 404) as host, pytest.raises(HTTPError) as caught:
        fetch(tmp_path, waits)
    assert caught.value.code == 404
    assert len(host.requests) == 1 and waits == [] and list(tmp_path.iterdir()) == []
    assert capsys.readouterr().err == ""


@pytest.mark.loopback
def test_a_fault_on_every_read_raises_the_last_error_after_the_waits(tmp_path, monkeypatch, capsys):
    waits: list[float] = []
    with raw_host(monkeypatch, *[503] * github.ATTEMPTS) as host, pytest.raises(HTTPError) as caught:
        fetch(tmp_path, waits)
    assert caught.value.code == 503
    assert len(host.requests) == github.ATTEMPTS == 4
    assert waits == list(github.WAITS) == [2, 8, 30]
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().err.splitlines() == [retry_line(1, 2), retry_line(2, 8), retry_line(3, 30)]


@pytest.mark.loopback
def test_bytes_that_are_not_the_pinned_ones_are_refused_after_a_read_made_again(tmp_path, monkeypatch):
    waits: list[float] = []
    with raw_host(monkeypatch, 502) as host, pytest.raises(ValueError, match="is not the pinned"):
        fetch(tmp_path, waits, sha256=hashlib.sha256(b"other").hexdigest())
    assert len(host.requests) == 2 and waits == [2] and list(tmp_path.iterdir()) == []


def faulting(faults: list[BaseException], calls: list[str]):
    """A stand-in for ``urlopen`` that raises each of ``faults`` in turn, then serves the file."""

    def urlopen(url: str, timeout: float):
        calls.append(url)
        if faults:
            raise faults.pop(0)
        return io.BytesIO(BODY)

    return urlopen


@pytest.mark.parametrize(
    "fault",
    [
        URLError(TimeoutError("timed out")),  # the connection, as urlopen wraps it
        TimeoutError("The read operation timed out"),  # the body, raw from the socket
        ConnectionResetError(104, "Connection reset by peer"),
        IncompleteRead(BODY[:4], len(BODY) - 4),
    ],
    ids=["url-error", "timeout", "reset", "incomplete-read"],
)
def test_a_passing_fault_of_the_connection_is_read_past(tmp_path, monkeypatch, capsys, fault):
    calls: list[str] = []
    monkeypatch.setattr(github, "urlopen", faulting([fault], calls))
    waits: list[float] = []
    assert fetch(tmp_path, waits).read_bytes() == BODY
    assert calls == [URL, URL] and waits == [2]
    line = capsys.readouterr().err
    assert line.startswith(f"{URL}: read 1 of {github.ATTEMPTS} failed (")
    assert line.endswith("); reading again in 2 s\n")
    assert "timed out" in line or "reset" in line or "IncompleteRead" in line


@pytest.mark.parametrize("status", [400, 403, 429])
def test_a_4xx_answer_is_raised_at_once(tmp_path, monkeypatch, status):
    calls: list[str] = []
    monkeypatch.setattr(github, "urlopen", faulting([HTTPError(URL, status, "refused", {}, None)], calls))
    waits: list[float] = []
    with pytest.raises(HTTPError) as caught:
        fetch(tmp_path, waits)
    assert caught.value.code == status and calls == [URL] and waits == []
