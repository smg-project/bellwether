"""Every import reads the license file it pins from the copy next to its sets before the cache or the network.

Each importer that pins a license file on GitHub (``LICENSE_SHA256``) writes its copy under the corpus root
(``LICENSE_COPY``) and hands ``github.fetch`` that copy, which serves it when its sha256 is the pinned one: on a cold
cache ``bellwether import <name> --check`` fetches no license, and a repository that leaves GitHub, moves the file or
answers 5xx stops no check. The copies are read here through the real ``fetch`` with the test suite's network refusal
in place, as ``test_import_swebench.py`` and ``test_import_swehero.py`` read their tables', so a copy that goes missing
or differs is caught by pytest rather than by a fetch on CI. The GSM8K import then runs against a stand-in for the
raw-file host that answers 503 to every request: it and its check pass from the copy without a request, and a copy that
holds other bytes is fetched again and named. Every other importer hands over its copy the same way, which its own
command test asserts.
"""

from __future__ import annotations

import hashlib
import http.server
import importlib
import json
import pkgutil
import threading
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from bellwether import importers
from bellwether.cli import main
from bellwether.importers import github, gsm8k

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
RAW = "https://raw.githubusercontent.com"
Pin = tuple[str, str, str, str, str]  # owner, repo, commit, path, sha256


def license_pins() -> dict[str, tuple[Pin, str]]:
    """Each importer's license pin with the copy it writes under the corpus root, by module name: every importer with
    a ``LICENSE_COPY`` and a ``LICENSE_SHA256``. SWE-bench and SWE-Hero pin theirs in tables, read in their own
    tests."""
    pins = {}
    for info in pkgutil.iter_modules(importers.__path__):
        module = importlib.import_module(f"{importers.__name__}.{info.name}")
        if not hasattr(module, "LICENSE_COPY") or not hasattr(module, "LICENSE_SHA256"):
            continue
        if hasattr(module, "LICENSE_OWNER"):  # a dataset elsewhere, its license file from a repository on GitHub
            where = (module.LICENSE_OWNER, module.LICENSE_REPO, module.LICENSE_COMMIT, module.LICENSE_PATH)
        else:  # a dataset in a repository on GitHub, its license file at the same commit
            where = (module.OWNER, module.REPO, module.COMMIT, module.LICENSE_FILE)
        pins[info.name] = ((*where, module.LICENSE_SHA256), module.LICENSE_COPY)
    return pins


PINS = license_pins()


def test_every_importer_that_copies_a_pinned_license_is_read():
    assert {"bfcl", "chartqa", "coco", "fleurs", "gsm8k", "hermes", "mgsm", "tau2"} <= set(PINS)
    assert all(relative.startswith("licenses/") for _, relative in PINS.values())


@pytest.mark.parametrize("name", sorted(PINS))
def test_the_committed_copy_holds_the_pinned_bytes_so_the_check_fetches_nothing(name, tmp_path):
    # No network in tests (conftest): a copy that is missing or differs would send the import to GitHub and fail here.
    pin, relative = PINS[name]
    assert github.fetch(*pin, cache=tmp_path / "cache", copy=CORPUS / relative) == CORPUS / relative
    assert not (tmp_path / "cache").exists()


class RawHost(http.server.ThreadingHTTPServer):
    """Answers every GET with ``status``, ``body`` behind a 200, and keeps each request's path."""

    block_on_close = False

    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.status, self.body = status, body
        self.requests: list[str] = []


class _Handler(http.server.BaseHTTPRequestHandler):
    server: RawHost

    def do_GET(self) -> None:
        self.server.requests.append(self.path)
        if self.server.status != 200:
            self.send_error(self.server.status)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.server.body)))
        self.end_headers()
        self.wfile.write(self.server.body)

    def log_message(self, *args: object) -> None:
        pass  # keep pytest's output to the tests


@contextmanager
def raw_host(monkeypatch: pytest.MonkeyPatch, status: int, body: bytes = b"") -> Iterator[RawHost]:
    """A ``RawHost`` that ``github.fetch`` reaches in place of the raw-file host, through the real ``urlopen`` with the
    host rewritten (``test_import_github.py`` reads its faults the same way)."""
    server = RawHost(status, body)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no proxy of the environment on the way
    host, port = server.server_address[:2]

    def urlopen(url: str, timeout: float):
        assert url.startswith(f"{RAW}/")
        return direct.open(f"http://{host}:{port}" + url.removeprefix(RAW), timeout=timeout)

    monkeypatch.setattr(github, "urlopen", urlopen)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


ROWS = {
    "train": {"question": "Two apples and two more?", "answer": "2 + 2 = <<2+2=4>>4\n#### 4"},
    "test": {"question": "Three pears and one more?", "answer": "3 + 1 = <<3+1=4>>4\n#### 4"},
}
LICENSE_REQUEST = f"/{gsm8k.OWNER}/{gsm8k.REPO}/{gsm8k.COMMIT}/{gsm8k.LICENSE_FILE}"


@pytest.fixture
def gsm8k_import(tmp_path, monkeypatch) -> tuple[list[str], Path, Path]:
    """The GSM8K import ready to run: its data files in the cache, one row each, and its LICENSE not; the corpus holds
    the committed copy, whose sha256 is the pinned one. Returns the command's arguments, the cached LICENSE's path and
    the copy."""
    cache, corpus = tmp_path / "cache", tmp_path / "corpus"
    cached = cache / "github" / gsm8k.OWNER / gsm8k.REPO / gsm8k.COMMIT
    pins = {}
    for split, row in ROWS.items():
        target = cached / gsm8k.data_file(split)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(json.dumps(row).encode() + b"\n")
        pins[split] = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(gsm8k, "SHA256", pins)
    copy = corpus / gsm8k.LICENSE_COPY
    copy.parent.mkdir(parents=True)
    copy.write_bytes((CORPUS / gsm8k.LICENSE_COPY).read_bytes())
    monkeypatch.setattr(github, "WAITS", (0, 0, 0))  # a read made again waits nothing here: a regression fails fast
    return ["import", "gsm8k", "--corpus", str(corpus), "--cache", str(cache)], cached / gsm8k.LICENSE_FILE, copy


@pytest.mark.loopback
def test_an_import_and_its_check_read_the_license_from_the_copy_while_the_raw_file_host_answers_503(
    gsm8k_import, monkeypatch, capsys
):
    argv, cached, copy = gsm8k_import
    with raw_host(monkeypatch, 503) as host:
        assert main(argv) == 0
        assert main([*argv, "--check"]) == 0
    assert host.requests == [] and not cached.exists()
    assert f"the GSM8K sets equal a fresh import of {gsm8k.SOURCE}" in capsys.readouterr().out


@pytest.mark.loopback
def test_a_copy_that_holds_other_bytes_is_fetched_and_the_check_names_it(gsm8k_import, monkeypatch, capsys):
    argv, cached, copy = gsm8k_import
    pinned_text = copy.read_bytes()
    with raw_host(monkeypatch, 200, pinned_text) as host:
        assert main(argv) == 0 and host.requests == []  # the sets written, the LICENSE from the copy
        copy.write_bytes(b"MIT License\n\nother\n")
        assert main([*argv, "--check"]) == 1
    assert host.requests == [LICENSE_REQUEST] and cached.read_bytes() == pinned_text
    assert capsys.readouterr().err == f"{copy}: differs from a fresh import\n"
    assert copy.read_bytes() == b"MIT License\n\nother\n"  # the check writes nothing
