"""Files pinned on GitHub by commit: downloaded once into a cache, and checked against their sha256 before any use.

A file is read from ``raw.githubusercontent.com`` at a commit, never a branch or a tag, so the bytes behind a pin cannot
move: ``fetch`` refuses any ref that is not a full commit id. A repository can still leave GitHub, so a copy the caller
holds from the pin (an import's copy next to its sets) stands in for the download when its sha256 is the pinned one.
A read that fails for a passing reason (a 5xx answer, a connection that stalls or is reset, a body cut short) is made
again after each of ``WAITS`` seconds, ``ATTEMPTS`` reads in all, before its error is raised, so one hiccup of the host
does not fail an import that reads hundreds of files; a 4xx answer is raised at once, as a file that is not there does
not come back. This module imports nothing beyond the standard library.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from bellwether import storage

from . import pinned

TIMEOUT = 300  # seconds, to connect and for each read of the body
WAITS = (2, 8, 30)  # seconds before each further read of a file whose read failed for a passing reason
ATTEMPTS = len(WAITS) + 1  # the reads a file gets before its last error is raised
# What a read raises when its fault passes: a connection refused, stalled or reset, a body cut short. A 5xx answer, an
# ``HTTPError``, passes as well; a 4xx answer does not.
PASSING = (URLError, TimeoutError, ConnectionResetError, IncompleteRead)


def fetch(
    owner: str,
    repo: str,
    commit: str,
    path: str,
    sha256: str,
    cache: Path = pinned.CACHE,
    copy: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """The cached copy of ``path`` in ``owner/repo`` at ``commit``, its bytes checked against ``sha256``.

    ``commit`` must be a full commit id, 40 lowercase hex digits; any other ref is refused before the cache or the
    network is read. ``copy``, when given, is a file written from this pin before, such as an import's copy next to
    its sets: it is returned as it is when its sha256 is the pinned one, before the cache or the network is read, so
    the pin resolves after its repository has left GitHub; a copy that is missing or holds other bytes is passed over.
    ``sleep`` waits between the reads of a file whose read failed for a passing reason; a test gives one that does not.
    """
    if not pinned.COMMIT_ID.fullmatch(commit):
        raise ValueError(f"{owner}/{repo}: {commit!r} is not a commit id (40 lowercase hex digits); pin a commit")
    if copy is not None and copy.is_file() and pinned.sha256_of_file(copy) == sha256:
        return copy
    target = cache / "github" / owner / repo / commit / path
    if target.is_file() and pinned.sha256_of_file(target) == sha256:
        return target
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"
    data = _read_patiently(url, sleep)
    pinned.check(url, pinned.sha256_of(data), sha256)
    return storage.write_whole(target, data)


def _read_patiently(url: str, sleep: Callable[[float], None]) -> bytes:
    """The bytes at ``url``, read again after each of ``WAITS`` while the read fails for a passing reason.

    The last read's error is raised; a 4xx answer's at once. Each read made again is named on stderr with its cause,
    so a host that falters shows in the log of a job that still passes.
    """
    for attempt, wait in enumerate(WAITS, start=1):
        try:
            return _read(url)
        except HTTPError as err:
            if err.code < 500:
                raise
            cause = f"HTTP {err.code} {err.reason}"
        except PASSING as err:
            cause = str(getattr(err, "reason", None) or err) or type(err).__name__
        print(f"{url}: read {attempt} of {ATTEMPTS} failed ({cause}); reading again in {wait} s", file=sys.stderr)
        sleep(wait)
    return _read(url)


def _read(url: str) -> bytes:
    with urlopen(url, timeout=TIMEOUT) as response:
        return response.read()
