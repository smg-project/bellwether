"""Files pinned on GitHub by commit: downloaded once into a cache, and checked against their sha256 before any use.

A file is read from ``raw.githubusercontent.com`` at a commit, never a branch or a tag, so the bytes behind a pin cannot
move: ``fetch`` refuses any ref that is not a full commit id. This module imports nothing beyond the standard library.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from . import pinned


def fetch(owner: str, repo: str, commit: str, path: str, sha256: str, cache: Path = pinned.CACHE) -> Path:
    """The cached copy of ``path`` in ``owner/repo`` at ``commit``, its bytes checked against ``sha256``.

    ``commit`` must be a full commit id, 40 lowercase hex digits; any other ref is refused before the cache or the
    network is read.
    """
    if not pinned.COMMIT_ID.fullmatch(commit):
        raise ValueError(f"{owner}/{repo}: {commit!r} is not a commit id (40 lowercase hex digits); pin a commit")
    target = cache / "github" / owner / repo / commit / path
    if target.is_file() and pinned.sha256_of_file(target) == sha256:
        return target
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"
    with urlopen(url, timeout=300) as response:
        data = response.read()
    pinned.check(url, pinned.sha256_of(data), sha256)
    return pinned.write(target, data)
