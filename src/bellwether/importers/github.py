"""Files pinned on GitHub by commit: downloaded once into a cache, and checked against their sha256 before any use.

A file is read from ``raw.githubusercontent.com`` at a commit, never a branch or a tag, so the bytes behind a pin cannot
move: ``fetch`` refuses any ref that is not a full commit id. This module imports nothing beyond the standard library.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from urllib.request import urlopen

CACHE = Path.home() / ".cache" / "bellwether" / "datasets"
# A full commit id, as GitHub writes it. A branch, a tag or a short id is not one.
COMMIT_ID = re.compile(r"[0-9a-f]{40}")


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(owner: str, repo: str, commit: str, path: str, sha256: str, cache: Path = CACHE) -> Path:
    """The cached copy of ``path`` in ``owner/repo`` at ``commit``, its bytes checked against ``sha256``.

    ``commit`` must be a full commit id, 40 lowercase hex digits; any other ref is refused before the cache or the
    network is read.
    """
    if not COMMIT_ID.fullmatch(commit):
        raise ValueError(f"{owner}/{repo}: {commit!r} is not a commit id (40 lowercase hex digits); pin a commit")
    target = cache / "github" / owner / repo / commit / path
    if target.is_file() and sha256_of(target.read_bytes()) == sha256:
        return target
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"
    with urlopen(url, timeout=300) as response:
        data = response.read()
    digest = sha256_of(data)
    if digest != sha256:
        raise ValueError(f"{url}: sha256 {digest} is not the pinned {sha256}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, target)
    return target
