"""Files pinned by sha256: the cache the importers keep them in, and the checks every fetcher makes before any use.

The fetchers keep their files under ``CACHE`` (``bellwether import --cache``), each in its own layout: ``pypi`` by file
name, ``github`` under ``github/<owner>/<repo>/<commit>/``. A file's bytes are checked against the sha256 pinned in the
importer, and a download is written whole or not at all. This module imports nothing beyond the standard library.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

CACHE = Path.home() / ".cache" / "bellwether" / "datasets"
# A full commit id, as GitHub and the Hugging Face Hub write it. A branch, a tag or a short id is not one.
COMMIT_ID = re.compile(r"[0-9a-f]{40}")


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    """The sha256 of the file's bytes, read in chunks rather than whole."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check(where: str | Path, digest: str, sha256: str) -> None:
    """Refuse bytes whose sha256 is ``digest`` unless it is the pinned ``sha256``, naming ``where`` the bytes are."""
    if digest != sha256:
        raise ValueError(f"{where}: sha256 {digest} is not the pinned {sha256}")


def write(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` through ``<name>.partial``, renamed into place: a write cut short leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, path)
    return path
