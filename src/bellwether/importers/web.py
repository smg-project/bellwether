"""Files pinned on the web by their sha256: downloaded once into a cache, and checked against it before any use.

A dataset that is not on the Hugging Face Hub, PyPI or GitHub is published as an archive at a URL with no revision to
pin, so the pin is the archive's sha256 alone: ``fetch`` keeps the file under ``<cache>/web/<sha256>/<name>``, reads
it from there while its bytes have the pinned sha256, and otherwise downloads it again, streamed to a partial file and
hashed as it comes, so an archive of hundreds of megabytes is never held in memory whole. Bytes whose sha256 is not the
pinned one are refused by their URL. This module imports nothing beyond the standard library.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from bellwether import storage

from . import pinned


def fetch(url: str, sha256: str, cache: Path = pinned.CACHE) -> Path:
    """The cached copy of the file at ``url``, its bytes checked against ``sha256``."""
    name = Path(urlsplit(url).path).name or "download"
    target = cache / "web" / sha256 / name
    if target.is_file() and pinned.sha256_of_file(target) == sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=300) as response, partial.open("wb") as handle:
            while piece := response.read(storage.READ_SIZE):
                digest.update(piece)
                handle.write(piece)
        pinned.check(url, digest.hexdigest(), sha256)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target
