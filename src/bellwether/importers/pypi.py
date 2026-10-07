"""Files pinned on PyPI: downloaded once into a cache, and checked against their sha256 before any use."""

from __future__ import annotations

from pathlib import Path

import httpx

from bellwether import storage

from . import pinned


def fetch(project: str, version: str, filename: str, sha256: str, cache: Path = pinned.CACHE) -> Path:
    """The cached path of ``filename`` from ``project==version`` on PyPI, its bytes checked against ``sha256``."""
    path = cache / filename
    if path.is_file() and pinned.sha256_of_file(path) == sha256:
        return path
    listing = httpx.get(f"https://pypi.org/pypi/{project}/{version}/json", timeout=60)
    listing.raise_for_status()
    urls = {entry["filename"]: entry["url"] for entry in listing.json()["urls"]}
    if filename not in urls:
        raise ValueError(f"{project}=={version} on PyPI has no file {filename}")
    download = httpx.get(urls[filename], timeout=300, follow_redirects=True)
    download.raise_for_status()
    pinned.check(filename, pinned.sha256_of(download.content), sha256)
    return storage.write_whole(path, download.content)
