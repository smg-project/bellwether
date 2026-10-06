"""Files pinned on PyPI: downloaded once into a cache, and checked against their sha256 before any use."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx

CACHE = Path.home() / ".cache" / "bellwether" / "datasets"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(project: str, version: str, filename: str, sha256: str, cache: Path = CACHE) -> Path:
    """The cached path of ``filename`` from ``project==version`` on PyPI, its bytes checked against ``sha256``."""
    path = cache / filename
    if path.is_file() and sha256_of(path.read_bytes()) == sha256:
        return path
    listing = httpx.get(f"https://pypi.org/pypi/{project}/{version}/json", timeout=60)
    listing.raise_for_status()
    urls = {entry["filename"]: entry["url"] for entry in listing.json()["urls"]}
    if filename not in urls:
        raise ValueError(f"{project}=={version} on PyPI has no file {filename}")
    download = httpx.get(urls[filename], timeout=300, follow_redirects=True)
    download.raise_for_status()
    digest = sha256_of(download.content)
    if digest != sha256:
        raise ValueError(f"{filename}: sha256 {digest} is not the pinned {sha256}")
    cache.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(download.content)
    os.replace(partial, path)
    return path
