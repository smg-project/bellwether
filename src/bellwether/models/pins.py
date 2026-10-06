"""The engine files the list is read from, each pinned to one commit and its sha256, and the cache that keeps them.

A file at a commit never changes, so a copy taken from a local checkout with ``git show`` and one
downloaded from GitHub are the same bytes. Either goes into the cache under its commit, which is
also where ``bellwether gaps`` keeps registry files, and every later run reads it there offline.
Every use checks the bytes against the pinned sha256, as the importers do with their files: a
cached file that differs is fetched again, and a fetched one that differs is refused.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from bellwether.gaps.fetch import write_whole


@dataclass(frozen=True)
class Pin:
    engine: str
    repo: str
    ref: str  # the tag or short commit the design cites
    commit: str
    files: Mapping[str, str]  # path in the repository -> the sha256 of its bytes at the commit
    dirs: Mapping[str, str] = field(default_factory=dict)  # directory -> the sha256 of its modules' sha256sum lines


# The pages SGLang's docs group under "Text Generation", and whether each lists multimodal models.
# The other supported-models pages list embedding, reranking, classification and reward models.
SGLANG_PAGES = {
    "docs/docs/supported-models/generative_models.mdx": False,
    "docs/docs/supported-models/multimodal_language_models.mdx": True,
    "docs/docs/supported-models/diffusion_language_models.mdx": False,
}

SGLANG_MODELS = "python/sglang/srt/models"
SGLANG_PROCESSORS = "python/sglang/srt/multimodal/processors"
VLLM_REGISTRY = "tests/models/registry.py"
VLLM = Pin(
    "vllm",
    "vllm-project/vllm",
    "v0.31.0",
    "db9527a46873454610df6dbedf79a36d6bf1a7f6",
    {VLLM_REGISTRY: "25a2f1e5d45d5970ff6a233dde10da6703806b285983804ce26881f28cf628f7"},
)
SGLANG = Pin(
    "sglang",
    "sgl-project/sglang",
    "7d22b7a8",
    "7d22b7a8750f53a04e41a5a5671f9a56ab6cd001",
    dict(
        zip(
            SGLANG_PAGES,
            (
                "86d7b3a71078044d1046762f5f73770fa1250414aca9da8ebb97673428d4c5c4",
                "68e0b307956271db7db6ffdbf45f6512f575574345c5574195e10e2a5d9359fa",
                "c4fb0bba849f4d35552ffc96d1d4f003da4019e351f7412375daed5ba3d026f3",
            ),
            strict=True,
        )
    ),
    {
        SGLANG_MODELS: "9581228057531b9f6bd16456c283c6f009a8547137c264cf046fde91ed77d681",  # 218 modules
        SGLANG_PROCESSORS: "2a8ab51314e77fee348c6af24843f10c32583342e37ba64f0b524621e493c288",  # 54 modules
    },
)


class PinError(RuntimeError):
    """A pinned file could not be had; the message says what to do instead."""


def pinned_root(pin: Pin, cache: Path, checkout: Path | None = None, client: httpx.Client | None = None) -> Path:
    """A directory laid out like a checkout that holds the pin's files at its commit, each its pinned bytes.

    A file the cache lacks, or holds with other bytes, comes from ``checkout`` when one is given,
    else from GitHub, and is written only once its sha256 is the pinned one. A pinned directory is
    its top-level modules, as SGLang's registry imports them, checked whole against the pin.
    """
    root = cache / pin.engine / pin.commit
    wanted = [rel for rel, digest in pin.files.items() if not _holds(root / rel, digest)]
    stale = [rel for rel, digest in pin.dirs.items() if not _holds_modules(root / rel, digest)]
    if (wanted or stale) and checkout is not None:
        for rel in wanted:
            write_whole(root / rel, _checked(pin, rel, _from_checkout(pin, checkout, rel)))
        for rel in stale:
            _write_modules(pin, root, rel, _modules_from_checkout(pin, checkout, rel))
    elif wanted or stale:
        own_client = client is None
        client = client or httpx.Client(timeout=30, follow_redirects=True)
        try:
            for rel in wanted:
                write_whole(root / rel, _checked(pin, rel, _from_github(pin, client, rel)))
            for rel in stale:
                _write_modules(pin, root, rel, _modules_from_github(pin, client, rel))
        finally:
            if own_client:
                client.close()
    return root


LISTED = ".listed"  # the modules a pinned directory was cached with, one name per line


def listed_modules(directory: Path) -> dict[str, bytes]:
    """The modules a pinned directory was cached with, by name; empty when it has not been cached."""
    listed = directory / LISTED
    if not listed.is_file():
        return {}
    return {name: (directory / name).read_bytes() for name in listed.read_text().split()}


def sha256sum(modules: Mapping[str, bytes]) -> str:
    """A directory's pin: the sha256 of the lines ``sha256sum`` prints for its modules, names in byte order.

    By hand, in the directory at the commit: ``sha256sum *.py | LC_ALL=C sort -k2 | sha256sum``.
    """
    lines = "".join(f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in sorted(modules.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def _holds_modules(directory: Path, digest: str) -> bool:
    try:
        modules = listed_modules(directory)
    except FileNotFoundError:  # a module the listing names is gone
        return False
    return bool(modules) and sha256sum(modules) == digest


def _write_modules(pin: Pin, root: Path, rel: str, modules: dict[str, bytes]) -> None:
    digest = sha256sum(modules)
    if digest != pin.dirs[rel]:
        raise PinError(
            f"{pin.repo} {rel}/ at {pin.commit} ({pin.ref}) came with sha256 {digest} over its modules, "
            f"not the pinned {pin.dirs[rel]}; nothing was written"
        )
    for name, content in modules.items():
        write_whole(root / rel / name, content)
    write_whole(root / rel / LISTED, "".join(f"{name}\n" for name in sorted(modules)).encode())


def _modules_from_checkout(pin: Pin, checkout: Path, rel: str) -> dict[str, bytes]:
    """The directory's top-level ``.py`` files at the commit, listed with ``git ls-tree``."""
    out = subprocess.run(["git", "-C", str(checkout), "ls-tree", f"{pin.commit}:{rel}"], capture_output=True, text=True)
    if out.returncode != 0:
        raise PinError(
            f"{checkout} has no {rel}/ at {pin.commit} ({pin.ref}): {out.stderr.strip()}; "
            f"fetch that commit there, or leave out --{pin.engine}-src to download the files"
        )
    names = []
    for line in out.stdout.splitlines():
        meta, name = line.split("\t", 1)
        if meta.split()[1] == "blob" and name.endswith(".py"):
            names.append(name)
    return {name: _from_checkout(pin, checkout, f"{rel}/{name}") for name in names}


def _modules_from_github(pin: Pin, client: httpx.Client, rel: str) -> dict[str, bytes]:
    """The directory's top-level ``.py`` files at the commit, listed through GitHub's contents API."""
    url = f"https://api.github.com/repos/{pin.repo}/contents/{rel}"
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):  # a CI runner's address is shared: 60 listings an hour without one
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = client.get(url, params={"ref": pin.commit}, headers=headers)
        resp.raise_for_status()
    except httpx.HTTPError as err:
        raise PinError(
            f"could not list {url} at {pin.commit}: {err}; give --{pin.engine}-src with a checkout that has it"
        ) from err
    names = sorted(item["name"] for item in resp.json() if item.get("type") == "file" and item["name"].endswith(".py"))
    return {name: _from_github(pin, client, f"{rel}/{name}") for name in names}


def _holds(path: Path, digest: str) -> bool:
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest


def _checked(pin: Pin, rel: str, content: bytes) -> bytes:
    digest = hashlib.sha256(content).hexdigest()
    if digest != pin.files[rel]:
        raise PinError(
            f"{pin.repo} {rel} at {pin.commit} ({pin.ref}) came with sha256 {digest}, not the pinned {pin.files[rel]}; "
            "nothing was written"
        )
    return content


def _from_checkout(pin: Pin, checkout: Path, rel: str) -> bytes:
    """``git show`` reads the commit itself, whatever the checkout has checked out or changed."""
    out = subprocess.run(["git", "-C", str(checkout), "show", f"{pin.commit}:{rel}"], capture_output=True)
    if out.returncode != 0:
        reason = out.stderr.decode(errors="replace").strip()
        raise PinError(
            f"{checkout} has no {rel} at {pin.commit} ({pin.ref}): {reason}; "
            f"fetch that commit there, or leave out --{pin.engine}-src to download the file"
        )
    return out.stdout


def _from_github(pin: Pin, client: httpx.Client, rel: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{pin.repo}/{pin.commit}/{rel}"
    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as err:
        raise PinError(
            f"could not download {url}: {err}; give --{pin.engine}-src with a checkout that has {pin.commit}"
        ) from err
    return resp.content
