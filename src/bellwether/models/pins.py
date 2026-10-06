"""The engine files the list is read from, each pinned to one commit, and the cache that keeps them.

A file at a commit never changes, so a copy taken from a local checkout with ``git show`` and one
downloaded from GitHub are the same bytes. Either goes into the cache under its commit, which is
also where ``bellwether gaps`` keeps registry files, and every later run reads it there offline.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from bellwether.gaps.fetch import write_whole


@dataclass(frozen=True)
class Pin:
    engine: str
    repo: str
    ref: str  # the tag or short commit the design cites
    commit: str
    files: tuple[str, ...]


# The pages SGLang's docs group under "Text Generation", and whether each lists multimodal models.
# The other supported-models pages list embedding, reranking, classification and reward models.
SGLANG_PAGES = {
    "docs/docs/supported-models/generative_models.mdx": False,
    "docs/docs/supported-models/multimodal_language_models.mdx": True,
    "docs/docs/supported-models/diffusion_language_models.mdx": False,
}

VLLM = Pin(
    "vllm", "vllm-project/vllm", "v0.31.0", "db9527a46873454610df6dbedf79a36d6bf1a7f6", ("tests/models/registry.py",)
)
SGLANG = Pin(
    "sglang", "sgl-project/sglang", "7d22b7a8", "7d22b7a8750f53a04e41a5a5671f9a56ab6cd001", tuple(SGLANG_PAGES)
)


class PinError(RuntimeError):
    """A pinned file could not be had; the message says what to do instead."""


def pinned_root(pin: Pin, cache: Path, checkout: Path | None = None, client: httpx.Client | None = None) -> Path:
    """A directory laid out like a checkout that holds the pin's files at its commit.

    Files missing from the cache come from ``checkout`` when one is given, else from GitHub.
    """
    root = cache / pin.engine / pin.commit
    missing = [rel for rel in pin.files if not (root / rel).is_file()]
    if missing and checkout is not None:
        for rel in missing:
            write_whole(root / rel, _from_checkout(pin, checkout, rel))
    elif missing:
        own_client = client is None
        client = client or httpx.Client(timeout=30, follow_redirects=True)
        try:
            for rel in missing:
                write_whole(root / rel, _from_github(pin, client, rel))
        finally:
            if own_client:
                client.close()
    return root


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
