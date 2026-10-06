"""Dataset files pinned on the Hugging Face Hub: read through a cache, and checked against their sha256 on every use.

``fetch`` reads a file with ``hf_hub_download`` into ``<cache>/huggingface``, the Hugging Face cache layout under the
importers' own cache (``bellwether import --cache``). Given a commit id as the revision, it serves a file already there
without a request for it, and ``HF_HUB_OFFLINE=1`` forbids the network outright. It does not check a cached file's
bytes, so ``fetch`` does, on every read, against the sha256 pinned in the importer.

huggingface_hub 1.33 keeps the bytes of a file stored with Xet once for the whole cache, in
``<cache>/huggingface/blobs/<2 hex>/<xet hash>``; the dataset's own folder (``datasets--<org>--<name>``) holds only
links to them. A CI cache of that folder alone keeps the links and loses the bytes, so CI caches ``--cache`` whole, in
the one step that also keeps the PyPI and GitHub files.

A dataset card (``README.md``) states its license in the YAML front matter; an importer pins the card by sha256 and
checks that the license is still the one it was reviewed for.
"""

from __future__ import annotations

from pathlib import Path

from . import pinned

CARD = "README.md"


def fetch(repo: str, revision: str, filename: str, sha256: str, cache: Path = pinned.CACHE) -> Path:
    """The path of ``filename`` in the dataset ``repo`` at commit ``revision``, kept under ``cache``, checked against
    ``sha256``.

    ``revision`` must be a full commit id, 40 lowercase hex digits, as ``github.fetch`` asks: a branch or a tag can
    move, and would send a request on every run to learn where it points. Any other is refused before the cache or the
    network is read.
    """
    if not pinned.COMMIT_ID.fullmatch(revision):
        raise ValueError(f"{repo}: {revision!r} is not a commit id (40 lowercase hex digits); pin a commit")
    import huggingface_hub

    # token=False: the files are public, and an importer that needs no token reads none; with the default, the call
    # would look one up (HF_TOKEN, then the token file, refreshing a browser login) and send it on every request.
    path = huggingface_hub.hf_hub_download(
        repo, filename, repo_type="dataset", revision=revision, cache_dir=cache / "huggingface", token=False
    )
    path = Path(path)
    pinned.check(f"{repo}@{revision} {filename}", pinned.sha256_of_file(path), sha256)
    return path


def card_license(card: str) -> str | None:
    """The top-level ``license`` of the card's YAML front matter, or None when it states none.

    A ``license`` nested under another key, or written after the front matter, does not count. Lines end at "\\n"
    only, as YAML's do.
    """
    lines = card.split("\n")
    if lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("license:"):
            return line.split(":", 1)[1].strip()
    return None


def check_card_license(repo: str, card: str, reviewed: str | None) -> None:
    """Refuse a card whose license is not the one the importer was reviewed for (None: a card that states none)."""
    found = card_license(card)
    if found != reviewed:
        raise ValueError(
            f"{repo} {CARD}: the card's license is {found!r}, not the reviewed {reviewed!r}; review it before importing"
        )
