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
    network is read. A cached file whose bytes are not the pinned ones is downloaded again, once, as ``pypi.fetch`` and
    ``github.fetch`` do; a file that is still not is refused, by its path in the cache.
    """
    if not pinned.COMMIT_ID.fullmatch(revision):
        raise ValueError(f"{repo}: {revision!r} is not a commit id (40 lowercase hex digits); pin a commit")
    import huggingface_hub

    def download(again: bool) -> Path:
        # token=False: the files are public, and an importer that needs no token reads none; with the default, the
        # call would look one up (HF_TOKEN, then the token file, refreshing a browser login) and send it on every
        # request.
        path = huggingface_hub.hf_hub_download(
            repo,
            filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache / "huggingface",
            force_download=again,
            token=False,
        )
        return Path(path)

    path = download(again=False)
    if pinned.sha256_of_file(path) != sha256:
        path = download(again=True)
    pinned.check(path, pinned.sha256_of_file(path), sha256)
    return path


def card_license(card: str) -> object:
    """The ``license`` of a dataset card's YAML front matter, as huggingface_hub reads a card; None when it states none.

    The front matter is read with ``huggingface_hub.DatasetCard``, the card reader of the Hub's own client, which
    parses it with PyYAML: a ``license`` nested under another key, or written after the front matter, is not the card's.
    The value is what YAML holds: a string, or a list for a card that states several. A card that opens with ``---``
    but in which DatasetCard finds no front matter is refused: read as stating none, it would pass where a card that
    states none was reviewed.
    """
    from huggingface_hub import DatasetCard

    read = DatasetCard(card)
    if read.text == card and card.lstrip().startswith("---"):
        raise ValueError("the card opens with --- but huggingface_hub finds no front matter in it; review it by hand")
    return read.data.license


def check_card_license(repo: str, card: str, reviewed: str | None) -> None:
    """Refuse a card whose license is not the one the importer was reviewed for (None: a card that states none)."""
    found = card_license(card)
    if found != reviewed:
        raise ValueError(
            f"{repo} {CARD}: the card's license is {found!r}, not the reviewed {reviewed!r}; review it before importing"
        )
