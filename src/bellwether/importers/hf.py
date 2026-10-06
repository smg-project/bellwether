"""Dataset files pinned on the Hugging Face Hub: read through its cache, and checked against their sha256 on every use.

``hf_hub_download`` keeps the files in the Hugging Face cache (``HF_HUB_CACHE``). Given a commit sha as the revision,
it answers from that cache without the network once the file is there, and ``HF_HUB_OFFLINE=1`` forbids the network
outright. It does not check a cached file against the Hub's hash, so ``fetch`` does, on every read.

A dataset card (``README.md``) states its license in the YAML front matter; an importer pins the card by sha256 and
checks that the license is still the one it was reviewed for.
"""

from __future__ import annotations

from pathlib import Path

from . import pinned

CARD = "README.md"


def fetch(repo: str, revision: str, filename: str, sha256: str) -> Path:
    """The cached path of ``filename`` in the dataset ``repo`` at commit ``revision``, checked against ``sha256``."""
    import huggingface_hub

    # token=False: the files are public, and an importer that needs no token reads none; with the default, the call
    # would look one up (HF_TOKEN, then the token file, refreshing a browser login) and send it on every request.
    path = Path(huggingface_hub.hf_hub_download(repo, filename, repo_type="dataset", revision=revision, token=False))
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
