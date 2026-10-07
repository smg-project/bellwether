"""The vendor-code oracle: a checkpoint whose tokenizer or prompt format is the vendor's own code, run in a sandbox.

Some checkpoints render or tokenize through code from their own repository: a tokenizer class that
``tokenizer_config.json`` names in ``auto_map``, such as Kimi-K3's tiktoken tokenizer, which renders its chat format
itself. bellwether runs that code only inside the sandbox ``bellwether sandbox-record`` starts. The sandbox is a
network-less container holding bellwether, the checkpoint's snapshot (checked against its manifest, every Python file
included) and the corpus, and nothing that can reach the Hugging Face token (docs/benchmark-sets.md, "Running vendor
code"). The image marks itself with ``SANDBOX_ENV``; anywhere else the oracle refuses before any vendor code is
imported.

A vendor tokenizer may have no fast ``tokenizers`` backend, so the round trip's output pieces come from an incremental
decode through the tokenizer's own ``decode`` (``incremental_pieces``), as vLLM's detokenizer reads a slow tokenizer.
"""

from __future__ import annotations

import os

SANDBOX_ENV = "BELLWETHER_SANDBOX"  # "1" inside the sandbox image only
IMAGE_ENV = "BELLWETHER_SANDBOX_IMAGE"  # the image the sandbox runs, for the provenance
SOURCE = "vendor-code"


class OutsideTheSandbox(RuntimeError):
    """The vendor-code oracle was asked for outside the sandbox."""


def require_sandbox(model: str) -> None:
    """Refuse unless this process runs inside the sandbox image, before anything of the vendor's is read."""
    if os.environ.get(SANDBOX_ENV) != "1":
        raise OutsideTheSandbox(
            f"{model}: the vendor-code oracle runs the vendor's code only inside the sandbox; record the checkpoint "
            "with `bellwether sandbox-record`, which checks its files against the manifest and starts the sandbox"
        )


def incremental_pieces(tokenizer, ids: list[int]) -> list[str]:
    """Each id's text as an incremental decode gives it: the text the id adds, or ``""`` while it ends inside a
    character, which a later id completes.

    A window of ids before the new one is decoded with it, from where the last piece started, so that a tokenizer
    whose decode depends on its left context reads it, and the cost stays linear in the output's length.
    """
    pieces: list[str] = []
    prefix = read = 0
    for end in range(1, len(ids) + 1):
        before = tokenizer.decode(ids[prefix:read], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        after = tokenizer.decode(ids[prefix:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if len(after) > len(before) and not after.endswith("\ufffd"):
            pieces.append(after[len(before) :])
            prefix, read = read, end
        else:
            pieces.append("")
    return pieces


def packages() -> dict:
    """What the provenance of a vendor-code case adds: the sandbox's image and the vendor's packages it imports."""
    import importlib.metadata

    found: dict = {"sandbox_image": os.environ.get(IMAGE_ENV, "")}
    for package in ("tiktoken", "sentencepiece", "protobuf", "mistral-common"):
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found
