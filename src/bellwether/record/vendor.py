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

The engine's server renders some vendor-code checkpoints with a renderer of its own, which rewrites the request's kwargs
before it calls the class: a Kimi K3 checkpoint (``KimiK3ForConditionalGeneration``) gets vLLM's Kimi K3 renderer
(``vllm/config/model.py:702-719`` picks the tokenizer mode ``kimi_k3`` for the architecture, ``vllm/renderers/
registry.py`` maps it to ``KimiK3Renderer``, at v0.31.0), whose ``_apply_k3_thinking_kwargs`` (``vllm/renderers/
kimi_k3.py:48-66``) turns the engine's ``enable_thinking`` alias into the class's ``thinking``, a ``reasoning_effort``
into ``thinking_effort`` (``none`` switches thinking off) and refuses an effort the class does not know, as the server
refuses the request. The oracle hands the class the kwargs as that renderer hands them (``kimi_k3_kwargs``), so the
reference is the engine's prompt and not the bare call's; the provenance names the renderer (``engine_renderer``).
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


# The architecture the engine's server renders with its Kimi K3 renderer, and the efforts that renderer admits
# (``_K3_THINKING_EFFORTS`` in ``vllm/renderers/kimi_k3.py`` at v0.31.0).
KIMI_K3_ARCHITECTURE = "KimiK3ForConditionalGeneration"
KIMI_K3_THINKING_EFFORTS = ("low", "high", "max")


def engine_renderer(architectures: list[str]) -> str | None:
    """The engine's own renderer for a vendor-code checkpoint whose ``config.json`` names ``architectures``, as the
    engine's server picks it (``vllm/config/model.py:702-719`` at v0.31.0): ``"kimi_k3"`` for a Kimi K3 checkpoint;
    None for every other, which the server renders through the class with the kwargs as the request gives them."""
    return "kimi_k3" if KIMI_K3_ARCHITECTURE in architectures else None


def kimi_k3_kwargs(kwargs: dict, reasoning_effort: object = None) -> dict:
    """``kwargs`` as the engine's Kimi K3 renderer hands them to the class for a request whose top-level
    ``reasoning_effort`` is ``reasoning_effort``.

    Two steps of the server at v0.31.0. ``build_chat_params`` (``vllm/entrypoints/openai/chat_completion/protocol.py:
    568-596``) folds the request's top-level ``reasoning_effort`` into the template kwargs over one of the same name,
    and, when the request's kwargs name no ``enable_thinking``, derives it as ``reasoning_effort != "none"``. Then
    ``_apply_k3_thinking_kwargs`` (``vllm/renderers/kimi_k3.py:48-66``): ``enable_thinking`` is taken out and becomes
    ``thinking`` unless one is given; ``reasoning_effort`` is taken out and ``none`` switches ``thinking`` off unless
    one is given, any other value becomes ``thinking_effort`` unless one is given; a ``thinking_effort`` outside the
    renderer's three is refused, as the server refuses the request with a 400. The request's own objects are never
    written to.
    """
    kwargs = dict(kwargs)
    if reasoning_effort is not None:
        kwargs.setdefault("enable_thinking", reasoning_effort != "none")
        kwargs["reasoning_effort"] = reasoning_effort
    enable_thinking = kwargs.pop("enable_thinking", None)
    if enable_thinking is not None:
        kwargs.setdefault("thinking", enable_thinking)
    effort = kwargs.pop("reasoning_effort", None)
    if effort == "none":
        kwargs.setdefault("thinking", False)
    elif effort is not None:
        kwargs.setdefault("thinking_effort", effort)
    thinking_effort = kwargs.get("thinking_effort")
    if thinking_effort is not None and thinking_effort not in KIMI_K3_THINKING_EFFORTS:
        supported = ", ".join(KIMI_K3_THINKING_EFFORTS)
        raise ValueError(f"Kimi K3 supports thinking_effort values: {supported}; the request gives {thinking_effort!r}")
    return kwargs


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
