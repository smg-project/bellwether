"""The vendor-code oracle: a checkpoint whose tokenizer or prompt format is the vendor's own code, run in a sandbox.

A checkpoint may ship a prompt encoder in place of a chat template: a module under ``encoding/`` whose
``encode_messages`` renders the chat format (DeepSeek V3.2's ``encoding_dsv32.py``, V4's ``encoding_dsv4.py``). The
engine's server renders such a checkpoint with its own copy of the module through a tokenizer mode picked by the
architecture (``ENCODER_RENDERERS``), never with a template. The vendor-code oracle runs the checkpoint's own module
(``prompt_encoder``), inside the sandbox only, and calls it as the engine's mode does (``ENCODER_RULES``, one rule per
mode, each the mirror of the engine's wrapper at v0.31.0), on the messages as the engine's renderer hands them to the
mode (``as_the_renderer_hands_them``: ``parse_chat_messages`` in the mode's content format, then
``_postprocess_messages``); the text is tokenized with the checkpoint's tokenizer and no special tokens added, as the
mode tokenizes it.

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

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from bellwether.inputs import ENCODER_MODES, checkpoint_dir, vendor_encoder

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


# The engine's tokenizer mode per architecture whose checkpoint ships a prompt encoder (``vllm/config/model.py:
# 709-717`` at v0.31.0): each mode wraps the checkpoint's tokenizer and renders with the engine's copy of the encoder.
ENCODER_RENDERERS = ENCODER_MODES


def engine_renderer(architectures: list[str]) -> str | None:
    """The engine's own renderer for a vendor-code checkpoint whose ``config.json`` names ``architectures``, as the
    engine's server picks it (``vllm/config/model.py:702-719`` at v0.31.0): ``"kimi_k3"`` for a Kimi K3 checkpoint,
    the tokenizer mode of ``ENCODER_RENDERERS`` for a checkpoint with a prompt encoder; None for every other, which
    the server renders through the class with the kwargs as the request gives them."""
    if KIMI_K3_ARCHITECTURE in architectures:
        return "kimi_k3"
    return next((ENCODER_RENDERERS[name] for name in architectures if name in ENCODER_RENDERERS), None)


def encoder_kwargs(kwargs: dict, reasoning_effort: object = None) -> dict:
    """``kwargs`` as ``build_chat_params`` (``vllm/entrypoints/openai/chat_completion/protocol.py:568-596`` at v0.31.0)
    hands them to one of the engine's encoder tokenizer modes for a request whose top-level ``reasoning_effort`` is
    ``reasoning_effort``: the top-level value joins the template kwargs over one of the same name and, when the
    request's kwargs name no ``enable_thinking``, sets it to ``reasoning_effort != "none"``. The request's own objects
    are never written to; the generation-prompt flags pass through, and no encoder mode reads them."""
    kwargs = dict(kwargs)
    if reasoning_effort is not None:
        kwargs.setdefault("enable_thinking", reasoning_effort != "none")
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


# The part types the engine's string content format writes as their text, joined with a newline
# (``vllm/entrypoints/chat_utils.py:1814-1853`` at v0.31.0); any other part takes the model's own placeholder there.
_TEXT_PARTS = ("text", "input_text", "output_text")


def as_the_renderer_hands_them(messages: list, content_format: str) -> list:
    """A deep copy of ``messages`` as the engine's renderer for a prompt-encoder checkpoint hands them to its tokenizer
    mode (``vllm/renderers/deepseek_v32.py:38-52`` and ``deepseek_v4.py:44-58`` at v0.31.0: ``parse_chat_messages`` in
    the mode's ``content_format``, then ``_postprocess_messages``, ``chat_utils.py:2084-2158``): under the ``string``
    format a list content becomes its text parts joined with a newline (a part of another type is refused: the engine
    writes the model's placeholder there, which the oracle does not derive); under ``openai`` the list stays. An
    assistant's empty ``tool_calls`` is dropped; a call whose arguments are a string gets the JSON text of the object
    they decode to (anything that does not decode to an object becomes ``{}``, as the engine coerces it), which the
    vendor's module decodes again; an object-form argument stays as given (the engine takes it as it is, the vendor's
    modules read a string and refuse it: the refusal is recorded as such)."""
    shaped = []
    for message in copy.deepcopy(messages):
        content = message.get("content")
        if content_format == "string" and isinstance(content, list):
            texts = []
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind not in _TEXT_PARTS:
                    raise ValueError(
                        f"the engine's string content format writes a {kind!r} part as the model's placeholder, which "
                        "the prompt-encoder oracle does not derive"
                    )
                texts.append(str(part.get("text", "")))
            message["content"] = "\n".join(texts)
        calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(calls, list):
            if not calls:
                del message["tool_calls"]
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or call.get("type", "function") != "function":
                    raise ValueError(
                        "the engine takes assistant tool_calls of type 'function' only; the request is refused"
                    )
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        decoded = json.loads(arguments)
                    except ValueError:
                        decoded = {}
                    function["arguments"] = json.dumps(decoded if isinstance(decoded, dict) else {}, ensure_ascii=False)
        shaped.append(message)
    return shaped


def deepseek_v32(module, messages: list, tools: list | None, kwargs: dict, last_role: str | None = None) -> str:
    """``encode_messages`` as the ``deepseek_v32`` mode calls it (``vllm/tokenizers/deepseek_v32.py:22-57``) on the
    messages its renderer hands it (``as_the_renderer_hands_them``, the ``string`` content format): thinking on under
    ``thinking`` or ``enable_thinking``, off by default; the request's tools on a new leading system message; earlier
    reasoning dropped when the conversation's last message is a user turn (``last_role``, the request's, for the round
    trip's renders of the appended turn); the generation-prompt flags are not read."""
    thinking = bool(kwargs.get("thinking", False) or kwargs.get("enable_thinking", False))
    messages = as_the_renderer_hands_them(messages, "string")
    drop_thinking = (last_role or messages[-1]["role"]) == "user"
    if tools:
        messages.insert(0, {"role": "system", "tools": copy.deepcopy(tools)})
    mode = "thinking" if thinking else "chat"
    return module.encode_messages(messages, thinking_mode=mode, drop_thinking=drop_thinking)


def deepseek_v4(module, messages: list, tools: list | None, kwargs: dict, last_role: str | None = None) -> str:
    """``encode_messages`` as the ``deepseek_v4`` mode calls it (``vllm/tokenizers/deepseek_v4.py:22-75``): thinking on
    unless ``thinking`` or ``enable_thinking`` says off; tools on the first system message, made when there is none;
    the effort: no string gives ``high`` when thinking and none otherwise, ``none`` switches to chat, ``max`` stays,
    ``low``, ``minimal`` and ``medium`` become ``low``, anything else ``high``; ``drop_thinking`` as the request's
    kwarg, true by default. The messages as its renderer hands them (``as_the_renderer_hands_them``, the ``openai``
    content format: a list content stays a list)."""
    thinking = bool(kwargs.get("thinking")) or bool(kwargs.get("enable_thinking"))
    if "thinking" not in kwargs and "enable_thinking" not in kwargs:
        thinking = True
    mode = "thinking" if thinking else "chat"
    messages = as_the_renderer_hands_them(messages, "openai")
    if tools:
        index = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if index is None:
            messages.insert(0, {"role": "system"})
            index = 0
        messages[index]["tools"] = copy.deepcopy(tools)
    effort = kwargs.get("reasoning_effort")
    if not isinstance(effort, str):
        effort = "high" if thinking else None
    elif effort == "none":
        mode, effort = "chat", None
    elif effort == "max":
        effort = "max"
    elif effort in ("low", "minimal", "medium"):
        effort = "low"
    else:
        effort = "high"
    drop_thinking = kwargs.get("drop_thinking", True)
    return module.encode_messages(messages, thinking_mode=mode, drop_thinking=drop_thinking, reasoning_effort=effort)


def deepseek_v41(module, messages: list, tools: list | None, kwargs: dict, last_role: str | None = None) -> str:
    """``encode_messages`` as the ``deepseek_v41`` mode calls it (``vllm/tokenizers/deepseek_v41.py:20-100``): the
    messages normalized (five roles, a ``reasoning`` becomes ``reasoning_content``, text parts joined by a blank line,
    an image part by the module's placeholder); tools on the first system message, made with empty content when there
    is none; thinking on unless ``thinking`` or ``enable_thinking`` says off, ``none`` switching it off; the effort
    ``high`` by default, an int in 1..100 or a key of the module's ``REASONING_EFFORT_MAPPINGS``, else refused as the
    server refuses the request; ``drop_thinking`` as the request's kwarg, true by default."""
    conversation = []
    for message in copy.deepcopy(messages):
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"the engine's DeepSeek V4.1 renderer refuses the role {role!r}")
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                kind = block.get("type") if isinstance(block, dict) else None
                if kind in ("text", "input_text", "output_text"):
                    parts.append(block.get("text", ""))
                elif kind in ("image_url", "input_image", "image_pil"):
                    parts.append(getattr(module, "IMAGE_PLACEHOLDER", "<｜deepseek_image｜>"))
                else:
                    raise ValueError(f"the engine's DeepSeek V4.1 renderer takes text and image parts, not {kind!r}")
            message["content"] = "\n\n".join(parts)
        conversation.append(message)
    if tools:
        system = next((m for m in conversation if m["role"] == "system"), None)
        if system is None:
            system = {"role": "system", "content": ""}
            conversation.insert(0, system)
        system["tools"] = copy.deepcopy(tools)
    thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
    if "thinking" not in kwargs and "enable_thinking" not in kwargs:
        thinking = True
    effort = kwargs.get("reasoning_effort")
    if effort == "none":
        thinking, effort = False, None
    if effort is None:
        effort = "high"
    mappings = getattr(module, "REASONING_EFFORT_MAPPINGS", {})
    if not ((type(effort) is int and 1 <= effort <= 100) or (isinstance(effort, str) and effort in mappings)):
        raise ValueError(
            f"DeepSeek V4.1 reasoning_effort must be one of {', '.join(sorted(mappings))} or an int in [1, 100]; "
            f"the request gives {effort!r}"
        )
    mode = "thinking" if thinking else "chat"
    drop_thinking = kwargs.get("drop_thinking", True)
    rendered = module.encode_messages(
        conversation, thinking_mode=mode, drop_thinking=drop_thinking, reasoning_effort=effort
    )
    return rendered[0] if isinstance(rendered, tuple) else rendered


ENCODER_RULES = {"deepseek_v32": deepseek_v32, "deepseek_v4": deepseek_v4, "deepseek_v41": deepseek_v41}


class PromptEncoder:
    """A checkpoint's prompt encoder, called as the engine's tokenizer mode for the checkpoint's architecture calls
    it: ``render`` gives the prompt text for a request's messages, tools and template kwargs (``encoder_kwargs``)."""

    def __init__(self, module, renderer: str, name: str, sha256: str) -> None:
        self.module, self.renderer, self.name, self.sha256 = module, renderer, name, sha256

    def render(self, messages: list, tools: list | None, kwargs: dict, last_role: str | None = None) -> str:
        text = ENCODER_RULES[self.renderer](self.module, messages, tools, kwargs, last_role)
        if not isinstance(text, str):
            raise ValueError(f"{self.name}: encode_messages gave {type(text).__name__}, not the prompt text")
        return text


def load_module(path: Path):
    """The prompt encoder imported from its staged file, by path: the module is the vendor's, run here only."""
    name = f"bellwether_vendor_{path.stem}_{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path.name}: not importable as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "encode_messages", None)):
        raise ValueError(f"{path.name}: the module defines no encode_messages")
    return module


def prompt_encoder(model: str, revision: str, renderer: str | None) -> PromptEncoder | None:
    """The checkpoint's prompt encoder for the vendor-code oracle; None for a checkpoint that ships none (one whose
    class renders the format, Kimi-K3's). Sandbox only, and only for an architecture the engine's server has a
    tokenizer mode for: without one, how the encoder is called is unknown, and the checkpoint is refused."""
    directory = checkpoint_dir(model, revision)
    name = vendor_encoder(directory)
    if name is None:
        return None
    require_sandbox(model)
    if renderer not in ENCODER_RULES:
        raise ValueError(
            f"{model} ships a prompt encoder ({name}) for an architecture the engine's server renders with no "
            "tokenizer mode of its own; the vendor-code oracle cannot tell how the encoder is called"
        )
    path = directory / name
    return PromptEncoder(load_module(path), renderer, name, hashlib.sha256(path.read_bytes()).hexdigest())


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
