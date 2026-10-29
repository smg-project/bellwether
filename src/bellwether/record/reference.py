"""The reference render oracle: the checkpoint's own chat template and tokenizer at a pinned revision.

This is the ``hf-template`` source of the authority order: ``transformers`` applies the template
that ships with the checkpoint, exactly as the vendor published it. What it is given from an
OpenAI-style request, and nothing else:

- ``messages`` and ``tools`` as sent;
- ``chat_template_kwargs`` as template variables (``enable_thinking`` and the like);
- ``continue_final_message`` and ``add_generation_prompt``; the latter defaults to the opposite
  of the former.

``tool_choice``, ``response_format`` and sampling parameters have no effect on the rendered
prompt in the reference, so they are kept in the request and ignored here.

With ``vendor_code``, the tokenizer is the class the checkpoint names in ``auto_map``, loaded with
``trust_remote_code``: the vendor-code oracle, which refuses outside the sandbox (``vendor``). Without it, a checkpoint
whose tokenizer is the vendor's code cannot be loaded, and nothing of the vendor's is run.

The template is resolved per request as the engine's server resolves it (``resolve_chat_template`` in
``vllm/renderers/hf.py:263-310`` at v0.31.0): a request without ``tools`` renders with the processor's template
when the engine gets one, and a request with ``tools`` with the tokenizer's (``tokenizer_config.json``'s or
``chat_template.jinja``'s). The engine gets a processor template by building the checkpoint's processor through
transformers' ``AutoProcessor`` (``_try_get_processor_chat_template``, ``:219-260``): a processor transformers knows
gives its template, the ``chat_template`` of ``chat_template.json`` (the legacy processor file, read before any
template file, ``processing_utils.py:1384-1391``); a processor that is the vendor's code is built by an engine that
runs that code, so its template counts too; a processor transformers 5.17 cannot build (a class it lacks) leaves the
engine the tokenizer's template for every request. ``processor_chat_template`` decides the same way (and refuses to
decide where the processor's backends are missing, torch, torchvision and pillow, or a package the processor imports
itself, SmolVLM's ``num2words``, all of which the engine's image ships: that is the environment's failure, not the
checkpoint's, and the refusal names the package to install when the error names one, ``missing_package``). A
tokenizer that ships no template renders with the processor's file for every request (Qwen3-Omni ships its template
only there), as before. The two templates differ on 35 of the 813 recorded checkpoints
(vision, OCR and audio families; 28 built, 5 the vendor's code, 2 not buildable), where every tool-less prompt the
engine's server writes comes from the processor's template; the provenance names the source of each line
(``template_source``), how the processor was had (``processor``) and the hash of both templates. The
tokenizer is ``AutoTokenizer``'s, except when the model config is
the vendor's class: ``AutoTokenizer`` builds the config first, which needs that code, and bellwether
never runs it outside the sandbox. Then the class is chosen from the two names ``AutoTokenizer`` reads, as it chooses
it: the ``tokenizer_class`` of ``tokenizer_config.json`` (Phi-4-multimodal's ``GPT2TokenizerFast``), unless transformers
registers a class for the config's ``model_type`` and lists the type among those whose hub class is wrong, where the
registered class wins (``phi3`` and ``phimoe`` name ``LlamaTokenizer`` and are encoded by the ``tokenizers`` backend,
the file's pipeline, as the engine encodes them). Nor does a model config of transformers' own class stop
the tokenizer when that class's validators refuse the checkpoint's values (ZAYA1's and an OLMo-3 fine-tune's rope
settings): ``AutoTokenizer`` is given a config of the same model type to choose the class by. Neither applies with
``vendor_code``: the vendor's class loads as the checkpoint names it, and renders with its own template or none
(Kimi-K3's). A vendor-code checkpoint the engine's server renders with a renderer of its own gets the request's
kwargs as that renderer hands them to the class: Kimi K3's turns the engine's ``enable_thinking`` alias into the
class's ``thinking``, a ``reasoning_effort`` (the request's kwarg or its top-level field) into ``thinking_effort``,
``none`` switching thinking off, and refuses an effort the class does not know (``vendor.kimi_k3_kwargs``, the mirror
of ``_apply_k3_thinking_kwargs`` in ``vllm/renderers/kimi_k3.py:48-66`` and of ``build_chat_params`` in
``vllm/entrypoints/openai/chat_completion/protocol.py:568-596`` at v0.31.0); the provenance names the renderer
(``engine_renderer``). The bare call rendered the engine's spellings as if absent: thinking stayed on under
``enable_thinking: false``, and ``reasoning_effort: low`` left the default ``thinking_effort=max`` directive. A
vendor-code checkpoint that ships a prompt encoder in place of a template (DeepSeek V3.2's
``encoding/encoding_dsv32.py``, V4's ``encoding_dsv4.py``; ``inputs.vendor_encoder``) is rendered by that module,
called as the engine's tokenizer mode for the architecture calls it (``vendor.prompt_encoder``,
``vendor.ENCODER_RULES``), the text encoded with no special tokens added as the mode encodes it; the provenance names
the module and its hash (``encoder_module``, ``encoder_sha256``) and the line's ``template_source`` is ``encoder``.

The template gets each message's content in the format vLLM selects for the template from its source
(``content_format``: a template that loops over a message's content takes a list of typed parts, and string content
reaches it as a one-item text part list; every other template takes the string), so a template whose two branches
differ is rendered as the engine renders it for the request as sent, which stays recorded unchanged. The provenance
names the format.

An assistant message's tool calls reach the template as the engine hands them (``as_vllm_gives_it``, the mirror of
``_postprocess_messages`` in ``vllm/entrypoints/chat_utils.py:2084-2158`` at v0.31.0, which the round trip applied to
parse cases before the render reference did): a call's ``arguments`` become the object they decode to, an object stays
one, and a missing, empty, malformed or non-object value becomes ``{}``; an empty ``tool_calls`` is dropped; a call
that is not an object or not of type ``function`` is refused, as the engine refuses the request. So a template that
writes the arguments with ``tojson``, concatenates them to a string or tests a key in them renders, or fails, as it
does on the engine, where it was given the request's JSON string before. The provenance names it
(``tool_call_arguments``).

A ``developer`` message reaches a template that names no developer role as the engine's renderer hands it
(``with_developer_as_system``, the mirror of ``safe_apply_chat_template`` in ``vllm/renderers/hf.py:800-808`` at
v0.31.0): the template's source is searched for the quoted word (``_detect_developer_role_support``, ``:510-511``);
without it every developer message becomes a system message without its ``tools`` (``_convert_developer_to_system``,
``:514-526``), and the system messages are merged into one at the front, their contents joined by a blank line
(``_consolidate_system_messages``, ``:529-566``). A template that names the role gets the message as written, and a
request without a developer message is never merged. The provenance names which (``developer_role``).

The ids of the rendered text come from the checkpoint's ``tokenizer.json`` itself, read with ``tokenizers``, when
the checkpoint ships one and the vendor's code is not in use: that file is the vendor's shipped encoder. A
transformers tokenizer class may install its own pre-tokenizer over the file's (``Qwen2Tokenizer`` splits with a
pattern that has no ``\\p{M}``, the Qwen3.5-generation ``tokenizer.json`` keeps combining marks with their letters),
and the ids then differ on scripts with combining marks although the text is the same. A checkpoint without
``tokenizer.json`` (a tiktoken or SentencePiece vocabulary) is encoded by the transformers tokenizer, as before.
So is a checkpoint whose class does not run the file at all: ``LlamaTokenizer`` builds its own pipeline (no
normalizer, a ``Metaspace`` pre-tokenizer) over a ``tokenizer.json`` that keeps the legacy ``Prepend`` normalizer, and
the ids transformers, and the engine with it, produce for such a checkpoint differ from the file's after every
special token (``<|im_start|>user`` reads ``user``, not ``\u2581user``): the reference encodes through the class
there (``class_rebuilds_the_pipeline``). The provenance names the encoder (``encoder``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bellwether.inputs import checkpoint_dir, vendor_encoder

from . import clock, vendor
from .content_format import ContentFormat, detect_content_format, in_content_format

SOURCE = "hf-template"


@dataclass
class Rendered:
    input_ids: list[int]
    text: str
    content_format: ContentFormat
    developer_role: str
    template_source: str


class HfTemplateOracle:
    def __init__(self, model: str, revision: str, vendor_code: bool = False) -> None:
        if vendor_code:
            vendor.require_sandbox(model)
        kwargs: dict = {} if Path(model).is_dir() else {"revision": revision}
        if vendor_code:
            kwargs["trust_remote_code"] = True
        self.vendor_code = vendor_code
        # The epoch the templates render at, read once here: what provenance names is what produced the render.
        self.source_date_epoch = clock.pinned_value()
        self.rendered_at = clock.pin_template_clock()
        self.tokenizer = load_tokenizer(model, revision, kwargs)
        # The engine's own renderer for the checkpoint, whose kwargs rewrite the oracle applies before the class renders
        # (``engine_kwargs``); None for a template, which the engine's server renders as the request gives the kwargs.
        self.engine_renderer = vendor.engine_renderer(architectures(model, revision)) if vendor_code else None
        # The checkpoint's prompt encoder when it ships one in place of a template (DeepSeek V3.2's, V4's): loaded in
        # the sandbox only, called as the engine's tokenizer mode for the architecture calls it; None otherwise.
        self.prompt_encoder = vendor.prompt_encoder(model, revision, self.engine_renderer) if vendor_code else None
        if not vendor_code and vendor_encoder(checkpoint_dir(model, revision)) is not None:
            raise ValueError(
                f"{model} at {revision} ships a prompt encoder the engine's server renders with; its reference is the "
                "vendor-code oracle, recorded with `bellwether sandbox-record`"
            )
        self.encoder = (
            None
            if vendor_code or class_rebuilds_the_pipeline(self.tokenizer, model, revision)
            else checkpoint_encoder(model, revision)
        )
        template = self.tokenizer.chat_template
        # The processor's template, which the engine's server renders a request without tools with when it builds
        # the checkpoint's processor (``resolve_chat_template``'s second priority), and how the processor was had;
        # none under the vendor's code, whose class renders its own format.
        self.processor_chat_template, self.processor = (
            (None, None) if vendor_code else processor_chat_template(model, revision, kwargs)
        )
        self.tokenizer_renders_the_processors = False
        if template is None and not vendor_code:
            template = processor_template(model, revision)
            if template is None:
                raise ValueError(
                    f"{model} at {revision} ships no chat template; the hf-template oracle cannot render it"
                )
            self.tokenizer.chat_template = template
            self.tokenizer_renders_the_processors = True
        if template is not None and not isinstance(template, str):
            template = json.dumps(template, sort_keys=True)
        # A vendor's class may render its chat format itself (Kimi-K3's does), with no template to hash; its code is
        # pinned by the manifest's inputs instead.
        if self.prompt_encoder is not None:
            template = None  # the encoder renders every request; a template the checkpoint ships too is not read
        self.template_sha256 = None if template is None else hashlib.sha256(template.encode("utf-8")).hexdigest()
        self.processor_template_sha256 = (
            None
            if self.processor_chat_template is None or self.tokenizer_renders_the_processors
            else hashlib.sha256(self.processor_chat_template.encode("utf-8")).hexdigest()
        )

    def render(self, request: dict) -> Rendered:
        continue_final = bool(request.get("continue_final_message", False))
        tools = request.get("tools")
        kwargs = {
            "tools": tools,
            "add_generation_prompt": bool(request.get("add_generation_prompt", not continue_final)),
            "continue_final_message": continue_final,
            **dict(request.get("chat_template_kwargs") or {}),
        }
        kwargs = self.engine_kwargs(request, kwargs)
        if self.prompt_encoder is not None:
            # The encoder renders the request as sent and the mode encodes its text with no special tokens added.
            text = self.prompt_encoder.render(request["messages"], tools, kwargs)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return Rendered([int(i) for i in ids], text, "string", "as-written", "encoder")
        content_format = self.content_format(tools)
        messages = self.for_template(request["messages"], tools)
        # The template the engine's server resolves for this request, handed over explicitly: the processor's for a
        # request without tools when the checkpoint ships one, else what the tokenizer resolves (None under a
        # vendor's class that renders its own format: then the class renders as it does).
        template = self.template_text(tools)
        if template is not None:
            kwargs["chat_template"] = template
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        if self.encoder is not None:
            ids = self.encoder.encode(str(text), add_special_tokens=False).ids
        else:
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
            ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return Rendered(
            [int(i) for i in ids], str(text), content_format, self.developer_role(tools), self.template_source(tools)
        )

    def engine_kwargs(self, request: dict, kwargs: dict) -> dict:
        """The template ``kwargs`` of ``request`` as the engine's server hands them to the checkpoint's class: rewritten
        by the engine's own renderer for the checkpoint when it has one (Kimi K3's, ``vendor.kimi_k3_kwargs``, which
        reads the request's top-level ``reasoning_effort`` too), as given for every other checkpoint."""
        if self.engine_renderer == "kimi_k3":
            return vendor.kimi_k3_kwargs(kwargs, request.get("reasoning_effort"))
        if self.prompt_encoder is not None:
            return vendor.encoder_kwargs(kwargs, request.get("reasoning_effort"))
        return kwargs

    def for_template(self, messages: list[dict], tools: list | None) -> list[dict]:
        """``messages`` as the engine hands them to the template for a request with ``tools``: each message's content
        in the format the template takes (``in_content_format``), each assistant message's tool calls as
        ``as_vllm_gives_it`` gives them, and a developer message as a system message when the template names no
        developer role (``with_developer_as_system``). The request's own objects are never written to."""
        prepared = in_content_format(messages, self.content_format(tools))
        prepared = [as_vllm_gives_it(m) if m.get("role") == "assistant" else m for m in prepared]
        return with_developer_as_system(prepared, self.template_text(tools))

    def template_text(self, tools: list | None) -> str | None:
        """The source of the template the engine's server applies to a request with ``tools``
        (``resolve_chat_template``, ``vllm/renderers/hf.py:263-310`` at v0.31.0): the processor's template when the
        request carries no tools (``tools is None``, as the engine tests it: an empty list is a request with tools)
        and the checkpoint ships one; else the tokenizer's (a checkpoint with named templates has one for requests
        with tools); None when the tokenizer resolves no template text, as a vendor's class that renders its own
        format."""
        if tools is None and self.processor_chat_template is not None:
            return self.processor_chat_template
        try:
            template = self.tokenizer.get_chat_template(None, tools=tools)
        except Exception:
            return None
        return template if isinstance(template, str) else None

    def template_source(self, tools: list | None) -> str:
        """Whose template renders a request with ``tools``: ``"processor"`` when the engine's server takes the
        processor's (a request without tools on a checkpoint that ships the file, or a tokenizer without a template
        of its own, which renders with the processor's for every request), ``"encoder"`` for a checkpoint whose
        prompt encoder renders every request, ``"tokenizer"`` otherwise."""
        if self.prompt_encoder is not None:
            return "encoder"
        if self.tokenizer_renders_the_processors or (tools is None and self.processor_chat_template is not None):
            return "processor"
        return "tokenizer"

    def content_format(self, tools: list | None) -> ContentFormat:
        """The content format vLLM selects for the template transformers applies to a request with ``tools``;
        "string" when the tokenizer resolves no template."""
        template = self.template_text(tools)
        return "string" if template is None else detect_content_format(template)

    def developer_role(self, tools: list | None) -> str:
        """How a developer message reaches the template transformers applies to a request with ``tools``:
        ``"as-written"`` when the template names the role (or the tokenizer resolves no template), ``"system"`` when
        the engine's renderer rewrites it (``with_developer_as_system``)."""
        template = self.template_text(tools)
        return "as-written" if template is None or names_developer_role(template) else "system"

    def provenance(self) -> dict:
        import jinja2
        import tokenizers
        import transformers

        found = {
            "oracle": vendor.SOURCE if self.vendor_code else "transformers.apply_chat_template",
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "jinja2": jinja2.__version__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "encoder": (
                "tokenizers:tokenizer.json"
                if self.encoder is not None
                else f"transformers:{type(self.tokenizer).__name__}"
            ),
            "tool_call_arguments": "string" if self.prompt_encoder is not None else "object",
            **({"chat_template_sha256": self.template_sha256} if self.template_sha256 else {}),
            **({"engine_renderer": self.engine_renderer} if self.engine_renderer else {}),
            **(
                {"encoder_module": self.prompt_encoder.name, "encoder_sha256": self.prompt_encoder.sha256}
                if self.prompt_encoder is not None
                else {}
            ),
            **({"processor": self.processor} if self.processor else {}),
            **(
                {"processor_chat_template_sha256": self.processor_template_sha256}
                if self.processor_template_sha256
                else {}
            ),
            **({"source_date_epoch": self.source_date_epoch} if self.source_date_epoch else {}),
        }
        if self.vendor_code:
            found |= vendor.packages()
        return found


def as_vllm_gives_it(message: dict) -> dict:
    """A copy of an assistant message as vLLM gives it to the template.

    This mirrors ``_postprocess_messages`` in ``vllm/entrypoints/chat_utils.py:2084-2158`` at v0.31.0, the release
    whose image bellwether pins: an empty ``tool_calls`` is dropped (``:2097-2100``); a call's arguments become the
    object they decode to, and anything that does not decode to a JSON object (missing, null, empty, invalid JSON, or
    JSON of another type) becomes ``{}`` (``:2119-2158``). A call that is not an object, or not of type ``function``,
    is refused as the engine refuses the request (``:2101-2117``).
    """
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message
    message = copy.deepcopy(message)
    if not tool_calls:
        del message["tool_calls"]
        return message
    for call in message["tool_calls"]:
        if not isinstance(call, dict):
            raise ValueError("assistant tool_calls entries must be objects")
        function = call.get("function")
        if call.get("type", "function") != "function" or not isinstance(function, dict):
            raise ValueError("chat completions only support assistant tool_calls of type 'function'")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        function["arguments"] = arguments if isinstance(arguments, dict) else {}
    return message


def names_developer_role(template: str) -> bool:
    """Whether a template's source names the ``developer`` role, as the engine's renderer tests it
    (``_detect_developer_role_support``, ``vllm/renderers/hf.py:510-511`` at v0.31.0): the quoted word anywhere in the
    text, a comment included."""
    return '"developer"' in template or "'developer'" in template


def with_developer_as_system(messages: list[dict], template: str | None) -> list[dict]:
    """``messages`` as the engine's renderer hands them to the template whose source is ``template``
    (``safe_apply_chat_template``, ``vllm/renderers/hf.py:800-808`` at v0.31.0): when a developer message is among them
    and the source names no developer role, each developer message becomes a system message without its ``tools``
    (``developer_as_system``) and the system messages are merged into one at the front
    (``consolidated_system_messages``). A template that names the role, or a request without a developer message, gets
    the messages as they are; so does a tokenizer that resolves no template text (None: a vendor's class rendering its
    own format), where the engine's renderer never runs."""
    if template is None or names_developer_role(template):
        return messages
    if not any(message.get("role") == "developer" for message in messages):
        return messages
    return consolidated_system_messages(developer_as_system(messages))


def developer_as_system(messages: list[dict]) -> list[dict]:
    """Each developer message as a system message without its ``tools``, the other messages as they are
    (``_convert_developer_to_system``, ``vllm/renderers/hf.py:514-526`` at v0.31.0)."""
    converted = []
    for message in messages:
        if message.get("role") == "developer":
            system = dict(message)
            system["role"] = "system"
            system.pop("tools", None)
            converted.append(system)
        else:
            converted.append(message)
    return converted


def consolidated_system_messages(messages: list[dict]) -> list[dict]:
    """The system messages merged into one at position 0, as the engine's renderer merges them after the developer
    rewrite (``_consolidate_system_messages``, ``vllm/renderers/hf.py:529-566`` at v0.31.0): nothing changes while the
    only system message is the first; otherwise every system message's content that is not empty (a parts list's texts
    joined by newlines) joins the merged message with a blank line between, as one string, and the other messages
    follow in their order."""
    contents: list[str] = []
    others: list[dict] = []
    needs_consolidation = False
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            if index > 0 or contents:
                needs_consolidation = True
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
                    elif isinstance(part, str):
                        parts.append(part)
                content = "\n".join(parts)
            if content:
                contents.append(content)
        else:
            others.append(message)
    if not needs_consolidation:
        return messages
    return [{"role": "system", "content": "\n\n".join(contents)}, *others]


def load_tokenizer(model: str, revision: str, kwargs: dict):
    """The checkpoint's tokenizer: ``AutoTokenizer``'s, or, when the model config is the vendor's class
    (``auto_map``'s ``AutoConfig``) and the vendor's code may not run, the class ``tokenizer_config.json`` names, as
    ``AutoTokenizer`` would choose it. A tokenizer that is the vendor's class itself is ``AutoTokenizer``'s to refuse,
    or, with ``trust_remote_code`` in the sandbox, to load. Outside the sandbox, a model config transformers refuses
    (``config_refused``) has ``AutoTokenizer`` choose by a config of the same model type (``config_stand_in``)."""
    from transformers import AutoTokenizer

    if not kwargs.get("trust_remote_code") and vendor_config(model, revision):
        path = checkpoint_file(model, revision, "tokenizer_config.json")
        config = json.loads(path.read_text()) if path is not None else {}
        named = config.get("tokenizer_class")
        if "auto_map" not in config and isinstance(named, str):
            return auto_tokenizer_class(named, model_type(model, revision)).from_pretrained(model, **kwargs)
    # A checkpoint that ships no config.json (Mistral's own layout): online, transformers takes the 404 for an empty
    # model config and goes on; offline, as the recorder runs, it cannot tell a file the cache marks absent from one it
    # never fetched, and refuses. The empty config it would have built loads the same tokenizer either way.
    if not Path(model).is_dir() and checkpoint_file(model, revision, "config.json") is None:
        from transformers import PreTrainedConfig

        return AutoTokenizer.from_pretrained(model, config=PreTrainedConfig(), **kwargs)
    from huggingface_hub.errors import StrictDataclassError

    try:
        return AutoTokenizer.from_pretrained(model, **kwargs)
    except (StrictDataclassError, AttributeError):
        if kwargs.get("trust_remote_code") or not config_refused(model, kwargs):
            raise
    return AutoTokenizer.from_pretrained(model, config=config_stand_in(model, revision), **kwargs)


def config_refused(model: str, kwargs: dict) -> bool:
    """Whether transformers refuses the checkpoint's model config as ``AutoTokenizer`` builds it: ``AutoConfig``, or,
    for a ``ValueError`` or ``OSError`` (a model type transformers does not know), the generic config. A refusal is a
    validator of the class rejecting the checkpoint's values: a strict-dataclass error, or an ``AttributeError`` that
    a validator raises and none wraps. Anything else leaves ``AutoTokenizer``'s error as it is."""
    from huggingface_hub.errors import StrictDataclassError
    from transformers import AutoConfig, PreTrainedConfig

    try:
        try:
            AutoConfig.from_pretrained(model, **kwargs)
        except (ValueError, OSError):
            PreTrainedConfig.from_pretrained(model, **kwargs)
    except (StrictDataclassError, AttributeError):
        return True
    except (ValueError, OSError):
        return False
    return False


def config_stand_in(model: str, revision: str):
    """The config ``AutoTokenizer`` chooses the tokenizer class by when transformers refuses the checkpoint's own: the
    default config of its model type, or the generic config for a type transformers does not know (as ``AutoTokenizer``
    falls back to), named as the checkpoint and carrying ``config.json``'s ``tokenizer_class`` and ``model_name``.
    Those and the config's class are what ``AutoTokenizer`` reads of it (``tokenization_auto.py`` at transformers
    5.17.0, lines 791-964); the tokenizer itself loads from the checkpoint's files."""
    from transformers import CONFIG_MAPPING, PreTrainedConfig

    path = checkpoint_file(model, revision, "config.json")
    values = json.loads(path.read_text()) if path is not None else {}
    model_type = values.get("model_type")
    known = isinstance(model_type, str) and model_type in CONFIG_MAPPING
    config = CONFIG_MAPPING[model_type]() if known else PreTrainedConfig()
    config.name_or_path = model
    for key in ("tokenizer_class", "model_name"):
        if isinstance(values.get(key), str):
            setattr(config, key, values[key])
    return config


# The classes ``AutoTokenizer`` never returns for a hub class name it overrides: a registered backend is the file's
# pipeline, not a class of its own (``tokenization_auto.py`` at transformers 5.17.0, lines 831-837 and 846-849).
BACKEND_CLASSES = ("TokenizersBackend", "PythonBackend", "PreTrainedTokenizerFast", "MistralCommonBackend")


def auto_tokenizer_class(named: str, model_type: str | None = None):
    """The class ``AutoTokenizer`` loads for the ``tokenizer_class`` that ``tokenizer_config.json`` names on a
    checkpoint of the model config's ``model_type``, as ``tokenization_auto.py`` chooses it at transformers 5.17.0. The
    type is read first (lines 815-860): when transformers registers a tokenizer class for it that is not the named one,
    the named class is overridden, by the registered class when the type is listed among those whose hub
    ``tokenizer_class`` is wrong (``MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS``: ``phi3``, ``phimoe``, ``deepseek_v3``,
    ``minimax_m2``...), by the ``tokenizers`` backend when the registered class is a backend. Otherwise the name decides
    (lines 875-938): a trailing ``Fast`` is dropped, the name is looked up as it is, then with ``Fast``; the Python base
    it finds for ``PreTrainedTokenizer``, which has no vocabulary of its own, becomes the ``tokenizers`` backend, and so
    does a name transformers does not ship."""
    from transformers import TokenizersBackend
    from transformers.models.auto.tokenization_auto import (
        MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS,
        TOKENIZER_MAPPING_NAMES,
        tokenizer_class_from_name,
    )

    name = named.removesuffix("Fast")
    registered = TOKENIZER_MAPPING_NAMES.get(model_type) if model_type else None
    if registered is not None and registered.removesuffix("Fast") != name:
        registered = registered.removesuffix("Fast")
        if registered not in BACKEND_CLASSES:
            chosen = registered if model_type in MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS else name
            tokenizer_class = tokenizer_class_from_name(chosen)
            if tokenizer_class is not None and tokenizer_class.__name__ not in BACKEND_CLASSES[:3]:
                return tokenizer_class
        return TokenizersBackend
    tokenizer_class = tokenizer_class_from_name(name) or tokenizer_class_from_name(f"{name}Fast")
    if tokenizer_class is None or tokenizer_class.__name__ == "PythonBackend":
        return TokenizersBackend
    return tokenizer_class


def model_type(model: str, revision: str) -> str | None:
    """The ``model_type`` of the checkpoint's ``config.json``, the key ``AutoTokenizer`` chooses the class by first;
    None without the file or the key."""
    path = checkpoint_file(model, revision, "config.json")
    value = json.loads(path.read_text()).get("model_type") if path is not None else None
    return value if isinstance(value, str) else None


def architectures(model: str, revision: str) -> list[str]:
    """The ``architectures`` of the checkpoint's ``config.json``, by which the engine's server picks a renderer of its
    own for the checkpoint (``vllm/config/model.py:702-719`` at v0.31.0); empty without the file or the key."""
    path = checkpoint_file(model, revision, "config.json")
    value = json.loads(path.read_text()).get("architectures") if path is not None else None
    return [name for name in value if isinstance(name, str)] if isinstance(value, list) else []


def class_rebuilds_the_pipeline(tokenizer, model: str, revision: str) -> bool:
    """Whether transformers encodes the checkpoint with a pipeline of the class's own rather than its
    ``tokenizer.json``: ``LlamaTokenizer`` builds its backend from the file's vocabulary with no normalizer and a
    ``Metaspace`` pre-tokenizer (``tokenization_llama.py`` at transformers 5.17.0) where the file keeps the legacy
    ``Prepend`` normalizer, so on such a checkpoint the file is not what transformers, and the engine with it, encodes
    with, and the reference encodes through the class. Every other checkpoint keeps the file as its encoder."""
    if type(tokenizer).__name__ != "LlamaTokenizer":
        return False
    from ..inputs import checkpoint_dir

    path = checkpoint_dir(model, revision) / "tokenizer.json"
    if not path.is_file():
        return False
    normalizer = json.loads(path.read_text(encoding="utf-8")).get("normalizer")
    if not isinstance(normalizer, dict):
        return False
    steps = normalizer.get("normalizers", [normalizer])
    return any(isinstance(step, dict) and step.get("type") == "Prepend" for step in steps)


def checkpoint_encoder(model: str, revision: str):
    """The checkpoint's ``tokenizer.json`` as a ``tokenizers.Tokenizer``, the vendor's shipped encoder, or None when
    the checkpoint ships none (a tiktoken or SentencePiece vocabulary). Read from the checkpoint's directory, which
    holds every oracle input the checkpoint ships, so a checkpoint without the file needs no absence marker."""
    from tokenizers import Tokenizer

    from ..inputs import checkpoint_dir

    path = checkpoint_dir(model, revision) / "tokenizer.json"
    return Tokenizer.from_file(str(path)) if path.is_file() else None


def vendor_config(model: str, revision: str) -> bool:
    """Whether the checkpoint's model config is the vendor's class, named by ``auto_map`` in ``config.json``."""
    path = checkpoint_file(model, revision, "config.json")
    auto_map = json.loads(path.read_text()).get("auto_map") if path is not None else None
    return isinstance(auto_map, dict) and "AutoConfig" in auto_map


PROCESSOR_BUILT = "built"
PROCESSOR_REMOTE_CODE = "remote-code"
PROCESSOR_NOT_BUILT = "not-built"


def built_processor(model: str, kwargs: dict):
    """What transformers' ``AutoProcessor`` builds for the checkpoint (the engine's ``cached_get_processor``): the
    processor, or the tokenizer for a checkpoint that has no processor class."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model, **kwargs)


# Where transformers' ``ProcessorMixin`` reads the processor's own template, in its order of precedence (a
# ``chat_template.jinja`` is the tokenizer's file too, so it never makes the processor's template a different one).
PROCESSOR_TEMPLATE_FILES = ("processor_config.json", "chat_template.json")

# The backends a processor build needs that transformers names when one is not installed; the message of the error.
PROCESSOR_BACKENDS = ("torch", "torchvision", "pillow", "pil ", "num2words")
# What a processor build needs beside transformers, as the engine's image ships it: the backends, and what a processor
# imports itself (SmolVLM's num2words). The refusal names the list when the error names no package.
PROCESSOR_PACKAGES = "torch, torchvision, pillow and num2words"
# How an error names the package it misses: Python's words for a module it could not import, a processor's for a
# package it asks for by name (SmolVLM's ``Package `num2words` is required to run SmolVLM processor``), transformers'
# for a backend (``requires the Torchvision library but it was not found``).
MISSING_PACKAGE = (
    re.compile(r"No module named ['\"`]?([A-Za-z0-9_][A-Za-z0-9_.-]*)"),
    re.compile(r"[Pp]ackage ['\"`]?([A-Za-z0-9_][A-Za-z0-9_.-]*)['\"`]? is required"),
    re.compile(r"requires the ([A-Za-z0-9_.-]+) library"),
)
# transformers' names for the backends whose package pip calls otherwise.
BACKEND_PACKAGES = {"pytorch": "torch", "pil": "pillow"}
# The top-level modules of the processors' own packages (PIL is pillow's): a symbol missing from one of them is the
# package at another release, the environment's failure; a symbol missing from transformers is a class it lacks.
PROCESSOR_MODULES = frozenset({"torch", "torchvision", "pil", "num2words"})


def is_missing_backend(message: str) -> bool:
    """Whether a processor build failed for a backend the environment lacks, read from the error's message: transformers
    5.17.0 says ``X requires the Torchvision library but it was not found`` (an ImportError, or a ValueError when the
    build resolves its classes) and ``Could not load any image processor class`` for a candidate it cannot import. A
    ``requires`` about anything but a backend (a config key, ``trust_remote_code``) is the checkpoint's failure, which
    the engine meets too, and is not a missing backend."""
    lower = message.lower()
    if "could not load any" in lower:
        return True
    return "requires" in lower and any(name in lower for name in PROCESSOR_BACKENDS)


def missing_package(err: BaseException) -> str | None:
    """The package the error says is missing, as pip names it, or None when it names none: the module Python could not
    import (a ``ModuleNotFoundError``'s ``name``, or ``No module named 'num2words'``), the package a processor asks for
    by name (SmolVLM's ``Package `num2words` is required to run SmolVLM processor``) or the backend transformers names
    (``requires the Torchvision library``; ``the PyTorch library`` is torch, ``the PIL library`` pillow). The top-level
    module names the package (``torchvision.transforms`` is torchvision's)."""
    name = getattr(err, "name", None)
    if not isinstance(name, str) or not name:
        message = str(err)
        found = next((match for match in (pattern.search(message) for pattern in MISSING_PACKAGE) if match), None)
        name = found.group(1) if found else None
    if not name:
        return None
    package = name.split(".")[0].lower()
    return BACKEND_PACKAGES.get(package, package)


def is_missing_dependency(err: ImportError) -> bool:
    """Whether an ``ImportError`` from a processor build is the environment's failure, not the checkpoint's: a module
    that is absent (``ModuleNotFoundError``), a package or backend the error asks for by name (``MISSING_PACKAGE``,
    ``is_missing_backend``), or a symbol missing from one of the processors' own packages (``PROCESSOR_MODULES``:
    torchvision at another release). A symbol missing from transformers or from any other module is a class the
    installed release lacks: the engine's build fails the same way and falls back to the tokenizer's template."""
    message = str(err)
    if isinstance(err, ModuleNotFoundError) or is_missing_backend(message):
        return True
    if any(pattern.search(message) for pattern in MISSING_PACKAGE):
        return True
    name = getattr(err, "name", None)
    return isinstance(name, str) and name.split(".")[0].lower() in PROCESSOR_MODULES


def backends_refusal(model: str, revision: str, err: BaseException) -> RuntimeError:
    """The refusal of a processor build that failed for the environment, not the checkpoint: it says what to install,
    the package the error names (``missing_package``) or, when it names none, the processors' list
    (``PROCESSOR_PACKAGES``), and carries the error's own words."""
    package = missing_package(err)
    return RuntimeError(
        f"{model} at {revision}: the processor's backends are missing, so whether the engine renders its template "
        f"cannot be decided; install {package or PROCESSOR_PACKAGES} ({err})"
    )


def processor_chat_template(model: str, revision: str, kwargs: dict) -> tuple[str | None, str | None]:
    """The processor's chat template as the engine's server gets it for a request without tools, and how it was had
    (``PROCESSOR_BUILT``, ``PROCESSOR_REMOTE_CODE``, ``PROCESSOR_NOT_BUILT``; None and None for a checkpoint
    without one).

    ``_try_get_processor_chat_template`` (``vllm/renderers/hf.py:219-260`` at v0.31.0) builds the checkpoint's
    processor and takes its ``chat_template``; a build that fails leaves the engine the tokenizer's template. The
    processor's own template (``processor_template``: a ``chat_template`` key in ``processor_config.json``, else the
    legacy ``chat_template.json``) is read first: without one a processor shares ``chat_template.jinja`` with the
    tokenizer and there is nothing to resolve, so nothing is built. With it, the processor is built as the engine
    builds it: one transformers knows gives its template (``built``); one that is the vendor's code (the checkpoint's
    ``auto_map`` names it, transformers refuses it without ``trust_remote_code``) is built by an engine that runs that
    code, so the file's template counts (``remote-code``); one transformers 5.17 cannot build, a class it lacks, leaves
    the engine the tokenizer's template (``not-built``). A build that fails for a missing backend (torch, torchvision,
    pillow) or a package the processor imports itself (num2words) is the environment's failure, not the checkpoint's,
    and is refused, naming the package to install (``backends_refusal``): the recorder decides as the engine does only
    with the processor's backends installed. An ``ImportError`` that names no missing module or package, a symbol the
    installed transformers no longer exports, is a class it lacks and is ``not-built`` like the other build errors
    (``is_missing_dependency``)."""
    template = processor_template(model, revision)
    if template is None:
        return None, None
    try:
        processor = built_processor(model, kwargs)
    except ImportError as err:
        if not is_missing_dependency(err):
            return None, PROCESSOR_NOT_BUILT
        raise backends_refusal(model, revision, err) from err
    except (
        Exception
    ) as err:  # transformers raises ValueError, OSError and its own errors for a processor it cannot build
        message = str(err)
        if "contains custom code" in message:
            return template, PROCESSOR_REMOTE_CODE
        if is_missing_backend(message):
            raise backends_refusal(model, revision, err) from err
        return None, PROCESSOR_NOT_BUILT
    from transformers import ProcessorMixin

    if not isinstance(processor, ProcessorMixin):
        return None, PROCESSOR_NOT_BUILT
    built = getattr(processor, "chat_template", None)
    if isinstance(built, dict):
        built = built.get("default")
    return (built if isinstance(built, str) else template), PROCESSOR_BUILT


def processor_template(model: str, revision: str) -> str | None:
    """The processor's own template, if the checkpoint ships one, in the order transformers' ``ProcessorMixin`` takes
    it (``processing_utils.py`` at 5.17.0): a ``chat_template`` key inside ``processor_config.json`` first
    (lines 1434-1440: the key, deprecated but honored, wins over every template file), else the ``chat_template`` of
    ``chat_template.json``, the legacy template file (lines 1384-1391). A checkpoint whose processor template lives in
    ``chat_template.jinja`` alone shares that file with the tokenizer, so there is nothing to resolve."""
    for filename in PROCESSOR_TEMPLATE_FILES:
        path = checkpoint_file(model, revision, filename)
        template = json.loads(path.read_text()).get("chat_template") if path is not None else None
        if isinstance(template, str):
            return template
    return None


def checkpoint_file(model: str, revision: str, filename: str) -> Path | None:
    """``filename`` of the checkpoint at ``revision``, or None when the checkpoint is known not to ship it.

    A local directory answers by what is on disk. For a hub id the hub cache must answer: the file, or the
    ``.no_exist`` marker that a download which got a 404 leaves. A file that is merely not cached is an error, because
    offline transformers would take it for one the repository does not ship and silently fall back.
    """
    if Path(model).is_dir():
        path = Path(model) / filename
        return path if path.is_file() else None
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    cached = try_to_load_from_cache(model, filename, revision=revision)
    if cached is _CACHED_NO_EXIST:
        return None
    if cached is None:
        raise FileNotFoundError(
            f"{filename} of {model} at {revision} is neither cached nor known to be absent; fetch it with "
            f"`hf download {model} {filename} --revision {revision}` (a 404 marks it absent)"
        )
    return Path(cached)
