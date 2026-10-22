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

The template is the tokenizer's, or, when the tokenizer ships none, the ``chat_template`` of
``chat_template.json``, the processor's file, as transformers' processor and vLLM read it (Qwen3-Omni
ships its template only there). The tokenizer is ``AutoTokenizer``'s, except when the model config is
the vendor's class: ``AutoTokenizer`` builds the config first, which needs that code, and bellwether
never runs it outside the sandbox. Then the class ``tokenizer_config.json`` names is loaded as ``AutoTokenizer``
would choose it (Phi-4-multimodal's ``GPT2TokenizerFast``). Nor does a model config of transformers' own class stop
the tokenizer when that class's validators refuse the checkpoint's values (ZAYA1's and an OLMo-3 fine-tune's rope
settings): ``AutoTokenizer`` is given a config of the same model type to choose the class by. Neither applies with
``vendor_code``: the vendor's class loads as the checkpoint names it, and renders with its own template or none
(Kimi-K3's).

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

The ids of the rendered text come from the checkpoint's ``tokenizer.json`` itself, read with ``tokenizers``, when
the checkpoint ships one and the vendor's code is not in use: that file is the vendor's shipped encoder. A
transformers tokenizer class may install its own pre-tokenizer over the file's (``Qwen2Tokenizer`` splits with a
pattern that has no ``\\p{M}``, the Qwen3.5-generation ``tokenizer.json`` keeps combining marks with their letters),
and the ids then differ on scripts with combining marks although the text is the same. A checkpoint without
``tokenizer.json`` (a tiktoken or SentencePiece vocabulary) is encoded by the transformers tokenizer, as before.
The provenance names the encoder (``encoder``).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import vendor
from .content_format import ContentFormat, detect_content_format, in_content_format

SOURCE = "hf-template"


@dataclass
class Rendered:
    input_ids: list[int]
    text: str
    content_format: ContentFormat


class HfTemplateOracle:
    def __init__(self, model: str, revision: str, vendor_code: bool = False) -> None:
        if vendor_code:
            vendor.require_sandbox(model)
        kwargs: dict = {} if Path(model).is_dir() else {"revision": revision}
        if vendor_code:
            kwargs["trust_remote_code"] = True
        self.vendor_code = vendor_code
        self.tokenizer = load_tokenizer(model, revision, kwargs)
        self.encoder = None if vendor_code else checkpoint_encoder(model, revision)
        template = self.tokenizer.chat_template
        if template is None and not vendor_code:
            template = processor_template(model, revision)
            if template is None:
                raise ValueError(
                    f"{model} at {revision} ships no chat template; the hf-template oracle cannot render it"
                )
            self.tokenizer.chat_template = template
        if template is not None and not isinstance(template, str):
            template = json.dumps(template, sort_keys=True)
        # A vendor's class may render its chat format itself (Kimi-K3's does), with no template to hash; its code is
        # pinned by the manifest's inputs instead.
        self.template_sha256 = None if template is None else hashlib.sha256(template.encode("utf-8")).hexdigest()

    def render(self, request: dict) -> Rendered:
        continue_final = bool(request.get("continue_final_message", False))
        tools = request.get("tools")
        kwargs = {
            "tools": tools,
            "add_generation_prompt": bool(request.get("add_generation_prompt", not continue_final)),
            "continue_final_message": continue_final,
            **dict(request.get("chat_template_kwargs") or {}),
        }
        content_format = self.content_format(tools)
        messages = self.for_template(request["messages"], tools)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        if self.encoder is not None:
            ids = self.encoder.encode(str(text), add_special_tokens=False).ids
        else:
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
            ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return Rendered([int(i) for i in ids], str(text), content_format)

    def for_template(self, messages: list[dict], tools: list | None) -> list[dict]:
        """``messages`` as the engine hands them to the template for a request with ``tools``: each message's content
        in the format the template takes (``in_content_format``), and each assistant message's tool calls as
        ``as_vllm_gives_it`` gives them. The request's own objects are never written to."""
        prepared = in_content_format(messages, self.content_format(tools))
        return [as_vllm_gives_it(message) if message.get("role") == "assistant" else message for message in prepared]

    def content_format(self, tools: list | None) -> ContentFormat:
        """The content format vLLM selects for the template transformers applies to a request with ``tools`` (a
        checkpoint with named templates has one for requests with tools); "string" when the tokenizer resolves no
        template, as a vendor's class that renders its own format."""
        try:
            template = self.tokenizer.get_chat_template(None, tools=tools)
        except Exception:
            return "string"
        return detect_content_format(template) if isinstance(template, str) else "string"

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
            "tool_call_arguments": "object",
            **({"chat_template_sha256": self.template_sha256} if self.template_sha256 else {}),
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
            return auto_tokenizer_class(named).from_pretrained(model, **kwargs)
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


def auto_tokenizer_class(named: str):
    """The class ``AutoTokenizer`` loads for the ``tokenizer_class`` that ``tokenizer_config.json`` names, as
    ``tokenization_auto.py`` chooses it at transformers 5.17.0 (lines 875-938): a trailing ``Fast`` is dropped, the
    name is looked up as it is, then with ``Fast``; the Python base it finds for ``PreTrainedTokenizer``, which has no
    vocabulary of its own, becomes the ``tokenizers`` backend, and so does a name transformers does not ship."""
    from transformers import TokenizersBackend
    from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

    name = named.removesuffix("Fast")
    tokenizer_class = tokenizer_class_from_name(name) or tokenizer_class_from_name(f"{name}Fast")
    if tokenizer_class is None or tokenizer_class.__name__ == "PythonBackend":
        return TokenizersBackend
    return tokenizer_class


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


def processor_template(model: str, revision: str) -> str | None:
    """The ``chat_template`` of ``chat_template.json``, the processor's template file, if the checkpoint ships one."""
    path = checkpoint_file(model, revision, "chat_template.json")
    template = json.loads(path.read_text()).get("chat_template") if path is not None else None
    return template if isinstance(template, str) else None


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
