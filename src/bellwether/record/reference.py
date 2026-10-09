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
would choose it (Phi-4-multimodal's ``GPT2TokenizerFast``). Neither applies with ``vendor_code``: the vendor's class
loads as the checkpoint names it, and renders with its own template or none (Kimi-K3's).

The ids of the rendered text come from the checkpoint's ``tokenizer.json`` itself, read with ``tokenizers``, when
the checkpoint ships one and the vendor's code is not in use: that file is the vendor's shipped encoder. A
transformers tokenizer class may install its own pre-tokenizer over the file's (``Qwen2Tokenizer`` splits with a
pattern that has no ``\\p{M}``, the Qwen3.5-generation ``tokenizer.json`` keeps combining marks with their letters),
and the ids then differ on scripts with combining marks although the text is the same. A checkpoint without
``tokenizer.json`` (a tiktoken or SentencePiece vocabulary) is encoded by the transformers tokenizer, as before.
The provenance names the encoder (``encoder``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import vendor

SOURCE = "hf-template"


@dataclass
class Rendered:
    input_ids: list[int]
    text: str


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
        kwargs = {
            "tools": request.get("tools"),
            "add_generation_prompt": bool(request.get("add_generation_prompt", not continue_final)),
            "continue_final_message": continue_final,
            **dict(request.get("chat_template_kwargs") or {}),
        }
        messages = request["messages"]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        if self.encoder is not None:
            ids = self.encoder.encode(str(text), add_special_tokens=False).ids
        else:
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
            ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return Rendered([int(i) for i in ids], str(text))

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
            **({"chat_template_sha256": self.template_sha256} if self.template_sha256 else {}),
        }
        if self.vendor_code:
            found |= vendor.packages()
        return found


def load_tokenizer(model: str, revision: str, kwargs: dict):
    """The checkpoint's tokenizer: ``AutoTokenizer``'s, or, when the model config is the vendor's class
    (``auto_map``'s ``AutoConfig``) and the vendor's code may not run, the class ``tokenizer_config.json`` names, as
    ``AutoTokenizer`` would choose it. A tokenizer that is the vendor's class itself is ``AutoTokenizer``'s to refuse,
    or, with ``trust_remote_code`` in the sandbox, to load."""
    from transformers import AutoTokenizer
    from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

    if not kwargs.get("trust_remote_code") and vendor_config(model, revision):
        path = checkpoint_file(model, revision, "tokenizer_config.json")
        config = json.loads(path.read_text()) if path is not None else {}
        named = config.get("tokenizer_class")
        if "auto_map" not in config and isinstance(named, str):
            tokenizer_class = tokenizer_class_from_name(named)
            if tokenizer_class is not None:
                return tokenizer_class.from_pretrained(model, **kwargs)
    return AutoTokenizer.from_pretrained(model, **kwargs)


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
