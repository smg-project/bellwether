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
        from transformers import AutoTokenizer

        kwargs: dict = {} if Path(model).is_dir() else {"revision": revision}
        if vendor_code:
            kwargs["trust_remote_code"] = True
        self.vendor_code = vendor_code
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        template = self.tokenizer.chat_template
        if template is None and not vendor_code:
            raise ValueError(f"{model} at {revision} ships no chat template; the hf-template oracle cannot render it")
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
            **({"chat_template_sha256": self.template_sha256} if self.template_sha256 else {}),
        }
        if self.vendor_code:
            found |= vendor.packages()
        return found
