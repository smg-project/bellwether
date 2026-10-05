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
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SOURCE = "hf-template"


@dataclass
class Rendered:
    input_ids: list[int]
    text: str


class HfTemplateOracle:
    def __init__(self, model: str, revision: str) -> None:
        from transformers import AutoTokenizer

        kwargs = {} if Path(model).is_dir() else {"revision": revision}
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        template = self.tokenizer.chat_template
        if template is None:
            raise ValueError(f"{model} at {revision} ships no chat template; the hf-template oracle cannot render it")
        if not isinstance(template, str):
            template = json.dumps(template, sort_keys=True)
        self.template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

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

        return {
            "oracle": "transformers.apply_chat_template",
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "jinja2": jinja2.__version__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "chat_template_sha256": self.template_sha256,
        }
