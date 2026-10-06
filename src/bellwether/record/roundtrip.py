"""The roundtrip parse oracle: the checkpoint's own template renders the message an output must parse to.

The design's first parse authority is the round trip: for a well-formed output, rendering the parsed
assistant message through the model's template must reproduce the output text. Recording runs it
the other way round. The corpus states the message; the template renders it as the final assistant
turn after the generation prompt; the text between the generation prompt and the end-of-turn token
is the output the parsers must turn back into that message, and its token ids are the engine chunks
a replay feeds.

A template that does not extend the generation prompt when the turn is appended cannot serve as
this oracle for that case (DeepSeek-R1 never renders ``<think>``, so no rendered turn extends its
``<think>`` prompt; Qwen3 with thinking switched off cannot render reasoning). Such a case is
reported and not recorded, and the run exits 1; recording it is left to the manifest's next
authority, which nothing here invokes.

The output follows the generation prompt, so a request that asks for no generation prompt or for
the final message to be continued cannot be recorded this way and is rejected.

The template gets every assistant message, the request's history and the final turn alike, as vLLM
gives it to a template (``_postprocess_messages`` in ``vllm/entrypoints/chat_utils.py`` at v0.31.0,
the release whose image bellwether pins): a call's arguments become the object they decode to,
anything that does not decode to a JSON object becomes ``{}``, and an empty ``tool_calls`` is
dropped. SGLang differs: at 7d22b7a8 (``normalize_assistant_tool_call_arguments``) it rejects a
string that is not a JSON object, except under Kimi-K3's encoding, and leaves missing or null
arguments as they are. That difference is recorded here, not decided. A parse case's own call must
carry its arguments as a JSON object string, the one a parser returns: that is a rule of the corpus,
not of an engine. A template that cannot take an object (DeepSeek's concatenate the string) fails the
case: that is a finding about the template and the engines, reported, never worked around. The
reference message keeps the JSON string.

Next to the ids the oracle records the text each token contributes under the tokenizer's incremental
decode (``DecodeStream``), the pieces a replay feeds with each id: a token that does not complete a
character contributes nothing, and the token that completes it carries the whole character. Joined,
the pieces must give back the output text; a case where they do not is reported and not recorded.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from tokenizers.decoders import DecodeStream

from .reference import HfTemplateOracle

SOURCE = "roundtrip"


@dataclass
class OutputText:
    text: str
    output_ids: list[int]
    output_pieces: list[str]
    finish_reason: str


class RoundtripOracle:
    def __init__(self, model: str, revision: str) -> None:
        self.renderer = HfTemplateOracle(model, revision)
        self.tokenizer = self.renderer.tokenizer
        if not self.tokenizer.eos_token:
            raise ValueError(
                f"{model} at {revision} has no end-of-turn token; the round trip cannot find the turn's end"
            )

    def render_output(self, request: dict, message: dict) -> OutputText:
        """The output text and ids for ``message`` as the final assistant turn of ``request``."""
        if request.get("add_generation_prompt") is False or request.get("continue_final_message"):
            raise ValueError(
                "a parse case's request must end at the generation prompt; `add_generation_prompt: false` and "
                "`continue_final_message` cannot be recorded by the round trip"
            )
        messages = [as_vllm_gives_it(m) if m.get("role") == "assistant" else m for m in request["messages"]]
        kwargs = {"tools": request.get("tools"), **dict(request.get("chat_template_kwargs") or {})}
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        check_parse_call_arguments(message)
        turn = {"role": "assistant", **as_vllm_gives_it(message)}
        rendered = self.tokenizer.apply_chat_template(
            [*messages, turn], tokenize=False, add_generation_prompt=False, **kwargs
        )
        if not rendered.startswith(prompt):
            raise ValueError(
                "the template does not extend the generation prompt when the turn is appended; "
                "the round trip is lossy for this case"
            )
        turn = rendered[len(prompt) :]
        end = self.tokenizer.eos_token
        if end not in turn:
            raise ValueError(f"the rendered turn carries no end-of-turn token {end!r}")
        text = turn[: turn.rindex(end)]
        output_ids = [int(i) for i in self.tokenizer.encode(text, add_special_tokens=False)]
        if self.tokenizer.decode(output_ids) != text:
            raise ValueError("the output text does not survive a tokenize-detokenize round trip")
        stream = DecodeStream(skip_special_tokens=False)
        output_pieces = [stream.step(self.tokenizer.backend_tokenizer, token) or "" for token in output_ids]
        if "".join(output_pieces) != text:
            raise ValueError("the output's tokens do not give back its text under the tokenizer's incremental decode")
        finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        return OutputText(text, output_ids, output_pieces, finish_reason)

    def provenance(self) -> dict:
        return self.renderer.provenance()


def as_vllm_gives_it(message: dict) -> dict:
    """A copy of an assistant message as vLLM gives it to the template.

    This mirrors ``_postprocess_messages`` in ``vllm/entrypoints/chat_utils.py:2084`` at v0.31.0, the release whose
    image bellwether pins: an empty ``tool_calls`` is dropped; a call's arguments become the object they decode to, and
    anything that does not decode to a JSON object (missing, null, empty, invalid JSON, or JSON of another type)
    becomes ``{}``.
    """
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message
    message = copy.deepcopy(message)
    if not tool_calls:
        del message["tool_calls"]
        return message
    for call in message["tool_calls"]:
        function = call["function"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        function["arguments"] = arguments if isinstance(arguments, dict) else {}
    return message


def check_parse_call_arguments(message: dict) -> None:
    """The corpus's rule for a parse case: each call carries its arguments as the JSON object string a parser returns.

    vLLM would give the template ``{}`` for no arguments, or another JSON value as it is; a case that relies on either
    states no string for a parser to return, so it is reported, not recorded.
    """
    for call in message.get("tool_calls") or []:
        arguments = call["function"].get("arguments")
        try:
            value = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            value = None
        if not isinstance(value, dict):
            raise ValueError(
                "a parse case's call must carry its arguments as a JSON object string (a rule of the corpus): "
                f"{arguments!r}"
            )
