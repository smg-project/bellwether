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
"""

from __future__ import annotations

from dataclasses import dataclass

from .reference import HfTemplateOracle

SOURCE = "roundtrip"


@dataclass
class OutputText:
    text: str
    output_ids: list[int]
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
        messages = request["messages"]
        kwargs = {"tools": request.get("tools"), **dict(request.get("chat_template_kwargs") or {})}
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        rendered = self.tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", **message}], tokenize=False, add_generation_prompt=False, **kwargs
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
        finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        return OutputText(text, output_ids, finish_reason)

    def provenance(self) -> dict:
        return self.renderer.provenance()
