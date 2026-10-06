"""The roundtrip parse oracle: the checkpoint's own template renders the message an output must parse to.

The design's first parse authority is the round trip: for a well-formed output, rendering the parsed
assistant message through the model's template must reproduce the output text. Recording runs it
the other way round. The corpus states the message; the template renders it as the final assistant
turn after the generation prompt; the turn up to where generation stops is the output the parsers
must turn back into that message, and its token ids are the engine chunks a replay feeds.

Generation stops on vLLM's stop set: the ``eos_token_id`` of the checkpoint's generation config
(``generation_config.json``, else ``config.json`` as ``GenerationConfig.from_model_config`` reads it)
and the tokenizer's eos, when it has one (``SamplingParams.update_from_generation_config``). That is
vLLM's stop set within one limit. vLLM reads ``config.json`` through the config class of its model
type, whose defaults count, and here that class is transformers' own for a model type transformers
knows; vendor code is never run. So a default eos that ``config.json`` does not state is missed when
the class is the vendor's (``auto_map``), whose code vLLM runs under ``--trust-remote-code``, or
vLLM's own (``_CONFIG_REGISTRY`` in ``vllm/transformers_utils/config.py``). Only a checkpoint that
ships no ``generation_config.json`` can be affected. The
output is the text before the first stop id in the rendered turn, after which the turn may hold only
whitespace and further stop ids. A template that writes no end marker (GLM's) gives the whole turn,
provided the next message, a user message after content or a tool message after tool calls, opens
with a stop id. Any other turn is reported and not recorded, and so is a message whose own text
(content, reasoning, a call's name or arguments) holds a stop id, since generation would stop inside
it. The line records the stop id and where it was found, in the turn or in the next message.
transformers' ``generate`` stops on the generation config's ids alone. Where that would end an output
elsewhere (Qwen3.5-9B's config lists only ``<|endoftext|>``, and its turns end with the tokenizer's
``<|im_end|>``), the output still ends where vLLM stops, as serving engines do. The ids ``generate``
stops on and the file they come from are a fact of the checkpoint, not of a case: ``record`` keeps
them in each parse set's table in ``sets.toml`` and prints one line for the model when some outputs
end on an id ``generate`` does not stop on. The generation config must be cached at the revision or
known absent: offline, transformers takes a file that is merely not cached for one the repository
does not ship.

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
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tokenizers.decoders import DecodeStream

from .reference import HfTemplateOracle

SOURCE = "roundtrip"
GENERATION_CONFIG = "generation_config.json"


@dataclass
class OutputText:
    text: str
    output_ids: list[int]
    output_pieces: list[str]
    finish_reason: str
    end_of_turn: dict


class RoundtripOracle:
    def __init__(self, model: str, revision: str) -> None:
        self.renderer = HfTemplateOracle(model, revision)
        self.tokenizer = self.renderer.tokenizer
        self.generate_eos_ids, self.generation_config_source = generation_eos_ids(model, revision)
        # vLLM's stop set: the generation config's eos_token_id and the tokenizer's eos, when it has one.
        eos = self.tokenizer.eos_token_id
        self.stop_ids = set(self.generate_eos_ids) | ({eos} if eos is not None else set())
        if not self.stop_ids:
            raise ValueError(
                f"{model} at {revision} has no stop id: neither its generation config nor its tokenizer names an eos, "
                "so nothing ends a turn"
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
        assistant = {"role": "assistant", **as_vllm_gives_it(message)}
        self.check_no_stop_id_in_the_message(assistant)
        rendered = self.tokenizer.apply_chat_template(
            [*messages, assistant], tokenize=False, add_generation_prompt=False, **kwargs
        )
        self.check_every_call_is_rendered(messages, kwargs, message, rendered)
        if not rendered.startswith(prompt):
            raise ValueError(
                "the template does not extend the generation prompt when the turn is appended; "
                "the round trip is lossy for this case"
            )
        turn = rendered[len(prompt) :]
        cut = stop_in_turn(self.tokenizer, turn, self.stop_ids)
        if cut is None:
            continued = self.render_next_message(messages, assistant, message, kwargs)
            cut = stop_at_next_message(self.tokenizer, prompt, rendered, continued, self.stop_ids)
        if cut.after.strip():
            name = self.tokenizer.convert_ids_to_tokens(cut.stop_id)
            raise ValueError(
                f"the turn goes on after stop id {cut.stop_id} ({name!r}): {turn[len(cut.text) :][:80]!r}; "
                "the template writes more of the turn after the point where generation stops"
            )
        text = cut.text
        output_ids = [int(i) for i in self.tokenizer.encode(text, add_special_tokens=False)]
        if self.tokenizer.decode(output_ids) != text:
            raise ValueError("the output text does not survive a tokenize-detokenize round trip")
        stream = DecodeStream(skip_special_tokens=False)
        output_pieces = [stream.step(self.tokenizer.backend_tokenizer, token) or "" for token in output_ids]
        if "".join(output_pieces) != text:
            raise ValueError("the output's tokens do not give back its text under the tokenizer's incremental decode")
        finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        end_of_turn = {"stop_id": cut.stop_id, "found_by": cut.found_by}
        return OutputText(text, output_ids, output_pieces, finish_reason, end_of_turn)

    def check_no_stop_id_in_the_message(self, assistant: dict) -> None:
        """No string of the message, as the template gets it, may hold a stop id.

        Generation stops at the first stop id, and the turn is searched by id, so a stop token in the message's own
        content, reasoning or call arguments would end the output inside the message: the case would be recorded
        short when only whitespace follows the token, and refused as if the template went on after generation stops
        otherwise. ``</s>`` is ordinary HTML and a stop token of several checkpoints. Each string is tokenized on its
        own, as the output is; a call's arguments are read as the template gets them, decoded, keys included.
        """
        for where, text in strings_of(assistant):
            for token in self.tokenizer.encode(text, add_special_tokens=False):
                if token in self.stop_ids:
                    name = self.tokenizer.convert_ids_to_tokens(token)
                    raise ValueError(
                        f"the message's own text holds stop id {token} ({name!r}) in {where}: generation stops there, "
                        "so no output carries the message whole"
                    )

    def render_next_message(self, messages: list, assistant: dict, message: dict, kwargs: dict) -> str:
        """The conversation rendered with the next message after the turn, for a turn that holds no stop id."""
        try:
            return self.tokenizer.apply_chat_template(
                [*messages, assistant, *next_messages(message)], tokenize=False, add_generation_prompt=False, **kwargs
            )
        except Exception as err:
            raise ValueError(
                f"no stop id in the turn, and the template cannot render the next message: {type(err).__name__}: {err}"
            ) from err

    def check_every_call_is_rendered(self, messages: list, kwargs: dict, message: dict, rendered: str) -> None:
        """Every call of the message must reach the rendered turn.

        Renaming a call, or adding a key to its arguments, must change what the template renders. A template that
        drops tool calls (Phi-4-mini's, Hunyuan-A13B's) or renders only some of them would otherwise give an output a
        parser cannot turn back into the message, and the case would be recorded lossy. Two renders per call; a
        template that fails on the change has read the call.
        """
        for index, call in enumerate(message.get("tool_calls") or []):
            for change, what in ((renamed, "renaming"), (with_marker_argument, "adding an argument to")):
                variant = {"role": "assistant", **as_vllm_gives_it(change(message, index))}
                try:
                    again = self.tokenizer.apply_chat_template(
                        [*messages, variant], tokenize=False, add_generation_prompt=False, **kwargs
                    )
                except Exception:
                    continue
                if again == rendered:
                    raise ValueError(
                        f"the template does not render every tool call: {what} call {index} "
                        f"({call['function']['name']}) leaves the rendered turn as it was, so the output would not "
                        "carry it"
                    )

    def provenance(self) -> dict:
        return self.renderer.provenance()

    def generate_stop(self) -> dict:
        """The ids transformers' ``generate`` stops on and the file they come from, as a parse set's table in sets.toml
        holds them; nothing for a checkpoint that ships neither file, which transformers could not load a model from."""
        if self.generation_config_source is None:
            return {}
        return {"generate_stop_ids": self.generate_eos_ids, "generate_stop_ids_from": self.generation_config_source}

    def stop_sets_differ(self, model: str, recorded: list[int]) -> str | None:
        """One line for the run when some of the ``recorded`` outputs end on a stop id ``generate`` does not stop on.

        vLLM stops on the generation config's ids and the tokenizer's eos, ``generate`` on the generation config's ids
        alone, so the two differ when an output ends on the tokenizer's eos and the generation config does not list it.
        The outputs still end where vLLM stops, as serving engines do.
        """
        if self.generation_config_source is None:
            return None
        outside = Counter(stop_id for stop_id in recorded if stop_id not in self.generate_eos_ids)
        if not outside:
            return None
        named = ", ".join(f"{i} ({self.tokenizer.convert_ids_to_tokens(i)!r})" for i in sorted(outside))
        return (
            f"stop sets differ for {model}: transformers' generate stops on {self.generate_eos_ids} "
            f"({self.generation_config_source}), so {outside.total()} of the {len(recorded)} outputs recorded here "
            f"end on a stop id it does not stop on: {named}"
        )


def generation_eos_ids(model: str, revision: str) -> tuple[list[int], str | None]:
    """The ``eos_token_id`` transformers' ``generate`` stops on, and the file it comes from.

    ``generate`` reads ``generation_config.json``, else ``config.json`` through ``GenerationConfig.from_model_config``,
    which takes a value the top level leaves unset from ``text_config`` (or ``decoder``, ``generator``). The model
    config it is given is the one ``model_config`` builds.
    """
    from transformers import GenerationConfig

    path = checkpoint_file(model, revision, GENERATION_CONFIG)
    if path is not None:
        source, config = GENERATION_CONFIG, GenerationConfig.from_dict(json.loads(path.read_text()))
    else:
        path = checkpoint_file(model, revision, "config.json")
        if path is None:
            return [], None
        source, config = "config.json", GenerationConfig.from_model_config(model_config(json.loads(path.read_text())))
    eos = config.eos_token_id
    return ([] if eos is None else [eos] if isinstance(eos, int) else [int(i) for i in eos]), source


def model_config(values: dict):
    """``config.json`` as the model config vLLM gives ``GenerationConfig.from_model_config``, without vendor code.

    vLLM gives it the config object, which carries its class's defaults: ``{"model_type": "llama"}`` states no eos,
    and ``LlamaConfig``'s is 2. For a model type transformers knows, that is transformers' class, built the way
    ``AutoConfig.from_pretrained`` builds it. Any other model type's class is the vendor's code, named by ``auto_map``,
    which bellwether never runs (nor ``trust_remote_code``): its values are read as written, and a default eos that
    class would set is not seen.
    """
    from transformers import CONFIG_MAPPING

    model_type = values.get("model_type")
    if isinstance(model_type, str) and model_type in CONFIG_MAPPING:
        return CONFIG_MAPPING[model_type].from_dict(values)
    return values


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


@dataclass
class Cut:
    """Where generation that stops on a set of ids ends the output: the text before the stop id, the stop id, the step
    that found it, and what the turn holds after the stop id other than further stop ids."""

    text: str
    stop_id: int
    found_by: str
    after: str = ""


def stop_in_turn(tokenizer, turn: str, stop_ids: set[int]) -> Cut | None:
    """Where the first stop id in the rendered ``turn`` ends the output; None when the turn holds none.

    What the turn holds after it, further stop ids left out, is kept in ``after``: the round trip records a turn only
    when that is whitespace (Phi-4-mini writes ``<|end|><|endoftext|>``), since anything else is part of the turn the
    template renders after generation stops.
    """
    encoded = tokenizer(turn, add_special_tokens=False, return_offsets_mapping=True)
    tokens = list(zip(encoded["input_ids"], encoded["offset_mapping"], strict=True))
    for index, (token, (start, end)) in enumerate(tokens):
        if token not in stop_ids:
            continue
        after, position = [], end
        for later, (later_start, later_end) in tokens[index + 1 :]:
            if later in stop_ids:
                after.append(turn[position:later_start])
                position = later_end
        after.append(turn[position:])
        return Cut(turn[:start], token, "turn", "".join(after))
    return None


def stop_at_next_message(tokenizer, prompt: str, rendered: str, continued: str, stop_ids: set[int]) -> Cut:
    """Where a turn with no stop id ends: the whole turn is the output when the next message opens with a stop id,
    right where the turn ends and on a token boundary (GLM's role tags); a ValueError says why it ends nowhere.

    ``rendered`` is the conversation that ends with the turn, after the generation ``prompt``; ``continued`` is the same
    conversation with the next message after the turn, and it must extend ``rendered``, or what follows the turn is
    unknown.
    """
    if not continued.startswith(rendered):
        raise ValueError(
            "no stop id in the turn, and the template renders the turn differently once the next message follows it, "
            "so what follows the turn is unknown"
        )
    turn, tail = rendered[len(prompt) :], continued[len(prompt) :]
    encoded = tokenizer(tail, add_special_tokens=False, return_offsets_mapping=True)
    for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True):
        if end <= len(turn):  # the turn's own, a zero-width token at its end (trim_offsets) among them
            continue
        name = tokenizer.convert_ids_to_tokens(token)
        if start == len(turn) and token in stop_ids:
            return Cut(turn, token, "next-message")
        if start == len(turn):
            raise ValueError(
                f"no stop id in the turn, and the next message opens with {name!r} ({token}), not a stop id"
            )
        raise ValueError(
            "no stop id in the turn, and no token starts where the turn ends once the next message follows it "
            f"(the first one after it is {name!r} ({token}))"
        )
    raise ValueError("no stop id in the turn, and nothing follows it")


def next_messages(message: dict) -> list[dict]:
    """What continues the conversation after ``message``: a user message after content, a tool message per call after
    tool calls, each with its call's id when the call has one."""
    calls = message.get("tool_calls") or []
    if not calls:
        return [{"role": "user", "content": "Thanks."}]
    return [
        {
            "role": "tool",
            **({"tool_call_id": call["id"]} if "id" in call else {}),
            "name": call["function"]["name"],
            "content": "{}",
        }
        for call in calls
    ]


def strings_of(value: object, path: str = "") -> Iterator[tuple[str, str]]:
    """Every string in ``value``, each key of a mapping included, with where it is (``tool_calls[0].function.name``)."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            where = f"{path}.{key}" if path else str(key)
            yield where, str(key)
            yield from strings_of(item, where)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from strings_of(item, f"{path}[{index}]")


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

    vLLM v0.31.0 would give the template ``{}`` for missing, empty or non-object arguments; a case that relies on that
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


MARKER = "bellwether_marker"


def renamed(message: dict, index: int) -> dict:
    """A copy of the message whose call ``index`` is named ``MARKER``."""
    message = copy.deepcopy(message)
    message["tool_calls"][index]["function"]["name"] = MARKER
    return message


def with_marker_argument(message: dict, index: int) -> dict:
    """A copy of the message whose call ``index`` carries one more argument, ``MARKER``."""
    message = copy.deepcopy(message)
    function = message["tool_calls"][index]["function"]
    function["arguments"] = json.dumps({**json.loads(function["arguments"]), MARKER: MARKER})
    return message
