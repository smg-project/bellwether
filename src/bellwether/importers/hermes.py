"""Hermes function-calling conversations as corpus sets: NousResearch/hermes-function-calling-v1.

The data is three JSON files of the Hugging Face dataset at a pinned revision, read through the Hugging Face cache and
checked against their sha256 on every use (``hf.fetch``), with the license read from the dataset card
(``hf.check_card_license``). Each row is a conversation in Hermes's own format; the importer turns it into OpenAI chat
requests and assistant messages:

- the request's ``tools`` are the row's ``tools`` field;
- the system turn loses the Hermes tool prompt (``TOOL_PROMPTS``), Hermes's own rendering of those tools, since the
  checkpoint's template renders them from ``tools``; what else it holds stays (``system_message``);
- a ``human`` turn is a user message;
- a ``gpt`` turn is an assistant message: its prose is ``content`` and each ``<tool_call>`` block one call
  (``split_calls``), with the id ``call_<n>``, numbered across the conversation;
- each ``<tool_response>`` block of a ``tool`` turn is a tool message answering the call in the same position.

Every assistant turn is a parse case and every user turn an assistant turn answers is a render case. A row with no
faithful OpenAI form gives no case, and the import names it with its reason (``Unmappable``); a case that repeats an
earlier one is left out, and the import names it with the case it repeats, so every case is distinct.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import corpus_sets, hf

REPO = "NousResearch/hermes-function-calling-v1"
REVISION = "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
SOURCE = f"hf:datasets/{REPO}@{REVISION}"
LICENSE = "Apache-2.0"
CARD_SHA256 = "a01aa1fe59858148f87d0a584e69bd4e33a9a3c70a177fe1517cad57ead45ae5"
CARD_LICENSE = "apache-2.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The card's configs that hold tool calls, with their files and sha256. The json_mode_* configs are structured-output
# cases, not tool calls, and are left for later.
CONFIGS = {
    "func_calling_singleturn": (
        "func-calling-singleturn.json",
        "fb2df9fe3f295dfd65b223c2dbf04d6b90006357d737f8f8927749f0bea09527",
    ),
    "func_calling": ("func-calling.json", "769478035a886678d525d057ea66e3fd1e247f43e35c8d6d7cc866e080b6742b"),
    "glaive_func_calling": (
        "glaive-function-calling-5k.json",
        "b98eb3f160359f27ad15018e974ce6db444f566eb5be4aa9e4aa690b34d50832",
    ),
}
# The rows taken: every STRIDE-th row of each file, from its FIRST_ROW. All rows would make 82.3 MB of plain JSON
# Lines, past the 50 MB one source's sets may take (corpus_sets.LIMIT, which write enforces); every second row makes
# 44.6 MB. func_calling's rows begin as func_calling_singleturn's rows of the same index (the first three turns are
# equal in 1883 of 1893 rows), so it takes the odd rows where the others take the even ones, and none of its cases
# repeats one of func_calling_singleturn's.
STRIDE = 2
FIRST_ROW = {"func_calling_singleturn": 0, "func_calling": 1, "glaive_func_calling": 0}

# The dataset's system prompts for tools, as (text before, text after) a ``<tools>`` element that holds the row's
# ``tools`` field: func_calling and func_calling_singleturn use the first two, glaive_func_calling the third. Each is
# Hermes's rendering of the tool list, which the checkpoint's template renders in its own words from ``tools``.
TOOL_PROMPTS = (
    (
        "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML"
        " tags. You may call one or more functions to assist with the user query. Don't make assumptions about what"
        " values to plug into functions.\n",
        "\nFor each function call return a json object with function name and arguments within <tool_call>"
        " </tool_call> tags with the following schema:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-dict>}\n</tool_call>\n',
    ),
    (
        "You are an expert structured information extraction AI model. You will be provided with documents to extract"
        " information from. You are also provided with the json schema to output extracted information in the"
        " function signatures within XML tags <tools></tools>. Don't make assumptions about what values to plug into"
        " json schema. \n",
        "\nFor each extraction function call return a json object with function name and arguments followed by a"
        " <tool_call> tag with the following schema:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-dict>}\n</tool_call>',
    ),
    (
        "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML"
        " tags.You may call one or more functions to assist with the user query. Don't make assumptions about what"
        " values to plug into functions.Here are the available tools:",
        "Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall',"
        " 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title':"
        " 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object"
        " with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n"
        "{tool_call}\n</tool_call>",
    ),
)


class Unmappable(ValueError):
    """A row that has no faithful OpenAI form: the import names it with this reason and takes no case from it."""


def read_rows(path: Path) -> list[dict]:
    return json.loads(path.read_bytes().decode("utf-8"))


def system_message(text: str, tools: str | None) -> str:
    """What is left of a system turn once the Hermes tool prompt carrying ``tools`` (the row's field) is taken out.

    ``tools`` is ``None`` for a row that declares no tool; its system message is kept as it is, unless it holds a
    ``<tools>`` element, whose tools the row would then not declare.
    """
    if tools is None:
        if "<tools>" in text:
            raise Unmappable("the system message lists tools the row does not declare")
        return text
    for before, after in TOOL_PROMPTS:
        prompt = f"{before}<tools>\n{tools}\n</tools>{after}"
        if prompt in text:
            return text.replace(prompt, "", 1).strip()
    raise Unmappable("the system message carries no Hermes tool prompt with the row's tools")


CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def split_calls(text: str) -> tuple[str, list[dict]]:
    """The content of an assistant turn and its calls, one ``{"name", "arguments"}`` object per ``<tool_call>``.

    A turn without a block is all content, byte for byte. In a turn with blocks the content is the text before the
    first block, less the whitespace that separates it from that block; between and after the blocks there may be
    only whitespace, since a message renders its content before its calls.
    """
    pieces = CALL.split(text)  # text, block, text, block, ..., text
    if any(tag in piece for piece in pieces[0::2] for tag in ("<tool_call>", "</tool_call>")):
        raise Unmappable("a <tool_call> tag without its pair")
    bodies = pieces[1::2]
    if not bodies:
        return text, []
    if any(piece.strip() for piece in pieces[2::2]):
        raise Unmappable("text after a <tool_call> block")
    return pieces[0].rstrip(), [_call(body) for body in bodies]


def _call(body: str) -> dict:
    try:
        call = json.loads(body)
    except json.JSONDecodeError:
        raise Unmappable("a <tool_call> block that is not JSON") from None
    if not (
        isinstance(call, dict)
        and set(call) == {"name", "arguments"}
        and isinstance(call["name"], str)
        and isinstance(call["arguments"], dict)
    ):
        raise Unmappable('a <tool_call> block that is not {"name": <string>, "arguments": {<object>}}')
    return call


RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


def responses(text: str) -> list:
    """The JSON of each ``<tool_response>`` block of a tool turn, in order; around the blocks only whitespace."""
    pieces = RESPONSE.split(text)  # text, block, text, block, ..., text
    if any(tag in piece for piece in pieces[0::2] for tag in ("<tool_response>", "</tool_response>")):
        raise Unmappable("a <tool_response> tag without its pair")
    if len(pieces) == 1:
        raise Unmappable("a tool turn without a <tool_response> block")
    if any(piece.strip() for piece in pieces[0::2]):
        raise Unmappable("text outside the <tool_response> blocks")
    found = []
    for body in pieces[1::2]:
        try:
            found.append(json.loads(body))
        except json.JSONDecodeError:
            raise Unmappable("a <tool_response> block that is not JSON") from None
    return found


def set_name(config: str) -> str:
    return "hermes-" + config.replace("_", "-")


def origin(config: str, row: dict, index: int, turn: int) -> dict:
    """Where a case came from: the file, the row by its index and its id, and the turn the case ends at."""
    filename, sha256 = CONFIGS[config]
    found = {"dataset": "hermes", "source": SOURCE, "sha256": sha256, "file": filename}
    return {**found, "row": index, "row_id": row["id"], "turn": turn, "license": LICENSE}


def row_cases(row: dict, index: int, config: str) -> tuple[list[dict], list[dict]]:
    """The render and parse cases of one row.

    The turns become OpenAI chat messages in order. A render case ends at each user turn an assistant turn answers,
    and a parse case is each assistant turn, its request every message before it. Calls get the ids ``call_<n>``,
    numbered across the row, and each tool result the id of the call it answers, by order. The ids leave the row out,
    so two rows that open with the same turns give the same cases there.
    """
    tools = _tools(row["tools"])
    declared = {tool["function"]["name"] for tool in tools}
    turns = row["conversations"]
    messages: list[dict] = []
    render: list[dict] = []
    parse: list[dict] = []
    calls: list[dict] = []  # the calls of the last assistant turn, until a tool turn answers them
    made = 0
    for turn, entry in enumerate(turns):
        source, text = entry["from"], entry["value"]
        if source == "tool":
            results = responses(text)
            if len(results) > len(calls):
                raise Unmappable("a response without a call")
            if len(results) < len(calls):
                raise Unmappable("a call without a response")
            for call, result in zip(calls, results, strict=True):
                content = json.dumps(result, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            calls = []
            continue
        if calls:
            raise Unmappable("a call without a response")
        if source == "system":
            content = system_message(text, row["tools"] if tools else None)
            if content:
                messages.append({"role": "system", "content": content})
        elif source == "human":
            messages.append({"role": "user", "content": text})
            if turn + 1 < len(turns) and turns[turn + 1]["from"] == "gpt":
                render.append(_line(config, row, index, turn, _request(messages, tools)))
        elif source == "gpt":
            content, found = split_calls(text)
            message: dict = {"content": content}
            calls = []
            for call in found:
                if call["name"] not in declared:
                    raise Unmappable(f"a call to {call['name']}, which the row's tools do not declare")
                arguments = json.dumps(call["arguments"], ensure_ascii=False)
                function = {"name": call["name"], "arguments": arguments}
                calls.append({"id": f"call_{made}", "type": "function", "function": function})
                made += 1
            if calls:
                message["tool_calls"] = calls
            parse.append(_line(config, row, index, turn, _request(messages, tools), message))
            messages.append({"role": "assistant", **message})
        else:
            raise Unmappable(f"a turn from {source!r}")
    return render, parse


def _tools(field: str) -> list[dict]:
    """The row's ``tools`` field as the request's tools, each ``{"type": "function", "function": {"name", ...}}``."""
    tools = json.loads(field) or []
    for tool in tools:
        if not (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
        ):
            raise Unmappable("a tool that is not an OpenAI function tool")
    return tools


def _request(messages: list[dict], tools: list[dict]) -> dict:
    request: dict = {"messages": list(messages)}
    if tools:
        request["tools"] = tools
    request["add_generation_prompt"] = True
    return request


def _line(config: str, row: dict, index: int, turn: int, request: dict, message: dict | None = None) -> dict:
    line = {"name": f"{set_name(config)}-{index}-{turn}", "request": request}
    if message is not None:
        line["message"] = message
    topic = " / ".join(part for part in (row.get("category"), row.get("subcategory")) if part)
    line["notes"] = f"Hermes {config} row {index} turn {turn}: {topic}"
    line["origin"] = origin(config, row, index, turn)
    return line


def build_sets(
    rows: dict[str, list[dict]],
    stride: int = STRIDE,
    skipped: list[tuple[str, int, str]] | None = None,
    repeated: list[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)`` from every ``stride``-th row of each config, from its ``FIRST_ROW``.

    A row that cannot be mapped gives no case, and is appended to ``skipped`` as ``(config, index, reason)``. A case
    that repeats an earlier one is left out and appended to ``repeated`` (``leave_out_repeats``): rows that open with
    the same turns give the cases of those turns once.
    """
    sets: dict[tuple[str, str], list[dict]] = {}
    for config, found in rows.items():
        cases: dict[str, list[dict]] = {"render": [], "parse": []}
        for index in range(FIRST_ROW[config], len(found), stride):
            try:
                row_render, row_parse = row_cases(found[index], index, config)
            except Unmappable as err:
                if skipped is not None:
                    skipped.append((config, index, str(err)))
                continue
            cases["render"] += row_render
            cases["parse"] += row_parse
        for kind, lines in cases.items():
            if lines:
                sets[(kind, set_name(config))] = lines
    return leave_out_repeats(sets, repeated)


def leave_out_repeats(
    sets: dict[tuple[str, str], list[dict]], repeated: list[tuple[str, str]] | None = None
) -> dict[tuple[str, str], list[dict]]:
    """The sets less each case whose request (and, for a parse case, message) is an earlier case's of its kind.

    Earlier is in any set before it, in the order of ``sets``, or before it in its own set. Each case left out is
    appended to ``repeated`` as ``(its name, the earlier case's name)``, and a set left with no case is dropped, so
    every case kept is distinct.
    """
    first: dict[str, dict[str, str]] = {}  # per kind: a case's request and message as written -> its name
    kept: dict[tuple[str, str], list[dict]] = {}
    for (kind, name), lines in sets.items():
        seen = first.setdefault(kind, {})
        for line in lines:
            key = json.dumps([line["request"], line.get("message")], ensure_ascii=False)
            if key in seen:
                if repeated is not None:
                    repeated.append((line["name"], seen[key]))
                continue
            seen[key] = line["name"]
            kept.setdefault((kind, name), []).append(line)
    return kept


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[Path]:
    """Write every set, and remove ``hermes-*`` files no config writes any more."""
    return corpus_sets.write(sets, corpus_dir, "hermes-")


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[str]:
    """One line per set file that differs from a fresh import; empty when the corpus is what the import writes."""
    return corpus_sets.check(sets, corpus_dir, "hermes-", "Hermes config")


def run(args: argparse.Namespace) -> int:
    card = hf.fetch(REPO, REVISION, hf.CARD, CARD_SHA256).read_bytes().decode("utf-8")
    hf.check_card_license(REPO, card, CARD_LICENSE)
    rows = {
        config: read_rows(hf.fetch(REPO, REVISION, filename, sha256)) for config, (filename, sha256) in CONFIGS.items()
    }
    skipped: list[tuple[str, int, str]] = []
    repeated: list[tuple[str, str]] = []
    sets = build_sets(rows, skipped=skipped, repeated=repeated)
    if args.check:
        problems = check_sets(sets, args.corpus)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the Hermes sets equal a fresh import of {SOURCE}")
        return 1 if problems else 0
    for config, found in rows.items():
        taken = len(range(FIRST_ROW[config], len(found), STRIDE))
        print(f"{config}: {taken} of {len(found)} rows, from row {FIRST_ROW[config]} in steps of {STRIDE}")
    for (kind, name), lines in sorted(sets.items()):
        print(f"{args.corpus / kind / f'{name}.jsonl'}: {len(lines)} cases")
    rows_by_reason: dict[tuple[str, str], list[str]] = {}
    for config, index, why in skipped:
        rows_by_reason.setdefault((config, why), []).append(str(index))
    for (config, why), indices in rows_by_reason.items():
        print(f"no cases for {len(indices)} row(s) of {config} ({', '.join(indices)}): {why}")
    for name, earlier in repeated:
        print(f"no case {name}: it repeats {earlier}")
    write_sets(sets, args.corpus)
    return 0
