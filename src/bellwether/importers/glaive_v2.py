"""glaive-function-calling-v2 as corpus sets: chats whose assistant turns call functions or answer in prose.

The data is the dataset's one JSON file at a pinned revision on the Hugging Face Hub, a list of rows ``{"system",
"chat"}``, both flat strings. ``hf.fetch`` keeps it under the import's ``--cache`` and checks it against a sha256 pinned
here on every use, and ``hf.check_card_license`` does the same for the dataset card, whose front matter carries the
license. A row becomes OpenAI chat messages:

- ``system`` is ``SYSTEM: `` and then either a lead-in sentence and one or more pretty-printed function objects, or a
  plain prompt. Each function becomes a tool, ``{"type": "function", "function": <the object>}``. The system message
  keeps what is neither the lead-in nor a function, and there is none when nothing remains.
- ``chat`` is turns headed ``USER: ``, ``ASSISTANT: `` or ``FUNCTION RESPONSE: ``, at its start or after a blank line.
  A call turn, ``<functioncall> {"name": ..., "arguments": '<JSON>'} <|endoftext|>``, is not valid JSON, since its
  arguments sit in single quotes. It becomes an assistant message with empty content and one call, whose id is
  ``call_<n>``, the chat's calls numbered from 0, and whose arguments are the quoted JSON written as the BFCL importer
  writes arguments. A function response becomes a ``tool`` message answering the call before it, and a prose turn its
  text without the ``<|endoftext|>`` that ends it. The ids leave the row out, so a chat that recurs in another row
  gives the same cases there.

Each assistant turn is a parse case, and each user turn an assistant answers a render case. The whole file maps to about
1.2 GB of plain JSON Lines, so the corpus holds a sample, every ``STEP``-th row by index, until a compressed corpus form
lands. A row these rules cannot map has no case, and the import names it with its reason. Beyond the standard library,
this module imports only what it shares with the other importers: the reader of pinned Hugging Face files (``hf``) and
the set writer (``corpus_sets``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import corpus_sets, hf

REPO = "glaiveai/glaive-function-calling-v2"
REVISION = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"
SOURCE = hf.source(REPO, REVISION)
DATA = "glaive-function-calling-v2.json"
DATA_SHA256 = "e9b5d671812b5ca2fbd7b625a37d5c99a19576c37252cdc806defe256aea6dad"
LICENSE = "Apache-2.0"
CARD_SHA256 = "39c78f1f56b86fcd159cadeb8feda8a9333db6ef5ca0ce6830731ac3c666838e"  # the dataset card, hf.CARD
CARD_LICENSE = "apache-2.0"  # the license in the reviewed card's front matter; LICENSE is its SPDX name
DATASET = "glaive-v2"
# The sample is every STEP-th row by index. The whole file maps to 1,235 MB of corpus; 31 keeps 40.1 MB of it, under
# corpus_sets.LIMIT, the 50 MB a source may keep plain.
STEP = 31
SET_SIZE = 5000  # cases per set file, of either kind, at most
NAMED = 50  # the rows a reason names at most; past that it names the first 50 and gives the count

SYSTEM = "SYSTEM: "
LEAD_IN = "You are a helpful assistant with access to the following functions. Use them if required -"
# A call as glaive writes it: the arguments are JSON inside single quotes, which makes the whole invalid JSON.
QUOTED = re.compile(r"""\{"name": "(?P<name>[^"]*)", "arguments": '(?P<arguments>.*)'\}""", re.S)
END = "<|endoftext|>"
CALL = "<functioncall>"
# A turn starts with its header at the start of the chat or after a blank line.
TURN = re.compile(r"(?:\A|\n\n+)(USER|ASSISTANT|FUNCTION RESPONSE): ")

CALL_FORM = 'a <functioncall> is not {"name": ..., "arguments": ...} with the arguments quoted or an object'
ARGUMENTS_NOT_JSON = "a call's arguments are not JSON"
ARGUMENTS_NOT_OBJECT = "a call's arguments are not a JSON object"
UNDECLARED = "a call names a function the row does not declare"
TEXT_BEFORE_CALL = "an assistant turn has text before its <functioncall>"
RESPONSE_WITHOUT_CALL = "a function response does not follow a call"
ASSISTANT_END = "an assistant turn does not end with its one <|endoftext|>"
STRAY_END = "a user turn or function response holds <|endoftext|>"


class Unmappable(ValueError):
    """A row the importer cannot map to cases; the message is the reason, the same for every row it applies to."""


def system_and_tools(system: str) -> tuple[str | None, list[dict]]:
    """The system message's text, or None when nothing remains, and the functions the row declares as tools.

    The field's ``SYSTEM: `` header, the lead-in sentence and the function objects after it are removed; the system
    message is the rest, without the whitespace around it.
    """
    text = system.removeprefix(SYSTEM)
    functions = []
    if text.startswith(LEAD_IN):
        text = text[len(LEAD_IN) :]
        decoder = json.JSONDecoder()
        position = _skip_space(text, 0)
        while text.startswith("{", position):
            function, position = decoder.raw_decode(text, position)
            functions.append(function)
            position = _skip_space(text, position)
        text = text[position:]
    return text.strip() or None, [{"type": "function", "function": function} for function in functions]


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def parse_call(text: str) -> tuple[str, str]:
    """The name and the arguments string of the call written between ``<functioncall>`` and ``<|endoftext|>``.

    The arguments are the text between the single quotes, read as JSON, or, in a call that is valid JSON as a whole,
    the object it holds. They are written as the BFCL importer writes them: ``json.dumps`` with raw Unicode, one line,
    the keys in their order.
    """
    match = QUOTED.fullmatch(text.strip())
    if match:
        name = match["name"]
        try:
            arguments = json.loads(match["arguments"], parse_constant=_not_json)
        except ValueError:
            raise Unmappable(ARGUMENTS_NOT_JSON) from None
    else:
        try:
            call = json.loads(text, parse_constant=_not_json)
        except ValueError:
            raise Unmappable(CALL_FORM) from None
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise Unmappable(CALL_FORM)
        name, arguments = call["name"], call["arguments"]
    if not isinstance(arguments, dict):
        raise Unmappable(ARGUMENTS_NOT_OBJECT)
    return name, json.dumps(arguments, ensure_ascii=False)


def _not_json(constant: str):
    """``NaN``, ``Infinity`` and ``-Infinity``: Python's json reads them, but they are not JSON."""
    raise ValueError(f"{constant} is not JSON")


def messages_for(row: dict) -> tuple[list[dict], list[dict]]:
    """A row as OpenAI chat messages, the system message first when there is one, and its tools.

    Each turn's text is taken without the whitespace around it, and the calls get the ids ``call_0``, ``call_1``, ...
    in the order they are made. ``Unmappable`` names the first rule the row breaks: an assistant turn ends with its one
    ``<|endoftext|>`` and no other turn holds one; a call turn is the call alone, to a function the row declares, with a
    JSON object for arguments; a function response follows a call.
    """
    system, tools = system_and_tools(row["system"])
    declared = {tool["function"]["name"] for tool in tools}
    messages = [] if system is None else [{"role": "system", "content": system}]
    calls = 0
    for header, text in _turns(row["chat"]):
        text = text.strip()
        if header == "ASSISTANT":
            if not text.endswith(END) or text.count(END) != 1:
                raise Unmappable(ASSISTANT_END)
            text = text[: -len(END)].strip()
            if CALL not in text:
                messages.append({"role": "assistant", "content": text})
                continue
            if not text.startswith(CALL):
                raise Unmappable(TEXT_BEFORE_CALL)
            name, arguments = parse_call(text[len(CALL) :])
            if name not in declared:
                raise Unmappable(UNDECLARED)
            call = {
                "id": f"call_{calls}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            calls += 1
            messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif END in text:
            raise Unmappable(STRAY_END)
        elif header == "USER":
            messages.append({"role": "user", "content": text})
        else:
            if not (messages and messages[-1].get("tool_calls")):
                raise Unmappable(RESPONSE_WITHOUT_CALL)
            messages.append({"role": "tool", "tool_call_id": messages[-1]["tool_calls"][0]["id"], "content": text})
    return messages, tools


def _turns(chat: str):
    """``(header, text)`` per turn of the chat, in order; a header after a single newline is text of the turn before."""
    headers = list(TURN.finditer(chat))
    for header, following in zip(headers, [*headers[1:], None], strict=True):
        yield header[1], chat[header.end() : following.start() if following else len(chat)]


def origin(row: int, turn: int) -> dict:
    """Where a case came from: the file, the row's index in it, and the chat turn the case ends at."""
    return {
        "dataset": DATASET,
        "source": SOURCE,
        "sha256": DATA_SHA256,
        "file": DATA,
        "row": row,
        "turn": turn,
        "license": LICENSE,
    }


def cases_for(index: int, messages: list[dict], tools: list[dict]) -> tuple[list[dict], list[dict]]:
    """Row ``index``'s render and parse cases, from its messages (``messages_for``).

    Each assistant turn is a parse case: its request is every message before it, its message is the turn. Each user
    turn an assistant turn answers is a render case: the request up to and including it. A case is named, and its
    origin located, by the turn's index in the chat, the system message not counted.
    """
    first = 1 if messages and messages[0]["role"] == "system" else 0
    render, parse = [], []
    for position in range(first, len(messages)):
        turn, message = position - first, messages[position]
        name = f"{DATASET}-{index}-{turn}"
        tail = {"notes": f"glaive-function-calling-v2 row {index} turn {turn}", "origin": origin(index, turn)}
        if message["role"] == "assistant":
            reply = {key: value for key, value in message.items() if key != "role"}
            request = _request(messages[:position], tools)
            parse.append({"name": name, "request": request, "message": reply, **tail})
        elif message["role"] == "user" and _role(messages, position + 1) == "assistant":
            request = _request(messages[: position + 1], tools)
            render.append({"name": name, "request": request, **tail})
    return render, parse


def _role(messages: list[dict], position: int) -> str | None:
    return messages[position]["role"] if position < len(messages) else None


def _request(messages: list[dict], tools: list[dict]) -> dict:
    """A chat completion body without ``model``: the messages, then the tools when the row declares any."""
    return {"messages": messages, "tools": tools} if tools else {"messages": messages}


def set_name(number: int) -> str:
    return f"{DATASET}-{number:02d}"


def build_sets(
    rows: list[dict], *, step: int, set_size: int, skipped: list[tuple[int, str]] | None = None
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)`` from every ``step``-th row, by index.

    Every row of the file is mapped, so that a row that cannot be is appended to ``skipped`` with its reason whether
    or not the sample takes it. A set holds whole rows, in index order, and at most ``set_size`` cases of either kind;
    the render set and the parse set of one number hold the same rows.
    """
    groups: list[tuple[list[dict], list[dict]]] = [([], [])]
    for index, row in enumerate(rows):
        try:
            messages, tools = messages_for(row)
        except Unmappable as err:
            if skipped is not None:
                skipped.append((index, str(err)))
            continue
        if index % step:
            continue
        row_render, row_parse = cases_for(index, messages, tools)
        render, parse = groups[-1]
        if (render or parse) and max(len(render) + len(row_render), len(parse) + len(row_parse)) > set_size:
            groups.append(([], []))
            render, parse = groups[-1]
        render += row_render
        parse += row_parse
    sets: dict[tuple[str, str], list[dict]] = {}
    for number, (render, parse) in enumerate(groups):
        for kind, lines in (("render", render), ("parse", parse)):
            if lines:
                sets[(kind, set_name(number))] = lines
    return sets


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[Path]:
    """Write every set, and remove ``glaive-v2-*`` files the sample no longer writes."""
    return corpus_sets.write(sets, corpus_dir, f"{DATASET}-")


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[str]:
    """One line per set file that differs from a fresh import; empty when the corpus is what the import writes."""
    return corpus_sets.check(sets, corpus_dir, f"{DATASET}-", "slice of the glaive-v2 sample")


def run(args: argparse.Namespace) -> int:
    hf.check_card_license(REPO, REVISION, CARD_SHA256, CARD_LICENSE, cache=args.cache)
    rows = json.loads(hf.fetch(REPO, REVISION, DATA, DATA_SHA256, cache=args.cache).read_bytes())
    skipped: list[tuple[int, str]] = []
    sets = build_sets(rows, step=STEP, set_size=SET_SIZE, skipped=skipped)
    if args.check:
        problems = check_sets(sets, args.corpus)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the glaive-v2 sets equal a fresh import of {SOURCE}")
        return 1 if problems else 0
    for (kind, name), lines in sorted(sets.items()):
        print(f"{args.corpus / kind / f'{name}.jsonl'}: {len(lines)} cases")
    counts = {kind: sum(len(lines) for (k, _), lines in sets.items() if k == kind) for kind in ("render", "parse")}
    print(
        f"sample: every row whose index is a multiple of {STEP}, {len(range(0, len(rows), STEP))} of {len(rows)} rows:"
        f" {counts['render']} render and {counts['parse']} parse cases"
    )
    report_skipped(skipped)
    write_sets(sets, args.corpus)
    return 0


def report_skipped(skipped: list[tuple[int, str]]) -> None:
    """Print each reason once, with the rows it left out across the whole file and how many of them are sampled."""
    rows_by_reason: dict[str, list[int]] = {}
    for index, why in skipped:
        rows_by_reason.setdefault(why, []).append(index)
    for why, rows in rows_by_reason.items():
        named = ", ".join(str(row) for row in rows[:NAMED])
        shown = f" ({named})" if len(rows) <= NAMED else f", the first {NAMED} ({named})"
        sampled = sum(1 for row in rows if row % STEP == 0)
        print(f"no case for {len(rows)} row(s){shown}, {sampled} in the sample: {why}")
