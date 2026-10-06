"""SWE-Hero's OpenHands trajectories as corpus sets: long agent conversations with tool calls and their results.

The data is one parquet shard of ``nvidia/SWE-Hero-openhands-trajectories`` at a pinned commit, with the dataset's
``tools.json`` and card, each read through the shared Hugging Face reader (``hf``), which checks its sha256 on every
use. The card's YAML front matter must carry the reviewed license (CC-BY-4.0), and a row is kept only when its
repository's license (the row's ``license``, an SPDX id) is one the card names: MIT, Apache-2.0, BSD-2-Clause or
BSD-3-Clause. Qwen3-Coder-480B-A35B-Instruct wrote the assistant turns, running in OpenHands.

A request is an OpenAI chat request: the trajectory's messages before a turn, unchanged, each tool message given the
id of the call it answers, and ``tools.json`` as the tools. A turn's parse case is its content and its calls, the
arguments the JSON strings the data holds. A case for every turn, each with its whole history, would grow with the
square of a trajectory's length, so the import samples: one row in ``STRIDE``, and in each the first, middle and last
of the assistant turns whose request fits in ``MAX_REQUEST_BYTES``. No message is ever cut to fit: a later turn gives
way to an earlier one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow.parquet as pq

from . import corpus_sets, hf

REPO = "nvidia/SWE-Hero-openhands-trajectories"
REVISION = "150bc119e52c647216fce285fd801f16b6fd745b"
SOURCE = hf.source(REPO, REVISION)
SHARD = 13
TOOLS_FILE = "tools.json"
SHARD_FILE = f"data/train-{SHARD:05d}-of-00014.parquet"
FILES = {  # every file the import reads, with its sha256, in the order it reads them
    hf.CARD: "018c3ca426aa3a4f8f3dc4082f8dd1da458a60ca5ce69175c890f60b7db95578",
    TOOLS_FILE: "d0f46e87e8d6b6d4eef8c01d6add2674ecaaff1260e08df5a4c2b162824c593c",
    SHARD_FILE: "936b195ed11b2b9be2e60cf9cb274dfc2f31d07745cbd6144658da988ce295a9",
}
COLUMNS = ["instance_id", "repo", "license", "trajectory_id", "trajectory"]  # the fields the import uses
LICENSE = "CC-BY-4.0"
CARD_LICENSE = "cc-by-4.0"  # the license in the reviewed card's YAML front matter
REPOSITORY_LICENSES = ("MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause")  # the card's filter, checked per row
SET = f"swehero-{SHARD}"
STRIDE = 20  # one row in STRIDE is sampled, counted from row 0
MAX_REQUEST_BYTES = 128_000  # a request's bytes in its corpus line
FUNCTION_KEYS = {"name", "description", "parameters", "strict"}  # OpenAI's function definition
FUNCTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")

LICENSE_NOT_ALLOWED = "the repository's license is not MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause"
UNPAIRED = "its tool results do not pair with the calls before them"
NOT_AN_OBJECT = "a call's arguments are not a JSON object string"
CALLS_OUTSIDE_A_TURN = "a message other than an assistant turn carries calls"
NO_TURN = "no assistant turn's request fits under the cap"


class Refused(ValueError):
    """A row the import leaves out: named under its reason, with a detail."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def tool_problem(tool) -> str | None:
    """What keeps ``tool`` from being an OpenAI chat tool, or None."""
    if not isinstance(tool, dict) or set(tool) != {"type", "function"} or tool["type"] != "function":
        return 'is not {"type": "function", "function": {...}}'
    function = tool["function"]
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or not FUNCTION_NAME.fullmatch(name):
        return "has no name of letters, digits, _ and -"
    extra = sorted(set(function) - FUNCTION_KEYS)
    if extra:
        return f"has keys OpenAI's function does not: {', '.join(extra)}"
    parameters = function.get("parameters", {"type": "object"})
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        return "has parameters that are not a JSON Schema object"
    return None


def check_tools(tools) -> list[dict]:
    """``tools.json`` as every request's ``tools``: checked to be OpenAI's shape, not converted."""
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{TOOLS_FILE}: not a list of tools")
    names: set[str] = set()
    for number, tool in enumerate(tools):
        problem = tool_problem(tool)
        if problem is None and tool["function"]["name"] in names:
            problem = f"repeats the name {tool['function']['name']}"
        if problem is not None:
            raise ValueError(f"{TOOLS_FILE}: tool {number} {problem}; review it before importing")
        names.add(tool["function"]["name"])
    return tools


def read_rows(path: Path) -> Iterator[dict]:
    """The shard's rows in order, with the fields the import uses, read a batch at a time."""
    with pq.ParquetFile(path) as shard:
        for batch in shard.iter_batches(batch_size=64, columns=COLUMNS):
            yield from batch.to_pylist()


def _not_json(constant: str):
    raise ValueError(f"{constant} is not JSON")


def openai_call(call: dict, index: int) -> dict:
    """One call of the assistant turn at ``index`` in OpenAI's key order, its arguments the JSON string the data holds.

    A call whose arguments are not a JSON object string refuses the row: it would go into every later request. The
    string must be JSON as RFC 8259 has it, so ``NaN`` and the infinities, which Python's json reads, refuse it too.
    """
    function = call["function"]
    arguments = function["arguments"]
    try:
        value = json.loads(arguments, parse_constant=_not_json) if isinstance(arguments, str) else None
    except ValueError:
        value = None
    if not isinstance(value, dict):
        raise Refused(NOT_AN_OBJECT, f"message {index} calls {function['name']}")
    return {"id": call["id"], "type": call["type"], "function": {"name": function["name"], "arguments": arguments}}


def messages_for(trajectory: list[dict]) -> list[dict]:
    """The trajectory as OpenAI chat messages, each tool message carrying the id of the call it answers.

    The data's tool messages carry no id: the k-th tool message after an assistant turn answers that turn's k-th call.
    A turn the conversation goes on past must be followed by exactly one result per call, or the row is refused; the
    last turn may have none, since the episode ends with its call (every trajectory in the shard ends with ``finish``).
    Keys come in OpenAI's order rather than the shard's alphabetical struct order, and ``tool_calls`` is kept on
    assistant turns that make calls only, since parquet gives every message the field and fills it with null. Calls on
    any other message refuse the row: the request has no place for them, and dropping them would change the
    conversation.
    """
    messages: list[dict] = []
    calls: list[str] = []
    answered = opened = 0
    for index, item in enumerate(trajectory):
        role = item["role"]
        if role != "assistant" and item["tool_calls"]:
            raise Refused(CALLS_OUTSIDE_A_TURN, f"message {index} is a {role} message with calls")
        if role == "tool":
            if answered == len(calls):
                raise Refused(UNPAIRED, f"message {index} is a tool result with no call to answer")
            messages.append({"role": "tool", "tool_call_id": calls[answered], "content": item["content"]})
            answered += 1
            continue
        if answered != len(calls):
            raise Refused(UNPAIRED, f"message {opened} makes {len(calls)} call(s) and {answered} result(s) follow")
        calls, answered = [], 0
        message = {"role": role, "content": item["content"]}
        if role == "assistant" and item["tool_calls"]:
            message["tool_calls"] = [openai_call(call, index) for call in item["tool_calls"]]
            calls, opened = [call["id"] for call in item["tool_calls"]], index
        messages.append(message)
    if 0 < answered < len(calls):
        raise Refused(UNPAIRED, f"message {opened} makes {len(calls)} call(s) and {answered} result(s) follow")
    return messages


def request_for(messages: list[dict], turn: int, tools: list[dict]) -> dict:
    """The request for the assistant turn at ``turn``: every message before it, unchanged, and the tools."""
    return {"messages": messages[:turn], "tools": tools}


def request_bytes(request: dict) -> int:
    """The bytes ``request`` takes in its corpus line."""
    return len(json.dumps(request, ensure_ascii=False).encode("utf-8"))


def turns(messages: list[dict], tools: list[dict], cap: int) -> list[int]:
    """The assistant turns a row gives cases for: the first, middle and last of those whose request fits in ``cap``.

    Each request holds the one before it, so the turns that fit are the first ones; a turn past the cap gives way to
    an earlier one, and no message is ever cut.
    """
    fitting = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if request_bytes(request_for(messages, index, tools)) > cap:
            break
        fitting.append(index)
    if not fitting:
        return []
    return sorted({fitting[0], fitting[(len(fitting) - 1) // 2], fitting[-1]})


def no_turn(messages: list[dict], tools: list[dict]) -> str:
    """Why a sampled row gives no turn: it has none, or its first request is past the cap."""
    first = next((index for index, message in enumerate(messages) if message["role"] == "assistant"), None)
    if first is None:
        return "the trajectory has no assistant turn"
    return f"the first assistant turn's request has {request_bytes(request_for(messages, first, tools))} bytes"


def origin(number: int, row: dict, turn: int) -> dict:
    """Where a case came from: the shard, its row and the turn in it, ``tools.json`` (the request's tools), and both
    licenses that bind the text, the repository's as the dataset labels it."""
    return {
        "dataset": "swehero",
        "source": SOURCE,
        "sha256": FILES[SHARD_FILE],
        "file": SHARD_FILE,
        "tools_sha256": FILES[TOOLS_FILE],
        "tools_file": TOOLS_FILE,
        "row": number,
        "turn": turn,
        "instance_id": row["instance_id"],
        "trajectory_id": row["trajectory_id"],
        "repository": row["repo"],
        "repository_license": row["license"],
        "license": LICENSE,
    }


def case_lines(number: int, row: dict, messages: list[dict], turn: int, tools: list[dict]) -> tuple[dict, dict]:
    """The render and the parse line for the assistant turn at ``turn`` of row ``number``.

    The parse case's message is the turn itself: its content and its calls, the arguments as the data holds them and
    no call id, since a parser makes its own. Its request, which the render case shares, is everything before it.
    """
    assistant = messages[turn]
    message = {"content": assistant["content"]}
    if "tool_calls" in assistant:
        message["tool_calls"] = [
            {"type": call["type"], "function": call["function"]} for call in assistant["tool_calls"]
        ]
    calls = ", ".join(call["function"]["name"] for call in message.get("tool_calls", [])) or "no call"
    name = f"{SET}-{number}-{turn}"
    request = request_for(messages, turn, tools)
    notes = f"SWE-Hero {row['instance_id']}: the assistant turn at message {turn} of {len(messages)} ({calls})"
    found = origin(number, row, turn)
    render = {"name": name, "request": request, "notes": notes, "origin": found}
    return render, {"name": name, "request": request, "message": message, "notes": notes, "origin": found}


def build_sets(
    rows: Iterable[dict], tools: list[dict], refused: list[tuple[int, str, str]] | None = None
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)``: a render and a parse case for each chosen turn of each sampled row.

    Every row is checked: one refused (its repository's license, its results, its arguments) is appended to
    ``refused`` with its reason and detail, sampled or not. A sampled row that gives no turn is appended too.
    """
    refused = [] if refused is None else refused
    render, parse = [], []
    for number, row in enumerate(rows):
        try:
            if row["license"] not in REPOSITORY_LICENSES:
                raise Refused(LICENSE_NOT_ALLOWED, str(row["license"]))
            messages = messages_for(row["trajectory"])
        except Refused as err:
            refused.append((number, err.reason, err.detail))
            continue
        if number % STRIDE:
            continue
        chosen = turns(messages, tools, MAX_REQUEST_BYTES)
        if not chosen:
            refused.append((number, NO_TURN, no_turn(messages, tools)))
        for turn in chosen:
            render_line, parse_line = case_lines(number, row, messages, turn, tools)
            render.append(render_line)
            parse.append(parse_line)
    return {("render", SET): render, ("parse", SET): parse} if render else {}


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[Path]:
    """Write every set, and remove ``swehero-*`` files no shard writes any more."""
    return corpus_sets.write(sets, corpus_dir, "swehero-")


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[str]:
    """One line per set file that differs from a fresh import; empty when the corpus is what the import writes."""
    return corpus_sets.check(sets, corpus_dir, "swehero-", "SWE-Hero shard")


def refusal_report(refused: list[tuple[int, str, str]]) -> list[str]:
    """One line per reason, with how many rows it refused and every one of them, each with its detail."""
    by_reason: dict[str, list[str]] = {}
    for number, reason, detail in refused:
        by_reason.setdefault(reason, []).append(f"{number}: {detail}")
    return [f"refused {len(rows)} row(s) ({', '.join(rows)}): {reason}" for reason, rows in by_reason.items()]


def run(args: argparse.Namespace) -> int:
    def fetch(name: str) -> Path:
        return hf.fetch(REPO, REVISION, name, FILES[name], cache=args.cache)

    # The card's license is checked before the shard is downloaded.
    hf.check_card_license(REPO, REVISION, FILES[hf.CARD], CARD_LICENSE, cache=args.cache)
    tools = check_tools(json.loads(fetch(TOOLS_FILE).read_text(encoding="utf-8")))
    refused: list[tuple[int, str, str]] = []
    sets = build_sets(read_rows(fetch(SHARD_FILE)), tools, refused)
    if args.check:
        problems = check_sets(sets, args.corpus)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the SWE-Hero sets equal a fresh import of {SOURCE} {SHARD_FILE}")
        return 1 if problems else 0
    for (kind, name), lines in sorted(sets.items()):
        size = len(corpus_sets.text(lines).encode("utf-8"))
        print(f"{args.corpus / kind / f'{name}.jsonl'}: {len(lines)} cases, {size / 1e6:.1f} MB")
    for line in refusal_report(refused):
        print(line)
    write_sets(sets, args.corpus)
    return 0
