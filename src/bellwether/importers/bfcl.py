"""BFCL's single-turn categories as corpus sets, as smg's weekly run sends them.

The data comes from the ``bfcl-eval`` wheel that smg's weekly run pins, read as a zip: nothing from it is
installed and none of its code runs. The weekly run uses BFCL's function-calling mode through
``OpenAICompletionsHandler``, so a request is the case's messages plus the functions turned into tools the way
that handler does: BFCL's language hint and Java/JavaScript rewrite (``_func_doc_language_specific_pre_processing``
in ``bfcl_eval/utils.py``), then ``convert_to_tool`` for OpenAI chat completions (``bfcl_eval/model_handler/utils.py``).
This module imports nothing beyond the standard library.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import zipfile
from pathlib import Path

PROJECT = "bfcl-eval"
VERSION = "2026.3.23"
WHEEL = "bfcl_eval-2026.3.23-py3-none-any.whl"
SHA256 = "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"
LICENSE = "Apache-2.0"
METADATA_LICENSE = "Apache 2.0"  # the License field of the reviewed wheel's METADATA; LICENSE is its SPDX name
SOURCE = f"pypi:{PROJECT}=={VERSION}"
DATA = "bfcl_eval/data"
TEMPERATURE = 0.001
# The single-turn categories of the weekly run (.github/workflows/nightly-bfcl.yml in smg), in its order.
CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
)

# bfcl_eval/constants/type_mappings.py, GORILLA_TO_OPENAPI
TYPE_MAP = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

HINTS = {
    "java": " Note that the provided function is in Java 8 SDK syntax.",
    "javascript": " Note that the provided function is in JavaScript syntax.",
    "python": " Note that the provided function is in Python 3 syntax.",
}


def language(category: str) -> str:
    if "javascript" in category:
        return "javascript"
    if "java" in category:
        return "java"
    return "python"


def prepare_functions(functions: list[dict], category: str) -> list[dict]:
    """BFCL's language hint on every description, and Java/JavaScript parameters rewritten as strings."""
    functions = copy.deepcopy(functions)
    lang = language(category)
    for item in functions:
        item["description"] = item["description"] + HINTS[lang]
        if lang == "python":
            continue
        name = "Java" if lang == "java" else "JavaScript"
        for value in item["parameters"]["properties"].values():
            if value["type"] == "any":
                value["description"] += f" This parameter can be of any type of {name} object in string representation."
            else:
                value["description"] += f" This is {name} {value['type']} type parameter in string representation."
            if (lang == "java" and value["type"] in ("ArrayList", "Array")) or (
                lang == "javascript" and value["type"] == "array"
            ):
                value["description"] += (
                    f" The list elements are of type {value['items']['type']}; they are not in string representation."
                )
                del value["items"]
            if lang == "javascript" and value["type"] == "dict" and "properties" in value:
                value["description"] += (
                    " The dictionary entries have the following schema; they are not in string representation. "
                    + json.dumps(value["properties"])
                )
                del value["properties"]
            value["type"] = "string"
    return functions


def to_tools(functions: list[dict]) -> list[dict]:
    """``convert_to_tool`` for OpenAI chat completions: the `tools` the weekly run sends."""
    tools = []
    for item in copy.deepcopy(functions):
        item["name"] = item["name"].replace(".", "_")
        item["parameters"]["type"] = "object"
        item["parameters"]["properties"] = _cast(item["parameters"]["properties"])
        tools.append({"type": "function", "function": item})
    return tools


def _cast(properties: dict) -> dict:
    """``_cast_to_openai_type`` with BFCL's map: types, the float note, nested properties and items."""
    for value in properties.values():
        if "type" not in value:
            value["type"] = "string"
        else:
            kind = value["type"]
            if kind == "float":
                value["format"] = "float"
                value["description"] += " This is a float type value."
            value["type"] = TYPE_MAP.get(kind, "string")
        if value["type"] in ("array", "object"):
            if "properties" in value:
                value["properties"] = _cast(value["properties"])
            elif "items" in value:
                items = value["items"]
                items["type"] = TYPE_MAP[items["type"]]
                if items["type"] == "array" and "items" in items:
                    items["items"]["type"] = TYPE_MAP[items["items"]["type"]]
                elif items["type"] == "object" and "properties" in items:
                    items["properties"] = _cast(items["properties"])
    return properties


OMIT = object()


def pick(options: list):
    """The first acceptable value that is not BFCL's "may be omitted" marker (``""``), or ``OMIT``."""
    for option in options:
        if option != "":
            return realize(option)
    return OMIT


def realize(option):
    """An acceptable value as a call argument.

    BFCL's checker (``dict_checker``, ``list_dict_checker``) reads a dict option as one list of acceptable values
    per key, and a list of dicts as such dict options in order; anything else is the value itself.
    """
    if isinstance(option, dict):
        return _realize_dict(option)
    if isinstance(option, list) and option and all(isinstance(item, dict) for item in option):
        return [_realize_dict(item) for item in option]
    return option


def _realize_dict(option: dict) -> dict:
    value = {}
    for key, acceptable in option.items():
        if not isinstance(acceptable, list):
            raise ValueError(f"dict option key {key!r} holds {acceptable!r}, not a list of acceptable values")
        chosen = next((item for item in acceptable if item != ""), OMIT)
        if chosen is not OMIT:
            value[key] = chosen
    return value


def string_valued(answer: dict) -> bool:
    """Whether every value the call would carry is a string.

    BFCL's Java and JavaScript categories ask for every argument in string representation, and its checker refuses
    any other type, but their ground truth stores the converted value (``5``, ``true``, a dict). Until the importer
    writes those values in the string form BFCL's converters read back (#26), such a row has no parse case.
    """
    for entry in answer["ground_truth"]:
        for options in next(iter(entry.values())).values():
            chosen = next((option for option in options if option != ""), OMIT)
            if chosen is not OMIT and not isinstance(chosen, str):
                return False
    return True


def message_for(answer: dict) -> dict:
    """The assistant message a parser must return for one BFCL ground truth: one call per entry, in order."""
    calls = []
    for entry in answer["ground_truth"]:
        ((name, params),) = entry.items()
        arguments = {}
        for param, options in params.items():
            value = pick(options)
            if value is not OMIT:
                arguments[param] = value
        call = {"name": name.replace(".", "_"), "arguments": json.dumps(arguments, ensure_ascii=False)}
        calls.append({"type": "function", "function": call})
    return {"content": "", "tool_calls": calls}


def question_file(category: str) -> str:
    return f"{DATA}/BFCL_v4_{category}.json"


def answer_file(category: str) -> str:
    return f"{DATA}/possible_answer/BFCL_v4_{category}.json"


def _jsonl(wheel: zipfile.ZipFile, member: str) -> list[dict]:
    return [json.loads(line) for line in wheel.read(member).decode("utf-8").splitlines() if line.strip()]


def read_rows(wheel: zipfile.ZipFile, category: str) -> list[dict]:
    return _jsonl(wheel, question_file(category))


def read_answers(wheel: zipfile.ZipFile, category: str) -> dict[str, dict]:
    if answer_file(category) not in wheel.namelist():
        return {}
    return {answer["id"]: answer for answer in _jsonl(wheel, answer_file(category))}


def request_for(row: dict, category: str) -> dict:
    """The body the weekly run sends for one row, without ``model``, in the order its client builds it."""
    if len(row["question"]) != 1:
        raise ValueError(f"{row['id']}: a single-turn row with {len(row['question'])} turns")
    request = {"messages": row["question"][0], "temperature": TEMPERATURE, "store": False}
    tools = to_tools(prepare_functions(row["function"], category))
    if tools:
        request["tools"] = tools
    return request


def case_name(row_id: str) -> str:
    return "bfcl-" + re.sub(r"[^a-z0-9]+", "-", row_id.lower()).strip("-")


def set_name(category: str) -> str:
    return "bfcl-" + category.replace("_", "-")


def origin(category: str, row_id: str, answered: bool = False) -> dict:
    """Where a case came from; a parse case also names the file its message came from."""
    found = {"dataset": "bfcl", "source": SOURCE, "sha256": SHA256, "file": question_file(category)}
    if answered:
        found["answer_file"] = answer_file(category)
    return {**found, "row": row_id, "license": LICENSE}


def build_sets(wheel: zipfile.ZipFile, categories: tuple[str, ...] | None = None) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)``: a render case for every row, a parse case where BFCL has an answer.

    A Java or JavaScript row gets its parse case only when every value it carries is a string (``string_valued``).
    """
    sets: dict[tuple[str, str], list[dict]] = {}
    seen: dict[str, str] = {}
    for category in categories or CATEGORIES:
        answers = read_answers(wheel, category)
        render, parse = [], []
        for row in read_rows(wheel, category):
            name = case_name(row["id"])
            if name in seen:
                raise ValueError(f"rows {seen[name]!r} and {row['id']!r} both become the case name {name}")
            seen[name] = row["id"]
            request = request_for(row, category)
            notes = f"BFCL {category} {row['id']}"
            render.append({"name": name, "request": request, "notes": notes, "origin": origin(category, row["id"])})
            answer = answers.get(row["id"])
            if answer is not None and (language(category) == "python" or string_valued(answer)):
                message = message_for(answer)
                line = {"name": name, "request": request, "message": message, "notes": notes}
                parse.append({**line, "origin": origin(category, row["id"], answered=True)})
        sets[("render", set_name(category))] = render
        if parse:
            sets[("parse", set_name(category))] = parse
    return sets


def _text(lines: list[dict]) -> str:
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[Path]:
    """Write every set, and remove ``bfcl-*`` files no category writes any more."""
    written = []
    for (kind, name), lines in sorted(sets.items()):
        path = corpus_dir / kind / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_text(lines).encode("utf-8"))
        written.append(path)
    for kind in ("render", "parse"):
        for stale in sorted((corpus_dir / kind).glob("bfcl-*.jsonl")):
            if stale not in written:
                stale.unlink()
    return written


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[str]:
    """One line per set file that differs from a fresh import; empty when the corpus is what the import writes."""
    expected = {corpus_dir / kind / f"{name}.jsonl": _text(lines) for (kind, name), lines in sets.items()}
    problems = []
    for path, text in sorted(expected.items()):
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif path.read_bytes().decode("utf-8") != text:
            problems.append(f"{path}: differs from a fresh import")
    for kind in ("render", "parse"):
        for path in sorted((corpus_dir / kind).glob("bfcl-*.jsonl")):
            if path not in expected:
                problems.append(f"{path}: no BFCL category writes it")
    return problems


def check_license(wheel: zipfile.ZipFile) -> None:
    """Refuse a wheel whose METADATA header does not carry the license this importer was reviewed for."""
    found = None
    members = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
    if members:
        for line in wheel.read(members[0]).decode("utf-8").splitlines():
            if not line.strip():
                break  # the header ends at the first blank line; the description follows
            if line.startswith(("License:", "License-Expression:")):
                found = line.split(":", 1)[1].strip()
                break
    if found != METADATA_LICENSE:
        raise ValueError(
            f"{WHEEL}: license {found!r} is not the reviewed {METADATA_LICENSE!r}; review it before importing"
        )


def run(args: argparse.Namespace) -> int:
    from . import pypi

    with zipfile.ZipFile(pypi.fetch(PROJECT, VERSION, WHEEL, SHA256, cache=args.cache)) as wheel:
        check_license(wheel)
        sets = build_sets(wheel)
    if args.check:
        problems = check_sets(sets, args.corpus)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the BFCL sets equal a fresh import of {SOURCE}")
        return 1 if problems else 0
    for (kind, name), lines in sorted(sets.items()):
        print(f"{args.corpus / kind / f'{name}.jsonl'}: {len(lines)} cases")
    for category in CATEGORIES:
        if language(category) != "python" and ("render", set_name(category)) in sets:
            render = len(sets[("render", set_name(category))])
            parse = len(sets.get(("parse", set_name(category)), []))
            print(f"{set_name(category)}: {render - parse} rows carry a value that is not a string; render only (#26)")
    write_sets(sets, args.corpus)
    return 0
