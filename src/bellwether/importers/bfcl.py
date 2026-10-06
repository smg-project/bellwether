"""BFCL's categories as corpus sets, as smg's weekly run sends them: each single-turn row, and the first turn of each
multi_turn row (``CATEGORIES`` says why multi_turn_long_context is left out).

The data comes from the ``bfcl-eval`` wheel that smg's weekly run pins, read as a zip: nothing from it is
installed and none of its code runs. The weekly run uses BFCL's function-calling mode through
``OpenAICompletionsHandler``, so a request is the case's messages plus the functions turned into tools the way
that handler does: BFCL's language hint and Java/JavaScript rewrite (``_func_doc_language_specific_pre_processing``
in ``bfcl_eval/utils.py``), then ``convert_to_tool`` for OpenAI chat completions (``bfcl_eval/model_handler/utils.py``).
The wheel has no LICENSE file, so the import copies the gorilla repository's, fetched by the commit the wheel was built
from.

A multi_turn row names its classes instead of carrying functions, and its first request is the first step of
``inference_multi_turn_FC`` (``bfcl_eval/model_handler/base_handler.py:95``): no system prompt of the handler's own
(line 169), the tools of the functions the first turn offers (line 170, ``first_turn_functions``), and the first turn's
messages (line 192). Every later request carries the results of the earlier calls, which only BFCL's simulators
produce, so later turns are not imported. This module, like the set writer and the fetcher it shares with the other
importers (``corpus_sets``, ``github``), imports nothing beyond the standard library.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
import zipfile
from pathlib import Path

from bellwether import jsonl

from . import corpus_sets, github

PROJECT = "bfcl-eval"
VERSION = "2026.3.23"
WHEEL = "bfcl_eval-2026.3.23-py3-none-any.whl"
SHA256 = "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"
LICENSE = "Apache-2.0"
METADATA_LICENSE = "Apache 2.0"  # the License field of the reviewed wheel's METADATA; LICENSE is its SPDX name
SOURCE = f"pypi:{PROJECT}=={VERSION}"
# Apache-2.0 4(a) asks that a copy of the License go with the work, and the wheel has none, so the import copies the
# repository's root LICENSE at the commit the wheel was built from. The wheel's 183 files under bfcl_eval/ are that
# commit's, byte for byte, and BFCL's publish workflow names a build of main by its UTC date, with a serial after the
# day's first commit: 2026.3.23 is the first commit of 2026-03-23, and its only one. berkeley-function-call-leaderboard/
# has no LICENSE of its own, and the repository has no NOTICE file for 4(d) to carry.
OWNER = "ShishirPatil"
REPO = "gorilla"
COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
LICENSE_FILE = "LICENSE"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
# Where the import writes the pinned LICENSE, under the corpus root.
LICENSE_COPY = "licenses/bfcl-LICENSE"
DATA = "bfcl_eval/data"
# The multi_turn classes' function docs, and the module that maps each class to its doc file and to its source.
FUNC_DOC = f"{DATA}/multi_turn_func_doc"
BACKEND_CONFIG = "bfcl_eval/constants/executable_backend_config.py"
FILE_MAPPING = "MULTI_TURN_FUNC_DOC_FILE_MAPPING"
CLASS_MAPPING = "CLASS_FILE_PATH_MAPPING"
TEMPERATURE = 0.001
# The categories of the weekly run (.github/workflows/nightly-bfcl.yml in smg), in its order: 13 single-turn, then
# multi_turn ones, of which the first turn is imported. multi_turn_long_context is left out for size: its first turns
# would take the BFCL corpus past the 50 MB a source may keep as plain JSON Lines (corpus/README.md has the figures).
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
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
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


def multi_turn(category: str) -> bool:
    return "multi_turn" in category


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
NOT_STRINGS = "carries a value that is not a string, which the category's checker refuses (#26)"
NO_FIRST_CALL = "the first turn has no gold call, so there is no message to parse"


class Unanswerable(ValueError):
    """A BFCL ground truth for which no call the rule builds passes BFCL's own checker."""


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


def message_for(answer: dict, functions: list[dict]) -> dict:
    """The assistant message a parser must return for one BFCL ground truth: one call per entry, in order.

    Each call is held to the rules BFCL's checker applies to it: a parameter its function does not declare is left
    out when the ground truth lets it be omitted, and a ground truth that leaves a required parameter without a value,
    or insists on one the function lacks, raises ``Unanswerable``: no call from it would pass.
    """
    declared = {function["name"]: function["parameters"] for function in functions}
    calls = []
    for entry in answer["ground_truth"]:
        ((name, params),) = entry.items()
        if name not in declared:
            raise Unanswerable(f"the ground truth calls {name}, which the row does not define")
        properties = declared[name].get("properties", {})
        arguments = {}
        for param, options in params.items():
            if param not in properties:
                if "" in options:
                    continue
                raise Unanswerable(f"{name} has no parameter {param!r}, which the ground truth requires")
            value = pick(options)
            if value is not OMIT:
                arguments[param] = value
        missing = [param for param in declared[name].get("required", []) if param not in arguments]
        if missing:
            raise Unanswerable(f"{name} requires {', '.join(missing)}, which the ground truth gives no value")
        call = {"name": name.replace(".", "_"), "arguments": json.dumps(arguments, ensure_ascii=False)}
        calls.append({"type": "function", "function": call})
    return {"content": "", "tool_calls": calls}


def first_turn_message(answer: dict, functions: list[dict], defs: dict[str, list[str]]) -> dict:
    """The assistant message a parser must return for a multi_turn row's first turn: its gold calls, in order.

    BFCL writes each gold call as Python source, ``cd(folder='document')``, and its executor runs it on the class.
    Positional values take the parameters of the method's ``def`` in order (``defs``: each method's, from
    ``read_func_defs``), which the function doc may list otherwise. Keyword values keep their names, and each value is
    read with ``ast.literal_eval``. A call this cannot read stops the import, naming the row and the call: one that is
    not a call to a named function, passes keywords by unpacking, gives a value that is not a literal, or passes values
    by position to a function with no ``def``. Each call is held to the rules ``message_for`` applies: a function the
    first turn offers, parameters it declares, each given once, and every required one given; a call that breaks one
    raises ``Unanswerable``.
    """
    declared = {function["name"]: function["parameters"] for function in functions}
    calls = []
    for text in answer["ground_truth"][0]:
        name, arguments = _read_call(answer["id"], text, declared, defs)
        call = {"name": name.replace(".", "_"), "arguments": json.dumps(arguments, ensure_ascii=False)}
        calls.append({"type": "function", "function": call})
    return {"content": "", "tool_calls": calls}


def _read_call(row: str, text: str, declared: dict[str, dict], defs: dict[str, list[str]]) -> tuple[str, dict]:
    """One call of the row's ground truth: the function's name, and its arguments by parameter name."""
    cannot = f"{row}: cannot read {text!r}:"
    try:
        node = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        node = None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"{cannot} it is not a call to a named function")
    if any(keyword.arg is None for keyword in node.keywords):
        raise ValueError(f"{cannot} it passes keywords by unpacking")
    values = [_literal(value, cannot) for value in node.args]
    keywords = [(keyword.arg, _literal(keyword.value, cannot)) for keyword in node.keywords]
    name = node.func.id
    if name not in declared:
        raise Unanswerable(f"the ground truth calls {name}, which the first turn does not offer")
    if values and name not in defs:
        raise ValueError(f"{cannot} BFCL's source has no def of {name} to bind its positional values to")
    positional = defs.get(name, [])
    if len(values) > len(positional):
        passed = len(values)
        raise Unanswerable(
            f"the def of {name} takes {len(positional)} parameters, and the ground truth passes {passed} by position"
        )
    properties = declared[name].get("properties", {})
    arguments = {}
    for param, value in [*zip(positional, values, strict=False), *keywords]:
        if param not in properties:
            raise Unanswerable(f"{name} has no parameter {param!r}, which the ground truth requires")
        if param in arguments:
            raise Unanswerable(f"{name} gets {param} both by position and by name")
        arguments[param] = value
    missing = [param for param in declared[name].get("required", []) if param not in arguments]
    if missing:
        raise Unanswerable(f"{name} requires {', '.join(missing)}, which the ground truth gives no value")
    return name, arguments


def _literal(node: ast.expr, cannot: str):
    """A value in a call, read with ``ast.literal_eval``; ``cannot`` starts the message when it is not a literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        raise ValueError(f"{cannot} {ast.unparse(node)} is not a literal value") from None


def func_doc_files(wheel: zipfile.ZipFile) -> dict[str, str]:
    """``MULTI_TURN_FUNC_DOC_FILE_MAPPING``, each multi_turn class's function doc file, read from the module's source.

    The value is parsed with ``ast`` and read with ``ast.literal_eval``, so the module is never imported or run.
    """
    for node in ast.parse(wheel.read(BACKEND_CONFIG).decode("utf-8")).body:
        if isinstance(node, ast.Assign) and [getattr(target, "id", None) for target in node.targets] == [FILE_MAPPING]:
            return ast.literal_eval(node.value)
    raise ValueError(f"{BACKEND_CONFIG} assigns no {FILE_MAPPING}")


def read_func_docs(wheel: zipfile.ZipFile) -> dict[str, list[dict]]:
    """Each multi_turn class's function docs, in file order."""
    return {name: _jsonl(wheel, f"{FUNC_DOC}/{file}") for name, file in func_doc_files(wheel).items()}


def class_files(wheel: zipfile.ZipFile) -> dict[str, str]:
    """``CLASS_FILE_PATH_MAPPING``, the module BFCL's executor loads each multi_turn class from, as its source file.

    The map's values are f-strings of ``BACKEND_PATH_PREFIX``, a string the module assigns before them. Each is joined
    from the parsed source, so the module is never imported or run.
    """
    strings: dict[str, str] = {}
    for node in ast.parse(wheel.read(BACKEND_CONFIG).decode("utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id == CLASS_MAPPING and isinstance(node.value, ast.Dict):
            pairs = zip(node.value.keys, node.value.values, strict=True)
            return {ast.literal_eval(key): _source_file(value, strings) for key, value in pairs}
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            strings[node.targets[0].id] = node.value.value
    raise ValueError(f"{BACKEND_CONFIG} assigns no {CLASS_MAPPING}")


def _source_file(node: ast.expr, strings: dict[str, str]) -> str:
    """The source file of a module named by a string, or by an f-string whose fields each name one of ``strings``."""
    module = ""
    for part in node.values if isinstance(node, ast.JoinedStr) else [node]:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            module += part.value
        elif (
            isinstance(part, ast.FormattedValue)
            and isinstance(part.value, ast.Name)
            and part.value.id in strings
            and part.conversion == -1
            and part.format_spec is None
        ):
            module += strings[part.value.id]
        else:
            raise ValueError(f"{BACKEND_CONFIG}: {ast.unparse(node)} is not a module name made of its strings")
    return module.replace(".", "/") + ".py"


def read_func_defs(wheel: zipfile.ZipFile) -> dict[str, dict[str, list[str]]]:
    """Each multi_turn class's methods, with the parameters each takes by position: its ``def``'s, after ``self``.

    BFCL's executor runs a gold call on an instance of the class (``execute_multi_turn_func_call``,
    ``bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py:13``), so a value passed by position takes the
    ``def``'s parameter, and a function doc may list them in another order (``TravelAPI.purchase_insurance`` swaps
    ``booking_id`` and ``insurance_cost``). Each class's source is parsed with ``ast``, never imported or run. Only the
    class's own ``def``s whose first parameter is ``self`` are listed: for any other, where a value goes is not known.
    """
    defs = {}
    for name, member in class_files(wheel).items():
        tree = ast.parse(wheel.read(member).decode("utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
        methods = [node for node in classes[-1].body if isinstance(node, ast.FunctionDef)] if classes else []
        params = {method.name: [arg.arg for arg in method.args.args] for method in methods}
        defs[name] = {method: names[1:] for method, names in params.items() if names[:1] == ["self"]}
    return defs


def first_turn_functions(row: dict, docs: dict[str, list[dict]]) -> list[dict]:
    """The functions a multi_turn row offers at its first turn, as BFCL builds them.

    BFCL gives a row the function docs of its ``involved_classes``, in that order, and takes out the first function of
    each name ``missed_function`` holds back (``populate_test_cases_with_predefined_functions``,
    ``bfcl_eval/utils.py:772``). The handler adds those back only at their own turn, with a fixed user message in place
    of the turn's (``inference_multi_turn_FC``, ``bfcl_eval/model_handler/base_handler.py:176``). A row that has a
    function added back at the first turn would change the first request, so it stops the import.
    """
    functions = [doc for name in row["involved_classes"] for doc in docs[name]]
    for turn, names in row.get("missed_function", {}).items():
        if turn == "0":
            raise ValueError(
                f"{row['id']}: BFCL adds {', '.join(names)} back at the first turn, with a fixed user message in place"
                " of the turn's, which this importer does not build"
            )
        for held in names:
            index = next((i for i, doc in enumerate(functions) if doc["name"] == held), None)
            if index is not None:
                functions.pop(index)
    return functions


def question_file(category: str) -> str:
    return f"{DATA}/BFCL_v4_{category}.json"


def answer_file(category: str) -> str:
    return f"{DATA}/possible_answer/BFCL_v4_{category}.json"


def _jsonl(wheel: zipfile.ZipFile, member: str) -> list[dict]:
    """The rows of a JSON Lines member of the wheel, in order."""
    return [row for _, row in jsonl.loads(wheel.read(member).decode("utf-8"), member)]


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
    return first_request(row, row["function"], category)


def first_request(row: dict, functions: list[dict], category: str) -> dict:
    """The first request for a row: its first turn's messages and the functions it offers there.

    ``OpenAICompletionsHandler`` starts every row with no message of its own (``_pre_query_processing_FC``,
    ``bfcl_eval/model_handler/api_inference/openai_completion.py:96``), adds the first turn's messages as they are
    (``add_first_turn_message_FC``, line 132) and sends ``tools`` only when there are any (``_query_FC``, line 79).
    """
    request = {"messages": row["question"][0], "temperature": TEMPERATURE, "store": False}
    tools = to_tools(prepare_functions(functions, category))
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


def build_sets(
    wheel: zipfile.ZipFile,
    categories: tuple[str, ...] | None = None,
    skipped: list[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)``: a render case for every row, a parse case where BFCL has an answer.

    A multi_turn row gives the request and the gold calls of its first turn. A Java or JavaScript row gets its parse
    case only when every value it carries is a string (``string_valued``), a multi_turn row only when its first turn
    has a gold call, and a row for which no call the rule builds passes BFCL's checker gets none (``Unanswerable``);
    each such row is appended to ``skipped`` with its reason.
    """
    sets: dict[tuple[str, str], list[dict]] = {}
    seen: dict[str, str] = {}
    docs: dict[str, list[dict]] | None = None
    defs: dict[str, dict[str, list[str]]] = {}
    for category in categories or CATEGORIES:
        answers = read_answers(wheel, category)
        render, parse = [], []
        for row in read_rows(wheel, category):
            name = case_name(row["id"])
            if name in seen:
                raise ValueError(f"rows {seen[name]!r} and {row['id']!r} both become the case name {name}")
            seen[name] = row["id"]
            notes = f"BFCL {category} {row['id']}"
            if multi_turn(category):
                if docs is None:
                    docs, defs = read_func_docs(wheel), read_func_defs(wheel)
                functions = first_turn_functions(row, docs)
                # BFCL's executor gives a method name two of the row's classes have to the later class.
                methods = {method: params for cls in row["involved_classes"] for method, params in defs[cls].items()}
                request = first_request(row, functions, category)
                notes += ", first turn"
            else:
                functions = row["function"]
                request = request_for(row, category)
            render.append({"name": name, "request": request, "notes": notes, "origin": origin(category, row["id"])})
            answer = answers.get(row["id"])
            if answer is None:
                continue
            if multi_turn(category) and not answer["ground_truth"][0]:
                why = NO_FIRST_CALL
            elif language(category) != "python" and not string_valued(answer):
                why = NOT_STRINGS
            else:
                try:
                    if multi_turn(category):
                        message = first_turn_message(answer, functions, methods)
                    else:
                        message = message_for(answer, functions)
                    why = None
                except Unanswerable as err:
                    why = str(err)
            if why is None:
                line = {"name": name, "request": request, "message": message, "notes": notes}
                parse.append({**line, "origin": origin(category, row["id"], answered=True)})
            elif skipped is not None:
                skipped.append((row["id"], why))
        sets[("render", set_name(category))] = render
        if parse:
            sets[("parse", set_name(category))] = parse
    return sets


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, license_text: bytes) -> list[Path]:
    """Write every set and the pinned LICENSE, and remove ``bfcl-*`` set files no category writes any more."""
    return corpus_sets.write(sets, corpus_dir, "bfcl-", {LICENSE_COPY: license_text})


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, license_text: bytes) -> list[str]:
    """One line per set file, or the LICENSE copy, that differs from a fresh import; empty when none does."""
    return corpus_sets.check(sets, corpus_dir, "bfcl-", "BFCL category", {LICENSE_COPY: license_text})


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


def check_license_file(text: bytes) -> None:
    """Refuse a LICENSE file whose heading is not the Apache License 2.0's.

    ``github.fetch`` holds the file to its pinned sha256. The heading is checked as well, as GSM8K's is, so that moving
    the pins to a commit whose LICENSE is no longer Apache-2.0 cannot pass on an updated hash alone.
    """
    if text.split()[:4] != [b"Apache", b"License", b"Version", b"2.0,"]:
        raise ValueError(f"{LICENSE_FILE} at {COMMIT}: not the Apache License 2.0; review it before importing")


def run(args: argparse.Namespace) -> int:
    from . import pypi

    license_text = github.fetch(OWNER, REPO, COMMIT, LICENSE_FILE, LICENSE_SHA256, cache=args.cache).read_bytes()
    check_license_file(license_text)
    skipped: list[tuple[str, str]] = []
    with zipfile.ZipFile(pypi.fetch(PROJECT, VERSION, WHEEL, SHA256, cache=args.cache)) as wheel:
        check_license(wheel)
        sets = build_sets(wheel, skipped=skipped)
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    if args.check:
        problems = check_sets(kept, args.corpus, license_text)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the BFCL sets equal a fresh import of {SOURCE}")
        return 1 if problems else 0
    corpus_sets.report("BFCL", sets, kept, repeats, args.corpus)
    corpus_sets.report_skipped(skipped, "parse case")
    write_sets(kept, args.corpus, license_text)
    return 0
