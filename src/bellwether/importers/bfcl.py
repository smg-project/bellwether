"""BFCL's single-turn categories as corpus sets, as smg's weekly run sends them.

The data comes from the ``bfcl-eval`` wheel that smg's weekly run pins, read as a zip: nothing from it is
installed and none of its code runs. The weekly run uses BFCL's function-calling mode through
``OpenAICompletionsHandler``, so a request is the case's messages plus the functions turned into tools the way
that handler does: BFCL's language hint and Java/JavaScript rewrite (``_func_doc_language_specific_pre_processing``
in ``bfcl_eval/utils.py``), then ``convert_to_tool`` for OpenAI chat completions (``bfcl_eval/model_handler/utils.py``).
This module imports nothing beyond the standard library.
"""

from __future__ import annotations

import copy
import json

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
