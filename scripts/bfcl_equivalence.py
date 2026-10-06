"""One-off check of the BFCL importer against BFCL's own code. Never run in bellwether's environment.

It runs bfcl-eval's code, so it runs in a throwaway virtual environment with a scrubbed environment: no
Hugging Face token and no other credential, HOME pointed at a temporary directory:

    UV=$(command -v uv); WHEEL=~/.cache/bellwether/datasets/bfcl_eval-2026.3.23-py3-none-any.whl
    SCRUB="env -i HOME=$(mktemp -d) PATH=/usr/bin:/bin"
    $SCRUB "$UV" venv /tmp/bfcl-check --python 3.12
    $SCRUB "$UV" pip install --python /tmp/bfcl-check/bin/python "bfcl-eval==2026.3.23" soundfile
    $SCRUB /tmp/bfcl-check/bin/python scripts/bfcl_equivalence.py "$WHEEL"

For every single-turn row it compares the importer's tools with BFCL's own conversion, key order included, and
for every row the importer writes a parse case for, runs BFCL's own checker on the call built from its ground truth.

For every row of the multi_turn categories the import writes, it compares the importer's first request with the one
BFCL's own handler sends: it runs ``OpenAICompletionsHandler.inference`` on the row as ``bfcl generate`` loads it and
stops it at its first request with a stub in place of the API call, so nothing reaches a model or the network. It also
compares the tools with ``convert_to_tool`` directly, and runs BFCL's ``multi_turn_checker`` on each first-turn parse
case's calls, decoded as the handler decodes a model's tool calls, against the first turn of the ground truth.
multi_turn categories named after the wheel are checked in place of those, so one the import does not write yet
can be checked too:

    $SCRUB /tmp/bfcl-check/bin/python scripts/bfcl_equivalence.py "$WHEEL" multi_turn_long_context
"""

import contextlib
import copy
import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

from bfcl_eval.constants.enums import Language, ModelStyle
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.utils import _func_doc_language_specific_pre_processing, load_dataset_entry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bellwether.importers import bfcl  # noqa: E402

LANGUAGES = {"python": Language.PYTHON, "java": Language.JAVA, "javascript": Language.JAVASCRIPT}
# The instances BFCL's executor keeps per row are named after this.
CHECKER_MODEL = "bellwether_equivalence"


def check_single_turn(wheel: zipfile.ZipFile) -> list[str]:
    # convert_func_name looks a model up by turning "_" back into "/", so the name must hold no "_".
    model = next(
        name
        for name, config in MODEL_CONFIG_MAPPING.items()
        if config.is_fc_model and config.underscore_to_dot and "_" not in name
    )
    tools_equal = tools_total = calls_valid = calls_total = unanswerable = 0
    failures: list[str] = []
    for category in bfcl.CATEGORIES:
        if bfcl.multi_turn(category):
            continue
        answers = bfcl.read_answers(wheel, category)
        for row in bfcl.read_rows(wheel, category):
            ours = bfcl.request_for(row, category).get("tools", [])
            prepared = _func_doc_language_specific_pre_processing(copy.deepcopy(row["function"]), category)
            theirs = convert_to_tool(prepared, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
            tools_total += 1
            if json.dumps(ours) == json.dumps(theirs):
                tools_equal += 1
            else:
                failures.append(f"tools differ: {row['id']}")
            answer = answers.get(row["id"])
            if answer is None or (bfcl.language(category) != "python" and not bfcl.string_valued(answer)):
                continue
            try:
                message = bfcl.message_for(answer, row["function"])
            except bfcl.Unanswerable:
                unanswerable += 1
                continue
            output = [{c["function"]["name"]: json.loads(c["function"]["arguments"])} for c in message["tool_calls"]]
            result = ast_checker(
                row["function"],
                output,
                answer["ground_truth"],
                LANGUAGES[bfcl.language(category)],
                category,
                model.replace("/", "_"),
            )
            calls_total += 1
            if result["valid"]:
                calls_valid += 1
            else:
                failures.append(f"call rejected: {row['id']}: {result['error']}")
    print(f"tools: {tools_equal} of {tools_total} equal to BFCL's convert_to_tool")
    print(f"calls: {calls_valid} of {calls_total} accepted by BFCL's ast_checker (as {model})")
    print(f"unanswerable ground truths left without a call: {unanswerable}")
    return failures


class FirstRequest(Exception):
    """Raised in place of the API call, carrying the arguments the handler would send."""


def first_request(handler: OpenAICompletionsHandler, entry: dict) -> dict:
    """The body BFCL's handler sends first for a multi_turn row, without ``model``."""

    def stop(**kwargs):
        raise FirstRequest(copy.deepcopy(kwargs))

    handler.generate_with_backoff = stop
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # the handler prints a line per step
            handler.inference(copy.deepcopy(entry), include_input_log=False, exclude_state_log=True)
    except FirstRequest as sent:
        body = sent.args[0]
    else:
        raise AssertionError(f"{entry['id']}: the handler finished without sending a request")
    del body["model"]
    return body


def check_multi_turn(wheel: zipfile.ZipFile, categories: tuple[str, ...]) -> list[str]:
    assert all(bfcl.multi_turn(category) for category in categories), categories
    # The handler builds an OpenAI client, which wants a key; the stub stops it before any request.
    os.environ.setdefault("OPENAI_API_KEY", "unused")
    handler = OpenAICompletionsHandler(
        model_name="bellwether-equivalence",
        temperature=bfcl.TEMPERATURE,
        registry_name="bellwether-equivalence-FC",
        is_fc_model=True,
    )
    skipped: list[tuple[str, str]] = []
    sets = bfcl.build_sets(wheel, categories=categories, skipped=skipped)
    requests_equal = tools_equal = total = calls_valid = calls_total = 0
    failures: list[str] = []
    for category in categories:
        entries = {entry["id"]: entry for entry in load_dataset_entry(category)}
        answers = bfcl.read_answers(wheel, category)
        for line in sets[("render", bfcl.set_name(category))]:
            entry = entries[line["origin"]["row"]]
            total += 1
            if json.dumps(line["request"]) == json.dumps(first_request(handler, entry)):
                requests_equal += 1
            else:
                failures.append(f"first request differs: {entry['id']}")
            theirs = convert_to_tool(entry["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
            if json.dumps(line["request"].get("tools", [])) == json.dumps(theirs):
                tools_equal += 1
            else:
                failures.append(f"tools differ: {entry['id']}")
        for line in sets.get(("parse", bfcl.set_name(category)), []):
            row_id = line["origin"]["row"]
            # The handler's model_responses for tool calls: one {name: arguments} per call, decoded to call strings.
            responses = [{c["function"]["name"]: c["function"]["arguments"]} for c in line["message"]["tool_calls"]]
            decoded = handler.decode_execute(responses, has_tool_call_tag=False)
            first_turn = [answers[row_id]["ground_truth"][0]]
            result = multi_turn_checker([[decoded]], first_turn, entries[row_id], category, CHECKER_MODEL)
            calls_total += 1
            if result["valid"]:
                calls_valid += 1
            else:
                failures.append(f"first turn rejected: {row_id}: {result.get('error_message')}")
    print(f"multi_turn first requests: {requests_equal} of {total} equal to what OpenAICompletionsHandler sends")
    print(f"multi_turn tools: {tools_equal} of {total} equal to BFCL's convert_to_tool")
    print(f"multi_turn first turns: {calls_valid} of {calls_total} accepted by BFCL's multi_turn_checker")
    for why, count in Counter(why for _, why in skipped).items():
        print(f"multi_turn rows without a parse case: {count}: {why}")
    return failures


def main(wheel_path: str, *multi_turn_categories: str) -> int:
    imported = tuple(category for category in bfcl.CATEGORIES if bfcl.multi_turn(category))
    with zipfile.ZipFile(wheel_path) as wheel:
        failures = check_single_turn(wheel) + check_multi_turn(wheel, multi_turn_categories or imported)
    for failure in failures:
        print(failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
