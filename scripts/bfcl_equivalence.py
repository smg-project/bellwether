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
"""

import copy
import json
import sys
import zipfile
from pathlib import Path

from bfcl_eval.constants.enums import Language, ModelStyle
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.utils import _func_doc_language_specific_pre_processing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bellwether.importers import bfcl  # noqa: E402

LANGUAGES = {"python": Language.PYTHON, "java": Language.JAVA, "javascript": Language.JAVASCRIPT}


def main(wheel_path: str) -> int:
    # convert_func_name looks a model up by turning "_" back into "/", so the name must hold no "_".
    model = next(
        name
        for name, config in MODEL_CONFIG_MAPPING.items()
        if config.is_fc_model and config.underscore_to_dot and "_" not in name
    )
    tools_equal = tools_total = calls_valid = calls_total = unanswerable = 0
    failures: list[str] = []
    with zipfile.ZipFile(wheel_path) as wheel:
        for category in bfcl.CATEGORIES:
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
                output = [
                    {c["function"]["name"]: json.loads(c["function"]["arguments"])} for c in message["tool_calls"]
                ]
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
    for failure in failures:
        print(failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
