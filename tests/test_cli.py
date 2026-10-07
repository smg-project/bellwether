import json

import pytest
from jsonschema import Draft202012Validator

from bellwether.cli import NOT_IMPLEMENTED, main
from bellwether.record.fixtures import schema_path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "gaps" in capsys.readouterr().out


def test_no_command_prints_help_and_fails(capsys):
    assert main([]) == NOT_IMPLEMENTED
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["record", "--model", "moonshotai/Kimi-K3", "--kind", "tokenize", "--oracle", "reference"],
        ["record", "--model", "moonshotai/Kimi-K3", "--kind", "render", "--oracle", "vllm"],
        ["verify", "--smg", "http://127.0.0.1:30000", "--capture", "capture.jsonl", "--kind", "parse"],
        ["verify", "--smg", "http://127.0.0.1:30000", "--capture", "capture.jsonl", "--kind", "detokenize"],
        ["report", "runs/a.json"],
    ],
)
def test_stubs_report_not_implemented(argv, capsys):
    assert main(argv) == NOT_IMPLEMENTED
    assert "not implemented" in capsys.readouterr().err


def test_case_schema_is_valid_and_accepts_examples():
    schema = json.loads(schema_path().read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    render = {
        "id": "kimi-k3/render/tools-parallel-001",
        "kind": "render",
        "model": "moonshotai/Kimi-K3",
        "request": {"messages": [{"role": "user", "content": "hi"}]},
        "reference": {"source": "vendor-code:encoding_k3.py", "input_ids": [1, 2, 3]},
        "witnesses": {"vllm": {"version": "0.29.1", "input_ids": [1, 2, 3]}},
    }
    parse = {
        "id": "kimi-k3/parse/tool-call-split-marker-003",
        "kind": "parse",
        "model": "moonshotai/Kimi-K3",
        "tools": [],
        "output_ids": [5, 6, 7],
        "output_pieces": ["<", "tool", ">"],
        "chunk_plans": {"whole": None, "per_token": None, "random-42": [2, 1]},
        "reference": {
            "source": "roundtrip",
            "message": {"role": "assistant"},
            "finish_reason": "tool_calls",
            "end_of_turn": {"stop_id": 8, "found_by": "turn"},
        },
    }
    validator.validate(render)
    validator.validate(parse)
    bad = dict(render, id="Kimi-K3/render/x")
    assert list(validator.iter_errors(bad)), "ids must be lowercase slugs"


VLLM_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"


def second_reference_lines() -> tuple[dict, dict]:
    """A render line with a second reference beside its reference, and a parse line kept for its second references."""
    render = {
        "id": "apertus-8b-instruct-2509/render/bfcl-simple-python-0",
        "kind": "render",
        "model": "swiss-ai/Apertus-8B-Instruct-2509",
        "request": {"messages": [{"role": "user", "content": "hi"}], "tools": []},
        "reference": {"source": "hf-template", "input_ids": [1, 2], "text": "hi"},
        "second_references": {
            "tool_chat_template_apertus.jinja": {
                "source": "hf-template",
                "input_ids": [1, 3, 2],
                "text": "tools hi",
                "provenance": {"vllm_commit": VLLM_COMMIT, "chat_template_sha256": "a" * 64},
            }
        },
    }
    parse = {
        "id": "mistral-7b-instruct-v0.3/parse/bfcl-simple-python-0",
        "kind": "parse",
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "request": {"messages": [{"role": "user", "content": "hi"}], "tools": []},
        "tools": [],
        "malformed": False,
        "reference": {"source": "roundtrip", "rejected": "the template wants call ids of nine letters or digits"},
        "second_references": {
            "tool_chat_template_mistral.jinja": {
                "source": "roundtrip",
                "ids": [5, 6],
                "pieces": ["[TOOL_CALLS]", "[]"],
                "text": "[TOOL_CALLS][]",
                "chunk_plans": {"whole": None, "per_token": None},
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "tool_calls",
                "end_of_turn": {"stop_id": 2, "found_by": "turn"},
                "provenance": {"vllm_commit": VLLM_COMMIT, "chat_template_sha256": "b" * 64},
            },
            "tool_chat_template_mistral_parallel.jinja": {
                "source": "roundtrip",
                "rejected": "a second reference refuses a case as the first one does",
                "provenance": {"vllm_commit": VLLM_COMMIT, "chat_template_sha256": "c" * 64},
            },
        },
    }
    return render, parse


def test_case_schema_accepts_second_references():
    validator = Draft202012Validator(json.loads(schema_path().read_text()))
    render, parse = second_reference_lines()
    validator.validate(render)
    validator.validate(parse)
    recorded_first = {
        **{key: value for key, value in parse.items() if key != "reference"},
        "reference": {
            "source": "roundtrip",
            "text": "x",
            "message": {"role": "assistant"},
            "end_of_turn": {"stop_id": 2, "found_by": "turn"},
        },
        "output_ids": [7],
        "output_pieces": ["x"],
    }
    validator.validate(recorded_first)


def lines_off_the_proposal() -> list:
    render, parse = second_reference_lines()
    entry = render["second_references"]["tool_chat_template_apertus.jinja"]
    recorded = parse["second_references"]["tool_chat_template_mistral.jinja"]
    without_second = {key: value for key, value in parse.items() if key != "second_references"}
    both_reasons = {"source": "roundtrip", "rejected": "a", "not_applicable": "b"}
    no_ids = {key: value for key, value in recorded.items() if key != "ids"}
    no_end_of_turn = {key: value for key, value in recorded.items() if key != "end_of_turn"}
    commitless = {**entry, "provenance": {"chat_template_sha256": "a" * 64}}
    return [
        pytest.param({**render, "second_references": {"apertus": entry}}, id="a key that is not a template file"),
        pytest.param({**render, "second_references": {}}, id="no second reference under the key"),
        pytest.param(
            {**render, "second_references": {"tool_chat_template_apertus.jinja": commitless}},
            id="an entry without the template's vLLM commit",
        ),
        pytest.param(
            {**render, "second_references": {"tool_chat_template_apertus.jinja": {**entry, "source": "vendor-code"}}},
            id="an entry from another oracle",
        ),
        pytest.param({**parse, "reference": both_reasons}, id="a reference with both reasons"),
        pytest.param(without_second, id="a reference without a result and no second reference"),
        pytest.param({**parse, "output_ids": [5, 6]}, id="the reference's output beside a reference without one"),
        pytest.param(
            {**parse, "second_references": {"tool_chat_template_mistral.jinja": no_ids}},
            id="a parse entry with a result but no output ids",
        ),
        pytest.param(
            {
                **parse,
                "reference": {"source": "roundtrip", "text": "x", "end_of_turn": {"stop_id": 2, "found_by": "turn"}},
            },
            id="a parse reference with a result but no output ids",
        ),
        pytest.param(
            {**parse, "second_references": {"tool_chat_template_mistral.jinja": no_end_of_turn}},
            id="a round-trip entry with a result but no end of turn",
        ),
    ]


@pytest.mark.parametrize("line", lines_off_the_proposal())
def test_case_schema_refuses_second_references_off_the_proposal(line):
    validator = Draft202012Validator(json.loads(schema_path().read_text()))
    assert list(validator.iter_errors(line))


def test_case_schema_accepts_a_render_line_kept_for_its_second_reference():
    render, _ = second_reference_lines()
    line = {
        **render,
        "reference": {"source": "hf-template", "rejected": "the template wants a description on every tool"},
    }
    Draft202012Validator(json.loads(schema_path().read_text())).validate(line)


def lines_with_one_fault() -> list:
    """Lines the schema refuses for one reason each, with where it reports it and the keyword that refuses it."""
    render, parse = second_reference_lines()
    apertus = "second_references/tool_chat_template_apertus.jinja"
    mistral = "second_references/tool_chat_template_mistral.jinja"
    entry = render["second_references"]["tool_chat_template_apertus.jinja"]
    recorded = parse["second_references"]["tool_chat_template_mistral.jinja"]
    no_result = {key: value for key, value in entry.items() if key not in ("input_ids", "text")}
    reason = "the template refuses the case"

    def render_entry(value: dict) -> dict:
        return {**render, "second_references": {"tool_chat_template_apertus.jinja": value}}

    def parse_entry(value: dict) -> dict:
        return {**parse, "second_references": {**parse["second_references"], "tool_chat_template_mistral.jinja": value}}

    detokenize = {
        "id": "apertus-8b-instruct-2509/detokenize/a",
        "kind": "detokenize",
        "model": "swiss-ai/Apertus-8B-Instruct-2509",
        "reference": {"source": "engine:vllm", "not_applicable": reason},
        "second_references": {"tool_chat_template_apertus.jinja": {**no_result, "rejected": reason}},
    }
    return [
        # A reference gives its result or a reason for recording none, never both and never neither.
        pytest.param(render_entry(no_result), apertus, "required", id="a render entry with no result and no reason"),
        pytest.param(
            {**render, "reference": {"source": "hf-template"}},
            "reference",
            "required",
            id="a render reference with no result and no reason",
        ),
        pytest.param(
            render_entry({**entry, "rejected": reason}), apertus, "not", id="a render entry with a result and a reason"
        ),
        pytest.param(
            parse_entry({**no_result, "source": "roundtrip", "not_applicable": reason, "ids": [5]}),
            mistral,
            "not",
            id="a parse entry not applicable with output ids",
        ),
        pytest.param(
            {**parse, "reference": {**parse["reference"], "message": {"role": "assistant"}}},
            "reference",
            "not",
            id="a reference with a reason and a result",
        ),
        # An entry comes from the reference's oracle for the case's kind.
        pytest.param(
            render_entry({**entry, "source": "roundtrip"}),
            f"{apertus}/source",
            "const",
            id="a render entry from the round trip",
        ),
        pytest.param(
            parse_entry({**recorded, "source": "hf-template"}),
            f"{mistral}/source",
            "const",
            id="a parse entry from hf-template",
        ),
        # Not applicable is a parse outcome.
        pytest.param(
            {**render, "reference": {"source": "hf-template", "not_applicable": reason}},
            "reference",
            "not",
            id="a render reference not applicable",
        ),
        pytest.param(
            render_entry({**no_result, "not_applicable": reason}), apertus, "not", id="a render entry not applicable"
        ),
        pytest.param(detokenize, "reference", "not", id="a detokenize reference not applicable"),
    ]


@pytest.mark.parametrize(("line", "where", "keyword"), lines_with_one_fault())
def test_case_schema_refuses_a_line_for_its_one_fault(line, where, keyword):
    validator = Draft202012Validator(json.loads(schema_path().read_text()))
    errors = [("/".join(map(str, error.absolute_path)), error.validator) for error in validator.iter_errors(line)]
    assert errors == [(where, keyword)]
