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
        ["verify", "--smg", "http://127.0.0.1:30000"],
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
        "chunk_plans": {"whole": None, "per_token": None, "random-42": [2, 1]},
        "reference": {"source": "roundtrip", "message": {"role": "assistant"}, "finish_reason": "tool_calls"},
    }
    validator.validate(render)
    validator.validate(parse)
    bad = dict(render, id="Kimi-K3/render/x")
    assert list(validator.iter_errors(bad)), "ids must be lowercase slugs"
