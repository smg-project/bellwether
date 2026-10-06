import hashlib
import json
import re

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bellwether.cli import main
from bellwether.importers import hf, swehero
from bellwether.record.corpus import read_cases

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Run a command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Done.",
            "parameters": {"type": "object", "required": ["message"], "properties": {"message": {"type": "string"}}},
        },
    },
]


def test_tools_in_openai_shape_pass_unchanged():
    assert swehero.check_tools(TOOLS) is TOOLS


@pytest.mark.parametrize(
    "tool, problem",
    [
        ({"name": "f", "parameters": {"type": "object"}}, 'tool 1 is not {"type": "function", "function": {...}}'),
        (
            {"type": "function", "function": {"name": "f", "parameters": {"type": "dict"}}},
            "tool 1 has parameters that are not a JSON Schema object",
        ),
        (
            {"type": "function", "function": {"name": "f", "response": {}}},
            "tool 1 has keys OpenAI's function does not: response",
        ),
        ({"type": "function", "function": {"name": "two words"}}, "tool 1 has no name of letters, digits, _ and -"),
        ({"type": "function", "function": {"name": "f\n"}}, "tool 1 has no name of letters, digits, _ and -"),
        ({"type": "function", "function": {"description": "D."}}, "tool 1 has no name of letters, digits, _ and -"),
        ({"type": "function", "function": {"name": "finish"}}, "tool 1 repeats the name finish"),
    ],
)
def test_tools_not_in_openai_shape_are_refused(tool, problem):
    with pytest.raises(ValueError, match=re.escape(f"tools.json: {problem}")):
        swehero.check_tools([TOOLS[1], tool])


def test_a_tools_file_that_is_not_a_list_of_tools_is_refused():
    for tools in ({"tools": TOOLS}, []):
        with pytest.raises(ValueError, match="tools.json: not a list of tools"):
            swehero.check_tools(tools)


def call(call_id: str, name: str, arguments) -> dict:
    """A call as the shard holds it: parquet gives its struct fields in alphabetical order."""
    return {"function": {"arguments": arguments, "name": name}, "id": call_id, "type": "function"}


def item(role: str, content: str, *calls: dict) -> dict:
    """A message as the shard holds it: every message has ``tool_calls``, null except on assistant turns."""
    return {"content": content, "role": role, "tool_calls": list(calls) if role == "assistant" else None}


TRAJECTORY = [
    item("system", "You are an agent."),
    item("user", "Fix the bug in café.py"),
    item("assistant", "Let me look.", call("call-a", "execute_bash", '{"command": "cat café.py"}')),
    item("tool", "print('hi')"),
    item(
        "assistant",
        "",
        call("call-b", "execute_bash", '{"command":"ls"}'),
        call("call-c", "execute_bash", '{"command": "pwd"}'),
    ),
    item("tool", "café.py"),
    item("tool", "/workspace"),
    item("assistant", "Done.", call("call-d", "finish", '{"message": "Fixed \\u2615."}')),
]


def openai_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_tool_messages_take_the_ids_of_the_calls_before_them_in_order():
    assert swehero.messages_for(TRAJECTORY) == [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the bug in café.py"},
        {
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [openai_call("call-a", "execute_bash", '{"command": "cat café.py"}')],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "print('hi')"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                openai_call("call-b", "execute_bash", '{"command":"ls"}'),
                openai_call("call-c", "execute_bash", '{"command": "pwd"}'),
            ],
        },
        {"role": "tool", "tool_call_id": "call-b", "content": "café.py"},
        {"role": "tool", "tool_call_id": "call-c", "content": "/workspace"},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [openai_call("call-d", "finish", '{"message": "Fixed \\u2615."}')],
        },
    ]


def test_messages_take_openai_key_order_not_the_shards():
    messages = swehero.messages_for(TRAJECTORY)
    assert [list(message) for message in messages[:4]] == [
        ["role", "content"],
        ["role", "content"],
        ["role", "content", "tool_calls"],
        ["role", "tool_call_id", "content"],
    ]
    assert list(messages[2]["tool_calls"][0]) == ["id", "type", "function"]
    assert list(messages[2]["tool_calls"][0]["function"]) == ["name", "arguments"]


def test_an_assistant_turn_without_calls_has_no_tool_calls_key():
    trajectory = [*TRAJECTORY[:2], item("assistant", "Which file?"), item("user", "café.py")]
    assert swehero.messages_for(trajectory)[2:] == [
        {"role": "assistant", "content": "Which file?"},
        {"role": "user", "content": "café.py"},
    ]


@pytest.mark.parametrize(
    "trajectory, detail",
    [
        ([*TRAJECTORY[:6], *TRAJECTORY[7:]], "message 4 makes 2 call(s) and 1 result(s) follow"),
        ([*TRAJECTORY[:3], item("user", "Go on.")], "message 2 makes 1 call(s) and 0 result(s) follow"),
        (
            [*TRAJECTORY[:4], item("tool", "again"), *TRAJECTORY[4:]],
            "message 4 is a tool result with no call to answer",
        ),
        (
            [*TRAJECTORY[:2], item("tool", "orphan"), *TRAJECTORY[2:]],
            "message 2 is a tool result with no call to answer",
        ),
        (TRAJECTORY[:6], "message 4 makes 2 call(s) and 1 result(s) follow"),
    ],
)
def test_a_row_whose_results_do_not_pair_with_its_calls_is_refused(trajectory, detail):
    # TRAJECTORY itself ends with a call no result answers, as every trajectory in the shard ends with `finish`:
    # the last turn may end the conversation, but a turn the conversation goes on past needs one result per call.
    with pytest.raises(swehero.Refused) as refused:
        swehero.messages_for(trajectory)
    assert (refused.value.reason, refused.value.detail) == (swehero.UNPAIRED, detail)


@pytest.mark.parametrize("arguments", ['["ls"]', '"ls"', "ls -la", '{"command": "ls"', None])
def test_a_call_whose_arguments_are_not_a_json_object_string_refuses_the_row(arguments):
    bad = item("assistant", "", call("call-x", "execute_bash", arguments))
    with pytest.raises(swehero.Refused) as refused:
        swehero.messages_for([*TRAJECTORY[:2], bad, item("tool", "x"), *TRAJECTORY[4:]])
    assert (refused.value.reason, refused.value.detail) == (swehero.NOT_AN_OBJECT, "message 2 calls execute_bash")


LONG = [
    item("system", "You are an agent."),
    item("user", "Fix it."),
    *[
        message
        for step in range(7)
        for message in (
            item("assistant", f"Step {step}.", call(f"call-{step}", "execute_bash", '{"command": "ls"}')),
            item("tool", "x" * 100),
        )
    ],
]  # assistant turns at messages 2, 4, ..., 14


def size(messages: list[dict], turn: int) -> int:
    return swehero.request_bytes(swehero.request_for(messages, turn, TOOLS))


def test_the_request_for_a_turn_is_every_message_before_it_and_the_tools():
    messages = swehero.messages_for(TRAJECTORY)
    request = swehero.request_for(messages, 4, TOOLS)
    assert request == {"messages": messages[:4], "tools": TOOLS}
    assert list(request) == ["messages", "tools"]


def test_a_request_is_measured_in_the_bytes_its_corpus_line_writes():
    request = {"messages": [{"role": "user", "content": "café ☕"}]}
    assert swehero.request_bytes(request) == len('{"messages": [{"role": "user", "content": "café ☕"}]}'.encode())


def test_the_turns_are_the_first_middle_and_last_whose_request_fits_under_the_cap():
    messages = swehero.messages_for(LONG)
    assert swehero.turns(messages, TOOLS, size(messages, 14)) == [2, 8, 14]
    assert swehero.turns(messages, TOOLS, size(messages, 10)) == [2, 6, 10]
    assert swehero.turns(messages, TOOLS, size(messages, 10) - 1) == [2, 4, 8]
    assert swehero.turns(messages, TOOLS, size(messages, 4)) == [2, 4]
    assert swehero.turns(messages, TOOLS, size(messages, 2)) == [2]
    assert swehero.turns(messages, TOOLS, size(messages, 2) - 1) == []


def row(number: int = 0, license: str | None = "MIT", trajectory: list[dict] = TRAJECTORY) -> dict:
    return {
        "instance_id": f"owner__repo-{number}",
        "repo": "owner/repo",
        "license": license,
        "trajectory_id": f"trajectory-{number}",
        "trajectory": trajectory,
    }


RENDER, PARSE = ("render", "swehero-13"), ("parse", "swehero-13")


def test_a_row_gives_a_render_and_a_parse_case_for_each_chosen_turn():
    sets = swehero.build_sets([row()], TOOLS)
    assert sorted(sets) == [PARSE, RENDER]
    render, parse = sets[RENDER], sets[PARSE]
    assert [line["name"] for line in render] == ["swehero-13-0-2", "swehero-13-0-4", "swehero-13-0-7"]
    assert [line["name"] for line in parse] == ["swehero-13-0-2", "swehero-13-0-4", "swehero-13-0-7"]
    origin = {
        "dataset": "swehero",
        "source": "hf:datasets/nvidia/SWE-Hero-openhands-trajectories@150bc119e52c647216fce285fd801f16b6fd745b",
        "sha256": "936b195ed11b2b9be2e60cf9cb274dfc2f31d07745cbd6144658da988ce295a9",
        "file": "data/train-00013-of-00014.parquet",
        "row": 0,
        "turn": 4,
        "instance_id": "owner__repo-0",
        "trajectory_id": "trajectory-0",
        "repository": "owner/repo",
        "repository_license": "MIT",
        "license": "CC-BY-4.0",
    }
    request = {"messages": swehero.messages_for(TRAJECTORY)[:4], "tools": TOOLS}
    notes = "SWE-Hero owner__repo-0: the assistant turn at message 4 of 8 (execute_bash, execute_bash)"
    assert render[1] == {"name": "swehero-13-0-4", "request": request, "notes": notes, "origin": origin}
    assert parse[1] == {
        "name": "swehero-13-0-4",
        "request": request,
        "message": {
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "execute_bash", "arguments": '{"command":"ls"}'}},
                {"type": "function", "function": {"name": "execute_bash", "arguments": '{"command": "pwd"}'}},
            ],
        },
        "notes": notes,
        "origin": origin,
    }
    assert list(render[1]) == ["name", "request", "notes", "origin"]
    assert list(parse[1]) == ["name", "request", "message", "notes", "origin"]
    assert list(parse[1]["origin"]) == list(origin)
    assert parse[2]["message"] == {
        "content": "Done.",
        "tool_calls": [
            {"type": "function", "function": {"name": "finish", "arguments": '{"message": "Fixed \\u2615."}'}}
        ],
    }


def test_a_turn_without_calls_is_a_parse_case_with_content_only():
    trajectory = [*TRAJECTORY[:7], item("assistant", "All done.")]
    [*_, last] = swehero.build_sets([row(trajectory=trajectory)], TOOLS)[PARSE]
    assert last["message"] == {"content": "All done."}
    assert last["notes"] == "SWE-Hero owner__repo-0: the assistant turn at message 7 of 8 (no call)"


def test_one_row_in_stride_is_sampled_and_every_refused_row_is_named_with_its_reason(monkeypatch):
    monkeypatch.setattr(swehero, "STRIDE", 2)
    rows = [row(number) for number in range(5)]
    rows[1]["trajectory"] = TRAJECTORY[:6]
    rows[2]["license"] = "GPL-3.0"
    rows[3]["license"] = None
    refused: list = []
    sets = swehero.build_sets(rows, TOOLS, refused)
    assert sorted({line["origin"]["row"] for line in sets[RENDER]}) == [0, 4]
    assert refused == [
        (1, swehero.UNPAIRED, "message 4 makes 2 call(s) and 1 result(s) follow"),
        (2, swehero.LICENSE_NOT_ALLOWED, "GPL-3.0"),
        (3, swehero.LICENSE_NOT_ALLOWED, "None"),
    ]


def test_rows_under_each_reviewed_repository_license_are_kept(monkeypatch):
    monkeypatch.setattr(swehero, "STRIDE", 1)
    licenses = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"]
    refused: list = []
    sets = swehero.build_sets([row(number, license) for number, license in enumerate(licenses)], TOOLS, refused)
    assert refused == []
    assert sorted({(line["origin"]["row"], line["origin"]["repository_license"]) for line in sets[PARSE]}) == list(
        enumerate(licenses)
    )


def test_a_sampled_row_that_gives_no_turn_is_named(monkeypatch):
    monkeypatch.setattr(swehero, "STRIDE", 2)
    monkeypatch.setattr(swehero, "MAX_REQUEST_BYTES", 100)
    refused: list = []
    assert swehero.build_sets([row(), row(1), row(2, trajectory=TRAJECTORY[:2])], TOOLS, refused) == {}
    first = size(swehero.messages_for(TRAJECTORY), 2)
    assert refused == [
        (0, swehero.NO_TURN, f"the first assistant turn's request has {first} bytes"),
        (2, swehero.NO_TURN, "the trajectory has no assistant turn"),
    ]


FUNCTION = pa.struct([("arguments", pa.string()), ("name", pa.string())])
CALL = pa.struct([("function", FUNCTION), ("id", pa.string()), ("type", pa.string())])
MESSAGE = pa.struct([("content", pa.string()), ("role", pa.string()), ("tool_calls", pa.list_(CALL))])
SHARD_SCHEMA = pa.schema(
    [
        ("instance_id", pa.string()),
        ("repo", pa.string()),
        ("license", pa.string()),
        ("trajectory_id", pa.string()),
        ("trajectory", pa.list_(MESSAGE)),
        ("model_patch", pa.string()),
        ("dataset", pa.string()),
    ]
)  # the schema of the pinned shard


def write_shard(path, rows: list[dict]):
    full = [{**r, "model_patch": "diff --git a/x b/x", "dataset": "nebius/SWE-rebench"} for r in rows]
    pq.write_table(pa.Table.from_pylist(full, schema=SHARD_SCHEMA), path)
    return path


def test_rows_are_read_from_the_shard_in_order_with_the_fields_the_import_uses(tmp_path):
    rows = [row(number) for number in range(3)]
    assert list(swehero.read_rows(write_shard(tmp_path / "shard.parquet", rows))) == rows


def test_written_sets_read_back_and_check_clean_and_a_changed_missing_or_stale_file_is_reported(tmp_path):
    sets, corpus = swehero.build_sets([row()], TOOLS), tmp_path / "corpus"
    (corpus / "render").mkdir(parents=True)
    for name in ("common", "bfcl-simple-python", "swehero-12"):
        (corpus / "render" / f"{name}.jsonl").write_text("{}\n")
    swehero.write_sets(sets, corpus)
    assert sorted(path.name for path in (corpus / "render").iterdir()) == [
        "bfcl-simple-python.jsonl",
        "common.jsonl",
        "swehero-13.jsonl",
    ]
    assert (corpus / "render" / "common.jsonl").read_text() == "{}\n"
    text = (corpus / "parse" / "swehero-13.jsonl").read_bytes().decode("utf-8")
    assert "café" in text and text.endswith("\n")
    cases = read_cases(corpus / "parse" / "swehero-13.jsonl")
    assert [(case.name, case.message, case.origin) for case in cases] == [
        (line["name"], line["message"], line["origin"]) for line in sets[PARSE]
    ]
    assert swehero.check_sets(sets, corpus) == []
    (corpus / "parse" / "swehero-13.jsonl").write_text("{}\n")
    (corpus / "render" / "swehero-13.jsonl").unlink()
    (corpus / "render" / "swehero-stale.jsonl").write_text("{}\n")
    assert swehero.check_sets(sets, corpus) == [
        f"{corpus / 'parse' / 'swehero-13.jsonl'}: differs from a fresh import",
        f"{corpus / 'render' / 'swehero-13.jsonl'}: missing",
        f"{corpus / 'render' / 'swehero-stale.jsonl'}: no SWE-Hero shard writes it",
    ]


# The front matter of the pinned card: a `license` feature column under dataset_info, then the dataset's license.
CARD = (
    "---\ndataset_info:\n  features:\n    - name: license\n      dtype: string\n"
    "license: cc-by-4.0\ntags:\n  - code\n---\n\n# SWE-Hero\n"
)


def serve(tmp_path, monkeypatch, rows: list[dict], card: str = CARD) -> list[tuple]:
    """Stand in for the Hub: the three files, written to ``tmp_path`` and pinned by their sha256 in place of the real
    pins; returns the downloads asked for."""
    files = {
        hf.CARD: tmp_path / "README.md",
        swehero.TOOLS_FILE: tmp_path / "tools.json",
        swehero.SHARD_FILE: write_shard(tmp_path / "shard.parquet", rows),
    }
    files[hf.CARD].write_text(card)
    files[swehero.TOOLS_FILE].write_text(json.dumps(TOOLS))
    pins = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    monkeypatch.setattr(swehero, "FILES", pins)
    downloads: list[tuple] = []

    def download(repo_id, filename, **kwargs):
        downloads.append((repo_id, filename, kwargs))
        return str(files[filename])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return downloads


def test_the_command_reads_the_pinned_files_under_its_cache_then_writes_and_checks(tmp_path, monkeypatch, capsys):
    downloads = serve(tmp_path, monkeypatch, [row()])
    corpus = tmp_path / "corpus"
    argv = ["import", "swehero", "--corpus", str(corpus), "--cache", str(tmp_path / "cache")]
    assert main([*argv, "--check"]) == 1
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    out = capsys.readouterr().out
    assert f"{corpus / 'render' / 'swehero-13.jsonl'}: 3 cases" in out
    assert f"{corpus / 'parse' / 'swehero-13.jsonl'}: 3 cases" in out
    assert f"{corpus}: the SWE-Hero sets equal a fresh import" in out
    at_the_pin = {
        "repo_type": "dataset",
        "revision": swehero.REVISION,
        "cache_dir": tmp_path / "cache" / "huggingface",
        "force_download": False,
        "token": False,
    }
    pinned = [(swehero.REPO, name, at_the_pin) for name in (hf.CARD, swehero.TOOLS_FILE, swehero.SHARD_FILE)]
    assert downloads == pinned * 3


def test_a_card_under_another_license_stops_the_import_before_the_shard_is_fetched(tmp_path, monkeypatch):
    downloads = serve(tmp_path, monkeypatch, [row()], card=CARD.replace("cc-by-4.0", "cc-by-nc-4.0"))
    with pytest.raises(ValueError, match="the card's license is 'cc-by-nc-4.0', not the reviewed 'cc-by-4.0'"):
        main(["import", "swehero", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
    assert [filename for _, filename, _ in downloads] == [hf.CARD]


def test_the_command_names_every_refused_row(tmp_path, monkeypatch, capsys):
    rows = [row(number, "GPL-3.0") for number in range(51)] + [row(51, trajectory=TRAJECTORY[:6]), row(52)]
    serve(tmp_path, monkeypatch, rows)
    assert main(["import", "swehero", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")]) == 0
    out = capsys.readouterr().out
    every = ", ".join(f"{number}: GPL-3.0" for number in range(51))
    assert f"refused 51 row(s) ({every}): {swehero.LICENSE_NOT_ALLOWED}" in out
    assert f"refused 1 row(s) (51: message 4 makes 2 call(s) and 1 result(s) follow): {swehero.UNPAIRED}" in out
