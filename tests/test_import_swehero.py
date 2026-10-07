import hashlib
import json
import re

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bellwether.cli import main
from bellwether.importers import corpus_sets, hf, swehero
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


# The license table the tests' rows read: owner/repo's code goes with one copied license file, and gone/repo has none.
# The committed table is the real one's (test_the_committed_license_table_pins_every_copy_it_names).
NOTICE = "licenses/swehero-owner-repo-0123456789ab-LICENSE"
LICENSES = swehero.Licenses(
    notices={"owner/repo": (NOTICE,)}, left_out={"gone/repo": "pull request 7 is not on GitHub"}
)
LICENSE_TEXTS = {NOTICE: b"MIT License\n"}
COMMITTED_LICENSES = swehero.committed_licenses


@pytest.fixture(autouse=True)
def license_table(monkeypatch):
    monkeypatch.setattr(swehero, "committed_licenses", lambda: LICENSES)
    monkeypatch.setattr(swehero, "license_files", lambda cache: LICENSE_TEXTS)


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
def test_a_row_whose_results_do_not_pair_with_its_calls_is_unusable(trajectory, detail):
    # TRAJECTORY itself ends with a call no result answers, as every trajectory in the shard ends with `finish`:
    # the last turn may end the conversation, but a turn the conversation goes on past needs one result per call.
    with pytest.raises(swehero.Unusable) as unusable:
        swehero.messages_for(trajectory)
    assert (unusable.value.reason, unusable.value.detail) == (swehero.UNPAIRED, detail)


@pytest.mark.parametrize(
    "arguments",
    [
        '["ls"]',
        '"ls"',
        "ls -la",
        '{"command": "ls"',
        None,
        # Python's json reads these, and JSON has no such values.
        '{"timeout": NaN}',
        '{"timeout": Infinity}',
        '{"timeout": -Infinity}',
    ],
)
def test_a_call_whose_arguments_are_not_a_json_object_string_makes_the_row_unusable(arguments):
    bad = item("assistant", "", call("call-x", "execute_bash", arguments))
    with pytest.raises(swehero.Unusable) as unusable:
        swehero.messages_for([*TRAJECTORY[:2], bad, item("tool", "x"), *TRAJECTORY[4:]])
    assert (unusable.value.reason, unusable.value.detail) == (swehero.NOT_AN_OBJECT, "message 2 calls execute_bash")


@pytest.mark.parametrize("index, role", [(0, "system"), (1, "user"), (3, "tool")])
def test_a_row_with_calls_on_a_message_that_is_not_an_assistant_turn_is_unusable(index, role):
    # OpenAI's request has no place for them, and dropping them would change the conversation.
    trajectory = list(TRAJECTORY)
    trajectory[index] = {**TRAJECTORY[index], "tool_calls": [call("call-x", "execute_bash", '{"command": "ls"}')]}
    with pytest.raises(swehero.Unusable) as unusable:
        swehero.messages_for(trajectory)
    detail = f"message {index} is a {role} message with calls"
    assert (unusable.value.reason, unusable.value.detail) == (swehero.CALLS_OUTSIDE_A_TURN, detail)


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
    sets = swehero.build_sets(13, [row()], TOOLS)
    assert sorted(sets) == [PARSE, RENDER]
    render, parse = sets[RENDER], sets[PARSE]
    assert [line["name"] for line in render] == ["swehero-13-0-2", "swehero-13-0-4", "swehero-13-0-7"]
    assert [line["name"] for line in parse] == ["swehero-13-0-2", "swehero-13-0-4", "swehero-13-0-7"]
    origin = {
        "dataset": "swehero",
        "source": "hf:datasets/nvidia/SWE-Hero-openhands-trajectories@150bc119e52c647216fce285fd801f16b6fd745b",
        "sha256": "936b195ed11b2b9be2e60cf9cb274dfc2f31d07745cbd6144658da988ce295a9",
        "file": "data/train-00013-of-00014.parquet",
        "tools_sha256": "d0f46e87e8d6b6d4eef8c01d6add2674ecaaff1260e08df5a4c2b162824c593c",
        "tools_file": "tools.json",
        "row": 0,
        "turn": 4,
        "instance_id": "owner__repo-0",
        "trajectory_id": "trajectory-0",
        "repository": "owner/repo",
        "repository_license": "MIT",
        "notices": [NOTICE],
        "license": "CC-BY-4.0",
        "written": ["tool result ids"],
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


def test_origin_marks_the_tool_result_ids_bellwether_writes_where_a_request_holds_a_tool_message():
    # The data's tool messages carry no id: each one's tool_call_id is bellwether's, the id of the call it answers by
    # position. The calls' own ids are the data's. A request with no tool message holds nothing bellwether wrote.
    sets = swehero.build_sets(13, [row()], TOOLS)
    for kind in (RENDER, PARSE):
        first, middle, last = sets[kind]
        assert [message["role"] for message in first["request"]["messages"]] == ["system", "user"]
        assert "written" not in first["origin"]
        assert middle["origin"]["written"] == last["origin"]["written"] == ["tool result ids"]
        assert list(middle["origin"])[-1] == "written"


def test_a_turn_without_calls_is_a_parse_case_with_content_only():
    trajectory = [*TRAJECTORY[:7], item("assistant", "All done.")]
    [*_, last] = swehero.build_sets(13, [row(trajectory=trajectory)], TOOLS)[PARSE]
    assert last["message"] == {"content": "All done."}
    assert last["notes"] == "SWE-Hero owner__repo-0: the assistant turn at message 7 of 8 (no call)"


# Every assistant content in the shard ends in whitespace, and tool results keep the output's own.
EDGES = [
    item("system", "You are an agent.\n"),
    item("user", "\n  Fix the bug.\n\n"),
    item("assistant", "Let me look.\n\n", call("call-a", "execute_bash", '{"command": "ls"}')),
    item("tool", "\n  café.py  \n\n"),
    item("assistant", "Done. \n\n", call("call-b", "finish", '{"message": "Fixed."}')),
]


def test_whitespace_at_the_edges_of_every_message_is_kept():
    sets = swehero.build_sets(13, [row(trajectory=EDGES)], TOOLS)
    [*_, last] = sets[RENDER]
    assert [message["content"] for message in last["request"]["messages"]] == [item["content"] for item in EDGES[:4]]
    assert [line["message"]["content"] for line in sets[PARSE]] == ["Let me look.\n\n", "Done. \n\n"]


def test_every_row_gives_cases_and_every_unusable_row_is_named_with_its_reason():
    rows = [row(number) for number in range(6)]
    rows[1]["trajectory"] = TRAJECTORY[:6]
    rows[2]["license"] = "GPL-3.0"
    rows[3]["license"] = None
    skipped: list = []
    sets = swehero.build_sets(13, rows, TOOLS, skipped)
    assert sorted({line["origin"]["row"] for line in sets[RENDER]}) == [0, 4, 5]
    names = [f"swehero-13-{number}-{turn}" for number in (0, 4, 5) for turn in (2, 4, 7)]
    assert [line["name"] for line in sets[RENDER]] == names
    assert [line["name"] for line in sets[PARSE]] == names
    assert skipped == [
        ("shard 13 row 1: message 4 makes 2 call(s) and 1 result(s) follow", swehero.UNPAIRED),
        ("shard 13 row 2: GPL-3.0", swehero.LICENSE_NOT_ALLOWED),
        ("shard 13 row 3: None", swehero.LICENSE_NOT_ALLOWED),
    ]


def test_rows_under_each_reviewed_repository_license_are_kept():
    licenses = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"]
    skipped: list = []
    sets = swehero.build_sets(13, [row(number, license) for number, license in enumerate(licenses)], TOOLS, skipped)
    assert skipped == []
    assert sorted({(line["origin"]["row"], line["origin"]["repository_license"]) for line in sets[PARSE]}) == list(
        enumerate(licenses)
    )


def test_a_row_whose_repository_has_no_pinned_license_file_is_named_and_left_out():
    skipped: list = []
    gone = {**row(), "repo": "gone/repo", "instance_id": "gone__repo-7"}
    sets = swehero.build_sets(13, [gone, row(1)], TOOLS, skipped)
    assert sorted({line["origin"]["row"] for line in sets[PARSE]}) == [1]
    assert skipped == [("shard 13 row 0: gone/repo: pull request 7 is not on GitHub", swehero.NO_LICENSE_FILE)]


def test_a_repository_the_license_table_does_not_name_stops_the_import():
    stranger = {**row(), "repo": "new/repo", "instance_id": "new__repo-1"}
    with pytest.raises(ValueError, match=re.escape("new/repo: swehero_licenses.json does not name it")):
        swehero.build_sets(13, [stranger], TOOLS)


def test_the_committed_license_table_pins_every_copy_it_names():
    table = swehero.load_license_table()
    files, repositories, left_out = table["files"], table["repositories"], table["left_out"]
    assert len(repositories) > 1000 and not set(repositories) & set(left_out)
    for repository, entry in repositories.items():
        assert set(entry["license"].split(" AND ")) <= set(swehero.REPOSITORY_LICENSES), repository
        own = [name for name in entry["files"] if files[name]["repository"] == repository]
        assert own and all(files[name]["commit"] == entry["commit"] for name in own), repository
    for name, pin in files.items():
        repository, commit, path = pin["repository"], pin["commit"], pin["path"]
        assert name == f"swehero-{repository.replace('/', '-')}-{commit[:12]}-{path.replace('/', '-')}"
        assert re.fullmatch(r"[0-9a-f]{40}", commit) and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]), name
    assert all(left_out.values())
    licenses = COMMITTED_LICENSES()
    assert set(licenses.notices) == set(repositories) and licenses.left_out == left_out
    some = next(iter(repositories))
    assert licenses.notices[some] == tuple(f"licenses/{name}" for name in repositories[some]["files"])


def test_a_row_that_gives_no_turn_is_named(monkeypatch):
    monkeypatch.setattr(swehero, "MAX_REQUEST_BYTES", 100)
    skipped: list = []
    assert swehero.build_sets(13, [row(), row(1), row(2, trajectory=TRAJECTORY[:2])], TOOLS, skipped) == {}
    first = size(swehero.messages_for(TRAJECTORY), 2)
    assert skipped == [
        (f"shard 13 row 0: the first assistant turn's request has {first} bytes", swehero.NO_TURN),
        (f"shard 13 row 1: the first assistant turn's request has {first} bytes", swehero.NO_TURN),
        ("shard 13 row 2: the trajectory has no assistant turn", swehero.NO_TURN),
    ]


def test_the_import_reads_every_shard_of_the_training_split_each_pinned_by_its_sha256():
    assert swehero.SHARD_FILES == [f"data/train-{shard:05d}-of-00014.parquet" for shard in range(14)]
    assert list(swehero.FILES) == [hf.CARD, swehero.TOOLS_FILE, *swehero.SHARD_FILES]
    assert all(re.fullmatch("[0-9a-f]{64}", sha256) for sha256 in swehero.FILES.values())
    assert swehero.FILES["data/train-00013-of-00014.parquet"] == (
        "936b195ed11b2b9be2e60cf9cb274dfc2f31d07745cbd6144658da988ce295a9"
    )


def test_each_shard_gives_its_own_sets_named_for_it_and_its_cases_name_the_file_they_come_from():
    sets = swehero.build_sets(0, [row()], TOOLS)
    assert sorted(sets) == [("parse", "swehero-0"), ("render", "swehero-0")]
    [first, *_] = sets["render", "swehero-0"]
    assert first["name"] == "swehero-0-0-2"
    assert (first["origin"]["file"], first["origin"]["sha256"]) == (
        "data/train-00000-of-00014.parquet",
        "5149ba06dc7bcfe2fc01a20036e35cd535fde91840253c15bc070d95cf217be6",
    )


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
    rows = [row(number) for number in range(2 * 64 + 1)]  # the shard is read 64 rows at a time
    assert list(swehero.read_rows(write_shard(tmp_path / "shard.parquet", rows))) == rows


@pytest.mark.usefixtures("within_the_limit")
def test_written_sets_read_back_and_check_clean_and_a_changed_missing_or_stale_file_is_reported(tmp_path):
    sets, corpus = swehero.build_sets(13, [row()], TOOLS), tmp_path / "corpus"
    (corpus / "render").mkdir(parents=True)
    for name in ("common", "bfcl-simple-python", "swehero-12"):
        (corpus / "render" / f"{name}.jsonl").write_text("{}\n")
    swehero.write_sets(sets, corpus, files=LICENSE_TEXTS)
    assert (corpus / NOTICE).read_bytes() == LICENSE_TEXTS[NOTICE]
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
    assert swehero.check_sets(sets, corpus, files=LICENSE_TEXTS) == []
    (corpus / NOTICE).write_bytes(b"changed\n")
    assert swehero.check_sets(sets, corpus, files=LICENSE_TEXTS) == [f"{corpus / NOTICE}: differs from a fresh import"]
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


def serve(tmp_path, monkeypatch, *shards: list[dict], card: str = CARD, tools: list = TOOLS) -> list[tuple]:
    """Stand in for the Hub: the card, tools.json and ``shards`` as the split's first shards, written to ``tmp_path``
    and pinned by their sha256 in place of the real pins; returns the downloads asked for."""
    shard_files = swehero.SHARD_FILES[: len(shards)]
    files = {hf.CARD: tmp_path / "README.md", swehero.TOOLS_FILE: tmp_path / "tools.json"}
    for number, (name, rows) in enumerate(zip(shard_files, shards, strict=True)):
        files[name] = write_shard(tmp_path / f"shard-{number}.parquet", rows)
    files[hf.CARD].write_text(card)
    files[swehero.TOOLS_FILE].write_text(json.dumps(tools))
    pins = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    monkeypatch.setattr(swehero, "SHARD_FILES", shard_files)
    monkeypatch.setattr(swehero, "FILES", pins)
    downloads: list[tuple] = []

    def download(repo_id, filename, **kwargs):
        downloads.append((repo_id, filename, kwargs))
        return str(files[filename])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return downloads


@pytest.fixture
def within_the_limit(monkeypatch):
    """The full import declares its sets compressed; a test that imports a few rows, whose sets stay within the limit,
    takes their form from their total instead, as an import that declares none does."""
    monkeypatch.setattr(swehero, "FORM", None)


def test_the_full_import_declares_its_sets_compressed():
    assert swehero.FORM == "zstd"


def test_a_shards_sets_come_before_the_next_shard_is_fetched(tmp_path, monkeypatch):
    downloads = serve(tmp_path, monkeypatch, [row()], [row(0, trajectory=EDGES)])
    fetched = []

    def fetch(name):
        fetched.append(name)
        return swehero.hf.fetch(swehero.REPO, swehero.REVISION, name, swehero.FILES[name], cache=tmp_path / "cache")

    sets = swehero.iter_sets(fetch, TOOLS, [])
    assert [next(sets)[:2], next(sets)[:2]] == [("render", "swehero-0"), ("parse", "swehero-0")]
    assert fetched == [swehero.SHARD_FILES[0]]
    assert [kind_name[:2] for kind_name in sets] == [("render", "swehero-1"), ("parse", "swehero-1")]
    assert fetched == swehero.SHARD_FILES
    assert len(downloads) == 2


@pytest.mark.usefixtures("within_the_limit")
def test_the_command_reads_every_pinned_shard_under_its_cache_then_writes_and_checks(tmp_path, monkeypatch, capsys):
    downloads = serve(tmp_path, monkeypatch, [row()], [row(0, trajectory=EDGES)])
    corpus = tmp_path / "corpus"
    argv = ["import", "swehero", "--corpus", str(corpus), "--cache", str(tmp_path / "cache")]
    assert main([*argv, "--check"]) == 1
    err = capsys.readouterr().err
    for name in ("swehero-0", "swehero-1"):
        assert f"{corpus / 'parse' / f'{name}.jsonl'}: missing" in err
        assert f"{corpus / 'render' / f'{name}.jsonl'}: missing" in err
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    out = capsys.readouterr().out
    assert f"{corpus / 'parse' / 'swehero-0.jsonl'}: 3 cases, 3 distinct messages\n" in out
    assert f"{corpus / 'parse' / 'swehero-1.jsonl'}: 2 cases, 2 distinct messages\n" in out
    assert f"{corpus / 'render' / 'swehero-0.jsonl'}: 3 cases\n" in out
    assert f"{corpus / 'render' / 'swehero-1.jsonl'}: 2 cases\n" in out
    assert f"{corpus}: 10 cases in the 4 SWE-Hero sets, 0 left out as repeats, 5 distinct messages\n" in out
    assert f"{corpus}: the SWE-Hero sets equal a fresh import of {swehero.SOURCE}, every row of its 2 shards\n" in out
    at_the_pin = {
        "repo_type": "dataset",
        "revision": swehero.REVISION,
        "cache_dir": tmp_path / "cache" / "huggingface",
        "force_download": False,
        "token": False,
    }
    pinned = [(swehero.REPO, name, at_the_pin) for name in (hf.CARD, swehero.TOOLS_FILE, *swehero.SHARD_FILES)]
    assert downloads == pinned * 3


def test_a_card_under_another_license_stops_the_import_before_the_shard_is_fetched(tmp_path, monkeypatch):
    downloads = serve(tmp_path, monkeypatch, [row()], card=CARD.replace("cc-by-4.0", "cc-by-nc-4.0"))
    with pytest.raises(ValueError, match="the card's license is 'cc-by-nc-4.0', not the reviewed 'cc-by-4.0'"):
        main(["import", "swehero", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
    assert [filename for _, filename, _ in downloads] == [hf.CARD]


def test_a_tools_file_not_in_openai_shape_stops_the_import_before_the_shard_is_fetched(tmp_path, monkeypatch):
    downloads = serve(tmp_path, monkeypatch, [row()], tools=[{"name": "execute_bash"}])
    problem = 'tools.json: tool 0 is not {"type": "function", "function": {...}}'
    with pytest.raises(ValueError, match=re.escape(problem)):
        main(["import", "swehero", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
    assert [filename for _, filename, _ in downloads] == [hf.CARD, swehero.TOOLS_FILE]


@pytest.mark.usefixtures("within_the_limit")
def test_the_command_names_every_row_it_leaves_out_in_the_words_of_the_other_imports(tmp_path, monkeypatch, capsys):
    rows = [row(number, "GPL-3.0") for number in range(51)] + [row(51, trajectory=TRAJECTORY[:6]), row(52)]
    serve(tmp_path, monkeypatch, rows)
    assert main(["import", "swehero", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")]) == 0
    out = capsys.readouterr().out
    every = ", ".join(f"shard 0 row {number}: GPL-3.0" for number in range(51))
    assert f"no case for 51 row(s) ({every}): {swehero.LICENSE_NOT_ALLOWED}\n" in out
    unpaired = "shard 0 row 51: message 4 makes 2 call(s) and 1 result(s) follow"
    assert f"no case for 1 row(s) ({unpaired}): {swehero.UNPAIRED}\n" in out


@pytest.mark.usefixtures("within_the_limit")
def test_the_command_leaves_out_a_case_that_repeats_an_earlier_one_across_shards_and_counts_distinct_messages(
    tmp_path, monkeypatch, capsys
):
    # Two trajectories of one task open with the same system prompt and issue, so the requests of their first turns
    # are the same: the second render case repeats the first, here from the next shard. Their parse cases differ, by
    # the turn each expects; the later turns, whose requests differ, expect the same messages, counted once.
    other = [*TRAJECTORY[:2], item("assistant", "Let me read it.", call("call-a", "execute_bash", '{"command": "ls"}'))]
    serve(tmp_path, monkeypatch, [row(0)], [row(0, trajectory=[*other, *TRAJECTORY[3:]])])
    corpus = tmp_path / "corpus"
    argv = ["import", "swehero", "--corpus", str(corpus), "--cache", str(tmp_path / "cache")]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "no case swehero-1-0-2: it repeats swehero-0-0-2\n" in out
    assert f"{corpus / 'parse' / 'swehero-1.jsonl'}: 3 cases, 3 distinct messages\n" in out
    assert f"{corpus / 'render' / 'swehero-1.jsonl'}: 2 cases, 1 left out as repeats\n" in out
    assert f"{corpus}: 11 cases in the 4 SWE-Hero sets, 1 left out as repeats, 4 distinct messages\n" in out
    render = [case.name for case in read_cases(corpus / "render" / "swehero-1.jsonl")]
    assert render == ["swehero-1-0-4", "swehero-1-0-7"]
    assert main([*argv, "--check"]) == 0


def test_past_the_limit_the_command_writes_every_set_compressed_and_checks_it(tmp_path, monkeypatch, capsys):
    # The full import takes 15 GB as plain JSON Lines; here the limit is moved below two shards' worth instead.
    serve(tmp_path, monkeypatch, [row()], [row(0, trajectory=EDGES)])
    corpus = tmp_path / "corpus"
    argv = ["import", "swehero", "--corpus", str(corpus), "--cache", str(tmp_path / "cache")]
    monkeypatch.setattr(swehero, "FORM", None)
    assert main(argv) == 0
    monkeypatch.setattr(swehero, "FORM", "zstd")
    monkeypatch.setattr(corpus_sets, "LIMIT", 1000)
    assert main([*argv, "--check"]) == 1
    assert "a fresh import writes this set as swehero-0.jsonl.zst" in capsys.readouterr().err
    assert main(argv) == 0
    assert sorted(path.relative_to(corpus).as_posix() for path in corpus.rglob("*") if path.is_file()) == [
        NOTICE,  # the copied license file, as it is
        *(f"{kind}/swehero-{shard}.jsonl.zst" for kind in ("parse", "render") for shard in (0, 1)),
    ]
    assert [case.name for case in read_cases(corpus / "parse" / "swehero-0.jsonl.zst")] == [
        "swehero-0-0-2",
        "swehero-0-0-4",
        "swehero-0-0-7",
    ]
    assert f"{corpus / 'render' / 'swehero-1.jsonl.zst'}: 2 cases\n" in capsys.readouterr().out
    assert main([*argv, "--check"]) == 0
