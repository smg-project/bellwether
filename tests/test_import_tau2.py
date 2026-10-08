import copy
import hashlib
import json

import pytest

from bellwether.cli import main
from bellwether.importers import corpus_sets, github, tau2
from bellwether.record.corpus import read_cases

# llm_agent.py as the release writes it, down to what the import reads: six prompt constants, each a triple-quoted
# string stripped, and the solo agent's stop tool and token as class attributes.
AGENT_SOURCE = '''
from tau2.agent.base import LocalAgent

AGENT_INSTRUCTION = """
You are an agent.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

AGENT_GT_INSTRUCTION = """
You are testing the user simulator.
""".strip()

SYSTEM_PROMPT_GT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<resolution_steps>
{resolution_steps}
</resolution_steps>
""".strip()

AGENT_SOLO_INSTRUCTION = """
Solve the ticket; call `{stop_function_name}` when done.
""".strip()

SYSTEM_PROMPT_SOLO = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
""".strip()


class LLMSoloAgent(LocalAgent):
    STOP_FUNCTION_NAME = "done"
    TRANSFER_TOOL_NAME = "transfer_to_human_agents"
    STOP_TOKEN = "###STOP###"
'''

TOOL = {"type": "function", "function": {"name": "get_customer", "description": "Get a customer.", "parameters": {}}}
DONE = {"type": "function", "function": {"name": "done", "description": "Call this when done.", "parameters": {}}}
TOOLS = {"telecom": {"default": [TOOL], "solo": [TOOL, DONE]}}


def raw(content, calls=None, finish="stop"):
    """``raw_data`` as the runs hold it: the provider's response, with each call's arguments as the model wrote them."""
    tool_calls = [
        {"function": {"arguments": arguments, "name": name}, "id": call_id, "type": "function"}
        for call_id, name, arguments in calls or []
    ]
    message = {"content": content, "role": "assistant", "tool_calls": tool_calls or None, "function_call": None}
    return {"finish_reason": finish, "index": 0, "message": message}


def assistant(content, calls=None):
    tool_calls = [
        {"id": call_id, "name": name, "arguments": json.loads(arguments), "requestor": "assistant"}
        for call_id, name, arguments in calls or []
    ]
    finish = "tool_calls" if calls else "stop"
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls or None,
        "raw_data": raw(content, calls, finish),
    }


GREETING = {"role": "assistant", "content": "Hi! How can I help you today?", "tool_calls": None, "raw_data": None}
CALL = ("call_1", "get_customer", '{"id":"c1","note":"café"}')


def run(agent: str = "llm_agent") -> dict:
    """One simulation of a telecom run: the greeting, a user turn, a call and its result, a reply, the user's own call
    on their device and its result (which the agent never sees), and a last exchange."""
    messages = [
        GREETING,
        {"role": "user", "content": "My phone is broken.", "tool_calls": None},
        assistant(None, [CALL]),
        {"role": "tool", "id": "call_1", "content": '{"name": "Ann"}', "requestor": "assistant", "error": False},
        assistant("Please turn airplane mode off."),
        {
            "role": "user",
            "content": None,
            "tool_calls": [{"id": "u_1", "name": "toggle_airplane_mode", "arguments": {}, "requestor": "user"}],
        },
        {"role": "tool", "id": "u_1", "content": "Airplane mode is off.", "requestor": "user", "error": False},
        {"role": "user", "content": "It works now.", "tool_calls": None},
        assistant("Glad to help!"),
    ]
    task = {
        "id": "[mobile]airplane_mode_on",
        "ticket": "The customer's phone has no service.",
        "evaluation_criteria": {
            "actions": [
                {"requestor": "assistant", "name": "get_customer", "arguments": {"id": "c1", "limit": 2.0}},
                {"requestor": "user", "name": "toggle_airplane_mode", "arguments": {"on": False, "tags": ["a"]}},
            ]
        },
    }
    return {
        "info": {
            "agent_info": {"implementation": agent, "llm": "model-1"},
            "environment_info": {"domain_name": "telecom", "policy": "Be kind.", "tool_defs": None},
            "user_info": {"implementation": "user_simulator", "llm": "gpt-4.1"},
        },
        "tasks": [task],
        "simulations": [{"id": "sim-uuid", "task_id": task["id"], "trial": 0, "messages": messages}],
    }


def solo_run() -> dict:
    """A solo run: no user; the stop turn is recorded as the stop token, and its raw response holds the call."""
    found = run("llm_agent_solo")
    stop = assistant(None, [("call_9", "done", "{}")])
    stop["content"], stop["tool_calls"] = "###STOP###", None
    found["simulations"][0]["messages"] = [
        assistant(None, [CALL]),
        {"role": "tool", "id": "call_1", "content": '{"name": "Ann"}', "requestor": "assistant", "error": False},
        stop,
    ]
    return found


@pytest.fixture
def prompts():
    return tau2.prompts(AGENT_SOURCE)


def cases(run_data, prompts, name="model-1_telecom_default_gpt-4.1-2025-04-14_4trials.json"):
    return tau2.cases_for_run(name, "0" * 64, run_data, prompts, TOOLS)


def test_the_prompt_constants_are_read_from_the_source_and_never_run(prompts):
    assert prompts["AGENT_INSTRUCTION"].startswith("You are an agent.")
    assert prompts["SYSTEM_PROMPT"].startswith("<instructions>\n{agent_instruction}")
    assert (prompts["STOP_FUNCTION_NAME"], prompts["STOP_TOKEN"]) == ("done", "###STOP###")
    with pytest.raises(ValueError, match="SYSTEM_PROMPT_SOLO"):
        tau2.prompts(AGENT_SOURCE.replace("SYSTEM_PROMPT_SOLO =", "OTHER ="))


def test_each_agent_gets_the_prompt_its_class_builds(prompts):
    task = run()["tasks"][0]
    default = tau2.system_prompt("llm_agent", prompts, "Be kind.", task)
    assert (
        default == "<instructions>\nYou are an agent.\nIn each turn you can either:\n- Send a message to the user.\n"
        "- Make a tool call.\n</instructions>\n<policy>\nBe kind.\n</policy>"
    )
    gt = tau2.system_prompt("llm_agent_gt", prompts, "Be kind.", task)
    assert gt.endswith(
        "<resolution_steps>\n[Step 1] Perform the following action: get_customer(id=c1, limit=2.0).\n"
        "[Step 2] Instruct the user to perform the following action: toggle_airplane_mode(on=False, tags=['a']).\n"
        "</resolution_steps>"
    )
    solo = tau2.system_prompt("llm_agent_solo", prompts, "Be kind.", task)
    assert "call `done` when done." in solo
    assert solo.endswith("<ticket>\nThe customer's phone has no service.\n</ticket>")


def test_every_turn_the_model_wrote_is_a_parse_case_and_its_request_a_render_case(prompts):
    render, parse = cases(run(), prompts)
    assert [line["name"] for line in parse] == [
        "tau2-model-1-telecom-default-0-2",
        "tau2-model-1-telecom-default-0-4",
        "tau2-model-1-telecom-default-0-8",
    ]
    assert [line["name"] for line in render] == [line["name"] for line in parse]
    for render_line, parse_line in zip(render, parse, strict=True):
        assert render_line["request"] == parse_line["request"]


def test_the_agent_sees_its_prompt_the_user_its_own_turns_and_its_own_results_and_nothing_else(prompts):
    _, parse = cases(run(), prompts)
    last = parse[-1]["request"]["messages"]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "tool", "assistant", "user"]
    # The greeting never reaches the agent's history, nor the user's own call on their device or its result.
    assert all("Hi! How can I help" not in json.dumps(m) for m in last)
    assert all("toggle_airplane_mode" not in json.dumps(m) for m in last)
    assert last[-1] == {"role": "user", "content": "It works now."}
    assert last[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"name": "Ann"}'}


def test_a_call_in_the_history_carries_its_arguments_as_tau2_sent_them(prompts):
    # tau2 wrote the history's arguments with json.dumps and its defaults: ASCII escapes and spaced separators,
    # whatever the model wrote.
    _, parse = cases(run(), prompts)
    call = parse[1]["request"]["messages"][2]["tool_calls"][0]
    assert call["id"] == "call_1" and call["type"] == "function" and call["function"]["name"] == "get_customer"
    arguments = call["function"]["arguments"]
    assert json.loads(arguments) == {"id": "c1", "note": "café"}
    assert arguments.isascii() and '"id": "c1", ' in arguments
    assert parse[1]["request"]["messages"][2]["content"] == ""


def test_a_parse_case_expects_the_turn_as_the_model_wrote_it(prompts):
    _, parse = cases(run(), prompts)
    assert parse[0]["message"] == {
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "get_customer", "arguments": CALL[2]}}],
    }
    assert parse[1]["message"] == {"content": "Please turn airplane mode off."}


def test_a_request_holds_the_tools_and_tool_choice_tau2_sent(prompts):
    _, parse = cases(run(), prompts)
    request = parse[0]["request"]
    assert request["tools"] == [TOOL] and request["tool_choice"] == "auto"
    assert request["messages"][0] == {
        "role": "system",
        "content": tau2.system_prompt("llm_agent", prompts, "Be kind.", run()["tasks"][0]),
    }


def test_a_solo_agent_starts_from_its_prompt_alone_with_its_stop_tool_and_must_call(prompts):
    render, parse = cases(solo_run(), prompts, "model-1_telecom_no-user_gpt-4.1-2025-04-14_4trials.json")
    assert [m["role"] for m in parse[0]["request"]["messages"]] == ["system"]
    assert parse[0]["request"]["tools"] == [TOOL, DONE] and parse[0]["request"]["tool_choice"] == "required"
    # The stop turn is recorded as the stop token; the model wrote the call, and the case expects that.
    assert parse[-1]["message"] == {
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "done", "arguments": "{}"}}],
    }
    assert len(render) == len(parse) == 2


def test_origin_names_the_run_the_turn_and_what_bellwether_wrote(prompts):
    _, parse = cases(run(), prompts)
    origin = parse[0]["origin"]
    assert origin["dataset"] == "tau2" and origin["source"] == tau2.SOURCE and origin["license"] == "MIT"
    assert origin["file"] == "data/tau2/results/final/model-1_telecom_default_gpt-4.1-2025-04-14_4trials.json"
    assert (origin["simulation"], origin["task"], origin["trial"], origin["turn"]) == (
        0,
        "[mobile]airplane_mode_on",
        0,
        2,
    )
    assert (origin["agent"], origin["model"]) == ("llm_agent", "model-1")
    assert origin["written"] == ["system prompt", "tools"]


def test_a_run_of_an_agent_with_no_public_prompt_is_left_out_with_its_reason(prompts):
    skipped = []
    runs = [
        ("a_telecom_no-user-op_gpt-4.1-2025-04-14_4trials.json", "0" * 64, run("llm_agent_solo_gt")),
        ("b_telecom_default_gpt-4.1-2025-04-14_4trials.json", "0" * 64, run()),
    ]
    sets = list(tau2.iter_sets(runs, prompts, TOOLS, skipped))
    assert [name for _, name, _ in sets] == ["tau2-b-telecom-default", "tau2-b-telecom-default"]
    assert skipped == [
        ("a_telecom_no-user-op_gpt-4.1-2025-04-14_4trials.json", tau2.LEFT_OUT_AGENTS["llm_agent_solo_gt"])
    ]


def test_a_set_is_named_by_its_run_without_the_user_model_and_trials():
    name = "claude-3-7-sonnet-20250219_telecom-workflow_no-user_gpt-4.1-2025-04-14_4trials.json"
    assert tau2.set_name(name) == "tau2-claude-3-7-sonnet-20250219-telecom-workflow-no-user"


def test_a_model_name_with_a_dot_gives_a_set_name_the_recorder_takes():
    # The recorder takes names of lowercase letters, digits and hyphens only; GPT-4.1's dot becomes a hyphen.
    assert tau2.set_name("gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json") == (
        "tau2-gpt-4-1-2025-04-14-airline-default"
    )
    assert tau2.set_name("gpt-4.1-mini-2025-04-14_retail_base_gpt-4.1-2025-04-14_4trials.json") == (
        "tau2-gpt-4-1-mini-2025-04-14-retail-base"
    )


def test_the_recorder_reads_every_set_the_import_writes(prompts, tmp_path, monkeypatch):
    # The import declares its sets compressed, which corpus_sets allows only past LIMIT; these sets are small.
    monkeypatch.setattr(corpus_sets, "LIMIT", 1000)
    runs = [("gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json", "0" * 64, run())]
    tau2.write_sets(tau2.iter_sets(runs, prompts, TOOLS, []), tmp_path, b"MIT License\n")
    written = sorted(tmp_path.glob("*/tau2-*.jsonl.zst"))
    assert [path.parent.name for path in written] == ["parse", "render"]
    for path in written:
        assert read_cases(path)


def test_written_sets_check_equal_and_a_changed_case_is_named(prompts, tmp_path, monkeypatch):
    # The import declares its sets compressed, which corpus_sets allows only past LIMIT; these sets are small.
    monkeypatch.setattr(corpus_sets, "LIMIT", 1000)
    runs = [("m_telecom_default_gpt-4.1-2025-04-14_4trials.json", "0" * 64, run())]
    tau2.write_sets(tau2.iter_sets(runs, prompts, TOOLS, []), tmp_path, b"MIT License\n")
    assert tau2.check_sets(tau2.iter_sets(runs, prompts, TOOLS, []), tmp_path, b"MIT License\n") == []
    changed = copy.deepcopy(runs[0][2])
    changed["simulations"][0]["messages"][1]["content"] = "My phone is fine."
    problems = tau2.check_sets(
        tau2.iter_sets([(runs[0][0], runs[0][1], changed)], prompts, TOOLS, []), tmp_path, b"MIT License\n"
    )
    assert problems and all("tau2-m-telecom-default" in problem for problem in problems)
    assert (tmp_path / tau2.LICENSE_COPY).read_bytes() == b"MIT License\n"


LICENSE_TEXT = b"MIT License\n\nCopyright (c) 2025 Sierra Technologies Inc.\n"


def serve(tmp_path, monkeypatch, runs: dict[str, dict]) -> list[tuple[str, str]]:
    """tau2-bench's files at their pins, from memory: the LICENSE and the runs at the runs' commit, llm_agent.py at the
    release. ``RUNS`` and the reviewed sha256s become theirs; each fetch's ``(commit, path)`` is recorded."""
    files = {
        (tau2.COMMIT, tau2.LICENSE_FILE): LICENSE_TEXT,
        (tau2.PROMPT_COMMIT, tau2.AGENT_FILE): AGENT_SOURCE.encode(),
    }
    for name, data in runs.items():
        files[(tau2.COMMIT, f"{tau2.RUNS_DIR}/{name}")] = json.dumps(data).encode()
    digest = {key: hashlib.sha256(data).hexdigest() for key, data in files.items()}
    monkeypatch.setattr(tau2, "RUNS", {name: digest[(tau2.COMMIT, f"{tau2.RUNS_DIR}/{name}")] for name in runs})
    monkeypatch.setattr(tau2, "LICENSE_SHA256", digest[(tau2.COMMIT, tau2.LICENSE_FILE)])
    monkeypatch.setattr(tau2, "AGENT_SHA256", digest[(tau2.PROMPT_COMMIT, tau2.AGENT_FILE)])
    monkeypatch.setattr(tau2, "load_tools", lambda: TOOLS)
    fetched = []

    def fetch(owner, repo, commit, path, sha256, cache):
        assert (owner, repo) == (tau2.OWNER, tau2.REPO)
        assert sha256 == digest[(commit, path)]
        fetched.append((commit, path))
        target = cache / "github" / owner / repo / commit / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[(commit, path)])
        return target

    monkeypatch.setattr(github, "fetch", fetch)
    return fetched


def test_the_command_fetches_each_file_at_its_pin_then_writes_and_checks_and_names_the_run_it_leaves_out(
    tmp_path, monkeypatch, capsys
):
    # The import declares its sets compressed, which corpus_sets allows only past LIMIT; these sets are small.
    monkeypatch.setattr(corpus_sets, "LIMIT", 1000)
    kept, left_out = (f"m_telecom_{variant}_gpt-4.1-2025-04-14_4trials.json" for variant in ("default", "no-user-op"))
    fetched = serve(tmp_path, monkeypatch, {kept: run(), left_out: run("llm_agent_solo_gt")})
    corpus = tmp_path / "corpus"
    argv = ["import", "tau2", "--corpus", str(corpus), "--cache", str(tmp_path / "cache")]
    assert main([*argv, "--check"]) == 1
    err = capsys.readouterr().err
    for kind in ("parse", "render"):
        assert f"{corpus / kind / 'tau2-m-telecom-default.jsonl.zst'}: missing" in err
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert f"no case for 1 row(s) ({left_out}): {tau2.LEFT_OUT_AGENTS['llm_agent_solo_gt']}\n" in out
    assert (corpus / tau2.LICENSE_COPY).read_bytes() == LICENSE_TEXT
    assert main([*argv, "--check"]) == 0
    assert f"{corpus}: the tau2 sets equal a fresh import of {tau2.SOURCE}\n" in capsys.readouterr().out
    pins = [(tau2.COMMIT, tau2.LICENSE_FILE), (tau2.PROMPT_COMMIT, tau2.AGENT_FILE)]
    runs = [(tau2.COMMIT, f"{tau2.RUNS_DIR}/{name}") for name in (kept, left_out)]
    assert fetched == (pins + runs) * 3
