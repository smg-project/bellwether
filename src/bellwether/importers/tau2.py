"""tau2-bench's published agent runs as corpus sets: every turn an agent model wrote, as its agent saw the conversation.

The data is the paper's runs in ``sierra-research/tau2-bench`` at a pinned commit, ``data/tau2/results/final``: 26 JSON
files, one per agent model, domain (airline, retail, telecom, telecom-workflow) and variant, each with four trials of
every task, the user played by GPT-4.1. Each file is fetched by its path at the commit and checked against its sha256.

A file records the conversation, not the agent's requests, so the import rebuilds what the agent was sent:

- **The system prompt** is tau2's own, read from ``src/tau2/agent/llm_agent.py`` at the release (``PROMPT_COMMIT``),
  since the commit the runs record (``c30d59aa``) is not public: ``SYSTEM_PROMPT`` with ``AGENT_INSTRUCTION`` and the
  run's domain policy; for the ``_gt`` agent, ``SYSTEM_PROMPT_GT`` with the task's resolution steps, written as
  ``make_agent_instructions_from_actions`` writes them; for the ``_solo`` agent, ``SYSTEM_PROMPT_SOLO`` with its stop
  tool and the task's ticket. The constants are read from the file's syntax tree, never run.
- **The tools** are the schemas tau2's ``as_tool`` builds at the release, read from ``tau2_tools.json``: the runs record
  none (``tool_defs`` is null), and ``scripts/tau2_tools.py`` builds the table by running tau2 in bellwether's sandbox.
  A solo agent also has the user's tools and ``done``, and must call one (``tool_choice`` ``required``); the others are
  sent ``auto``.
- **The history** is the agent's own, as ``to_litellm_messages`` gives it: the user's messages, the agent's turns with
  their calls' arguments as ``json.dumps`` writes them, and the results of the agent's calls. The greeting tau2 opens
  with never reaches the agent's history, nor do the calls the user makes on their device or their results.

A turn the model wrote (one with ``raw_data``) is a parse case, which expects the turn as the model wrote it, each
call's arguments as ``raw_data`` holds them and no call id; its request is a render case. A call-only turn's content,
null in the provider's response, is written as "", as the other imports write it. The prompt and the tools are text
bellwether rebuilt, so ``origin.written`` names them. A run whose agent has no public prompt (``LEFT_OUT_AGENTS``) is
left out and named. The sets take more than ``corpus_sets.LIMIT`` as plain JSON Lines, so the import declares them
compressed and streams them one run at a time. The repository's MIT License goes beside them (``LICENSE_COPY``).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from . import corpus_sets, github

OWNER, REPO = "sierra-research", "tau2-bench"
COMMIT = "4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699"  # the runs and the LICENSE
SOURCE = f"github:{OWNER}/{REPO}@{COMMIT}"
DATASET = "tau2"
LICENSE = "MIT"
LICENSE_FILE = "LICENSE"
LICENSE_SHA256 = "e67c5aa0074dfcaefd3c3a1aedb94cb539234aecd15d5a972574e3200e6252fe"
LICENSE_TITLE = "MIT License"  # the reviewed LICENSE file's first line
# Where the import writes the LICENSE, under the corpus root.
LICENSE_COPY = "licenses/tau2-bench-LICENSE"
# The release: the public commit nearest the runs, whose llm_agent.py holds the agents' prompts.
PROMPT_COMMIT = "37199f36924c8896f5e048360691f8476cd89ba1"
AGENT_FILE = "src/tau2/agent/llm_agent.py"
AGENT_SHA256 = "5b5b404ae261f4cbaf5769cb7b3cec09a1442f28b5975bb72e4dc857ead03742"
PROMPT_CONSTANTS = (
    "AGENT_INSTRUCTION",
    "SYSTEM_PROMPT",
    "AGENT_GT_INSTRUCTION",
    "SYSTEM_PROMPT_GT",
    "AGENT_SOLO_INSTRUCTION",
    "SYSTEM_PROMPT_SOLO",
)
SOLO_CLASS = "LLMSoloAgent"
SOLO_CONSTANTS = ("STOP_FUNCTION_NAME", "STOP_TOKEN")
TOOLS_TABLE = Path(__file__).with_name("tau2_tools.json")
RUNS_DIR = "data/tau2/results/final"
# Each run's file, by the sha256 of its bytes, in the order the sets are built.
RUNS = {
    "claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json": (
        "40a2c6a246eab27db5cdefda895fbe7f44be78a23d300817248534aa606d66de"
    ),
    "claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json": (
        "ed41dbd18c080154156484e3a0122c095e324a11367a640d88e15956daed7b9d"
    ),
    "claude-3-7-sonnet-20250219_telecom_default_gpt-4.1-2025-04-14_4trials.json": (
        "f49e540896fe91ab8631647f02eb777ef6fed5a6ebec545f43e71d0504e227b2"
    ),
    "gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json": (
        "31d19e79b8ddc5934a60701f913ff5fefb41dee5e68f7b813c26254ab2e2d5c8"
    ),
    "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json": (
        "5fc5b96ada0fe46a463eaed98d1bfed9947fe073bac052290162dae18d71394e"
    ),
    "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json": (
        "8e1793f57a9e216514dbce1f93b44b0592d5904f1c5b97ab711906782107a1f2"
    ),
    "gpt-4.1-2025-04-14_telecom_no-user_gpt-4.1-2025-04-14_4trials.json": (
        "6dd63dc15ffe0b6e4e79434f4d292b1c0af10db3c91c08e152258795852f8dd1"
    ),
    "gpt-4.1-2025-04-14_telecom_no-user-op_gpt-4.1-2025-04-14_4trials.json": (
        "2065bec4a0f9917af8e031624a8ccc3d69a014f7da6761840fc95c1262ce696a"
    ),
    "gpt-4.1-2025-04-14_telecom_op_gpt-4.1-2025-04-14_4trials.json": (
        "16c0bcb5fa4ad621b50141f5f93d73c46557adca86d1894d1686a57603c34f29"
    ),
    "gpt-4.1-2025-04-14_telecom-workflow_default_gpt-4.1-2025-04-14_4trials.json": (
        "edf9d65e122c850966615488e181a51a7c9ff1d7244e6f908aced9dbc5e56960"
    ),
    "gpt-4.1-2025-04-14_telecom-workflow_no-user_gpt-4.1-2025-04-14_4trials.json": (
        "c2a5208ee3c9dc68f4c539a66244d032b8adb2ac5f33e3339246c3dca8af6c47"
    ),
    "gpt-4.1-2025-04-14_telecom-workflow_no-user-op_gpt-4.1-2025-04-14_4trials.json": (
        "ae28ecfe2a3870fb739e4d1a2ebd0fc8ade9571e38b4a4c167e5c3716800e749"
    ),
    "gpt-4.1-2025-04-14_telecom-workflow_op_gpt-4.1-2025-04-14_4trials.json": (
        "3d6a5b58e89dc13fd71829ba53000c80768ea36da1851b6ea96a212a116ba7a0"
    ),
    "gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json": (
        "41788956547a100fb8e34baf159f36dd1f07f0765798a5fdaf917576c5dba9d0"
    ),
    "gpt-4.1-mini-2025-04-14_retail_base_gpt-4.1-2025-04-14_4trials.json": (
        "6d6badb43b716adca31591b0b40e15fd493b49adddaa8e2c47035bb557549257"
    ),
    "gpt-4.1-mini-2025-04-14_telecom_base_gpt-4.1-2025-04-14_4trials.json": (
        "04fad1a6fb3ff8804be31c54cf5a993b64868e37683c226e014e7adbf60841d9"
    ),
    "o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json": (
        "6a3a354c9bce42b52905af096e9cea6f555548acbb17b567fd403c356df7b4ec"
    ),
    "o4-mini-2025-04-16_retail_default_gpt-4.1-2025-04-14_4trials.json": (
        "7135f38bbbbd46d6babe830574c8623bb276d8fd6da316580b905cf513853f98"
    ),
    "o4-mini-2025-04-16_telecom_default_gpt-4.1-2025-04-14_4trials.json": (
        "1f46a6f296694bdb3ae203e6b379cdada0b8f17b648a4cf90d6343de1f1fb070"
    ),
    "o4-mini-2025-04-16_telecom_no-user_gpt-4.1-2025-04-14_4trials.json": (
        "b9a1310bcd27fdbb04d1a095812791b12f97fde6b3586063613bcd521490e834"
    ),
    "o4-mini-2025-04-16_telecom_no-user-op_gpt-4.1-2025-04-14_4trials.json": (
        "e4f0e4827bdd36438f6daf80ceecffe7f9fd175cef80ae1f821422cb4e5743c4"
    ),
    "o4-mini-2025-04-16_telecom_op_gpt-4.1-2025-04-14_4trials.json": (
        "b7291c709a0ac3636306355656bcd5f81646b49dc1099454fc392937981a0380"
    ),
    "o4-mini-2025-04-16_telecom-workflow_default_gpt-4.1-2025-04-14_4trials.json": (
        "918bea6c28adb6be4ce308d2897a6549ff06b6852fa2c784143a28c511c7b97f"
    ),
    "o4-mini-2025-04-16_telecom-workflow_no-user_gpt-4.1-2025-04-14_4trials.json": (
        "9fa6d8bd2dce2c53d3bf500a4ec914e6f13c9ee83827193301bc5fcfd285cdb0"
    ),
    "o4-mini-2025-04-16_telecom-workflow_no-user-op_gpt-4.1-2025-04-14_4trials.json": (
        "671b820995cd858a3e3abe3da0f6cf943ad9b95220fafa2c729f2e9d8b8c4160"
    ),
    "o4-mini-2025-04-16_telecom-workflow_op_gpt-4.1-2025-04-14_4trials.json": (
        "c6fc8f601d0bf398d2e06f21aa0b4d832dbfc0998d2b193cd27331578790177d"
    ),
}
# Agents whose prompt no public commit of tau2 defines: their runs are left out, named with this reason.
LEFT_OUT_AGENTS = {
    "llm_agent_solo_gt": (
        "no public commit of tau2 defines this agent's prompt (it is registered only at the unpublished commit the "
        "runs record)"
    ),
}
SOLO_AGENTS = ("llm_agent_solo",)
WRITTEN = ["system prompt", "tools"]
# The sets take about 6 GB as plain JSON Lines, past corpus_sets.LIMIT: the import streams them, one run at a time, and
# declares their form rather than let corpus_sets write them plain first.
FORM = "zstd"
SET_PREFIX = f"{DATASET}-"
USER_MODEL_SUFFIX = "_gpt-4.1-2025-04-14_4trials.json"


def prompts(source: str) -> dict[str, str]:
    """The prompt constants and the solo agent's stop tool and token, read from ``llm_agent.py``'s syntax tree.

    Each constant is a string literal, or one with ``.strip()`` called on it, as the release writes them; the file is
    parsed, never run.
    """
    found: dict[str, str] = {}
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in PROMPT_CONSTANTS:
                found[node.targets[0].id] = _literal(node.value)
        elif isinstance(node, ast.ClassDef) and node.name == SOLO_CLASS:
            for item in node.body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                    if item.targets[0].id in SOLO_CONSTANTS:
                        found[item.targets[0].id] = _literal(item.value)
    missing = [name for name in (*PROMPT_CONSTANTS, *SOLO_CONSTANTS) if name not in found]
    if missing:
        raise ValueError(f"{AGENT_FILE} at {PROMPT_COMMIT}: no {', '.join(missing)}")
    return found


def _literal(node: ast.expr) -> str:
    """A string literal, stripped when ``.strip()`` is called on it."""
    stripped = (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "strip"
        and not node.args
        and not node.keywords
    )
    value = node.func.value if stripped else node
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
        raise ValueError(f"{AGENT_FILE}: a prompt constant is not a string literal (line {node.lineno})")
    return value.value.strip() if stripped else value.value


def system_prompt(agent: str, constants: dict[str, str], policy: str, task: dict) -> str:
    """The system prompt ``agent``'s class builds, for the run's domain policy and the simulation's task."""
    if agent == "llm_agent":
        return constants["SYSTEM_PROMPT"].format(domain_policy=policy, agent_instruction=constants["AGENT_INSTRUCTION"])
    if agent == "llm_agent_gt":
        return constants["SYSTEM_PROMPT_GT"].format(
            agent_instruction=constants["AGENT_GT_INSTRUCTION"],
            domain_policy=policy,
            resolution_steps=resolution_steps(task),
        )
    if agent in SOLO_AGENTS:
        instruction = constants["AGENT_SOLO_INSTRUCTION"].format(
            stop_function_name=constants["STOP_FUNCTION_NAME"], stop_token=constants["STOP_TOKEN"]
        )
        return constants["SYSTEM_PROMPT_SOLO"].format(
            agent_instruction=instruction, domain_policy=policy, ticket=task["ticket"]
        )
    raise ValueError(f"no prompt for the agent {agent!r}")


def resolution_steps(task: dict) -> str:
    """The task's expected actions as the ``_gt`` agent lists them, with their arguments (its default): one step per
    action, an agent's action to perform or a user's to ask for, written ``name(key=value, ...)`` with each value as
    Python prints it."""
    lines = []
    for number, action in enumerate(task["evaluation_criteria"]["actions"], start=1):
        call = f"{action['name']}({', '.join(f'{key}={value}' for key, value in action['arguments'].items())})"
        if action["requestor"] == "user":
            lines.append(f"[Step {number}] Instruct the user to perform the following action: {call}.")
        else:
            lines.append(f"[Step {number}] Perform the following action: {call}.")
    return "\n".join(lines)


def history(message: dict) -> dict | None:
    """``message`` as the agent's history holds it, or None when the agent never sees it: the greeting tau2 opens with
    (no ``raw_data``), the user's calls on their device, and their results."""
    role = message["role"]
    if role == "assistant":
        if not message.get("raw_data"):
            return None
        found = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            found["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                }
                for call in message["tool_calls"]
            ]
        return found
    if role == "user":
        return None if message.get("tool_calls") else {"role": "user", "content": message.get("content") or ""}
    if role == "tool":
        if message.get("requestor") != "assistant":
            return None
        return {"role": "tool", "tool_call_id": message["id"], "content": message.get("content") or ""}
    raise ValueError(f"a message of the role {role!r}")


def expected(raw: dict) -> dict:
    """The turn as the model wrote it, from the provider's response: its content, and each call with its arguments as
    written and no id, since a parser makes up its own."""
    message = raw["message"]
    found: dict = {"content": message.get("content") or ""}
    calls = message.get("tool_calls") or []
    if calls:
        found["tool_calls"] = [
            {
                "type": "function",
                "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]},
            }
            for call in calls
        ]
    return found


def set_name(file_name: str) -> str:
    """The run's set: its agent model, domain and variant, without the user model and trial count every run shares."""
    stem = file_name.removesuffix(USER_MODEL_SUFFIX).removesuffix(".json")
    return SET_PREFIX + stem.replace("_", "-")


def origin(file_name: str, sha256: str, simulation: int, sim: dict, turn: int, agent: str, model: str) -> dict:
    return {
        "dataset": DATASET,
        "source": SOURCE,
        "sha256": sha256,
        "file": f"{RUNS_DIR}/{file_name}",
        "simulation": simulation,
        "task": sim["task_id"],
        "trial": sim["trial"],
        "turn": turn,
        "agent": agent,
        "model": model,
        "license": LICENSE,
        "written": list(WRITTEN),
    }


def cases_for_run(
    file_name: str, sha256: str, run: dict, constants: dict[str, str], tools: dict
) -> tuple[list[dict], list[dict]]:
    """The run's render and parse cases: for every turn the model wrote, the request the agent sent and the turn."""
    info = run["info"]
    agent, model = info["agent_info"]["implementation"], info["agent_info"]["llm"]
    domain, policy = info["environment_info"]["domain_name"], info["environment_info"]["policy"]
    kind = "solo" if agent in SOLO_AGENTS else "default"
    run_tools = tools[domain][kind]
    tool_choice = "required" if kind == "solo" else "auto"
    tasks = {task["id"]: task for task in run["tasks"]}
    name = set_name(file_name)
    render, parse = [], []
    for simulation, sim in enumerate(run["simulations"]):
        system = {"role": "system", "content": system_prompt(agent, constants, policy, tasks[sim["task_id"]])}
        seen: list[dict] = [system]
        for turn, message in enumerate(sim["messages"]):
            if message["role"] == "assistant" and message.get("raw_data"):
                request = {"messages": list(seen), "tools": run_tools, "tool_choice": tool_choice}
                case = f"{name}-{simulation}-{turn}"
                notes = f"tau2-bench {file_name} simulation {simulation} turn {turn}"
                where = origin(file_name, sha256, simulation, sim, turn, agent, model)
                render.append({"name": case, "request": request, "notes": notes, "origin": where})
                parse.append(
                    {
                        "name": case,
                        "request": request,
                        "message": expected(message["raw_data"]),
                        "notes": notes,
                        "origin": where,
                    }
                )
            kept = history(message)
            if kept is not None:
                seen.append(kept)
    return render, parse


def iter_sets(
    runs: Iterable[tuple[str, str, dict]], constants: dict[str, str], tools: dict, skipped: list[tuple[str, str]]
) -> Iterator[tuple[str, str, list[dict]]]:
    """Each run's render and parse sets, ``(kind, set name, lines)``, one run at a time; a run whose agent has no
    public prompt is appended to ``skipped`` with its reason instead."""
    for file_name, sha256, run in runs:
        agent = run["info"]["agent_info"]["implementation"]
        if agent in LEFT_OUT_AGENTS:
            skipped.append((file_name, LEFT_OUT_AGENTS[agent]))
            continue
        render, parse = cases_for_run(file_name, sha256, run, constants, tools)
        yield "render", set_name(file_name), render
        yield "parse", set_name(file_name), parse


def fetch(path: str, sha256: str, cache: Path, commit: str = COMMIT) -> Path:
    return github.fetch(OWNER, REPO, commit, path, sha256, cache=cache)


def read_runs(cache: Path) -> Iterator[tuple[str, str, dict]]:
    """Each run, fetched, checked and read, one at a time."""
    for file_name, sha256 in RUNS.items():
        yield file_name, sha256, json.loads(fetch(f"{RUNS_DIR}/{file_name}", sha256, cache).read_bytes())


def load_tools() -> dict:
    return json.loads(TOOLS_TABLE.read_text("utf-8"))["tools"]


def check_license(text: bytes) -> None:
    if text.decode("utf-8").splitlines()[0].strip() != LICENSE_TITLE:
        raise ValueError(f"{LICENSE_FILE} at {COMMIT}: its first line is not {LICENSE_TITLE!r}")


def write_sets(
    sets: corpus_sets.Sets, corpus_dir: Path, license_text: bytes, summary: corpus_sets.Summary | None = None
) -> list[Path]:
    files = {LICENSE_COPY: license_text}
    return corpus_sets.write(sets, corpus_dir, SET_PREFIX, files, form=FORM, summary=summary)


def check_sets(sets: corpus_sets.Sets, corpus_dir: Path, license_text: bytes) -> list[str]:
    files = {LICENSE_COPY: license_text}
    return corpus_sets.check(sets, corpus_dir, SET_PREFIX, "tau2-bench run", files, form=FORM)


def run(args: argparse.Namespace) -> int:
    license_text = fetch(LICENSE_FILE, LICENSE_SHA256, args.cache).read_bytes()
    check_license(license_text)
    constants = prompts(fetch(AGENT_FILE, AGENT_SHA256, args.cache, PROMPT_COMMIT).read_text("utf-8"))
    tools = load_tools()
    skipped: list[tuple[str, str]] = []
    sets = iter_sets(read_runs(args.cache), constants, tools, skipped)
    if args.check:
        problems = check_sets(sets, args.corpus, license_text)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the tau2 sets equal a fresh import of {SOURCE}")
        return 1 if problems else 0
    summary = corpus_sets.Summary()
    write_sets(sets, args.corpus, license_text, summary)
    corpus_sets.report_summary("tau2-bench", summary, args.corpus)
    corpus_sets.report_skipped(skipped)
    return 0
