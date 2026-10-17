"""The hand-written sets committed under ``corpus/``, the inputs written in this repository (corpus/README.md).

Nothing reads them until ``record`` does for a checkpoint, so a malformed line would first show on the recorder; this
reads them as ``record`` reads them, and holds them to the rules the README states for a parse case's message and a
history's tool messages.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from bellwether.record.corpus import read_cases

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
HAND_WRITTEN = {"render": ["common"], "parse": ["common", "turns"]}


def cases(kind: str, name: str):
    return read_cases(CORPUS / kind / f"{name}.jsonl")


@pytest.mark.parametrize("kind", sorted(HAND_WRITTEN))
def test_every_hand_written_set_reads_as_record_reads_it_and_its_names_are_its_own(kind):
    seen: dict[str, str] = {}
    for name in HAND_WRITTEN[kind]:
        read = cases(kind, name)
        assert read, f"{kind}/{name} holds no case"
        for case in read:
            assert case.name not in seen, f"{kind}/{name}/{case.name} is already a case of {seen[case.name]}"
            seen[case.name] = name
            assert case.notes, f"{kind}/{name}/{case.name}: say what the case probes"


@pytest.mark.parametrize("name", HAND_WRITTEN["parse"])
def test_a_hand_written_parse_case_gives_its_message_whose_calls_carry_no_id(name):
    for case in cases("parse", name):
        assert case.message is not None, f"{name}/{case.name}: a parse case needs the message the output parses to"
        for call in case.message.get("tool_calls") or []:
            assert set(call) == {"type", "function"}, f"{name}/{case.name}: a call holds type and function only"
            assert isinstance(call["function"]["arguments"], str), f"{name}/{case.name}: arguments are the JSON string"
            json.loads(call["function"]["arguments"])


@pytest.mark.parametrize(("kind", "name"), [(kind, name) for kind, names in HAND_WRITTEN.items() for name in names])
def test_a_tool_message_in_a_hand_written_history_answers_a_call_of_that_history(kind, name):
    for case in cases(kind, name):
        ids = {
            call["id"]
            for message in case.request["messages"]
            if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        }
        for message in case.request["messages"]:
            if message.get("role") == "tool":
                where = f"{kind}/{name}/{case.name}"
                assert message.get("tool_call_id") in ids, f"{where}: the tool message answers no call of the history"
