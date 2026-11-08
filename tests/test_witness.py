"""``bellwether witness``: a library's reading of recorded parse cases, written beside each case's reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bellwether import witness
from bellwether.manifest import find_manifest
from bellwether.record import sets as set_tables
from bellwether.record.fixtures import read_fixture_file, schema_path, write_fixture_file

REVISION = "0" * 40
MODEL = "org/Tiny-Chat"
LIBRARY = "deepseek-recipe"


def parse_line(name: str, text: str, witnesses: dict | None = None) -> dict:
    line = {
        "id": f"tiny-chat/parse/{name}",
        "kind": "parse",
        "model": MODEL,
        "request": {
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "output_ids": [5, 6],
        "output_pieces": [text[:1], text[1:]],
        "reference": {
            "source": "roundtrip",
            "text": text,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
            "end_of_turn": {"stop_id": 7, "found_by": "turn"},
        },
    }
    if witnesses:
        line["witnesses"] = witnesses
    return line


def a_tree(tmp_path: Path) -> Path:
    fixtures = tmp_path / "fixtures"
    slug = fixtures / "tiny-chat"
    (slug / "parse").mkdir(parents=True)
    (slug / "manifest.toml").write_text(f'model = "{MODEL}"\nrevision = "{REVISION}"\nencoder = "deepseek_v41"\n')
    lines = {
        "tiny-chat/parse/a": parse_line("a", "ab"),
        "tiny-chat/parse/b": parse_line("b", "cd", {"vllm": {"version": "0.31.0"}}),
    }
    write_fixture_file(slug / "parse" / "common.jsonl", lines)
    table = set_tables.entry(
        "plain",
        (slug / "parse" / "common.jsonl").read_text(),
        2,
        1,
        generate_stop_ids=[7],
        generate_stop_ids_from="config.json",
    )
    set_tables.update(slug / "sets.toml", "parse", {"common": table})
    return fixtures


def a_reader(case: dict) -> dict:
    if case["id"].endswith("/b"):
        return {"version": "0.0-test", "rejected": "the test refuses b"}
    return {
        "version": "0.0-test",
        "message": {"role": "assistant", "content": case["reference"]["text"]},
        "finish_reason": "stop",
    }


def test_a_witness_reading_is_written_beside_the_reference_and_the_set_table_follows(tmp_path):
    fixtures = a_tree(tmp_path)
    manifest = find_manifest(fixtures, MODEL)
    path = fixtures / "tiny-chat" / "parse" / "common.jsonl"
    before = read_fixture_file(path)

    summaries = witness.witness_sets(fixtures, manifest, LIBRARY, a_reader)

    after = read_fixture_file(path)
    assert summaries == [f"{path}: 2 cases witnessed by {LIBRARY}, 1 refused by it, {summaries[0].rsplit(', ', 1)[1]}"]
    assert after["tiny-chat/parse/a"]["witnesses"] == {
        LIBRARY: {"version": "0.0-test", "message": {"role": "assistant", "content": "ab"}, "finish_reason": "stop"}
    }
    # The engine witness stays beside the library's; a refusal records its reason and no message.
    assert after["tiny-chat/parse/b"]["witnesses"] == {
        "vllm": {"version": "0.31.0"},
        LIBRARY: {"version": "0.0-test", "rejected": "the test refuses b"},
    }
    for case_id, case in after.items():
        assert {key: value for key, value in case.items() if key != "witnesses"} == {
            key: value for key, value in before[case_id].items() if key != "witnesses"
        }
    table = set_tables.read(fixtures / "tiny-chat" / "sets.toml")[("parse", "common")]
    assert (table["cases"], table["rejected"], table["generate_stop_ids"], table["generate_stop_ids_from"]) == (
        2,
        1,
        [7],
        "config.json",
    )
    assert table["plain_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert table["plain_bytes"] == len(path.read_bytes())

    # A second run reads the same cases and writes the same bytes.
    first = path.read_bytes()
    witness.witness_sets(fixtures, manifest, LIBRARY, a_reader)
    assert path.read_bytes() == first


def test_only_the_named_sets_are_witnessed(tmp_path):
    fixtures = a_tree(tmp_path)
    slug = fixtures / "tiny-chat"
    write_fixture_file(slug / "parse" / "turns.jsonl", {"tiny-chat/parse/t": parse_line("t", "ef")})
    manifest = find_manifest(fixtures, MODEL)

    summaries = witness.witness_sets(fixtures, manifest, LIBRARY, a_reader, ["turns"])

    assert len(summaries) == 1 and str(slug / "parse" / "turns.jsonl") in summaries[0]
    assert "witnesses" not in read_fixture_file(slug / "parse" / "common.jsonl")["tiny-chat/parse/a"]
    assert LIBRARY in read_fixture_file(slug / "parse" / "turns.jsonl")["tiny-chat/parse/t"]["witnesses"]


def test_the_schema_takes_a_witness_message_or_a_refusal_and_never_both():
    validator = Draft202012Validator(json.loads(schema_path().read_text()))
    message = {"version": "0.2.0", "message": {"role": "assistant", "content": "ab"}, "finish_reason": "stop"}
    refusal = {"version": "0.2.0", "rejected": "no"}
    good = [message, refusal, {"version": "0.31.0", "image": "an engine image"}]
    bad = [
        {**message, "rejected": "no"},
        {"version": "0.2.0", "message": {"role": "assistant", "content": "ab"}},
        {"version": "0.2.0", "finish_reason": "stop"},
        {"message": {"role": "assistant"}, "finish_reason": "stop"},
    ]
    for reading in good:
        assert not list(validator.iter_errors(parse_line("a", "ab", {LIBRARY: reading}))), reading
    for reading in bad:
        assert list(validator.iter_errors(parse_line("a", "ab", {LIBRARY: reading}))), reading
    assert list(validator.iter_errors(parse_line("a", "ab", {"some-other-reader": message})))


def test_the_library_body_spells_the_template_switches_as_the_librarys_fields():
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"},
    }
    body = witness.library_body(request, MODEL)
    assert body == {
        "model": MODEL,
        "messages": request["messages"],
        "tools": request["tools"],
        "thinking": {"type": "disabled"},
        "reasoning_effort": "low",
    }
    # A request without the switches hands the library the messages alone; one that spells them itself keeps them.
    assert witness.library_body({"messages": []}, MODEL) == {"model": MODEL, "messages": []}
    spelled = {"messages": [], "thinking": {"type": "enabled"}, "chat_template_kwargs": {"thinking": False}}
    assert witness.library_body(spelled, MODEL)["thinking"] == {"type": "enabled"}


def test_run_says_when_the_library_is_missing_or_the_kind_is_not_served(tmp_path, monkeypatch, capsys):
    fixtures = a_tree(tmp_path)

    def missing(manifest):
        raise ImportError("No module named 'deepseek_recipe'")

    monkeypatch.setitem(witness.LIBRARIES, LIBRARY, missing)
    args = argparse.Namespace(model=MODEL, library=LIBRARY, kind="parse", fixtures=fixtures, sets=None)
    assert witness.run(args) == 2
    assert "pip install -e '.[deepseek]'" in capsys.readouterr().err
    assert witness.run(argparse.Namespace(**{**vars(args), "kind": "render"})) == 2
    assert witness.run(argparse.Namespace(**{**vars(args), "model": "org/Other"})) == 1


def test_run_refuses_a_checkpoint_the_library_does_not_read(tmp_path, capsys):
    pytest.importorskip("deepseek_recipe")
    fixtures = a_tree(tmp_path)
    (fixtures / "tiny-chat" / "manifest.toml").write_text(f'model = "{MODEL}"\nrevision = "{REVISION}"\n')
    args = argparse.Namespace(model=MODEL, library=LIBRARY, kind="parse", fixtures=fixtures, sets=None)
    assert witness.run(args) == 1
    assert "reads the DeepSeek V4 and V4.1 encoders" in capsys.readouterr().err


def test_the_library_reads_a_recorded_case_through_its_stream_processor(tmp_path):
    """Opt-in: the library installed and ``BELLWETHER_DEEPSEEK_TOKENIZER`` naming a V4.1 ``tokenizer.json``."""
    recipe = pytest.importorskip("deepseek_recipe")
    tokenizer_path = os.environ.get("BELLWETHER_DEEPSEEK_TOKENIZER")
    if not tokenizer_path:
        pytest.skip("BELLWETHER_DEEPSEEK_TOKENIZER names no tokenizer.json")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tokenizer.json").write_bytes(Path(tokenizer_path).read_bytes())
    fixtures = a_tree(tmp_path)
    (fixtures / "tiny-chat" / "manifest.toml").write_text(
        f'model = "{checkpoint}"\nrevision = "local"\nencoder = "deepseek_v41"\n'
    )
    manifest = find_manifest(fixtures, str(checkpoint))
    version, read = witness.deepseek_recipe_reader(manifest)
    assert version == recipe.__version__
    tokenizer = recipe.Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    text = "Paris is sunny."
    case = parse_line("a", text)
    case["output_ids"] = tokenizer.encode(text)
    reading = read(case)
    assert reading == {
        "version": version,
        "message": {"role": "assistant", "content": text},
        "finish_reason": "stop",
    }
