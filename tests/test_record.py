import fcntl
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess

import huggingface_hub.constants
import pytest
import zstandard
from conftest import TEMPLATE
from tokenizers import Tokenizer, decoders, normalizers, processors

from bellwether import storage
from bellwether import unpack as unpack_module
from bellwether.cli import main
from bellwether.inputs import oracle_inputs
from bellwether.manifest import find_manifest, load_manifest, slug_for
from bellwether.record import sets as set_tables
from bellwether.record.chunks import chunk_plans
from bellwether.record.corpus import load_corpus, read_cases
from bellwether.record.fixtures import canonical_line, read_fixture_file, schema_path, validator, write_fixture_file
from bellwether.record.reference import HfTemplateOracle
from bellwether.record.roundtrip import RoundtripOracle, as_vllm_gives_it, generation_eos_ids, next_messages
from bellwether.storage import is_lfs_pointer, lfs_pull_command, plain_text

ROOT = pathlib.Path(__file__).resolve().parents[1]


def write_jsonl(path: pathlib.Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


# A Hub id is pinned to a commit. The tiny model is a directory, which has no commits: its revision is "local".
HUB_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def write_manifest(
    fixtures: pathlib.Path,
    slug: str,
    model: str,
    *,
    revision: str | None = None,
    inputs: dict[str, str] | None = None,
    tier: int | str | None = None,
    group: str | None = None,
) -> pathlib.Path:
    """A manifest as a person would write it; ``tier`` may be given as raw TOML text, to write a wrong type."""
    path = fixtures / slug / "manifest.toml"
    path.parent.mkdir(parents=True)
    if revision is None:
        revision = "local" if pathlib.Path(model).is_absolute() else HUB_REVISION
    lines = [f'model = "{model}"', f'revision = "{revision}"']
    lines += [f"tier = {tier}"] if tier is not None else []
    lines += [f'group = "{group}"'] if group is not None else []
    lines += ["", "[authority]", 'render = ["hf-template"]', "", "[smg]", 'tool_parser = "qwen"']
    if inputs is not None:
        lines += ["", "[inputs]", *(f'"{name}" = "{digest}"' for name, digest in inputs.items())]
    path.write_text("\n".join(lines) + "\n")
    return path


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def test_manifest_reads_model_revision_authority_and_names(tmp_path):
    path = write_manifest(tmp_path, "tiny-chat", "acme/Tiny-Chat")
    manifest = load_manifest(path)
    assert (manifest.slug, manifest.model, manifest.revision) == ("tiny-chat", "acme/Tiny-Chat", HUB_REVISION)
    assert manifest.authority == {"render": ["hf-template"]}
    assert manifest.smg == {"tool_parser": "qwen"}


@pytest.mark.parametrize("revision", ["main", "v1.0", "b968826d", HUB_REVISION.upper(), "local"])
def test_manifest_refuses_a_revision_that_is_not_a_commit_hash(tmp_path, revision):
    path = write_manifest(tmp_path, "tiny-chat", "acme/Tiny-Chat", revision=revision)
    with pytest.raises(ValueError, match=re.escape(f"{path}: revision {revision!r} is not a commit")):
        load_manifest(path)


def test_manifest_takes_a_commit_hash_and_local_only_for_a_checkpoint_directory(tmp_path, tiny_model):
    hub = write_manifest(tmp_path / "hub", "tiny-chat", "acme/Tiny-Chat", revision=HUB_REVISION)
    assert load_manifest(hub).revision == HUB_REVISION
    local = write_manifest(tmp_path / "local", "tiny-chat", str(tiny_model), revision="local")
    assert load_manifest(local).revision == "local"


def test_manifest_reads_the_oracle_inputs_group_and_tier(tmp_path):
    inputs = {"tokenizer.json": "a" * 64, "tokenizer_config.json": "b" * 64}
    path = write_manifest(tmp_path, "tiny-chat-mini", "acme/Tiny-Chat-Mini", inputs=inputs, tier=2, group="tiny-chat")
    manifest = load_manifest(path)
    assert (manifest.inputs, manifest.group, manifest.tier) == (inputs, "tiny-chat", 2)


def test_a_manifest_from_before_groups_loads_without_inputs_group_or_tier(tmp_path):
    manifest = load_manifest(write_manifest(tmp_path, "tiny-chat", "acme/Tiny-Chat"))
    assert (manifest.inputs, manifest.group, manifest.tier) == ({}, None, None)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"tier": 4}, "`tier` must be 1, 2 or 3"),
        ({"tier": '"1"'}, "`tier` must be 1, 2 or 3"),
        ({"tier": "true"}, "`tier` must be 1, 2 or 3"),
        ({"group": "Tiny Chat"}, "`group` must be the slug of its group's primary"),
        ({"group": "tiny-chat"}, "`group` names this manifest's own directory; a group's primary has no `group`"),
        ({"inputs": {"tokenizer.json": "abc"}}, "[inputs] must give each file's sha256"),
        ({"inputs": {"tokenizer.json": "A" * 64}}, "[inputs] must give each file's sha256"),
    ],
)
def test_manifest_rejects_a_malformed_tier_group_or_inputs(tmp_path, fields, message):
    path = write_manifest(tmp_path, "tiny-chat", "acme/Tiny-Chat", **fields)
    with pytest.raises(ValueError, match=re.escape(f"{path}: {message}")):
        load_manifest(path)


def test_manifest_rejects_an_unknown_kind_in_the_authority_order(tmp_path):
    path = write_manifest(tmp_path, "tiny-chat", "acme/Tiny-Chat")
    path.write_text(path.read_text().replace("render =", "prompt ="))
    with pytest.raises(ValueError, match="unknown kind `prompt`"):
        load_manifest(path)


def test_find_manifest_matches_the_model_id_exactly(tmp_path):
    write_manifest(tmp_path, "tiny-chat", "acme/Tiny-Chat")
    assert find_manifest(tmp_path, "acme/Tiny-Chat").slug == "tiny-chat"
    with pytest.raises(FileNotFoundError, match="manifests exist for: acme/Tiny-Chat"):
        find_manifest(tmp_path, "acme/tiny-chat")


def test_slug_is_the_lowercase_last_path_segment():
    assert slug_for("Qwen/Qwen3-8B") == "qwen3-8b"
    assert slug_for("deepseek-ai/DeepSeek-V4.1-Flash") == "deepseek-v4.1-flash"
    assert slug_for("meta-models/Muse Glimmer_30B") == "muse-glimmer-30b"


def test_corpus_rejects_names_that_are_not_slugs_and_duplicates(tmp_path):
    path = tmp_path / "set.jsonl"
    write_jsonl(path, [{"name": "Bad Name", "request": {"messages": []}}])
    with pytest.raises(ValueError, match="`name` must be a lowercase slug"):
        read_cases(path)
    write_jsonl(path, [{"name": "a", "request": {"messages": []}}, {"name": "a", "request": {"messages": []}}])
    with pytest.raises(ValueError, match="duplicate case name 'a'"):
        read_cases(path)
    write_jsonl(path, [{"name": "a", "request": []}])
    with pytest.raises(ValueError, match="`request` must be an object"):
        read_cases(path)


def test_corpus_adds_the_model_specific_file_to_the_shared_set(tmp_path):
    write_jsonl(tmp_path / "render" / "common.jsonl", [{"name": "a", "request": {"messages": []}}])
    write_jsonl(tmp_path / "render" / "tiny-chat" / "common.jsonl", [{"name": "b", "request": {"messages": []}}])
    write_jsonl(tmp_path / "render" / "tiny-chat" / "own.jsonl", [{"name": "c", "request": {"messages": []}}])
    sets = load_corpus(tmp_path, "render", "tiny-chat")
    assert {name: [c.name for c in cases] for name, cases in sets.items()} == {"common": ["a", "b"], "own": ["c"]}
    assert load_corpus(tmp_path, "render", "other-model") == {"common": sets["common"][:1]}


def test_corpus_rejects_a_model_file_that_reuses_a_shared_name(tmp_path):
    write_jsonl(tmp_path / "render" / "common.jsonl", [{"name": "a", "request": {"messages": []}}])
    write_jsonl(tmp_path / "render" / "tiny-chat" / "common.jsonl", [{"name": "a", "request": {"messages": []}}])
    with pytest.raises(ValueError, match="case name 'a' is already used in .*common.jsonl"):
        load_corpus(tmp_path, "render", "tiny-chat")


def test_corpus_rejects_the_same_name_in_two_sets(tmp_path):
    write_jsonl(tmp_path / "render" / "common.jsonl", [{"name": "a", "request": {"messages": []}}])
    write_jsonl(tmp_path / "render" / "tools.jsonl", [{"name": "a", "request": {"messages": []}}])
    with pytest.raises(ValueError, match="tools.jsonl: case name 'a' is already used in .*common.jsonl"):
        load_corpus(tmp_path, "render", "tiny-chat")


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any directory")
def test_corpus_fails_when_a_directory_that_exists_cannot_be_read(tmp_path):
    write_jsonl(tmp_path / "render" / "common.jsonl", [{"name": "a", "request": {"messages": []}}])
    own = tmp_path / "render" / "tiny-chat"
    write_jsonl(own / "common.jsonl", [{"name": "b", "request": {"messages": []}}])
    own.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            load_corpus(tmp_path, "render", "tiny-chat")
    finally:
        own.chmod(0o755)


# Line breaks other than "\n" that JSON writes raw when ensure_ascii is off, as the corpus and fixture writers do.
BREAKS = "one two\u0085three"


def test_corpus_reads_a_case_whose_text_holds_unicode_line_breaks_intact(tmp_path):
    path = tmp_path / "set.jsonl"
    case = {"name": "a", "request": {"messages": [user(BREAKS)]}, "message": {"content": BREAKS}}
    path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    [read] = read_cases(path)
    assert read.request == {"messages": [user(BREAKS)]} and read.message == {"content": BREAKS}


def write_compressed(path: pathlib.Path, lines: list[dict]) -> None:
    """A corpus set in the compressed form an import past corpus_sets.LIMIT writes."""
    storage.write(path, "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode("utf-8"))


def test_corpus_reads_a_compressed_set_under_the_sets_name_with_its_lines_intact(tmp_path):
    imported = {"name": "x-0", "request": {"messages": [user(BREAKS)]}, "origin": {"dataset": "x"}}
    write_compressed(tmp_path / "render" / "x-a.jsonl.zst", [imported])
    write_jsonl(tmp_path / "render" / "common.jsonl", [{"name": "a", "request": {"messages": []}}])
    sets = load_corpus(tmp_path, "render", "tiny-chat")
    assert {name: [case.name for case in cases] for name, cases in sets.items()} == {"common": ["a"], "x-a": ["x-0"]}
    assert sets["x-a"][0].request == {"messages": [user(BREAKS)]} and sets["x-a"][0].origin == {"dataset": "x"}


def test_corpus_refuses_a_set_stored_in_both_forms(tmp_path):
    write_jsonl(tmp_path / "render" / "x-a.jsonl", [{"name": "a", "request": {"messages": []}}])
    write_compressed(tmp_path / "render" / "x-a.jsonl.zst", [{"name": "b", "request": {"messages": []}}])
    with pytest.raises(ValueError, match="x-a.jsonl.zst: set x-a is stored in both forms, beside .*x-a.jsonl$"):
        load_corpus(tmp_path, "render", "tiny-chat")


def test_corpus_names_a_set_git_lfs_has_not_fetched_with_the_command_that_fetches_it(tmp_path):
    (tmp_path / "render").mkdir()
    (tmp_path / "render" / "x-a.jsonl.zst").write_text(LFS_POINTER)
    with pytest.raises(ValueError, match="x-a.jsonl.zst is a Git LFS pointer; fetch it first: git lfs pull"):
        load_corpus(tmp_path, "render", "tiny-chat")


def test_reference_oracle_renders_the_checkpoint_template_and_its_ids(tiny_model):
    oracle = HfTemplateOracle(str(tiny_model), "local")
    rendered = oracle.render({"messages": [user("What is the capital of France?")]})
    assert rendered.text == "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    assert rendered.input_ids == oracle.tokenizer.encode(rendered.text, add_special_tokens=False)
    assert rendered.input_ids[0] == oracle.tokenizer.convert_tokens_to_ids("<|im_start|>")
    provenance = oracle.provenance()
    assert provenance["oracle"] == "transformers.apply_chat_template"
    assert len(provenance["chat_template_sha256"]) == 64
    assert provenance["tokenizer_class"] == type(oracle.tokenizer).__name__


def test_reference_oracle_passes_template_kwargs_tools_and_continuation_through(tiny_model):
    oracle = HfTemplateOracle(str(tiny_model), "local")
    off = oracle.render({"messages": [user("Hi")], "chat_template_kwargs": {"enable_thinking": False}})
    assert off.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    with_tools = oracle.render({"messages": [user("Hi")], "tools": tools})
    assert with_tools.text.startswith("<|im_start|>system\n" + json.dumps(tools))
    continued = oracle.render(
        {"messages": [user("Finish."), {"role": "assistant", "content": "The quick"}], "continue_final_message": True}
    )
    assert continued.text.endswith("<|im_start|>assistant\nThe quick")


def record(
    tmp_path, tiny_model, *corpus_sets: tuple[str, list[dict]], kind: str = "render"
) -> tuple[int, pathlib.Path]:
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    if not (fixtures / "tiny-chat").exists():
        write_manifest(fixtures, "tiny-chat", str(tiny_model), inputs=oracle_inputs(str(tiny_model), "local"))
    for name, lines in corpus_sets:
        write_jsonl(corpus / kind / f"{name}.jsonl", lines)
    argv = ["record", "--model", str(tiny_model), "--kind", kind, "--oracle", "reference"]
    argv += ["--fixtures", str(fixtures), "--corpus", str(corpus)]
    return main(argv), fixtures / "tiny-chat" / kind


def test_record_writes_sorted_canonical_lines_with_provenance(tmp_path, tiny_model, capsys):
    cases = [
        {"name": "zulu", "request": {"messages": [user("Z")]}},
        {"name": "alpha", "request": {"messages": [user("A")]}},
    ]
    status, out_dir = record(tmp_path, tiny_model, ("common", cases))
    assert status == 0
    text = (out_dir / "common.jsonl").read_text()
    lines = [json.loads(line) for line in text.removesuffix("\n").split("\n")]
    assert [line["id"] for line in lines] == ["tiny-chat/render/alpha", "tiny-chat/render/zulu"]
    first = lines[0]
    assert first["kind"] == "render" and first["model"] == str(tiny_model)
    assert first["reference"]["source"] == "hf-template"
    assert first["reference"]["text"] == "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\n"
    assert first["reference"]["provenance"]["revision"] == "local"
    assert "transformers" in first["reference"]["provenance"]
    for line in lines:
        validator().validate(line)
    assert text == "".join(f"{canonical_line(line)}\n" for line in lines)
    assert list(lines[0]) == ["id", "kind", "model", "request", "reference"]
    assert "common.jsonl: 2 cases recorded\n" in capsys.readouterr().out


def test_record_rerun_replaces_the_reference_and_keeps_witnesses(tmp_path, tiny_model):
    status, out_dir = record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    assert status == 0
    path = out_dir / "common.jsonl"
    recorded = read_fixture_file(path)
    recorded["tiny-chat/render/a"]["witnesses"] = {"vllm": {"version": "0.30.0", "input_ids": [1]}}
    recorded["tiny-chat/render/a"]["reference"]["text"] = "stale"
    write_fixture_file(path, recorded)
    status, _ = record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    assert status == 0
    line = read_fixture_file(path)["tiny-chat/render/a"]
    assert line["witnesses"] == {"vllm": {"version": "0.30.0", "input_ids": [1]}}
    assert line["reference"]["text"] != "stale"


def test_record_rebuilds_the_file_from_the_cases_rendered_this_run(tmp_path, tiny_model, capsys):
    first = [{"name": "a", "request": {"messages": [user("A")]}}, {"name": "b", "request": {"messages": [user("B")]}}]
    status, out_dir = record(tmp_path, tiny_model, ("common", first))
    assert status == 0
    assert list(read_fixture_file(out_dir / "common.jsonl")) == ["tiny-chat/render/a", "tiny-chat/render/b"]
    # "a" leaves the corpus and "b" can no longer be rendered: neither may stay in the file.
    second = [{"name": "b", "request": {"prompt": "no messages"}}, {"name": "c", "request": {"messages": [user("C")]}}]
    status, _ = record(tmp_path, tiny_model, ("common", second))
    assert status == 1
    assert list(read_fixture_file(out_dir / "common.jsonl")) == ["tiny-chat/render/c"]
    captured = capsys.readouterr()
    assert "1 cases recorded, 2 old cases removed" in captured.out
    assert "not recorded tiny-chat/render/b: KeyError" in captured.err
    # Nothing renders: the file goes away rather than keeping stale lines.
    status, _ = record(tmp_path, tiny_model, ("common", [{"name": "c", "request": {"prompt": "x"}}]))
    assert status == 1
    assert not (out_dir / "common.jsonl").exists()


def test_record_removes_fixture_files_of_sets_the_corpus_no_longer_has(tmp_path, tiny_model, capsys):
    common = [{"name": "a", "request": {"messages": [user("A")]}}]
    tools = [{"name": "t", "request": {"messages": [user("T")]}}]
    status, out_dir = record(tmp_path, tiny_model, ("common", common), ("tools", tools))
    assert status == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["common.jsonl", "tools.jsonl"]
    (tmp_path / "corpus" / "render" / "tools.jsonl").unlink()
    status, _ = record(tmp_path, tiny_model, ("common", common))
    assert status == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["common.jsonl"]
    assert "tools.jsonl: removed, the corpus has no set of that name" in capsys.readouterr().out


def test_record_keeps_the_request_key_order_the_oracle_rendered(tmp_path, tiny_model):
    tool = {"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {}}}
    cases = [{"name": "tools", "request": {"messages": [user("Hi")], "tools": [tool]}}]
    status, out_dir = record(tmp_path, tiny_model, ("common", cases))
    assert status == 0
    raw = (out_dir / "common.jsonl").read_text()
    # The request keeps the corpus order (type before function, name before description) ...
    assert (
        '"tools":[{"type":"function","function":{"name":"get_weather","description":"Weather","parameters":{}}}]' in raw
    )
    # ... and the reference shows the template rendered that order, so the two agree.
    rendered = json.loads(raw)["reference"]["text"]
    assert '{"type": "function", "function": {"name": "get_weather", "description": "Weather"' in rendered
    assert raw.startswith('{"id":"tiny-chat/render/tools","kind":"render","model":')


def test_record_drops_witnesses_when_the_request_changed(tmp_path, tiny_model, capsys):
    status, out_dir = record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    assert status == 0
    path = out_dir / "common.jsonl"
    recorded = read_fixture_file(path)
    recorded["tiny-chat/render/a"]["witnesses"] = {"vllm": {"version": "0.30.0", "input_ids": [1]}}
    write_fixture_file(path, recorded)
    status, _ = record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A!")]}}]))
    assert status == 0
    assert "witnesses" not in read_fixture_file(path)["tiny-chat/render/a"]
    assert "1 witnesses dropped because the request changed" in capsys.readouterr().out


def test_record_reports_cases_the_template_cannot_render_and_records_the_rest(tmp_path, tiny_model, capsys):
    cases = [
        {"name": "ok", "request": {"messages": [user("A")]}},
        {"name": "broken", "request": {"prompt": "no messages"}},
    ]
    status, out_dir = record(tmp_path, tiny_model, ("common", cases))
    assert status == 1
    assert list(read_fixture_file(out_dir / "common.jsonl")) == ["tiny-chat/render/ok"]
    assert "not recorded tiny-chat/render/broken: KeyError" in capsys.readouterr().err


WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}


def weather_call() -> dict:
    return {"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}


def test_chunk_plans_cover_the_output_and_are_deterministic():
    plans = chunk_plans(10)
    assert plans["whole"] is None and plans["per_token"] is None
    assert plans["size-3"] == [3, 3, 3, 1]
    assert "size-1" not in plans and "size-11" not in plans
    assert plans["split-4"] == [4, 6] and sum(1 for name in plans if name.startswith("split-")) == 9
    assert sum(1 for name in plans if name.startswith("random-")) == 30
    assert all(sum(lengths) == 10 for lengths in plans.values() if lengths is not None)
    assert all(1 <= length <= 8 for name, lengths in plans.items() if name.startswith("random-") for length in lengths)
    assert chunk_plans(10) == plans
    assert "split-1" not in chunk_plans(33)
    with pytest.raises(ValueError, match="at least one token"):
        chunk_plans(0)


def test_roundtrip_records_the_turn_between_the_prompt_and_the_end_of_turn(tiny_model):
    oracle = RoundtripOracle(str(tiny_model), "local")
    request = {"messages": [user("Weather?")], "tools": [WEATHER_TOOL]}
    message = {"reasoning_content": "think", "content": "Sure.", "tool_calls": [weather_call()]}
    out = oracle.render_output(request, message)
    call_json = json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
    assert out.text == f"<think>\nthink\n</think>\n\nSure.\n<tool_call>\n{call_json}\n</tool_call>"
    assert out.output_ids == oracle.tokenizer.encode(out.text, add_special_tokens=False)
    assert out.finish_reason == "tool_calls"
    plain = oracle.render_output({"messages": [user("Hi")]}, {"content": "Hello"})
    assert (plain.text, plain.finish_reason) == ("Hello", "stop")


def tiny_variant(
    tiny_model, tmp_path_factory, name: str, template: str, *, tokens: tuple[str, ...] = (), without_eos: bool = False
) -> pathlib.Path:
    """The tiny model's tokenizer with another chat template, optionally more special tokens or no eos token."""
    directory = tmp_path_factory.mktemp(name)
    tokenizer = Tokenizer.from_file(str(tiny_model / "tokenizer.json"))
    tokenizer.add_special_tokens(list(tokens))
    tokenizer.save(str(directory / "tokenizer.json"))
    config = json.loads((tiny_model / "tokenizer_config.json").read_text())
    if without_eos:
        del config["eos_token"]
    (directory / "tokenizer_config.json").write_text(json.dumps({**config, "chat_template": template}))
    return directory


def assistant_template(call: str) -> str:
    return (
        "{%- for m in messages %}"
        "{%- if m['role'] == 'assistant' %}{{ '<|im_start|>assistant\\n' + (m['content'] or '') }}"
        "{%- for c in (m['tool_calls'] or []) %}" + call + "{%- endfor %}{{ '<|im_end|>\\n' }}"
        "{%- else %}{{ '<|im_start|>' + m['role'] + '\\n' + (m['content'] or '') + '<|im_end|>\\n' }}{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
    )


@pytest.fixture(scope="session")
def items_model(tiny_model, tmp_path_factory) -> pathlib.Path:
    """A template that iterates a call's arguments, as most current templates do."""
    call = (
        "{{ '<tool_call>' + c['function']['name'] }}"
        "{%- for k, v in c['function']['arguments'].items() %}{{ ' ' + k + '=' + v | string }}{%- endfor %}"
        "{{ '</tool_call>' }}"
    )
    return tiny_variant(tiny_model, tmp_path_factory, "items-chat", assistant_template(call))


@pytest.fixture(scope="session")
def concat_model(tiny_model, tmp_path_factory) -> pathlib.Path:
    """A template that concatenates a call's arguments as a string, as DeepSeek's do."""
    call = "{{ '<tool_call>' + c['function']['name'] + ' ' + c['function']['arguments'] + '</tool_call>' }}"
    return tiny_variant(tiny_model, tmp_path_factory, "concat-chat", assistant_template(call))


def test_roundtrip_gives_the_template_the_arguments_as_an_object_as_vllm_does(items_model):
    out = RoundtripOracle(str(items_model), "local").render_output(
        {"messages": [user("Weather?")]}, {"content": "", "tool_calls": [weather_call()]}
    )
    assert out.text == "<tool_call>get_weather city=Paris</tool_call>"


def test_a_template_that_cannot_take_object_arguments_fails_the_case(concat_model):
    with pytest.raises(TypeError, match="concatenate"):
        RoundtripOracle(str(concat_model), "local").render_output(
            {"messages": [user("Weather?")]}, {"content": "", "tool_calls": [weather_call()]}
        )


def test_roundtrip_reports_arguments_that_are_not_a_json_object(tiny_model):
    # A rule of the corpus, not an engine's: a parse case's call carries the JSON object string a parser returns.
    oracle = RoundtripOracle(str(tiny_model), "local")
    for arguments in ('["a"]', "not json", "", None):
        call = {"type": "function", "function": {"name": "f", "arguments": arguments}}
        with pytest.raises(ValueError, match="a JSON object"):
            oracle.render_output({"messages": [user("Go")]}, {"content": "", "tool_calls": [call]})


@pytest.mark.parametrize(
    ("arguments", "given"),
    [
        ("", {}),
        (None, {}),
        ("null", {}),
        ([], {}),
        ("{}", {}),
        ('{"a": 1}', {"a": 1}),
        ('["a"]', {}),
        ('"x"', {}),
        ("2", {}),
        ("{a", {}),
        ({"a": 1}, {"a": 1}),
        (["a"], {}),
    ],
)
def test_a_history_call_reaches_the_template_with_the_arguments_vllm_gives_it(arguments, given):
    call = {"type": "function", "function": {"name": "f", "arguments": arguments}}
    message = {"role": "assistant", "content": "", "tool_calls": [call]}
    assert as_vllm_gives_it(message)["tool_calls"][0]["function"]["arguments"] == given
    assert call["function"]["arguments"] == arguments


def test_a_history_call_without_arguments_reaches_the_template_with_an_empty_object():
    message = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "f"}}]}
    assert as_vllm_gives_it(message)["tool_calls"][0]["function"]["arguments"] == {}


def test_an_empty_tool_calls_list_is_dropped_as_vllm_drops_it():
    message = {"role": "assistant", "content": "Hi", "tool_calls": []}
    assert as_vllm_gives_it(message) == {"role": "assistant", "content": "Hi"}
    assert message["tool_calls"] == []


def test_a_call_with_empty_arguments_in_the_history_renders_as_vllm_renders_it(items_model):
    empty = {"type": "function", "function": {"name": "get_time", "arguments": ""}}
    request = {
        "messages": [
            user("Time?"),
            {"role": "assistant", "content": "", "tool_calls": [empty]},
            {"role": "tool", "content": "noon"},
            user("And the weather in Paris?"),
        ]
    }
    out = RoundtripOracle(str(items_model), "local").render_output(
        request, {"content": "", "tool_calls": [weather_call()]}
    )
    assert out.text == "<tool_call>get_weather city=Paris</tool_call>"


def test_a_template_that_renders_no_tool_calls_fails_the_case(tiny_model, tmp_path_factory):
    # Phi-4-mini's and Hunyuan-A13B's templates drop the calls; the output would be the end of the turn alone.
    model = tiny_variant(tiny_model, tmp_path_factory, "no-calls-chat", assistant_template(""))
    with pytest.raises(
        ValueError,
        match=re.escape("does not render every tool call: renaming call 0 (get_weather) leaves"),
    ):
        RoundtripOracle(str(model), "local").render_output(
            {"messages": [user("Weather?")]}, {"content": "", "tool_calls": [weather_call()]}
        )


def test_a_template_that_renders_only_the_first_call_fails_the_case(tiny_model, tmp_path_factory):
    call = "{%- if loop.first %}{{ '<tool_call>' + c['function']['name'] + '</tool_call>' }}{%- endif %}"
    model = tiny_variant(tiny_model, tmp_path_factory, "first-call-chat", assistant_template(call))
    with pytest.raises(
        ValueError,
        match=re.escape("does not render every tool call: adding an argument to call 0 (get_weather) leaves"),
    ):
        RoundtripOracle(str(model), "local").render_output(
            {"messages": [user("Weather?")]}, {"content": "", "tool_calls": [weather_call(), weather_call()]}
        )


def test_a_template_that_renders_only_the_last_call_fails_even_when_the_names_still_add_up(
    tiny_model, tmp_path_factory
):
    # bfcl-live-parallel-8-4-0's shape: the second call's argument holds the first call's name, so counting names
    # in the output finds two even when the first call is never rendered.
    call = (
        "{%- if loop.last %}{{ '<tool_call>' + c['function']['name'] + ' ' + c['function']['arguments'] | tojson"
        " + '</tool_call>' }}{%- endif %}"
    )
    model = tiny_variant(tiny_model, tmp_path_factory, "last-call-chat", assistant_template(call))
    calls = [
        {"type": "function", "function": {"name": "todo", "arguments": '{"type": "add"}'}},
        {"type": "function", "function": {"name": "todo", "arguments": '{"content": "todo random"}'}},
    ]
    with pytest.raises(
        ValueError,
        match=re.escape("does not render every tool call: renaming call 0 (todo) leaves"),
    ):
        RoundtripOracle(str(model), "local").render_output(
            {"messages": [user("Two todos")]}, {"content": "", "tool_calls": calls}
        )


def test_a_template_that_renders_the_names_without_the_arguments_fails_the_case(tiny_model, tmp_path_factory):
    # The rename changes the rendered turn, so only the marker argument finds that the arguments never reach it.
    call = "{{ '<tool_call>' + c['function']['name'] + '</tool_call>' }}"
    model = tiny_variant(tiny_model, tmp_path_factory, "names-only-chat", assistant_template(call))
    with pytest.raises(
        ValueError,
        match=re.escape("does not render every tool call: adding an argument to call 0 (get_weather) leaves"),
    ):
        RoundtripOracle(str(model), "local").render_output(
            {"messages": [user("Weather?")]}, {"content": "", "tool_calls": [weather_call()]}
        )


def test_a_template_that_renders_the_arguments_without_the_names_fails_the_case(tiny_model, tmp_path_factory):
    # The marker argument changes the rendered turn, so only the rename finds that the name never reaches it.
    call = "{{ '<tool_call>' + c['function']['arguments'] | tojson + '</tool_call>' }}"
    model = tiny_variant(tiny_model, tmp_path_factory, "arguments-only-chat", assistant_template(call))
    with pytest.raises(
        ValueError,
        match=re.escape("does not render every tool call: renaming call 0 (get_weather) leaves"),
    ):
        RoundtripOracle(str(model), "local").render_output(
            {"messages": [user("Weather?")]}, {"content": "", "tool_calls": [weather_call()]}
        )


def test_a_template_that_raises_on_an_unknown_tool_name_still_records_the_case(tiny_model, tmp_path_factory):
    # Some templates look the call's tool up in `tools` and raise when there is none. Failing on the renamed call
    # shows the template read the name; the marker argument still changes the rendered turn, so the case is recorded.
    call = (
        "{%- if c['function']['name'] not in tools | map(attribute='function.name') | list %}"
        "{{ raise_exception('no tool named ' + c['function']['name']) }}{%- endif %}"
        "{{ '<tool_call>' + c['function']['name'] }}"
        "{%- for k, v in c['function']['arguments'].items() %}{{ ' ' + k + '=' + v | string }}{%- endfor %}"
        "{{ '</tool_call>' }}"
    )
    model = tiny_variant(tiny_model, tmp_path_factory, "tool-lookup-chat", assistant_template(call))
    out = RoundtripOracle(str(model), "local").render_output(
        {"messages": [user("Weather?")], "tools": [WEATHER_TOOL]}, {"content": "", "tool_calls": [weather_call()]}
    )
    assert out.text == "<tool_call>get_weather city=Paris</tool_call>"


# Qwen3.5 to 3.8's shape: each argument is a tag, a string written as it is and any other value as JSON, so a null
# and the string "null" give the same text.
TAGGED_CALL = (
    "{{ '<tool_call>\\n<function=' + c['function']['name'] + '>\\n' }}"
    "{%- for k, v in c['function']['arguments'].items() %}"
    "{{ '<parameter=' + k + '>\\n' + (v if v is string else v | tojson) + '\\n</parameter>\\n' }}"
    "{%- endfor %}{{ '</function>\\n</tool_call>' }}"
)


@pytest.fixture(scope="session")
def tagged_model(tiny_model, tmp_path_factory) -> pathlib.Path:
    """A template that writes each argument as a tag, as Qwen3.5 to 3.8 do."""
    return tiny_variant(tiny_model, tmp_path_factory, "tagged-chat", assistant_template(TAGGED_CALL))


def call_weather(model: pathlib.Path, properties: dict | None, arguments: dict):
    """The round trip of one get_weather call; the request's tool declares ``properties``, or there is no tool."""
    request = {"messages": [user("Weather?")]}
    if properties is not None:
        parameters = {"type": "object", "properties": properties}
        request["tools"] = [{"type": "function", "function": {"name": "get_weather", "parameters": parameters}}]
    call = {"type": "function", "function": {"name": "get_weather", "arguments": json.dumps(arguments)}}
    return RoundtripOracle(str(model), "local").render_output(request, {"content": "", "tool_calls": [call]})


STRING_UNIT = {"city": {"type": "string"}, "unit": {"type": "string"}}


def test_a_null_under_a_declared_string_fails_the_case_under_a_tagged_template(tagged_model):
    # vLLM keeps the text `null` as the string "null" for a parameter declared a string, and smg does the same.
    reason = (
        "the template renders get_weather's unit null as it renders the string 'null', and the tool declares unit a "
        "string, so no reader gets null back from the output"
    )
    with pytest.raises(ValueError, match=re.escape(reason)):
        call_weather(tagged_model, STRING_UNIT, {"city": "Paris", "unit": None})


def test_a_null_under_a_declared_string_is_recorded_under_a_template_that_writes_json(tiny_model):
    out = call_weather(tiny_model, STRING_UNIT, {"city": "Paris", "unit": None})
    assert '"arguments": {"city": "Paris", "unit": null}' in out.text


@pytest.mark.parametrize(
    ("declared", "value"),
    [({"type": "boolean"}, False), ({"type": "number"}, 2), ({"type": ["string", "null"]}, None)],
    ids=["boolean", "int-as-number", "type-list"],
)
def test_a_value_of_its_declared_type_is_not_checked(tagged_model, declared, value):
    out = call_weather(tagged_model, {"unit": declared}, {"unit": value})
    assert f"<parameter=unit>\n{json.dumps(value)}\n</parameter>" in out.text


@pytest.mark.parametrize(
    ("declared", "value", "reading"),
    [
        ({"type": ["string", "null"]}, "null", "null"),
        ({"type": ["string", "integer"]}, "5", "5"),
        ({"type": ["string", "boolean"]}, "true", "true"),
    ],
    ids=["null", "integer", "boolean"],
)
def test_a_string_that_reads_as_another_type_its_union_admits_fails_the_case(tagged_model, declared, value, reading):
    # A tagged template writes a string as it is and other values as JSON, so the string "null" and null are the same
    # text; vLLM's parser tries every other declared type before a string, so a reader hands back the other value.
    reason = f"vLLM's parser tries {reading} before a string, so it hands back {reading}"
    with pytest.raises(ValueError, match=re.escape(reason)):
        call_weather(tagged_model, {"unit": declared}, {"unit": value})


def test_the_text_none_under_a_nullable_string_is_recorded(tagged_model):
    # vLLM reads only the text null as null, so the string "None" comes back as the string.
    out = call_weather(tagged_model, {"unit": {"type": ["string", "null"]}}, {"unit": "None"})
    assert "<parameter=unit>\nNone\n</parameter>" in out.text


def test_a_non_string_under_a_union_with_a_string_is_recorded(tagged_model):
    # The other way round a reader gets the value back: it tries the integer before the string.
    out = call_weather(tagged_model, {"unit": {"type": ["string", "integer"]}}, {"unit": 5})
    assert "<parameter=unit>\n5\n</parameter>" in out.text


@pytest.mark.parametrize(("model", "text"), [("tagged_model", "false"), ("items_model", "False")])
def test_a_boolean_under_a_declared_string_fails_the_case(request, model, text):
    # Qwen3.5 to 3.8 write a boolean as JSON, `false`; Qwen3-Coder writes it with Jinja's `string`, `False`. Either
    # text is also the text of a string.
    reason = (
        f"the template renders get_weather's unit false as it renders the string '{text}', and the tool declares unit "
        "a string, so no reader gets false back from the output"
    )
    with pytest.raises(ValueError, match=re.escape(reason)):
        call_weather(request.getfixturevalue(model), STRING_UNIT, {"city": "Paris", "unit": False})


def test_an_array_holding_non_ascii_text_under_a_declared_string_fails_the_case(tagged_model):
    # transformers' tojson writes non-ASCII text as it is, so the JSON candidate must leave it unescaped too.
    reason = (
        'the template renders get_weather\'s unit ["é"] as it renders the string \'["é"]\', and the tool declares '
        'unit a string, so no reader gets ["é"] back from the output'
    )
    with pytest.raises(ValueError, match=re.escape(reason)):
        call_weather(tagged_model, STRING_UNIT, {"city": "Paris", "unit": ["é"]})


def test_a_string_that_reads_as_its_declared_type_fails_the_case(tagged_model):
    reason = (
        "the template renders get_weather's days the string '2' as it renders 2, and the tool declares days an "
        "integer, so no reader gets the string '2' back from the output"
    )
    with pytest.raises(ValueError, match=re.escape(reason)):
        call_weather(tagged_model, {"days": {"type": "integer"}}, {"days": "2"})


@pytest.mark.parametrize("declared", [{"type": "integer"}, {"type": ["string", "integer"]}], ids=["integer", "list"])
def test_a_null_written_as_null_under_a_declared_type_other_than_string_is_recorded(tagged_model, declared):
    # vLLM and smg read the text `null` as null under any declared type but a string alone.
    out = call_weather(tagged_model, {"days": declared}, {"days": None})
    assert "<parameter=days>\nnull\n</parameter>" in out.text


@pytest.mark.parametrize(
    ("declared", "kinds"),
    [({"type": "integer"}, "an integer"), ({"type": ["string", "null"]}, "a string or null")],
    ids=["integer", "optional-string"],
)
def test_a_null_written_as_none_fails_the_case_under_any_declared_type(items_model, declared, kinds):
    # Qwen3-Coder writes a null with Jinja's `string`, `None`. vLLM's parser (vllm/tool_parsers/utils.py,
    # coerce_to_schema_type) reads only the text `null` as null, so `None` comes back as the string "None".
    reason = (
        "the template renders get_weather's days null as it renders the string 'None', and vLLM reads the text None "
        f"as the string 'None' under {kinds}, so it does not get null back from the output"
    )
    with pytest.raises(ValueError, match=re.escape(reason)):
        call_weather(items_model, {"days": declared}, {"days": None})


def test_an_undeclared_number_fails_the_case(tagged_model):
    # Without a declared type the readers differ (smg infers JSON, vLLM keeps a string), so the text must say which.
    reason = (
        "the template renders get_weather's days 3 as it renders the string '3', and the tools declare no type for "
        "days, so no reader can tell the two apart in the output"
    )
    with pytest.raises(ValueError, match=re.escape(reason)):
        call_weather(tagged_model, {"city": {"type": "string"}}, {"city": "Paris", "days": 3})


@pytest.mark.parametrize(
    ("model", "value", "candidate"),
    [("tagged_model", "true", "true"), ("items_model", "None", "null"), ("items_model", "False", "false")],
)
def test_an_undeclared_string_that_reads_as_another_type_fails_the_case(request, model, value, candidate):
    # Where no type is declared, smg reads JSON, and Python's True, False and None.
    reason = (
        f"the template renders get_weather's unit the string '{value}' as it renders {candidate}, and the tools "
        "declare no type for unit, so no reader can tell the two apart in the output"
    )
    with pytest.raises(ValueError, match=re.escape(reason)):
        call_weather(request.getfixturevalue(model), None, {"unit": value})


def test_an_undeclared_string_that_is_not_json_is_recorded(tagged_model):
    out = call_weather(tagged_model, None, {"city": "Paris"})
    assert "<parameter=city>\nParis\n</parameter>" in out.text


def test_a_template_that_fails_on_the_candidate_records_the_case(tiny_model, tmp_path_factory):
    # This template takes no string value: the candidates, the strings 'null' and 'None', fail to render, so neither
    # reads as the null.
    call = (
        "{{ '<tool_call>\\n<function=' + c['function']['name'] + '>\\n' }}"
        "{%- for k, v in c['function']['arguments'].items() %}"
        "{%- if v is string %}{{ raise_exception('no string arguments') }}{%- endif %}"
        "{{ '<parameter=' + k + '>\\n' + v | tojson + '\\n</parameter>\\n' }}"
        "{%- endfor %}{{ '</function>\\n</tool_call>' }}"
    )
    model = tiny_variant(tiny_model, tmp_path_factory, "no-strings-chat", assistant_template(call))
    out = call_weather(model, {"unit": {"type": "string"}}, {"unit": None})
    assert "<parameter=unit>\nnull\n</parameter>" in out.text


def test_roundtrip_records_the_text_each_output_token_contributes(tiny_model):
    oracle = RoundtripOracle(str(tiny_model), "local")
    out = oracle.render_output({"messages": [user("Hi")]}, {"reasoning_content": "r", "content": "Café 🌍"})
    # The tiny tokenizer has no merge for a byte outside ASCII, so é is two byte tokens and the globe
    # four; a token that does not complete a character contributes nothing, the one that does
    # carries the whole character. A marker token contributes its marker.
    assert out.output_pieces[0] == "<think>"
    assert out.output_pieces[-7:] == ["", "é", " ", "", "", "", "🌍"]
    assert len(out.output_pieces) == len(out.output_ids)
    assert "".join(out.output_pieces) == "<think>\nr\n</think>\n\nCafé 🌍"


# BENGALI LETTER YYA as one code point, as seven of MGSM's Bengali exemplars write it (#57). It is a
# composition exclusion: NFC writes it as YA and NUKTA, and never puts the two back together.
YYA = chr(0x09DF)
YA_NUKTA = chr(0x09AF) + chr(0x09BC)
# LOWER ONE EIGHTH BLOCK, which SentencePiece's normalizer writes for a space and its decoder reads back as one.
SPACE_MARK = chr(0x2581)


def normalizing_variant(tiny_model, tmp_path_factory, name: str, normalizer, decoder=None) -> pathlib.Path:
    """The tiny model whose tokenizer has ``normalizer`` (and ``decoder``), saved the way a checkpoint ships one."""
    directory = tmp_path_factory.mktemp(name)
    tokenizer = Tokenizer.from_file(str(tiny_model / "tokenizer.json"))
    tokenizer.normalizer = normalizer
    if decoder is not None:
        tokenizer.decoder = decoder
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_bytes((tiny_model / "tokenizer_config.json").read_bytes())
    return directory


@pytest.fixture(scope="session")
def nfc_model(tiny_model, tmp_path_factory) -> pathlib.Path:
    """The tiny model with Qwen3-8B's normalizer, NFC."""
    return normalizing_variant(tiny_model, tmp_path_factory, "nfc-chat", normalizers.NFC())


@pytest.fixture(scope="session")
def spaces_model(tiny_model, tmp_path_factory) -> pathlib.Path:
    """NFC, then every space written as U+2581, which the decoder reads back as a space."""
    return normalizing_variant(
        tiny_model,
        tmp_path_factory,
        "nfc-spaces-chat",
        normalizers.Sequence([normalizers.NFC(), normalizers.Replace(" ", SPACE_MARK)]),
        decoders.Sequence([decoders.ByteLevel(), decoders.Replace(SPACE_MARK, " ")]),
    )


def encode_with(model: pathlib.Path, normalizer, text: str) -> list[int]:
    """``text``'s ids under ``model``'s tokenizer with ``normalizer`` in place of its own."""
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    tokenizer.normalizer = normalizer
    return tokenizer.encode(text, add_special_tokens=False).ids


def test_roundtrip_builds_the_ids_of_text_nfc_would_change_from_the_text_as_written(nfc_model):
    oracle = RoundtripOracle(str(nfc_model), "local")
    text = f"Paris {YYA}"
    assert oracle.tokenizer.decode(oracle.tokenizer.encode(text, add_special_tokens=False)) == f"Paris {YA_NUKTA}"
    out = oracle.render_output({"messages": [user("Hi")]}, {"content": text})
    # The output is the turn up to its stop id, as written, and the ids are those of that text.
    assert out.text == text
    assert out.end_of_turn == {"stop_id": token_id(nfc_model, "<|im_end|>"), "found_by": "turn"}
    assert out.output_ids == encode_with(nfc_model, None, text)
    assert oracle.tokenizer.decode(out.output_ids) == text
    # The pieces come from the ids recorded: YYA's three bytes give two empty pieces, then the letter.
    assert out.output_pieces[-3:] == ["", "", YYA]
    assert len(out.output_pieces) == len(out.output_ids)
    assert "".join(out.output_pieces) == text
    # The ids came from a copy: the oracle's own tokenizer still normalizes.
    assert isinstance(oracle.tokenizer.backend_tokenizer.normalizer, normalizers.NFC)


def test_roundtrip_leaves_out_only_the_unicode_normalization_of_a_normalizer_sequence(spaces_model):
    oracle = RoundtripOracle(str(spaces_model), "local")
    text = f"Paris {YYA} fox"
    out = oracle.render_output({"messages": [user("Hi")]}, {"content": text})
    assert out.text == text
    assert oracle.tokenizer.decode(out.output_ids) == text
    # The Replace step stays: the ids write each space as U+2581, the way this tokenizer writes one.
    assert out.output_ids == encode_with(spaces_model, normalizers.Replace(" ", SPACE_MARK), text)
    assert out.output_ids != encode_with(spaces_model, None, text)
    assert len(out.output_pieces) == len(out.output_ids)
    assert "".join(out.output_pieces) == text


@pytest.mark.parametrize("model", ["nfc_model", "spaces_model"])
def test_roundtrip_keeps_the_tokenizers_own_ids_when_they_decode_back_to_the_text(model, request):
    # YYA as NFC writes it: the normalizer leaves its letters alone; the spaces model still rewrites the spaces.
    path = request.getfixturevalue(model)
    oracle = RoundtripOracle(str(path), "local")
    text = f"Paris {YA_NUKTA} fox"
    out = oracle.render_output({"messages": [user("Hi")]}, {"content": text})
    assert out.output_ids == oracle.tokenizer.encode(text, add_special_tokens=False)
    assert "".join(out.output_pieces) == text


def test_roundtrip_rejects_text_whose_ids_lose_it_with_and_without_the_unicode_normalization(spaces_model):
    # The decoder reads every U+2581 back as a space, so text that holds one cannot come back either way.
    oracle = RoundtripOracle(str(spaces_model), "local")
    with pytest.raises(ValueError, match="^the output text does not survive a tokenize-detokenize round trip$"):
        oracle.render_output({"messages": [user("Hi")]}, {"content": f"Paris {YYA}{SPACE_MARK}"})


def test_roundtrip_encodes_once_for_a_tokenizer_without_unicode_normalization(tiny_model, tmp_path_factory):
    # Spaces written as U+2581 and read back, and no Unicode normalization form: there is nothing to leave out,
    # so a text whose ids lose it is rejected as before, and no copy of the tokenizer encodes it a second time.
    model = normalizing_variant(
        tiny_model,
        tmp_path_factory,
        "spaces-only-chat",
        normalizers.Replace(" ", SPACE_MARK),
        decoders.Sequence([decoders.ByteLevel(), decoders.Replace(SPACE_MARK, " ")]),
    )
    oracle = RoundtripOracle(str(model), "local")
    with pytest.raises(ValueError, match="^the output text does not survive a tokenize-detokenize round trip$"):
        oracle.render_output({"messages": [user("Hi")]}, {"content": f"Paris{SPACE_MARK}"})
    assert oracle.tokenizer_without_unicode_normalization is None


def test_roundtrip_reports_a_turn_the_template_cannot_extend(tiny_model):
    oracle = RoundtripOracle(str(tiny_model), "local")
    request = {"messages": [user("Hi")], "chat_template_kwargs": {"enable_thinking": False}}
    with pytest.raises(ValueError, match="does not extend the generation prompt"):
        oracle.render_output(request, {"reasoning_content": "r", "content": "c"})


def test_roundtrip_rejects_a_request_that_does_not_end_at_the_generation_prompt(tiny_model):
    oracle = RoundtripOracle(str(tiny_model), "local")
    for request in (
        {"messages": [user("Hi")], "add_generation_prompt": False},
        {"messages": [user("Hi"), {"role": "assistant", "content": "The"}], "continue_final_message": True},
    ):
        with pytest.raises(ValueError, match="must end at the generation prompt"):
            oracle.render_output(request, {"content": "Hello"})


def test_record_parse_writes_the_output_its_chunk_plans_and_the_message(tmp_path, tiny_model, capsys):
    cases = [
        {
            "name": "call",
            "request": {"messages": [user("Weather?")], "tools": [WEATHER_TOOL]},
            "message": {"content": "", "tool_calls": [weather_call()]},
        },
        {
            "name": "lossy",
            "request": {"messages": [user("Hi")], "chat_template_kwargs": {"enable_thinking": False}},
            "message": {"reasoning_content": "r", "content": "c"},
        },
    ]
    status, out_dir = record(tmp_path, tiny_model, ("common", cases), kind="parse")
    assert status == 1
    lines = read_fixture_file(out_dir / "common.jsonl")
    assert list(lines) == ["tiny-chat/parse/call"]
    line = lines["tiny-chat/parse/call"]
    assert line["kind"] == "parse" and line["tools"] == [WEATHER_TOOL] and line["malformed"] is False
    assert line["request"] == cases[0]["request"]
    assert line["output_ids"] and len(line["output_ids"]) == sum(line["chunk_plans"]["size-2"])
    assert line["chunk_plans"]["whole"] is None
    reference = line["reference"]
    assert len(line["output_pieces"]) == len(line["output_ids"])
    assert "".join(line["output_pieces"]) == reference["text"]
    assert reference["source"] == "roundtrip"
    assert reference["message"] == {"role": "assistant", "content": "", "tool_calls": [weather_call()]}
    assert reference["finish_reason"] == "tool_calls"
    assert reference["text"].startswith("\n<tool_call>\n")
    assert reference["provenance"]["revision"] == "local"
    assert "not recorded tiny-chat/parse/lossy: ValueError: the template does not extend" in capsys.readouterr().err


def test_record_parse_reports_an_output_whose_tokens_do_not_give_back_its_text(tmp_path, tiny_model, capsys):
    # The incremental decode holds back text that ends in U+FFFD, waiting for the bytes that would
    # complete a character; at the end of an output none come, so the pieces fall short of the text.
    cases = [
        {"name": "held", "request": {"messages": [user("Hi")]}, "message": {"content": "odd \ufffd"}},
        {"name": "plain", "request": {"messages": [user("Hi")]}, "message": {"content": "Hello"}},
    ]
    status, out_dir = record(tmp_path, tiny_model, ("common", cases), kind="parse")
    assert status == 1
    assert list(read_fixture_file(out_dir / "common.jsonl")) == ["tiny-chat/parse/plain"]
    err = capsys.readouterr().err
    assert "not recorded tiny-chat/parse/held: ValueError: the output's tokens do not give back its text" in err


def test_record_parse_counts_the_cases_whose_ids_leave_out_the_unicode_normalization(tmp_path, nfc_model, capsys):
    cases = [
        {"name": "as-written", "request": {"messages": [user("Hi")]}, "message": {"content": f"Paris {YYA}"}},
        {"name": "plain", "request": {"messages": [user("Hi")]}, "message": {"content": "Paris"}},
    ]
    status, out_dir = record(tmp_path, nfc_model, ("common", cases), kind="parse")
    captured = capsys.readouterr()
    assert captured.err == ""
    assert status == 0
    line = read_fixture_file(out_dir / "common.jsonl")["tiny-chat/parse/as-written"]
    assert line["reference"]["text"] == f"Paris {YYA}"
    assert "".join(line["output_pieces"]) == f"Paris {YYA}"
    expected = "common.jsonl: 2 cases recorded, 1 with output ids built without the tokenizer's Unicode normalization\n"
    assert expected in captured.out
    # The count is reported, not stored: sets.toml keeps its fields.
    assert list(sets_tables(tmp_path)["parse"]["common"]) == list(set_tables.FIELDS)


def test_record_parse_needs_the_message(tmp_path, tiny_model, capsys):
    status, out_dir = record(
        tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("Hi")]}}]), kind="parse"
    )
    assert status == 1
    assert not (out_dir / "common.jsonl").exists()
    assert "a parse case needs `message`" in capsys.readouterr().err


def test_corpus_rejects_a_message_that_is_not_an_object(tmp_path):
    path = tmp_path / "set.jsonl"
    write_jsonl(path, [{"name": "a", "request": {"messages": []}, "message": "text"}])
    with pytest.raises(ValueError, match="`message` must be an object"):
        read_cases(path)


def test_corpus_keeps_the_origin_of_an_imported_case(tmp_path):
    path = tmp_path / "set.jsonl"
    origin = {"dataset": "bfcl", "row": "simple_python_0"}
    write_jsonl(
        path,
        [
            {"name": "a", "request": {"messages": []}, "origin": origin},
            {"name": "b", "request": {"messages": []}},
        ],
    )
    assert [case.origin for case in read_cases(path)] == [origin, None]


def test_corpus_rejects_an_origin_that_is_not_an_object(tmp_path):
    path = tmp_path / "set.jsonl"
    write_jsonl(path, [{"name": "a", "request": {"messages": []}, "origin": "bfcl"}])
    with pytest.raises(ValueError, match="`origin` must be an object"):
        read_cases(path)


def test_corpus_rejects_an_origin_that_names_no_dataset(tmp_path):
    path = tmp_path / "set.jsonl"
    write_jsonl(path, [{"name": "a", "request": {"messages": []}, "origin": {"row": "x"}}])
    with pytest.raises(ValueError, match=r"set\.jsonl:1: `origin` must name its `dataset`"):
        read_cases(path)


def test_record_other_kinds_and_oracles_are_not_implemented(tmp_path, tiny_model, capsys):
    argv = ["record", "--model", str(tiny_model), "--kind", "render", "--oracle", "sglang", "--fixtures", str(tmp_path)]
    assert main(argv) == 2
    assert "not implemented" in capsys.readouterr().err
    argv = [
        "record",
        "--model",
        str(tiny_model),
        "--kind",
        "tokenize",
        "--oracle",
        "reference",
        "--fixtures",
        str(tmp_path),
    ]
    assert main(argv) == 2


def test_the_case_schema_is_packaged_with_the_module():
    from importlib import resources

    packaged = resources.files("bellwether") / "schemas" / "case.schema.json"
    assert packaged.is_file()
    assert schema_path() == pathlib.Path(str(packaged))
    assert not (ROOT / "schemas").exists()


def test_fixture_writer_rejects_a_line_off_the_schema(tmp_path):
    good = {"id": "tiny-chat/render/a", "kind": "render", "model": "m", "reference": {"source": "hf-template"}}
    with pytest.raises(ValueError, match="tiny-chat/render/a: does not match the case schema at reference/source"):
        write_fixture_file(tmp_path / "x.jsonl", {"tiny-chat/render/a": {**good, "reference": {"source": "guess"}}})
    with pytest.raises(ValueError, match="at \\(root\\)"):
        write_fixture_file(tmp_path / "x.jsonl", {"tiny-chat/render/a": {**good, "extra": 1}})
    write_fixture_file(tmp_path / "x.jsonl", {"tiny-chat/render/a": good})
    assert read_fixture_file(tmp_path / "x.jsonl") == {"tiny-chat/render/a": good}


def test_fixture_writer_rejects_a_parse_line_without_its_ids_or_pieces(tmp_path):
    line = {
        "id": "tiny-chat/parse/a",
        "kind": "parse",
        "model": "tiny-chat",
        "output_ids": [5, 6],
        "output_pieces": ["a", "b"],
        "reference": {"source": "roundtrip", "text": "ab", "end_of_turn": {"stop_id": 7, "found_by": "turn"}},
    }
    write_fixture_file(tmp_path / "x.jsonl", {"tiny-chat/parse/a": line})
    for missing in ("output_ids", "output_pieces"):
        partial = {key: value for key, value in line.items() if key != missing}
        with pytest.raises(ValueError, match="tiny-chat/parse/a: does not match the case schema at \\(root\\)"):
            write_fixture_file(tmp_path / "y.jsonl", {"tiny-chat/parse/a": partial})


def test_a_fixture_file_reads_a_case_whose_text_holds_unicode_line_breaks_intact(tmp_path):
    line = {
        "id": "tiny-chat/render/a",
        "kind": "render",
        "model": "m",
        "request": {"messages": [user(BREAKS)]},
        "reference": {"source": "hf-template", "text": BREAKS},
    }
    write_fixture_file(tmp_path / "x.jsonl", {"tiny-chat/render/a": line})
    assert BREAKS in (tmp_path / "x.jsonl").read_text(encoding="utf-8")
    assert read_fixture_file(tmp_path / "x.jsonl") == {"tiny-chat/render/a": line}


def roundtrip_line(**reference) -> dict:
    """A round-trip parse line whose reference holds ``reference`` besides its source and text."""
    return {
        "id": "tiny-chat/parse/a",
        "kind": "parse",
        "model": "tiny-chat",
        "output_ids": [5],
        "output_pieces": ["a"],
        "reference": {"source": "roundtrip", "text": "a", **reference},
    }


@pytest.mark.parametrize(
    "end_of_turn",
    [
        None,
        {"stop_id": 7},
        {"found_by": "turn"},
        {"stop_id": 7, "found_by": "banana"},
        {"stop_id": 7, "found_by": "next_message"},
        {"stop_id": "x", "found_by": "turn"},
        {"stop_id": -1, "found_by": "turn"},
        {"stop_id": 7.5, "found_by": "turn"},
        {"stop_id": True, "found_by": "turn"},
        {"stop_id": 7, "found_by": "turn", "stop_ids": [7]},
        7,
    ],
    ids=[
        "missing",
        "no-found-by",
        "no-stop-id",
        "found-by-banana",
        "found-by-misspelled",
        "stop-id-string",
        "stop-id-negative",
        "stop-id-fraction",
        "stop-id-boolean",
        "misspelled-key",
        "not-an-object",
    ],
)
def test_the_case_schema_refuses_a_round_trip_parse_line_without_a_well_formed_end_of_turn(end_of_turn):
    # A round-trip parse line says which stop id ends its output and which step found it, beside finish_reason.
    line = roundtrip_line() if end_of_turn is None else roundtrip_line(end_of_turn=end_of_turn)
    errors = list(validator().iter_errors(line))
    assert errors, f"the schema took end_of_turn {end_of_turn!r}"


def test_the_case_schema_takes_a_well_formed_end_of_turn_and_asks_it_of_round_trip_parse_lines_only():
    for found_by in ("turn", "next-message"):
        validator().validate(roundtrip_line(end_of_turn={"stop_id": 0, "found_by": found_by}))
    engine = roundtrip_line()
    engine["reference"]["source"] = "engine:vllm"
    validator().validate(engine)
    validator().validate(
        {"id": "tiny-chat/render/a", "kind": "render", "model": "m", "reference": {"source": "roundtrip"}}
    )


COMMITTED_SETS = sorted([*ROOT.glob("fixtures/*/*/*.jsonl"), *ROOT.glob("fixtures/*/*/*.jsonl.zst")])


@pytest.mark.parametrize("path", COMMITTED_SETS, ids=lambda p: str(p.relative_to(ROOT)))
def test_committed_fixtures_are_canonical_sorted_and_valid(path):
    check_committed_set(path, ROOT)


def check_committed_set(path: pathlib.Path, root: pathlib.Path) -> None:
    """One committed set against its manifest, its ``sets.toml`` table and the schema; skipped while a pointer."""
    if is_lfs_pointer(path):
        pytest.skip(f"Git LFS has not fetched this set; fetch it with: {lfs_pull_command([path], root)}")
    manifest = load_manifest(path.parent.parent / "manifest.toml")
    plain = plain_text(path)
    name = path.name.removesuffix(".zst").removesuffix(".jsonl")
    table = set_tables.read(path.parent.parent / set_tables.FILE).get((path.parent.name, name))
    assert table is not None, f"{path}: sets.toml has no table for this set"
    form = "zstd" if path.name.endswith(".zst") else "plain"
    generate = {field: table[field] for field in set_tables.GENERATE_FIELDS if field in table}
    if path.parent.name == "parse":  # recorded from a hub checkpoint, which ships config.json at least
        assert list(generate) == list(set_tables.GENERATE_FIELDS), f"{path}: sets.toml names no stop ids of generate's"
    expected = set_tables.entry(form, plain, table["cases"], table["rejected"], **generate)
    assert table == expected, f"{path}: its sets.toml table does not match the file"
    # Only "\n" ends a line: U+2028 and the like stay raw inside a case's strings.
    lines = plain.split("\n")
    assert lines.pop() == "", f"{path}: the last line does not end in a newline"
    assert len(lines) == table["cases"]
    cases = [json.loads(line) for line in lines]
    assert [c["id"] for c in cases] == sorted(c["id"] for c in cases)
    for raw, case in zip(lines, cases, strict=True):
        validator().validate(case)
        assert raw == canonical_line(case)
        assert case["id"].startswith(f"{manifest.slug}/{path.parent.name}/")
        assert case["model"] == manifest.model
        assert case["reference"]["provenance"]["revision"] == manifest.revision
        if case["kind"] == "parse":
            assert len(case["output_pieces"]) == len(case["output_ids"])
            assert "".join(case["output_pieces"]) == case["reference"]["text"]


def record_argv(tmp_path, tiny_model, *extra: str) -> list[str]:
    argv = ["record", "--model", str(tiny_model), "--kind", "render", "--oracle", "reference"]
    return argv + ["--fixtures", str(tmp_path / "fixtures"), "--corpus", str(tmp_path / "corpus"), *extra]


def test_record_set_records_only_the_named_sets_and_leaves_the_others(tmp_path, tiny_model):
    status, out_dir = record(
        tmp_path,
        tiny_model,
        ("common", [{"name": "a", "request": {"messages": [user("A")]}}]),
        ("extra", [{"name": "b", "request": {"messages": [user("B")]}}]),
    )
    assert status == 0
    extra_before = (out_dir / "extra.jsonl").read_text()
    write_jsonl(tmp_path / "corpus" / "render" / "common.jsonl", [{"name": "a2", "request": {"messages": [user("A")]}}])
    (tmp_path / "corpus" / "render" / "extra.jsonl").unlink()

    assert main(record_argv(tmp_path, tiny_model, "--set", "common")) == 0

    text = (out_dir / "common.jsonl").read_text()
    ids = [json.loads(line)["id"] for line in text.removesuffix("\n").split("\n")]
    assert ids == ["tiny-chat/render/a2"]
    assert (out_dir / "extra.jsonl").read_text() == extra_before


def test_record_without_set_keeps_the_fixtures_of_an_imported_set(tmp_path, tiny_model):
    imported = {"name": "bfcl-x-0", "request": {"messages": [user("B")]}, "origin": {"dataset": "bfcl"}}
    record(
        tmp_path,
        tiny_model,
        ("common", [{"name": "a", "request": {"messages": [user("A")]}}]),
        ("bfcl-x", [imported]),
    )
    assert main(record_argv(tmp_path, tiny_model, "--set", "bfcl-x")) == 0
    out_dir = tmp_path / "fixtures" / "tiny-chat" / "render"
    recorded = plain_text(out_dir / "bfcl-x.jsonl.zst")

    assert main(record_argv(tmp_path, tiny_model)) == 0

    assert plain_text(out_dir / "bfcl-x.jsonl.zst") == recorded
    assert not (out_dir / "bfcl-x.jsonl").exists()


def test_record_set_rejects_a_set_the_corpus_does_not_have(tmp_path, tiny_model, capsys):
    record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    assert main(record_argv(tmp_path, tiny_model, "--set", "missing")) == 1
    assert "no corpus set named missing" in capsys.readouterr().err


def token_id(model: pathlib.Path, token: str) -> int:
    return Tokenizer.from_file(str(model / "tokenizer.json")).token_to_id(token)


def test_record_parse_names_the_stop_id_that_ends_the_output(tmp_path, tiny_model):
    # Today's case: the turn ends with the tokenizer's eos, and the output is the text before it, as before.
    cases = [{"name": "plain", "request": {"messages": [user("Hi")]}, "message": {"content": "Hello"}}]
    status, out_dir = record(tmp_path, tiny_model, ("common", cases), kind="parse")
    assert status == 0
    line = read_fixture_file(out_dir / "common.jsonl")["tiny-chat/parse/plain"]
    im_end = token_id(tiny_model, "<|im_end|>")
    assert line["reference"]["text"] == "Hello"
    assert im_end not in line["output_ids"]
    assert line["reference"]["end_of_turn"] == {"stop_id": im_end, "found_by": "turn"}
    assert "end_of_turn" not in line["reference"]["provenance"]


def turn_end_template(end: str) -> str:
    """A ChatML-like template whose assistant turn ends with ``end`` (Jinja string syntax), not ``<|im_end|>``."""
    return (
        "{%- for m in messages %}"
        "{%- if m['role'] == 'assistant' %}{{ '<|im_start|>assistant\\n' + (m['content'] or '') + '" + end + "' }}"
        "{%- else %}{{ '<|im_start|>' + m['role'] + '\\n' + (m['content'] or '') + '<|im_end|>\\n' }}{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
    )


def write_generation_config(model: pathlib.Path, *eos_tokens: str) -> None:
    """``generation_config.json`` as a checkpoint ships it, its ``eos_token_id`` listing ``eos_tokens``."""
    eos_token_id = [token_id(model, token) for token in eos_tokens]
    (model / "generation_config.json").write_text(json.dumps({"eos_token_id": eos_token_id}))


def test_the_output_ends_at_a_stop_id_the_generation_config_lists(tiny_model, tmp_path_factory):
    # MiniCPM5, Laguna, dots3, Inkling: the template ends the turn with a marker of the generation config's
    # eos_token_id, not with the tokenizer's eos. The output is the text before the marker.
    model = tiny_variant(tiny_model, tmp_path_factory, "end-chat", turn_end_template("<|end|>\\n"), tokens=("<|end|>",))
    write_generation_config(model, "<|end|>")
    out = RoundtripOracle(str(model), "local").render_output({"messages": [user("Hi")]}, {"content": "Hello"})
    assert out.text == "Hello"
    assert token_id(model, "<|end|>") not in out.output_ids
    assert out.end_of_turn == {"stop_id": token_id(model, "<|end|>"), "found_by": "turn"}


def test_a_turn_that_goes_on_after_its_stop_id_fails_the_case(tiny_model, tmp_path_factory):
    # The template renders part of the message after the point where generation stops, so no generation
    # returns this turn: the case is reported, not recorded.
    end = turn_end_template("<|end|>\\nSources: none.<|im_end|>\\n")
    model = tiny_variant(tiny_model, tmp_path_factory, "goes-on-chat", end, tokens=("<|end|>",))
    write_generation_config(model, "<|end|>")
    with pytest.raises(ValueError, match=r"the turn goes on after stop id \d+ \('<\|end\|>'\)"):
        RoundtripOracle(str(model), "local").render_output({"messages": [user("Hi")]}, {"content": "Hello"})
    # Whitespace and further stop ids are not the turn going on: Phi-4-mini writes `<|end|><|endoftext|>`.
    end = turn_end_template("<|end|>\\n<|im_end|>\\n")
    model = tiny_variant(tiny_model, tmp_path_factory, "end-eos-chat", end, tokens=("<|end|>",))
    write_generation_config(model, "<|end|>")
    out = RoundtripOracle(str(model), "local").render_output({"messages": [user("Hi")]}, {"content": "Hello"})
    assert (out.text, out.end_of_turn["stop_id"]) == ("Hello", token_id(model, "<|end|>"))


@pytest.fixture(scope="session")
def endoftext_model(tiny_model, tmp_path_factory) -> pathlib.Path:
    """Qwen3's stop set on the tiny model: `<|im_end|>`, which ends its turns, and `<|endoftext|>`."""
    model = tiny_variant(tiny_model, tmp_path_factory, "endoftext-chat", TEMPLATE, tokens=("<|endoftext|>",))
    write_generation_config(model, "<|im_end|>", "<|endoftext|>")
    return model


def call_with(arguments: str) -> dict:
    return {"type": "function", "function": {"name": "get_weather", "arguments": arguments}}


# `<|endoftext|>` with its angle brackets written as JSON escapes; the template gets the decoded text.
ESCAPED_ENDOFTEXT = chr(92) + "u003c|endoftext|" + chr(92) + "u003e"


@pytest.mark.parametrize(
    ("message", "where"),
    [
        ({"content": "It is <|endoftext|>"}, "content"),
        ({"content": "It is <|endoftext|> and more"}, "content"),
        ({"reasoning_content": "Say <|endoftext|> first.", "content": "Done."}, "reasoning_content"),
        (
            {"content": "", "tool_calls": [call_with('{"city": "<|endoftext|>"}')]},
            "tool_calls[0].function.arguments.city",
        ),
        (
            {"content": "", "tool_calls": [call_with('{"city": "' + ESCAPED_ENDOFTEXT + '"}')]},
            "tool_calls[0].function.arguments.city",
        ),
        (
            {"content": "", "tool_calls": [call_with('{"<|endoftext|>": "Paris"}')]},
            "tool_calls[0].function.arguments.<|endoftext|>",
        ),
    ],
    ids=["content-end", "content-middle", "reasoning", "call-arguments", "call-arguments-escaped", "argument-name"],
)
def test_a_stop_id_in_the_message_s_own_text_fails_the_case(endoftext_model, message, where):
    # Generation stops at the first stop id, so no output carries a message whose own text holds one. The turn is
    # searched by id: with the stop token at the end of the content the case was recorded short, and anywhere else it
    # was refused as if the template had rendered the rest of the message after generation stops. `</s>`, a stop token
    # of several checkpoints, is also ordinary HTML.
    request = {"messages": [user("Weather?")], "tools": [WEATHER_TOOL]}
    stop = token_id(endoftext_model, "<|endoftext|>")
    expected = f"the message's own text holds stop id {stop} ('<|endoftext|>') in {where}:"
    with pytest.raises(ValueError, match=re.escape(expected)):
        RoundtripOracle(str(endoftext_model), "local").render_output(request, message)


def test_a_special_token_that_is_not_a_stop_id_stays_in_the_output(endoftext_model):
    # The corpus has messages whose text holds `</think>` or `<tool_call>`: only a stop id fails a case.
    out = RoundtripOracle(str(endoftext_model), "local").render_output(
        {"messages": [user("Hi")]}, {"content": "Write <think> here."}
    )
    assert out.text == "Write <think> here."


# GLM's shape: no message is closed; each opens with its role tag, and a tool message needs its call's id here.
ROLE_TAG_TEMPLATE = (
    "{%- for m in messages %}"
    "{%- if m['role'] == 'assistant' %}{{ '<|assistant|>' + (m['content'] or '') }}"
    "{%- for c in (m['tool_calls'] or []) %}"
    "{{ '<tool_call>' + c['function']['name'] + ' ' + c['function']['arguments'] | tojson + '</tool_call>' }}"
    "{%- endfor %}"
    "{%- elif m['role'] == 'tool' %}{{ '<|observation|>' + m['tool_call_id'] + ' ' + m['content'] }}"
    "{%- else %}{{ '<|user|>' + m['content'] }}{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|assistant|>' }}{%- endif %}"
)
ROLE_TAGS = ("<|user|>", "<|assistant|>", "<|observation|>")


def test_a_turn_without_a_stop_id_ends_where_the_next_message_opens_with_one(tiny_model, tmp_path_factory):
    # GLM writes no end marker: generation stops on the tag that opens the next message, `<|user|>` after content
    # and `<|observation|>` after tool calls, both in its generation config. The output is the whole turn.
    model = tiny_variant(tiny_model, tmp_path_factory, "role-tag-chat", ROLE_TAG_TEMPLATE, tokens=ROLE_TAGS)
    write_generation_config(model, "<|user|>", "<|observation|>")
    oracle = RoundtripOracle(str(model), "local")
    content = oracle.render_output({"messages": [user("Hi")]}, {"content": "Hello"})
    assert content.text == "Hello"
    assert content.end_of_turn == {"stop_id": token_id(model, "<|user|>"), "found_by": "next-message"}
    calls = [{**weather_call(), "id": "call_1"}, {**weather_call(), "id": "call_2"}]
    request = {"messages": [user("Weather?")], "tools": [WEATHER_TOOL]}
    called = oracle.render_output(request, {"content": "", "tool_calls": calls})
    call_text = '<tool_call>get_weather {"city": "Paris"}</tool_call>'
    assert called.text == call_text * 2
    assert called.end_of_turn == {"stop_id": token_id(model, "<|observation|>"), "found_by": "next-message"}
    # The tag that opens the next message must be a stop id: without `<|user|>` listed, nothing ends a content turn.
    model = tiny_variant(tiny_model, tmp_path_factory, "role-tag-unlisted-chat", ROLE_TAG_TEMPLATE, tokens=ROLE_TAGS)
    write_generation_config(model, "<|observation|>")
    with pytest.raises(ValueError, match=r"no stop id in the turn, and the next message opens with '<\|user\|>'"):
        RoundtripOracle(str(model), "local").render_output({"messages": [user("Hi")]}, {"content": "Hello"})


def test_a_turn_the_template_renders_differently_once_the_next_message_follows_fails_the_case(
    tiny_model, tmp_path_factory
):
    # GLM-4.6, GLM-4.7-Flash and ERNIE-4.5 write no stop id in the turn and drop an earlier turn's reasoning once a
    # message follows it: the turn as the conversation's end is not a prefix of the conversation that goes on, so what
    # follows the turn is unknown.
    template = (
        "{%- for m in messages %}"
        "{%- if m['role'] == 'assistant' %}{{ '<|assistant|>' }}"
        "{%- if loop.last and m['reasoning_content'] %}"
        "{{ '<think>' + m['reasoning_content'] + '</think>' }}"
        "{%- endif %}"
        "{{ m['content'] or '' }}"
        "{%- else %}{{ '<|user|>' + m['content'] }}{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}{{ '<|assistant|>' }}{%- endif %}"
    )
    model = tiny_variant(tiny_model, tmp_path_factory, "drops-reasoning-chat", template, tokens=ROLE_TAGS)
    write_generation_config(model, "<|user|>", "<|observation|>")
    with pytest.raises(ValueError, match="the template renders the turn differently once the next message follows it"):
        RoundtripOracle(str(model), "local").render_output(
            {"messages": [user("Hi")]}, {"reasoning_content": "Think.", "content": "Hello"}
        )


def test_a_template_that_cannot_render_the_next_message_fails_the_case_with_its_error(tiny_model, tmp_path_factory):
    # The role-tag template needs each tool message's call id, and these calls have none: its error is the reason.
    model = tiny_variant(tiny_model, tmp_path_factory, "role-tag-no-id-chat", ROLE_TAG_TEMPLATE, tokens=ROLE_TAGS)
    write_generation_config(model, "<|user|>", "<|observation|>")
    with pytest.raises(ValueError, match="the template cannot render the next message: UndefinedError"):
        RoundtripOracle(str(model), "local").render_output(
            {"messages": [user("Weather?")], "tools": [WEATHER_TOOL]}, {"content": "", "tool_calls": [weather_call()]}
        )


# llava-1.5's shape: every message is `<role>: <text> `, so a turn ends in a space and the next message's role word
# starts with it: `Ġuser` covers the turn's last character and the next message's first ones.
TRAILING_SPACE_TEMPLATE = (
    "{%- for m in messages %}{{ m['role'] + ': ' + (m['content'] or '') + ' ' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ 'assistant:' }}{%- endif %}"
)


@pytest.mark.parametrize("listed", [False, True], ids=["not-a-stop-id", "a-stop-id"])
def test_a_token_that_straddles_the_end_of_the_turn_fails_the_case(tiny_model, tmp_path_factory, listed):
    # No token starts where the turn ends, so no generation stops there, whether or not the straddling token is a
    # stop id: llava-1.5-7b-hf's 16 common cases.
    model = tiny_variant(tiny_model, tmp_path_factory, "trailing-space-chat", TRAILING_SPACE_TEMPLATE)
    if listed:
        write_generation_config(model, "Ġuser")
    first = f"(the first one after it is 'Ġuser' ({token_id(model, 'Ġuser')}))"
    reason = f"no token starts where the turn ends once the next message follows it {first}"
    with pytest.raises(ValueError, match=re.escape(reason)):
        RoundtripOracle(str(model), "local").render_output({"messages": [user("Hi")]}, {"content": "Hello"})


def trimmed_offsets_variant(tiny_model, tmp_path_factory, name: str, template: str) -> pathlib.Path:
    """The role-tag variant with GPT-2's ByteLevel post-processor and ``trim_offsets``, as JetBrains' Mellum2 ships it:
    a token of spaces alone gets a zero-width offset, after its spaces."""
    model = tiny_variant(tiny_model, tmp_path_factory, name, template, tokens=ROLE_TAGS)
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.save(str(model / "tokenizer.json"))
    write_generation_config(model, "<|user|>", "<|observation|>")
    return model


def test_a_zero_width_token_neither_hides_nor_stands_in_for_the_stop_id_that_opens_the_next_message(
    tiny_model, tmp_path_factory
):
    # Spaces the template writes between the turn and the next message are a token that starts after the turn's end,
    # zero-width under trim_offsets: no token starts where the turn ends, and the reason names that first token.
    template = ROLE_TAG_TEMPLATE.replace(
        "{%- else %}{{ '<|user|>' + m['content'] }}", "{%- else %}{{ '  <|user|>' + m['content'] }}"
    )
    model = trimmed_offsets_variant(tiny_model, tmp_path_factory, "spaced-tag-chat", template)
    with pytest.raises(
        ValueError, match=r"no token starts where the turn ends .*\(the first one after it is 'Ġ+' \(\d+\)\)"
    ):
        RoundtripOracle(str(model), "local").render_output({"messages": [user("Hi")]}, {"content": "Hello"})
    # Spaces that end the turn itself are a zero-width token at its end, and the stop id after them still ends it.
    model = trimmed_offsets_variant(tiny_model, tmp_path_factory, "trimmed-role-tag-chat", ROLE_TAG_TEMPLATE)
    out = RoundtripOracle(str(model), "local").render_output({"messages": [user("Hi")]}, {"content": "Hello  "})
    assert (out.text, out.end_of_turn) == (
        "Hello  ",
        {"stop_id": token_id(model, "<|user|>"), "found_by": "next-message"},
    )


def test_a_turn_with_no_stop_id_that_nothing_follows_fails_the_case(tiny_model, tmp_path_factory):
    # This template writes nothing for a tool message, so the conversation that goes on ends where the turn ends.
    template = ROLE_TAG_TEMPLATE.replace(
        "{%- elif m['role'] == 'tool' %}{{ '<|observation|>' + m['tool_call_id'] + ' ' + m['content'] }}",
        "{%- elif m['role'] == 'tool' %}",
    )
    model = tiny_variant(tiny_model, tmp_path_factory, "silent-tool-chat", template, tokens=ROLE_TAGS)
    write_generation_config(model, "<|user|>", "<|observation|>")
    with pytest.raises(ValueError, match="no stop id in the turn, and nothing follows it"):
        RoundtripOracle(str(model), "local").render_output(
            {"messages": [user("Weather?")], "tools": [WEATHER_TOOL]}, {"content": "", "tool_calls": [weather_call()]}
        )


def test_a_generation_config_without_an_eos_token_id_stops_on_the_tokenizer_s_eos(tiny_model, tmp_path_factory):
    model = tiny_variant(tiny_model, tmp_path_factory, "no-eos-generation-chat", TEMPLATE)
    (model / "generation_config.json").write_text(json.dumps({"bos_token_id": 1}))
    assert generation_eos_ids(str(model), "local") == ([], "generation_config.json")
    out = RoundtripOracle(str(model), "local").render_output({"messages": [user("Hi")]}, {"content": "Hello"})
    assert out.end_of_turn == {"stop_id": token_id(model, "<|im_end|>"), "found_by": "turn"}


def test_the_next_message_is_a_user_message_after_content_and_one_tool_message_per_call_after_calls():
    assert next_messages({"content": "Hello"}) == [{"role": "user", "content": "Thanks."}]
    calls = [{**weather_call(), "id": "call_1"}, {"type": "function", "function": {"name": "f", "arguments": "{}"}}]
    assert next_messages({"content": "", "tool_calls": calls}) == [
        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "{}"},
        {"role": "tool", "name": "f", "content": "{}"},
    ]


def test_a_tokenizer_without_an_eos_token_stops_on_the_model_config_eos(tiny_model, tmp_path_factory):
    # Inkling: the tokenizer declares no eos, no generation_config.json ships, and config.json's eos_token_id is
    # the template's end marker.
    model = tiny_variant(tiny_model, tmp_path_factory, "no-eos-chat", TEMPLATE, without_eos=True)
    (model / "config.json").write_text(json.dumps({"eos_token_id": token_id(model, "<|im_end|>")}))
    out = RoundtripOracle(str(model), "local").render_output({"messages": [user("Hi")]}, {"content": "Hello"})
    assert out.text == "Hello"
    assert out.end_of_turn == {"stop_id": token_id(model, "<|im_end|>"), "found_by": "turn"}


def test_a_checkpoint_with_no_stop_id_at_all_is_refused(tiny_model, tmp_path_factory):
    # No eos in the tokenizer and no generation or model config naming one: nothing ends a turn.
    model = tiny_variant(tiny_model, tmp_path_factory, "no-stop-chat", TEMPLATE, without_eos=True)
    with pytest.raises(ValueError, match="has no stop id"):
        RoundtripOracle(str(model), "local")


def test_a_hub_checkpoint_needs_its_generation_config_cached_or_known_absent(tmp_path, monkeypatch):
    # Offline, transformers takes a generation_config.json that is merely not cached for one the repository does not
    # ship, and falls back to config.json. Here the file must be cached at the revision, or known absent: the hub
    # cache's .no_exist marker, which a download that got a 404 leaves.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    revision = "0123456789abcdef0123456789abcdef01234567"
    repo = tmp_path / "models--acme--Tiny-Chat"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 7}}))
    fetch = f"hf download acme/Tiny-Chat generation_config.json --revision {revision}"
    with pytest.raises(FileNotFoundError, match=f"generation_config.json .*`{fetch}`"):
        generation_eos_ids("acme/Tiny-Chat", revision)
    absent = repo / ".no_exist" / revision / "generation_config.json"
    absent.parent.mkdir(parents=True)
    absent.touch()
    assert generation_eos_ids("acme/Tiny-Chat", revision) == ([7], "config.json")
    absent.unlink()
    (snapshot / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 8]}))
    assert generation_eos_ids("acme/Tiny-Chat", revision) == ([7, 8], "generation_config.json")


def test_config_json_is_read_with_the_defaults_of_the_class_transformers_has_for_its_model_type(
    tiny_model, tmp_path_factory
):
    # vLLM gives GenerationConfig.from_model_config the config object, which carries its class's defaults: a
    # config.json of {"model_type": "llama"} states no eos, and LlamaConfig's is 2.
    model = tiny_variant(tiny_model, tmp_path_factory, "llama-config-chat", TEMPLATE)
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert generation_eos_ids(str(model), "local") == ([2], "config.json")


def test_config_json_of_a_model_type_transformers_does_not_know_is_read_as_written(tiny_model, tmp_path_factory):
    # Its class is the vendor's code, named by auto_map, which bellwether never runs: the file is read as written.
    model = tiny_variant(tiny_model, tmp_path_factory, "vendor-config-chat", TEMPLATE)
    (model / "configuration_acme.py").write_text('raise RuntimeError("the vendor\'s code ran")\n')
    config = {
        "model_type": "acme_chat",
        "auto_map": {"AutoConfig": "configuration_acme.AcmeConfig"},
        "text_config": {"eos_token_id": 7},
    }
    (model / "config.json").write_text(json.dumps(config))
    assert generation_eos_ids(str(model), "local") == ([7], "config.json")


def test_where_hf_generate_would_not_stop_the_run_says_so_once_and_sets_toml_names_its_stop_ids(
    tmp_path, tiny_model, tmp_path_factory, capsys
):
    # Qwen3.5-9B ships no generation_config.json, its config.json lists only <|endoftext|>, and its turns end with
    # <|im_end|>, the tokenizer's eos: vLLM stops there, transformers' generate does not. The outputs end where vLLM,
    # the serving engine, stops, and each line carries only its stop id. What generate stops on is a fact of the
    # checkpoint: the set's table in sets.toml holds the ids and the file they come from, and the run says once for
    # the model how many outputs generate would not end.
    model = tiny_variant(tiny_model, tmp_path_factory, "eos-unlisted-chat", TEMPLATE, tokens=("<|endoftext|>",))
    endoftext, im_end = token_id(model, "<|endoftext|>"), token_id(model, "<|im_end|>")
    (model / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": endoftext}}))
    cases = [
        {"name": "plain", "request": {"messages": [user("Hi")]}, "message": {"content": "Hello"}},
        {"name": "other", "request": {"messages": [user("Hi")]}, "message": {"content": "Paris."}},
    ]
    status, out_dir = record(tmp_path, model, ("common", cases), kind="parse")
    assert status == 0
    for line in read_fixture_file(out_dir / "common.jsonl").values():
        assert line["reference"]["end_of_turn"] == {"stop_id": im_end, "found_by": "turn"}
    table = sets_tables(tmp_path)["parse"]["common"]
    assert (table["generate_stop_ids"], table["generate_stop_ids_from"]) == ([endoftext], "config.json")
    err = capsys.readouterr().err
    assert err.count("stop sets differ") == 1
    assert (
        f"stop sets differ for {model}: transformers' generate stops on [{endoftext}] (config.json), so 2 of the 2 "
        f"outputs recorded here end on a stop id it does not stop on: {im_end} ('<|im_end|>')\n"
    ) in err


def test_where_hf_generate_stops_where_vllm_does_the_run_says_nothing(tmp_path, tiny_model, tmp_path_factory, capsys):
    # The generation config lists the id the turns end on, as most checkpoints' do: generate and vLLM agree.
    model = tiny_variant(tiny_model, tmp_path_factory, "eos-listed-chat", TEMPLATE, tokens=("<|endoftext|>",))
    write_generation_config(model, "<|im_end|>", "<|endoftext|>")
    cases = [{"name": "plain", "request": {"messages": [user("Hi")]}, "message": {"content": "Hello"}}]
    status, _ = record(tmp_path, model, ("common", cases), kind="parse")
    assert status == 0
    table = sets_tables(tmp_path)["parse"]["common"]
    expected = ([token_id(model, "<|im_end|>"), token_id(model, "<|endoftext|>")], "generation_config.json")
    assert (table["generate_stop_ids"], table["generate_stop_ids_from"]) == expected
    assert "stop sets differ" not in capsys.readouterr().err


def test_a_checkpoint_without_a_generation_or_model_config_has_no_generate_to_name(tmp_path, tiny_model, capsys):
    # The tiny model ships neither file, so transformers could not load a model to call generate on: its parse table
    # names no stop ids of generate's, and the run compares nothing.
    cases = [{"name": "plain", "request": {"messages": [user("Hi")]}, "message": {"content": "Hello"}}]
    status, _ = record(tmp_path, tiny_model, ("common", cases), kind="parse")
    assert status == 0
    assert set(sets_tables(tmp_path)["parse"]["common"]) == set(set_tables.FIELDS)
    assert "stop sets differ" not in capsys.readouterr().err


def test_tool_calls_in_the_history_also_reach_the_template_as_objects(items_model):
    request = {
        "messages": [
            user("Weather in Paris?"),
            {"role": "assistant", "content": "", "tool_calls": [weather_call()]},
            {"role": "tool", "content": "Sunny"},
            user("And now?"),
        ]
    }
    out = RoundtripOracle(str(items_model), "local").render_output(
        request, {"content": "", "tool_calls": [weather_call()]}
    )
    assert out.text == "<tool_call>get_weather city=Paris</tool_call>"
    assert request["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'


def test_record_records_a_checkpoint_whose_oracle_inputs_are_the_ones_its_manifest_lists(tmp_path, tiny_model):
    path = write_manifest(
        tmp_path / "fixtures", "tiny-chat", str(tiny_model), inputs=oracle_inputs(str(tiny_model), "local")
    )
    assert "tokenizer_config.json" in load_manifest(path).inputs
    status, out_dir = record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    assert status == 0
    assert list(read_fixture_file(out_dir / "common.jsonl")) == ["tiny-chat/render/a"]


def test_record_refuses_a_checkpoint_whose_oracle_input_changed_and_names_the_file(tmp_path, tiny_model, capsys):
    model = tmp_path / "tiny-chat"
    shutil.copytree(tiny_model, model)
    listed = oracle_inputs(str(model), "local")
    path = write_manifest(tmp_path / "fixtures", "tiny-chat", str(model), inputs=listed)
    config = json.loads((model / "tokenizer_config.json").read_text())
    config["chat_template"] = "{{ messages[0]['content'] }}"
    (model / "tokenizer_config.json").write_text(json.dumps(config))
    found = oracle_inputs(str(model), "local")["tokenizer_config.json"]
    write_jsonl(tmp_path / "corpus" / "render" / "common.jsonl", [{"name": "a", "request": {"messages": [user("A")]}}])

    assert main(record_argv(tmp_path, model)) == 1

    err = capsys.readouterr().err
    listing = f"tokenizer_config.json (listed {listed['tokenizer_config.json'][:12]}, found {found[:12]})"
    assert f"its oracle inputs at local differ from {path}: {listing};" in err
    assert not (tmp_path / "fixtures" / "tiny-chat" / "render").exists()


def test_record_refuses_a_group_member_and_names_its_group(tmp_path, tiny_model, capsys):
    fixtures = tmp_path / "fixtures"
    inputs = oracle_inputs(str(tiny_model), "local")
    write_manifest(fixtures, "tiny-chat", str(tiny_model), inputs=inputs)
    write_manifest(fixtures, "tiny-chat-mini", "acme/Tiny-Chat-Mini", inputs=inputs, group="tiny-chat")
    write_jsonl(tmp_path / "corpus" / "render" / "common.jsonl", [{"name": "a", "request": {"messages": [user("A")]}}])

    assert main(record_argv(tmp_path, "acme/Tiny-Chat-Mini")) == 1

    err = capsys.readouterr().err
    assert "acme/Tiny-Chat-Mini is a member of checkpoint group tiny-chat" in err
    assert f"record the group instead: bellwether record --model {tiny_model}" in err
    assert not (fixtures / "tiny-chat-mini" / "render").exists() and not (fixtures / "tiny-chat" / "render").exists()


def test_record_refuses_a_manifest_that_lists_no_oracle_inputs(tmp_path, tiny_model, capsys):
    path = write_manifest(tmp_path / "fixtures", "tiny-chat", str(tiny_model))
    assert main(record_argv(tmp_path, tiny_model)) == 1
    assert f"{path} lists no oracle inputs; `bellwether manifests` writes them" in capsys.readouterr().err


def render_cases(*names: str) -> dict[str, dict]:
    cases = {}
    for name in names:
        case_id = f"m/render/{name}"
        cases[case_id] = {
            "id": case_id,
            "kind": "render",
            "model": "m",
            "reference": {"source": "hf-template", "text": "Café ☕"},
        }
    return cases


def test_a_compressed_fixture_file_holds_the_same_lines_as_the_plain_one(tmp_path):
    cases = render_cases("b", "a")
    write_fixture_file(tmp_path / "set.jsonl", cases)
    write_fixture_file(tmp_path / "set.jsonl.zst", cases)
    plain = (tmp_path / "set.jsonl").read_bytes()
    assert (tmp_path / "set.jsonl.zst").read_bytes()[:4] == bytes.fromhex("28b52ffd")  # the zstd frame magic
    assert plain_text(tmp_path / "set.jsonl.zst").encode("utf-8") == plain
    assert plain_text(tmp_path / "set.jsonl") == plain.decode("utf-8")
    assert read_fixture_file(tmp_path / "set.jsonl.zst") == read_fixture_file(tmp_path / "set.jsonl")
    assert list(read_fixture_file(tmp_path / "set.jsonl.zst")) == ["m/render/a", "m/render/b"]


def imported(name: str, text: str) -> dict:
    return {"name": name, "request": {"messages": [user(text)]}, "origin": {"dataset": "bench"}}


def sets_tables(tmp_path) -> dict:
    import tomllib

    return tomllib.loads((tmp_path / "fixtures" / "tiny-chat" / "sets.toml").read_text())


def test_record_writes_an_imported_set_compressed_and_a_hand_written_set_plain(tmp_path, tiny_model):
    status, out_dir = record(
        tmp_path,
        tiny_model,
        ("common", [{"name": "a", "request": {"messages": [user("A")]}}]),
        ("bench-x", [imported("bench-x-0", "X"), imported("bench-x-1", "Y")]),
    )
    assert status == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["bench-x.jsonl.zst", "common.jsonl"]
    tables = sets_tables(tmp_path)
    for name, form, cases in (("common", "plain", 1), ("bench-x", "zstd", 2)):
        file = out_dir / (f"{name}.jsonl" if form == "plain" else f"{name}.jsonl.zst")
        plain = plain_text(file).encode("utf-8")
        assert tables["render"][name] == {
            "form": form,
            "cases": cases,
            "rejected": 0,
            "plain_bytes": len(plain),
            "plain_sha256": hashlib.sha256(plain).hexdigest(),
        }


def test_a_set_that_changes_form_leaves_one_file(tmp_path, tiny_model):
    status, out_dir = record(tmp_path, tiny_model, ("x", [{"name": "x-0", "request": {"messages": [user("X")]}}]))
    assert status == 0 and (out_dir / "x.jsonl").exists()
    status, _ = record(tmp_path, tiny_model, ("x", [imported("x-0", "X")]))
    assert status == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["x.jsonl.zst"]
    assert sets_tables(tmp_path)["render"]["x"]["form"] == "zstd"


def test_a_set_that_changes_back_to_plain_leaves_one_file_with_its_witnesses(tmp_path, tiny_model):
    status, out_dir = record(tmp_path, tiny_model, ("x", [imported("x-0", "X")]))
    assert status == 0 and (out_dir / "x.jsonl.zst").exists()
    recorded = read_fixture_file(out_dir / "x.jsonl.zst")
    recorded["tiny-chat/render/x-0"]["witnesses"] = {"vllm": {"version": "0.30.0", "input_ids": [1]}}
    write_fixture_file(out_dir / "x.jsonl.zst", recorded)
    status, _ = record(tmp_path, tiny_model, ("x", [{"name": "x-0", "request": {"messages": [user("X")]}}]))
    assert status == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["x.jsonl"]
    line = read_fixture_file(out_dir / "x.jsonl")["tiny-chat/render/x-0"]
    assert line.get("witnesses") == {"vllm": {"version": "0.30.0", "input_ids": [1]}}
    assert sets_tables(tmp_path)["render"]["x"]["form"] == "plain"


def test_record_set_updates_only_its_own_table_and_counts_rejections(tmp_path, tiny_model):
    record(
        tmp_path,
        tiny_model,
        ("common", [{"name": "a", "request": {"messages": [user("A")]}}]),
        ("extra", [{"name": "b", "request": {"messages": [user("B")]}}]),
    )
    before = sets_tables(tmp_path)["render"]["extra"]
    bad = {
        "name": "c",
        "request": {"messages": [user("C")], "add_generation_prompt": True, "continue_final_message": True},
    }
    write_jsonl(
        tmp_path / "corpus" / "render" / "common.jsonl", [{"name": "a", "request": {"messages": [user("A")]}}, bad]
    )
    assert main(record_argv(tmp_path, tiny_model, "--set", "common")) == 1
    tables = sets_tables(tmp_path)["render"]
    assert tables["extra"] == before
    assert (tables["common"]["cases"], tables["common"]["rejected"]) == (1, 1)


@pytest.mark.parametrize("last", ["parse", "render"])
def test_record_runs_of_the_two_kinds_at_once_keep_each_others_tables(tmp_path, tiny_model, monkeypatch, last):
    # The other kind's run records from start to end while this run writes its set file, after this run started and
    # before it writes sets.toml. A run that wrote back the tables it read at its start would drop the other kind's.
    other = "render" if last == "parse" else "parse"
    write_jsonl(tmp_path / "corpus" / "render" / "common.jsonl", [{"name": "a", "request": {"messages": [user("A")]}}])
    write_jsonl(
        tmp_path / "corpus" / "parse" / "common.jsonl",
        [{"name": "a", "request": {"messages": [user("Hi")]}, "message": {"content": "Hello"}}],
    )
    statuses = []

    def write_while_the_other_kind_records(path, cases):
        write_fixture_file(path, cases)
        if path.parent.name == last:
            statuses.append(record(tmp_path, tiny_model, kind=other)[0])

    monkeypatch.setattr("bellwether.record.write_fixture_file", write_while_the_other_kind_records)
    statuses.append(record(tmp_path, tiny_model, kind=last)[0])

    assert statuses == [0, 0]
    assert sorted(set_tables.read(tmp_path / "fixtures" / "tiny-chat" / "sets.toml")) == [
        ("parse", "common"),
        ("render", "common"),
    ]


def test_record_drops_only_its_own_kinds_table_of_a_set_it_no_longer_writes(tmp_path, tiny_model):
    names = ("common", "extra", "tools")
    for kind, message in (("render", {}), ("parse", {"message": {"content": "Hello"}})):
        cases = {name: [{"name": f"{name}-0", "request": {"messages": [user(name)]}, **message}] for name in names}
        assert record(tmp_path, tiny_model, *cases.items(), kind=kind)[0] == 0
    (tmp_path / "corpus" / "render" / "tools.jsonl").unlink()  # the corpus no longer has the set
    write_jsonl(tmp_path / "corpus" / "render" / "extra.jsonl", [{"name": "extra-0", "request": {"prompt": "x"}}])

    status, _ = record(tmp_path, tiny_model)  # extra-0 no longer renders, so the run writes no extra set

    assert status == 1
    assert sorted(set_tables.read(tmp_path / "fixtures" / "tiny-chat" / "sets.toml")) == [
        ("parse", "common"),
        ("parse", "extra"),
        ("parse", "tools"),
        ("render", "common"),
    ]


def test_record_reads_and_writes_sets_toml_under_the_lock_beside_it(tmp_path, tiny_model, monkeypatch):
    # An exclusive flock on sets.toml.lock: while a run holds it, no other open of that file can take it.
    lock = tmp_path / "fixtures" / "tiny-chat" / "sets.toml.lock"
    seen: list[tuple[str, bool]] = []

    def held() -> bool:
        with open(lock, "a") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False

    real_read, real_write = set_tables.read, set_tables.write

    def read(path):
        seen.append(("read", held()))
        return real_read(path)

    def write(path, tables):
        seen.append(("write", held()))
        real_write(path, tables)

    monkeypatch.setattr(set_tables, "read", read)
    monkeypatch.setattr(set_tables, "write", write)
    status, _ = record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))

    assert status == 0
    assert seen == [("read", True), ("write", True)]


def test_sets_toml_is_replaced_whole_never_rewritten_in_place(tmp_path):
    # A reader, or a run cut off mid-write, would otherwise find half a table.
    path = tmp_path / "sets.toml"
    set_tables.write(path, {("render", "common"): set_tables.entry("plain", "a\n", 1, 0)})
    inode = path.stat().st_ino

    set_tables.write(path, {("render", "common"): set_tables.entry("plain", "b\n", 1, 0)})

    assert path.stat().st_ino != inode, "sets.toml was rewritten in place"
    assert [p.name for p in tmp_path.iterdir()] == ["sets.toml"]


# What Git LFS 3.7.1 leaves in place of a file it has not fetched; this one stands for the 13 bytes "compressed-x\n".
LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:af16c249d1a79d093931e0317035d67af8c1f47632feca7b2216bc4787d7983e\n"
    "size 13\n"
)


def test_unpack_writes_one_plain_tree_of_both_forms_and_the_manifests(tmp_path, tiny_model):
    record(
        tmp_path,
        tiny_model,
        ("common", [{"name": "a", "request": {"messages": [user("A")]}}]),
        ("bench-x", [imported("bench-x-0", "X")]),
    )
    fixtures, out = tmp_path / "fixtures", tmp_path / "plain"
    assert main(["unpack", "--fixtures", str(fixtures), "--out", str(out)]) == 0
    for name in ("common", "bench-x"):
        source = next((fixtures / "tiny-chat" / "render").glob(f"{name}.jsonl*"))
        assert (out / "tiny-chat" / "render" / f"{name}.jsonl").read_text() == plain_text(source)
    for name in ("manifest.toml", "sets.toml"):
        assert (out / "tiny-chat" / name).read_bytes() == (fixtures / "tiny-chat" / name).read_bytes()


def test_a_git_lfs_pointer_is_named_not_decompressed(tmp_path):
    pointer = tmp_path / "set.jsonl.zst"
    pointer.write_text(LFS_POINTER)
    with pytest.raises(ValueError, match="Git LFS pointer"):
        plain_text(pointer)


@pytest.mark.parametrize("path", sorted(ROOT.glob("fixtures/*/sets.toml")), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_sets_toml_table_has_its_set(path):
    for kind, name in set_tables.read(path):
        files = [path.parent / kind / f"{name}.jsonl", path.parent / kind / f"{name}.jsonl.zst"]
        assert sum(f.is_file() for f in files) == 1, f"{path}: [{kind}.{name}] needs exactly one set file"


@pytest.mark.parametrize("path", sorted(ROOT.glob("fixtures/*/manifest.toml")), ids=lambda p: str(p.relative_to(ROOT)))
def test_committed_manifests_list_inputs_and_tier_and_a_member_names_a_primary_with_equal_inputs(path):
    manifest = load_manifest(path)
    assert manifest.inputs, f"{path}: no [inputs]; `bellwether manifests` writes them"
    assert manifest.tier is not None, f"{path}: no tier; `bellwether manifests` writes it"
    if manifest.group is not None:
        primary = load_manifest(ROOT / "fixtures" / manifest.group / "manifest.toml")
        assert primary.group is None, f"{path}: its group's manifest is itself a member"
        assert primary.inputs == manifest.inputs, f"{path}: its oracle inputs differ from its group's"
        assert [p.name for p in path.parent.iterdir()] == ["manifest.toml"], f"{path}: a member holds no fixtures"


def git(cwd: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def runs(*argv: str) -> bool:
    try:
        return subprocess.run(argv, capture_output=True).returncode == 0
    except OSError:
        return False


needs_git = pytest.mark.skipif(not runs("git", "--version"), reason="needs git")


@pytest.fixture
def git_sandbox(tmp_path, monkeypatch) -> pathlib.Path:
    """``tmp_path``, with git reading a config of its own and no repository inherited from the environment."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


# Git LFS matches --include against the path from the repository root, and .lfsconfig's fetchexclude wins over it
# unless --exclude '' clears it: an absolute path, one relative to a subdirectory, or no --exclude fetches nothing.
FETCH_BENCH_X = "git lfs pull --include 'fixtures/m/render/bench-x.jsonl.zst' --exclude ''"


@needs_git
def test_a_pointer_error_names_the_command_that_fetches_it_from_the_repository_root(git_sandbox, monkeypatch):
    git(git_sandbox, "init", "-q", "clone")
    pointer = git_sandbox / "clone" / "fixtures" / "m" / "render" / "bench-x.jsonl.zst"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(LFS_POINTER)
    monkeypatch.chdir(pointer.parent.parent)
    for given in (pointer, pathlib.Path("render/bench-x.jsonl.zst")):
        with pytest.raises(ValueError, match=re.escape(FETCH_BENCH_X)):
            plain_text(given)


def test_the_committed_fixture_check_skips_a_set_git_lfs_has_not_fetched(tmp_path):
    pointer = tmp_path / "fixtures" / "m" / "render" / "bench-x.jsonl.zst"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(LFS_POINTER)
    with pytest.raises(pytest.skip.Exception, match=re.escape(FETCH_BENCH_X)):
        check_committed_set(pointer, tmp_path)


needs_git_lfs = pytest.mark.skipif(not runs("git", "lfs", "version"), reason="needs git and git-lfs")


@pytest.fixture
def lfs_clone(git_sandbox, monkeypatch) -> pathlib.Path:
    """A default clone of a repository (``source``, beside it) that keeps a benchmark set in Git LFS under this
    repository's .gitattributes and .lfsconfig, pushed to a local remote: the clone holds the set's pointer."""
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "bellwether")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "bellwether@example.com")
    source, remote, clone = git_sandbox / "source", git_sandbox / "remote.git", git_sandbox / "clone"
    git(git_sandbox, "init", "-q", "--bare", "-b", "main", str(remote))
    git(git_sandbox, "init", "-q", "-b", "main", str(source))
    git(source, "lfs", "install")  # the LFS filters, in the sandbox's config, and the hook that pushes LFS objects
    for name in (".gitattributes", ".lfsconfig"):
        (source / name).write_bytes((ROOT / name).read_bytes())
    write_manifest(source / "fixtures", "m", "acme/M")
    write_fixture_file(source / "fixtures" / "m" / "render" / "common.jsonl", render_cases("a"))
    write_fixture_file(source / "fixtures" / "m" / "render" / "bench-x.jsonl.zst", render_cases("x-0", "x-1"))
    write_compressed(source / "corpus" / "render" / "big-x.jsonl.zst", [{"name": "big-x-0", "request": {}}])
    git(source, "add", "-A")
    git(source, "commit", "-q", "-m", "fixtures")
    git(source, "push", "-q", remote.as_uri(), "main")
    git(git_sandbox, "clone", "-q", remote.as_uri(), str(clone))
    return clone


@needs_git_lfs
def test_unpack_fetches_the_sets_a_default_clone_holds_as_pointers(lfs_clone, monkeypatch):
    assert is_lfs_pointer(lfs_clone / "fixtures" / "m" / "render" / "bench-x.jsonl.zst")
    monkeypatch.chdir(lfs_clone / "fixtures")
    out = lfs_clone.parent / "plain"
    assert main(["unpack", "--fixtures", ".", "--out", str(out)]) == 0
    source = lfs_clone.parent / "source" / "fixtures" / "m" / "render"
    assert (out / "m" / "render" / "bench-x.jsonl").read_text() == plain_text(source / "bench-x.jsonl.zst")
    assert (out / "m" / "render" / "common.jsonl").read_text() == (source / "common.jsonl").read_text()


@needs_git_lfs
def test_a_compressed_corpus_set_is_kept_in_git_lfs_and_a_default_clone_fetches_it(lfs_clone):
    # .lfsconfig keeps a clone from fetching the fixture sets only: record and import --check read the corpus whole.
    listed = subprocess.run(["git", "lfs", "ls-files", "--name-only"], cwd=lfs_clone, capture_output=True, text=True)
    assert "corpus/render/big-x.jsonl.zst" in listed.stdout.split()
    fetched = lfs_clone / "corpus" / "render" / "big-x.jsonl.zst"
    assert not is_lfs_pointer(fetched)
    assert plain_text(fetched) == plain_text(lfs_clone.parent / "source" / "corpus" / "render" / "big-x.jsonl.zst")
    assert is_lfs_pointer(lfs_clone / "fixtures" / "m" / "render" / "bench-x.jsonl.zst")


@needs_git
def test_unpack_names_the_sets_it_could_not_fetch_and_writes_nothing(git_sandbox, tiny_model, monkeypatch, capsys):
    git(git_sandbox, "init", "-q")
    common = [{"name": "a", "request": {"messages": [user("A")]}}]
    record(git_sandbox, tiny_model, ("common", common), ("bench-x", [imported("bench-x-0", "X")]))
    pointer = git_sandbox / "fixtures" / "tiny-chat" / "render" / "bench-x.jsonl.zst"
    pointer.write_text(LFS_POINTER)
    asked: list[list[pathlib.Path]] = []
    monkeypatch.setattr("bellwether.unpack.fetch", asked.append)  # a pull that fetches nothing: no git-lfs, no network
    out = git_sandbox / "plain"
    assert main(["unpack", "--fixtures", str(git_sandbox / "fixtures"), "--out", str(out)]) == 1
    assert asked == [[pointer]]
    err = capsys.readouterr().err
    assert str(pointer) in err
    assert "git lfs pull --include 'fixtures/tiny-chat/render/bench-x.jsonl.zst' --exclude ''" in err
    assert not out.exists()


def test_is_lfs_pointer_tells_a_pointer_from_a_set_in_either_form(tmp_path):
    (tmp_path / "pointer.jsonl.zst").write_text(LFS_POINTER)
    for name in ("set.jsonl", "set.jsonl.zst"):
        write_fixture_file(tmp_path / name, render_cases("a"))
    assert is_lfs_pointer(tmp_path / "pointer.jsonl.zst")
    assert not is_lfs_pointer(tmp_path / "set.jsonl")
    assert not is_lfs_pointer(tmp_path / "set.jsonl.zst")


def test_unpack_writes_a_model_that_has_only_hand_written_sets(tmp_path, tiny_model, monkeypatch):
    record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    asked: list[list[pathlib.Path]] = []
    monkeypatch.setattr("bellwether.unpack.fetch", asked.append)
    fixtures, out = tmp_path / "fixtures", tmp_path / "plain"
    assert main(["unpack", "--fixtures", str(fixtures), "--out", str(out)]) == 0
    assert asked == []  # nothing to fetch; an empty --include would fetch every set
    written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert written == ["tiny-chat/manifest.toml", "tiny-chat/render/common.jsonl", "tiny-chat/sets.toml"]
    assert (out / "tiny-chat" / "render" / "common.jsonl").read_bytes() == (
        fixtures / "tiny-chat" / "render" / "common.jsonl"
    ).read_bytes()


def test_unpack_model_writes_and_fetches_that_model_only(tmp_path, tiny_model, monkeypatch):
    common = [{"name": "a", "request": {"messages": [user("A")]}}]
    record(tmp_path, tiny_model, ("common", common), ("bench-x", [imported("bench-x-0", "X")]))
    write_manifest(tmp_path / "fixtures", "other", "acme/Other")
    write_fixture_file(tmp_path / "fixtures" / "other" / "render" / "common.jsonl", render_cases("a"))
    (tmp_path / "fixtures" / "other" / "render" / "bench-y.jsonl.zst").write_text(LFS_POINTER)
    asked: list[list[pathlib.Path]] = []
    monkeypatch.setattr("bellwether.unpack.fetch", asked.append)
    out = tmp_path / "plain"
    argv = ["unpack", "--fixtures", str(tmp_path / "fixtures"), "--out", str(out), "--model", str(tiny_model)]
    assert main(argv) == 0
    assert asked == []
    assert sorted(p.name for p in out.iterdir()) == ["tiny-chat"]
    assert sorted(p.name for p in (out / "tiny-chat" / "render").iterdir()) == ["bench-x.jsonl", "common.jsonl"]


def test_unpack_again_drops_a_set_the_fixtures_no_longer_have(tmp_path, tiny_model, monkeypatch):
    monkeypatch.setattr("bellwether.unpack.fetch", lambda paths: None)
    record(
        tmp_path,
        tiny_model,
        ("common", [{"name": "a", "request": {"messages": [user("A")]}}]),
        ("extra", [{"name": "b", "request": {"messages": [user("B")]}}]),
    )
    fixtures, out = tmp_path / "fixtures", tmp_path / "plain"
    assert main(["unpack", "--fixtures", str(fixtures), "--out", str(out)]) == 0
    (tmp_path / "corpus" / "render" / "extra.jsonl").unlink()
    assert main(record_argv(tmp_path, tiny_model)) == 0

    assert main(["unpack", "--fixtures", str(fixtures), "--out", str(out)]) == 0

    assert not (out / "tiny-chat" / "render" / "extra.jsonl").exists()
    assert (out / "tiny-chat" / "render" / "common.jsonl").is_file()


def test_unpack_refuses_to_write_over_the_fixtures_it_reads(tmp_path, tiny_model, capsys):
    record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    fixtures = tmp_path / "fixtures"
    before = (fixtures / "tiny-chat" / "render" / "common.jsonl").read_bytes()
    assert main(["unpack", "--fixtures", str(fixtures), "--out", str(fixtures)]) == 1
    assert "--out" in capsys.readouterr().err
    assert (fixtures / "tiny-chat" / "render" / "common.jsonl").read_bytes() == before


def test_unpack_model_names_the_known_models_when_none_matches(tmp_path, tiny_model, capsys):
    record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    out = tmp_path / "plain"
    assert main(["unpack", "--fixtures", str(tmp_path / "fixtures"), "--out", str(out), "--model", "acme/None"]) == 1
    assert "acme/None" in capsys.readouterr().err
    assert not out.exists()


def test_fetch_pulls_in_batches_that_stay_under_the_argument_limit(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("bellwether.unpack.repository_root", lambda path: tmp_path)
    monkeypatch.setattr("bellwether.unpack.subprocess.run", lambda argv, cwd: calls.append(argv))
    paths = [tmp_path / "fixtures" / "m" / "render" / f"set-{i:05d}-{'x' * 40}.jsonl.zst" for i in range(3000)]
    unpack_module.fetch(paths)
    includes = [argv[argv.index("--include") + 1] for argv in calls]
    assert len(calls) > 1
    assert all(len(include.encode()) < 100_000 for include in includes)
    assert sorted(p for include in includes for p in include.split(",")) == sorted(
        p.relative_to(tmp_path).as_posix() for p in paths
    )


def test_rewriting_an_unchanged_compressed_set_keeps_its_bytes(tmp_path):
    # A zstandard upgrade may compress the same content to other bytes; that must not make a new LFS object.
    path = tmp_path / "set.jsonl.zst"
    lines = render_cases("a", "b")
    write_fixture_file(path, lines)
    other_bytes = zstandard.ZstdCompressor(level=3).compress(plain_text(path).encode("utf-8"))
    path.write_bytes(other_bytes)

    write_fixture_file(path, lines)

    assert path.read_bytes() == other_bytes
    write_fixture_file(path, render_cases("a"))
    assert path.read_bytes() != other_bytes


def test_a_compressed_fixture_set_cut_short_is_replaced_by_the_next_write(tmp_path):
    # A record run cut short leaves a frame zstd cannot read; the next run must rewrite it, not stop on it.
    path = tmp_path / "set.jsonl.zst"
    write_fixture_file(path, render_cases("a", "b"))
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    write_fixture_file(path, render_cases("a", "b"))
    assert list(read_fixture_file(path)) == ["m/render/a", "m/render/b"]


def test_record_keeps_the_old_form_when_the_new_file_cannot_be_written(tmp_path, tiny_model, monkeypatch):
    status, out_dir = record(
        tmp_path, tiny_model, ("bench-x", [{"name": "bench-x-0", "request": {"messages": [user("X")]}}])
    )
    assert status == 0 and (out_dir / "bench-x.jsonl").is_file()
    write_jsonl(tmp_path / "corpus" / "render" / "bench-x.jsonl", [imported("bench-x-0", "X")])

    def refuse(path, cases):
        raise ValueError("schema")

    monkeypatch.setattr("bellwether.record.write_fixture_file", refuse)
    with pytest.raises(ValueError, match="schema"):
        main(record_argv(tmp_path, tiny_model))
    assert (out_dir / "bench-x.jsonl").is_file()


def test_unpack_everything_again_drops_a_model_the_fixtures_no_longer_have(tmp_path, tiny_model, monkeypatch):
    monkeypatch.setattr("bellwether.unpack.fetch", lambda paths: None)
    record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    fixtures, out = tmp_path / "fixtures", tmp_path / "plain"
    write_manifest(fixtures, "other", "acme/Other")
    write_fixture_file(fixtures / "other" / "render" / "common.jsonl", render_cases("a"))
    assert main(["unpack", "--fixtures", str(fixtures), "--out", str(out)]) == 0
    assert (out / "other" / "render" / "common.jsonl").is_file()
    shutil.rmtree(fixtures / "other")
    (out / "notes").mkdir()
    (out / "notes" / "todo.txt").write_text("keep")

    assert main(["unpack", "--fixtures", str(fixtures), "--out", str(out)]) == 0

    assert not (out / "other").exists()
    assert (out / "tiny-chat" / "render" / "common.jsonl").is_file()
    assert (out / "notes" / "todo.txt").read_text() == "keep"


def test_unpack_refuses_to_replace_a_directory_it_did_not_write(tmp_path, tiny_model, monkeypatch, capsys):
    monkeypatch.setattr("bellwether.unpack.fetch", lambda paths: None)
    record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    out = tmp_path / "other"
    (out / "tiny-chat").mkdir(parents=True)
    (out / "tiny-chat" / "notes.txt").write_text("notes")

    assert main(["unpack", "--fixtures", str(tmp_path / "fixtures"), "--out", str(out)]) == 1

    assert "did not write" in capsys.readouterr().err
    assert (out / "tiny-chat" / "notes.txt").read_text() == "notes"
    assert not (out / "tiny-chat" / "render").exists()


NOT_BELLWETHER = 'name = "not bellwether"\n'


def test_unpack_everything_leaves_a_directory_whose_manifest_toml_is_not_a_bellwether_manifest(
    tmp_path, tiny_model, monkeypatch
):
    monkeypatch.setattr("bellwether.unpack.fetch", lambda paths: None)
    record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    out = tmp_path / "shared"
    (out / "some-project").mkdir(parents=True)
    (out / "some-project" / "manifest.toml").write_text(NOT_BELLWETHER)

    assert main(["unpack", "--fixtures", str(tmp_path / "fixtures"), "--out", str(out)]) == 0

    assert (out / "some-project" / "manifest.toml").read_text() == NOT_BELLWETHER
    assert (out / "tiny-chat" / "render" / "common.jsonl").is_file()


def test_unpack_refuses_to_replace_a_directory_whose_manifest_toml_is_not_a_bellwether_manifest(
    tmp_path, tiny_model, monkeypatch, capsys
):
    monkeypatch.setattr("bellwether.unpack.fetch", lambda paths: None)
    record(tmp_path, tiny_model, ("common", [{"name": "a", "request": {"messages": [user("A")]}}]))
    out = tmp_path / "shared"
    (out / "tiny-chat").mkdir(parents=True)
    (out / "tiny-chat" / "manifest.toml").write_text(NOT_BELLWETHER)

    assert main(["unpack", "--fixtures", str(tmp_path / "fixtures"), "--out", str(out)]) == 1

    assert "did not write" in capsys.readouterr().err
    assert (out / "tiny-chat" / "manifest.toml").read_text() == NOT_BELLWETHER
    assert not (out / "tiny-chat" / "render").exists()
