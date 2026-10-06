import hashlib
import json
import os
import pathlib
import re
import subprocess

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from bellwether import unpack as unpack_module
from bellwether.cli import main
from bellwether.manifest import find_manifest, load_manifest, slug_for
from bellwether.record import sets as set_tables
from bellwether.record.chunks import chunk_plans
from bellwether.record.corpus import load_corpus, read_cases
from bellwether.record.fixtures import (
    canonical_line,
    is_lfs_pointer,
    lfs_pull_command,
    plain_text,
    read_fixture_file,
    schema_path,
    validator,
    write_fixture_file,
)
from bellwether.record.reference import HfTemplateOracle
from bellwether.record.roundtrip import RoundtripOracle, as_vllm_gives_it

ROOT = pathlib.Path(__file__).resolve().parents[1]

# A ChatML-shaped template with a thinking switch, enough to see every request field arrive.
TEMPLATE = (
    "{%- if tools %}{{ '<|im_start|>system\\n' + (tools | tojson) + '<|im_end|>\\n' }}{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m['role'] == 'assistant' %}{{ '<|im_start|>assistant\\n' }}"
    "{%- if m['reasoning_content'] %}{{ '<think>\\n' + m['reasoning_content'] + '\\n</think>\\n\\n' }}{%- endif %}"
    "{{ m['content'] or '' }}"
    "{%- for c in (m['tool_calls'] or []) %}"
    "{{ '\\n<tool_call>\\n' + (c['function'] | tojson) + '\\n</tool_call>' }}{%- endfor %}"
    "{{ '<|im_end|>\\n' }}"
    "{%- else %}{{ '<|im_start|>' + m['role'] + '\\n' + (m['content'] or '') + '<|im_end|>\\n' }}{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and not enable_thinking %}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
    "{%- endif %}"
)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> pathlib.Path:
    """A byte-level BPE tokenizer trained on a few sentences, saved the way a checkpoint ships one."""
    directory = tmp_path_factory.mktemp("tiny-chat")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<unk>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    sentences = ["system user assistant What is the capital of France? Paris. The quick brown fox"] * 4
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(str(directory / "tokenizer.json"))
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": TEMPLATE,
        "unk_token": "<unk>",
        "eos_token": "<|im_end|>",
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(config))
    return directory


def write_jsonl(path: pathlib.Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def write_manifest(fixtures: pathlib.Path, slug: str, model: str) -> pathlib.Path:
    path = fixtures / slug / "manifest.toml"
    path.parent.mkdir(parents=True)
    lines = [
        f'model = "{model}"',
        'revision = "local"',
        "",
        "[authority]",
        'render = ["hf-template"]',
        "",
        "[smg]",
        'tool_parser = "qwen"',
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def test_manifest_reads_model_revision_authority_and_names(tmp_path):
    path = write_manifest(tmp_path, "tiny-chat", "acme/Tiny-Chat")
    manifest = load_manifest(path)
    assert (manifest.slug, manifest.model, manifest.revision) == ("tiny-chat", "acme/Tiny-Chat", "local")
    assert manifest.authority == {"render": ["hf-template"]}
    assert manifest.smg == {"tool_parser": "qwen"}


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
        write_manifest(fixtures, "tiny-chat", str(tiny_model))
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
    lines = [json.loads(line) for line in text.splitlines()]
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


def tiny_variant(tiny_model, tmp_path_factory, name: str, template: str) -> pathlib.Path:
    """The tiny model's tokenizer with another chat template."""
    directory = tmp_path_factory.mktemp(name)
    (directory / "tokenizer.json").write_bytes((tiny_model / "tokenizer.json").read_bytes())
    config = json.loads((tiny_model / "tokenizer_config.json").read_text())
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
        "reference": {"source": "roundtrip", "text": "ab"},
    }
    write_fixture_file(tmp_path / "x.jsonl", {"tiny-chat/parse/a": line})
    for missing in ("output_ids", "output_pieces"):
        partial = {key: value for key, value in line.items() if key != missing}
        with pytest.raises(ValueError, match="tiny-chat/parse/a: does not match the case schema at \\(root\\)"):
            write_fixture_file(tmp_path / "y.jsonl", {"tiny-chat/parse/a": partial})


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
    expected = set_tables.entry(form, plain, table["cases"], table["rejected"])
    assert table == expected, f"{path}: its sets.toml table does not match the file"
    lines = plain.splitlines()
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

    ids = [json.loads(line)["id"] for line in (out_dir / "common.jsonl").read_text().splitlines()]
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
