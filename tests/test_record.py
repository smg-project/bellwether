import json
import os
import pathlib

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from bellwether.cli import main
from bellwether.manifest import find_manifest, load_manifest, slug_for
from bellwether.record.corpus import load_corpus, read_cases
from bellwether.record.fixtures import canonical_line, read_fixture_file, schema_path, validator, write_fixture_file
from bellwether.record.reference import HfTemplateOracle

ROOT = pathlib.Path(__file__).resolve().parents[1]

# A ChatML-shaped template with a thinking switch, enough to see every request field arrive.
TEMPLATE = (
    "{%- if tools %}{{ '<|im_start|>system\\n' + (tools | tojson) + '<|im_end|>\\n' }}{%- endif %}"
    "{%- for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + (m['content'] or '') + '<|im_end|>\\n' }}"
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
    config = {"tokenizer_class": "PreTrainedTokenizerFast", "chat_template": TEMPLATE, "unk_token": "<unk>"}
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


def record(tmp_path, tiny_model, *corpus_sets: tuple[str, list[dict]]) -> tuple[int, pathlib.Path]:
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    if not (fixtures / "tiny-chat").exists():
        write_manifest(fixtures, "tiny-chat", str(tiny_model))
    for name, lines in corpus_sets:
        write_jsonl(corpus / "render" / f"{name}.jsonl", lines)
    argv = ["record", "--model", str(tiny_model), "--kind", "render", "--oracle", "reference"]
    argv += ["--fixtures", str(fixtures), "--corpus", str(corpus)]
    return main(argv), fixtures / "tiny-chat" / "render"


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


def test_record_other_kinds_and_oracles_are_not_implemented(tmp_path, tiny_model, capsys):
    argv = ["record", "--model", str(tiny_model), "--kind", "render", "--oracle", "sglang", "--fixtures", str(tmp_path)]
    assert main(argv) == 2
    assert "not implemented" in capsys.readouterr().err


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


@pytest.mark.parametrize("path", sorted(ROOT.glob("fixtures/*/*/*.jsonl")), ids=lambda p: str(p.relative_to(ROOT)))
def test_committed_fixtures_are_canonical_sorted_and_valid(path):
    manifest = load_manifest(path.parent.parent / "manifest.toml")
    lines = path.read_text().splitlines()
    cases = [json.loads(line) for line in lines]
    assert [c["id"] for c in cases] == sorted(c["id"] for c in cases)
    for raw, case in zip(lines, cases, strict=True):
        validator().validate(case)
        assert raw == canonical_line(case)
        assert case["id"].startswith(f"{manifest.slug}/{path.parent.name}/")
        assert case["model"] == manifest.model
        assert case["reference"]["provenance"]["revision"] == manifest.revision
