import hashlib
import json
import pathlib
import shutil

import pytest

from bellwether.inputs import oracle_inputs


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def checkpoint(tmp_path, tiny_model) -> pathlib.Path:
    """The tiny model with a generation config, a model config and files the oracle never reads, as a checkpoint
    ships them."""
    directory = tmp_path / "tiny-chat"
    shutil.copytree(tiny_model, directory)
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": 2, "temperature": 0.6}))
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 64}))
    (directory / "model.safetensors").write_bytes(b"weights")
    (directory / "README.md").write_text("A model card.")
    (directory / "modeling_tiny.py").write_text("raise SystemExit('never run')\n")
    return directory


def test_oracle_inputs_are_the_files_the_oracle_reads_each_with_its_sha256(checkpoint):
    inputs = oracle_inputs(str(checkpoint), "local")
    assert list(inputs) == ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
    for name in ("tokenizer.json", "tokenizer_config.json"):
        assert inputs[name] == sha256((checkpoint / name).read_bytes())


def test_every_tokenizer_file_and_chat_template_file_is_an_input(checkpoint):
    names = [
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "chat_template.jinja",
    ]
    for name in names:
        (checkpoint / name).write_text(f"{{}} {name}")
    inputs = oracle_inputs(str(checkpoint), "local")
    for name in names:
        assert inputs[name] == sha256(f"{{}} {name}".encode())


def test_a_narrowed_file_is_hashed_over_the_canonical_json_of_its_fields_absent_ones_as_null(checkpoint):
    inputs = oracle_inputs(str(checkpoint), "local")
    generation = json.dumps({"bos_token_id": None, "eos_token_id": 2, "pad_token_id": None}, sort_keys=True)
    config = json.dumps({"model_type": "qwen3", "tokenizer_class": None}, sort_keys=True)
    assert inputs["generation_config.json"] == sha256(generation.encode())
    assert inputs["config.json"] == sha256(config.encode())


def test_chat_template_json_is_hashed_over_its_template_so_its_whitespace_does_not_split_a_group(checkpoint):
    # No oracle reads the file's bytes: vLLM reads it through the processor, parsed, and takes its chat_template.
    path = checkpoint / "chat_template.json"
    path.write_text(json.dumps({"chat_template": "{{ messages }}"}))
    inputs = oracle_inputs(str(checkpoint), "local")
    assert inputs["chat_template.json"] == sha256(json.dumps({"chat_template": "{{ messages }}"}).encode())
    path.write_text(json.dumps({"chat_template": "{{ messages }}", "processor": "Acme"}, indent=2) + "\n  \n")
    assert oracle_inputs(str(checkpoint), "local") == inputs
    path.write_text(json.dumps({"chat_template": "{{ messages[0] }}"}))
    assert oracle_inputs(str(checkpoint), "local")["chat_template.json"] != inputs["chat_template.json"]


def test_sampling_defaults_leave_the_inputs_as_they_are_and_token_ids_change_them(checkpoint):
    before = oracle_inputs(str(checkpoint), "local")
    generation = checkpoint / "generation_config.json"
    generation.write_text(json.dumps({"eos_token_id": 2, "temperature": 1.0, "top_p": 0.8}))
    assert oracle_inputs(str(checkpoint), "local") == before
    generation.write_text(json.dumps({"eos_token_id": [2, 3], "temperature": 0.6}))
    assert oracle_inputs(str(checkpoint), "local")["generation_config.json"] != before["generation_config.json"]


def test_only_the_fields_that_choose_the_tokenizer_count_in_the_model_config(checkpoint):
    before = oracle_inputs(str(checkpoint), "local")
    config = checkpoint / "config.json"
    config.write_text(json.dumps({"model_type": "qwen3", "hidden_size": 128, "architectures": ["Qwen3ForCausalLM"]}))
    assert oracle_inputs(str(checkpoint), "local") == before
    config.write_text(json.dumps({"model_type": "qwen3", "tokenizer_class": "Qwen2Tokenizer"}))
    assert oracle_inputs(str(checkpoint), "local")["config.json"] != before["config.json"]


def canonical(**fields) -> str:
    return sha256(json.dumps(fields, sort_keys=True).encode())


def test_without_a_generation_config_the_model_config_gives_the_stop_ids_text_config_included(checkpoint):
    # The end of a turn reads generation_config.json, else config.json through GenerationConfig.from_model_config,
    # which takes a value the top level leaves unset from text_config. A model type transformers does not know is read
    # as written: its class is the vendor's code, which never runs.
    (checkpoint / "generation_config.json").unlink()
    (checkpoint / "configuration_acme.py").write_text('raise RuntimeError("the vendor\'s code ran")\n')
    config = {"model_type": "acme_chat", "auto_map": {"AutoConfig": "configuration_acme.AcmeConfig"}}
    (checkpoint / "config.json").write_text(json.dumps({**config, "text_config": {"eos_token_id": 7}}))
    inputs = oracle_inputs(str(checkpoint), "local")
    assert "generation_config.json" not in inputs
    expected = canonical(
        model_type="acme_chat", tokenizer_class=None, bos_token_id=None, eos_token_id=7, pad_token_id=None
    )
    assert inputs["config.json"] == expected


def test_the_stop_ids_of_a_model_type_transformers_knows_carry_its_class_defaults(checkpoint):
    # {"model_type": "llama"} states no token ids; LlamaConfig's are bos 1 and eos 2, which from_model_config gives.
    (checkpoint / "generation_config.json").unlink()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "llama"}))
    expected = canonical(model_type="llama", tokenizer_class=None, bos_token_id=1, eos_token_id=2, pad_token_id=None)
    assert oracle_inputs(str(checkpoint), "local")["config.json"] == expected


def test_the_model_config_s_stop_ids_count_only_where_no_generation_config_gives_them(checkpoint):
    config = checkpoint / "config.json"
    before = oracle_inputs(str(checkpoint), "local")
    config.write_text(json.dumps({"model_type": "qwen3", "hidden_size": 64, "eos_token_id": 9}))
    assert oracle_inputs(str(checkpoint), "local") == before
    (checkpoint / "generation_config.json").unlink()
    without = oracle_inputs(str(checkpoint), "local")["config.json"]
    config.write_text(json.dumps({"model_type": "qwen3", "hidden_size": 64, "eos_token_id": 10}))
    assert oracle_inputs(str(checkpoint), "local")["config.json"] != without


def test_each_named_chat_template_is_an_input_and_nothing_else_in_their_directory(checkpoint):
    # transformers reads every additional_chat_templates/<name>.jinja, and takes tool_use when a request has tools.
    named = checkpoint / "additional_chat_templates"
    (named / "older").mkdir(parents=True)
    (named / "tool_use.jinja").write_text("{{ tools }}")
    (named / "rag.jinja").write_text("{{ documents }}")
    (named / "README.md").write_text("Notes.")
    (named / "older" / "tool_use.jinja").write_text("{{ messages }}")
    inputs = oracle_inputs(str(checkpoint), "local")
    assert [name for name in inputs if name.startswith("additional_chat_templates/")] == [
        "additional_chat_templates/rag.jinja",
        "additional_chat_templates/tool_use.jinja",
    ]
    assert inputs["additional_chat_templates/tool_use.jinja"] == sha256(b"{{ tools }}")


OLD, NEW = "1" * 40, "2" * 40


def cached(hub: pathlib.Path, files: pathlib.Path, revision: str, *, listed: list[str] | None = None) -> pathlib.Path:
    """``files`` as the Hugging Face cache holds a snapshot of acme/tiny-chat at ``revision``, with the commit's file
    list (``trees/<commit>.json``) naming ``listed``, every file of the snapshot by default; an empty ``listed`` leaves
    the list out, as a cache filled one file at a time has none."""
    repo = hub / "models--acme--tiny-chat"
    snapshot = repo / "snapshots" / revision
    shutil.copytree(files, snapshot)
    names = sorted(p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file())
    if listed != []:
        listing = {name: {"size": 1, "blob_id": "0" * 40} for name in (names if listed is None else listed)}
        (repo / "trees").mkdir(exist_ok=True)
        (repo / "trees" / f"{revision}.json").write_text(json.dumps({"format_version": 1, "files": listing}))
    return snapshot


@pytest.fixture
def offline_hub(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    return tmp_path / "hub"


def test_oracle_inputs_read_the_hugging_face_cache_at_the_exact_revision(offline_hub, tiny_model):
    cached(offline_hub, tiny_model, OLD)
    newer = cached(
        offline_hub, tiny_model, NEW, listed=["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]
    )
    (newer / "chat_template.jinja").write_text("{{ messages }}")
    (offline_hub / "models--acme--tiny-chat" / "refs").mkdir()
    (offline_hub / "models--acme--tiny-chat" / "refs" / "main").write_text(NEW)

    assert oracle_inputs("acme/tiny-chat", OLD) == oracle_inputs(str(tiny_model), "local")
    assert "chat_template.jinja" in oracle_inputs("acme/tiny-chat", NEW)


def test_oracle_inputs_name_a_revision_the_offline_cache_does_not_hold(offline_hub):
    with pytest.raises(FileNotFoundError, match=f"acme/tiny-chat at {OLD}: "):
        oracle_inputs("acme/tiny-chat", OLD)


def test_offline_a_snapshot_without_the_commit_s_file_list_is_refused(offline_hub, tiny_model):
    # Filled one file at a time, the cache cannot tell a file it lacks from one the checkpoint does not ship, which
    # would drop out of the inputs unnoticed.
    cached(offline_hub, tiny_model, OLD, listed=[])
    with pytest.raises(
        FileNotFoundError, match=f"acme/tiny-chat at {OLD}: the cache holds no list of the commit's files"
    ):
        oracle_inputs("acme/tiny-chat", OLD)


@pytest.mark.parametrize("missing", ["vocab.json", "additional_chat_templates/tool_use.jinja"])
def test_offline_a_snapshot_that_lacks_a_file_the_commit_lists_is_refused(offline_hub, tiny_model, missing):
    names = ["tokenizer.json", "tokenizer_config.json", missing]
    cached(offline_hub, tiny_model, OLD, listed=names)
    with pytest.raises(FileNotFoundError, match=rf"acme/tiny-chat at {OLD}: .*incomplete.*{missing}"):
        oracle_inputs("acme/tiny-chat", OLD)


def test_offline_a_file_the_commit_does_not_list_is_one_the_checkpoint_does_not_ship(offline_hub, tiny_model):
    cached(offline_hub, tiny_model, OLD, listed=["tokenizer.json", "tokenizer_config.json", "model.safetensors"])
    assert oracle_inputs("acme/tiny-chat", OLD) == oracle_inputs(str(tiny_model), "local")
