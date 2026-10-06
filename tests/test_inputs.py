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
        "chat_template.json",
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


OLD, NEW = "1" * 40, "2" * 40


def test_oracle_inputs_read_the_hugging_face_cache_at_the_exact_revision(tmp_path, tiny_model, monkeypatch):
    repo = tmp_path / "hub" / "models--acme--tiny-chat"
    for revision in (OLD, NEW):
        shutil.copytree(tiny_model, repo / "snapshots" / revision)
    (repo / "snapshots" / NEW / "chat_template.jinja").write_text("{{ messages }}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(NEW)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)

    assert oracle_inputs("acme/tiny-chat", OLD) == oracle_inputs(str(tiny_model), "local")
    assert "chat_template.jinja" in oracle_inputs("acme/tiny-chat", NEW)


def test_oracle_inputs_name_a_revision_the_offline_cache_does_not_hold(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    with pytest.raises(FileNotFoundError, match=f"acme/tiny-chat at {OLD}: "):
        oracle_inputs("acme/tiny-chat", OLD)
