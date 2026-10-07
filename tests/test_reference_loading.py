import json
import pathlib
import shutil

import pytest
import transformers

from bellwether.record import reference, vendor
from bellwether.record.reference import HfTemplateOracle

REQUEST = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}


@pytest.fixture
def checkpoint(tiny_model, tmp_path) -> pathlib.Path:
    """A copy of the tiny checkpoint, to change one file of."""
    directory = tmp_path / "checkpoint"
    shutil.copytree(tiny_model, directory)
    return directory


def test_a_template_only_the_processor_ships_renders(checkpoint):
    # Qwen3-Omni ships its template in chat_template.json, the processor's file, and none in tokenizer_config.json;
    # transformers' processor and vLLM read it from there.
    config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    template = config.pop("chat_template")
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(config))
    (checkpoint / "chat_template.json").write_text(json.dumps({"chat_template": template}))
    rendered = HfTemplateOracle(str(checkpoint), "local").render(REQUEST)
    assert rendered.text.startswith("<|im_start|>user\nWhat is the capital of France?<|im_end|>")


def test_a_model_config_that_needs_the_vendors_code_does_not_stop_the_named_tokenizer(checkpoint):
    # Phi-4-multimodal: its config is the vendor's class (auto_map), which bellwether never runs, and without it
    # transformers cannot build the config its rope settings need; its tokenizer is a class transformers ships.
    (checkpoint / "configuration_x.py").write_text("import pathlib\npathlib.Path(__file__).with_name('ran').touch()\n")
    config = {
        "model_type": "x-vendor",
        "auto_map": {"AutoConfig": "configuration_x.XConfig"},
        "max_position_embeddings": 131072,
        "original_max_position_embeddings": 4096,
        "rope_scaling": {"type": "longrope", "long_factor": [1.0, 1.1], "short_factor": [1.0, 1.0]},
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    rendered = HfTemplateOracle(str(checkpoint), "local").render(REQUEST)
    assert rendered.text.startswith("<|im_start|>user\nWhat is the capital of France?<|im_end|>")
    assert not (checkpoint / "ran").exists()


def test_inside_the_sandbox_the_tokenizer_is_auto_tokenizers_whatever_the_model_config(checkpoint, monkeypatch):
    # With trust_remote_code the vendor's config may be built, so the named-class path, for loads that may not run the
    # vendor's code, is not taken and reads none of the checkpoint's configs.
    consulted = []
    monkeypatch.setattr(reference, "vendor_config", lambda *args: consulted.append(args) or True)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda model, **kwargs: ("auto", model, kwargs))
    loaded = reference.load_tokenizer(str(checkpoint), "local", {"trust_remote_code": True})
    assert loaded == ("auto", str(checkpoint), {"trust_remote_code": True})
    assert consulted == []


def test_a_vendor_class_without_a_template_keeps_none_though_the_processor_ships_one(checkpoint, monkeypatch):
    # A vendor's class may render its chat format itself (Kimi-K3's does); a processor template set on it would change
    # what it renders, so the vendor-code oracle leaves the class as the checkpoint ships it.
    monkeypatch.setenv(vendor.SANDBOX_ENV, "1")
    config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    template = config.pop("chat_template")
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(config))
    (checkpoint / "chat_template.json").write_text(json.dumps({"chat_template": template}))
    oracle = HfTemplateOracle(str(checkpoint), "local", vendor_code=True)
    assert oracle.tokenizer.chat_template is None
    assert oracle.template_sha256 is None
