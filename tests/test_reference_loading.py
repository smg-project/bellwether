import json
import pathlib
import shutil

import pytest
import transformers
from huggingface_hub.errors import StrictDataclassFieldValidationError

from bellwether.record import reference, vendor
from bellwether.record.reference import HfTemplateOracle

REQUEST = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}


@pytest.fixture
def checkpoint(tiny_model, tmp_path) -> pathlib.Path:
    """A copy of the tiny checkpoint, to change one file of."""
    directory = tmp_path / "checkpoint"
    shutil.copytree(tiny_model, directory)
    return directory


@pytest.fixture
def legacy_llama_model(tmp_path) -> pathlib.Path:
    """A checkpoint of transformers' Llama class whose ``tokenizer.json`` keeps the legacy pipeline: the ``Prepend``
    normalizer the class replaces with a ``Metaspace`` pre-tokenizer when it builds its own backend."""
    from tokenizers import Tokenizer, decoders, models, normalizers, trainers

    directory = tmp_path / "legacy-llama"
    directory.mkdir()
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Prepend("\u2581"), normalizers.Replace(" ", "\u2581")])
    tokenizer.decoder = decoders.Sequence(
        [decoders.Replace("\u2581", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
    )
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ?.,\n\u2581"),
    )
    sentences = ["system user assistant What is the capital of France? Paris. The quick brown fox"] * 4
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(str(directory / "tokenizer.json"))
    template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    config = {
        "tokenizer_class": "LlamaTokenizer",
        "chat_template": template,
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "add_bos_token": False,
        "add_prefix_space": True,
        "legacy": False,
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(config))
    return directory


def test_a_llama_class_checkpoint_with_the_legacy_file_pipeline_is_encoded_by_the_class(legacy_llama_model):
    """transformers' Llama class does not run such a file: its own pipeline gives ``user`` after ``<|im_start|>``
    where the file's ``Prepend`` gives ``\u2581user``; the reference's ids are the class's, and the provenance says
    so."""
    from tokenizers import Tokenizer

    oracle = HfTemplateOracle(str(legacy_llama_model), "local")
    assert type(oracle.tokenizer).__name__ == "LlamaTokenizer"
    rendered = oracle.render(REQUEST)
    encoded = oracle.tokenizer.apply_chat_template(REQUEST["messages"], tokenize=True, add_generation_prompt=True)
    class_ids = list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)
    file_ids = (
        Tokenizer.from_file(str(legacy_llama_model / "tokenizer.json"))
        .encode(rendered.text, add_special_tokens=False)
        .ids
    )
    assert rendered.input_ids == class_ids
    assert rendered.input_ids != file_ids
    assert oracle.tokenizer.convert_ids_to_tokens(rendered.input_ids[:2]) == ["<|im_start|>", "user"]
    assert oracle.tokenizer.convert_ids_to_tokens(file_ids[:2]) == ["<|im_start|>", "\u2581user"]
    assert oracle.provenance()["encoder"] == "transformers:LlamaTokenizer"


@pytest.mark.parametrize(
    ("model_type", "encoder"),
    [
        # Phi-3-mini-4k-instruct, Phi-tiny-MoE-instruct: the hub names LlamaTokenizer, transformers lists the type among
        # those whose hub class is wrong and registers the tokenizers backend for it, so AutoTokenizer, and the engine
        # with it, runs the file's pipeline (the prefix space stays after an added token).
        pytest.param("phi3", "tokenizers:tokenizer.json", id="listed-type-keeps-the-file"),
        # A type transformers does not know (Phi-4-multimodal's, MiniCPM4's): the named class, as before.
        pytest.param("x-vendor", "transformers:LlamaTokenizer", id="unknown-type-keeps-the-named-class"),
    ],
)
def test_a_vendor_config_names_the_llama_class_and_the_model_type_decides_the_encoder(
    legacy_llama_model, model_type, encoder
):
    """Behind a vendor's model config the recorder chooses the class from the names ``AutoTokenizer`` reads, the hub's
    ``tokenizer_class`` AND the config's ``model_type``: for a type whose hub class transformers overrides the ids are
    the file's, not the class's rebuilt pipeline's, and the provenance says which."""
    from tokenizers import Tokenizer

    config = {"model_type": model_type, "auto_map": {"AutoConfig": "configuration_x.XConfig"}}
    (legacy_llama_model / "config.json").write_text(json.dumps(config))
    oracle = HfTemplateOracle(str(legacy_llama_model), "local")
    rendered = oracle.render(REQUEST)
    file_ids = (
        Tokenizer.from_file(str(legacy_llama_model / "tokenizer.json"))
        .encode(rendered.text, add_special_tokens=False)
        .ids
    )
    assert oracle.provenance()["encoder"] == encoder
    if encoder == "tokenizers:tokenizer.json":
        assert type(oracle.tokenizer).__name__ == "TokenizersBackend"
        assert rendered.input_ids == file_ids
        assert oracle.tokenizer.convert_ids_to_tokens(rendered.input_ids[:2]) == ["<|im_start|>", "\u2581user"]
    else:
        assert type(oracle.tokenizer).__name__ == "LlamaTokenizer"
        assert rendered.input_ids != file_ids
        assert oracle.tokenizer.convert_ids_to_tokens(rendered.input_ids[:2]) == ["<|im_start|>", "user"]


@pytest.mark.parametrize(
    ("named", "model_type", "chosen"),
    [
        pytest.param("LlamaTokenizer", "phi3", "TokenizersBackend", id="listed-type-registers-a-backend"),
        pytest.param(
            "LlamaTokenizerFast", "deepseek_v3", "TokenizersBackend", id="fast-suffix-dropped-before-the-compare"
        ),
        pytest.param("GPT2Tokenizer", "minimax_m2", "TokenizersBackend", id="any-name-yields-to-the-backend"),
        pytest.param("LlamaTokenizer", "qwen2", "Qwen2Tokenizer", id="listed-type-registers-its-own-class"),
        pytest.param("Qwen2Tokenizer", "qwen2", "Qwen2Tokenizer", id="registered-equals-named"),
        pytest.param("LlamaTokenizer", "llama", "LlamaTokenizer", id="type-without-a-registered-class"),
        pytest.param("LlamaTokenizer", "x-vendor", "LlamaTokenizer", id="unknown-type"),
        pytest.param("LlamaTokenizer", None, "LlamaTokenizer", id="no-config"),
        pytest.param("PreTrainedTokenizer", "x-vendor", "TokenizersBackend", id="python-base"),
    ],
)
def test_the_class_auto_tokenizer_picks_for_a_named_class_and_a_model_type(named, model_type, chosen):
    """The table of ``tokenization_auto.py`` at transformers 5.17.0: the registered class of the model type overrides
    the hub's name for the types transformers lists, the name decides elsewhere."""
    assert reference.auto_tokenizer_class(named, model_type).__name__ == chosen


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


@pytest.mark.parametrize(
    "named",
    [
        pytest.param("PreTrainedTokenizer", id="python-base"),  # NVIDIA-Nemotron-Nano-9B-v2, Ling-lite-1.5, Ultravox
        pytest.param("XVendorTokenizer", id="unknown-to-transformers"),
    ],
)
def test_a_vendor_config_loads_the_tokenizer_class_auto_tokenizer_picks_for_the_named_one(checkpoint, named):
    """``AutoTokenizer`` does not load a named ``PreTrainedTokenizer`` as it is: that is the Python base, with no
    vocabulary of its own, and a name transformers does not ship is no class at all. Both load the ``tokenizers``
    backend from ``tokenizer.json`` (``tokenization_auto.py`` at transformers 5.17.0), and so does the recorder when
    the model config is the vendor's class."""
    config = {"model_type": "x-vendor", "auto_map": {"AutoConfig": "configuration_x.XConfig"}}
    (checkpoint / "config.json").write_text(json.dumps(config))
    tokenizer_config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    (checkpoint / "tokenizer_config.json").write_text(json.dumps({**tokenizer_config, "tokenizer_class": named}))
    oracle = HfTemplateOracle(str(checkpoint), "local")
    assert type(oracle.tokenizer).__name__ == "TokenizersBackend"
    rendered = oracle.render(REQUEST)
    assert rendered.text.startswith("<|im_start|>user\nWhat is the capital of France?<|im_end|>")


@pytest.mark.parametrize(
    "config",
    [
        # ZAYA1-reasoning-base: ZayaConfig takes "rope_scaling": false for its rope_parameters, a field false does not
        # fit (a StrictDataclassFieldValidationError).
        pytest.param({"model_type": "zaya", "rope_scaling": False}, id="field"),
        # kanana-nano-2.1b-instruct: validate_architecture refuses 24 heads over a hidden size of 1792, though head_dim
        # is set (a StrictDataclassClassValidationError).
        pytest.param(
            {"model_type": "llama", "hidden_size": 1792, "num_attention_heads": 24, "head_dim": 128}, id="class"
        ),
        # olmo3-7b-sdf-sft-clean150: Olmo3Config takes one set of rope parameters for one per layer type, and
        # validate_rope fails on it with an AttributeError, which no strict-dataclass error wraps.
        pytest.param(
            {"model_type": "olmo3", "rope_parameters": {"rope_type": "default", "rope_theta": 500000}}, id="attribute"
        ),
    ],
)
def test_a_model_config_transformers_refuses_does_not_stop_the_tokenizer(checkpoint, config):
    (checkpoint / "config.json").write_text(json.dumps(config))
    rendered = HfTemplateOracle(str(checkpoint), "local").render(REQUEST)
    assert rendered.text.startswith("<|im_start|>user\nWhat is the capital of France?<|im_end|>")


@pytest.mark.parametrize(
    "named", [pytest.param("GemmaTokenizerFast", id="named"), pytest.param(None, id="by-model-type")]
)
def test_a_refused_model_config_loads_the_tokenizer_class_its_model_type_picks(checkpoint, named):
    """``AutoTokenizer`` chooses the class by the model config's type as well as by the class ``tokenizer_config.json``
    names, and by the type alone when it names none (ZAYA1's type picks ``GemmaTokenizer``); the recorder gives it the
    same choice for a config of that type that transformers refuses."""
    tokenizer_config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    tokenizer_config.pop("tokenizer_class")
    if named is not None:
        tokenizer_config["tokenizer_class"] = named
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "zaya"}))
    picked = type(transformers.AutoTokenizer.from_pretrained(str(checkpoint)))
    assert picked.__name__ == "GemmaTokenizer"
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "zaya", "rope_scaling": False}))
    assert type(reference.load_tokenizer(str(checkpoint), "local", {})) is picked


def test_an_error_the_model_config_does_not_cause_is_the_tokenizers_own(checkpoint, monkeypatch):
    def refuse(model, **kwargs):
        raise AttributeError("the tokenizer's own")

    (checkpoint / "config.json").write_text(json.dumps({"model_type": "llama"}))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", refuse)
    with pytest.raises(AttributeError, match="the tokenizer's own"):
        reference.load_tokenizer(str(checkpoint), "local", {})


def test_inside_the_sandbox_a_refused_model_config_stops_the_tokenizer(checkpoint):
    # The sandbox loads what AutoTokenizer loads, as it builds the config; the stand-in config is for loads outside it.
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "zaya", "rope_scaling": False}))
    with pytest.raises(StrictDataclassFieldValidationError):
        reference.load_tokenizer(str(checkpoint), "local", {"trust_remote_code": True})


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
