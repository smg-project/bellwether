import json
import pathlib

import pytest

from bellwether.record import vendor
from bellwether.record.reference import HfTemplateOracle

# The vendor's tokenizer class, as a checkpoint ships one: one token per UTF-8 byte, and three special tokens. On import
# it leaves a file beside itself, so a test can tell whether bellwether ran it.
VENDOR_CODE = """
import pathlib

from transformers.tokenization_utils import PreTrainedTokenizer

pathlib.Path(__file__).with_name("imported").write_text("yes")

SPECIAL = ["<|im_start|>", "<|im_end|>", "<unk>"]


class ByteTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]
    vocab_files_names = {}

    @property
    def vocab_size(self):
        return 3 + 256

    def get_vocab(self):
        vocab = {token: index for index, token in enumerate(SPECIAL)}
        vocab.update({f"<0x{byte:02X}>": 3 + byte for byte in range(256)})
        return vocab

    def _tokenize(self, text):
        return [f"<0x{byte:02X}>" for byte in text.encode("utf-8")]

    def _convert_token_to_id(self, token):
        return SPECIAL.index(token) if token in SPECIAL else 3 + int(token[3:5], 16)

    def _convert_id_to_token(self, index):
        return SPECIAL[index] if index < 3 else f"<0x{index - 3:02X}>"

    def convert_tokens_to_string(self, tokens):
        text, held = [], bytearray()
        for token in tokens:
            if token in SPECIAL:
                text.append(held.decode("utf-8", errors="replace"))
                held = bytearray()
                text.append(token)
            else:
                held.append(int(token[3:5], 16))
        text.append(held.decode("utf-8", errors="replace"))
        return "".join(text)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


class EncoderTokenizer(ByteTokenizer):
    # Renders its chat format itself, with no template, as Kimi-K3's class does.

    def apply_chat_template(self, conversation, tools=None, tokenize=True, add_generation_prompt=False, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\\n{m['content']}<|im_end|>\\n" for m in conversation)
        if add_generation_prompt:
            text += "<|im_start|>assistant\\n"
        return self.encode(text, add_special_tokens=False) if tokenize else text


class ThinkingTokenizer(ByteTokenizer):
    # Reads the thinking kwargs as Kimi-K3's class reads them (`thinking`, `thinking_effort`, the effort defaulting to
    # max and checked against the class's three), and writes what it read into the prompt; every other kwarg is ignored.

    def apply_chat_template(
        self, conversation, tools=None, tokenize=True, add_generation_prompt=False, thinking=True, **kwargs
    ):
        kwargs.setdefault("thinking_effort", "max")
        effort = kwargs["thinking_effort"]
        assert effort in ("low", "high", "max"), f"Unsupported thinking_effort={effort!r}"
        text = f"<|effort|>{effort}\\n" if thinking else ""
        for m in conversation:
            text += f"<|im_start|>{m['role']}\\n{m['content']}<|im_end|>\\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\\n" + ("<think>" if thinking else "")
        return self.encode(text, add_special_tokens=False) if tokenize else text
"""
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture
def vendor_checkpoint(tmp_path, monkeypatch) -> pathlib.Path:
    """A checkpoint whose tokenizer class is the vendor's code, named in auto_map."""
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "modules"))
    directory = tmp_path / "vendor-chat"
    directory.mkdir()
    (directory / "tokenization_byte.py").write_text(VENDOR_CODE)
    config = {
        "auto_map": {"AutoTokenizer": ["tokenization_byte.ByteTokenizer", None]},
        "tokenizer_class": "ByteTokenizer",
        "chat_template": TEMPLATE,
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "added_tokens_decoder": {
            str(index): {"content": token, "special": True, "lstrip": False, "rstrip": False, "normalized": False}
            for index, token in enumerate(["<|im_start|>", "<|im_end|>", "<unk>"])
        },
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(config))
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    return directory


REQUEST = {"messages": [{"role": "user", "content": "café"}]}
KIMI_K3 = "KimiK3ForConditionalGeneration"  # the architecture the engine's server renders with its Kimi K3 renderer


def with_thinking_class(checkpoint: pathlib.Path, architecture: str) -> None:
    """Make the checkpoint's class the one that reads the thinking kwargs, with no template, under ``architecture``."""
    config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    config["auto_map"] = {"AutoTokenizer": ["tokenization_byte.ThinkingTokenizer", None]}
    config["tokenizer_class"] = "ThinkingTokenizer"
    del config["chat_template"]
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(config))
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "x", "architectures": [architecture]}))


@pytest.fixture
def kimi_k3_checkpoint(vendor_checkpoint, monkeypatch) -> pathlib.Path:
    monkeypatch.setenv(vendor.SANDBOX_ENV, "1")
    with_thinking_class(vendor_checkpoint, KIMI_K3)
    return vendor_checkpoint


def without_a_template(checkpoint: pathlib.Path) -> None:
    """Make the checkpoint's class one that renders its chat format itself, and drop the template."""
    config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    config["auto_map"] = {"AutoTokenizer": ["tokenization_byte.EncoderTokenizer", None]}
    config["tokenizer_class"] = "EncoderTokenizer"
    del config["chat_template"]
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(config))


def test_in_the_sandbox_a_vendor_class_that_renders_itself_needs_no_template(vendor_checkpoint, monkeypatch):
    monkeypatch.setenv(vendor.SANDBOX_ENV, "1")
    without_a_template(vendor_checkpoint)
    oracle = HfTemplateOracle(str(vendor_checkpoint), "local", vendor_code=True)
    rendered = oracle.render(REQUEST)
    assert rendered.text == "<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n"
    assert rendered.input_ids[0] == 0
    assert "chat_template_sha256" not in oracle.provenance()


def test_outside_the_sandbox_the_vendor_oracle_refuses_before_any_vendor_code_runs(vendor_checkpoint, monkeypatch):
    monkeypatch.delenv(vendor.SANDBOX_ENV, raising=False)
    with pytest.raises(vendor.OutsideTheSandbox, match="runs the vendor's code only inside the sandbox"):
        HfTemplateOracle(str(vendor_checkpoint), "local", vendor_code=True)
    assert not (vendor_checkpoint / "imported").exists()


def test_in_the_sandbox_the_vendor_tokenizer_renders(vendor_checkpoint, monkeypatch):
    monkeypatch.setenv(vendor.SANDBOX_ENV, "1")
    oracle = HfTemplateOracle(str(vendor_checkpoint), "local", vendor_code=True)
    rendered = oracle.render(REQUEST)
    assert rendered.text == "<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n"
    assert rendered.input_ids[0] == 0 and rendered.input_ids.count(1) == 1
    assert len(rendered.input_ids) == 3 + len("user\ncafé\n".encode()) + len(b"assistant\n")
    assert oracle.provenance()["tokenizer_class"] == "ByteTokenizer"
    assert oracle.provenance()["oracle"] == "vendor-code"


@pytest.mark.parametrize(
    ("request_fields", "rendered"),
    [
        # The engine's alias becomes the class's own kwarg: thinking off, no effort directive, the response opener.
        pytest.param(
            {"chat_template_kwargs": {"enable_thinking": False}},
            "<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n",
            id="enable-thinking-off",
        ),
        pytest.param(
            {"chat_template_kwargs": {"enable_thinking": True}},
            "<|effort|>max\n<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n<think>",
            id="enable-thinking-on",
        ),
        # A reasoning effort in the kwargs is the class's thinking effort; `none` switches thinking off.
        pytest.param(
            {"chat_template_kwargs": {"reasoning_effort": "low"}},
            "<|effort|>low\n<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n<think>",
            id="reasoning-effort-low",
        ),
        pytest.param(
            {"chat_template_kwargs": {"reasoning_effort": "none"}},
            "<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n",
            id="reasoning-effort-none",
        ),
        # The request's top-level reasoning_effort reaches the kwargs too, over one of the same name in them.
        pytest.param(
            {"reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": "low"}},
            "<|effort|>high\n<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n<think>",
            id="top-level-reasoning-effort",
        ),
        pytest.param(
            {"reasoning_effort": "none"},
            "<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n",
            id="top-level-reasoning-effort-none",
        ),
        # The class's own kwargs win over the engine's spellings.
        pytest.param(
            {
                "chat_template_kwargs": {
                    "thinking": True,
                    "enable_thinking": False,
                    "thinking_effort": "low",
                    "reasoning_effort": "high",
                }
            },
            "<|effort|>low\n<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n<think>",
            id="the-class-kwargs-win",
        ),
    ],
)
def test_in_the_sandbox_a_kimi_k3_checkpoint_gets_the_kwargs_as_the_engines_renderer_hands_them(
    kimi_k3_checkpoint, request_fields, rendered
):
    oracle = HfTemplateOracle(str(kimi_k3_checkpoint), "local", vendor_code=True)
    assert oracle.render({**REQUEST, **request_fields}).text == rendered
    assert oracle.provenance()["engine_renderer"] == "kimi_k3"


def test_in_the_sandbox_a_kimi_k3_checkpoint_refuses_an_effort_the_engines_renderer_refuses(kimi_k3_checkpoint):
    oracle = HfTemplateOracle(str(kimi_k3_checkpoint), "local", vendor_code=True)
    with pytest.raises(ValueError, match="Kimi K3 supports thinking_effort values: low, high, max"):
        oracle.render({**REQUEST, "chat_template_kwargs": {"thinking_effort": "medium"}})
    with pytest.raises(ValueError, match="thinking_effort"):
        oracle.render({**REQUEST, "reasoning_effort": "medium"})


def test_in_the_sandbox_a_vendor_class_of_another_architecture_gets_the_kwargs_as_the_request_gives_them(
    vendor_checkpoint, monkeypatch
):
    """The engine's server renders every other vendor-code checkpoint through the class with the request's kwargs:
    the alias reaches a class that does not read it, and thinking stays on."""
    monkeypatch.setenv(vendor.SANDBOX_ENV, "1")
    with_thinking_class(vendor_checkpoint, "OtherForCausalLM")
    oracle = HfTemplateOracle(str(vendor_checkpoint), "local", vendor_code=True)
    rendered = oracle.render({**REQUEST, "chat_template_kwargs": {"enable_thinking": False}})
    assert rendered.text == "<|effort|>max\n<|im_start|>user\ncafé<|im_end|>\n<|im_start|>assistant\n<think>"
    assert "engine_renderer" not in oracle.provenance()


def test_in_the_sandbox_the_round_trip_of_a_kimi_k3_checkpoint_renders_with_the_engines_kwargs(kimi_k3_checkpoint):
    """The parse oracle's prompt and turn come from the same rewrite: under the engine's alias the generation prompt
    opens no thought, and the output is the message alone."""
    from bellwether.record.roundtrip import RoundtripOracle

    oracle = RoundtripOracle(str(kimi_k3_checkpoint), "local", vendor_code=True)
    request = {**REQUEST, "chat_template_kwargs": {"enable_thinking": False}}
    output = oracle.render_output(request, {"content": "Paris."})
    assert output.text == "Paris."
    assert oracle.provenance()["engine_renderer"] == "kimi_k3"


def test_without_the_vendor_flag_a_vendor_tokenizer_is_never_loaded(vendor_checkpoint, monkeypatch):
    monkeypatch.setenv(vendor.SANDBOX_ENV, "1")
    with pytest.raises(ValueError):
        HfTemplateOracle(str(vendor_checkpoint), "local")
    assert not (vendor_checkpoint / "imported").exists()


class Bytes:
    """A tokenizer that decodes ids as UTF-8 bytes, as a byte-level vocabulary does."""

    def decode(self, ids, **_):
        return bytes(ids).decode("utf-8", errors="replace")


def test_incremental_pieces_give_back_the_text_and_hold_a_split_character_until_it_is_whole():
    text = "aé b"
    ids = list(text.encode("utf-8"))
    pieces = vendor.incremental_pieces(Bytes(), ids)
    assert pieces == ["a", "", "é", " ", "b"]
    assert "".join(pieces) == text


def record(tmp_path, checkpoint, kind: str, lines: list[dict]) -> tuple[int, pathlib.Path]:
    """``bellwether record --oracle vendor`` over one corpus set, with a manifest listing the checkpoint's inputs."""
    from bellwether.cli import main
    from bellwether.inputs import oracle_inputs

    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    manifest = fixtures / "vendor-chat" / "manifest.toml"
    if not manifest.exists():
        manifest.parent.mkdir(parents=True)
        inputs = oracle_inputs(str(checkpoint), "local")
        text = [f'model = "{checkpoint}"', 'revision = "local"', "", "[authority]", 'render = ["vendor-code"]']
        text += ["", "[inputs]", *(f'"{name}" = "{digest}"' for name, digest in inputs.items())]
        manifest.write_text("\n".join(text) + "\n")
    (corpus / kind).mkdir(parents=True, exist_ok=True)
    (corpus / kind / "common.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    argv = ["record", "--model", str(checkpoint), "--kind", kind, "--oracle", "vendor"]
    return main([*argv, "--fixtures", str(fixtures), "--corpus", str(corpus)]), fixtures / "vendor-chat" / kind


def test_record_with_the_vendor_oracle_outside_the_sandbox_names_the_command_and_runs_nothing(
    tmp_path, vendor_checkpoint, monkeypatch, capsys
):
    monkeypatch.delenv(vendor.SANDBOX_ENV, raising=False)
    status, out = record(tmp_path, vendor_checkpoint, "render", [{"name": "a", "request": REQUEST}])
    assert status == 1
    assert "bellwether sandbox-record" in capsys.readouterr().err
    assert not out.exists() and not (vendor_checkpoint / "imported").exists()


def test_record_with_the_vendor_oracle_in_the_sandbox_writes_vendor_code_references(
    tmp_path, vendor_checkpoint, monkeypatch
):
    monkeypatch.setenv(vendor.SANDBOX_ENV, "1")
    monkeypatch.setenv(vendor.IMAGE_ENV, "bellwether-vendor@sha256:" + "0" * 64)
    status, out = record(tmp_path, vendor_checkpoint, "render", [{"name": "a", "request": REQUEST}])
    assert status == 0
    [line] = [json.loads(text) for text in (out / "common.jsonl").read_text().splitlines()]
    assert line["reference"]["source"] == "vendor-code"
    assert line["reference"]["text"].startswith("<|im_start|>user\ncafé")
    assert line["reference"]["provenance"]["oracle"] == "vendor-code"
    assert line["reference"]["provenance"]["sandbox_image"].startswith("bellwether-vendor@sha256:")
    message = {"content": "Paris é."}
    status, out = record(tmp_path, vendor_checkpoint, "parse", [{"name": "b", "request": REQUEST, "message": message}])
    assert status == 0
    [line] = [json.loads(text) for text in (out / "common.jsonl").read_text().splitlines()]
    assert line["reference"]["source"] == "roundtrip:vendor-code"
    assert "".join(line["output_pieces"]) == line["reference"]["text"] == "Paris é."
    assert "" in line["output_pieces"]  # the first byte of é waits for the second
