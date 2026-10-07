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
