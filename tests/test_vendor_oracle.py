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
