"""What every test shares: no network. Tests read files and fakes; a connection attempt fails at once.

The refusal is a ``RuntimeError``, not an ``OSError``, so no client mistakes it for a passing network
failure and carries on: ``httpx`` maps only ``OSError`` to its connection errors, and the code under
test turns those into statuses (``tests/test_network.py`` proves the refusal).

A test marked ``loopback`` may reach a server it runs itself on this machine: it may connect to
127.0.0.1 and ::1, written as numbers, and resolve those two. Every other address and every name,
``localhost`` included, stay refused for it too.

A test that needs a checkpoint takes ``tiny_model``: a tokenizer and a chat template saved as a checkpoint ships
them.
"""

from __future__ import annotations

import ipaddress
import json
import pathlib
import socket
from collections.abc import Callable

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

LOOPBACK = (ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1"))


class NetworkRefused(RuntimeError):
    """A test tried to reach the network."""


def is_loopback(host: object) -> bool:
    """Whether ``host`` is 127.0.0.1 or ::1 written as numbers; a name never is, whatever it resolves to."""
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    try:
        return ipaddress.ip_address(host) in LOOPBACK
    except ValueError:
        return False


def host_of(address: object) -> object:
    """The host of a socket address: the first item of an IPv4 or IPv6 tuple, else None."""
    return address[0] if isinstance(address, tuple) and address else None


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    loopback = request.node.get_closest_marker("loopback") is not None

    def refuse(*args: object, **kwargs: object) -> None:
        address = next((arg for arg in args if not isinstance(arg, socket.socket)), kwargs)
        raise NetworkRefused(f"no network in tests: {address!r}")

    def guard(original: Callable, host: Callable[..., object]) -> Callable:
        def call(*args: object, **kwargs: object) -> object:
            if loopback and is_loopback(host(*args, **kwargs)):
                return original(*args, **kwargs)
            return refuse(*args, **kwargs)

        return call

    monkeypatch.setattr(socket.socket, "connect", guard(socket.socket.connect, lambda sock, address: host_of(address)))
    monkeypatch.setattr(
        socket.socket, "connect_ex", guard(socket.socket.connect_ex, lambda sock, address: host_of(address))
    )
    monkeypatch.setattr(
        socket, "create_connection", guard(socket.create_connection, lambda address, *a, **k: host_of(address))
    )
    monkeypatch.setattr(socket, "getaddrinfo", guard(socket.getaddrinfo, lambda host, *a, **k: host))


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
