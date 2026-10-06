"""What every test shares: no network. Tests read files and fakes; a connection attempt fails at once.

The refusal is a ``RuntimeError``, not an ``OSError``, so no client mistakes it for a passing network
failure and carries on: ``httpx`` maps only ``OSError`` to its connection errors, and the code under
test turns those into statuses (``tests/test_network.py`` proves the refusal).
"""

from __future__ import annotations

import socket

import pytest


class NetworkRefused(RuntimeError):
    """A test tried to reach the network."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        address = next((arg for arg in args if not isinstance(arg, socket.socket)), kwargs)
        raise NetworkRefused(f"no network in tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
