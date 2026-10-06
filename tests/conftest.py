"""What every test shares: no network. Tests read files and fakes; a connection attempt fails at once.

The refusal is a ``RuntimeError``, not an ``OSError``, so no client mistakes it for a passing network
failure and carries on: ``httpx`` maps only ``OSError`` to its connection errors, and the code under
test turns those into statuses (``tests/test_network.py`` proves the refusal).

A test marked ``loopback`` may reach a server it runs itself on this machine: it may connect to
127.0.0.1 and ::1, written as numbers, and resolve those two. Every other address and every name,
``localhost`` included, stay refused for it too.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable

import pytest

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
