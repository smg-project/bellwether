"""The suite cannot reach a network: ``conftest.py`` refuses every connection, even to this machine.

A server listening on the loopback interface stands in for the Hub, so the proof itself sends
nothing off this machine, and a refused connection is told apart from one that got through.
"""

from __future__ import annotations

import socket

import httpx
import pytest


def listening() -> socket.socket:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    return server


def test_a_test_cannot_open_a_connection_even_to_this_machine() -> None:
    with listening() as server:
        with pytest.raises(RuntimeError, match="network"):
            socket.create_connection(server.getsockname(), timeout=1)
        with socket.socket() as client, pytest.raises(RuntimeError, match="network"):
            client.connect(server.getsockname())


def test_an_http_client_cannot_reach_a_server() -> None:
    with listening() as server:
        host, port = server.getsockname()
        with pytest.raises(RuntimeError, match="network"):
            httpx.get(f"http://{host}:{port}/api/models/Qwen/Qwen3-8B", timeout=1)


def test_a_test_cannot_resolve_a_name() -> None:
    with pytest.raises(RuntimeError, match="network"):
        socket.getaddrinfo("localhost", 443)
