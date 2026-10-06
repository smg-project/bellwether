"""The suite cannot reach a network: ``conftest.py`` refuses every connection, even to this machine.

A server listening on the loopback interface stands in for the Hub, so the proof itself sends
nothing off this machine, and a refused connection is told apart from one that got through.

A test marked ``loopback`` may reach a server it runs itself at 127.0.0.1 or ::1, as the verify tests
reach their stand-in for SMG; every other address and every name stay refused for it too.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest


def listening(host: str = "127.0.0.1", family: socket.AddressFamily = socket.AF_INET) -> socket.socket:
    server = socket.socket(family)
    server.bind((host, 0))
    server.listen()
    return server


class NoContent(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


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


def test_a_test_without_the_marker_cannot_reach_loopback_on_either_family() -> None:
    with listening() as server:
        host, port = server.getsockname()
        with pytest.raises(RuntimeError, match="network"):
            socket.getaddrinfo(host, port)
        with socket.socket() as client, pytest.raises(RuntimeError, match="network"):
            client.connect_ex((host, port))
    with socket.socket(socket.AF_INET6) as client, pytest.raises(RuntimeError, match="network"):
        client.connect(("::1", 9))


@pytest.mark.loopback
def test_a_loopback_test_reaches_a_server_it_runs_on_this_machine() -> None:
    with listening() as server:
        host, port = server.getsockname()
        assert socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4] == (host, port)
        with socket.create_connection((host, port), timeout=1):
            pass
        with socket.socket() as client:
            client.connect((host, port))
        with socket.socket() as client:
            assert client.connect_ex((host, port)) == 0
    with ThreadingHTTPServer(("127.0.0.1", 0), NoContent) as http_server:
        threading.Thread(target=http_server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        host, port = http_server.server_address[:2]
        try:
            assert httpx.get(f"http://{host}:{port}/", timeout=5).status_code == 204
        finally:
            http_server.shutdown()


@pytest.mark.loopback
def test_a_loopback_test_reaches_a_server_on_the_ipv6_loopback_too() -> None:
    try:
        server = listening("::1", socket.AF_INET6)
    except OSError:
        pytest.skip("this machine has no IPv6 loopback")
    with server:
        host, port = server.getsockname()[:2]
        with socket.create_connection((host, port), timeout=1):
            pass


@pytest.mark.loopback
def test_a_loopback_test_is_still_refused_any_other_address_and_any_name() -> None:
    # 192.0.2.1 is TEST-NET-1 (RFC 5737) and never routed; 127.0.0.2 is loopback, but not one of the two addresses.
    for host in ("192.0.2.1", "10.0.0.1", "127.0.0.2"):
        with pytest.raises(RuntimeError, match="network"):
            socket.create_connection((host, 443), timeout=1)
        with socket.socket() as client, pytest.raises(RuntimeError, match="network"):
            client.connect((host, 443))
        with socket.socket() as client, pytest.raises(RuntimeError, match="network"):
            client.connect_ex((host, 443))
        with pytest.raises(RuntimeError, match="network"):
            socket.getaddrinfo(host, 443)
    for name in ("localhost", "huggingface.co"):
        with pytest.raises(RuntimeError, match="network"):
            socket.getaddrinfo(name, 443)
        with pytest.raises(RuntimeError, match="network"):
            socket.create_connection((name, 443), timeout=1)
    with pytest.raises(RuntimeError, match="network"):
        httpx.get("http://localhost:1/", timeout=1)
