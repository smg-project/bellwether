"""Tests for the vLLM render server client, against a fake server that answers in vLLM v0.31.0's shapes.

Every reply is built from the vLLM source at v0.31.0 (db9527a4), cited where it is built. The token ids,
pieces and decode windows are Qwen3-8B's at the revision in fixtures/qwen3-8b/manifest.toml, worked out
offline with vLLM's own decode function; they were not captured from a running server.
"""

from __future__ import annotations

import http.server
import importlib
import json
import socket
import sys
import threading
import time

import pytest

# Every test here talks to a fake render server it runs on 127.0.0.1, or to a closed port there.
pytestmark = pytest.mark.loopback

# fixtures/qwen3-8b/render/common.jsonl, case text-user-only: the request and the ids its template renders.
CHAT_REQUEST = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
PROMPT_IDS = [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198]

# "晴れ 🌤" in Qwen3-8B: the emoji spans three byte-level tokens, so the first two contribute nothing.
OUTPUT_IDS = [105212, 32021, 11162, 234, 97]
OUTPUT_TEXT = "晴れ 🌤"
OUTPUT_PIECES = ["晴", "れ", "", "", " 🌤"]
# The decode window (prev_tokens, read_offset) vLLM hands back after each of those tokens: what
# detokenize_incrementally leaves (vllm/tokenizers/detokenizer_utils.py:176-269) once _detokenize_delta
# trims it and rebases the offsets (vllm/renderers/online_derenderer.py:275-286).
WINDOWS = [
    (["æĻ´"], 1),
    (["ãĤĮ"], 1),
    (["ãĤĮ", "ĠðŁ"], 1),
    (["ãĤĮ", "ĠðŁ", "Į"], 1),
    (["ĠðŁ", "Į", "¤"], 3),
]


class FakeRenderServer(http.server.ThreadingHTTPServer):
    """Answers each POST with the next queued reply and keeps what the client sent.

    A reply is ``(status, body)`` or ``(status, body, headers)``. Every answer is HTTP/1.1 with a Content-Length,
    so a client can keep one connection open from call to call; ``peers`` has the client port each request came
    from and ``closed`` the client port of each connection that ended. ``delay`` holds every answer back that many
    seconds.
    """

    block_on_close = False  # a connection a client keeps open must not hold up the fixture's teardown

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.replies: list[tuple] = []
        self.received: list[tuple[str, str, bytes]] = []
        self.peers: list[int] = []
        self.closed: list[int] = []
        self.delay = 0.0

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def bodies(self) -> list[dict]:
        return [json.loads(raw) for _, _, raw in self.received]

    def handle_error(self, request: object, client_address: object) -> None:
        if not isinstance(sys.exc_info()[1], ConnectionError):  # a client that stopped waiting closed its end
            super().handle_error(request, client_address)


class _Handler(http.server.BaseHTTPRequestHandler):
    server: FakeRenderServer
    protocol_version = "HTTP/1.1"

    def handle(self) -> None:
        try:
            super().handle()  # one request after another until the client closes the connection
        finally:
            self.server.closed.append(self.client_address[1])

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        target = self.requestline.split()[1]  # as sent: self.path turns a leading "//" into "/"
        self.server.received.append((target, self.headers.get("Content-Type", ""), raw))
        self.server.peers.append(self.client_address[1])
        reply = self.server.replies.pop(0) if self.server.replies else (500, "the fake has no reply queued")
        status, body, headers = reply if len(reply) == 3 else (*reply, {})
        time.sleep(self.server.delay)
        text = body if isinstance(body, str) else json.dumps(body)
        data = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain" if isinstance(body, str) else "application/json")
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass  # keep pytest's output to the tests


@pytest.fixture
def fake_server():
    server = FakeRenderServer()
    # serve_forever looks for shutdown() once per poll_interval, 0.5 s unless told; each test waits for it once.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def vllm_render():
    """The module under test, imported per test so a missing or broken module fails each test on its own."""
    return importlib.import_module("bellwether.engines.vllm_render")


def generate_request(token_ids: list[int]) -> dict:
    """``GenerateRequest.model_dump()``, the body /v1/chat/completions/render answers with.

    The fields in declaration order (vllm/entrypoints/scale_out/token_in_token_out/protocol.py:145-253), filled
    as render_chat_request fills them (vllm/entrypoints/scale_out/render/serving.py:127-140). There is no field
    for the prompt text. ``sampling_params`` is the omit-defaults dump with private fields removed
    (vllm/v1/serial_utils.py:606-625) of what ChatCompletionRequest.to_sampling_params builds
    (vllm/entrypoints/openai/chat_completion/protocol.py:661-732) from Qwen3-8B's generation_config.json
    defaults (vllm/config/model.py:1804-1859), with max_tokens = max_model_len - len(token_ids)
    (vllm/entrypoints/serve/utils/api_utils.py:169-205). stop_token_ids is the request's own, empty.
    """
    return {
        "request_id": "chatcmpl-5c1d9e0e8f3b4a2c9d7e6f5a4b3c2d1e",
        "token_ids": token_ids,
        "token_offsets": None,
        "features": None,
        "content_parts": None,
        "sampling_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "stop": [],
            "stop_token_ids": [],
            "max_tokens": 40960 - len(token_ids),
            "output_kind": 2,
            "skip_clone": True,
            "bad_words": [],
            "skip_reading_prefix_cache": False,
        },
        "model": None,
        "return_token_ids": None,
        "stream": False,
        "stream_options": None,
        "cache_salt": None,
        "priority": 0,
        "kv_transfer_params": None,
        "ec_transfer_params": None,
    }


def completion_response(text: str, completion_tokens: int) -> dict:
    """``CompletionResponse.model_dump()``, the body non-streaming /v1/completions/derender answers with.

    The fields in declaration order (vllm/entrypoints/openai/completion/protocol.py:613-658), filled by
    derender_completion_response (vllm/entrypoints/scale_out/derender/serving.py:206-252) with one choice per
    generate-response choice, its text a whole decode (vllm/renderers/online_derenderer.py:615-666).
    """
    return {
        "id": "4f2b6c1d8e9a4b7c9d0e1f2a3b4c5d6e",
        "object": "text_completion",
        "created": 1791331200,
        "model": "Qwen/Qwen3-8B",
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": "stop",
                "stop_reason": None,
                "token_ids": None,
                "prompt_logprobs": None,
                "prompt_token_ids": None,
                "routed_experts": None,
            }
        ],
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": completion_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        },
        "kv_transfer_params": None,
        "ec_transfer_params": None,
        "metrics": None,
    }


def stream_reply(piece: str, prev_tokens: list[str], read_offset: int) -> dict:
    """``DerenderCompletionStreamResponse.model_dump()``, the body streaming /v1/completions/derender answers with.

    ``chunk`` is a CompletionStreamResponse (vllm/entrypoints/openai/completion/protocol.py:661-690) as
    derender_completion_stream fills it (vllm/renderers/online_derenderer.py:668-750); ``stream_state`` is a
    DerenderStreamState (vllm/entrypoints/scale_out/token_in_token_out/protocol.py:478-590) whose parser-path
    fields stay at their defaults on the completions route.
    """
    return {
        "chunk": {
            "id": "4f2b6c1d8e9a4b7c9d0e1f2a3b4c5d6e",
            "object": "text_completion",
            "created": 1791331200,
            "model": "Qwen/Qwen3-8B",
            "choices": [
                {
                    "index": 0,
                    "text": piece,
                    "logprobs": None,
                    "finish_reason": None,
                    "stop_reason": None,
                    "prompt_token_ids": None,
                    "token_ids": None,
                }
            ],
            "usage": None,
            "system_fingerprint": None,
            "metrics": None,
        },
        "stream_state": {
            "prev_tokens": prev_tokens,
            "prefix_offset": 0,
            "read_offset": read_offset,
            "role_sent": False,
            "output_token_ids": [],
            "output_chunk_lens": [],
            "tools_streamed": False,
            "last_tool_call_ids": [],
        },
    }


def test_render_chat_sends_the_request_as_given_and_returns_the_ids_as_answered(fake_server, vllm_render):
    """/v1/chat/completions/render answers with a token-in request: ids and sampling parameters, no prompt text.

    The route: vllm/entrypoints/scale_out/render/api_router.py:28-51; the body: generate_request() above.
    """
    reply = generate_request(PROMPT_IDS)
    fake_server.replies.append((200, reply))

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        rendered = server.render_chat(CHAT_REQUEST)

    path, content_type, raw = fake_server.received[0]
    assert path == "/v1/chat/completions/render"
    # Any other content type is refused with a 400 (vllm/entrypoints/serve/utils/api_utils.py:350-356).
    assert content_type == "application/json"
    # The request goes as given, key order included: a template can render it.
    assert json.dumps(json.loads(raw)) == json.dumps(CHAT_REQUEST)
    assert rendered.token_ids == PROMPT_IDS
    assert rendered.sampling_params == reply["sampling_params"]
    assert rendered.raw == reply
    assert not hasattr(rendered, "text")  # the server returns none to pass on


def test_derender_decodes_the_ids_whole_and_keeps_special_tokens_unless_told(fake_server, vllm_render):
    """Non-streaming /v1/completions/derender: one generate response in, one whole-text choice out.

    The route: vllm/entrypoints/scale_out/derender/api_router.py:81-126; the request:
    DerenderCompletionRequest (vllm/entrypoints/scale_out/token_in_token_out/protocol.py:434-475); the body:
    completion_response() above. The server skips special tokens unless the completion request says
    otherwise (vllm/renderers/online_derenderer.py:626-630), while output_pieces keep them.
    """
    fake_server.replies += [(200, completion_response(OUTPUT_TEXT, 5)), (200, completion_response(OUTPUT_TEXT, 5))]

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        assert server.derender(OUTPUT_IDS) == OUTPUT_TEXT
        server.derender(OUTPUT_IDS, skip_special_tokens=True)

    assert [path for path, _, _ in fake_server.received] == ["/v1/completions/derender"] * 2
    kept, skipped = fake_server.bodies()
    assert kept["stream"] is False
    assert kept["generate_responses"] == [{"choices": [{"index": 0, "token_ids": OUTPUT_IDS}]}]
    assert kept["completion_request"]["skip_special_tokens"] is False
    # CompletionRequest refuses a request without a prompt (vllm/entrypoints/openai/completion/protocol.py:550-567).
    assert kept["completion_request"]["prompt"]
    assert skipped["completion_request"]["skip_special_tokens"] is True


def test_derender_of_no_ids_is_empty_text_without_a_call(fake_server, vllm_render):
    """No ids decode to no text, as derender_pieces gives no pieces for them, and the server is not asked.

    It would refuse: a choice with empty token_ids is a 400 (vllm/renderers/online_derenderer.py:639-643, mapped by
    vllm/entrypoints/serve/exception_handling/error_response.py:69-72), the reply queued here.
    """
    fake_server.replies.append(
        (
            400,
            {
                "error": {
                    "message": "choice 0 in response 4f2b6c1d has empty or null token_ids",
                    "type": "BadRequestError",
                    "param": None,
                    "code": 400,
                }
            },
        )
    )

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        assert server.derender([]) == ""
    assert fake_server.received == []


def test_derender_pieces_streams_one_token_per_call_and_carries_the_state_back(fake_server, vllm_render):
    """Streaming /v1/completions/derender, one token per call, gives the text each token contributes.

    The request: DerenderCompletionStreamRequest (vllm/entrypoints/scale_out/token_in_token_out/protocol.py:
    637-660); the body: stream_reply() above. The server keeps nothing between calls: each call carries the
    stream_state the previous one returned, and the first carries none (protocol.py:652-653).
    """
    replies = [stream_reply(piece, *window) for piece, window in zip(OUTPUT_PIECES, WINDOWS, strict=True)]
    fake_server.replies += [(200, reply) for reply in replies]

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        pieces = server.derender_pieces(OUTPUT_IDS)

    assert pieces == OUTPUT_PIECES
    bodies = fake_server.bodies()
    assert [path for path, _, _ in fake_server.received] == ["/v1/completions/derender"] * len(OUTPUT_IDS)
    assert [body["generate_chunk"]["choices"] for body in bodies] == [
        [{"index": 0, "token_ids": [token]}] for token in OUTPUT_IDS
    ]
    assert all(body["stream"] is True for body in bodies)
    assert all(body["completion_request"]["skip_special_tokens"] is False for body in bodies)
    assert [body["stream_state"] for body in bodies] == [None] + [reply["stream_state"] for reply in replies[:-1]]


@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_every_derender_call_carries_a_one_token_prompt_whatever_the_ids(fake_server, vllm_render, skip_special_tokens):
    """The completion request is there only for skip_special_tokens, so its prompt is one token, id 0.

    Neither derender form reads anything else from it (vllm/renderers/online_derenderer.py:626-630 and 711-715),
    but it is validated as a whole CompletionRequest, which refuses one without a prompt
    (vllm/entrypoints/openai/completion/protocol.py:550-567). With the ids as that prompt, each of derender_pieces'
    n calls would carry all n ids.
    """
    replies = [stream_reply(piece, *window) for piece, window in zip(OUTPUT_PIECES, WINDOWS, strict=True)]
    fake_server.replies += [(200, completion_response(OUTPUT_TEXT, 5))] + [(200, reply) for reply in replies]

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        server.derender(OUTPUT_IDS, skip_special_tokens=skip_special_tokens)
        server.derender_pieces(OUTPUT_IDS, skip_special_tokens=skip_special_tokens)

    assert [body["completion_request"] for body in fake_server.bodies()] == [
        {"prompt": [0], "skip_special_tokens": skip_special_tokens}
    ] * (1 + len(OUTPUT_IDS))


def closed_port() -> int:
    """A port on 127.0.0.1 that nothing listens on: the system hands it out, and it is closed again at once."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_calls_share_one_connection(fake_server, vllm_render):
    """One client keeps one connection open from call to call, rather than opening one per token."""
    replies = [stream_reply(piece, *window) for piece, window in zip(OUTPUT_PIECES, WINDOWS, strict=True)]
    fake_server.replies += [(200, completion_response(OUTPUT_TEXT, 5))] + [(200, reply) for reply in replies]

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        server.derender(OUTPUT_IDS)
        server.derender_pieces(OUTPUT_IDS)

    assert len(fake_server.peers) == 1 + len(OUTPUT_IDS)
    assert len(set(fake_server.peers)) == 1


def test_closing_the_client_closes_its_connection(fake_server, vllm_render):
    """Leaving the ``with`` block closes the connection the calls kept open."""
    fake_server.replies.append((200, completion_response(OUTPUT_TEXT, 5)))

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        server.derender(OUTPUT_IDS)

    deadline = time.monotonic() + 2
    while fake_server.closed != fake_server.peers and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake_server.closed == fake_server.peers


def test_the_environments_proxy_settings_are_ignored(fake_server, vllm_render, monkeypatch):
    """The client talks to the server it was given, whatever proxy the environment names."""
    proxy = f"http://127.0.0.1:{closed_port()}"
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, proxy)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    fake_server.replies.append((200, completion_response(OUTPUT_TEXT, 5)))

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        assert server.derender(OUTPUT_IDS) == OUTPUT_TEXT


def test_a_trailing_slash_on_the_base_url_is_dropped(fake_server, vllm_render):
    """``http://host:port/`` reaches the same routes as ``http://host:port``."""
    fake_server.replies.append((200, generate_request(PROMPT_IDS)))

    with vllm_render.RenderServer(fake_server.url + "/", timeout=5) as server:
        server.render_chat(CHAT_REQUEST)

    assert [path for path, _, _ in fake_server.received] == ["/v1/chat/completions/render"]


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            {
                "message": "choice 0 in response 4f2b6c1d has empty or null token_ids",
                "type": "BadRequestError",
                "param": None,
                "code": 400,
            },
            id="route",  # vllm/renderers/online_derenderer.py:639-643, mapped by error_response.py:69-72
        ),
        pytest.param(
            {
                "message": "1 validation error:\n  {'type': 'missing', 'loc': 'body.generate_responses', "
                "'msg': 'Field required', 'input': '<dict of 2 items>'}",
                "type": "Bad Request",
                "param": "body.generate_responses",
                "code": 400,
            },
            id="validation",  # vllm/entrypoints/serve/exception_handling/handlers/validation.py:149-195
        ),
    ],
)
def test_a_400_in_vllms_error_shape_is_a_refusal(fake_server, vllm_render, error):
    """vLLM refused the request: a 400 whose body is its ErrorResponse (engine/protocol.py:79-87).

    vLLM builds it for a request that fails validation or that a route rejects; the two cases are one of each.
    """
    fake_server.replies.append((400, {"error": error}))

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        with pytest.raises(vllm_render.Refused) as refused:
            server.derender(OUTPUT_IDS)

    assert refused.value.message == error["message"]
    assert refused.value.error == error
    assert error["message"] in str(refused.value)
    assert not isinstance(refused.value, vllm_render.MeasurementFailed)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        # vLLM's own errors that are not about the request.
        pytest.param(
            404,
            {"error": {"message": "Not Found", "type": "Not Found", "param": None, "code": 404}},
            id="unknown-route",  # vllm/entrypoints/serve/exception_handling/handlers/http.py:16-31
        ),
        pytest.param(
            404,
            {
                "error": {
                    "message": "The model `Qwen/Qwen3-8B-typo` does not exist.",
                    "type": "NotFoundError",
                    "param": "model",
                    "code": 404,
                }
            },
            id="unknown-model",  # vllm/entrypoints/serve/engine/serving.py:66-71
        ),
        pytest.param(
            501,
            {
                "error": {
                    "message": "The model does not support Completions Derender API",
                    "type": "NotImplementedError",
                    "param": None,
                    "code": 501,
                }
            },
            id="route-not-served",  # vllm/entrypoints/scale_out/derender/api_router.py:106-108, error_response.py:73-76
        ),
        pytest.param(
            500,
            {
                "error": {
                    "message": "list index out of range",
                    "type": "InternalServerError",
                    "param": None,
                    "code": 500,
                }
            },
            id="vllm-failed",  # vllm/entrypoints/serve/exception_handling/error_response.py:82-85
        ),
        pytest.param(
            401,
            {"error": "Unauthorized"},
            id="api-key",  # vllm/entrypoints/serve/middleware/authenticate.py:59-61
        ),
        # Answers that are not vLLM's.
        pytest.param(
            502, "<html><head><title>502 Bad Gateway</title></head><body>nginx</body></html>", id="proxy-page"
        ),
        pytest.param(203, completion_response(OUTPUT_TEXT, 5), id="a-2xx-not-200"),
        pytest.param(400, "<html><head><title>400 Bad Request</title></head><body>nginx</body></html>", id="400-page"),
        pytest.param(400, ["Bad Request"], id="400-json-array"),
        pytest.param(400, {"detail": "Bad Request"}, id="400-without-error"),
        pytest.param(400, {"error": "Bad Request"}, id="400-error-string"),
        pytest.param(400, {"error": {"type": "BadRequestError", "param": None, "code": 400}}, id="400-without-message"),
        pytest.param(
            400,
            {
                "error": {
                    "message": "Invalid value",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_value",
                }
            },
            id="400-other-code",
        ),
    ],
)
def test_any_other_answer_is_a_failed_measurement(fake_server, vllm_render, status, body):
    """Anything but vLLM's 200 or its refusal leaves nothing measured, and is never taken for a refusal.

    vLLM's own 404, 501, 500 and the API-key check's 401 are about the setup or the server, not the request; the
    rest did not come from vLLM.
    """
    fake_server.replies.append((status, body))

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        with pytest.raises(vllm_render.MeasurementFailed) as failed:
            server.derender(OUTPUT_IDS)

    assert failed.value.status == status
    assert fake_server.url in str(failed.value)
    assert not isinstance(failed.value, vllm_render.Refused)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_a_failed_measurement_and_is_not_followed(fake_server, vllm_render, status):
    """vLLM's routes answer where they are, so a redirect means something else answered; it is not followed.

    The reply after it is a good answer, which a client that followed the redirect would take.
    """
    fake_server.replies += [
        (status, "", {"Location": "/v1/completions/derender"}),
        (200, completion_response(OUTPUT_TEXT, 5)),
    ]

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        with pytest.raises(vllm_render.MeasurementFailed) as failed:
            server.derender(OUTPUT_IDS)

    assert failed.value.status == status
    assert len(fake_server.received) == 1


def test_no_server_is_a_failed_measurement(vllm_render):
    """Nothing listening at the URL, as with a stopped container: no answer, so nothing measured."""
    url = f"http://127.0.0.1:{closed_port()}"

    with pytest.raises(vllm_render.MeasurementFailed) as failed:
        with vllm_render.RenderServer(url, timeout=5) as server:
            server.derender(OUTPUT_IDS)

    assert failed.value.status is None
    assert url in str(failed.value)
    assert not isinstance(failed.value, vllm_render.Refused)


def test_an_answer_later_than_the_timeout_is_a_failed_measurement(fake_server, vllm_render):
    """A server that holds its answer past ``timeout`` fails the call, though the answer it holds is a good one."""
    fake_server.delay = 1.0
    fake_server.replies.append((200, completion_response(OUTPUT_TEXT, 5)))

    with vllm_render.RenderServer(fake_server.url, timeout=0.2) as server:
        with pytest.raises(vllm_render.MeasurementFailed) as failed:
            server.derender(OUTPUT_IDS)

    assert failed.value.status is None


def without(answer: dict, name: str) -> dict:
    """``answer`` without its field ``name``."""
    return {key: value for key, value in answer.items() if key != name}


RENDER_ANSWER = generate_request(PROMPT_IDS)
DERENDER_ANSWER = completion_response(OUTPUT_TEXT, 5)
STREAM_ANSWER = stream_reply(OUTPUT_PIECES[0], *WINDOWS[0])
CALLS = {
    "render": lambda server: server.render_chat(CHAT_REQUEST),
    "derender": lambda server: server.derender(OUTPUT_IDS),
    "pieces": lambda server: server.derender_pieces(OUTPUT_IDS[:1]),
}


@pytest.mark.parametrize(
    ("call", "answer"),
    [
        pytest.param("render", "<html><body>It works!</body></html>", id="render-not-json"),
        pytest.param("render", PROMPT_IDS, id="render-not-an-object"),
        pytest.param("render", without(RENDER_ANSWER, "token_ids"), id="render-without-token-ids"),
        pytest.param("render", {**RENDER_ANSWER, "token_ids": "151644 872 198"}, id="render-token-ids-a-string"),
        pytest.param("render", {**RENDER_ANSWER, "token_ids": [str(i) for i in PROMPT_IDS]}, id="render-ids-not-ints"),
        pytest.param("render", {**RENDER_ANSWER, "sampling_params": None}, id="render-sampling-params-null"),
        pytest.param("derender", [OUTPUT_TEXT], id="derender-not-an-object"),
        pytest.param("derender", without(DERENDER_ANSWER, "choices"), id="derender-without-choices"),
        pytest.param("derender", {**DERENDER_ANSWER, "choices": []}, id="derender-no-choice"),
        pytest.param(
            "derender", {**DERENDER_ANSWER, "choices": DERENDER_ANSWER["choices"] * 2}, id="derender-two-choices"
        ),
        pytest.param(
            "derender",
            {**DERENDER_ANSWER, "choices": [{**DERENDER_ANSWER["choices"][0], "text": None}]},
            id="derender-text-null",
        ),
        pytest.param(
            "derender",
            {**DERENDER_ANSWER, "choices": [{**DERENDER_ANSWER["choices"][0], "text": OUTPUT_PIECES}]},
            id="derender-text-not-a-string",
        ),
        pytest.param("pieces", without(STREAM_ANSWER, "chunk"), id="pieces-without-chunk"),
        pytest.param(
            "pieces",
            {
                **STREAM_ANSWER,
                "chunk": {**STREAM_ANSWER["chunk"], "choices": [without(STREAM_ANSWER["chunk"]["choices"][0], "text")]},
            },
            id="pieces-without-text",
        ),
        pytest.param("pieces", without(STREAM_ANSWER, "stream_state"), id="pieces-without-stream-state"),
    ],
)
def test_a_200_without_what_the_call_reads_is_a_failed_measurement(fake_server, vllm_render, call, answer):
    """A 200 is vLLM's answer only when it has what the call reads, in the type vLLM gives it.

    Each answer is a good one with one thing missing or wrong; nothing stands in for what is missing.
    """
    fake_server.replies.append((200, answer))

    with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
        with pytest.raises(vllm_render.MeasurementFailed) as failed:
            CALLS[call](server)

    assert failed.value.status == 200
    assert not isinstance(failed.value, vllm_render.Refused)


def test_derender_pieces_stops_at_a_failed_call_with_no_pieces(fake_server, vllm_render):
    """A failed call ends derender_pieces with its error: no pieces come back as if whole, and no later token goes."""
    fake_server.replies += [(200, stream_reply(OUTPUT_PIECES[0], *WINDOWS[0])), (502, "Bad Gateway")]

    with pytest.raises(vllm_render.MeasurementFailed) as failed:
        with vllm_render.RenderServer(fake_server.url, timeout=5) as server:
            server.derender_pieces(OUTPUT_IDS)

    assert failed.value.status == 502
    assert len(fake_server.received) == 2
