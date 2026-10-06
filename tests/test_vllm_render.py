"""Tests for the vLLM render server client, against a fake server that answers in vLLM v0.31.0's shapes.

Every reply is built from the vLLM source at v0.31.0 (db9527a4), cited where it is built. The token ids,
pieces and decode windows are Qwen3-8B's at the revision in fixtures/qwen3-8b/manifest.toml, worked out
offline with vLLM's own decode function; they were not captured from a running server.
"""

from __future__ import annotations

import http.server
import importlib
import json
import threading

import pytest

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


class FakeRenderServer(http.server.HTTPServer):
    """Answers each POST with the next queued (status, body) and keeps what the client sent."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.replies: list[tuple[int, object]] = []
        self.received: list[tuple[str, str, bytes]] = []

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def bodies(self) -> list[dict]:
        return [json.loads(raw) for _, _, raw in self.received]


class _Handler(http.server.BaseHTTPRequestHandler):
    server: FakeRenderServer

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.received.append((self.path, self.headers.get("Content-Type", ""), raw))
        status, body = self.server.replies.pop(0) if self.server.replies else (500, "the fake has no reply queued")
        text = body if isinstance(body, str) else json.dumps(body)
        data = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain" if isinstance(body, str) else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass  # keep pytest's output to the tests


@pytest.fixture
def fake_server():
    server = FakeRenderServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
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

    rendered = vllm_render.RenderServer(fake_server.url, timeout=5).render_chat(CHAT_REQUEST)

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
    server = vllm_render.RenderServer(fake_server.url, timeout=5)

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


def test_derender_pieces_streams_one_token_per_call_and_carries_the_state_back(fake_server, vllm_render):
    """Streaming /v1/completions/derender, one token per call, gives the text each token contributes.

    The request: DerenderCompletionStreamRequest (vllm/entrypoints/scale_out/token_in_token_out/protocol.py:
    637-660); the body: stream_reply() above. The server keeps nothing between calls: each call carries the
    stream_state the previous one returned, and the first carries none (protocol.py:652-653).
    """
    replies = [stream_reply(piece, *window) for piece, window in zip(OUTPUT_PIECES, WINDOWS, strict=True)]
    fake_server.replies += [(200, reply) for reply in replies]

    pieces = vllm_render.RenderServer(fake_server.url, timeout=5).derender_pieces(OUTPUT_IDS)

    assert pieces == OUTPUT_PIECES
    bodies = fake_server.bodies()
    assert [path for path, _, _ in fake_server.received] == ["/v1/completions/derender"] * len(OUTPUT_IDS)
    assert [body["generate_chunk"]["choices"] for body in bodies] == [
        [{"index": 0, "token_ids": [token]}] for token in OUTPUT_IDS
    ]
    assert all(body["stream"] is True for body in bodies)
    assert all(body["completion_request"]["skip_special_tokens"] is False for body in bodies)
    assert [body["stream_state"] for body in bodies] == [None] + [reply["stream_state"] for reply in replies[:-1]]


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
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
            "The model `Qwen/Qwen3-8B-typo` does not exist.",
            id="unknown-model",  # vllm/entrypoints/serve/engine/serving.py:66-71
        ),
        pytest.param(
            400,
            {
                "error": {
                    "message": "choice 0 in response 4f2b6c1d has empty or null token_ids",
                    "type": "BadRequestError",
                    "param": None,
                    "code": 400,
                }
            },
            "choice 0 in response 4f2b6c1d has empty or null token_ids",
            id="empty-ids",  # vllm/renderers/online_derenderer.py:639-643, mapped by error_response.py:69-72
        ),
        pytest.param(
            401,
            {"error": "Unauthorized"},
            "Unauthorized",
            id="api-key",  # vllm/entrypoints/serve/middleware/authenticate.py:59-61
        ),
        pytest.param(502, "Bad Gateway", "Bad Gateway", id="not-vllm"),
    ],
)
def test_a_non_200_answer_raises_with_its_status_and_the_server_message(
    fake_server, vllm_render, status, body, message
):
    """vLLM answers errors as ``{"error": {"message", "type", "param", "code"}}``.

    ErrorResponse: vllm/entrypoints/serve/engine/protocol.py:79-87, sent by every route on this server
    (vllm/entrypoints/scale_out/derender/api_router.py:123-125); anything that is not vLLM's shape is kept as text.
    """
    fake_server.replies.append((status, body))

    with pytest.raises(vllm_render.RenderServerError) as raised:
        vllm_render.RenderServer(fake_server.url, timeout=5).derender(OUTPUT_IDS)

    assert raised.value.status == status
    assert raised.value.message == message
