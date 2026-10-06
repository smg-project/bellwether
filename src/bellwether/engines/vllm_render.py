"""A client for vLLM's GPU-less render server (``vllm launch render``), the source of the vLLM witness.

The server runs vLLM's own request preprocessing and output detokenization with no engine behind it, so
bellwether can ask vLLM which prompt tokens it builds for a request and which text it gives output tokens,
without a GPU and without importing vLLM. Citations are file:line in vLLM v0.31.0 (db9527a4), the image
bellwether pins.

What the two routes used here answer, and what they do not:

- ``POST /v1/chat/completions/render`` takes a chat completion request and answers with the token-in request
  vLLM would hand its engine: prompt token ids and sampling parameters
  (vllm/entrypoints/scale_out/token_in_token_out/protocol.py:145-253). It builds the prompt with the code
  ``vllm serve`` runs (vllm/renderers/online_renderer.py:175-268) and never returns the prompt text.
- Those sampling parameters carry no end-of-sequence ids. ``stop_token_ids`` is the request's own; vLLM adds
  the generation config's eos ids later, in the engine's input processor
  (vllm/v1/engine/input_processor.py:440-443), which the render server does not run.
- ``POST /v1/completions/derender`` decodes output ids: whole without ``stream``, or one chunk per call with
  ``stream: true`` and a decode state the caller carries from call to call
  (vllm/renderers/online_derenderer.py:221-286 and 668-750). One token per call gives the text each token
  contributes, which is what ``output_pieces`` records.
- Both derender forms skip special tokens unless the request says otherwise
  (vllm/renderers/online_derenderer.py:626-630 and 711-715), so every call here says which.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

CHAT_RENDER = "/v1/chat/completions/render"
COMPLETIONS_DERENDER = "/v1/completions/derender"


class RenderServerError(Exception):
    """The render server answered with a status other than 200; ``message`` is what it said."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"vLLM render server answered {status}: {message}")
        self.status = status
        self.message = message


@dataclass
class RenderedPrompt:
    """What ``/v1/chat/completions/render`` answered for one request, as it answered.

    There is no text: the answer's only prompt is ``token_ids``, and the text vLLM rendered stays on the
    server (vllm/renderers/hf.py:1127-1156). ``/detokenize`` would decode the ids
    (vllm/renderers/base.py:583-585), which is not the template's text, so nothing stands in for it. ``raw``
    keeps the whole answer, ``token_offsets`` included when the request set ``return_token_offsets``.
    """

    token_ids: list[int]
    sampling_params: dict[str, Any]
    raw: dict[str, Any]


class RenderServer:
    """A running ``vllm launch render`` at ``base_url``; each call is one POST that waits ``timeout`` seconds."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def render_chat(self, request: dict) -> RenderedPrompt:
        """The prompt ids vLLM builds for a chat completion request.

        The request is sent as given, key order included, since a template can render it. It needs no
        ``model``: without one the served model answers (vllm/entrypoints/serve/engine/serving.py:73-78).
        """
        answer = self._post(CHAT_RENDER, request)
        return RenderedPrompt(answer["token_ids"], answer["sampling_params"], answer)

    def derender(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        """vLLM's text for ``ids`` decoded whole: its tokenizer's ``decode``
        (vllm/renderers/online_derenderer.py:645-647).

        ``skip_special_tokens`` defaults to False, as bellwether records an output (record/roundtrip.py:77-80);
        the server's own default is True. No ids give empty text without a call, as ``derender_pieces`` gives no
        pieces: the server refuses a choice with no token ids (vllm/renderers/online_derenderer.py:639-643).
        """
        if not ids:
            return ""
        answer = self._post(
            COMPLETIONS_DERENDER,
            {
                "stream": False,
                "generate_responses": [{"choices": [{"index": 0, "token_ids": ids}]}],
                "completion_request": _completion_request(skip_special_tokens),
            },
        )
        return _only_choice(answer)["text"]

    def derender_pieces(self, ids: list[int], *, skip_special_tokens: bool = False) -> list[str]:
        """The text each token contributes under vLLM's incremental decode: one streaming derender call per token.

        The server keeps no state between calls, so each call carries the ``stream_state`` the previous one
        returned, unchanged, and the first carries none
        (vllm/entrypoints/scale_out/token_in_token_out/protocol.py:637-660). A chunk of several tokens decodes
        to the concatenation of its tokens' texts (vllm/renderers/online_derenderer.py:258-286 goes token by
        token), so these pieces also give vLLM's text for every chunk plan.

        vLLM decodes with ``detokenize_incrementally`` over token strings
        (vllm/tokenizers/detokenizer_utils.py:176-269) from an empty window; ``output_pieces`` come from
        tokenizers' ``DecodeStream``, also started empty (record/roundtrip.py:79-80). Both decode a window of
        recent tokens and emit only what extends it, holding back a trailing replacement character, so for
        Hugging Face fast tokenizers with special tokens kept they should agree. Where they can part:
        transformers and tokenizers do not mean the same tokens by "special" when skipping them, and tokenizers
        vLLM loads outside transformers (``--tokenizer-mode mistral`` and the other custom modes) convert token
        strings their own way. Neither is what ``vllm serve`` streams: its engine primes ``DecodeStream`` with
        the prompt (vllm/v1/engine/detokenizer.py:182-183), which changes the first piece of SentencePiece
        tokenizers.
        """
        completion_request = _completion_request(skip_special_tokens)
        pieces: list[str] = []
        state = None
        for token in ids:
            answer = self._post(
                COMPLETIONS_DERENDER,
                {
                    "stream": True,
                    "generate_chunk": {"choices": [{"index": 0, "token_ids": [token]}]},
                    "stream_state": state,
                    "completion_request": completion_request,
                },
            )
            pieces.append(_only_choice(answer["chunk"])["text"])
            state = answer["stream_state"]
        return pieces

    def _post(self, path: str, body: dict) -> Any:
        """POST ``body`` as JSON, the only content type the server takes.

        Any other is refused with a 400 (vllm/entrypoints/serve/utils/api_utils.py:350-356).
        """
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status, text = response.status, response.read().decode()
        except urllib.error.HTTPError as err:
            raise RenderServerError(err.code, _error_message(err.read().decode(errors="replace"))) from err
        if status != 200:
            raise RenderServerError(status, _error_message(text))
        return json.loads(text)


def _completion_request(skip_special_tokens: bool) -> dict:
    """The ``completion_request`` a derender call carries, there only to set ``skip_special_tokens``.

    The derenderer reads nothing else from it (vllm/renderers/online_derenderer.py:626-630 and 711-715), but
    it is validated as a whole CompletionRequest, which refuses one without a prompt
    (vllm/entrypoints/openai/completion/protocol.py:550-567). Its prompt is therefore one token, id 0 (a prompt
    may be a list of non-negative ids, protocol.py:50-56), whatever the ids being decoded: with those ids as the
    prompt, each of ``derender_pieces``' calls would carry all of them.
    """
    return {"prompt": [0], "skip_special_tokens": skip_special_tokens}


def _only_choice(answer: dict) -> dict:
    """The one choice a single-sequence derender answers with.

    One choice per generate-response choice (vllm/renderers/online_derenderer.py:637-664 and 719-730), and
    every call here sends one.
    """
    choices = answer["choices"]
    if len(choices) != 1:
        raise ValueError(f"expected one choice in the derender answer, got {len(choices)}")
    return choices[0]


def _error_message(text: str) -> str:
    """The message in a vLLM error body, or the body itself when it is not vLLM's shape.

    vLLM answers ``{"error": {"message", "type", "param", "code"}}`` (vllm/entrypoints/serve/engine/protocol.py:
    79-87), except the API-key check, which answers ``{"error": "Unauthorized"}``
    (vllm/entrypoints/serve/middleware/authenticate.py:59-61).
    """
    try:
        error = json.loads(text)["error"]
    except (ValueError, TypeError, KeyError):
        return text
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return error if isinstance(error, str) else text
