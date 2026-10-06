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

A call ends one of three ways. vLLM answers: a 200 with what the call reads. vLLM refuses: a 400 carrying its
ErrorResponse, raised as Refused, an answer about the request. Or nothing is measured: MeasurementFailed, for no
answer, a redirect, any other status or body, or a 200 without what the call reads. A refusal can be evidence
about a case; a failed measurement never is, so a stopped container cannot pass for vLLM refusing every case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

CHAT_RENDER = "/v1/chat/completions/render"
COMPLETIONS_DERENDER = "/v1/completions/derender"


class Refused(Exception):
    """vLLM answered and refused the request: a 400 whose body is vLLM's ErrorResponse.

    That body is ``{"error": {"message", "type", "param", "code"}}`` (vllm/entrypoints/serve/engine/protocol.py:
    79-87), and vLLM answers it with a 400 for a request it will not take: one that fails validation
    (vllm/entrypoints/serve/exception_handling/handlers/validation.py:149-195) or that a route rejects
    (vllm/entrypoints/serve/exception_handling/error_response.py:41-44, 53-57, 69-72 and 77-81). It is about the
    request, so it is the answer a recorder may keep as vLLM rejecting the case. ``error`` is the error object as
    answered and ``message`` its message.
    """

    def __init__(self, error: dict) -> None:
        super().__init__(f"vLLM refused the request: {error['message']}")
        self.error = error
        self.message = error["message"]


class MeasurementFailed(Exception):
    """Nothing was measured: vLLM gave no answer to the request, so nothing can be said about the case.

    The call failed (no connection, a timeout, a broken response), was redirected, or was answered with anything
    but a 200 or a refusal: a proxy's page, the API-key check's ``{"error": "Unauthorized"}``
    (vllm/entrypoints/serve/middleware/authenticate.py:59-61), or one of vLLM's errors about the setup or the
    server rather than the request: an unknown model (404, vllm/entrypoints/serve/engine/serving.py:66-71) or
    route (404, vllm/entrypoints/serve/exception_handling/handlers/http.py:16-31), a route the model does not
    serve (501, vllm/entrypoints/scale_out/derender/api_router.py:106-108), or its own failure (500,
    vllm/entrypoints/serve/exception_handling/error_response.py:82-85). A 200 whose body is not JSON, or lacks a
    field the call reads, is one too. It is not a Refused and is never caught as one. ``status`` is the status
    answered, None when there was none.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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
    """A running ``vllm launch render`` at ``base_url``, over one HTTP connection kept open from call to call.

    Close it when done, or use it as a context manager. It follows no redirect, since vLLM's routes answer where
    they are, and it ignores the environment's proxy settings, so it talks to the server it was given with nothing
    in between.

    ``timeout`` bounds each step of a call, not the whole call: httpx's client has no timeout for a whole call.
    ``httpx.Timeout(timeout)`` gives each of its four steps ``timeout`` seconds: connecting, each write of the
    request, each read of the answer, and waiting for a free connection. A server that stops answering fails the
    call once a step waits that long; one that sends a few bytes within every ``timeout`` can keep a call going
    longer.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False)

    def __enter__(self) -> RenderServer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection the calls kept open."""
        self._client.close()

    def render_chat(self, request: dict) -> RenderedPrompt:
        """The prompt ids vLLM builds for a chat completion request.

        The request is sent as given, key order included, since a template can render it. It needs no
        ``model``: without one the served model answers (vllm/entrypoints/serve/engine/serving.py:73-78).
        """
        answer = self._post(CHAT_RENDER, request)
        token_ids = _field(answer, "token_ids", list)
        if not all(isinstance(token, int) for token in token_ids):
            raise MeasurementFailed(f"the answer's token_ids are not all integers: {token_ids!r:.200}", 200)
        return RenderedPrompt(token_ids, _field(answer, "sampling_params", dict), answer)

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
        return _only_text(answer)

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
            pieces.append(_only_text(_field(answer, "chunk", dict)))
            state = _field(answer, "stream_state", dict)
        return pieces

    def _post(self, path: str, body: dict) -> Any:
        """POST ``body`` as JSON, the only content type the server takes, and return what a 200 answers.

        Any other content type is refused with a 400 (vllm/entrypoints/serve/utils/api_utils.py:350-356). A refusal
        raises Refused; any other answer raises MeasurementFailed.
        """
        url = self.base_url + path
        try:
            response = self._client.post(url, content=json.dumps(body), headers={"Content-Type": "application/json"})
        except httpx.HTTPError as err:
            raise MeasurementFailed(f"no answer from {url}: {type(err).__name__}: {err}") from err
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as err:
                raise MeasurementFailed(
                    f"{url} answered 200 with a body that is not JSON: {response.text[:200]!r}", 200
                ) from err
        error = _refusal(response)
        if error is not None:
            raise Refused(error)
        raise MeasurementFailed(f"{url} answered {response.status_code}: {response.text[:200]!r}", response.status_code)


def _completion_request(skip_special_tokens: bool) -> dict:
    """The ``completion_request`` a derender call carries, there only to set ``skip_special_tokens``.

    The derenderer reads nothing else from it (vllm/renderers/online_derenderer.py:626-630 and 711-715), but
    it is validated as a whole CompletionRequest, which refuses one without a prompt
    (vllm/entrypoints/openai/completion/protocol.py:550-567). Its prompt is therefore one token, id 0 (a prompt
    may be a list of non-negative ids, protocol.py:50-56), whatever the ids being decoded: with those ids as the
    prompt, each of ``derender_pieces``' calls would carry all of them.
    """
    return {"prompt": [0], "skip_special_tokens": skip_special_tokens}


def _field(answer: Any, name: str, kind: type) -> Any:
    """``answer[name]``, which must be a ``kind``; a 200 without it measured nothing.

    Nothing stands in for a missing field: an empty default would be recorded as what vLLM answered.
    """
    value = answer.get(name) if isinstance(answer, dict) else None
    if not isinstance(value, kind):
        raise MeasurementFailed(f"the answer has no {kind.__name__} {name!r}: {answer!r:.200}", 200)
    return value


def _only_text(answer: Any) -> str:
    """The text of the one choice a single-sequence derender answers with.

    One choice per generate-response choice (vllm/renderers/online_derenderer.py:637-664 and 719-730), and
    every call here sends one.
    """
    choices = _field(answer, "choices", list)
    if len(choices) != 1:
        raise MeasurementFailed(f"the answer has {len(choices)} choices, not one: {answer!r:.200}", 200)
    return _field(choices[0], "text", str)


def _refusal(response: httpx.Response) -> dict | None:
    """vLLM's error object when ``response`` is a refusal, a 400 whose body is vLLM's ErrorResponse; else None.

    vLLM answers with the status its error object carries in ``code``
    (vllm/entrypoints/serve/exception_handling/handlers/exception.py:23-24, http.py:24-31 and
    validation.py:187-195), and the message is always a string (vllm/entrypoints/serve/engine/protocol.py:79-83).
    """
    if response.status_code != 400:
        return None
    try:
        error = response.json()["error"]
    except (ValueError, TypeError, KeyError):
        return None
    if isinstance(error, dict) and isinstance(error.get("message"), str) and error.get("code") == response.status_code:
        return error
    return None
