"""Render verify: does SMG send the engine the prompt token ids the reference renders?

Each render fixture's request goes to SMG's chat endpoint with the fixture id as ``rid``. SMG passes a
client ``rid`` through as the engine request's ``request_id``, verbatim or, under prefill-decode
disaggregation, with a UUID after it, and the mock worker behind SMG writes every Generate request it
receives to its capture file as one JSON line, so a case's capture line is found by its id and the
``input_ids`` on it are compared with ``reference.input_ids``.

Cases go one at a time: the mock writes a request's line before the engine answers, so once SMG has answered a
case, every line for it is in the file, and the lines written since the previous answer are the case's or another
client's.

SMG's answer is read as the vLLM render client reads vLLM's (``bellwether.engines.vllm_render``): a 400 whose body is
SMG's error object is SMG refusing the request, an answer about the case; anything else that is not a 200 is about the
setup or SMG's own failure, and measures nothing. A capture line, when there is one, is compared whatever SMG answered.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

import httpx

from bellwether.manifest import Manifest

TIMEOUT = 60.0  # seconds per request; a render case takes milliseconds, so this only catches a stuck SMG
ID_WINDOW = 8  # ids shown on each side of the first difference
TEXT_WINDOW = 40  # characters shown on each side of the first difference in the prompt text
# Under prefill-decode disaggregation SMG's gRPC router sends a rid as `{rid}-{uuid}`, with a fresh UUIDv7 for each
# attempt (smg's resolve_request_id_stamp and IdStamp::restamp). Any UUID in canonical form is taken off.
PREFILL_DECODE_ID = re.compile(r"(.+)-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class CannotVerify(Exception):
    """The run cannot give verdicts: no such model or cases, a manifest, set or capture file verify cannot read, or no
    answer from SMG."""


class Capture:
    """The mock's capture file, opened before the first request and read as it grows.

    The mock creates the file when it starts and appends to it, so what is there at the start is an earlier run's.
    Each read takes the complete lines written since the last one; the text after the last newline is a line still
    being written, kept for the next read.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self._file = path.open("rb")
        except OSError as err:
            raise CannotVerify(f"cannot read the capture file {path}: {err.strerror or err}") from None
        self._at = self._file.seek(0, os.SEEK_END)  # where the text not yet split into lines starts
        self._pending = b""

    def __enter__(self) -> Capture:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._file.close()

    def lines(self) -> list[tuple[int, object]]:
        """``(byte offset, line)`` for each complete line written since the last read; blank lines are skipped."""
        *complete, self._pending = (self._pending + self._file.read()).split(b"\n")
        found = []
        for raw in complete:
            at, self._at = self._at, self._at + len(raw) + 1
            if not raw.strip():
                continue
            try:
                found.append((at, json.loads(raw)))
            except ValueError:
                raise CannotVerify(
                    f"{self.path}: the line at byte {at} is not JSON; is this the mock's capture file?"
                ) from None
        return found


def client() -> httpx.Client:
    """The HTTP client for one run. It follows no redirect, since SMG answers where it is, and it ignores the
    environment's proxy settings: verify talks to the SMG it was given, with nothing in between."""
    return httpx.Client(timeout=TIMEOUT, follow_redirects=False, trust_env=False)


def served_models(http: httpx.Client, url: str) -> set[str]:
    """The model ids SMG lists at ``/v1/models``, asked before the first case.

    SMG lists each worker's model id there, not its aliases, and answers 503 when it has no worker. Anything but a
    200 with a list of ids stops the run: a model SMG does not serve would fail every case as a failed measurement.
    """
    target = f"{url.rstrip('/')}/v1/models"
    try:
        response = http.get(target)
    except (httpx.HTTPError, httpx.InvalidURL) as err:
        raise CannotVerify(f"no answer from {target}: {type(err).__name__}: {err}") from err
    if response.status_code != 200:
        hint = "; is --smg SMG's base URL, without /v1?" if response.status_code == 404 else ""
        raise CannotVerify(f"{target} answered {response.status_code}: {response.text.strip()[:200]}{hint}")
    try:
        models = response.json()["data"]
        ids = {model["id"] for model in models}
    except (ValueError, TypeError, KeyError):
        raise CannotVerify(f"{target} answered 200 without SMG's list of models: {response.text[:200]!r}") from None
    if not all(isinstance(model_id, str) for model_id in ids):
        raise CannotVerify(f"{target} answered 200 without SMG's list of models: {response.text[:200]!r}")
    return ids


def verify_case(http: httpx.Client, url: str, capture: Capture, manifest: Manifest, set_name: str, case: dict) -> dict:
    """Send one case and judge it on the first capture line written for it since the previous case's answer."""
    try:
        status, body, location = send(http, url, request_body(case, manifest.model))
    except (httpx.HTTPError, httpx.InvalidURL) as err:
        raise CannotVerify(f"no answer from {url} for {case['id']}: {type(err).__name__}: {err}") from err
    line = None
    for at, candidate in capture.lines():
        request_id = candidate.get("request_id") if isinstance(candidate, dict) else None
        if line is not None or not carries(request_id, case["id"]):
            continue
        ids = candidate.get("input_ids")
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            raise CannotVerify(
                f"{capture.path}: the line at byte {at} for {request_id} has no list of integer input_ids"
            )
        line = candidate
    refused = refusal(status, body)
    if line is not None:
        # SMG sent the engine its prompt, so the ids are compared whatever it answered afterwards.
        outcome = compare(case, line)
    elif status == 200:
        outcome = {"verdict": "missing"}
    elif refused is not None:
        outcome = {"verdict": "rejected", "code": refused["code"], "message": refused["message"]}
    else:
        outcome = {"verdict": "measurement_failed"}
    if status != 200 and "message" not in outcome:
        outcome["message"] = f"redirected to {location}" if location is not None else error_message(body)
    return {"id": case["id"], "model": manifest.model, "set": set_name, **outcome, "status": status, "body": body}


def request_body(case: dict, model: str) -> dict:
    """The recorded request in its own key order, plus what SMG needs to route, join and answer it.

    None of the added fields reaches the chat template, so the prompt SMG renders is the recorded one's.
    """
    body = dict(case["request"])
    body["model"] = model
    body["rid"] = case["id"]
    body["stream"] = False
    if "max_tokens" not in body and "max_completion_tokens" not in body:
        body["max_tokens"] = 1
    return body


def send(http: httpx.Client, url: str, body: dict) -> tuple[int, object, str | None]:
    """SMG's status, its body (JSON, else the text), and where a redirect points."""
    response = http.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )
    location = response.headers.get("location") if response.is_redirect else None
    try:
        return response.status_code, response.json(), location
    except ValueError:
        return response.status_code, response.text, location


def refusal(status: int, body: object) -> dict | None:
    """SMG's error object when the answer is SMG refusing the request, else None.

    That is a 400 whose body is ``{"error": {...}}`` with a string ``message`` and a ``code``: SMG's own errors carry
    a string code (``create_error`` in smg's external_router), and its validation errors the number 400
    (``ValidatedJson`` in smg's protocols). A proxy's 400 is a page, and every other status (404 for a model no worker
    serves, 429, 5xx) is about the setup or SMG's own failure, not the request.
    """
    error = body.get("error") if status == 400 and isinstance(body, dict) else None
    if not isinstance(error, dict) or not isinstance(error.get("message"), str):
        return None
    code = error.get("code")
    return error if isinstance(code, str) or (isinstance(code, int) and not isinstance(code, bool)) else None


def carries(request_id: object, case_id: str) -> bool:
    """Whether a captured ``request_id`` is the case's: its id as sent, or with one prefill-decode suffix after it.

    The id as sent is tried first, so a case whose id ends in something like that suffix keeps its own lines.
    """
    if not isinstance(request_id, str):
        return False
    if request_id == case_id:
        return True
    match = PREFILL_DECODE_ID.fullmatch(request_id)
    return match is not None and match[1] == case_id


def compare(case: dict, line: dict) -> dict:
    """``match``, or where the ids SMG sent leave the reference's and whether the text it rendered did too."""
    reference, smg = case["reference"]["input_ids"], line["input_ids"]
    if smg == reference:
        return {"verdict": "match"}
    index = first_difference(reference, smg)
    outcome = {
        "verdict": "regression",
        "index": index,
        "lengths": {"reference": len(reference), "smg": len(smg)},
        "window": window(reference, smg, index, ID_WINDOW),
    }
    # Equal text in other ids points at tokenization; other text points at rendering.
    expected, sent = case["reference"].get("text"), line.get("original_text")
    if not isinstance(expected, str) or not isinstance(sent, str):
        outcome["text_equal"] = None
    elif sent == expected:
        outcome["text_equal"] = True
    else:
        at = first_difference(expected, sent)
        outcome["text_equal"] = False
        outcome["text"] = {"index": at, **window(expected, sent, at, TEXT_WINDOW)}
    return outcome


def first_difference(reference: Sequence, smg: Sequence) -> int:
    """The first index where the two differ; the shorter length when one is a prefix of the other."""
    pairs = zip(reference, smg, strict=False)  # the lengths may differ
    return next((i for i, (a, b) in enumerate(pairs) if a != b), min(len(reference), len(smg)))


def window(reference: Sequence, smg: Sequence, index: int, width: int) -> dict:
    """Both sides from ``width`` before ``index`` to ``width`` after it, with where that span starts."""
    start = max(0, index - width)
    return {"start": start, "reference": reference[start : index + width + 1], "smg": smg[start : index + width + 1]}


def error_message(body: object) -> str:
    """SMG's own errors are ``{"error": {"message": ...}}``; something in front of it may answer in plain text."""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return body.strip()[:500] if isinstance(body, str) else json.dumps(body)[:500]
