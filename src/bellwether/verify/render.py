"""Render verify: does SMG send the engine the prompt token ids the reference renders?

Each render fixture's request goes to SMG's chat endpoint with the fixture id as ``rid``. SMG passes a
client ``rid`` through as the engine request's ``request_id``, verbatim or, under prefill-decode
disaggregation, with a UUID after it, and the mock worker behind SMG writes every Generate request it
receives to its capture file as one JSON line, so a case's capture line is found by its id and the
``input_ids`` on it are compared with ``reference.input_ids``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

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


def verify(url: str, capture: Path, cases: list[tuple[Manifest, str, dict]]) -> list[dict]:
    """Send every case, then join the capture file to the answers on ``request_id``."""
    with open_capture(capture) as file:
        start = file.seek(0, os.SEEK_END)
        answers = []
        # The environment's proxy settings are ignored: verify talks to the SMG it was given, with nothing in between.
        with httpx.Client(timeout=TIMEOUT, trust_env=False) as client:
            for manifest, _, case in cases:
                try:
                    answers.append(send(client, url, request_body(case, manifest.model)))
                except (httpx.HTTPError, httpx.InvalidURL) as err:
                    raise CannotVerify(f"no answer from {url} for {case['id']}: {type(err).__name__}: {err}") from err
        captured = read_capture(file, capture, start, {case["id"] for _, _, case in cases})
    results = []
    for (manifest, set_name, case), (status, body) in zip(cases, answers, strict=True):
        if status != 200:
            # A non-200 is SMG's answer to the case, so it is a verdict, not an error of the run.
            outcome = {"verdict": "rejected", "message": error_message(body)}
        elif case["id"] not in captured:
            outcome = {"verdict": "missing"}
        else:
            outcome = compare(case, captured[case["id"]])
        results.append(
            {"id": case["id"], "model": manifest.model, "set": set_name, **outcome, "status": status, "body": body}
        )
    return results


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


def send(client: httpx.Client, url: str, body: dict) -> tuple[int, object]:
    response = client.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, response.text


def open_capture(path: Path) -> BinaryIO:
    """The capture file, opened before the first request so that one verify cannot read stops the run at once.

    The mock creates it when it starts and appends to it, so what is there already is an earlier run's.
    """
    try:
        return path.open("rb")
    except OSError as err:
        raise CannotVerify(f"cannot read the capture file {path}: {err.strerror or err}") from None


def read_capture(file: BinaryIO, path: Path, start: int, wanted: set[str]) -> dict[str, dict]:
    """The first capture line for each wanted fixture id among the complete lines written since ``start``.

    Lines for other request ids are other clients'. The text after the last newline is a line the mock is still
    writing for one of them: every request this run sent was answered, and the mock writes a request's line
    before the engine answers. A later line for a case is a retry or, under prefill-decode, the other leg's copy
    of the request.
    """
    file.seek(start)
    data = file.read()
    found: dict[str, dict] = {}
    end = start
    for raw in data.split(b"\n")[:-1]:
        at, end = end, end + len(raw) + 1
        if not raw.strip():
            continue
        try:
            line = json.loads(raw)
        except ValueError:
            raise CannotVerify(f"{path}: the line at byte {at} is not JSON; is this the mock's capture file?") from None
        request_id = line.get("request_id") if isinstance(line, dict) else None
        case_id = fixture_id(request_id, wanted)
        if case_id is None or case_id in found:
            continue
        ids = line.get("input_ids")
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            raise CannotVerify(f"{path}: the line at byte {at} for {request_id} has no list of integer input_ids")
        found[case_id] = line
    return found


def fixture_id(request_id: object, wanted: set[str]) -> str | None:
    """The wanted fixture id a captured ``request_id`` carries, or None.

    An id that is a wanted fixture id as it stands is that case's, so no line joins two cases; otherwise one
    prefill-decode suffix is taken off its end.
    """
    if not isinstance(request_id, str):
        return None
    if request_id in wanted:
        return request_id
    match = PREFILL_DECODE_ID.fullmatch(request_id)
    return match[1] if match and match[1] in wanted else None


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
    return body.strip() if isinstance(body, str) else json.dumps(body)
