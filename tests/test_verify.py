"""``bellwether verify`` against a fake gateway.

The fake stands in for SMG and the mock worker behind it, as verify sees them: it answers
``/v1/chat/completions``, and for a request that reaches the engine it appends the capture line the mock
writes before SMG answers. The engine receives each case's own reference unless a test changes it.

The fake listens on 127.0.0.1, so every test here is marked ``loopback`` (``tests/conftest.py``): it may
connect to that address and to no other.
"""

import hashlib
import json
import re
import socket
import threading
import tracemalloc
import xml.etree.ElementTree as ET
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import zstandard

from bellwether import __version__
from bellwether.cli import main
from bellwether.record.fixtures import write_fixture_file
from bellwether.verify import render

pytestmark = pytest.mark.loopback


class FakeGateway:
    """An HTTP server on a free local port standing in for SMG and the mock behind it, answering by ``rid``."""

    def __init__(self, capture):
        self.capture = capture
        capture.touch()  # the mock creates its capture file when it starts
        self.models: set[str] = set()  # what GET /v1/models lists; with none it answers 503, as SMG does
        self.models_body: dict | None = None  # a 200 body for GET /v1/models other than the list of models
        self.prompts: dict[str, tuple[list[int] | None, str | None]] = {}  # rid -> what the engine receives
        self.request_ids: dict[str, str] = {}  # rid -> the engine's request_id, where it is not the rid as sent
        # rid -> what SMG answers instead of sending the request on: a status and a body. A str is SMG's own error
        # message, bytes a plain body from something in front of it, a dict a JSON body as given; a 3xx redirects.
        self.rejections: dict[str, tuple[int, str | bytes | dict]] = {}
        self.after_line: dict[str, tuple[int, str | bytes | dict]] = {}  # rid -> the same, once the engine has it
        self.appended_after: dict[str, str] = {}  # rid -> raw text another client appends after it
        self.drops: set[str] = set()  # rids whose connection closes without an answer
        self.on_answer: dict[str, Callable[[], None]] = {}  # rid -> something done before answering it
        self.received: list[dict] = []
        # A prompt computed from the request instead of looked up by rid, and no record of what was received: a
        # fake whose memory does not grow with the number of cases.
        self.prompt_for: Callable[[dict], tuple[list[int], str]] | None = None
        self.keep_received = True
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                if self.path != "/v1/models":
                    self.reply(404, b"")
                elif gateway.models_body is not None:
                    self.reply(200, gateway.models_body)
                elif not gateway.models:
                    self.reply(503, b"No models available")
                else:
                    models = [
                        {"id": m, "object": "model", "created": 0, "owned_by": "self_hosted"} for m in gateway.models
                    ]
                    self.reply(200, {"object": "list", "data": sorted(models, key=lambda m: m["id"])})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if gateway.keep_received:
                    gateway.received.append({"path": self.path, "body": body})
                if body["rid"] in gateway.drops:
                    self.close_connection = True
                    return
                self.reply(*gateway.answer(body))

            def reply(self, status, payload):
                plain = isinstance(payload, bytes)
                data = payload if plain else json.dumps(payload).encode()
                self.send_response(status)
                if 300 <= status < 400:
                    self.send_header("Location", "/elsewhere")
                self.send_header("Content-Type", "text/plain" if plain else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def serve(self, cases: dict[str, dict]) -> None:
        """Send each case's reference to the engine, the way a conformant SMG would."""
        for case_id, case in cases.items():
            self.models.add(case["model"])
            self.prompts[case_id] = (case["reference"]["input_ids"], case["reference"]["text"])

    def answer(self, body: dict) -> tuple[int, dict | bytes]:
        rid = body["rid"]
        if rid in self.on_answer:
            self.on_answer[rid]()
        if rid in self.rejections:
            return smg_answer(*self.rejections[rid])
        if self.prompt_for is not None:
            self.prompts = {rid: self.prompt_for(body)}
        ids, text = self.prompts.get(rid, ([], ""))
        if rid in self.prompts:
            line = {
                "request_id": self.request_ids.get(rid, rid),
                "input_ids": ids,
                "original_text": text,
                "stream": body.get("stream"),
                "temperature": None,
                "top_p": None,
                "max_new_tokens": body.get("max_tokens"),
                "constraint": None,
                "logprob_start_len": -1,
            }
            self.append(json.dumps(line) + "\n")
        if rid in self.appended_after:
            self.append(self.appended_after[rid])
        if rid in self.after_line:
            return smg_answer(*self.after_line[rid])
        message = {"role": "assistant", "content": "ok"}
        prompt_tokens = len(ids or [])
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1}
        return 200, {
            "id": rid,
            "choices": [{"index": 0, "message": message, "finish_reason": "length"}],
            "usage": usage,
        }

    def append(self, text: str) -> None:
        with self.capture.open("a", encoding="utf-8") as f:
            f.write(text)


def smg_answer(status: int, body: str | bytes | dict) -> tuple[int, dict | bytes]:
    """A str body is SMG's own error message, in the shape SMG's create_error answers with."""
    if not isinstance(body, str):
        return status, body
    phrase = HTTPStatus(status).phrase
    return status, {"error": {"type": phrase, "code": phrase.lower().replace(" ", "_"), "message": body, "param": None}}


@pytest.fixture
def gateway(tmp_path):
    fake = FakeGateway(tmp_path / "capture.jsonl")
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


def revision_of(slug: str) -> str:
    """A commit hash for the manifest of ``slug``, which pins a revision; each slug has its own."""
    return hashlib.sha1(slug.encode()).hexdigest()


def write_model(fixtures, slug: str, model: str, cases: dict[str, tuple[dict, list[int], str]]) -> dict[str, dict]:
    """A manifest and one render set; ``cases`` maps a case name to its request and reference ids and text."""
    (fixtures / slug).mkdir(parents=True)
    (fixtures / slug / "manifest.toml").write_text(f'model = "{model}"\nrevision = "{revision_of(slug)}"\n')
    lines = {}
    for name, (request, ids, text) in cases.items():
        case_id = f"{slug}/render/{name}"
        reference = {"source": "hf-template", "input_ids": ids, "text": text}
        lines[case_id] = {"id": case_id, "kind": "render", "model": model, "request": request, "reference": reference}
    write_fixture_file(fixtures / slug / "render" / "common.jsonl", lines)
    return lines


HELLO = {"messages": [{"role": "user", "content": "Hello."}]}
# A whole fixture line, for tests that write a set of their own.
CASE_LINE = {
    "id": "m1/render/hello",
    "kind": "render",
    "model": "org/M1",
    "request": HELLO,
    "reference": {"source": "hf-template", "input_ids": [10, 11], "text": "<u>Hello.</u><a>"},
}
BYE = {"messages": [{"role": "user", "content": "Bye."}]}
CASES = {
    "hello": (HELLO, list(range(10, 30)), "<u>Hello.</u><a>"),
    "budget": ({"messages": [{"role": "user", "content": "Hi."}], "max_tokens": 7, "temperature": 0}, [10, 11], "Hi"),
    "tools": (
        {
            "messages": [{"role": "user", "content": "Weather?"}],
            # Keys out of sorted order at every depth: a template renders them in the order it is given.
            "tools": [{"type": "function", "function": {"name": "w", "parameters": {"zip": {}, "city": {}}}}],
            "max_completion_tokens": 5,
        },
        [10, 12, 13],
        "<t>Weather?",
    ),
}
# What SMG appends to a rid under prefill-decode disaggregation: a UUIDv7, a fresh one for each attempt.
PD_SUFFIX = "-0199b9d4-6a52-7c3e-8f21-{:012x}"
ISSUE = "https://github.com/smg-project/smg/issues/1"


def write_known(path, entries: dict[str, dict]) -> None:
    """A known-differences file: one table per fixture id, each with the issue unless the entry gives its own."""
    text = ""
    for case_id, entry in entries.items():
        text += f"[{json.dumps(case_id)}]\n"
        text += "".join(f"{key} = {json.dumps(value)}\n" for key, value in {"issue": ISSUE, **entry}.items())
    path.write_text(text)


def verify(gateway, fixtures, *extra) -> int:
    return main(
        ["verify", "--smg", gateway.url, "--capture", str(gateway.capture), "--fixtures", str(fixtures), *extra]
    )


def test_each_render_case_goes_to_smg_and_its_capture_line_matches(tmp_path, gateway):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)

    assert verify(gateway, fixtures, "--report", str(report)) == 0

    sent = {received["body"]["rid"]: received for received in gateway.received}
    assert sorted(sent) == sorted(cases)
    for case_id, case in cases.items():
        assert sent[case_id]["path"] == "/v1/chat/completions"
        body, request = sent[case_id]["body"], case["request"]
        recorded = json.dumps(dict(list(body.items())[: len(request)]))
        assert recorded == json.dumps(request), "the request as recorded, in its key order at every depth"
        added = {"model": "org/M1", "rid": case_id, "stream": False}
        if not {"max_tokens", "max_completion_tokens"} & set(request):
            added["max_tokens"] = 1  # the request's own limit, under either name, is left alone
        assert dict(list(body.items())[len(request) :]) == added
    verdicts = {case["id"]: case["verdict"] for case in json.loads(report.read_text())["cases"]}
    assert verdicts == dict.fromkeys(cases, "match")


def test_capture_lines_from_before_the_run_do_not_count(tmp_path, gateway):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    # The mock appends: an earlier run against it left lines with the same request ids and other ids.
    gateway.append("".join(json.dumps({"request_id": case_id, "input_ids": [99]}) + "\n" for case_id in cases))

    assert verify(gateway, fixtures, "--report", str(report)) == 0
    assert {case["verdict"] for case in json.loads(report.read_text())["cases"]} == {"match"}


def test_other_clients_lines_and_a_line_still_being_written_are_skipped_and_counted(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    # Another client's request whose id starts with a case's id, a blank line, and a line still being written.
    other = json.dumps({"request_id": "m1/render/hello-again", "input_ids": [1]})
    gateway.appended_after["m1/render/hello"] = other + "\n\n" + json.dumps({"input_ids": [1]}) + "\n"
    gateway.appended_after["m1/render/tools"] = '{"request_id": "another-client-2", "input_'

    assert verify(gateway, fixtures, "--report", str(report)) == 0

    written = json.loads(report.read_text())
    assert [(case["verdict"], case["lines"]) for case in written["cases"]] == [("match", 1)] * 3
    assert written["capture"] == {"lines": 6, "joined": 3, "other": 3, "unfinished": True}
    out = capsys.readouterr().out
    assert "capture: 6 lines written during the run, 3 joined a case, 3 did not; a line was still being written" in out


def test_every_line_for_a_case_is_compared(tmp_path, gateway):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    # A second request for the case, such as a retry, whose prompt is not the reference's.
    again = {"request_id": "m1/render/hello", "input_ids": [*range(10, 22), 7, *range(23, 30)]}
    gateway.appended_after["m1/render/hello"] = json.dumps(again) + "\n"

    assert verify(gateway, fixtures, "--report", str(report)) == 1

    results = {case["id"]: case for case in json.loads(report.read_text())["cases"]}
    hello = results["m1/render/hello"]
    assert (hello["verdict"], hello["index"], hello["lines"]) == ("regression", 12, 2)
    assert results["m1/render/budget"]["lines"] == 1


def test_lines_under_ids_no_case_carries_are_counted(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    for n, case_id in enumerate(cases):
        gateway.request_ids[case_id] = f"chatcmpl-{n}"  # an SMG that does not pass the rid through

    assert verify(gateway, fixtures, "--report", str(report)) == 1

    written = json.loads(report.read_text())
    assert {case["verdict"] for case in written["cases"]} == {"missing"}
    assert written["capture"] == {"lines": 3, "joined": 0, "other": 3, "unfinished": False}
    assert "capture: 3 lines written during the run, 0 joined a case, 3 did not" in capsys.readouterr().out


def test_a_difference_gives_the_first_differing_index_and_the_ids_and_text_around_it(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    # The same text as the reference in other ids points at tokenization ...
    gateway.prompts["m1/render/hello"] = ([*range(10, 22), 500, 501, *range(23, 30)], "<u>Hello.</u><a>")
    # ... and other text points at rendering.
    gateway.prompts["m1/render/tools"] = ([10, 12, 13, 14], "<t>Weather?<g>")
    # A capture line without the text cannot say which.
    gateway.prompts["m1/render/budget"] = ([10, 99], None)

    assert verify(gateway, fixtures, "--report", str(report)) == 1

    results = {case["id"]: case for case in json.loads(report.read_text())["cases"]}
    hello, tools, budget = results["m1/render/hello"], results["m1/render/tools"], results["m1/render/budget"]
    assert hello["verdict"] == "regression"
    assert hello["index"] == 12
    assert hello["lengths"] == {"reference": 20, "smg": 21}
    assert hello["window"] == {
        "start": 4,
        "reference": list(range(14, 30)),
        "smg": [*range(14, 22), 500, 501, *range(23, 30)],
    }
    assert hello["text_equal"] is True
    assert "text" not in hello
    assert tools["verdict"] == "regression"
    assert (tools["index"], tools["lengths"]) == (3, {"reference": 3, "smg": 4})
    assert tools["window"] == {"start": 0, "reference": [10, 12, 13], "smg": [10, 12, 13, 14]}
    assert tools["text_equal"] is False
    assert tools["text"] == {"index": 11, "start": 0, "reference": "<t>Weather?", "smg": "<t>Weather?<g>"}
    assert (budget["verdict"], budget["index"], budget["text_equal"]) == ("regression", 1, None)
    out = capsys.readouterr().out.splitlines()
    assert (
        "regression m1/render/hello: ids differ from index 12 (reference 20 ids, smg 21); same text, so tokenization"
        in out
    )
    assert (
        "regression m1/render/tools: ids differ from index 3 (reference 3 ids, smg 4); "
        "text differs from character 11, so rendering"
    ) in out
    assert (
        "regression m1/render/budget: ids differ from index 1 (reference 2 ids, smg 2); "
        "the capture line does not carry the text"
    ) in out


SMG_VALIDATION = {"error": {"message": "temperature: must be at most 2", "type": "invalid_request_error", "code": 400}}


def smg_error(status: int, code: str, message: str) -> dict:
    return {"error": {"type": HTTPStatus(status).phrase, "code": code, "message": message, "param": None}}


@pytest.mark.parametrize(
    "status, body, verdict",
    [
        (400, "message content cannot be empty", "rejected"),
        (400, SMG_VALIDATION, "rejected"),
        (404, smg_error(404, "model_not_found", "No worker available for model 'org/M1'"), "measurement_failed"),
        (503, smg_error(503, "no_available_workers", "No available workers"), "measurement_failed"),
        (500, smg_error(500, "internal_error", "the worker went away"), "measurement_failed"),
        (502, b"upstream connect error", "measurement_failed"),
        (400, b"<html>400 Bad Request</html>", "measurement_failed"),
        (307, b"", "measurement_failed"),
        (400, {"error": {"message": "x", "type": "invalid_request_error"}}, "measurement_failed"),
        (400, {"error": {"message": "x", "type": "invalid_request_error", "code": True}}, "measurement_failed"),
    ],
    ids=[
        "refusal",
        "validation",
        "no worker",
        "unavailable",
        "internal",
        "proxy",
        "proxy 400",
        "redirect",
        "no code",
        "a code that is true",
    ],
)
def test_only_smgs_refusal_of_the_request_is_rejected(tmp_path, gateway, capsys, status, body, verdict):
    fixtures, report, known = tmp_path / "fixtures", tmp_path / "report.json", tmp_path / "known.toml"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    gateway.rejections["m1/render/hello"] = (status, body)
    if verdict == "rejected":
        code = smg_answer(status, body)[1]["error"]["code"]
    else:
        code = "bad_request"  # an answer about the setup is never excused, even listed as a refusal
    write_known(known, {"m1/render/hello": {"verdict": "rejected", "code": code, "reason": "SMG refuses it"}})

    assert verify(gateway, fixtures, "--known", str(known), "--report", str(report)) == (
        0 if verdict == "rejected" else 1
    )

    results = {case["id"]: case for case in json.loads(report.read_text())["cases"]}
    hello = results["m1/render/hello"]
    assert (hello["verdict"], hello["status"], hello["passed"]) == (verdict, status, verdict == "rejected")
    answered = smg_answer(status, body)[1]
    assert hello["body"] == (answered.decode() if isinstance(answered, bytes) else answered)
    if verdict == "rejected":
        assert (hello["code"], hello["message"]) == (answered["error"]["code"], answered["error"]["message"])
    else:
        assert "code" not in hello
        if status == 307:
            assert hello["message"] == "redirected to /elsewhere"
        elif isinstance(answered, dict):
            assert hello["message"] == answered["error"]["message"]
        else:
            assert hello["message"] == answered.decode()
    assert results["m1/render/budget"]["verdict"] == "match"
    out = capsys.readouterr().out
    assert f"{verdict} m1/render/hello: HTTP {status}" in out


def test_a_capture_line_is_compared_whatever_smg_answers_after_it(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    gateway.prompts["m1/render/budget"] = ([10, 12], "Hi")
    gateway.after_line["m1/render/budget"] = (500, smg_error(500, "internal_error", "the worker went away"))
    gateway.after_line["m1/render/hello"] = (400, "message content cannot be empty")

    assert verify(gateway, fixtures, "--report", str(report)) == 1

    results = {case["id"]: case for case in json.loads(report.read_text())["cases"]}
    budget, hello = results["m1/render/budget"], results["m1/render/hello"]
    assert (budget["verdict"], budget["index"], budget["lengths"]) == ("regression", 1, {"reference": 2, "smg": 2})
    assert (budget["status"], budget["message"]) == (500, "the worker went away")
    assert (hello["verdict"], hello["passed"], hello["status"]) == ("match", True, 400)
    out = capsys.readouterr().out
    assert "regression m1/render/budget: ids differ from index 1" in out and "SMG then answered HTTP 500" in out
    assert "match m1/render/hello: SMG then answered HTTP 400" in out


def test_an_answered_request_with_no_capture_line_is_missing(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    del gateway.prompts["m1/render/hello"]  # SMG answered, but the engine that captures never saw the request

    assert verify(gateway, fixtures, "--report", str(report)) == 1

    results = {case["id"]: case for case in json.loads(report.read_text())["cases"]}
    assert (results["m1/render/hello"]["verdict"], results["m1/render/hello"]["status"]) == ("missing", 200)
    assert results["m1/render/tools"]["verdict"] == "match"
    assert (
        "missing m1/render/hello: SMG answered, but no capture line carries this case's id" in capsys.readouterr().out
    )


def test_a_capture_file_nobody_writes_leaves_every_answered_case_missing(tmp_path, gateway):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    (tmp_path / "elsewhere.jsonl").touch()
    argv = ["--smg", gateway.url, "--capture", str(tmp_path / "elsewhere.jsonl"), "--fixtures", str(fixtures)]

    assert main(["verify", *argv, "--report", str(report)]) == 1
    written = json.loads(report.read_text())
    assert {case["verdict"] for case in written["cases"]} == {"missing"}
    assert written["capture"] == {"lines": 0, "joined": 0, "other": 0, "unfinished": False}


def test_a_request_id_with_the_prefill_decode_suffix_joins_to_its_case(tmp_path, gateway):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    for n, case_id in enumerate(cases):
        gateway.request_ids[case_id] = case_id + PD_SUFFIX.format(n)
    # A retry goes out under a fresh suffix, with the same prompt; its line joins the case too.
    retry = {"request_id": "m1/render/hello" + PD_SUFFIX.format(99), "input_ids": list(range(10, 30))}
    gateway.appended_after["m1/render/hello"] = json.dumps(retry) + "\n"

    assert verify(gateway, fixtures, "--report", str(report)) == 0

    results = {case["id"]: (case["verdict"], case["lines"]) for case in json.loads(report.read_text())["cases"]}
    assert results == {**dict.fromkeys(cases, ("match", 1)), "m1/render/hello": ("match", 2)}


@pytest.mark.parametrize("prefill_decode", [False, True], ids=["verbatim", "prefill-decode"])
def test_a_capture_line_joins_one_case_only(tmp_path, gateway, prefill_decode):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    # One case's id is the other's with a suffix of SMG's form after it.
    twin = "hello" + PD_SUFFIX.format(7)
    cases = write_model(fixtures, "m1", "org/M1", {"hello": CASES["hello"], twin: (BYE, [7, 8], "b")})
    gateway.serve(cases)
    del gateway.prompts["m1/render/hello"]  # so the only line that could join it is the twin's
    if prefill_decode:
        gateway.request_ids[f"m1/render/{twin}"] = f"m1/render/{twin}" + PD_SUFFIX.format(8)

    assert verify(gateway, fixtures, "--report", str(report)) == 1

    verdicts = {case["id"]: case["verdict"] for case in json.loads(report.read_text())["cases"]}
    assert verdicts == {"m1/render/hello": "missing", f"m1/render/{twin}": "match"}


def test_a_listed_case_passes_while_it_has_the_outcome_its_entry_states(tmp_path, gateway, capsys):
    fixtures, report, known = tmp_path / "fixtures", tmp_path / "report.json", tmp_path / "known.toml"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    write_model(fixtures, "m2", "org/M2", {"hello": (HELLO, [5, 6], "h")})
    gateway.serve(cases)
    gateway.prompts["m1/render/hello"] = ([10, 11], "<u>Hi.</u><a>")
    gateway.rejections["m1/render/tools"] = (400, "tool definitions need a name")
    write_known(
        known,
        {
            "m1/render/hello": {"verdict": "regression", "reason": "SMG drops the period"},
            "m1/render/tools": {"verdict": "rejected", "code": "bad_request", "reason": "SMG requires a name"},
            "m2/render/hello": {"verdict": "regression", "reason": "another model, not verified in this run"},
            "m1/parse/hello": {"verdict": "regression", "reason": "another kind, not verified in this run"},
        },
    )

    assert verify(gateway, fixtures, "--model", "org/M1", "--known", str(known), "--report", str(report)) == 0

    written = json.loads(report.read_text())
    results = {case["id"]: case for case in written["cases"]}
    hello, tools = results["m1/render/hello"], results["m1/render/tools"]
    assert (hello["verdict"], hello["passed"]) == ("regression", True)
    assert hello["known"] == {"issue": ISSUE, "verdict": "regression", "reason": "SMG drops the period"}
    assert (tools["verdict"], tools["code"], tools["passed"]) == ("rejected", "bad_request", True)
    assert [results["m1/render/budget"][key] for key in ("verdict", "known", "passed")] == ["match", None, True]
    assert written["known_outside_run"] == ["m1/parse/hello", "m2/render/hello"]
    out = capsys.readouterr().out
    assert f"known: SMG drops the period ({ISSUE})" in out
    assert "2 listed known differences are for cases this run does not verify" in out


@pytest.mark.parametrize(
    "entry, answer",
    [
        ({"verdict": "regression"}, "match"),
        ({"verdict": "regression"}, "rejected"),
        ({"verdict": "rejected", "code": "bad_request"}, "regression"),
        ({"verdict": "rejected", "code": "bad_request"}, "rejected for another code"),
        ({"verdict": "rejected", "code": 400}, "rejected"),
        ({"verdict": "regression"}, "missing"),
        ({"verdict": "rejected", "code": "bad_request"}, "measurement_failed"),
    ],
    ids=["fixed", "now refused", "now sent", "other code", "code of another type", "missing", "measurement failed"],
)
def test_a_listed_case_with_any_other_outcome_fails(tmp_path, gateway, capsys, entry, answer):
    fixtures, report, known = tmp_path / "fixtures", tmp_path / "report.json", tmp_path / "known.toml"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    if answer == "rejected":
        gateway.rejections["m1/render/hello"] = (400, "message content cannot be empty")
    elif answer == "rejected for another code":
        gateway.rejections["m1/render/hello"] = (400, smg_error(400, "invalid_tool", "tool definitions need a name"))
    elif answer == "regression":
        gateway.prompts["m1/render/hello"] = ([10, 11], "<u>Hi.</u><a>")
    elif answer == "missing":
        del gateway.prompts["m1/render/hello"]
    elif answer == "measurement_failed":
        gateway.rejections["m1/render/hello"] = (503, smg_error(503, "no_available_workers", "No available workers"))
    write_known(known, {"m1/render/hello": {**entry, "reason": "SMG differs here"}})

    assert verify(gateway, fixtures, "--known", str(known), "--report", str(report)) == 1

    results = {case["id"]: case for case in json.loads(report.read_text())["cases"]}
    assert results["m1/render/hello"]["passed"] is False
    assert [results[case_id]["passed"] for case_id in ("m1/render/budget", "m1/render/tools")] == [True, True]
    listed = entry["verdict"] + (f" {entry['code']}" if "code" in entry else "")
    line = next(line for line in capsys.readouterr().out.splitlines() if "m1/render/hello:" in line)
    assert f"listed as a known {listed}" in line
    if answer == "match":
        assert line.startswith("match m1/render/hello: matches, but it is listed") and "remove the entry" in line
    elif answer in ("missing", "measurement_failed"):
        assert line.endswith(f"but {answer.replace('_', ' ')} is about the setup")
    else:
        assert f"which this is not ({ISSUE})" in line


def test_a_listed_id_that_can_name_no_case_fails_the_run_on_its_own(tmp_path, gateway, capsys):
    fixtures, report, known = tmp_path / "fixtures", tmp_path / "report.json", tmp_path / "known.toml"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    gone = ["m1/render/renamed", "m9/render/hello", "m1/rendr/hello", "hello"]
    write_known(known, {case_id: {"verdict": "regression", "reason": "listed long ago"} for case_id in gone})

    assert verify(gateway, fixtures, "--known", str(known), "--report", str(report)) == 1

    written = json.loads(report.read_text())
    assert all(case["passed"] for case in written["cases"])
    assert (written["passed"], written["known_without_case"], written["known_outside_run"]) == (False, sorted(gone), [])
    out = capsys.readouterr().out
    assert all(f"known {case_id}: listed, but there is no such case" in out for case_id in gone)
    assert "3 pass, 0 fail, 4 listed without a case; failed" in out


@pytest.mark.parametrize(
    "entry, message",
    [
        ('"m1/render/hello" = "SMG drops the period"', "is not a table"),
        ('["m1/render/hello"]\nverdict = "regression"\nreason = "SMG drops the period"', "issue"),
        ('["m1/render/hello"]\nverdict = "regression"\nreason = "x"\nissue = "#12"', "issue"),
        (f'["m1/render/hello"]\nverdict = "missing"\nreason = "x"\nissue = "{ISSUE}"', "verdict"),
        (f'["m1/render/hello"]\nverdict = "rejected"\nreason = "x"\nissue = "{ISSUE}"', "code"),
        (f'["m1/render/hello"]\nverdict = "regression"\ncode = 400\nreason = "x"\nissue = "{ISSUE}"', "code"),
        (f'["m1/render/hello"]\nverdict = "regression"\nreason = ""\nissue = "{ISSUE}"', "reason"),
        (f'["m1/render/hello"]\nverdict = "regression"\nreason = "x"\nissue = "{ISSUE}"\nwhy = "y"', "why"),
    ],
    ids=[
        "a reason only",
        "no issue",
        "no issue link",
        "a setup verdict",
        "no code",
        "a stray code",
        "no reason",
        "a stray key",
    ],
)
def test_a_known_entry_states_its_verdict_its_reason_and_its_issue(tmp_path, gateway, capsys, entry, message):
    fixtures, known = tmp_path / "fixtures", tmp_path / "known.toml"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    known.write_text(entry + "\n")

    assert verify(gateway, fixtures, "--known", str(known)) == 2

    err = capsys.readouterr().err
    assert str(known) in err and "m1/render/hello" in err and message in err
    assert gateway.received == []


def test_the_json_report_and_the_junit_xml_carry_every_case_and_the_provenance(tmp_path, gateway):
    fixtures, out, known = tmp_path / "fixtures", tmp_path / "out", tmp_path / "known.toml"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    cases |= write_model(fixtures, "m2", "org/M2", {"hello": (HELLO, [5, 6], "h"), "bye": (BYE, [7], "b")})
    gateway.serve(cases)
    gateway.prompts["m1/render/hello"] = ([*range(10, 22), 500], "<u>Hello.</u><a>")
    gateway.rejections["m1/render/tools"] = (400, "tool definitions need a name")
    del gateway.prompts["m2/render/hello"]
    gateway.rejections["m2/render/bye"] = (400, "goodbyes are refused")
    write_known(
        known,
        {
            "m2/render/bye": {"verdict": "rejected", "code": "bad_request", "reason": "SMG refuses goodbyes"},
            "m2/render/gone": {"verdict": "regression", "reason": "a case the corpus dropped"},
        },
    )
    argv = [
        "--known",
        str(known),
        "--report",
        str(out / "a" / "report.json"),
        "--junit",
        str(out / "b" / "c" / "junit.xml"),
    ]

    assert verify(gateway, fixtures, *argv) == 1

    written = json.loads((out / "a" / "report.json").read_text())
    assert written["kind"] == "render"
    provenance = written["provenance"]
    assert provenance["bellwether"]["version"] == __version__
    assert provenance["bellwether"]["commit"] is None or re.fullmatch(
        "[0-9a-f]{40}", provenance["bellwether"]["commit"]
    )
    assert provenance["bellwether"]["dirty"] in (None, True, False)
    assert provenance["smg"] == gateway.url
    assert provenance["capture"] == str(gateway.capture)
    assert provenance["known"] == str(known)
    assert provenance["manifests"] == [
        {"model": "org/M1", "revision": revision_of("m1"), "path": str(fixtures / "m1" / "manifest.toml")},
        {"model": "org/M2", "revision": revision_of("m2"), "path": str(fixtures / "m2" / "manifest.toml")},
    ]
    assert written["summary"] == {
        "cases": 5,
        "match": 1,
        "regression": 1,
        "rejected": 2,
        "missing": 1,
        "measurement_failed": 0,
        "passed": 2,
        "failed": 3,
        "not_sent": 0,
    }
    assert (written["passed"], written["stopped"], written["not_sent"]) == (False, None, [])
    assert written["known_without_case"] == ["m2/render/gone"]
    entries = {entry["id"]: entry for entry in written["cases"]}
    assert list(entries) == sorted(cases)
    assert list(entries["m1/render/hello"])[:6] == ["id", "model", "set", "verdict", "passed", "known"]
    assert (entries["m2/render/bye"]["model"], entries["m2/render/bye"]["set"]) == ("org/M2", "common")

    suites = ET.parse(out / "b" / "c" / "junit.xml").getroot()
    assert suites.tag == "testsuites"
    assert {key: suites.get(key) for key in ("tests", "failures", "errors", "skipped")} == {
        "tests": "6",
        "failures": "3",
        "errors": "1",
        "skipped": "1",
    }
    outcome = {}
    for suite in suites:
        for testcase in suite:
            child = next(iter(testcase), None)
            outcome[testcase.get("name")] = None if child is None else (child.tag, child.get("type"))
    assert outcome == {
        "m1/render/budget": None,
        "m1/render/hello": ("failure", "regression"),
        "m1/render/tools": ("failure", "rejected"),
        "m2/render/bye": ("skipped", None),
        "m2/render/hello": ("error", "missing"),
        "m2/render/gone": ("failure", "known-without-case"),
    }
    assert [suite.get("name") for suite in suites] == ["org/M1 render", "org/M2 render", "known differences"]
    hello = suites.find("./testsuite/testcase[@name='m1/render/hello']/failure")
    assert "index 12" in hello.get("message")
    assert json.loads(hello.text)["window"]["smg"][-1] == 500
    assert "SMG refuses goodbyes" in suites.find("./testsuite/testcase[@name='m2/render/bye']/skipped").get("message")


def test_model_selects_the_manifests_to_verify(tmp_path, gateway):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    cases |= write_model(fixtures, "m2", "org/M2", {"hello": (HELLO, [5, 6], "h")})
    gateway.serve(cases)

    assert verify(gateway, fixtures, "--model", "org/M2", "--report", str(report)) == 0

    assert [received["body"]["rid"] for received in gateway.received] == ["m2/render/hello"]
    assert [manifest["model"] for manifest in json.loads(report.read_text())["provenance"]["manifests"]] == ["org/M2"]


def test_set_selects_the_render_sets_to_verify(tmp_path, gateway, capsys):
    fixtures, report, known = tmp_path / "fixtures", tmp_path / "report.json", tmp_path / "known.toml"
    plain = write_model(fixtures, "m1", "org/M1", {"hello": CASES["hello"]})
    bench = {"m1/render/bench-a": {**plain["m1/render/hello"], "id": "m1/render/bench-a", "request": BYE}}
    write_fixture_file(fixtures / "m1" / "render" / "bench.jsonl.zst", bench)
    gateway.serve({**plain, **bench})
    # A case of a set the run does not select may still be there: its entry is counted, not judged.
    write_known(known, {"m1/render/hello": {"verdict": "regression", "reason": "in the set not selected"}})
    (fixtures / "m1" / "sets.toml").write_text('[render.gone]\nform = "zstd"\ncases = 1\n')  # not selected

    assert verify(gateway, fixtures, "--set", "bench", "--known", str(known), "--report", str(report)) == 0

    assert [received["body"]["rid"] for received in gateway.received] == ["m1/render/bench-a"]
    written = json.loads(report.read_text())
    assert (written["known_without_case"], written["known_outside_run"]) == ([], ["m1/render/hello"])
    assert "render: 1 cases (1 match" in capsys.readouterr().out


def test_the_smg_url_is_shown_without_its_user_and_password(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    gateway.drops.add("m1/render/tools")
    url = gateway.url.replace("http://", "http://ops:s3cret@")

    assert (
        main(
            [
                "verify",
                "--smg",
                url,
                "--capture",
                str(gateway.capture),
                "--fixtures",
                str(fixtures),
                "--report",
                str(report),
            ]
        )
        == 2
    )

    written = report.read_text()
    assert json.loads(written)["provenance"]["smg"] == gateway.url
    captured = capsys.readouterr()
    assert "s3cret" not in written + captured.out + captured.err
    assert f"no answer from {gateway.url} for m1/render/tools" in captured.err


def test_a_model_without_render_cases_is_named_in_the_report(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    (fixtures / "m3").mkdir()
    (fixtures / "m3" / "manifest.toml").write_text(f'model = "org/M3"\nrevision = "{revision_of("m3")}"\n')

    assert verify(gateway, fixtures, "--report", str(report)) == 0

    assert json.loads(report.read_text())["models_without_cases"] == ["org/M3"]
    assert "org/M3: no render cases" in capsys.readouterr().out


def test_a_benchmark_set_stored_compressed_is_verified_like_a_plain_one(tmp_path, gateway, capsys):
    fixtures = tmp_path / "fixtures"
    plain = write_model(fixtures, "m1", "org/M1", {"hello": CASES["hello"]})
    bench = {
        "m1/render/bench-a": {**plain["m1/render/hello"], "id": "m1/render/bench-a", "request": BYE},
    }
    write_fixture_file(fixtures / "m1" / "render" / "bench.jsonl.zst", bench)
    write_fixture_file(fixtures / "m1" / "render" / "empty.jsonl.zst", {})  # a frame that declares no bytes
    gateway.serve({**plain, **bench})

    assert verify(gateway, fixtures) == 0

    assert [body["body"]["rid"] for body in gateway.received] == ["m1/render/bench-a", "m1/render/hello"]
    assert "render: 2 cases (2 match" in capsys.readouterr().out


def test_memory_does_not_grow_with_the_number_of_cases(tmp_path, gateway):
    # The prompt for "case <i>" is the ids from 1000 + i on: long enough that keeping each case would show.
    def prompt(i: int) -> tuple[list[int], str]:
        return list(range(1000 + i, 1300 + i)), f"<u>case {i}</u>"

    gateway.prompt_for = lambda body: prompt(int(body["messages"][0]["content"].split()[1]))
    gateway.keep_received = False
    gateway.models = {"org/M1"}
    peaks = {}
    for n in (20, 100, 400):  # the first run takes what a process allocates once
        fixtures, out = tmp_path / f"fixtures-{n}", tmp_path / f"out-{n}"
        (fixtures / "m1").mkdir(parents=True)
        (fixtures / "m1" / "manifest.toml").write_text(f'model = "org/M1"\nrevision = "{revision_of("m1")}"\n')
        cases = {}
        for i in range(n):
            ids, text = prompt(i)
            request = {"messages": [{"role": "user", "content": f"case {i}"}]}
            reference = {"source": "hf-template", "input_ids": ids, "text": text}
            case_id = f"m1/render/case-{i:04d}"
            cases[case_id] = {
                "id": case_id,
                "kind": "render",
                "model": "org/M1",
                "request": request,
                "reference": reference,
            }
        write_fixture_file(fixtures / "m1" / "render" / "bench.jsonl.zst", cases)
        argv = ["--report", str(out / "report.json"), "--junit", str(out / "junit.xml")]
        tracemalloc.start()
        try:
            assert verify(gateway, fixtures, *argv) == 0
            peaks[n] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert json.loads((out / "report.json").read_text())["summary"]["match"] == n

    per_case = (peaks[400] - peaks[100]) / 300
    assert per_case < 1024, f"the peak grows by {per_case:.0f} bytes for each case"


def test_a_set_that_cannot_be_read_again_stops_the_run_where_it_is(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    cases |= write_model(fixtures, "m2", "org/M2", {"hello": (HELLO, [5, 6], "h")})
    gateway.serve(cases)
    later = fixtures / "m2" / "render" / "common.jsonl"
    gateway.on_answer["m1/render/tools"] = lambda: later.write_text("not a line of a set\n")

    assert verify(gateway, fixtures, "--report", str(report)) == 2

    written = json.loads(report.read_text())
    assert [case["verdict"] for case in written["cases"]] == ["match"] * 3
    assert written["stopped"] == {"case": None, "error": f"{later}:1: not a JSON line"}
    assert written["not_sent"] == []
    captured = capsys.readouterr()
    assert f"{later}:1: not a JSON line" in captured.err
    assert "stopped at a set it could not read; 0 cases not sent" in captured.out


def test_a_set_is_read_whole_however_its_lines_end(tmp_path, gateway):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    path = fixtures / "m1" / "render" / "common.jsonl"
    path.write_text("\n\n".join(path.read_text().splitlines()))  # blank lines between, and no newline at the end

    assert verify(gateway, fixtures, "--report", str(report)) == 0

    assert [case["id"] for case in json.loads(report.read_text())["cases"]] == sorted(cases)


def test_lines_written_after_the_last_answer_are_counted(tmp_path, gateway, monkeypatch):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    counts = render.Capture.counts

    def late(capture):
        gateway.append(json.dumps({"request_id": "another-client", "input_ids": [1]}) + "\n")
        return counts(capture)

    monkeypatch.setattr(render.Capture, "counts", late)

    assert verify(gateway, fixtures, "--report", str(report)) == 0

    assert json.loads(report.read_text())["capture"] == {"lines": 4, "joined": 3, "other": 1, "unfinished": False}


def closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.parametrize(
    "problem",
    [
        "chunk plans",
        "unknown model",
        "no render cases",
        "manifest without a revision",
        "manifest not TOML",
        "manifest not readable",
        "group names no manifest",
        "a named model without render cases",
        "bad known file",
        "no gateway",
        "model not served",
        "no model served",
        "url ends in /v1",
        "capture not found",
        "capture is a directory",
        "set not fetched",
        "no such set",
        "set not zstd",
        "set cut short",
        "set listed but not there",
        "reference without ids",
        "id in two sets",
        "no manifests",
        "no render cases anywhere",
        "only empty sets",
        "sets.toml not TOML",
        "sets.toml set without a form",
        "set is a directory",
        "set corrupt",
        "line without an id",
        "case without a request",
        "known file not TOML",
        "model list not SMG's",
        "model ids not strings",
        "url not valid",
        "sets.toml render not a table",
        "set without a declared size",
    ],
)
def test_a_run_that_cannot_give_verdicts_exits_2_and_writes_no_report(tmp_path, gateway, capsys, problem):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    (fixtures / "m3").mkdir()
    (fixtures / "m3" / "manifest.toml").write_text(f'model = "org/M3"\nrevision = "{revision_of("m3")}"\n')
    url, capture, extra = gateway.url, gateway.capture, []
    if problem == "chunk plans":
        extra, message = ["--chunk-plan", "whole"], "--chunk-plan"
    elif problem == "unknown model":
        extra, message = ["--model", "org/Nope"], "no manifest under"
    elif problem == "no render cases":
        extra, message = ["--model", "org/M3"], "no render fixtures"
    elif problem == "manifest without a revision":
        (fixtures / "m3" / "manifest.toml").write_text('model = "org/M3"\n')
        message = "`revision` is required"
    elif problem == "manifest not TOML":
        # --model finds a manifest by reading each one in turn, so the error must name the file it stopped at.
        (fixtures / "m3" / "manifest.toml").write_text('model = "org/M3\n')
        extra, message = ["--model", "org/M3"], str(fixtures / "m3" / "manifest.toml")
    elif problem == "group names no manifest":
        manifest = f'model = "org/M3"\nrevision = "{revision_of("m3")}"\ngroup = "gone"\n'
        (fixtures / "m3" / "manifest.toml").write_text(manifest)
        message = "group gone names no manifest"
    elif problem == "manifest not readable":
        (fixtures / "m4" / "manifest.toml").mkdir(parents=True)
        message = str(fixtures / "m4" / "manifest.toml")
    elif problem == "a named model without render cases":
        extra, message = ["--model", "org/M1", "--model", "org/M3"], "no render fixtures under"
    elif problem == "capture not found":
        capture = tmp_path / "nowhere.jsonl"
        message = str(capture)
    elif problem == "capture is a directory":
        capture = tmp_path / "captures"
        capture.mkdir()
        message = str(capture)
    elif problem == "bad known file":
        (tmp_path / "known.toml").write_text('"m1/render/hello" = ""\n')
        extra, message = ["--known", str(tmp_path / "known.toml")], "known.toml"
    elif problem == "no gateway":
        url = f"http://127.0.0.1:{closed_port()}"
        message = f"no answer from {url}"
    elif problem == "model not served":
        gateway.models = {"org/Other"}
        message = "SMG serves no model org/M1; it serves org/Other"
    elif problem == "no model served":
        gateway.models = set()
        message = "/v1/models answered 503: No models available"
    elif problem == "url ends in /v1":
        url = f"{gateway.url}/v1"
        message = "/v1/v1/models answered 404; is --smg SMG's base URL, without /v1?"
    elif problem == "no manifests":
        fixtures = tmp_path / "empty"
        fixtures.mkdir()
        message = f"no manifests under {fixtures}"
    elif problem in ("no render cases anywhere", "only empty sets"):
        fixtures = tmp_path / "other"
        (fixtures / "m1" / "render").mkdir(parents=True)
        (fixtures / "m1" / "manifest.toml").write_text(f'model = "org/M1"\nrevision = "{revision_of("m1")}"\n')
        if problem == "only empty sets":
            (fixtures / "m1" / "render" / "common.jsonl").write_text("")
        message = f"no render fixtures under {fixtures} for org/M1"
    elif problem == "sets.toml not TOML":
        (fixtures / "m1" / "sets.toml").write_text("[render.common\n")
        message = str(fixtures / "m1" / "sets.toml")
    elif problem == "sets.toml set without a form":
        (fixtures / "m1" / "sets.toml").write_text("[render.common]\ncases = 3\n")
        message = "render.common has no form of plain or zstd"
    elif problem == "set is a directory":
        (fixtures / "m1" / "render" / "odd.jsonl").mkdir()
        message = str(fixtures / "m1" / "render" / "odd.jsonl")
    elif problem == "set corrupt":
        # A frame that declares 16 bytes, then a block of the type zstd reserves.
        (fixtures / "m1" / "render" / "bench.jsonl.zst").write_bytes(bytes.fromhex("28b52ffd2010070000"))
        message = "bench.jsonl.zst is not a zstd stream verify can read"
    elif problem == "line without an id":
        (fixtures / "m1" / "render" / "odd.jsonl").write_text('{"kind": "render"}\n')
        message = f"{fixtures / 'm1' / 'render' / 'odd.jsonl'}:1: not a case with an id"
    elif problem == "case without a request":
        (fixtures / "m1" / "render" / "odd.jsonl").write_text('{"id": "m1/render/odd", "reference": {}}\n')
        message = "m1/render/odd has no request"
    elif problem == "known file not TOML":
        (tmp_path / "known.toml").write_text("[[[\n")
        extra, message = ["--known", str(tmp_path / "known.toml")], str(tmp_path / "known.toml")
    elif problem == "model list not SMG's":
        gateway.models_body = {"object": "list"}
        message = "/v1/models answered 200 without SMG's list of models"
    elif problem == "model ids not strings":
        gateway.models_body = {"object": "list", "data": [{"id": 7}]}
        message = "/v1/models answered 200 without SMG's list of models"
    elif problem == "url not valid":
        url = "http://[::1"
        message = "no answer from http://[::1/v1/models: InvalidURL"
    elif problem == "sets.toml render not a table":
        (fixtures / "m1" / "sets.toml").write_text("render = 1\n")
        message = "[render] is not a table of sets"
    elif problem == "set without a declared size":
        line = json.dumps({**CASE_LINE, "id": "m1/render/bench-a"}) + "\n"
        compressed = zstandard.ZstdCompressor(write_content_size=False).compress(line.encode())
        (fixtures / "m1" / "render" / "bench.jsonl.zst").write_bytes(compressed)
        message = "its zstd frame does not declare its content size"
    elif problem == "set not fetched":
        pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 10\n"
        (fixtures / "m1" / "render" / "bench.jsonl.zst").write_text(pointer)
        message = "git lfs pull --include"
    elif problem == "no such set":
        extra, message = ["--set", "common", "--set", "nope"], "no render set nope for org/M1, org/M3"
    elif problem in ("set not zstd", "set cut short"):
        bench = fixtures / "m1" / "render" / "bench.jsonl.zst"
        write_fixture_file(bench, {"m1/render/bench-a": {**CASE_LINE, "id": "m1/render/bench-a"}})
        bench.write_bytes(b"(\xb5/\xfd" + bytes(16) if problem == "set not zstd" else bench.read_bytes()[:-4])
        message = str(bench)
    elif problem == "set listed but not there":
        (fixtures / "m1" / "sets.toml").write_text('[render.bench]\nform = "zstd"\ncases = 1\n')
        message = str(fixtures / "m1" / "render" / "bench.jsonl.zst")
    elif problem == "reference without ids":
        reference = {"source": "hf-template", "text": "<u>Bye.</u><a>"}
        write_fixture_file(
            fixtures / "m1" / "render" / "bye.jsonl", {"m1/render/bye": {**CASE_LINE, "reference": reference}}
        )
        message = "m1/render/bye"
    else:
        write_fixture_file(fixtures / "m1" / "render" / "again.jsonl", {"m1/render/hello": CASE_LINE})
        message = "m1/render/hello"
    argv = ["--smg", url, "--capture", str(capture), "--fixtures", str(fixtures), "--report", str(report)]

    assert main(["verify", *argv, *extra]) == 2

    assert message in capsys.readouterr().err
    assert not report.exists()
    assert gateway.received == [], "found before the first request"


@pytest.mark.parametrize("problem", ["no answer", "not a capture file", "capture line without ids"])
def test_a_run_stopped_partway_reports_what_was_answered_and_names_what_was_not_sent(
    tmp_path, gateway, capsys, problem
):
    fixtures, out, known = tmp_path / "fixtures", tmp_path / "out", tmp_path / "known.toml"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    cases |= write_model(fixtures, "m2", "org/M2", {"hello": (HELLO, [5, 6], "h"), "bye": (BYE, [7], "b")})
    gateway.serve(cases)
    gateway.prompts["m1/render/budget"] = ([10, 99], "Hi")  # a regression before the run stops
    if problem == "no answer":
        gateway.drops.add("m1/render/tools")
        message = f"no answer from {gateway.url} for m1/render/tools"
    elif problem == "not a capture file":
        gateway.appended_after["m1/render/tools"] = "INFO smg: request done\n"
        message = "is not JSON; is this the mock's capture file?"
    else:
        gateway.prompts["m1/render/tools"] = (None, "<t>Weather?")
        message = "has no list of integer input_ids"
    write_known(known, {"m2/render/bye": {"verdict": "rejected", "code": "bad_request", "reason": "not reached"}})
    argv = ["--known", str(known), "--report", str(out / "report.json"), "--junit", str(out / "junit.xml")]

    assert verify(gateway, fixtures, *argv) == 2

    written = json.loads((out / "report.json").read_text())
    verdicts = [(case["id"], case["verdict"]) for case in written["cases"]]
    assert verdicts == [("m1/render/budget", "regression"), ("m1/render/hello", "match")]
    assert written["stopped"]["case"] == "m1/render/tools"
    assert message in written["stopped"]["error"]
    if problem == "not a capture file":
        at = gateway.capture.read_bytes().index(b"INFO smg")
        assert f"the line at byte {at} is not JSON" in written["stopped"]["error"]
    assert written["not_sent"] == ["m2/render/bye", "m2/render/hello"]
    assert written["known_without_case"] == []
    assert (written["passed"], written["summary"]["not_sent"]) == (False, 2)
    outcome = {}
    for testcase in ET.parse(out / "junit.xml").getroot().iter("testcase"):
        child = next(iter(testcase), None)
        outcome[testcase.get("name")] = None if child is None else (child.tag, child.get("type"))
    assert outcome == {
        "m1/render/budget": ("failure", "regression"),
        "m1/render/hello": None,
        "m1/render/tools": ("error", "stopped"),
        "m2/render/bye": ("error", "not-sent"),
        "m2/render/hello": ("error", "not-sent"),
    }
    not_sent = {error.get("message") for error in ET.parse(out / "junit.xml").getroot().iter("error")}
    assert "not sent: the run stopped at m1/render/tools" in not_sent
    captured = capsys.readouterr()
    assert message in captured.err
    assert "regression m1/render/budget" in captured.out
    assert "stopped at m1/render/tools; 2 cases not sent" in captured.out
