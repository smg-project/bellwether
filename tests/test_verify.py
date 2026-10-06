"""``bellwether verify`` against a fake gateway.

The fake stands in for SMG and the mock worker behind it, as verify sees them: it answers
``/v1/chat/completions``, and for a request that reaches the engine it appends the capture line the mock
writes before SMG answers. The engine receives each case's own reference unless a test changes it.
"""

import json
import re
import socket
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bellwether import __version__
from bellwether.cli import main
from bellwether.record.fixtures import write_fixture_file


class FakeGateway:
    """An HTTP server on a free local port, answering each request by its ``rid``."""

    def __init__(self, capture):
        self.capture = capture
        self.prompts: dict[str, tuple[list[int] | None, str | None]] = {}  # rid -> what the engine receives
        # rid -> the status and the message SMG answers with; bytes are a plain body from something in front of it
        self.rejections: dict[str, tuple[int, str | bytes]] = {}
        self.appended_after: dict[str, str] = {}  # rid -> raw text another client appends after it
        self.received: list[dict] = []
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                gateway.received.append({"path": self.path, "body": body})
                status, payload = gateway.answer(body)
                plain = isinstance(payload, bytes)
                data = payload if plain else json.dumps(payload).encode()
                self.send_response(status)
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
            self.prompts[case_id] = (case["reference"]["input_ids"], case["reference"]["text"])

    def answer(self, body: dict) -> tuple[int, dict | bytes]:
        rid = body["rid"]
        if rid in self.rejections:
            status, message = self.rejections[rid]
            if isinstance(message, bytes):
                return status, message
            return status, {"error": {"type": "Bad Request", "code": "bad_request", "message": message, "param": None}}
        ids, text = self.prompts.get(rid, ([], ""))
        if rid in self.prompts:
            line = {
                "request_id": rid,
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


@pytest.fixture
def gateway(tmp_path):
    fake = FakeGateway(tmp_path / "capture.jsonl")
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


def write_model(fixtures, slug: str, model: str, cases: dict[str, tuple[dict, list[int], str]]) -> dict[str, dict]:
    """A manifest and one render set; ``cases`` maps a case name to its request and reference ids and text."""
    (fixtures / slug).mkdir(parents=True)
    (fixtures / slug / "manifest.toml").write_text(f'model = "{model}"\nrevision = "rev-{slug}"\n')
    lines = {}
    for name, (request, ids, text) in cases.items():
        case_id = f"{slug}/render/{name}"
        reference = {"source": "hf-template", "input_ids": ids, "text": text}
        lines[case_id] = {"id": case_id, "kind": "render", "model": model, "request": request, "reference": reference}
    write_fixture_file(fixtures / slug / "render" / "common.jsonl", lines)
    return lines


HELLO = {"messages": [{"role": "user", "content": "Hello."}]}
BYE = {"messages": [{"role": "user", "content": "Bye."}]}
CASES = {
    "hello": (HELLO, list(range(10, 30)), "<u>Hello.</u><a>"),
    "budget": ({"messages": [{"role": "user", "content": "Hi."}], "max_tokens": 7, "temperature": 0}, [10, 11], "Hi"),
    "tools": (
        {
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{"type": "function", "function": {}}],
            "max_completion_tokens": 5,
        },
        [10, 12, 13],
        "<t>Weather?",
    ),
}


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
        assert list(body.items())[: len(request)] == list(request.items()), "the request as recorded, in its key order"
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


def test_other_clients_lines_and_a_line_still_being_written_are_skipped(tmp_path, gateway):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    first, last = sorted(cases)[0], sorted(cases)[-1]
    gateway.appended_after[first] = json.dumps({"request_id": "another-client", "input_ids": [1]}) + "\n"
    gateway.appended_after[last] = '{"request_id": "another-client-2", "input_'

    assert verify(gateway, fixtures, "--report", str(report)) == 0
    assert {case["verdict"] for case in json.loads(report.read_text())["cases"]} == {"match"}


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
    assert hello["verdict"] == "differs"
    assert hello["index"] == 12
    assert hello["lengths"] == {"reference": 20, "smg": 21}
    assert hello["window"] == {
        "start": 4,
        "reference": list(range(14, 30)),
        "smg": [*range(14, 22), 500, 501, *range(23, 30)],
    }
    assert hello["text_equal"] is True
    assert "text" not in hello
    assert tools["verdict"] == "differs"
    assert (tools["index"], tools["lengths"]) == (3, {"reference": 3, "smg": 4})
    assert tools["window"] == {"start": 0, "reference": [10, 12, 13], "smg": [10, 12, 13, 14]}
    assert tools["text_equal"] is False
    assert tools["text"] == {"index": 11, "start": 0, "reference": "<t>Weather?", "smg": "<t>Weather?<g>"}
    assert (budget["verdict"], budget["index"], budget["text_equal"]) == ("differs", 1, None)
    out = capsys.readouterr().out
    assert "m1/render/hello" in out and "index 12" in out


def test_a_non_200_answer_is_recorded_as_rejected_with_its_status_and_message(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    gateway.rejections["m1/render/hello"] = (400, "message content cannot be empty")
    gateway.rejections["m1/render/tools"] = (502, b"upstream connect error")

    assert verify(gateway, fixtures, "--report", str(report)) == 1

    results = {case["id"]: case for case in json.loads(report.read_text())["cases"]}
    hello, tools = results["m1/render/hello"], results["m1/render/tools"]
    assert (hello["verdict"], hello["status"], hello["message"]) == ("rejected", 400, "message content cannot be empty")
    assert hello["body"]["error"]["message"] == "message content cannot be empty"
    assert (tools["verdict"], tools["status"], tools["message"]) == ("rejected", 502, "upstream connect error")
    assert tools["body"] == "upstream connect error"
    assert results["m1/render/budget"]["verdict"] == "match"
    assert "rejected m1/render/hello: HTTP 400: message content cannot be empty" in capsys.readouterr().out


def test_an_answered_request_with_no_capture_line_is_missing(tmp_path, gateway, capsys):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    del gateway.prompts["m1/render/hello"]  # SMG answered, but the engine that captures never saw the request

    assert verify(gateway, fixtures, "--report", str(report)) == 1

    results = {case["id"]: case for case in json.loads(report.read_text())["cases"]}
    assert (results["m1/render/hello"]["verdict"], results["m1/render/hello"]["status"]) == ("missing", 200)
    assert results["m1/render/tools"]["verdict"] == "match"
    assert "missing m1/render/hello" in capsys.readouterr().out


def test_a_capture_file_nobody_writes_leaves_every_answered_case_missing(tmp_path, gateway):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    argv = ["--smg", gateway.url, "--capture", str(tmp_path / "elsewhere.jsonl"), "--fixtures", str(fixtures)]

    assert main(["verify", *argv, "--report", str(report)]) == 1
    assert {case["verdict"] for case in json.loads(report.read_text())["cases"]} == {"missing"}


def test_listed_known_differences_pass_while_they_differ_or_are_rejected(tmp_path, gateway, capsys):
    fixtures, report, known = tmp_path / "fixtures", tmp_path / "report.json", tmp_path / "known.toml"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    gateway.prompts["m1/render/hello"] = ([10, 11], "<u>Hi.</u><a>")
    gateway.rejections["m1/render/tools"] = (400, "tool definitions need a name")
    known.write_text(
        '"m1/render/hello" = "SMG drops the period"\n'
        '"m1/render/tools" = "SMG requires a function name"\n'
        '"m2/render/hello" = "another model, not verified in this run"\n'
        '"m1/parse/hello" = "another kind, not verified in this run"\n'
    )

    assert verify(gateway, fixtures, "--known", str(known), "--report", str(report)) == 0

    results = {case["id"]: case for case in json.loads(report.read_text())["cases"]}
    assert [results["m1/render/hello"][key] for key in ("verdict", "known", "passed")] == [
        "differs",
        "SMG drops the period",
        True,
    ]
    assert [results["m1/render/tools"][key] for key in ("verdict", "known", "passed")] == [
        "rejected",
        "SMG requires a function name",
        True,
    ]
    assert [results["m1/render/budget"][key] for key in ("verdict", "known", "passed")] == ["match", None, True]
    assert "known: SMG drops the period" in capsys.readouterr().out


def test_a_listed_case_that_matches_or_is_missing_or_does_not_exist_fails(tmp_path, gateway, capsys):
    fixtures, report, known = tmp_path / "fixtures", tmp_path / "report.json", tmp_path / "known.toml"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    gateway.serve(cases)
    del gateway.prompts["m1/render/tools"]
    known.write_text(
        '"m1/render/hello" = "fixed in SMG since"\n'
        '"m1/render/tools" = "a missing capture line is the setup, not a difference"\n'
        '"m1/render/renamed" = "a case the corpus no longer has"\n'
    )

    assert verify(gateway, fixtures, "--known", str(known), "--report", str(report)) == 1

    written = json.loads(report.read_text())
    results = {case["id"]: case for case in written["cases"]}
    assert [results["m1/render/hello"][key] for key in ("verdict", "passed")] == ["match", False]
    assert [results["m1/render/tools"][key] for key in ("verdict", "passed")] == ["missing", False]
    assert written["known_without_case"] == ["m1/render/renamed"]
    out = capsys.readouterr().out
    assert "m1/render/hello" in out and "remove the entry" in out
    assert "m1/render/renamed" in out


def test_the_json_report_and_the_junit_xml_carry_every_case_and_the_provenance(tmp_path, gateway):
    fixtures, out, known = tmp_path / "fixtures", tmp_path / "out", tmp_path / "known.toml"
    cases = write_model(fixtures, "m1", "org/M1", CASES)
    cases |= write_model(fixtures, "m2", "org/M2", {"hello": (HELLO, [5, 6], "h"), "bye": (BYE, [7], "b")})
    gateway.serve(cases)
    gateway.prompts["m1/render/hello"] = ([*range(10, 22), 500], "<u>Hello.</u><a>")
    gateway.rejections["m1/render/tools"] = (400, "tool definitions need a name")
    del gateway.prompts["m2/render/hello"]
    gateway.rejections["m2/render/bye"] = (400, "goodbyes are refused")
    known.write_text('"m2/render/bye" = "SMG refuses goodbyes"\n"m2/render/gone" = "a case the corpus dropped"\n')
    argv = ["--known", str(known), "--report", str(out / "report.json"), "--junit", str(out / "junit.xml")]

    assert verify(gateway, fixtures, *argv) == 1

    written = json.loads((out / "report.json").read_text())
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
        {"model": "org/M1", "revision": "rev-m1", "path": str(fixtures / "m1" / "manifest.toml")},
        {"model": "org/M2", "revision": "rev-m2", "path": str(fixtures / "m2" / "manifest.toml")},
    ]
    assert written["summary"] == {
        "cases": 5,
        "match": 1,
        "differs": 1,
        "rejected": 2,
        "missing": 1,
        "passed": 2,
        "failed": 3,
    }
    assert written["passed"] is False
    assert written["known_without_case"] == ["m2/render/gone"]
    entries = {entry["id"]: entry for entry in written["cases"]}
    assert list(entries) == sorted(cases)
    assert list(entries["m1/render/hello"])[:6] == ["id", "model", "set", "verdict", "passed", "known"]
    assert (entries["m2/render/bye"]["model"], entries["m2/render/bye"]["set"]) == ("org/M2", "common")

    suites = ET.parse(out / "junit.xml").getroot()
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
        "m1/render/hello": ("failure", "differs"),
        "m1/render/tools": ("error", "rejected"),
        "m2/render/bye": ("skipped", None),
        "m2/render/hello": ("failure", "missing"),
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


def test_a_benchmark_set_stored_compressed_is_verified_like_a_plain_one(tmp_path, gateway, capsys):
    fixtures = tmp_path / "fixtures"
    plain = write_model(fixtures, "m1", "org/M1", {"hello": CASES["hello"]})
    bench = {
        "m1/render/bench-a": {**plain["m1/render/hello"], "id": "m1/render/bench-a", "request": BYE},
    }
    write_fixture_file(fixtures / "m1" / "render" / "bench.jsonl.zst", bench)
    gateway.serve({**plain, **bench})

    assert verify(gateway, fixtures) == 0

    assert [body["body"]["rid"] for body in gateway.received] == ["m1/render/bench-a", "m1/render/hello"]
    assert "render: 2 cases (2 match" in capsys.readouterr().out


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
        "bad known file",
        "no gateway",
        "not a capture file",
        "capture line without ids",
        "set not fetched",
    ],
)
def test_a_run_that_cannot_give_verdicts_exits_2_and_writes_no_report(tmp_path, gateway, capsys, problem):
    fixtures, report = tmp_path / "fixtures", tmp_path / "report.json"
    gateway.serve(write_model(fixtures, "m1", "org/M1", CASES))
    (fixtures / "m3").mkdir()
    (fixtures / "m3" / "manifest.toml").write_text('model = "org/M3"\nrevision = "r"\n')
    url, extra = gateway.url, []
    if problem == "chunk plans":
        extra, message = ["--chunk-plan", "whole"], "--chunk-plan"
    elif problem == "unknown model":
        extra, message = ["--model", "org/Nope"], "no manifest under"
    elif problem == "no render cases":
        extra, message = ["--model", "org/M3"], "no render fixtures"
    elif problem == "bad known file":
        (tmp_path / "known.toml").write_text('"m1/render/hello" = ""\n')
        extra, message = ["--known", str(tmp_path / "known.toml")], "known.toml"
    elif problem == "no gateway":
        url = f"http://127.0.0.1:{closed_port()}"
        message = f"no answer from {url}"
    elif problem == "set not fetched":
        pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 10\n"
        (fixtures / "m1" / "render" / "bench.jsonl.zst").write_text(pointer)
        message = "git lfs pull --include"
    elif problem == "not a capture file":
        gateway.appended_after["m1/render/hello"] = "INFO smg: request done\n"
        message = "not JSON"
    else:
        gateway.prompts["m1/render/hello"] = (None, "<u>Hello.</u><a>")
        message = "input_ids"
    argv = ["--smg", url, "--capture", str(gateway.capture), "--fixtures", str(fixtures), "--report", str(report)]

    assert main(["verify", *argv, *extra]) == 2

    assert message in capsys.readouterr().err
    assert not report.exists()
