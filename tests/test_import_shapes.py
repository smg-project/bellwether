import hashlib
import json
import zipfile

import pytest

from bellwether.cli import main
from bellwether.count import counts
from bellwether.importers import bfcl, github, gsm8k, pypi, shapes
from bellwether.record import sets as set_tables
from bellwether.record.corpus import read_cases


def test_the_calls_interleave_their_categories_one_case_at_a_time():
    simple, parallel, live = ["s0", "s1", "s2"], ["p0"], ["l0", "l1"]
    assert shapes.interleave([simple, parallel, live]) == ["s0", "p0", "l0", "s1", "l1", "s2"]
    assert shapes.interleave([[], ["p0"]]) == ["p0"]


def test_pairs_go_by_index_and_cycle_the_shorter_list_up_to_the_size():
    assert shapes.pair(["c0", "c1", "c2"], ["t0"], size=10) == [("c0", "t0"), ("c1", "t0"), ("c2", "t0")]
    assert shapes.pair(["c0"], ["t0", "t1"], size=10) == [("c0", "t0"), ("c0", "t1")]
    assert shapes.pair(["c0", "c1", "c2"], ["t0", "t1"], size=2) == [("c0", "t0"), ("c1", "t1")]


def test_pairing_refuses_a_side_with_nothing_to_pair():
    for calls, texts in [([], ["t0"]), (["c0"], []), ([], [])]:
        with pytest.raises(ValueError, match="nothing to pair"):
            shapes.pair(calls, texts, size=10)


CALLS = [{"type": "function", "function": {"name": "ping", "arguments": "{}"}}]


def test_each_shape_is_named_by_the_parts_its_message_holds_in_order():
    messages = {
        "reasoning": {"reasoning_content": "R", "content": ""},
        "content": {"content": "C"},
        "reasoning-content": {"reasoning_content": "R", "content": "C"},
        "reasoning-calls": {"reasoning_content": "R", "content": "", "tool_calls": CALLS},
        "content-calls": {"content": "C", "tool_calls": CALLS},
        "reasoning-content-calls": {"reasoning_content": "R", "content": "C", "tool_calls": CALLS},
    }
    assert shapes.SHAPES == tuple(messages)
    for shape, expected in messages.items():
        message = shapes.message_for(shape, "R", "C", CALLS)
        assert message == expected
        assert list(message) == list(expected)


def jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


ORDER = {
    "name": "Cafe.order",
    "description": "Orders.",
    "parameters": {
        "type": "dict",
        "properties": {"drink": {"type": "string", "description": "D."}},
        "required": ["drink"],
    },
}


def bfcl_row(row_id: str, text: str) -> dict:
    return {"id": row_id, "question": [[{"role": "user", "content": text}]], "function": [ORDER]}


def bfcl_answer(row_id: str, *drinks: str) -> dict:
    return {"id": row_id, "ground_truth": [{"Cafe.order": {"drink": [drink]}} for drink in drinks]}


DATA = "bfcl_eval/data"
MEMBERS = {
    f"{DATA}/BFCL_v4_simple_python.json": [
        bfcl_row("simple_python_0", "Order a Café ☕"),
        bfcl_row("simple_python_1", "Tea"),
    ],
    f"{DATA}/possible_answer/BFCL_v4_simple_python.json": [
        bfcl_answer("simple_python_0", "Café ☕"),
        bfcl_answer("simple_python_1", "tea"),
    ],
    f"{DATA}/BFCL_v4_parallel.json": [bfcl_row("parallel_0", "Tea and coffee")],
    f"{DATA}/possible_answer/BFCL_v4_parallel.json": [bfcl_answer("parallel_0", "tea", "coffee")],
}


def fake_wheel(tmp_path, license: str = "Apache 2.0"):
    """The members of the pinned wheel the shapes import reads, under a METADATA header naming ``license``."""
    path = tmp_path / "bfcl.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            "bfcl_eval-2026.3.23.dist-info/METADATA", f"Metadata-Version: 2.1\nName: bfcl-eval\nLicense: {license}\n"
        )
        for name, rows in MEMBERS.items():
            wheel.writestr(name, jsonl(rows))
    return path


SOLUTION = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day."
TICKETS = "They sold 2,000 + 125 = <<2000+125=2125>>2,125 tickets."
TEST_FILE = jsonl(
    [
        {"question": "How much does Janet make?", "answer": SOLUTION + "\n#### 18"},
        {"question": "How many tickets?", "answer": TICKETS + "\n#### 2,125"},
    ]
)


def build(tmp_path, monkeypatch, size: int | None = None) -> dict[tuple[str, str], list[dict]]:
    monkeypatch.setattr(shapes, "CATEGORIES", ("simple_python", "parallel"))
    with zipfile.ZipFile(fake_wheel(tmp_path)) as wheel:
        return shapes.build_sets(wheel, TEST_FILE, size=size)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Cafe_order",
            "description": "Orders. Note that the provided function is in Python 3 syntax.",
            "parameters": {
                "type": "object",
                "properties": {"drink": {"type": "string", "description": "D."}},
                "required": ["drink"],
            },
        },
    }
]
BFCL_ORIGIN = {
    "dataset": "bfcl",
    "source": "pypi:bfcl-eval==2026.3.23",
    "sha256": "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a",
    "file": "bfcl_eval/data/BFCL_v4_simple_python.json",
    "answer_file": "bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json",
    "row": "simple_python_0",
    "license": "Apache-2.0",
}
GSM8K_ORIGIN = {
    "dataset": "gsm8k",
    "source": "github:openai/grade-school-math@3101c7d5072418e28b9008a6636bde82a006892c",
    "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "file": "grade_school_math/data/test.jsonl",
    "row": 0,
    "license": "MIT",
}


def call(drink: str) -> dict:
    return {
        "type": "function",
        "function": {"name": "Cafe_order", "arguments": json.dumps({"drink": drink}, ensure_ascii=False)},
    }


def test_a_case_takes_its_request_and_calls_from_bfcl_and_its_text_from_gsm8k(tmp_path, monkeypatch):
    [line, *_] = build(tmp_path, monkeypatch)[("parse", "shapes-reasoning-content-calls")]
    assert line == {
        "name": "shapes-reasoning-content-calls-0",
        "request": {
            "messages": [{"role": "user", "content": "Order a Café ☕"}],
            "temperature": 0.001,
            "store": False,
            "tools": TOOLS,
        },
        "message": {"reasoning_content": SOLUTION, "content": "The answer is 18.", "tool_calls": [call("Café ☕")]},
        "notes": "shape reasoning-content-calls: BFCL simple_python simple_python_0, GSM8K test row 0",
        "origin": {"dataset": "shapes", "parts": [BFCL_ORIGIN, GSM8K_ORIGIN]},
    }
    assert list(line) == ["name", "request", "message", "notes", "origin"]
    assert list(line["origin"]) == ["dataset", "parts"]
    assert [list(part) for part in line["origin"]["parts"]] == [list(BFCL_ORIGIN), list(GSM8K_ORIGIN)]


def test_every_set_holds_the_same_pairs_in_its_own_shape(tmp_path, monkeypatch):
    sets = build(tmp_path, monkeypatch)
    assert list(sets) == [
        ("parse", "shapes-reasoning"),
        ("parse", "shapes-content"),
        ("parse", "shapes-reasoning-content"),
        ("parse", "shapes-reasoning-calls"),
        ("parse", "shapes-content-calls"),
        ("parse", "shapes-reasoning-content-calls"),
    ]
    for (_, name), lines in sets.items():
        assert [line["name"] for line in lines] == [f"{name}-0", f"{name}-1", f"{name}-2"]
        rows = [tuple(part["row"] for part in line["origin"]["parts"]) for line in lines]
        assert rows == [("simple_python_0", 0), ("parallel_0", 1), ("simple_python_1", 0)]
        assert [line["request"]["messages"][0]["content"] for line in lines] == [
            "Order a Café ☕",
            "Tea and coffee",
            "Tea",
        ]
        assert all(line["request"]["tools"] == TOOLS for line in lines)
    assert sets[("parse", "shapes-content-calls")][1]["message"] == {
        "content": "The answer is 2,125.",
        "tool_calls": [call("tea"), call("coffee")],
    }
    assert sets[("parse", "shapes-reasoning")][2]["message"] == {"reasoning_content": SOLUTION, "content": ""}
    assert sets[("parse", "shapes-content")][1]["message"] == {"content": "The answer is 2,125."}


def test_the_size_caps_every_set(tmp_path, monkeypatch):
    assert [len(lines) for lines in build(tmp_path, monkeypatch, size=2).values()] == [2, 2, 2, 2, 2, 2]


SET_FILES = [
    "shapes-content-calls.jsonl",
    "shapes-content.jsonl",
    "shapes-reasoning-calls.jsonl",
    "shapes-reasoning-content-calls.jsonl",
    "shapes-reasoning-content.jsonl",
    "shapes-reasoning.jsonl",
]


def test_written_sets_replace_only_shapes_files_and_check_names_each_difference(tmp_path, monkeypatch):
    sets, parse = build(tmp_path, monkeypatch), tmp_path / "corpus" / "parse"
    parse.mkdir(parents=True)
    others = ["common.jsonl", "bfcl-simple-python.jsonl", "gsm8k-test-reasoning-content.jsonl"]
    for name in [*others, "shapes-old.jsonl"]:
        (parse / name).write_text("{}\n")
    written = shapes.write_sets(sets, tmp_path / "corpus")
    assert sorted(path.name for path in written) == SET_FILES
    assert sorted(path.name for path in parse.iterdir()) == sorted(others + SET_FILES)
    assert all((parse / name).read_text() == "{}\n" for name in others)
    assert "Café ☕" in (parse / "shapes-content-calls.jsonl").read_bytes().decode("utf-8")
    assert shapes.check_sets(sets, tmp_path / "corpus") == []
    (parse / "shapes-content.jsonl").write_text("{}\n")
    (parse / "shapes-reasoning-calls.jsonl").unlink()
    (parse / "shapes-stale.jsonl").write_text("{}\n")
    assert shapes.check_sets(sets, tmp_path / "corpus") == [
        f"{parse / 'shapes-content.jsonl'}: differs from a fresh import",
        f"{parse / 'shapes-reasoning-calls.jsonl'}: missing",
        f"{parse / 'shapes-stale.jsonl'}: no message shape writes it",
    ]


def test_the_written_cases_read_back_and_count_under_shapes(tmp_path, monkeypatch):
    corpus, fixtures = tmp_path / "corpus", tmp_path / "fixtures"
    shapes.write_sets(build(tmp_path, monkeypatch), corpus)
    for name in SET_FILES:
        cases = read_cases(corpus / "parse" / name)
        assert [[part["row"] for part in case.origin["parts"]] for case in cases] == [
            ["simple_python_0", 0],
            ["parallel_0", 1],
            ["simple_python_1", 0],
        ]
    (fixtures / "m1").mkdir(parents=True)
    (fixtures / "m1" / "manifest.toml").write_text('model = "org/M1"\nrevision = "r"\n')
    # count reads each model's sets.toml, never a fixture set: write the table as record does for two cases.
    recorded = "".join(json.dumps({"id": f"m1/parse/shapes-content-calls-{i}"}) + "\n" for i in (0, 2))
    tables = {("parse", "shapes-content-calls"): set_tables.entry("zstd", recorded, cases=2, rejected=0)}
    set_tables.write(fixtures / "m1" / set_tables.FILE, tables)
    assert counts(fixtures, corpus) == [{"model": "org/M1", "kind": "parse", "source": "shapes", "cases": 2}]


MIT = b"MIT License\n\nCopyright (c) 2021 OpenAI\n"
TEST = "grade_school_math/data/test.jsonl"


def serve(tmp_path, monkeypatch, wheel_license: str = "Apache 2.0", gsm8k_license: bytes = MIT) -> None:
    """Stand in for both fetchers, so the pinned files come from ``tmp_path`` and never the network.

    The import then reads the two categories the fake wheel holds.
    """
    wheel = fake_wheel(tmp_path, wheel_license)
    mit_sha256 = hashlib.sha256(MIT).hexdigest()

    def fetch_wheel(project, version, filename, sha256, cache):
        assert (project, version, filename, sha256, cache) == (
            "bfcl-eval",
            "2026.3.23",
            bfcl.WHEEL,
            bfcl.SHA256,
            tmp_path,
        )
        return wheel

    def fetch_file(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, cache) == ("openai", "grade-school-math", gsm8k.COMMIT, tmp_path)
        assert sha256 == {"LICENSE": mit_sha256, TEST: gsm8k.SHA256["test"]}[path]
        served = tmp_path / "served" / path
        served.parent.mkdir(parents=True, exist_ok=True)
        served.write_bytes({"LICENSE": gsm8k_license, TEST: TEST_FILE}[path])
        return served

    monkeypatch.setattr(pypi, "fetch", fetch_wheel)
    monkeypatch.setattr(github, "fetch", fetch_file)
    monkeypatch.setattr(gsm8k, "LICENSE_SHA256", mit_sha256)
    monkeypatch.setattr(shapes, "CATEGORIES", ("simple_python", "parallel"))


UNREADABLE = "not JSON: the shapes sets are built from the pinned sources, never from these sets\n"


def test_the_command_writes_then_checks_from_the_pinned_sources_alone(tmp_path, monkeypatch, capsys):
    serve(tmp_path, monkeypatch)
    monkeypatch.setattr(shapes, "SIZE", 2)
    corpus = tmp_path / "corpus"
    (corpus / "parse").mkdir(parents=True)
    committed = ["bfcl-simple-python.jsonl", "bfcl-parallel.jsonl", "gsm8k-test-reasoning-content.jsonl"]
    for name in committed:
        (corpus / "parse" / name).write_text(UNREADABLE)
    argv = ["import", "shapes", "--corpus", str(corpus), "--cache", str(tmp_path)]
    assert main([*argv, "--check"]) == 1
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    out = capsys.readouterr().out
    assert f"{corpus / 'parse' / 'shapes-content-calls.jsonl'}: 2 cases" in out
    assert f"{corpus}: the shapes sets equal a fresh import of {bfcl.SOURCE} and {gsm8k.SOURCE}" in out
    assert sorted(path.name for path in (corpus / "parse").iterdir()) == sorted(committed + SET_FILES)
    assert all((corpus / "parse" / name).read_text() == UNREADABLE for name in committed)


@pytest.mark.parametrize(
    ("wheel_license", "gsm8k_license", "refusal"),
    [
        ("MIT", MIT, "license 'MIT' is not the reviewed 'Apache 2.0'"),
        ("Apache 2.0", b"Apache License\nVersion 2.0, January 2004\n", "not the MIT License"),
    ],
    ids=["bfcl", "gsm8k"],
)
def test_the_command_refuses_a_source_under_another_license_and_writes_nothing(
    tmp_path, monkeypatch, wheel_license, gsm8k_license, refusal
):
    serve(tmp_path, monkeypatch, wheel_license, gsm8k_license)
    with pytest.raises(ValueError, match=refusal):
        main(["import", "shapes", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path)])
    assert not (tmp_path / "corpus").exists()
