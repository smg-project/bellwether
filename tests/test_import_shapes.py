import hashlib
import json
import zipfile

import pytest

from bellwether.cli import main
from bellwether.count import counts
from bellwether.importers import bfcl, corpus_sets, github, gsm8k, pypi, shapes
from bellwether.record import sets as set_tables
from bellwether.record.corpus import read_cases


def test_the_bfcl_cases_interleave_their_categories_one_at_a_time():
    simple, parallel, live = ["s0", "s1", "s2"], ["p0"], ["l0", "l1"]
    assert shapes.interleave([simple, parallel, live]) == ["s0", "p0", "l0", "s1", "l1", "s2"]
    assert shapes.interleave([[], ["p0"]]) == ["p0"]


def test_pairs_go_by_index_up_to_the_size():
    assert shapes.pair(["c0", "c1", "c2"], ["t0", "t1"], size=2) == [("c0", "t0"), ("c1", "t1")]
    assert shapes.pair(["c0", "c1"], ["t0", "t1", "t2"], size=2) == [("c0", "t0"), ("c1", "t1")]


def test_a_side_shorter_than_the_size_is_refused_rather_than_used_twice():
    for cases, texts, there_are in [
        (["c0"], ["t0", "t1"], "1 and 2"),
        (["c0", "c1"], ["t0"], "2 and 1"),
        ([], [], "0 and 0"),
    ]:
        with pytest.raises(
            ValueError, match=f"2 pairs need 2 BFCL parse cases and 2 GSM8K rows; there are {there_are}"
        ):
            shapes.pair(cases, texts, size=2)


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


def fake_wheel(tmp_path, license: str = "Apache 2.0", members: dict[str, list[dict]] = MEMBERS):
    """The members of the pinned wheel the shapes import reads, under a METADATA header naming ``license``."""
    path = tmp_path / "bfcl.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            "bfcl_eval-2026.3.23.dist-info/METADATA", f"Metadata-Version: 2.1\nName: bfcl-eval\nLicense: {license}\n"
        )
        for name, rows in members.items():
            wheel.writestr(name, jsonl(rows))
    return path


# Each solution as GSM8K writes it, one step per line with calculator annotations: the steps before the last line,
# which become the reasoning, and the last line, which becomes the content.
JANET_STEPS = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day."
JANET_LAST = "She makes 9 * 2 = $<<9*2=18>>18 every day."
TICKETS_STEPS = "They sold 2,000 tickets, then 125 more."
TICKETS_LAST = "They sold 2,000 + 125 = <<2000+125=2125>>2,125 tickets."
JANET = {"question": "How much does Janet make?", "answer": f"{JANET_STEPS}\n{JANET_LAST}\n#### 18"}
TICKETS = {"question": "How many tickets?", "answer": f"{TICKETS_STEPS}\n{TICKETS_LAST}\n#### 2,125"}
ROSES_STEPS = "Each bunch holds 12 roses, so 3 bunches hold 3 * 12 = <<3*12=36>>36 roses."
ROSES_LAST = "She gives away 36 - 6 = <<36-6=30>>30 roses."
ROSES = {"question": "How many roses does she give away?", "answer": f"{ROSES_STEPS}\n{ROSES_LAST}\n#### 30"}
# Three rows for the three BFCL parse cases of MEMBERS, so the default three pairs use each side once.
TEST_FILE = jsonl([JANET, TICKETS, ROSES])


def build(
    tmp_path, monkeypatch, size: int = 3, members=MEMBERS, test_file: bytes = TEST_FILE, skipped=None
) -> dict[tuple[str, str], list[dict]]:
    """The sets from the fake wheel and ``test_file``, for the categories it holds and ``size`` pairs."""
    monkeypatch.setattr(shapes, "CATEGORIES", ("simple_python", "parallel"))
    monkeypatch.setattr(shapes, "SIZE", size)
    with zipfile.ZipFile(fake_wheel(tmp_path, members=members)) as wheel:
        return shapes.build_sets(wheel, test_file, skipped=skipped)


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
        "message": {"reasoning_content": JANET_STEPS, "content": JANET_LAST, "tool_calls": [call("Café ☕")]},
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
        assert rows == [("simple_python_0", 0), ("parallel_0", 1), ("simple_python_1", 2)]
        assert [line["request"]["messages"][0]["content"] for line in lines] == [
            "Order a Café ☕",
            "Tea and coffee",
            "Tea",
        ]
        assert all(line["request"]["tools"] == TOOLS for line in lines)
    assert sets[("parse", "shapes-content-calls")][1]["message"] == {
        "content": TICKETS_LAST,
        "tool_calls": [call("tea"), call("coffee")],
    }
    assert sets[("parse", "shapes-reasoning")][2]["message"] == {"reasoning_content": ROSES_STEPS, "content": ""}
    assert sets[("parse", "shapes-content")][1]["message"] == {"content": TICKETS_LAST}


def test_the_reasoning_and_the_content_are_the_solution_split_at_its_last_line():
    assert shapes.split_solution(f"{JANET_STEPS}\n{JANET_LAST}") == (JANET_STEPS, JANET_LAST)
    # Lines before the last stay as written, a blank one included.
    assert shapes.split_solution("A = <<1+1=2>>2.\n\nB.\nSo C = <<2*2=4>>4.") == (
        "A = <<1+1=2>>2.\n\nB.",
        "So C = <<2*2=4>>4.",
    )
    for solution, reason in [
        ("Two and two make <<2+2=4>>4.", "the solution has no text before its last line"),
        (" \nTwo and two make <<2+2=4>>4.", "the solution has no text before its last line"),
        ("Two and two.\n ", "the solution's last line is empty"),
    ]:
        with pytest.raises(gsm8k.Unusable, match=reason):
            shapes.split_solution(solution)


def test_a_gsm8k_row_whose_solution_gives_no_reasoning_is_left_out_with_its_reason(tmp_path, monkeypatch):
    one_line = {"question": "How many?", "answer": "Two and two make <<2+2=4>>4.\n#### 4"}
    skipped: list[tuple[str, str]] = []
    sets = build(tmp_path, monkeypatch, test_file=jsonl([JANET, one_line, TICKETS, ROSES]), skipped=skipped)
    for lines in sets.values():
        assert [line["origin"]["parts"][1]["row"] for line in lines] == [0, 2, 3]
    assert skipped == [("GSM8K test row 1", "the solution has no text before its last line")]


def test_the_size_caps_every_set(tmp_path, monkeypatch):
    assert [len(lines) for lines in build(tmp_path, monkeypatch, size=2).values()] == [2, 2, 2, 2, 2, 2]


def test_a_source_with_fewer_usable_rows_than_the_pairs_stops_the_import(tmp_path, monkeypatch):
    # Three GSM8K rows, but one gives no reasoning: two texts for three pairs.
    one_line = {"question": "How many?", "answer": "Two and two make <<2+2=4>>4.\n#### 4"}
    with pytest.raises(ValueError, match="3 pairs need 3 BFCL parse cases and 3 GSM8K rows; there are 3 and 2"):
        build(tmp_path, monkeypatch, test_file=jsonl([JANET, one_line, TICKETS]))


NO_JUICE = "the ground truth calls Cafe.juice, which the row does not define"
# Five simple_python rows, of which simple_python_1 has no parse case: its ground truth calls a function the row lacks.
# Each row asks its own question, so no case repeats another.
LEFT_OUT = {
    f"{DATA}/BFCL_v4_simple_python.json": [bfcl_row(f"simple_python_{i}", f"Tea number {i}") for i in range(5)],
    f"{DATA}/possible_answer/BFCL_v4_simple_python.json": [
        bfcl_answer("simple_python_0", "tea"),
        {"id": "simple_python_1", "ground_truth": [{"Cafe.juice": {"fruit": ["apple"]}}]},
        *[bfcl_answer(f"simple_python_{i}", "tea") for i in (2, 3, 4)],
    ],
    f"{DATA}/BFCL_v4_parallel.json": [bfcl_row("parallel_0", "Tea")],
    f"{DATA}/possible_answer/BFCL_v4_parallel.json": [bfcl_answer("parallel_0", "tea")],
}
NO_PARALLEL_ANSWERS = {name: rows for name, rows in MEMBERS.items() if "possible_answer/BFCL_v4_parallel" not in name}


def test_a_bfcl_row_with_no_parse_case_and_the_cases_after_the_pairs_are_left_out_by_name(tmp_path, monkeypatch):
    skipped: list[tuple[str, str]] = []
    sets = build(tmp_path, monkeypatch, size=3, members=LEFT_OUT, skipped=skipped)
    # simple_python_1 falls inside the range the pairs use: simple_python_0 and simple_python_2 are both used.
    for lines in sets.values():
        rows = [line["origin"]["parts"][0]["row"] for line in lines]
        assert rows == ["simple_python_0", "parallel_0", "simple_python_2"]
    assert skipped == [
        ("BFCL simple_python_1", NO_JUICE),
        ("2 BFCL simple_python parse cases, simple_python_3 to simple_python_4", "after the first 3 pairs"),
    ]


def test_the_gsm8k_rows_after_the_pairs_are_left_out_as_one_run(tmp_path, monkeypatch):
    more = [
        {"question": f"Q{row}?", "answer": f"Add 0.\nSo {row} + 0 = <<{row}+0={row}>>{row}\n#### {row}"}
        for row in (3, 4)
    ]
    skipped: list[tuple[str, str]] = []
    build(tmp_path, monkeypatch, size=2, test_file=TEST_FILE + jsonl(more), skipped=skipped)
    assert skipped == [
        ("BFCL simple_python_1", "after the first 2 pairs"),
        ("3 GSM8K test rows, 2 to 4", "after the first 2 pairs"),
    ]


def test_a_category_without_parse_cases_stops_the_import(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no parse case in BFCL parallel"):
        build(tmp_path, monkeypatch, members=NO_PARALLEL_ANSWERS)


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
            ["simple_python_1", 2],
        ]
    (fixtures / "m1").mkdir(parents=True)
    (fixtures / "m1" / "manifest.toml").write_text(
        'model = "org/M1"\nrevision = "0123456789abcdef0123456789abcdef01234567"\n'
    )
    # count reads each model's sets.toml, never a fixture set: write the table as record does for two cases.
    recorded = "".join(json.dumps({"id": f"m1/parse/shapes-content-calls-{i}"}) + "\n" for i in (0, 2))
    tables = {("parse", "shapes-content-calls"): set_tables.entry("zstd", recorded, cases=2, rejected=0)}
    set_tables.write(fixtures / "m1" / set_tables.FILE, tables)
    assert counts(fixtures, corpus) == [{"model": "org/M1", "kind": "parse", "source": "shapes", "cases": 2}]


MIT = b"MIT License\n\nCopyright (c) 2021 OpenAI\n"
TEST = "grade_school_math/data/test.jsonl"


def serve(
    tmp_path,
    monkeypatch,
    wheel_license: str = "Apache 2.0",
    gsm8k_license: bytes = MIT,
    members=MEMBERS,
    test_file: bytes = TEST_FILE,
) -> None:
    """Stand in for both fetchers, so the pinned files come from ``tmp_path`` and never the network.

    The import then reads the two categories the fake wheel holds.
    """
    wheel = fake_wheel(tmp_path, wheel_license, members)
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
        served.write_bytes({"LICENSE": gsm8k_license, TEST: test_file}[path])
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
    assert f"{corpus / 'parse' / 'shapes-content-calls.jsonl'}: 2 cases, 2 distinct messages" in out
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


def test_the_command_names_every_row_it_leaves_out_with_its_reason(tmp_path, monkeypatch, capsys):
    serve(tmp_path, monkeypatch, members=LEFT_OUT)
    monkeypatch.setattr(shapes, "SIZE", 3)
    assert main(["import", "shapes", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines()[-2:] == [
        f"no case for BFCL simple_python_1: {NO_JUICE}",
        "no case for 2 BFCL simple_python parse cases, simple_python_3 to simple_python_4: after the first 3 pairs",
    ]


def test_the_command_stops_on_a_category_without_parse_cases_and_writes_nothing(tmp_path, monkeypatch):
    serve(tmp_path, monkeypatch, members=NO_PARALLEL_ANSWERS)
    with pytest.raises(ValueError, match="no parse case in BFCL parallel"):
        main(["import", "shapes", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path)])
    assert not (tmp_path / "corpus").exists()


def test_the_command_leaves_out_and_names_each_case_that_repeats_an_earlier_one(tmp_path, monkeypatch, capsys):
    # Each category's two rows ask the same and are answered the same, and the four GSM8K rows hold two solutions twice:
    # pairs 2 and 3 repeat pairs 0 and 1 in every set.
    members = {
        f"{DATA}/BFCL_v4_simple_python.json": [bfcl_row(f"simple_python_{i}", "Tea") for i in range(2)],
        f"{DATA}/possible_answer/BFCL_v4_simple_python.json": [
            bfcl_answer(f"simple_python_{i}", "tea") for i in range(2)
        ],
        f"{DATA}/BFCL_v4_parallel.json": [bfcl_row(f"parallel_{i}", "Coffee") for i in range(2)],
        f"{DATA}/possible_answer/BFCL_v4_parallel.json": [bfcl_answer(f"parallel_{i}", "coffee") for i in range(2)],
    }
    serve(tmp_path, monkeypatch, members=members, test_file=jsonl([JANET, TICKETS, JANET, TICKETS]))
    monkeypatch.setattr(shapes, "SIZE", 4)
    corpus = tmp_path / "corpus"
    argv = ["import", "shapes", "--corpus", str(corpus), "--cache", str(tmp_path)]
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    out = capsys.readouterr().out.splitlines()
    names = [shapes.set_name(shape) for shape in shapes.SHAPES]
    each = "2 cases, 2 left out as repeats, 2 distinct messages"
    assert out == [
        *[f"no case {name}-{i}: it repeats {name}-{i - 2}" for name in names for i in (2, 3)],
        *[f"{corpus / 'parse' / f'{name}.jsonl'}: {each}" for name in sorted(names)],
        f"{corpus}: 12 cases in the 6 shapes sets, 12 left out as repeats, 12 distinct messages",
        f"{corpus}: the shapes sets equal a fresh import of {bfcl.SOURCE} and {gsm8k.SOURCE}",
    ]
    assert [case.name for case in read_cases(corpus / "parse" / "shapes-content.jsonl")] == [
        "shapes-content-0",
        "shapes-content-1",
    ]


def test_the_command_stops_when_a_source_has_fewer_cases_than_the_pairs_and_writes_nothing(tmp_path, monkeypatch):
    serve(tmp_path, monkeypatch)
    monkeypatch.setattr(shapes, "SIZE", 4)
    with pytest.raises(ValueError, match="4 pairs need 4 BFCL parse cases and 4 GSM8K rows; there are 3 and 3"):
        main(["import", "shapes", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path)])
    assert not (tmp_path / "corpus").exists()


def test_the_command_hands_the_built_sets_to_the_shared_rule_and_keeps_what_it_keeps(tmp_path, monkeypatch, capsys):
    handed: list[dict] = []

    def leave_out_repeats(sets):
        """Keep each set's first case and report the others as repeats of it."""
        handed.append({key: [line["name"] for line in lines] for key, lines in sets.items()})
        kept = {key: lines[:1] for key, lines in sets.items()}
        return kept, [(line["name"], lines[0]["name"]) for lines in sets.values() for line in lines[1:]]

    monkeypatch.setattr(corpus_sets, "leave_out_repeats", leave_out_repeats)
    serve(tmp_path, monkeypatch)
    monkeypatch.setattr(shapes, "SIZE", 3)
    corpus = tmp_path / "corpus"
    argv = ["import", "shapes", "--corpus", str(corpus), "--cache", str(tmp_path)]
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    names = [shapes.set_name(shape) for shape in shapes.SHAPES]
    assert handed == [{("parse", name): [f"{name}-{i}" for i in range(3)] for name in names}] * 2
    assert [case.name for case in read_cases(corpus / "parse" / "shapes-content.jsonl")] == ["shapes-content-0"]
    assert capsys.readouterr().out.splitlines()[:12] == [
        f"no case {name}-{i}: it repeats {name}-0" for name in names for i in (1, 2)
    ]
