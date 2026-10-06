import hashlib
import io
import json

import pytest

from bellwether.cli import main
from bellwether.importers import github, gsm8k
from bellwether.record.corpus import read_cases

FILE = "data/test.jsonl"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def opener(data: bytes, calls: list[str]):
    """A stand-in for ``urlopen`` that serves ``data`` and records each URL: tests never touch the network."""

    def urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(data)

    return urlopen


def test_fetch_downloads_the_file_at_the_commit_into_the_cache(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(github, "urlopen", opener(b"rows", calls))
    path = github.fetch("acme", "sums", "c0ffee", FILE, sha(b"rows"), cache=tmp_path)
    assert path == tmp_path / "github" / "acme" / "sums" / "c0ffee" / "data" / "test.jsonl"
    assert path.read_bytes() == b"rows"
    assert calls == ["https://raw.githubusercontent.com/acme/sums/c0ffee/data/test.jsonl"]
    assert [p.name for p in path.parent.iterdir()] == ["test.jsonl"]


def cached(tmp_path, data: bytes):
    path = tmp_path / "github" / "acme" / "sums" / "c0ffee" / FILE
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def test_fetch_uses_a_cached_file_whose_hash_matches_without_the_network(tmp_path, monkeypatch):
    path = cached(tmp_path, b"rows")
    monkeypatch.setattr(github, "urlopen", lambda *a, **k: pytest.fail("no download expected"))
    assert github.fetch("acme", "sums", "c0ffee", FILE, sha(b"rows"), cache=tmp_path) == path


def test_fetch_rejects_a_download_that_is_not_the_pinned_one_and_caches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(github, "urlopen", opener(b"tampered", []))
    with pytest.raises(ValueError, match="is not the pinned"):
        github.fetch("acme", "sums", "c0ffee", FILE, sha(b"rows"), cache=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_replaces_a_cached_file_whose_hash_does_not_match(tmp_path, monkeypatch):
    cached(tmp_path, b"stale")
    monkeypatch.setattr(github, "urlopen", opener(b"rows", []))
    assert github.fetch("acme", "sums", "c0ffee", FILE, sha(b"rows"), cache=tmp_path).read_bytes() == b"rows"


MIT = b"MIT License\n\nCopyright (c) 2021 OpenAI\n"


def test_the_reviewed_license_passes(monkeypatch):
    monkeypatch.setattr(gsm8k, "LICENSE_SHA256", sha(MIT))
    gsm8k.check_license(MIT)


def test_a_license_file_other_than_the_reviewed_one_is_refused(monkeypatch):
    monkeypatch.setattr(gsm8k, "LICENSE_SHA256", sha(MIT))
    with pytest.raises(ValueError, match="is not the reviewed"):
        gsm8k.check_license(MIT + b"Additional terms apply.\n")


def test_a_license_that_is_not_mit_is_refused_even_when_its_hash_is_pinned(monkeypatch):
    apache = b"Apache License\nVersion 2.0, January 2004\n"
    monkeypatch.setattr(gsm8k, "LICENSE_SHA256", sha(apache))
    with pytest.raises(ValueError, match="not the MIT License"):
        gsm8k.check_license(apache)


def jsonl(rows: list[dict]) -> bytes:
    """A split's file: a JSON object per line, each ending in "\\n", with non-ASCII text written raw."""
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def test_rows_come_in_file_order_with_their_line_index_and_unicode_line_breaks_intact():
    rows = [
        {"question": "Clive opens a box.\u2028It holds 6 balls.\u0085How many?", "answer": "6\n#### 6"},
        {"question": "Two?", "answer": "2\n#### 2"},
    ]
    assert gsm8k.read_rows(jsonl(rows)) == [(0, rows[0]), (1, rows[1])]


SOLUTION = "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day."
ROW = {"question": "Janet’s ducks lay 16 eggs per day. How much does she make?", "answer": SOLUTION + "\n#### 18"}


def test_the_answer_splits_into_the_solution_as_written_and_the_final_answer():
    assert gsm8k.split_answer(ROW["answer"]) == (SOLUTION, "18")
    assert gsm8k.split_answer("1 + 1 = <<1+1=2>>2\n#### 2 ") == ("1 + 1 = <<1+1=2>>2", "2")


def test_an_answer_whose_last_line_is_not_the_final_answer_is_unusable():
    for answer in ["She makes $18 every day.", "#### 18\nShe makes $18 every day.", "So $18.\n####18"]:
        with pytest.raises(gsm8k.Unusable, match="last line is not '#### <final answer>'"):
            gsm8k.split_answer(answer)


TEST_ORIGIN = {
    "dataset": "gsm8k",
    "source": "github:openai/grade-school-math@3101c7d5072418e28b9008a6636bde82a006892c",
    "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "file": "grade_school_math/data/test.jsonl",
    "row": 0,
    "license": "MIT",
}
REQUEST = {"messages": [{"role": "user", "content": ROW["question"]}]}


def test_a_render_case_is_the_question_as_one_user_turn_with_its_origin():
    sets = gsm8k.build_sets({"test": jsonl([ROW])})
    assert sets[("render", "gsm8k-test")] == [
        {"name": "gsm8k-test-0", "request": REQUEST, "notes": "GSM8K test row 0", "origin": TEST_ORIGIN}
    ]
    [line] = sets[("render", "gsm8k-test")]
    assert list(line) == ["name", "request", "notes", "origin"]
    assert list(line["origin"]) == ["dataset", "source", "sha256", "file", "row", "license"]


def test_the_parse_cases_hold_the_solution_as_reasoning_or_the_whole_answer_as_content():
    sets = gsm8k.build_sets({"test": jsonl([ROW])})
    [reasoning] = sets[("parse", "gsm8k-test-reasoning")]
    [content] = sets[("parse", "gsm8k-test-content")]
    assert reasoning == {
        "name": "gsm8k-test-reasoning-0",
        "request": REQUEST,
        "message": {"reasoning_content": SOLUTION, "content": "18"},
        "notes": "GSM8K test row 0",
        "origin": TEST_ORIGIN,
    }
    assert content == {
        "name": "gsm8k-test-content-0",
        "request": REQUEST,
        "message": {"content": ROW["answer"]},
        "notes": "GSM8K test row 0",
        "origin": TEST_ORIGIN,
    }
    assert list(reasoning) == list(content) == ["name", "request", "message", "notes", "origin"]
    assert list(reasoning["message"]) == ["reasoning_content", "content"]


def test_rows_that_cannot_become_a_case_are_left_out_of_every_set_of_their_split_with_their_reason():
    no_final = {"question": "How much?", "answer": "She makes $18 every day."}
    no_question = {"question": " ", "answer": SOLUTION + "\n#### 18"}
    skipped: list[tuple[str, str]] = []
    sets = gsm8k.build_sets({"train": jsonl([ROW, no_final, no_question, ROW]), "test": jsonl([ROW])}, skipped=skipped)
    assert list(sets) == [
        ("render", "gsm8k-train"),
        ("parse", "gsm8k-train-reasoning"),
        ("parse", "gsm8k-train-content"),
        ("render", "gsm8k-test"),
        ("parse", "gsm8k-test-reasoning"),
        ("parse", "gsm8k-test-content"),
    ]
    assert [line["name"] for line in sets[("parse", "gsm8k-train-content")]] == [
        "gsm8k-train-content-0",
        "gsm8k-train-content-3",
    ]
    for key in [("render", "gsm8k-train"), ("parse", "gsm8k-train-reasoning")]:
        assert [line["origin"]["row"] for line in sets[key]] == [0, 3]
    assert sets[("render", "gsm8k-train")][0]["origin"]["file"] == "grade_school_math/data/train.jsonl"
    assert sets[("render", "gsm8k-train")][0]["origin"]["sha256"] == gsm8k.SHA256["train"]
    assert skipped == [
        ("train row 1", "the answer's last line is not '#### <final answer>'"),
        ("train row 2", "the question is empty"),
    ]


def test_written_sets_are_raw_unicode_and_a_rewrite_is_byte_identical(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "parse").mkdir(parents=True)
    (corpus / "parse" / "common.jsonl").write_text("{}\n")
    (corpus / "parse" / "gsm8k-old.jsonl").write_text("{}\n")
    written = gsm8k.write_sets(gsm8k.build_sets({"test": jsonl([ROW])}), corpus)
    assert sorted(path.relative_to(corpus).as_posix() for path in written) == [
        "parse/gsm8k-test-content.jsonl",
        "parse/gsm8k-test-reasoning.jsonl",
        "render/gsm8k-test.jsonl",
    ]
    assert not (corpus / "parse" / "gsm8k-old.jsonl").exists()
    assert (corpus / "parse" / "common.jsonl").read_text() == "{}\n"
    first = {path: path.read_bytes() for path in written}
    text = first[corpus / "render" / "gsm8k-test.jsonl"].decode("utf-8")
    assert "Janet’s" in text and text.endswith("\n")
    gsm8k.write_sets(gsm8k.build_sets({"test": jsonl([ROW])}), corpus)
    assert {path: path.read_bytes() for path in written} == first


def test_check_passes_on_a_fresh_import_and_names_each_set_file_that_differs(tmp_path):
    sets, corpus = gsm8k.build_sets({"test": jsonl([ROW])}), tmp_path / "corpus"
    gsm8k.write_sets(sets, corpus)
    assert gsm8k.check_sets(sets, corpus) == []
    (corpus / "parse" / "gsm8k-test-content.jsonl").write_text("{}\n")
    (corpus / "render" / "gsm8k-test.jsonl").unlink()
    (corpus / "render" / "gsm8k-stale.jsonl").write_text("{}\n")
    (corpus / "render" / "bfcl-other.jsonl").write_text("{}\n")
    assert gsm8k.check_sets(sets, corpus) == [
        f"{corpus / 'parse' / 'gsm8k-test-content.jsonl'}: differs from a fresh import",
        f"{corpus / 'render' / 'gsm8k-test.jsonl'}: missing",
        f"{corpus / 'render' / 'gsm8k-stale.jsonl'}: no GSM8K split writes it",
    ]


def test_check_names_a_set_file_that_is_not_utf_8_instead_of_stopping(tmp_path):
    sets, corpus = gsm8k.build_sets({"test": jsonl([ROW])}), tmp_path / "corpus"
    gsm8k.write_sets(sets, corpus)
    (corpus / "render" / "gsm8k-test.jsonl").write_bytes(b"\xff\n")
    assert gsm8k.check_sets(sets, corpus) == [f"{corpus / 'render' / 'gsm8k-test.jsonl'}: differs from a fresh import"]


def test_a_question_holding_unicode_line_breaks_reads_back_from_the_written_corpus(tmp_path):
    row = dict(ROW, question="Clive opens a box of balls.  \u2028It holds 6 blue balls.  \u2028How many?")
    gsm8k.write_sets(gsm8k.build_sets({"train": jsonl([row])}), tmp_path)
    for kind, name in [("render", "gsm8k-train"), ("parse", "gsm8k-train-reasoning"), ("parse", "gsm8k-train-content")]:
        [case] = read_cases(tmp_path / kind / f"{name}.jsonl")
        assert case.request == {"messages": [{"role": "user", "content": row["question"]}]}


TRAIN, TEST = "grade_school_math/data/train.jsonl", "grade_school_math/data/test.jsonl"


def serve(tmp_path, monkeypatch, files: dict[str, bytes]) -> None:
    """Stand in for ``github.fetch``: each pinned path is served from ``files``, and nothing is downloaded."""
    pins = {"LICENSE": sha(MIT), TRAIN: gsm8k.SHA256["train"], TEST: gsm8k.SHA256["test"]}

    def fetch(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit) == ("openai", "grade-school-math", gsm8k.COMMIT)
        assert (sha256, cache) == (pins[path], tmp_path)
        served = tmp_path / "served" / path
        served.parent.mkdir(parents=True, exist_ok=True)
        served.write_bytes(files[path])
        return served

    monkeypatch.setattr(github, "fetch", fetch)
    monkeypatch.setattr(gsm8k, "LICENSE_SHA256", sha(MIT))


def test_the_command_writes_then_checks_and_names_the_rows_it_leaves_out(tmp_path, monkeypatch, capsys):
    no_question = {"question": "", "answer": SOLUTION + "\n#### 18"}
    serve(tmp_path, monkeypatch, {"LICENSE": MIT, TRAIN: jsonl([ROW, no_question]), TEST: jsonl([ROW])})
    corpus = tmp_path / "corpus"
    argv = ["import", "gsm8k", "--corpus", str(corpus), "--cache", str(tmp_path)]
    assert main([*argv, "--check"]) == 1
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    out = capsys.readouterr().out
    assert f"{corpus / 'parse' / 'gsm8k-train-reasoning.jsonl'}: 1 cases" in out
    assert "no case for 1 row(s) (train row 1): the question is empty" in out
    assert f"{corpus}: the GSM8K sets equal a fresh import of {gsm8k.SOURCE}" in out


def test_the_command_refuses_a_license_that_is_not_mit_and_writes_nothing(tmp_path, monkeypatch):
    serve(tmp_path, monkeypatch, {"LICENSE": b"Apache License\n", TRAIN: jsonl([ROW]), TEST: jsonl([ROW])})
    with pytest.raises(ValueError, match="not the MIT License"):
        main(["import", "gsm8k", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path)])
    assert not (tmp_path / "corpus").exists()
