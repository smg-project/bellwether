import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bellwether.cli import main
from bellwether.importers import corpus_sets, hf, swebench

COMMIT = "d26b2424437dabeeca94d7900b37d2df4410da0c"
ISSUE = "UsernameValidator allows a trailing newline\r\nDescription\n"
PATCH = (
    "diff --git a/django/core/validators.py b/django/core/validators.py\n"
    "--- a/django/core/validators.py\n"
    "+++ b/django/core/validators.py\n"
    "@@ -1 +1 @@\n"
    "-    regex = r'^[\\w.@+-]+$'\n"
    "+    regex = r'^[\\w.@+-]+\\Z'\t# \"Café\"\n"
)


def row(**fields) -> dict:
    base = {
        "repo": "django/django",
        "instance_id": "django__django-11099",
        "base_commit": COMMIT,
        "patch": PATCH,
        "problem_statement": ISSUE,
        "hints_text": "",
    }
    return {**base, **fields}


SUBMIT_PATCH = {
    "type": "function",
    "function": {
        "name": "submit_patch",
        "description": "Submit a patch that resolves the issue.",
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "A unified diff that resolves the issue, applied at the repository's base commit.",
                }
            },
            "required": ["patch"],
        },
    },
}


def test_the_request_is_the_repository_at_its_commit_the_issue_verbatim_and_one_tool():
    request = swebench.request_for(row())
    assert request == {
        "messages": [
            {"role": "system", "content": f"You are working on the django/django repository at commit {COMMIT}."},
            {"role": "user", "content": ISSUE},
        ],
        "tools": [SUBMIT_PATCH],
    }
    assert list(request) == ["messages", "tools"]
    assert list(request["tools"][0]) == ["type", "function"]
    assert list(request["tools"][0]["function"]) == ["name", "description", "parameters"]
    assert list(request["tools"][0]["function"]["parameters"]) == ["type", "properties", "required"]


def test_hints_follow_the_issue_after_a_fixed_separator_and_blank_hints_add_nothing():
    hinted = swebench.request_for(row(hints_text="Use \\Z, not $.\n"))
    assert hinted["messages"][1]["content"] == ISSUE + "\n\nHints:\nUse \\Z, not $.\n"
    for blank in ("", "\n", "\n\n", None):
        assert swebench.request_for(row(hints_text=blank))["messages"][1]["content"] == ISSUE


def test_the_call_carries_the_patch_as_its_one_argument_in_json_with_raw_unicode():
    assert swebench.call_message('a\t"b"\\c é\n') == {
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "submit_patch", "arguments": '{"patch": "a\\t\\"b\\"\\\\c é\\n"}'},
            }
        ],
    }
    assert json.loads(swebench.call_message(PATCH)["tool_calls"][0]["function"]["arguments"]) == {"patch": PATCH}


def test_the_content_is_the_patch_in_a_diff_block_whose_closing_fence_starts_a_line():
    assert swebench.content_message(PATCH) == {"content": "```diff\n" + PATCH + "```"}
    assert swebench.content_message("-a\n+b") == {"content": "```diff\n-a\n+b\n```"}


def write_parquet(path, rows: list[dict]):
    pq.write_table(pa.Table.from_pylist([{**r, "test_patch": "", "FAIL_TO_PASS": ["t"]} for r in rows]), path)
    return path


def test_rows_are_read_from_parquet_in_file_order_with_the_columns_the_import_uses(tmp_path):
    rows = [row(instance_id="django__django-11100", hints_text="Hint.\n"), row()]
    assert swebench.read_rows(write_parquet(tmp_path / "test.parquet", rows)) == rows


CARD = "---\ndataset_info:\n  features:\n  - name: patch\n---\n\nlicense: mit, in the text, does not count\n"


VERIFIED_ORIGIN = {
    "dataset": "swebench",
    "source": "hf://datasets/SWE-bench/SWE-bench_Verified@78f471bf655a3137b2e8a75af1501690ec009ec3",
    "sha256": "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25",
    "file": "data/test-00000-of-00001.parquet",
}
PYLINT = row(repo="pylint-dev/pylint", instance_id="pylint-dev__pylint-4551")


def test_sets_per_family_and_form_with_copyleft_rows_in_their_own_sets():
    test_row = row(instance_id="django__django-10097")
    sets = swebench.build_sets([(swebench.VERIFIED, [row(), PYLINT]), (swebench.TEST, [test_row])])
    assert sorted(sets) == [
        ("parse", "swebench-test-call"),
        ("parse", "swebench-test-content"),
        ("parse", "swebench-verified-call"),
        ("parse", "swebench-verified-call-copyleft"),
        ("parse", "swebench-verified-content"),
        ("parse", "swebench-verified-content-copyleft"),
        ("render", "swebench-test"),
        ("render", "swebench-verified"),
        ("render", "swebench-verified-copyleft"),
    ]
    origin = {**VERIFIED_ORIGIN, "row": "django__django-11099", "license": "BSD-3-Clause"}
    request = swebench.request_for(row())
    assert sets[("render", "swebench-verified")] == [
        {
            "name": "swebench-verified-django-django-11099",
            "request": request,
            "notes": "SWE-bench Verified django__django-11099",
            "origin": origin,
        }
    ]
    [call] = sets[("parse", "swebench-verified-call")]
    assert call == {
        "name": "swebench-verified-call-django-django-11099",
        "request": request,
        "message": swebench.call_message(PATCH),
        "notes": "SWE-bench Verified django__django-11099: the gold patch as one submit_patch call",
        "origin": origin,
    }
    assert list(call) == ["name", "request", "message", "notes", "origin"]
    assert list(call["origin"]) == ["dataset", "source", "sha256", "file", "row", "license"]
    [content] = sets[("parse", "swebench-verified-content")]
    assert content["name"] == "swebench-verified-content-django-django-11099"
    assert content["message"] == swebench.content_message(PATCH)
    assert content["notes"] == "SWE-bench Verified django__django-11099: the gold patch in a diff block"
    [copyleft] = sets[("render", "swebench-verified-copyleft")]
    assert copyleft["name"] == "swebench-verified-pylint-dev-pylint-4551"
    assert copyleft["origin"] == {**VERIFIED_ORIGIN, "row": "pylint-dev__pylint-4551", "license": "GPL-2.0"}
    [tested] = sets[("render", "swebench-test")]
    assert tested["name"] == "swebench-test-django-django-10097"
    assert tested["notes"] == "SWE-bench test django__django-10097"
    assert tested["origin"] == {
        "dataset": "swebench",
        "source": "hf://datasets/SWE-bench/SWE-bench@c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2",
        "sha256": "d4f5a245c75319fa8240c540674958c4d491e82edf274b144d43836bdcbc4567",
        "file": "data/test-00000-of-00001.parquet",
        "row": "django__django-10097",
        "license": "BSD-3-Clause",
    }


def names(lines: list[dict]) -> list[str]:
    return [line["name"] for line in lines]


def test_test_rows_that_are_verified_rows_are_left_to_the_verified_sets_and_reported():
    sources = [(swebench.VERIFIED, [row()]), (swebench.TEST, [row(), row(instance_id="django__django-10097")])]
    sets = swebench.build_sets(sources)
    assert names(sets[("render", "swebench-test")]) == ["swebench-test-django-django-10097"]
    assert names(sets[("parse", "swebench-test-call")]) == ["swebench-test-call-django-django-10097"]
    assert names(sets[("render", "swebench-verified")]) == ["swebench-verified-django-django-11099"]
    repeated: list[str] = []
    swebench.build_sets(sources, repeated=repeated)
    assert repeated == ["django__django-11099"]


def test_a_repeated_row_that_differs_from_the_first_stops_the_import():
    sources = [(swebench.VERIFIED, [row()]), (swebench.TEST, [row(hints_text="New.\n", patch="-a\n+b\n")])]
    with pytest.raises(
        ValueError,
        match="django__django-11099: the SWE-bench test row differs from the SWE-bench "
        "Verified row in patch, hints_text",
    ):
        swebench.build_sets(sources)


def test_rows_with_an_empty_patch_or_problem_statement_are_skipped_and_each_is_named():
    rows = [
        row(instance_id="django__django-1", patch=""),
        row(instance_id="django__django-2", problem_statement="\n"),
        row(),
    ]
    sets = swebench.build_sets([(swebench.VERIFIED, rows)])
    assert names(sets[("render", "swebench-verified")]) == ["swebench-verified-django-django-11099"]
    assert names(sets[("parse", "swebench-verified-content")]) == ["swebench-verified-content-django-django-11099"]
    skipped: list[tuple[str, str]] = []
    swebench.build_sets(
        [(swebench.VERIFIED, [*rows, row(instance_id="django__django-3", patch=None)])], skipped=skipped
    )
    assert skipped == [
        ("django__django-1", "the patch is empty"),
        ("django__django-2", "the problem statement is empty"),
        ("django__django-3", "the patch is empty"),
    ]


def test_a_repository_whose_license_was_not_reviewed_stops_the_import():
    rows = [row(repo="numpy/numpy", instance_id="numpy__numpy-1")]
    with pytest.raises(ValueError, match="numpy__numpy-1: numpy/numpy is not in the reviewed license table"):
        swebench.build_sets([(swebench.VERIFIED, rows)])


def test_two_rows_with_one_case_name_stop_the_import():
    twin = row(instance_id="Django__Django-11099")
    with pytest.raises(
        ValueError,
        match="rows 'django__django-11099' and 'Django__Django-11099' both become the case "
        "name swebench-verified-django-django-11099",
    ):
        swebench.build_sets([(swebench.VERIFIED, [row(), twin])])


def test_written_sets_check_clean_and_a_changed_missing_or_stale_file_is_reported(tmp_path):
    sets, corpus = swebench.build_sets([(swebench.VERIFIED, [row(), PYLINT])]), tmp_path / "corpus"
    (corpus / "render").mkdir(parents=True)
    for other in ("common.jsonl", "bfcl-simple-python.jsonl", "swebench-old.jsonl"):
        (corpus / "render" / other).write_text("{}\n")
    swebench.write_sets(sets, corpus)
    assert not (corpus / "render" / "swebench-old.jsonl").exists()
    assert (corpus / "render" / "common.jsonl").read_text() == "{}\n"
    assert (corpus / "render" / "bfcl-simple-python.jsonl").read_text() == "{}\n"
    text = (corpus / "parse" / "swebench-verified-call.jsonl").read_bytes().decode("utf-8")
    assert "Café" in text and "\\r\\n" in text and text.endswith("}\n") and text.count("\n") == 1
    assert swebench.check_sets(sets, corpus) == []
    (corpus / "parse" / "swebench-verified-call.jsonl").write_text("{}\n")
    (corpus / "parse" / "swebench-verified-content-copyleft.jsonl").unlink()
    (corpus / "render" / "swebench-stale.jsonl").write_text("{}\n")
    assert swebench.check_sets(sets, corpus) == [
        f"{corpus / 'parse' / 'swebench-verified-call.jsonl'}: differs from a fresh import",
        f"{corpus / 'parse' / 'swebench-verified-content-copyleft.jsonl'}: missing",
        f"{corpus / 'render' / 'swebench-stale.jsonl'}: no SWE-bench set writes it",
    ]


def fake_hub(tmp_path, verified: list[dict], test: list[dict], cards: dict[str, str] | None = None):
    """``hf.fetch`` over local files: each source's parquet file and card, recording what was asked for.

    Each source's card is ``CARD``, which states no license, unless ``cards`` gives its repository another.
    """
    files = {
        (swebench.VERIFIED.repo, swebench.VERIFIED.file): write_parquet(tmp_path / "verified.parquet", verified),
        (swebench.TEST.repo, swebench.TEST.file): write_parquet(tmp_path / "test.parquet", test),
    }
    for source in swebench.SOURCES:
        card = tmp_path / f"{source.family}-README.md"
        card.write_text((cards or {}).get(source.repo, CARD))
        files[(source.repo, "README.md")] = card
    calls: list[tuple] = []

    def fetch(repo, revision, filename, sha256):
        calls.append((repo, revision, filename, sha256))
        return files[(repo, filename)]

    return fetch, calls


def test_the_command_writes_then_checks_and_names_what_it_leaves_out(tmp_path, monkeypatch, capsys):
    empty = row(instance_id="django__django-1", patch="")
    fetch, calls = fake_hub(tmp_path, [row(), PYLINT, empty], [row(), row(instance_id="django__django-10097")])
    monkeypatch.setattr(hf, "fetch", fetch)
    corpus = tmp_path / "corpus"
    assert main(["import", "swebench", "--corpus", str(corpus), "--check"]) == 1
    assert main(["import", "swebench", "--corpus", str(corpus)]) == 0
    assert main(["import", "swebench", "--corpus", str(corpus), "--check"]) == 0
    out = capsys.readouterr().out
    assert f"{corpus / 'render' / 'swebench-verified.jsonl'}: 1 cases" in out
    assert f"{corpus / 'parse' / 'swebench-verified-call-copyleft.jsonl'}: 1 cases" in out
    assert f"{corpus / 'render' / 'swebench-test.jsonl'}: 1 cases" in out
    assert "no cases for 1 row(s) (django__django-1): the patch is empty" in out
    assert (
        "1 SWE-bench test row(s) are also SWE-bench Verified rows; each is imported once, in the Verified sets" in out
    )
    assert f"{corpus}: the SWE-bench sets equal a fresh import" in out
    cards = {(s.repo, s.revision, "README.md", s.card_sha256) for s in swebench.SOURCES}
    rows = {(s.repo, s.revision, s.file, s.sha256) for s in swebench.SOURCES}
    assert set(calls) == cards | rows and len(cards | rows) == 4


@pytest.mark.parametrize("source", swebench.SOURCES, ids=lambda source: source.family)
def test_the_command_refuses_a_dataset_card_that_states_a_license_and_writes_nothing(tmp_path, monkeypatch, source):
    cards = {source.repo: "---\nlicense: mit\n---\n"}
    fetch, _ = fake_hub(tmp_path, [row()], [row(instance_id="django__django-10097")], cards=cards)
    monkeypatch.setattr(hf, "fetch", fetch)
    with pytest.raises(
        ValueError,
        match=f"^{re.escape(source.repo)} README.md: the card's license is 'mit', not the reviewed None; review it",
    ):
        main(["import", "swebench", "--corpus", str(tmp_path / "corpus")])
    assert not (tmp_path / "corpus").exists()


def test_the_command_refuses_sets_past_the_limit_and_writes_nothing(tmp_path, monkeypatch):
    fetch, _ = fake_hub(tmp_path, [row()], [row(instance_id="django__django-10097")])
    monkeypatch.setattr(hf, "fetch", fetch)
    monkeypatch.setattr(corpus_sets, "LIMIT", 1)
    with pytest.raises(ValueError, match=r"^the swebench-\* sets take [0-9]+ bytes, past the 1 one source may take"):
        main(["import", "swebench", "--corpus", str(tmp_path / "corpus")])
    assert not (tmp_path / "corpus").exists()
