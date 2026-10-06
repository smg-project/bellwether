import pytest
import zstandard

from bellwether import storage
from bellwether.importers import corpus_sets


def render(name: str, request: dict) -> dict:
    """A render line as an importer builds it; its notes and origin differ from every other line's."""
    return {"name": name, "request": request, "notes": f"row {name}", "origin": {"dataset": "test", "row": name}}


def parse(name: str, request: dict, message: dict) -> dict:
    origin = {"dataset": "test", "row": name}
    return {"name": name, "request": request, "message": message, "notes": f"row {name}", "origin": origin}


def ask(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def names(sets: dict) -> dict:
    return {key: [line["name"] for line in lines] for key, lines in sets.items()}


def test_a_render_case_whose_request_equals_an_earlier_ones_is_left_out_naming_the_case_kept():
    sets = {
        ("render", "s"): [
            render("a", ask("Hi")),
            render("b", ask("Hi")),
            render("c", ask("Bye")),
            render("d", ask("Hi")),
        ]
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert kept == {("render", "s"): [render("a", ask("Hi")), render("c", ask("Bye"))]}
    assert repeats == [("b", "a"), ("d", "a")]
    assert names(sets) == {("render", "s"): ["a", "b", "c", "d"]}


def test_a_parse_case_repeats_an_earlier_one_only_when_its_message_is_equal_too():
    sets = {
        ("parse", "s"): [
            parse("a", ask("1 + 1?"), {"content": "2"}),
            parse("b", ask("1 + 1?"), {"content": "two"}),
            parse("c", ask("1 + 1?"), {"content": "2"}),
        ]
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert names(kept) == {("parse", "s"): ["a", "b"]}
    assert repeats == [("c", "a")]


def tool(*params: str) -> dict:
    properties = {param: {"type": "string"} for param in params}
    return {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": properties}}}


def test_key_order_counts_since_a_template_can_see_it():
    sets = {
        ("render", "s"): [
            render("a", {**ask("Hi"), "tools": [tool("x", "y")]}),
            render("b", {**ask("Hi"), "tools": [tool("y", "x")]}),
            render("c", {**ask("Hi"), "tools": [tool("x", "y")]}),
        ],
        ("parse", "s"): [
            parse("a", ask("Hi"), {"reasoning_content": "Greet.", "content": "Hello"}),
            parse("b", ask("Hi"), {"content": "Hello", "reasoning_content": "Greet."}),
        ],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert names(kept) == {("render", "s"): ["a", "b"], ("parse", "s"): ["a", "b"]}
    assert repeats == [("c", "a")]


def test_the_set_built_first_keeps_the_case_whatever_the_sets_are_named():
    sets = {
        ("render", "zeta"): [render("z", ask("Hi"))],
        ("render", "alpha"): [render("a", ask("Hi")), render("b", ask("Bye"))],
        ("render", "beta"): [render("c", ask("Bye"))],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert list(kept) == list(sets)
    assert names(kept) == {("render", "zeta"): ["z"], ("render", "alpha"): ["b"], ("render", "beta"): []}
    assert repeats == [("a", "z"), ("c", "b")]


def test_a_render_case_and_a_parse_case_never_repeat_each_other():
    sets = {("render", "s"): [render("a", ask("Hi"))], ("parse", "s"): [parse("a", ask("Hi"), {"content": "Hello"})]}
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert kept == sets and repeats == []


def test_when_nothing_repeats_the_sets_come_back_whole_and_the_list_is_empty():
    sets = {
        ("render", "s"): [render("a", ask("Hi")), render("b", ask("Bye"))],
        ("parse", "s"): [parse("c", ask("1 + 1?"), {"content": "2"}), parse("d", ask("2 + 2?"), {"content": "4"})],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert kept == sets and list(kept) == list(sets) and repeats == []


def test_the_report_names_each_repeat_then_each_set_by_kind_and_name_with_its_count_then_the_total(tmp_path, capsys):
    sets = {
        ("render", "zeta"): [render("a", ask("Hi")), render("b", ask("Hi"))],
        ("render", "alpha"): [render("c", ask("Hi")), render("d", ask("Bye"))],
        ("parse", "zeta"): [parse("e", ask("Hi"), {"content": "Hello"}), parse("f", ask("Hi"), {"content": "Hello"})],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    corpus_sets.report("Test", sets, kept, repeats, tmp_path)
    assert capsys.readouterr().out.splitlines() == [
        "no case b: it repeats a",
        "no case c: it repeats a",
        "no case f: it repeats e",
        f"{tmp_path / 'parse' / 'zeta.jsonl'}: 1 cases, 1 left out as repeats, 1 distinct messages",
        f"{tmp_path / 'render' / 'alpha.jsonl'}: 1 cases, 1 left out as repeats",
        f"{tmp_path / 'render' / 'zeta.jsonl'}: 1 cases, 1 left out as repeats",
        f"{tmp_path}: 3 cases in the 3 Test sets, 3 left out as repeats, 1 distinct messages",
    ]


def test_the_report_counts_distinct_messages_as_written_per_parse_set_and_once_across_the_sets(tmp_path, capsys):
    # Cases that are not repeats can carry the same message, which is all a parser sees: the counts keep them apart.
    sets = {
        ("parse", "s"): [
            parse("a", ask("Hi"), {"content": "Hello"}),
            parse("b", ask("Hey"), {"content": "Hello"}),
            parse("c", ask("Hi"), {"reasoning_content": "Greet.", "content": "Hello"}),
            parse("d", ask("Hi"), {"content": "Hello", "reasoning_content": "Greet."}),
        ],
        ("parse", "t"): [parse("e", ask("Yo"), {"content": "Hello"}), parse("f", ask("Yo"), {"content": "Bye"})],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    corpus_sets.report("Test", sets, kept, repeats, tmp_path)
    assert capsys.readouterr().out.splitlines() == [
        f"{tmp_path / 'parse' / 's.jsonl'}: 4 cases, 3 distinct messages",
        f"{tmp_path / 'parse' / 't.jsonl'}: 2 cases, 2 distinct messages",
        f"{tmp_path}: 6 cases in the 2 Test sets, 0 left out as repeats, 4 distinct messages",
    ]


def test_each_reason_is_printed_once_naming_every_row_it_left_out_and_what_the_rows_do_not_get(capsys):
    skipped = [(f"row {row}", "no final answer" if row == 2 else "empty") for row in range(1, 6)]
    corpus_sets.report_skipped(skipped)
    corpus_sets.report_skipped([("simple_java_1", "not strings")], "parse case")
    assert capsys.readouterr().out.splitlines() == [
        "no case for 4 row(s) (row 1, row 3, row 4, row 5): empty",
        "no case for 1 row(s) (row 2): no final answer",
        "no parse case for 1 row(s) (simple_java_1): not strings",
    ]


ZSTD_MAGIC = bytes.fromhex("28b52ffd")


def an_import() -> dict:
    """The sets of one import, ``x-``: a render set and a parse set, as an importer builds them."""
    return {
        ("render", "x-a"): [render("x-a-0", ask("Hi")), render("x-a-1", ask("Bye"))],
        ("parse", "x-a"): [parse("x-a-2", ask("Hi"), {"content": "Hello"})],
    }


def plain_size(sets: dict) -> int:
    return sum(len(corpus_sets.text(lines).encode("utf-8")) for lines in sets.values())


def files(root) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def test_the_limit_is_fifty_megabytes():
    assert corpus_sets.LIMIT == 50_000_000


def test_an_import_within_the_limit_is_written_as_plain_json_lines(tmp_path, monkeypatch):
    sets = an_import()
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets))
    written = corpus_sets.write(sets, tmp_path, "x-")
    assert written == [tmp_path / "parse" / "x-a.jsonl", tmp_path / "render" / "x-a.jsonl"]
    assert (tmp_path / "render" / "x-a.jsonl").read_bytes() == corpus_sets.text(sets["render", "x-a"]).encode()


def test_an_import_past_the_limit_writes_every_set_compressed_holding_the_same_lines(tmp_path, monkeypatch):
    sets = an_import()
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets) - 1)
    written = corpus_sets.write(sets, tmp_path, "x-")
    assert written == [tmp_path / "parse" / "x-a.jsonl.zst", tmp_path / "render" / "x-a.jsonl.zst"]
    for (kind, name), lines in sets.items():
        path = tmp_path / kind / f"{name}.jsonl.zst"
        assert path.read_bytes()[:4] == ZSTD_MAGIC
        assert storage.plain_bytes(path) == corpus_sets.text(lines).encode("utf-8")


def test_the_limit_counts_bytes_not_characters(tmp_path, monkeypatch):
    sets = {("render", "x-a"): [render("x-a-0", ask("\xe9" * 100))]}
    monkeypatch.setattr(corpus_sets, "LIMIT", len(corpus_sets.text(sets["render", "x-a"])))
    assert corpus_sets.write(sets, tmp_path, "x-") == [tmp_path / "render" / "x-a.jsonl.zst"]


def test_an_imports_other_files_do_not_count_toward_the_limit(tmp_path, monkeypatch):
    sets = {("render", "x-a"): [render("x-a-0", ask("Hi"))]}
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets))
    written = corpus_sets.write(sets, tmp_path, "x-", files={"licenses/x-LICENSE": b"MIT License\n"})
    assert written == [tmp_path / "render" / "x-a.jsonl", tmp_path / "licenses" / "x-LICENSE"]


def test_an_import_that_changes_form_keeps_one_file_per_set_and_leaves_other_sets_alone(tmp_path, monkeypatch):
    sets = an_import()
    for other in ("render/common.jsonl", "render/y-a.jsonl.zst"):  # a hand-written set, and another import's
        (tmp_path / other).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / other).write_bytes(b"not this import's\n")
    others = ["render/common.jsonl", "render/y-a.jsonl.zst"]
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets) - 1)
    corpus_sets.write(sets, tmp_path, "x-")
    assert files(tmp_path) == sorted(["parse/x-a.jsonl.zst", "render/x-a.jsonl.zst", *others])
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets))
    corpus_sets.write(sets, tmp_path, "x-")
    assert files(tmp_path) == sorted(["parse/x-a.jsonl", "render/x-a.jsonl", *others])


def test_a_write_that_fails_part_way_removes_none_of_the_old_files(tmp_path, monkeypatch):
    sets = an_import()
    corpus_sets.write(sets, tmp_path, "x-")
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets) - 1)
    write = storage.write
    written: list = []

    def fail_on_the_second(path, data):
        if written:
            raise OSError("no space left on device")
        written.append(path)
        write(path, data)

    monkeypatch.setattr(storage, "write", fail_on_the_second)
    with pytest.raises(OSError, match="no space left"):
        corpus_sets.write(sets, tmp_path, "x-")
    assert files(tmp_path) == ["parse/x-a.jsonl", "parse/x-a.jsonl.zst", "render/x-a.jsonl"]


def test_the_report_names_each_set_file_in_the_form_the_import_writes(tmp_path, monkeypatch, capsys):
    sets = an_import()
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets) - 1)
    corpus_sets.report("X", sets, sets, [], tmp_path)
    assert capsys.readouterr().out.splitlines()[:2] == [
        f"{tmp_path / 'parse' / 'x-a.jsonl.zst'}: 1 cases, 1 distinct messages",
        f"{tmp_path / 'render' / 'x-a.jsonl.zst'}: 2 cases",
    ]


def check(sets: dict, root) -> list[str]:
    return corpus_sets.check(sets, root, "x-", "x part")


@pytest.mark.parametrize("past", [False, True], ids=["plain", "compressed"])
def test_check_passes_an_import_as_write_left_it_in_either_form(tmp_path, monkeypatch, past):
    sets = an_import()
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets) - past)
    corpus_sets.write(sets, tmp_path, "x-")
    assert check(sets, tmp_path) == []


def test_check_compares_a_compressed_sets_plain_content_not_its_bytes(tmp_path, monkeypatch):
    sets = an_import()
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets) - 1)
    corpus_sets.write(sets, tmp_path, "x-")
    path = tmp_path / "render" / "x-a.jsonl.zst"
    path.write_bytes(zstandard.ZstdCompressor(level=3).compress(storage.plain_bytes(path)))
    assert check(sets, tmp_path) == []
    path.write_bytes(zstandard.ZstdCompressor(level=3).compress(b'{"name": "x-a-0"}\n'))
    assert check(sets, tmp_path) == [f"{path}: differs from a fresh import"]


@pytest.mark.parametrize("past", [False, True], ids=["now plain", "now compressed"])
def test_check_applies_the_limit_to_a_corpus_written_in_the_other_form(tmp_path, monkeypatch, past):
    sets = an_import()
    size = plain_size(sets)
    monkeypatch.setattr(corpus_sets, "LIMIT", size - (not past))
    corpus_sets.write(sets, tmp_path, "x-")
    monkeypatch.setattr(corpus_sets, "LIMIT", size - past)
    fresh, stored = (".jsonl.zst", ".jsonl") if past else (".jsonl", ".jsonl.zst")
    limit = f"the x-* sets take {size} bytes as plain JSON Lines, {'past' if past else 'within'} the {size - past}"
    reason = f"a fresh import writes this set as x-a{fresh}: {limit} that stay plain"
    assert check(sets, tmp_path) == [
        f"{tmp_path / 'parse' / f'x-a{fresh}'}: missing",
        f"{tmp_path / 'render' / f'x-a{fresh}'}: missing",
        f"{tmp_path / 'parse' / f'x-a{stored}'}: {reason}",
        f"{tmp_path / 'render' / f'x-a{stored}'}: {reason}",
    ]


def test_check_names_a_set_git_lfs_has_not_fetched_with_the_command_that_fetches_it(tmp_path, monkeypatch):
    sets = an_import()
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets) - 1)
    corpus_sets.write(sets, tmp_path, "x-")
    pointer = tmp_path / "render" / "x-a.jsonl.zst"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 13\n")
    fetch = f"git lfs pull --include '{pointer}' --exclude ''"
    assert check(sets, tmp_path) == [f"{pointer} is a Git LFS pointer; fetch it first: {fetch}"]


def test_check_names_a_set_file_no_part_of_the_import_writes_in_either_form(tmp_path, monkeypatch):
    sets = an_import()
    corpus_sets.write(sets, tmp_path, "x-")
    for stale in ("render/x-gone.jsonl", "render/x-gone.jsonl.zst"):
        (tmp_path / stale).write_bytes(b"")
    assert check(sets, tmp_path) == [
        f"{tmp_path / 'render' / 'x-gone.jsonl'}: no x part writes it",
        f"{tmp_path / 'render' / 'x-gone.jsonl.zst'}: no x part writes it",
    ]
