import re
import tracemalloc

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
    limit = f"the import's sets take {size} bytes as plain JSON Lines, {'past' if past else 'within'} the {size - past}"
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


def test_check_names_a_compressed_set_zstd_cannot_read_and_write_replaces_it(tmp_path, monkeypatch):
    sets = an_import()
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(sets) - 1)
    corpus_sets.write(sets, tmp_path, "x-")
    path = tmp_path / "render" / "x-a.jsonl.zst"
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    [problem] = check(sets, tmp_path)
    assert problem.startswith(f"{path} cannot be decompressed: ")
    corpus_sets.write(sets, tmp_path, "x-")
    assert check(sets, tmp_path) == []


# Streamed imports: the sets one at a time, as an importer that builds them as it reads would yield them.


def as_stream(sets: dict):
    for (kind, name), lines in sets.items():
        yield kind, name, lines


def an_import_with_repeats() -> dict:
    """Sets whose cases repeat across sets, and whose parse cases share a message, as a real import's do."""
    return {
        ("render", "x-b"): [render("x-b-0", ask("Hi")), render("x-b-1", ask("Bye")), render("x-b-2", ask("Hi"))],
        ("parse", "x-b"): [
            parse("x-b-3", ask("Hi"), {"content": "Hello"}),
            parse("x-b-4", ask("Yo"), {"content": "Hello"}),
        ],
        ("render", "x-a"): [render("x-a-0", ask("Bye")), render("x-a-1", ask("Hey"))],
        ("parse", "x-a"): [
            parse("x-a-2", ask("Hi"), {"content": "Hello"}),
            parse("x-a-3", ask("Hey"), {"content": "Hey"}),
        ],
    }


def kept_size(sets: dict) -> int:
    kept, _ = corpus_sets.leave_out_repeats(sets)
    return plain_size(kept)


def files_and_bytes(root) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def written_held(sets: dict, root, capsys, **declared) -> tuple[dict, str]:
    """An importer that holds its sets, as every importer did: the repeat rule, the report, then the write."""
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    corpus_sets.report("X", sets, kept, repeats, root)
    corpus_sets.write(kept, root, "x-", **declared)
    return files_and_bytes(root), capsys.readouterr().out.replace(str(root), "<root>")


def written_streamed(sets: dict, root, capsys, **declared) -> tuple[dict, str]:
    """An importer that streams its sets: write leaves out the repeats as they come, and the report follows."""
    summary = corpus_sets.Summary()
    corpus_sets.write(as_stream(sets), root, "x-", summary=summary, **declared)
    corpus_sets.report_summary("X", summary, root)
    return files_and_bytes(root), capsys.readouterr().out.replace(str(root), "<root>")


@pytest.mark.parametrize("past", [False, True], ids=["plain", "compressed"])
def test_a_streamed_import_writes_the_files_and_report_of_one_that_holds_its_sets(tmp_path, monkeypatch, capsys, past):
    sets = an_import_with_repeats()
    monkeypatch.setattr(corpus_sets, "LIMIT", kept_size(sets) - past)
    held = written_held(sets, tmp_path / "held", capsys)
    assert written_streamed(sets, tmp_path / "streamed", capsys) == held
    suffix = ".jsonl.zst" if past else ".jsonl"
    assert held[1].splitlines() == [
        "no case x-b-2: it repeats x-b-0",
        "no case x-a-0: it repeats x-b-1",
        "no case x-a-2: it repeats x-b-3",
        f"<root>/parse/x-a{suffix}: 1 cases, 1 left out as repeats, 1 distinct messages",
        f"<root>/parse/x-b{suffix}: 2 cases, 1 distinct messages",
        f"<root>/render/x-a{suffix}: 1 cases, 1 left out as repeats",
        f"<root>/render/x-b{suffix}: 2 cases, 1 left out as repeats",
        "<root>: 6 cases in the 4 X sets, 3 left out as repeats, 2 distinct messages",
    ]


def test_a_streamed_import_declared_compressed_writes_what_an_import_past_the_limit_writes(
    tmp_path, monkeypatch, capsys
):
    sets = an_import_with_repeats()
    monkeypatch.setattr(corpus_sets, "LIMIT", kept_size(sets) - 1)
    held = written_held(sets, tmp_path / "held", capsys)
    assert written_streamed(sets, tmp_path / "streamed", capsys, form="zstd") == held


@pytest.mark.parametrize("past", [False, True], ids=["plain", "compressed"])
def test_a_streamed_check_names_what_a_check_of_held_sets_names(tmp_path, monkeypatch, past):
    sets = an_import_with_repeats()
    kept, _ = corpus_sets.leave_out_repeats(sets)
    monkeypatch.setattr(corpus_sets, "LIMIT", plain_size(kept) - past)
    corpus_sets.write(kept, tmp_path, "x-")
    assert corpus_sets.check(as_stream(sets), tmp_path, "x-", "x part") == []
    form, other = (".jsonl.zst", ".jsonl") if past else (".jsonl", ".jsonl.zst")
    (tmp_path / "parse" / f"x-a{form}").unlink()
    storage.write(tmp_path / "render" / f"x-a{form}", b"{}\n")
    (tmp_path / "render" / f"x-b{other}").write_bytes(b"{}\n")
    (tmp_path / "render" / "x-gone.jsonl").write_bytes(b"{}\n")
    held = check(kept, tmp_path)
    assert len(held) == 4
    assert corpus_sets.check(as_stream(sets), tmp_path, "x-", "x part") == held


def test_a_streamed_import_declared_plain_is_refused_before_it_writes_past_the_limit(tmp_path, monkeypatch):
    sets = an_import_with_repeats()
    (tmp_path / "render").mkdir()
    (tmp_path / "render" / "x-gone.jsonl").write_bytes(b"{}\n")
    monkeypatch.setattr(corpus_sets, "LIMIT", 10)
    with pytest.raises(ValueError, match=re.escape("but the import declares form 'plain'; declare form 'zstd'")):
        corpus_sets.write(as_stream(sets), tmp_path, "x-", form="plain")
    assert files(tmp_path) == ["render/x-gone.jsonl"]


def test_a_streamed_import_declared_compressed_within_the_limit_is_refused_and_removes_nothing(tmp_path):
    sets = an_import_with_repeats()
    kept, _ = corpus_sets.leave_out_repeats(sets)
    corpus_sets.write(kept, tmp_path, "x-")
    plain = ["parse/x-a.jsonl", "parse/x-b.jsonl", "render/x-a.jsonl", "render/x-b.jsonl"]
    assert files(tmp_path) == plain
    with pytest.raises(ValueError, match=re.escape("but the import declares form 'zstd'; declare form 'plain'")):
        corpus_sets.write(as_stream(sets), tmp_path, "x-", form="zstd")
    assert set(plain) <= set(files(tmp_path))


@pytest.mark.parametrize(
    "form, limit, wording",
    [("plain", 10, "past the 10 that stay plain"), ("zstd", None, "within the 50000000 that stay plain")],
    ids=["plain past the limit", "zstd within it"],
)
def test_a_check_names_a_declared_form_the_total_contradicts(tmp_path, monkeypatch, form, limit, wording):
    sets = an_import_with_repeats()
    if limit:
        monkeypatch.setattr(corpus_sets, "LIMIT", limit)
    problems = corpus_sets.check(as_stream(sets), tmp_path, "x-", "x part", form=form)
    size = kept_size(sets)
    other = "zstd" if form == "plain" else "plain"
    assert problems[-1] == (
        f"{tmp_path}: the x-* sets take {size} bytes as plain JSON Lines, {wording}, "
        f"but the import declares form {form!r}; declare form {other!r}"
    )


def test_a_form_that_is_not_plain_or_zstd_is_refused(tmp_path):
    with pytest.raises(ValueError, match=re.escape("form 'gzip' is not 'plain' or 'zstd'")):
        corpus_sets.write(as_stream(an_import_with_repeats()), tmp_path, "x-", form="gzip")


# Memory: an import streamed one set at a time holds about one set, however many sets it has.

CASE_BYTES = 10_000
CASES = 20
SETS = 20
ONE_SET = CASES * CASE_BYTES  # about one set's plain bytes; the import takes SETS times as many


def many_sets():
    for n in range(SETS):
        yield "render", f"x-{n:02d}", [render(f"x-{n}-{i}", ask(f"{n}.{i} " + "a" * CASE_BYTES)) for i in range(CASES)]


def traced_peak(function, *args, **kwargs) -> int:
    """The most memory Python held at once, in bytes, while ``function`` ran, past what it held when it started."""
    tracemalloc.start()
    try:
        held = tracemalloc.get_traced_memory()[0]
        function(*args, **kwargs)
        return tracemalloc.get_traced_memory()[1] - held
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("form, limit", [("plain", None), ("zstd", 1000), (None, None), (None, 1000)])
def test_a_streamed_write_and_check_hold_about_one_set_at_a_time(tmp_path, monkeypatch, form, limit):
    if limit:
        monkeypatch.setattr(corpus_sets, "LIMIT", limit)
    assert traced_peak(corpus_sets.write, many_sets(), tmp_path, "x-", form=form) < 4 * ONE_SET
    assert traced_peak(corpus_sets.check, many_sets(), tmp_path, "x-", "x part", form=form) < 4 * ONE_SET
    assert corpus_sets.check(many_sets(), tmp_path, "x-", "x part", form=form) == []
    assert len(files(tmp_path)) == SETS
