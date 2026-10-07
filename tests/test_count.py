import json

import pytest
import zstandard

from bellwether import storage
from bellwether.cli import main
from bellwether.count import counts, set_sources

REVISION = "0123456789abcdef0123456789abcdef01234567"
TABLE = 'form = "{}"\ncases = {}\nrejected = 0\nplain_bytes = 1\nplain_sha256 = "x"\n'


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def write_manifest(fixtures, slug, model, tier, group=None):
    (fixtures / slug).mkdir(parents=True)
    named = f'group = "{group}"\n' if group else ""
    (fixtures / slug / "manifest.toml").write_text(
        f'model = "{model}"\nrevision = "{REVISION}"\ntier = {tier}\n{named}'
    )


def test_count_groups_fixture_cases_by_model_kind_and_source(tmp_path, capsys):
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    write_manifest(fixtures, "m1", "org/M1", 1)
    (fixtures / "m1" / "sets.toml").write_text(
        "[render.common]\n"
        + TABLE.format("plain", 2)
        + "[render.bfcl-x]\n"
        + TABLE.format("zstd", 1)
        + "[parse.bfcl-x]\n"
        + TABLE.format("zstd", 1)
    )
    request = {"messages": []}
    write_lines(
        corpus / "render" / "common.jsonl", [{"name": "a", "request": request}, {"name": "b", "request": request}]
    )
    imported = {"name": "bfcl-x-0", "request": request, "origin": {"dataset": "bfcl"}}
    write_lines(corpus / "render" / "bfcl-x.jsonl", [imported])
    write_lines(corpus / "parse" / "bfcl-x.jsonl", [imported])

    assert counts(fixtures, corpus) == [
        {"model": "org/M1", "group": "m1", "tier": 1, "kind": "parse", "source": "bfcl", "cases": 1},
        {"model": "org/M1", "group": "m1", "tier": 1, "kind": "render", "source": "bfcl", "cases": 1},
        {"model": "org/M1", "group": "m1", "tier": 1, "kind": "render", "source": "hand-written", "cases": 2},
    ]
    assert main(["count", "--fixtures", str(fixtures), "--corpus", str(corpus)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "| Model | Group | Tier | Kind | Source | Cases |",
        "|---|---|---:|---|---|---:|",
        "| org/M1 | m1 | 1 | parse | bfcl | 1 |",
        "| org/M1 | m1 | 1 | render | bfcl | 1 |",
        "| org/M1 | m1 | 1 | render | hand-written | 2 |",
        "| all | | | | | 4 |",
    ]


def test_every_checkpoint_is_a_row_and_another_member_shows_its_group_s_cases(tmp_path, capsys):
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    write_manifest(fixtures, "m1", "org/M1", 1)
    write_manifest(fixtures, "m1-small", "org/M1-Small", 2, group="m1")
    write_manifest(fixtures, "m2", "org/M2", 3)
    (fixtures / "m1" / "sets.toml").write_text("[render.common]\n" + TABLE.format("plain", 2))
    request = {"messages": []}
    write_lines(
        corpus / "render" / "common.jsonl", [{"name": "a", "request": request}, {"name": "b", "request": request}]
    )

    assert counts(fixtures, corpus) == [
        {"model": "org/M1", "group": "m1", "tier": 1, "kind": "render", "source": "hand-written", "cases": 2},
        {"model": "org/M1-Small", "group": "m1", "tier": 2, "kind": "render", "source": "hand-written", "cases": 2},
        {"model": "org/M2", "group": "m2", "tier": 3, "kind": None, "source": None, "cases": 0},
    ]
    assert main(["count", "--fixtures", str(fixtures), "--corpus", str(corpus)]) == 0
    # Another member repeats its group's cases; `all` counts each group's once.
    assert capsys.readouterr().out.splitlines()[2:] == [
        "| org/M1 | m1 | 1 | render | hand-written | 2 |",
        "| org/M1-Small | m1 | 2 | render | hand-written | 2 |",
        "| org/M2 | m2 | 3 | | | 0 |",
        "| all | | | | | 2 |",
    ]


def test_count_finds_the_source_of_a_compressed_corpus_set(tmp_path):
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    (fixtures / "m1").mkdir(parents=True)
    (fixtures / "m1" / "manifest.toml").write_text(f'model = "org/M1"\nrevision = "{REVISION}"\n')
    table = 'form = "zstd"\ncases = 2\nrejected = 0\nplain_bytes = 1\nplain_sha256 = "x"\n'
    (fixtures / "m1" / "sets.toml").write_text("[render.big-x]\n" + table)
    lines = [{"name": f"big-x-{n}", "request": {"messages": []}, "origin": {"dataset": "big"}} for n in range(2)]
    storage.write(corpus / "render" / "big-x.jsonl.zst", "".join(json.dumps(line) + "\n" for line in lines).encode())
    assert counts(fixtures, corpus) == [
        {"model": "org/M1", "group": "m1", "tier": None, "kind": "render", "source": "big", "cases": 2}
    ]


FIRST = {"name": "big-x-0", "request": {}, "origin": {"dataset": "big"}}


def set_path(corpus, form: str):
    path = corpus / "render" / ("big-x.jsonl.zst" if form == "compressed" else "big-x.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("form", ["plain", "compressed"])
def test_count_reads_a_sets_first_record_and_nothing_after_it(tmp_path, form):
    # The rest of the set cannot be read: its second line is not JSON, and past 2.8 MB of cases a plain set's bytes are
    # not UTF-8 and a compressed set's frame is cut short. Only a reader that stops at the first record gives a source.
    cases = (json.dumps({"name": f"big-x-{n}", "request": {"text": "lorem ipsum " * 8}}) for n in range(1, 20000))
    rest = "{not json\n" + "".join(case + "\n" for case in cases)
    text = (json.dumps(FIRST) + "\n" + rest).encode()
    path = set_path(tmp_path / "corpus", form)
    if form == "compressed":
        frame = zstandard.ZstdCompressor(level=19).compress(text)
        path.write_bytes(frame[: len(frame) // 2])
    else:
        path.write_bytes(text + b"\xff\xfe\n")
    assert set_sources(tmp_path / "corpus") == {("render", "big-x"): "big"}


@pytest.mark.parametrize("form", ["plain", "compressed"])
def test_count_keeps_a_record_whole_across_the_line_breaks_json_writes_raw(tmp_path, form):
    # Only "\n" ends a line: U+2028 and U+0085 inside a string, written raw by ensure_ascii=False, stay in it.
    text = "one" + chr(0x2028) + "two" + chr(0x85) + "three"
    first = {**FIRST, "request": {"messages": [{"role": "user", "content": text}]}}
    line = json.dumps(first, ensure_ascii=False) + "\n"
    assert chr(0x2028) in line and chr(0x85) in line
    storage.write(set_path(tmp_path / "corpus", form), line.encode())
    assert set_sources(tmp_path / "corpus") == {("render", "big-x"): "big"}


def test_count_names_a_set_git_lfs_has_not_fetched_with_the_command_that_fetches_it(tmp_path):
    pointer = set_path(tmp_path / "corpus", "compressed")
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 13\n")
    with pytest.raises(ValueError, match="big-x.jsonl.zst is a Git LFS pointer; fetch it first: git lfs pull"):
        set_sources(tmp_path / "corpus")


def test_count_refuses_a_manifest_whose_group_names_no_manifest(tmp_path, capsys):
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    write_manifest(fixtures, "m1-small", "org/M1-Small", 2, group="m1")
    assert main(["count", "--fixtures", str(fixtures), "--corpus", str(corpus)]) == 1
    assert f"bellwether count: {fixtures / 'm1-small' / 'manifest.toml'}: group m1 names no manifest" in (
        capsys.readouterr().err
    )


def test_a_member_of_a_group_that_recorded_nothing_has_a_row_with_no_cases(tmp_path):
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    write_manifest(fixtures, "m1", "org/M1", 1)
    write_manifest(fixtures, "m1-small", "org/M1-Small", 2, group="m1")
    assert counts(fixtures, corpus) == [
        {"model": "org/M1", "group": "m1", "tier": 1, "kind": None, "source": None, "cases": 0},
        {"model": "org/M1-Small", "group": "m1", "tier": 2, "kind": None, "source": None, "cases": 0},
    ]


def test_count_s_help_says_it_counts_per_checkpoint_with_its_group_and_tier(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "cases per checkpoint, kind and source, with each checkpoint's group and tier" in " ".join(
        capsys.readouterr().out.split()
    )


README = "# x\n\nintro\n\n## What bellwether holds\n\n{begin}\nstale\n{end}\n\n## Next\n\nrest\n"


def holdings(tmp_path):
    """A fixture root with one recorded group (two checkpoints) and one that recorded nothing, and a corpus of two
    sources: hand-written and an imported one whose lines name two licenses."""
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    write_manifest(fixtures, "m1", "org/M1", 1)
    write_manifest(fixtures, "m1-small", "org/M1-Small", 1, group="m1")
    write_manifest(fixtures, "m2", "org/M2", 2)
    (fixtures / "m1" / "sets.toml").write_text(
        "[render.common]\n"
        + TABLE.format("plain", 2)
        + "[parse.bfcl-x]\n"
        + TABLE.format("plain", 3).replace("rejected = 0", "rejected = 1")
    )
    request = {"messages": []}
    write_lines(
        corpus / "render" / "common.jsonl", [{"name": "a", "request": request}, {"name": "b", "request": request}]
    )
    origin = {"dataset": "bfcl", "source": "pypi:bfcl-eval==1", "license": "Apache-2.0"}
    lines = [{"name": f"bfcl-x-{n}", "request": request, "origin": {**origin, "row": n}} for n in range(3)]
    lines.append({"name": "bfcl-x-3", "request": request, "origin": {**origin, "row": 3, "license": "MIT"}})
    write_lines(corpus / "parse" / "bfcl-x.jsonl", lines)
    return fixtures, corpus


def test_readme_tables_are_written_between_the_markers_and_nothing_else_changes(tmp_path):
    from bellwether.count import README_BEGIN, README_END

    fixtures, corpus = holdings(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README.format(begin=README_BEGIN, end=README_END))
    args = ["count", "--fixtures", str(fixtures), "--corpus", str(corpus), "--readme", str(readme)]
    assert main(args) == 0
    text = readme.read_text()
    before, _, rest = text.partition(README_BEGIN)
    tables, _, after = rest.partition(README_END)
    assert before == "# x\n\nintro\n\n## What bellwether holds\n\n"
    assert after == "\n\n## Next\n\nrest\n"
    assert "stale" not in tables
    assert [line for line in tables.splitlines() if line.startswith("| [m1]")] == [
        "| [m1](fixtures/m1/sets.toml) | org/M1 @ 01234567 | 2 | 1 | 2 | 3 | 1 | bfcl, hand-written |"
    ]
    assert "1 more group (1 checkpoint) has a manifest and nothing recorded yet." in tables
    assert [line for line in tables.splitlines() if line.startswith(("| bfcl ", "| hand-written "))] == [
        "| bfcl | pypi:bfcl-eval==1 | Apache-2.0, MIT | 0 | 0 | 1 | 4 | 0.0 MB | 0.0 MB, plain |",
        "| hand-written | written in this repository | | 1 | 2 | 0 | 0 | 0.0 MB | 0.0 MB, plain |",
    ]


def test_readme_check_passes_when_current_and_fails_when_a_count_changed(tmp_path, capsys):
    from bellwether.count import README_BEGIN, README_END

    fixtures, corpus = holdings(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README.format(begin=README_BEGIN, end=README_END))
    args = ["count", "--fixtures", str(fixtures), "--corpus", str(corpus), "--readme", str(readme)]
    assert main([*args, "--check"]) == 1
    assert "run `bellwether count --readme" in capsys.readouterr().err
    assert main(args) == 0
    assert main([*args, "--check"]) == 0
    write_lines(corpus / "render" / "common.jsonl", [{"name": "a", "request": {"messages": []}}])
    assert main([*args, "--check"]) == 1


def test_readme_without_the_markers_is_refused(tmp_path, capsys):
    fixtures, corpus = holdings(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("# x\n")
    assert main(["count", "--fixtures", str(fixtures), "--corpus", str(corpus), "--readme", str(readme)]) == 1
    assert "has no tables to replace" in capsys.readouterr().err


def test_readme_cuts_a_full_commit_id_to_eight_characters_and_writes_a_gigabyte_as_gigabytes():
    from bellwether.count import _short, _size

    commit = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"
    assert _short(f"hf:datasets/glaiveai/glaive-function-calling-v2@{commit}") == (
        "hf:datasets/glaiveai/glaive-function-calling-v2@e7f4b645"
    )
    assert _short("pypi:bfcl-eval==2026.3.23") == "pypi:bfcl-eval==2026.3.23"
    assert (_size(37_040_000), _size(1_243_300_000)) == ("37.0 MB", "1.24 GB")


@pytest.mark.parametrize(
    ("line", "why"),
    [
        ("[1, 2]", ": line 1 is not a JSON object"),
        ('"text"', ": line 1 is not a JSON object"),
        ("{not json", ":1: not JSON"),
    ],
)
def test_readme_names_a_corpus_line_that_is_not_a_json_object(tmp_path, capsys, line, why):
    from bellwether.count import README_BEGIN, README_END

    fixtures, corpus = holdings(tmp_path)
    (corpus / "parse" / "bfcl-x.jsonl").write_text(line + "\n")
    readme = tmp_path / "README.md"
    readme.write_text(README.format(begin=README_BEGIN, end=README_END))
    assert main(["count", "--fixtures", str(fixtures), "--corpus", str(corpus), "--readme", str(readme)]) == 1
    assert f"{corpus / 'parse' / 'bfcl-x.jsonl'}{why}" in capsys.readouterr().err
