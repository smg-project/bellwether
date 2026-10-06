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


def test_every_checkpoint_is_a_row_and_a_member_shows_its_group_s_cases(tmp_path, capsys):
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
    # A member repeats its group's cases; `all` counts each group's once.
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


def test_count_refuses_a_member_whose_group_names_no_manifest(tmp_path, capsys):
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
