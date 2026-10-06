import json

from bellwether import storage
from bellwether.cli import main
from bellwether.count import counts


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_count_groups_fixture_cases_by_model_kind_and_source(tmp_path, capsys):
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    (fixtures / "m1").mkdir(parents=True)
    (fixtures / "m1" / "manifest.toml").write_text('model = "org/M1"\nrevision = "r"\n')
    table = 'form = "{}"\ncases = {}\nrejected = 0\nplain_bytes = 1\nplain_sha256 = "x"\n'
    (fixtures / "m1" / "sets.toml").write_text(
        "[render.common]\n"
        + table.format("plain", 2)
        + "[render.bfcl-x]\n"
        + table.format("zstd", 1)
        + "[parse.bfcl-x]\n"
        + table.format("zstd", 1)
    )
    request = {"messages": []}
    write_lines(
        corpus / "render" / "common.jsonl", [{"name": "a", "request": request}, {"name": "b", "request": request}]
    )
    imported = {"name": "bfcl-x-0", "request": request, "origin": {"dataset": "bfcl"}}
    write_lines(corpus / "render" / "bfcl-x.jsonl", [imported])
    write_lines(corpus / "parse" / "bfcl-x.jsonl", [imported])

    assert counts(fixtures, corpus) == [
        {"model": "org/M1", "kind": "parse", "source": "bfcl", "cases": 1},
        {"model": "org/M1", "kind": "render", "source": "bfcl", "cases": 1},
        {"model": "org/M1", "kind": "render", "source": "hand-written", "cases": 2},
    ]
    assert main(["count", "--fixtures", str(fixtures), "--corpus", str(corpus)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "| Model | Kind | Source | Cases |",
        "|---|---|---|---:|",
        "| org/M1 | parse | bfcl | 1 |",
        "| org/M1 | render | bfcl | 1 |",
        "| org/M1 | render | hand-written | 2 |",
        "| all | | | 4 |",
    ]


def test_count_finds_the_source_of_a_compressed_corpus_set(tmp_path):
    fixtures, corpus = tmp_path / "fixtures", tmp_path / "corpus"
    (fixtures / "m1").mkdir(parents=True)
    (fixtures / "m1" / "manifest.toml").write_text('model = "org/M1"\nrevision = "r"\n')
    table = 'form = "zstd"\ncases = 2\nrejected = 0\nplain_bytes = 1\nplain_sha256 = "x"\n'
    (fixtures / "m1" / "sets.toml").write_text("[render.big-x]\n" + table)
    lines = [{"name": f"big-x-{n}", "request": {"messages": []}, "origin": {"dataset": "big"}} for n in range(2)]
    storage.write(corpus / "render" / "big-x.jsonl.zst", "".join(json.dumps(line) + "\n" for line in lines).encode())
    assert counts(fixtures, corpus) == [{"model": "org/M1", "kind": "render", "source": "big", "cases": 2}]
