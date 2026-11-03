import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import chartqa, github, hf, image_sets

GPL = b"                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n"
CARD = b"---\nlicense: gpl-3.0\n---\n# ChartQA\n"
FILE = "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(charts: list[bytes | None]) -> pa.Table:
    """Rows of one question each, with the chart in the ``image`` column as the dataset stores it, null where
    absent."""
    kind = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    cells = [({"bytes": item, "path": None} if item else None) for item in charts]
    return pa.table(
        {
            "question": pa.array([f"q{n}" for n in range(len(charts))]),
            "image": pa.array(cells, type=kind),
        }
    )


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """The pinned file served from ``tmp_path``, with the card and the licence text, through the real ``hf.fetch``
    and ``github.fetch`` signatures with no network."""
    first, second, third = png(8, 6), jpeg(10, 4), png(3, 3)
    pq.write_table(
        table([first, first, second, b"not an image", None, third, second]),
        tmp_path / "test.parquet",
    )
    files = {FILE: tmp_path / "test.parquet"}
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (
            chartqa.REPO,
            chartqa.REVISION,
        ) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            chartqa.LICENSE_OWNER,
            chartqa.LICENSE_REPO,
            chartqa.LICENSE_COMMIT,
            chartqa.LICENSE_PATH,
        )
        assert (sha256, cache) == (chartqa.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(GPL)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {
        "first": first,
        "second": second,
        "third": third,
        "served": served,
        "files": files,
    }


def test_the_file_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", chartqa.REVISION)
    assert list(chartqa.FILES) == [FILE] and re.fullmatch(r"[0-9a-f]{64}", chartqa.FILES[FILE])
    assert chartqa.SOURCE == f"hf:datasets/HuggingFaceM4/ChartQA@{chartqa.REVISION}"
    assert chartqa.CARD_LICENSE == "gpl-3.0" and chartqa.LICENSE == "GPL-3.0"


def test_read_gives_every_chart_in_row_order_with_its_origin(pinned):
    found = list(chartqa.read(pinned["files"]))
    assert [(f.origin["file"], f.origin["row"]) for f in found] == [
        (FILE, 0),
        (FILE, 1),
        (FILE, 2),
        (FILE, 3),
        (FILE, 5),
        (FILE, 6),
    ]
    assert found[0].data == pinned["first"] and found[2].data == pinned["second"]
    assert found[0].origin == {
        "dataset": "chartqa",
        "source": chartqa.SOURCE,
        "sha256": chartqa.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "image",
        "license": "GPL-3.0",
    }
    assert found[0].notes == "ChartQA test row 0, image" and found[4].notes == "ChartQA test row 5, image"


def test_the_licence_text_passes_and_another_text_is_refused():
    chartqa.check_license(GPL)
    with pytest.raises(ValueError, match="not the GNU General Public License v3"):
        chartqa.check_license(b"MIT License\n")


def test_import_writes_the_set_the_charts_and_the_licence_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "chartqa", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "chartqa.jsonl")
    assert [line["name"] for line in lines] == [
        f"chartqa-00001-{sha(first)[:8]}",
        f"chartqa-00002-{sha(second)[:8]}",
        f"chartqa-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/chartqa/chartqa-00001-{sha(first)[:8]}.png"
    assert lines[1]["path"].endswith(".jpg")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "chartqa-LICENSE").read_bytes() == GPL
    assert pinned["served"][hf.CARD] == chartqa.CARD_SHA256 and pinned["served"][FILE] == chartqa.FILES[FILE]
    assert out == [
        f"no line for ChartQA test row 1, image: it repeats chartqa-00001-{sha(first)[:8]}",
        f"no line for ChartQA test row 6, image: it repeats chartqa-00002-{sha(second)[:8]}",
        "no line for ChartQA test row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'chartqa.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert (
        main(
            [
                "import",
                "chartqa",
                "--corpus",
                str(corpus),
                "--cache",
                str(cache),
                "--check",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the ChartQA set equals a fresh import of {chartqa.SOURCE}"
    ]
    text = (corpus / "image" / "chartqa.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "chartqa.jsonl").write_text("".join(line + "\n" for line in text))
    assert (
        main(
            [
                "import",
                "chartqa",
                "--corpus",
                str(corpus),
                "--cache",
                str(cache),
                "--check",
            ]
        )
        == 1
    )
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'chartqa.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: cc-by-4.0\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(
        ValueError,
        match="the card's license is 'cc-by-4.0', not the reviewed 'gpl-3.0'",
    ):
        main(
            [
                "import",
                "chartqa",
                "--corpus",
                str(tmp_path / "corpus"),
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
