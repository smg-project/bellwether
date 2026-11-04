import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, websight

CC_BY = b"Creative Commons Attribution 4.0 International\n\n Creative Commons Corporation is not a law firm\n"
CARD = b"---\nlicense: cc-by-4.0\n---\n# WebSight\n"
FILE2 = "v0.2/train-00001-of-00738-b79d703dc79dce91.parquet"
FILE = "v0.2/train-00000-of-00738-80a58552f2fb3344.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(photos: list[bytes | None]) -> pa.Table:
    """Rows of one question each, with the photograph in the ``image`` column as the dataset stores it, null where
    absent."""
    kind = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    cells = [({"bytes": photo, "path": None} if photo else None) for photo in photos]
    return pa.table({"question": pa.array([f"q{n}" for n in range(len(photos))]), "image": pa.array(cells, type=kind)})


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """The pinned shard served from ``tmp_path``, with the card and the licence text, through the real ``hf.fetch``
    and ``github.fetch`` signatures with no network."""
    first, second, third = jpeg(8, 6), jpeg(10, 4), png(3, 3)
    pq.write_table(table([first, first, second, b"not an image", None, third, second]), tmp_path / "shard.parquet")
    pq.write_table(table([]), tmp_path / "empty.parquet")
    files = {FILE: tmp_path / "shard.parquet", FILE2: tmp_path / "empty.parquet"}
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (websight.REPO, websight.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            websight.LICENSE_OWNER,
            websight.LICENSE_REPO,
            websight.LICENSE_COMMIT,
            websight.LICENSE_PATH,
        )
        assert (sha256, cache) == (websight.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(CC_BY)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", websight.REVISION)
    assert list(websight.FILES) == [FILE, FILE2]
    assert all(re.fullmatch(r"[0-9a-f]{64}", websight.FILES[f]) for f in (FILE, FILE2))
    assert websight.SOURCE == f"hf:datasets/HuggingFaceM4/WebSight@{websight.REVISION}"
    assert websight.CARD_LICENSE == "cc-by-4.0" and websight.LICENSE == "CC-BY-4.0"


def test_read_gives_every_screenshot_in_row_order_with_its_origin(pinned):
    found = list(websight.read(pinned["files"]))
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
        "dataset": "websight",
        "source": websight.SOURCE,
        "sha256": websight.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "image",
        "license": "CC-BY-4.0",
    }
    assert found[0].notes == "WebSight train row 0, image" and found[4].notes == "WebSight train row 5, image"


def test_the_cc_by_licence_passes_and_another_text_is_refused():
    websight.check_license(CC_BY)
    with pytest.raises(ValueError, match="not the CC BY 4.0 licence text"):
        websight.check_license(b"Creative Commons Attribution-ShareAlike 4.0 International\n")


def test_import_writes_the_set_the_screenshots_and_the_licence_copy_and_check_passes_then_fails(
    pinned, tmp_path, capsys
):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "websight", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "websight.jsonl")
    assert [line["name"] for line in lines] == [
        f"websight-00001-{sha(first)[:8]}",
        f"websight-00002-{sha(second)[:8]}",
        f"websight-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/websight/websight-00001-{sha(first)[:8]}.jpg"
    assert lines[2]["path"].endswith(".png")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "websight-LICENSE").read_bytes() == CC_BY
    assert pinned["served"][hf.CARD] == websight.CARD_SHA256 and pinned["served"][FILE] == websight.FILES[FILE]
    assert pinned["served"][FILE2] == websight.FILES[FILE2]
    assert out == [
        f"no line for WebSight train row 1, image: it repeats websight-00001-{sha(first)[:8]}",
        f"no line for WebSight train row 6, image: it repeats websight-00002-{sha(second)[:8]}",
        "no line for WebSight train row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'websight.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "websight", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the WebSight set equals a fresh import of {websight.SOURCE}"
    ]
    text = (corpus / "image" / "websight.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "websight.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "websight", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'websight.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: mit\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match="the card's license is 'mit', not the reviewed 'cc-by-4.0'"):
        main(["import", "websight", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
