import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, pathvqa

LICENSE_TEXT = b"MIT License\n\n(the body of the text)\n"
CARD = b"---\nlicense: mit\n---\n# PathVQA\n"
FILE = "data/test-00000-of-00003-e9adadb4799f44d3.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(images: list[bytes | None]) -> pa.Table:
    """Rows of one slide each, in the ``image`` column as the dataset stores it, null where absent."""
    kind = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    cells = [({"bytes": data, "path": None} if data else None) for data in images]
    return pa.table({"id": pa.array([f"r{n}" for n in range(len(images))]), "image": pa.array(cells, type=kind)})


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """The pinned shard served from ``tmp_path``, with the card and the licence text, through the real ``hf.fetch``
    and ``github.fetch`` signatures with no network."""
    first, second, third = jpeg(8, 6), jpeg(10, 4), png(3, 3)
    pq.write_table(table([first, first, second, b"not an image", None, third, second]), tmp_path / "shard.parquet")
    files = {FILE: tmp_path / "shard.parquet"}
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (pathvqa.REPO, pathvqa.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            pathvqa.LICENSE_OWNER,
            pathvqa.LICENSE_REPO,
            pathvqa.LICENSE_COMMIT,
            pathvqa.LICENSE_PATH,
        )
        assert (sha256, cache) == (pathvqa.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(LICENSE_TEXT)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", pathvqa.REVISION)
    assert list(pathvqa.FILES) == [FILE] and re.fullmatch(r"[0-9a-f]{64}", pathvqa.FILES[FILE])
    assert pathvqa.SOURCE == f"hf:datasets/flaviagiammarino/path-vqa@{pathvqa.REVISION}"
    assert pathvqa.CARD_LICENSE == "mit" and pathvqa.LICENSE == "MIT"


def test_read_gives_every_slide_in_row_order_with_its_origin(pinned):
    found = list(pathvqa.read(pinned["files"]))
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
        "dataset": "pathvqa",
        "source": pathvqa.SOURCE,
        "sha256": pathvqa.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "image",
        "license": "MIT",
    }
    assert found[0].notes == "PathVQA test row 0, image"
    assert found[4].notes == "PathVQA test row 5, image"


def test_the_licence_text_passes_and_another_text_is_refused():
    pathvqa.check_license(LICENSE_TEXT)
    with pytest.raises(ValueError, match=re.escape("not the MIT License")):
        pathvqa.check_license(b"Apache License\n")


def test_import_writes_the_set_the_slides_and_the_licence_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "pathvqa", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "pathvqa.jsonl")
    assert [line["name"] for line in lines] == [
        f"pathvqa-00001-{sha(first)[:8]}",
        f"pathvqa-00002-{sha(second)[:8]}",
        f"pathvqa-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/pathvqa/pathvqa-00001-{sha(first)[:8]}.jpg"
    assert lines[2]["path"].endswith(".png")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "pathvqa-LICENSE").read_bytes() == LICENSE_TEXT
    assert pinned["served"][hf.CARD] == pathvqa.CARD_SHA256 and pinned["served"][FILE] == pathvqa.FILES[FILE]
    assert out == [
        f"no line for PathVQA test row 1, image: it repeats pathvqa-00001-{sha(first)[:8]}",
        f"no line for PathVQA test row 6, image: it repeats pathvqa-00002-{sha(second)[:8]}",
        "no line for PathVQA test row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'pathvqa.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "pathvqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the PathVQA set equals a fresh import of {pathvqa.SOURCE}"
    ]
    text = (corpus / "image" / "pathvqa.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "pathvqa.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "pathvqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'pathvqa.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: cc-by-4.0\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match=re.escape("the card's license is 'cc-by-4.0', not the reviewed 'mit'")):
        main(["import", "pathvqa", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
