import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import docmatix, github, hf, image_sets

MIT_TEXT = b"MIT License\n\nCopyright (c) <year> <copyright holders>\n\nPermission is hereby granted\n"
CARD = b"---\nlicense: mit\n---\n# Docmatix\n"
FILE2 = "data/train-00001-of-01106-97f9db5e7314e467.parquet"
FILE = "data/train-00000-of-01106-8ce5201aa40b0141.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(documents: list[list[bytes | None] | None]) -> pa.Table:
    """Rows of one document each, with its pages as a list in the ``images`` column as the dataset stores them."""
    page = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    cells = [
        None if pages is None else [({"bytes": data, "path": None} if data else None) for data in pages]
        for pages in documents
    ]
    texts = pa.array([[{"user": "q", "assistant": "a", "source": "s"}] for _ in documents])
    return pa.table({"images": pa.array(cells, type=pa.list_(page)), "texts": texts})


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """The pinned shard served from ``tmp_path``, with the card and the licence text, through the real ``hf.fetch``
    and ``github.fetch`` signatures with no network: a two-page document, a document that repeats a page, an empty
    one, one with a null page, one whose page is not an image, a null row, and a one-page document."""
    first, second, third = png(8, 6), jpeg(10, 4), png(3, 3)
    documents = [[first, second], [first], [], [None], [b"not an image"], None, [third]]
    pq.write_table(table(documents), tmp_path / "shard.parquet")
    pq.write_table(table([]), tmp_path / "empty.parquet")
    files = {FILE: tmp_path / "shard.parquet", FILE2: tmp_path / "empty.parquet"}
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (docmatix.REPO, docmatix.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            docmatix.LICENSE_OWNER,
            docmatix.LICENSE_REPO,
            docmatix.LICENSE_COMMIT,
            docmatix.LICENSE_PATH,
        )
        assert (sha256, cache) == (docmatix.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(MIT_TEXT)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", docmatix.REVISION)
    assert list(docmatix.FILES) == [FILE, FILE2]
    assert all(re.fullmatch(r"[0-9a-f]{64}", docmatix.FILES[f]) for f in (FILE, FILE2))
    assert docmatix.SOURCE == f"hf:datasets/HuggingFaceM4/Docmatix@{docmatix.REVISION}"
    assert docmatix.CARD_LICENSE == "mit" and docmatix.LICENSE == "MIT"


def test_read_gives_every_page_in_row_and_page_order_with_its_origin(pinned):
    found = list(docmatix.read(pinned["files"]))
    assert [(f.origin["row"], f.origin["column"]) for f in found] == [
        (0, "images[0]"),
        (0, "images[1]"),
        (1, "images[0]"),
        (4, "images[0]"),
        (6, "images[0]"),
    ]
    assert found[0].data == pinned["first"] and found[1].data == pinned["second"]
    assert found[0].origin == {
        "dataset": "docmatix",
        "source": docmatix.SOURCE,
        "sha256": docmatix.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "images[0]",
        "license": "MIT",
    }
    assert found[1].notes == "Docmatix train row 0, images[1]"


def test_the_mit_text_passes_and_another_text_is_refused():
    docmatix.check_license(MIT_TEXT)
    with pytest.raises(ValueError, match="not the MIT License"):
        docmatix.check_license(b"Apache License\n")


def test_import_writes_the_set_the_pages_and_the_licence_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "docmatix", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "docmatix.jsonl")
    assert [line["name"] for line in lines] == [
        f"docmatix-00001-{sha(first)[:8]}",
        f"docmatix-00002-{sha(second)[:8]}",
        f"docmatix-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/docmatix/docmatix-00001-{sha(first)[:8]}.png"
    assert lines[1]["path"].endswith(".jpg")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "docmatix-LICENSE").read_bytes() == MIT_TEXT
    assert pinned["served"][hf.CARD] == docmatix.CARD_SHA256 and pinned["served"][FILE] == docmatix.FILES[FILE]
    assert pinned["served"][FILE2] == docmatix.FILES[FILE2]
    assert out == [
        f"no line for Docmatix train row 1, images[0]: it repeats docmatix-00001-{sha(first)[:8]}",
        "no line for Docmatix train row 4, images[0]: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'docmatix.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "1 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "docmatix", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the Docmatix set equals a fresh import of {docmatix.SOURCE}"
    ]
    text = (corpus / "image" / "docmatix.jsonl").read_text().splitlines()
    tampered = json.loads(text[1])
    tampered["w"] = 11
    text[1] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "docmatix.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "docmatix", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'docmatix.jsonl'}:2: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: cc-by-4.0\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match="the card's license is 'cc-by-4.0', not the reviewed 'mit'"):
        main(["import", "docmatix", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
