import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import docvqa, github, hf, image_sets

APACHE = b"\n                                 Apache License\n                           Version 2.0, January 2004\n"
CARD = b"---\nlicense: apache-2.0\n---\n# DocVQA\n"
FIRST = "DocVQA/validation-00000-of-00006.parquet"
LAST = "DocVQA/validation-00005-of-00006.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(pages: list[bytes | None]) -> pa.Table:
    """Rows of one question each, with the page in the ``image`` column as the dataset stores it, null where absent."""
    kind = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    cells = [({"bytes": page, "path": None} if page else None) for page in pages]
    return pa.table({"question": pa.array([f"q{n}" for n in range(len(pages))]), "image": pa.array(cells, type=kind)})


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """The pinned files served from ``tmp_path``: two of the six files with pages (the other four empty), the card and
    the Apache License, through the real ``hf.fetch`` and ``github.fetch`` signatures with no network."""
    first, second, third = png(8, 6), jpeg(10, 4), png(3, 3)
    pq.write_table(table([first, first, second, b"not an image"]), tmp_path / "first.parquet")
    pq.write_table(table([third, None, second]), tmp_path / "last.parquet")
    pq.write_table(table([]), tmp_path / "empty.parquet")
    files = {
        name: {FIRST: tmp_path / "first.parquet", LAST: tmp_path / "last.parquet"}.get(name, tmp_path / "empty.parquet")
        for name in docvqa.FILES
    }
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (docvqa.REPO, docvqa.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            docvqa.LICENSE_OWNER,
            docvqa.LICENSE_REPO,
            docvqa.LICENSE_COMMIT,
            docvqa.LICENSE_PATH,
        )
        assert (sha256, cache) == (docvqa.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(APACHE)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_every_file_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", docvqa.REVISION)
    assert len(docvqa.FILES) == 6 and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in docvqa.FILES.values())
    assert list(docvqa.FILES) == [f"DocVQA/validation-0000{n}-of-00006.parquet" for n in range(6)]
    assert docvqa.SOURCE == f"hf:datasets/lmms-lab-encoder/DocVQA@{docvqa.REVISION}"


def test_read_gives_every_page_in_file_and_row_order_with_its_origin(pinned):
    found = list(docvqa.read(pinned["files"]))
    assert [(f.origin["file"], f.origin["row"]) for f in found] == [
        (FIRST, 0),
        (FIRST, 1),
        (FIRST, 2),
        (FIRST, 3),
        (LAST, 0),
        (LAST, 2),
    ]
    assert found[0].data == pinned["first"] and found[2].data == pinned["second"]
    assert found[0].origin == {
        "dataset": "docvqa",
        "source": docvqa.SOURCE,
        "sha256": docvqa.FILES[FIRST],
        "file": FIRST,
        "row": 0,
        "column": "image",
        "license": "Apache-2.0",
    }
    assert found[0].notes == "DocVQA validation row 0, image" and found[4].notes == "DocVQA validation row 0, image"


def test_the_apache_license_passes_and_another_text_is_refused():
    docvqa.check_license(APACHE)
    with pytest.raises(ValueError, match="not the Apache License, Version 2.0"):
        docvqa.check_license(b"MIT License\n")


def test_import_writes_the_set_the_pages_and_the_license_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "docvqa", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "docvqa.jsonl")
    assert [line["name"] for line in lines] == [
        f"docvqa-00001-{sha(first)[:8]}",
        f"docvqa-00002-{sha(second)[:8]}",
        f"docvqa-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/docvqa/docvqa-00001-{sha(first)[:8]}.png" and lines[1]["path"].endswith(".jpg")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "docvqa-LICENSE").read_bytes() == APACHE
    assert pinned["served"][hf.CARD] == docvqa.CARD_SHA256 and pinned["served"][FIRST] == docvqa.FILES[FIRST]
    assert out == [
        f"no line for DocVQA validation row 1, image: it repeats docvqa-00001-{sha(first)[:8]}",
        f"no line for DocVQA validation row 2, image: it repeats docvqa-00002-{sha(second)[:8]}",
        "no line for DocVQA validation row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'docvqa.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "docvqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the DocVQA set equals a fresh import of {docvqa.SOURCE}"
    ]
    text = (corpus / "image" / "docvqa.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "docvqa.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "docvqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'docvqa.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_license_is_not_the_reviewed_one_stops_the_import(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: cc-by-4.0\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match="the card's license is 'cc-by-4.0', not the reviewed 'apache-2.0'"):
        main(["import", "docvqa", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
