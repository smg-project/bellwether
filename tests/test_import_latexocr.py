import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, latexocr

LICENSE_TEXT = b"Apache License\n\n(the body of the text)\n"
CARD = b"---\nlicense: apache-2.0\n---\n# LaTeX-OCR\n"
FILE2 = "human_handwrite/test-00000-of-00001.parquet"
FILE = "full/test-00000-of-00001.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(images: list[bytes | None]) -> pa.Table:
    """Rows of one formula each, in the ``image`` column as the dataset stores it, null where absent."""
    kind = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    cells = [({"bytes": data, "path": None} if data else None) for data in images]
    return pa.table({"id": pa.array([f"r{n}" for n in range(len(images))]), "image": pa.array(cells, type=kind)})


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
        assert (repo, revision) == (latexocr.REPO, latexocr.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache, copy):
        assert (owner, repo, commit, path) == (
            latexocr.LICENSE_OWNER,
            latexocr.LICENSE_REPO,
            latexocr.LICENSE_COMMIT,
            latexocr.LICENSE_PATH,
        )
        assert (sha256, cache) == (latexocr.LICENSE_SHA256, tmp_path / "cache")
        assert copy == tmp_path / "corpus" / latexocr.LICENSE_COPY
        (tmp_path / "LICENSE").write_bytes(LICENSE_TEXT)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    # The window covers the whole seven-row shard here; the window test narrows it.
    monkeypatch.setattr(latexocr, "SHARDS", ((FILE, 0, 7),))
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", latexocr.REVISION)
    assert list(latexocr.FILES) == [FILE, FILE2]
    assert all(re.fullmatch(r"[0-9a-f]{64}", latexocr.FILES[f]) for f in (FILE, FILE2))
    assert latexocr.SOURCE == f"hf:datasets/linxy/LaTeX_OCR@{latexocr.REVISION}"
    assert latexocr.CARD_LICENSE == "apache-2.0" and latexocr.LICENSE == "Apache-2.0"
    assert latexocr.SHARDS == ((FILE, 0, 2000), (FILE2, 0, 70))


def test_read_gives_every_formula_in_row_order_with_its_origin(pinned):
    found = list(latexocr.read(pinned["files"]))
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
        "dataset": "latexocr",
        "source": latexocr.SOURCE,
        "sha256": latexocr.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "image",
        "license": "Apache-2.0",
    }
    assert found[0].notes == "LaTeX-OCR test row 0, image"
    assert found[4].notes == "LaTeX-OCR test row 5, image"


def test_read_keeps_to_the_window_of_rows(pinned, monkeypatch):
    monkeypatch.setattr(latexocr, "SHARDS", ((FILE, 2, 3),))
    assert [f.origin["row"] for f in latexocr.read(pinned["files"])] == [2, 3]
    monkeypatch.setattr(latexocr, "SHARDS", ((FILE, 5, 1), (FILE, 0, 1)))
    assert [f.origin["row"] for f in latexocr.read(pinned["files"])] == [5, 0]


def test_the_licence_text_passes_and_another_text_is_refused():
    latexocr.check_license(LICENSE_TEXT)
    with pytest.raises(ValueError, match=re.escape("not the Apache License 2.0 text")):
        latexocr.check_license(b"MIT License\n")


def test_import_writes_the_set_the_formulas_and_the_licence_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "latexocr", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "latexocr.jsonl")
    assert [line["name"] for line in lines] == [
        f"latexocr-00001-{sha(first)[:8]}",
        f"latexocr-00002-{sha(second)[:8]}",
        f"latexocr-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/latexocr/latexocr-00001-{sha(first)[:8]}.jpg"
    assert lines[2]["path"].endswith(".png")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "latexocr-LICENSE").read_bytes() == LICENSE_TEXT
    assert pinned["served"][hf.CARD] == latexocr.CARD_SHA256 and pinned["served"][FILE] == latexocr.FILES[FILE]
    assert out == [
        f"no line for LaTeX-OCR test row 1, image: it repeats latexocr-00001-{sha(first)[:8]}",
        f"no line for LaTeX-OCR test row 6, image: it repeats latexocr-00002-{sha(second)[:8]}",
        "no line for LaTeX-OCR test row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'latexocr.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "latexocr", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the LaTeX-OCR set equals a fresh import of {latexocr.SOURCE}"
    ]
    text = (corpus / "image" / "latexocr.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "latexocr.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "latexocr", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'latexocr.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: mit\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match=re.escape("the card's license is 'mit', not the reviewed 'apache-2.0'")):
        main(["import", "latexocr", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
