import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, textvqa

CC_BY = b"Creative Commons Attribution 4.0 International\n\n Creative Commons Corporation is not a law firm\n"
CARD = b"---\ndataset_info:\n  features:\n  - name: image\n    dtype: image\n---\n# TextVQA\n"
FIRST = "data/validation-00000-of-00003.parquet"
LAST = "data/validation-00002-of-00003.parquet"


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
    """The pinned files served from ``tmp_path``: two of the three files with photographs (the middle one empty), the
    card and the licence text, through the real ``hf.fetch`` and ``github.fetch`` signatures with no network."""
    first, second, third = jpeg(8, 6), jpeg(10, 4), png(3, 3)
    pq.write_table(table([first, first, second, b"not an image"]), tmp_path / "first.parquet")
    pq.write_table(table([third, None, second]), tmp_path / "last.parquet")
    pq.write_table(table([]), tmp_path / "empty.parquet")
    files = {
        name: {FIRST: tmp_path / "first.parquet", LAST: tmp_path / "last.parquet"}.get(name, tmp_path / "empty.parquet")
        for name in textvqa.FILES
    }
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (textvqa.REPO, textvqa.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache, copy):
        assert (owner, repo, commit, path) == (
            textvqa.LICENSE_OWNER,
            textvqa.LICENSE_REPO,
            textvqa.LICENSE_COMMIT,
            textvqa.LICENSE_PATH,
        )
        assert (sha256, cache) == (textvqa.LICENSE_SHA256, tmp_path / "cache")
        assert copy == tmp_path / "corpus" / textvqa.LICENSE_COPY
        (tmp_path / "LICENSE").write_bytes(CC_BY)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_every_file_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", textvqa.REVISION)
    assert len(textvqa.FILES) == 3 and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in textvqa.FILES.values())
    assert list(textvqa.FILES) == [f"data/validation-0000{n}-of-00003.parquet" for n in range(3)]
    assert textvqa.SOURCE == f"hf:datasets/lmms-lab-encoder/textvqa@{textvqa.REVISION}"
    assert textvqa.CARD_LICENSE is None and textvqa.LICENSE == "CC-BY-4.0"


def test_read_gives_every_photograph_in_file_and_row_order_with_its_origin(pinned):
    found = list(textvqa.read(pinned["files"]))
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
        "dataset": "textvqa",
        "source": textvqa.SOURCE,
        "sha256": textvqa.FILES[FIRST],
        "file": FIRST,
        "row": 0,
        "column": "image",
        "license": "CC-BY-4.0",
    }
    assert found[0].notes == "TextVQA validation row 0, image" and found[4].notes == "TextVQA validation row 0, image"


def test_the_cc_by_licence_passes_and_another_text_is_refused():
    textvqa.check_license(CC_BY)
    with pytest.raises(ValueError, match="not the CC BY 4.0 licence text"):
        textvqa.check_license(b"Creative Commons Attribution-ShareAlike 4.0 International\n")


def test_import_writes_the_set_the_photographs_and_the_licence_copy_and_check_passes_then_fails(
    pinned, tmp_path, capsys
):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "textvqa", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "textvqa.jsonl")
    assert [line["name"] for line in lines] == [
        f"textvqa-00001-{sha(first)[:8]}",
        f"textvqa-00002-{sha(second)[:8]}",
        f"textvqa-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/textvqa/textvqa-00001-{sha(first)[:8]}.jpg"
    assert lines[2]["path"].endswith(".png")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "textvqa-LICENSE").read_bytes() == CC_BY
    assert pinned["served"][hf.CARD] == textvqa.CARD_SHA256 and pinned["served"][FIRST] == textvqa.FILES[FIRST]
    assert out == [
        f"no line for TextVQA validation row 1, image: it repeats textvqa-00001-{sha(first)[:8]}",
        f"no line for TextVQA validation row 2, image: it repeats textvqa-00002-{sha(second)[:8]}",
        "no line for TextVQA validation row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'textvqa.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "textvqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the TextVQA set equals a fresh import of {textvqa.SOURCE}"
    ]
    text = (corpus / "image" / "textvqa.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "textvqa.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "textvqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'textvqa.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_that_gains_a_licence_statement_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: cc-by-4.0\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match="the card's license is 'cc-by-4.0', not the reviewed None"):
        main(["import", "textvqa", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
