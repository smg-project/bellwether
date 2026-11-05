import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, realworldqa

LICENSE_TEXT = b"Creative Commons Attribution 4.0 International\n\n(the body of the text)\n"
CARD = b"---\nlicense: cc-by-4.0\n---\n# RealWorldQA\n"
FILE = "data/test-00000-of-00002.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(images: list[bytes | None]) -> pa.Table:
    """Rows of one photograph each, in the ``image`` column as the dataset stores it, null where absent."""
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
        assert (repo, revision) == (realworldqa.REPO, realworldqa.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache, copy):
        assert (owner, repo, commit, path) == (
            realworldqa.LICENSE_OWNER,
            realworldqa.LICENSE_REPO,
            realworldqa.LICENSE_COMMIT,
            realworldqa.LICENSE_PATH,
        )
        assert (sha256, cache) == (realworldqa.LICENSE_SHA256, tmp_path / "cache")
        assert copy == tmp_path / "corpus" / realworldqa.LICENSE_COPY
        (tmp_path / "LICENSE").write_bytes(LICENSE_TEXT)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", realworldqa.REVISION)
    assert list(realworldqa.FILES) == [FILE] and re.fullmatch(r"[0-9a-f]{64}", realworldqa.FILES[FILE])
    assert realworldqa.SOURCE == f"hf:datasets/lmms-lab-encoder/RealWorldQA@{realworldqa.REVISION}"
    assert realworldqa.CARD_LICENSE == "cc-by-4.0" and realworldqa.LICENSE == "CC-BY-4.0"


def test_read_gives_every_photograph_in_row_order_with_its_origin(pinned):
    found = list(realworldqa.read(pinned["files"]))
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
        "dataset": "realworldqa",
        "source": realworldqa.SOURCE,
        "sha256": realworldqa.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "image",
        "license": "CC-BY-4.0",
    }
    assert found[0].notes == "RealWorldQA test row 0, image"
    assert found[4].notes == "RealWorldQA test row 5, image"


def test_the_licence_text_passes_and_another_text_is_refused():
    realworldqa.check_license(LICENSE_TEXT)
    with pytest.raises(ValueError, match=re.escape("not the CC BY 4.0 licence text")):
        realworldqa.check_license(b"Apache License\n")


def test_import_writes_the_set_the_photographs_and_the_licence_copy_and_check_passes_then_fails(
    pinned, tmp_path, capsys
):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "realworldqa", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "realworldqa.jsonl")
    assert [line["name"] for line in lines] == [
        f"realworldqa-00001-{sha(first)[:8]}",
        f"realworldqa-00002-{sha(second)[:8]}",
        f"realworldqa-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/realworldqa/realworldqa-00001-{sha(first)[:8]}.jpg"
    assert lines[2]["path"].endswith(".png")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "realworldqa-LICENSE").read_bytes() == LICENSE_TEXT
    assert pinned["served"][hf.CARD] == realworldqa.CARD_SHA256 and pinned["served"][FILE] == realworldqa.FILES[FILE]
    assert out == [
        f"no line for RealWorldQA test row 1, image: it repeats realworldqa-00001-{sha(first)[:8]}",
        f"no line for RealWorldQA test row 6, image: it repeats realworldqa-00002-{sha(second)[:8]}",
        "no line for RealWorldQA test row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'realworldqa.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "realworldqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the RealWorldQA set equals a fresh import of {realworldqa.SOURCE}"
    ]
    text = (corpus / "image" / "realworldqa.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "realworldqa.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "realworldqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'realworldqa.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: mit\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match=re.escape("the card's license is 'mit', not the reviewed 'cc-by-4.0'")):
        main(["import", "realworldqa", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
