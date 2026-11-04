import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, gqa, hf, image_sets

LICENSE_TEXT = b"MIT License\n\n(the body of the text)\n"
CARD = b"---\nlicense: mit\n---\n# GQA\n"
FILE2 = "challenge_balanced_images/challenge-00000-of-00001.parquet"
FILE = "testdev_balanced_images/testdev-00000-of-00001.parquet"


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
    pq.write_table(table([]), tmp_path / "empty.parquet")
    files = {FILE: tmp_path / "shard.parquet", FILE2: tmp_path / "empty.parquet"}
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (gqa.REPO, gqa.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            gqa.LICENSE_OWNER,
            gqa.LICENSE_REPO,
            gqa.LICENSE_COMMIT,
            gqa.LICENSE_PATH,
        )
        assert (sha256, cache) == (gqa.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(LICENSE_TEXT)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", gqa.REVISION)
    assert list(gqa.FILES) == [FILE, FILE2]
    assert all(re.fullmatch(r"[0-9a-f]{64}", gqa.FILES[f]) for f in (FILE, FILE2))
    assert gqa.SOURCE == f"hf:datasets/lmms-lab-encoder/GQA@{gqa.REVISION}"
    assert gqa.CARD_LICENSE == "mit" and gqa.LICENSE == "MIT"


def test_read_gives_every_photograph_in_row_order_with_its_origin(pinned):
    found = list(gqa.read(pinned["files"]))
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
        "dataset": "gqa",
        "source": gqa.SOURCE,
        "sha256": gqa.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "image",
        "license": "MIT",
    }
    assert found[0].notes == "GQA testdev row 0, image"
    assert found[4].notes == "GQA testdev row 5, image"


def test_the_licence_text_passes_and_another_text_is_refused():
    gqa.check_license(LICENSE_TEXT)
    with pytest.raises(ValueError, match=re.escape("not the MIT License")):
        gqa.check_license(b"Apache License\n")


def test_import_writes_the_set_the_photographs_and_the_licence_copy_and_check_passes_then_fails(
    pinned, tmp_path, capsys
):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "gqa", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "gqa.jsonl")
    assert [line["name"] for line in lines] == [
        f"gqa-00001-{sha(first)[:8]}",
        f"gqa-00002-{sha(second)[:8]}",
        f"gqa-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/gqa/gqa-00001-{sha(first)[:8]}.jpg"
    assert lines[2]["path"].endswith(".png")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "gqa-LICENSE").read_bytes() == LICENSE_TEXT
    assert pinned["served"][hf.CARD] == gqa.CARD_SHA256 and pinned["served"][FILE] == gqa.FILES[FILE]
    assert pinned["served"][FILE2] == gqa.FILES[FILE2]
    assert out == [
        f"no line for GQA testdev row 1, image: it repeats gqa-00001-{sha(first)[:8]}",
        f"no line for GQA testdev row 6, image: it repeats gqa-00002-{sha(second)[:8]}",
        "no line for GQA testdev row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'gqa.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "gqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{corpus}: the GQA set equals a fresh import of {gqa.SOURCE}"]
    text = (corpus / "image" / "gqa.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "gqa.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "gqa", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [f"{corpus / 'image' / 'gqa.jsonl'}:3: differs from a fresh import"]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: cc-by-4.0\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match=re.escape("the card's license is 'cc-by-4.0', not the reviewed 'mit'")):
        main(["import", "gqa", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
