import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, pubtabnet

LICENSE_TEXT = (
    b"Community Data License Agreement \xe2\x80\x93 Permissive \xe2\x80\x93 Version 1.0\n\n(the body of the text)\n"
)
CARD = b"---\nlicense: cdla-permissive-1.0\n---\n# PubTabNet\n"
FILE = "data/validation-00000-of-00001.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(images: list[bytes | None]) -> pa.Table:
    """Rows of one table each, in the ``image`` column as the dataset stores it, null where absent."""
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
        assert (repo, revision) == (pubtabnet.REPO, pubtabnet.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            pubtabnet.LICENSE_OWNER,
            pubtabnet.LICENSE_REPO,
            pubtabnet.LICENSE_COMMIT,
            pubtabnet.LICENSE_PATH,
        )
        assert (sha256, cache) == (pubtabnet.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(LICENSE_TEXT)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    # The window covers the whole seven-row shard here; the window test narrows it.
    monkeypatch.setattr(pubtabnet, "SHARDS", ((FILE, 0, 7),))
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", pubtabnet.REVISION)
    assert list(pubtabnet.FILES) == [FILE] and re.fullmatch(r"[0-9a-f]{64}", pubtabnet.FILES[FILE])
    assert pubtabnet.SOURCE == f"hf:datasets/apoidea/pubtabnet-html@{pubtabnet.REVISION}"
    assert pubtabnet.CARD_LICENSE == "cdla-permissive-1.0" and pubtabnet.LICENSE == "CDLA-Permissive-1.0"
    assert pubtabnet.SHARDS == ((FILE, 0, 2000),)


def test_read_gives_every_table_in_row_order_with_its_origin(pinned):
    found = list(pubtabnet.read(pinned["files"]))
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
        "dataset": "pubtabnet",
        "source": pubtabnet.SOURCE,
        "sha256": pubtabnet.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "image",
        "license": "CDLA-Permissive-1.0",
    }
    assert found[0].notes == "PubTabNet validation row 0, image"
    assert found[4].notes == "PubTabNet validation row 5, image"


def test_read_keeps_to_the_window_of_rows(pinned, monkeypatch):
    monkeypatch.setattr(pubtabnet, "SHARDS", ((FILE, 2, 3),))
    assert [f.origin["row"] for f in pubtabnet.read(pinned["files"])] == [2, 3]
    monkeypatch.setattr(pubtabnet, "SHARDS", ((FILE, 5, 1), (FILE, 0, 1)))
    assert [f.origin["row"] for f in pubtabnet.read(pinned["files"])] == [5, 0]


def test_the_licence_text_passes_and_another_text_is_refused():
    pubtabnet.check_license(LICENSE_TEXT)
    with pytest.raises(ValueError, match=re.escape("not the CDLA-Permissive-1.0 text")):
        pubtabnet.check_license(b"Apache License\n")


def test_import_writes_the_set_the_tables_and_the_licence_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "pubtabnet", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "pubtabnet.jsonl")
    assert [line["name"] for line in lines] == [
        f"pubtabnet-00001-{sha(first)[:8]}",
        f"pubtabnet-00002-{sha(second)[:8]}",
        f"pubtabnet-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/pubtabnet/pubtabnet-00001-{sha(first)[:8]}.jpg"
    assert lines[2]["path"].endswith(".png")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "pubtabnet-LICENSE").read_bytes() == LICENSE_TEXT
    assert pinned["served"][hf.CARD] == pubtabnet.CARD_SHA256 and pinned["served"][FILE] == pubtabnet.FILES[FILE]
    assert out == [
        f"no line for PubTabNet validation row 1, image: it repeats pubtabnet-00001-{sha(first)[:8]}",
        f"no line for PubTabNet validation row 6, image: it repeats pubtabnet-00002-{sha(second)[:8]}",
        "no line for PubTabNet validation row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'pubtabnet.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "pubtabnet", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the PubTabNet set equals a fresh import of {pubtabnet.SOURCE}"
    ]
    text = (corpus / "image" / "pubtabnet.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "pubtabnet.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "pubtabnet", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'pubtabnet.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: mit\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(
        ValueError, match=re.escape("the card's license is 'mit', not the reviewed 'cdla-permissive-1.0'")
    ):
        main(["import", "pubtabnet", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
