import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, ocrbench

MIT_TEXT = b"MIT License\n\nCopyright (c) 2023 Yuliang Liu\n\nPermission is hereby granted\n"
CARD = b"---\ndataset_info:\n  features:\n  - name: image\n    dtype: image\n---\n# OCRBench\n"
FILE = "data/test-00000-of-00001.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(pictures: list[bytes | None]) -> pa.Table:
    """Rows of one question each, with the picture in the ``image`` column as the dataset stores it, null where
    absent."""
    kind = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    cells = [({"bytes": item, "path": None} if item else None) for item in pictures]
    return pa.table(
        {
            "question": pa.array([f"q{n}" for n in range(len(pictures))]),
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
            ocrbench.REPO,
            ocrbench.REVISION,
        ) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache, copy):
        assert (owner, repo, commit, path) == (
            ocrbench.LICENSE_OWNER,
            ocrbench.LICENSE_REPO,
            ocrbench.LICENSE_COMMIT,
            ocrbench.LICENSE_PATH,
        )
        assert (sha256, cache) == (ocrbench.LICENSE_SHA256, tmp_path / "cache")
        assert copy == tmp_path / "corpus" / ocrbench.LICENSE_COPY
        (tmp_path / "LICENSE").write_bytes(MIT_TEXT)
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
    assert re.fullmatch(r"[0-9a-f]{40}", ocrbench.REVISION)
    assert list(ocrbench.FILES) == [FILE] and re.fullmatch(r"[0-9a-f]{64}", ocrbench.FILES[FILE])
    assert ocrbench.SOURCE == f"hf:datasets/echo840/OCRBench@{ocrbench.REVISION}"
    assert ocrbench.CARD_LICENSE is None and ocrbench.LICENSE == "MIT"


def test_read_gives_every_picture_in_row_order_with_its_origin(pinned):
    found = list(ocrbench.read(pinned["files"]))
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
        "dataset": "ocrbench",
        "source": ocrbench.SOURCE,
        "sha256": ocrbench.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "image",
        "license": "MIT",
    }
    assert found[0].notes == "OCRBench test row 0, image" and found[4].notes == "OCRBench test row 5, image"


def test_the_licence_text_passes_and_another_text_is_refused():
    ocrbench.check_license(MIT_TEXT)
    with pytest.raises(ValueError, match="not the MIT License"):
        ocrbench.check_license(b"Apache License\n")


def test_import_writes_the_set_the_pictures_and_the_licence_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "ocrbench", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "ocrbench.jsonl")
    assert [line["name"] for line in lines] == [
        f"ocrbench-00001-{sha(first)[:8]}",
        f"ocrbench-00002-{sha(second)[:8]}",
        f"ocrbench-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/ocrbench/ocrbench-00001-{sha(first)[:8]}.png"
    assert lines[1]["path"].endswith(".jpg")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "ocrbench-LICENSE").read_bytes() == MIT_TEXT
    assert pinned["served"][hf.CARD] == ocrbench.CARD_SHA256 and pinned["served"][FILE] == ocrbench.FILES[FILE]
    assert out == [
        f"no line for OCRBench test row 1, image: it repeats ocrbench-00001-{sha(first)[:8]}",
        f"no line for OCRBench test row 6, image: it repeats ocrbench-00002-{sha(second)[:8]}",
        "no line for OCRBench test row 3, image: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'ocrbench.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no image",
    ]
    assert (
        main(
            [
                "import",
                "ocrbench",
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
        f"{corpus}: the OCRBench set equals a fresh import of {ocrbench.SOURCE}"
    ]
    text = (corpus / "image" / "ocrbench.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["bits"] = 16
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "ocrbench.jsonl").write_text("".join(line + "\n" for line in text))
    assert (
        main(
            [
                "import",
                "ocrbench",
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
        f"{corpus / 'image' / 'ocrbench.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: apache-2.0\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match="the card's license is 'apache-2.0', not the reviewed None"):
        main(
            [
                "import",
                "ocrbench",
                "--corpus",
                str(tmp_path / "corpus"),
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
