import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, mmmu

APACHE = b"\n                                 Apache License\n                           Version 2.0, January 2004\n"
CARD = b"---\nlicense: apache-2.0\n---\n# MMMU\n"
ACCOUNTING = "Accounting/validation-00000-of-00001.parquet"
MATH = "Math/validation-00000-of-00001.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(rows: list[list[bytes | None]]) -> pa.Table:
    """Rows of up to seven image cells, as the dataset stores them: a struct of bytes and path, null where absent."""
    kind = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    columns = {}
    for index, column in enumerate(mmmu.COLUMNS):
        cells = [({"bytes": row[index], "path": None} if index < len(row) and row[index] else None) for row in rows]
        columns[column] = pa.array(cells, type=kind)
    columns["question"] = pa.array([f"q{n}" for n in range(len(rows))])
    return pa.table(columns)


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """The pinned files served from ``tmp_path``: two subjects' parquet files (the other 28 empty), the card and the
    Apache License, each fetched through the real ``hf.fetch`` and ``github.fetch`` signatures with no network."""
    files = {}
    first, second, third = png(8, 6), jpeg(10, 4), png(3, 3)
    pq.write_table(table([[first, None, second], [None], [first], [b"not an image"]]), tmp_path / "accounting.parquet")
    pq.write_table(table([[third]]), tmp_path / "math.parquet")
    empty = tmp_path / "empty.parquet"
    pq.write_table(table([]), empty)
    for name in mmmu.FILES:
        files[name] = {ACCOUNTING: tmp_path / "accounting.parquet", MATH: tmp_path / "math.parquet"}.get(name, empty)
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (mmmu.REPO, mmmu.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            mmmu.LICENSE_OWNER,
            mmmu.LICENSE_REPO,
            mmmu.LICENSE_COMMIT,
            mmmu.LICENSE_PATH,
        )
        assert (sha256, cache) == (mmmu.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(APACHE)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "tmp": tmp_path}


def test_every_file_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", mmmu.REVISION)
    assert len(mmmu.FILES) == 30 and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in mmmu.FILES.values())
    assert list(mmmu.FILES) == sorted(mmmu.FILES) and all(
        name.endswith("/validation-00000-of-00001.parquet") for name in mmmu.FILES
    )
    assert mmmu.SOURCE == f"hf:datasets/MMMU/MMMU@{mmmu.REVISION}"


def test_read_gives_every_image_cell_in_file_row_and_column_order_with_its_origin(pinned):
    files = {name: pinned["tmp"] / "x" for name in mmmu.FILES}
    files[ACCOUNTING], files[MATH] = pinned["tmp"] / "accounting.parquet", pinned["tmp"] / "math.parquet"
    for name, path in files.items():
        if path.name == "x":
            files[name] = pinned["tmp"] / "empty.parquet"
    found = list(mmmu.read(files))
    assert [(f.origin["file"], f.origin["row"], f.origin["column"]) for f in found] == [
        (ACCOUNTING, 0, "image_1"),
        (ACCOUNTING, 0, "image_3"),
        (ACCOUNTING, 2, "image_1"),
        (ACCOUNTING, 3, "image_1"),
        (MATH, 0, "image_1"),
    ]
    assert found[0].data == pinned["first"] and found[1].data == pinned["second"]
    assert found[0].origin == {
        "dataset": "mmmu",
        "source": mmmu.SOURCE,
        "sha256": mmmu.FILES[ACCOUNTING],
        "file": ACCOUNTING,
        "row": 0,
        "column": "image_1",
        "license": "Apache-2.0",
    }
    assert found[0].notes == "MMMU Accounting validation row 0, image_1"
    assert found[4].notes == "MMMU Math validation row 0, image_1"


def test_the_subject_is_the_files_directory_with_spaces_for_its_underscores():
    assert (
        mmmu.subject_of("Architecture_and_Engineering/validation-00000-of-00001.parquet")
        == "Architecture and Engineering"
    )


def test_the_apache_license_passes_and_another_text_is_refused():
    mmmu.check_license(APACHE)
    with pytest.raises(ValueError, match="not the Apache License, Version 2.0"):
        mmmu.check_license(b"MIT License\n")


def test_import_writes_the_set_the_images_and_the_license_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "mmmu", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "mmmu.jsonl")
    assert [line["name"] for line in lines] == [
        f"mmmu-00001-{sha(first)[:8]}",
        f"mmmu-00002-{sha(second)[:8]}",
        f"mmmu-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/mmmu/mmmu-00001-{sha(first)[:8]}.png" and lines[1]["path"].endswith(".jpg")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "mmmu-LICENSE").read_bytes() == APACHE
    assert pinned["served"][hf.CARD] == mmmu.CARD_SHA256 and pinned["served"][ACCOUNTING] == mmmu.FILES[ACCOUNTING]
    assert out == [
        f"no line for MMMU Accounting validation row 2, image_1: it repeats mmmu-00001-{sha(first)[:8]}",
        "no line for MMMU Accounting validation row 3, image_1: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'mmmu.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "1 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "mmmu", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{corpus}: the MMMU set equals a fresh import of {mmmu.SOURCE}"]
    tampered = json.loads((corpus / "image" / "mmmu.jsonl").read_text().splitlines()[1])
    tampered["w"] += 1
    text = (corpus / "image" / "mmmu.jsonl").read_text().splitlines()
    text[1] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "mmmu.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "mmmu", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [f"{corpus / 'image' / 'mmmu.jsonl'}:2: differs from a fresh import"]


def test_a_card_whose_license_is_not_the_reviewed_one_stops_the_import(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: cc-by-nc-sa-4.0\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match="the card's license is 'cc-by-nc-sa-4.0', not the reviewed 'apache-2.0'"):
        main(["import", "mmmu", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
