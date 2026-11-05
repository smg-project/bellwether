import hashlib
import io
import json
import re
import zipfile

import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import coco, github, image_sets, pinned, web

TERMS = (
    b"<h1>Terms of Use</h1><p>licensed under a <a>Creative Commons Attribution 4.0 License</a>.</p>"
    b"<p>Use of the images must abide by the Flickr Terms of Use.</p>"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive(members: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return out.getvalue()


@pytest.fixture
def pinned_archive(tmp_path, monkeypatch):
    """The archive and the terms page served from ``tmp_path`` through the real ``web.fetch`` and ``github.fetch``
    signatures with no network: three photographs, one repeated under another name, and a text file the import
    does not read."""
    first, second, third = jpeg(8, 6), jpeg(10, 4), png(3, 3)
    members = {
        "val2017/000000000139.jpg": first,
        "val2017/000000000285.jpg": second,
        "val2017/000000000632.jpg": first,
        "val2017/000000000724.jpg": third,
        "val2017/readme.txt": b"not an image",
    }
    (tmp_path / "val2017.zip").write_bytes(archive(members))
    served = {}

    def fetch_web(url, sha256, cache):
        assert (url, sha256, cache) == (coco.URL, coco.ARCHIVE_SHA256, tmp_path / "cache")
        served["archive"] = sha256
        return tmp_path / "val2017.zip"

    def fetch_github(owner, repo, commit, path, sha256, cache, copy):
        assert (owner, repo, commit, path) == (
            coco.LICENSE_OWNER,
            coco.LICENSE_REPO,
            coco.LICENSE_COMMIT,
            coco.LICENSE_PATH,
        )
        assert (sha256, cache) == (coco.LICENSE_SHA256, tmp_path / "cache")
        assert copy == tmp_path / "corpus" / coco.LICENSE_COPY
        (tmp_path / "termsofuse.htm").write_bytes(TERMS)
        return tmp_path / "termsofuse.htm"

    monkeypatch.setattr(web, "fetch", fetch_web)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "archive": tmp_path / "val2017.zip"}


def test_the_archive_and_the_terms_are_pinned_by_sha256_and_a_full_commit():
    assert re.fullmatch(r"[0-9a-f]{64}", coco.ARCHIVE_SHA256) and re.fullmatch(r"[0-9a-f]{64}", coco.LICENSE_SHA256)
    assert re.fullmatch(r"[0-9a-f]{40}", coco.LICENSE_COMMIT)
    assert coco.SOURCE == "http://images.cocodataset.org/zips/val2017.zip"
    assert coco.LICENSE == "images: Flickr Terms of Use; annotations: CC-BY-4.0"


def test_read_gives_every_jpeg_member_in_name_order_with_its_origin(pinned_archive):
    found = list(coco.read(pinned_archive["archive"]))
    assert [f.origin["member"] for f in found] == [
        "val2017/000000000139.jpg",
        "val2017/000000000285.jpg",
        "val2017/000000000632.jpg",
        "val2017/000000000724.jpg",
    ]
    assert found[0].data == pinned_archive["first"] and found[2].data == pinned_archive["first"]
    assert found[0].origin == {
        "dataset": "coco",
        "source": coco.SOURCE,
        "sha256": coco.ARCHIVE_SHA256,
        "file": "val2017.zip",
        "member": "val2017/000000000139.jpg",
        "license": coco.LICENSE,
    }
    assert found[0].notes == "COCO val2017 val2017/000000000139.jpg"


def test_the_terms_page_passes_and_a_page_without_both_statements_is_refused():
    coco.check_license(TERMS)
    with pytest.raises(ValueError, match="not the terms of use the import was reviewed for"):
        coco.check_license(b"<p>All rights reserved.</p>")


def test_import_writes_the_set_the_photographs_and_the_terms_copy_and_check_passes_then_fails(
    pinned_archive, tmp_path, capsys
):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "coco", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned_archive["first"], pinned_archive["second"], pinned_archive["third"]
    lines = image_sets.read_set(corpus / "image" / "coco.jsonl")
    assert [line["name"] for line in lines] == [
        f"coco-00001-{sha(first)[:8]}",
        f"coco-00002-{sha(second)[:8]}",
        f"coco-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/coco/coco-00001-{sha(first)[:8]}.jpg" and lines[2]["path"].endswith(".png")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "coco-LICENSE").read_bytes() == TERMS
    assert pinned_archive["served"]["archive"] == coco.ARCHIVE_SHA256
    assert out == [
        f"no line for COCO val2017 val2017/000000000632.jpg: it repeats coco-00001-{sha(first)[:8]}",
        f"{corpus / 'image' / 'coco.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "1 left out as repeats, 0 rows with no image",
    ]
    assert main(["import", "coco", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{corpus}: the COCO set equals a fresh import of {coco.SOURCE}"]
    text = (corpus / "image" / "coco.jsonl").read_text().splitlines()
    tampered = json.loads(text[1])
    tampered["w"] = 11
    text[1] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "coco.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "coco", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [f"{corpus / 'image' / 'coco.jsonl'}:2: differs from a fresh import"]


def test_web_fetch_serves_a_cached_file_with_the_pinned_sha256_and_refuses_other_bytes(tmp_path, monkeypatch):
    data = b"archive bytes"
    cached = tmp_path / "web" / sha(data) / "val2017.zip"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(data)
    assert web.fetch("http://example.invalid/zips/val2017.zip", sha(data), tmp_path) == cached

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    monkeypatch.setattr(web, "urlopen", lambda url, timeout: Response(b"other bytes"))
    with pytest.raises(ValueError, match="is not the pinned"):
        web.fetch("http://example.invalid/zips/other.zip", sha(data), tmp_path)
    assert not (tmp_path / "web" / sha(data) / "other.zip").exists()
    assert not (tmp_path / "web" / sha(data) / "other.zip.partial").exists()
    monkeypatch.setattr(web, "urlopen", lambda url, timeout: Response(b"other bytes"))
    fetched = web.fetch("http://example.invalid/zips/other.zip", sha(b"other bytes"), tmp_path)
    assert fetched == tmp_path / "web" / sha(b"other bytes") / "other.zip" and fetched.read_bytes() == b"other bytes"
    assert pinned.sha256_of_file(fetched) == sha(b"other bytes")
