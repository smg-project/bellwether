import hashlib
import json
import re
import zipfile

import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, screenspot

APACHE_TEXT = b"Apache License\nVersion 2.0, January 2004\n\n(the body of the text)\n"
CARD = b"---\nlicense: apache-2.0\n---\n"
DIR = "screenspotv2_image/"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """The pinned archive served from ``tmp_path``, with the card and the licence text, through the real ``hf.fetch``
    and ``github.fetch`` signatures with no network: screenshots in both suffixes, a repeat, a member that is not an
    image, and members of other kinds that are not read."""
    first, second, third = png(8, 6), jpeg(10, 4), png(3, 3)
    archive = tmp_path / "screenspotv2_image.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(DIR, b"")
        z.writestr(DIR + "a_first.png", first)
        z.writestr(DIR + "b_second.jpg", second)
        z.writestr(DIR + "c_repeat.png", first)
        z.writestr(DIR + "d_bad.png", b"not an image")
        z.writestr(DIR + "e_third.png", third)
        z.writestr(DIR + "f_archive.tar", b"not read")
        z.writestr(DIR + "g_notes.txt", b"not read")
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (screenspot.REPO, screenspot.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        assert filename == screenspot.ARCHIVE
        return archive

    def fetch_github(owner, repo, commit, path, sha256, cache, copy):
        assert (owner, repo, commit, path) == (
            screenspot.LICENSE_OWNER,
            screenspot.LICENSE_REPO,
            screenspot.LICENSE_COMMIT,
            screenspot.LICENSE_PATH,
        )
        assert (sha256, cache) == (screenspot.LICENSE_SHA256, tmp_path / "cache")
        assert copy == tmp_path / "corpus" / screenspot.LICENSE_COPY
        (tmp_path / "LICENSE").write_bytes(APACHE_TEXT)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "archive": archive}


def test_the_archive_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", screenspot.REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", screenspot.ARCHIVE_SHA256)
    assert screenspot.SOURCE == f"hf:datasets/OS-Copilot/ScreenSpot-v2@{screenspot.REVISION}"
    assert screenspot.CARD_LICENSE == "apache-2.0" and screenspot.LICENSE == "Apache-2.0"


def test_read_gives_every_png_and_jpeg_member_in_name_order_with_its_origin(pinned):
    found = list(screenspot.read(pinned["archive"]))
    assert [f.origin["member"] for f in found] == [
        DIR + "a_first.png",
        DIR + "b_second.jpg",
        DIR + "c_repeat.png",
        DIR + "d_bad.png",
        DIR + "e_third.png",
    ]
    assert found[0].data == pinned["first"] and found[1].data == pinned["second"]
    assert found[0].origin == {
        "dataset": "screenspot",
        "source": screenspot.SOURCE,
        "sha256": screenspot.ARCHIVE_SHA256,
        "file": "screenspotv2_image.zip",
        "member": DIR + "a_first.png",
        "license": "Apache-2.0",
    }
    assert found[1].notes == "ScreenSpot-v2 " + DIR + "b_second.jpg"


def test_the_apache_text_passes_and_another_text_is_refused():
    screenspot.check_license(APACHE_TEXT)
    with pytest.raises(ValueError, match="not the Apache License 2.0 text"):
        screenspot.check_license(b"MIT License\n")


def test_import_writes_the_set_the_screenshots_and_the_licence_copy_and_check_passes_then_fails(
    pinned, tmp_path, capsys
):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "screenspot", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = image_sets.read_set(corpus / "image" / "screenspot.jsonl")
    assert [line["name"] for line in lines] == [
        f"screenspot-00001-{sha(first)[:8]}",
        f"screenspot-00002-{sha(second)[:8]}",
        f"screenspot-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"images/screenspot/screenspot-00001-{sha(first)[:8]}.png"
    assert lines[1]["path"].endswith(".jpg")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "screenspot-LICENSE").read_bytes() == APACHE_TEXT
    assert pinned["served"][hf.CARD] == screenspot.CARD_SHA256
    assert pinned["served"][screenspot.ARCHIVE] == screenspot.ARCHIVE_SHA256
    assert out == [
        f"no line for ScreenSpot-v2 {DIR}c_repeat.png: it repeats screenspot-00001-{sha(first)[:8]}",
        f"no line for ScreenSpot-v2 {DIR}d_bad.png: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'screenspot.jsonl'}: 3 images, {len(first) + len(second) + len(third)} bytes, "
        "1 left out as repeats, 1 rows with no image",
    ]
    assert main(["import", "screenspot", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the ScreenSpot-v2 set equals a fresh import of {screenspot.SOURCE}"
    ]
    text = (corpus / "image" / "screenspot.jsonl").read_text().splitlines()
    tampered = json.loads(text[1])
    tampered["h"] = 5
    text[1] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "screenspot.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "screenspot", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'image' / 'screenspot.jsonl'}:2: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: mit\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match="the card's license is 'mit', not the reviewed 'apache-2.0'"):
        main(["import", "screenspot", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
