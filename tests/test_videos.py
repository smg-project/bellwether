import hashlib
import json
import struct

import pytest

from bellwether import videos
from bellwether.cli import NOT_IMPLEMENTED, main

LFS_POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:{}\nsize {}\n"


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", 8 + len(body)) + kind + body


def mvhd(timescale: int, duration: int) -> bytes:
    return box(b"mvhd", bytes(4) + struct.pack(">II", 0, 0) + struct.pack(">II", timescale, duration) + bytes(80))


def tkhd(w: int, h: int) -> bytes:
    return box(b"tkhd", bytes(4) + bytes(20) + bytes(8) + bytes(8) + bytes(36) + struct.pack(">II", w << 16, h << 16))


def trak(handler: bytes, codec: bytes, w: int = 0, h: int = 0) -> bytes:
    hdlr = box(b"hdlr", bytes(8) + handler + bytes(13))
    entry = box(codec, bytes(70))
    stsd = box(b"stsd", bytes(4) + struct.pack(">I", 1) + entry)
    return box(b"trak", tkhd(w, h) + box(b"mdia", hdlr + box(b"minf", box(b"stbl", stsd))))


def mp4(brand: bytes = b"isom", seconds: float = 5.0, w: int = 320, h: int = 240, audio: bool = False) -> bytes:
    tracks = trak(b"vide", b"avc1", w, h) + (trak(b"soun", b"mp4a") if audio else b"")
    return (
        box(b"ftyp", brand + bytes(4) + brand)
        + box(b"moov", mvhd(1000, int(seconds * 1000)) + tracks)
        + box(b"mdat", b"\x00")
    )


def ebml_element(element_id: bytes, body: bytes) -> bytes:
    assert len(body) < 127
    return element_id + bytes([0x80 | len(body)]) + body


def webm(seconds: float = 5.0, w: int = 320, h: int = 240, codec: bytes = b"V_VP9", audio: bool = False) -> bytes:
    header = ebml_element(b"\x1a\x45\xdf\xa3", ebml_element(b"\x42\x82", b"webm"))
    info = ebml_element(
        b"\x15\x49\xa9\x66",
        ebml_element(b"\x2a\xd7\xb1", (1_000_000).to_bytes(3, "big"))
        + ebml_element(b"\x44\x89", struct.pack(">d", seconds * 1000)),
    )
    video = ebml_element(
        b"\xae",
        ebml_element(b"\x83", b"\x01")
        + ebml_element(b"\x86", codec)
        + ebml_element(
            b"\xe0", ebml_element(b"\xb0", w.to_bytes(2, "big")) + ebml_element(b"\xba", h.to_bytes(2, "big"))
        ),
    )
    sound = ebml_element(b"\xae", ebml_element(b"\x83", b"\x02") + ebml_element(b"\x86", b"A_OPUS")) if audio else b""
    tracks = ebml_element(b"\x16\x54\xae\x6b", video + sound)
    return header + ebml_element(b"\x18\x53\x80\x67", info + tracks)


def test_an_mp4_is_catalogued_from_its_boxes():
    assert videos.catalogue(mp4()) == videos.Catalogue("mp4", 5.0, 320, 240, "avc1", False)
    assert videos.catalogue(mp4(b"qt  ", 1.5, 1280, 720, audio=True)) == videos.Catalogue(
        "mov", 1.5, 1280, 720, "avc1", True
    )


def test_a_webm_is_catalogued_from_its_elements():
    assert videos.catalogue(webm()) == videos.Catalogue("webm", 5.0, 320, 240, "V_VP9", False)
    assert videos.catalogue(webm(12.0, 640, 360, b"V_AV1", audio=True)) == videos.Catalogue(
        "webm", 12.0, 640, 360, "V_AV1", True
    )


def test_bytes_that_are_not_a_video_or_headers_cut_short_are_refused():
    with pytest.raises(videos.NotAVideo, match="not an MP4, QuickTime, WebM or Matroska file"):
        videos.catalogue(b"RIFF\x00\x00\x00\x00AVI ")
    with pytest.raises(videos.NotAVideo, match="not whole"):
        videos.catalogue(mp4()[:40])
    with pytest.raises(videos.NotAVideo, match="not whole"):
        videos.catalogue(b"\x1a\x45\xdf\xa3" + b"\x85" + b"\x42")


def test_a_box_or_an_element_that_runs_past_its_parent_is_a_file_that_is_not_whole():
    with pytest.raises(videos.NotAVideo, match="runs past its parent"):
        videos.catalogue(mp4()[:-1])  # the mdat box declares one more byte than the file has
    with pytest.raises(videos.NotAVideo, match="runs past its parent"):
        videos.catalogue(webm()[:-1])  # PixelHeight declares two bytes, one is left
    with pytest.raises(videos.NotAVideo, match="runs past its parent"):
        videos.catalogue(webm(w=1280, h=720)[:-1])


def test_an_ebml_document_that_is_not_webm_or_matroska_is_refused():
    with pytest.raises(videos.NotAVideo, match="DocType 'xiph' is not webm or matroska"):
        videos.catalogue(webm().replace(b"webm", b"xiph", 1))
    header = ebml_element(b"\x1a\x45\xdf\xa3", ebml_element(b"\x42\x82", b"webm"))
    assert webm().startswith(header)
    matroska = ebml_element(b"\x1a\x45\xdf\xa3", ebml_element(b"\x42\x82", b"matroska")) + webm()[len(header) :]
    assert videos.catalogue(matroska).container == "matroska"


def write_corpus(root, clips: dict[str, bytes], pointer_for: set[str] = frozenset()) -> None:
    (root / "video").mkdir(parents=True)
    (root / "videos" / "clips").mkdir(parents=True)
    (root / "licenses").mkdir()
    (root / "licenses" / "clips-LICENSE").write_text("licence")
    lines = []
    for name, data in clips.items():
        digest = hashlib.sha256(data).hexdigest()
        ext = "webm" if data.startswith(b"\x1a\x45") else "mp4"
        what = videos.catalogue(data)
        line = {"name": name, "path": f"videos/clips/{name}.{ext}", "bytes": len(data), "sha256": digest}
        line.update(
            container=what.container,
            duration_s=what.duration_s,
            w=what.w,
            h=what.h,
            video_codec=what.video_codec,
            audio=what.audio,
        )
        line.update(notes=name, origin={"dataset": "clips", "source": "made here", "license": "Apache-2.0"})
        lines.append(json.dumps(line) + "\n")
        path = root / line["path"]
        if name in pointer_for:
            path.write_text(LFS_POINTER.format(digest, len(data)))
        else:
            path.write_bytes(data)
    (root / "video" / "clips.jsonl").write_text("".join(lines))


def test_a_corpus_of_content_or_pointers_passes_and_the_command_counts_it(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    write_corpus(corpus, {"a": mp4(), "b": webm(), "c": mp4(audio=True)}, pointer_for={"b"})
    assert main(["videos", "check", "--corpus", str(corpus)]) == 0
    total = len(mp4()) + len(webm()) + len(mp4(audio=True))
    assert capsys.readouterr().out.splitlines() == [f"{corpus}: 3 clips in 1 sets, {total} bytes"]


def test_every_kind_of_disagreement_is_named(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    write_corpus(corpus, {"a": mp4(), "b": webm(), "c": mp4(seconds=2.0)})
    (corpus / "videos" / "clips" / "a.mp4").write_bytes(mp4(seconds=9.0))  # other bytes than its line's
    (corpus / "videos" / "clips" / "stray.mp4").write_bytes(mp4())  # a file no line names
    (corpus / "videos" / "other").mkdir()  # a directory with no set
    (corpus / "licenses" / "clips-LICENSE").unlink()
    text = (corpus / "video" / "clips.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["duration_s"] = 3.0  # a line whose catalogue is not the file's
    text[2] = json.dumps(tampered)
    (corpus / "video" / "clips.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["videos", "check", "--corpus", str(corpus)]) == 1
    err = capsys.readouterr().err
    assert "videos/clips/a.mp4: sha256" in err and "the set says" in err
    assert "clips.jsonl:3: the headers say" in err and "'duration_s': 2.0" in err and "'duration_s': 3.0" in err
    assert "videos/clips/stray.mp4: no line of clips.jsonl names it" in err
    assert "videos/other: no set video/other.jsonl names it" in err
    assert "licenses/clips-LICENSE: the licence copy of clips.jsonl is missing" in err
    assert err.rstrip().endswith("bellwether videos check: 5 problem(s)")


def test_a_lines_fields_a_nested_file_and_a_path_outside_the_dataset_are_named_on_pointers_alone(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    write_corpus(corpus, {"a": mp4(), "b": webm(), "c": mp4(audio=True)}, pointer_for={"a", "b", "c"})
    text = (corpus / "video" / "clips.jsonl").read_text().splitlines()
    first = json.loads(text[0])
    del first["audio"]  # a field missing from a line, which the pointer check cannot read from the file
    second = json.loads(text[1])
    second["w"] = "320"  # a field of another type
    third = json.loads(text[2])
    third["path"] = "videos/clips/../clips/c.mp4"  # not a file directly under the dataset directory
    text[:3] = [json.dumps(first), json.dumps(second), json.dumps(third)]
    (corpus / "video" / "clips.jsonl").write_text("".join(line + "\n" for line in text))
    (corpus / "videos" / "clips" / "nested").mkdir()
    (corpus / "videos" / "clips" / "nested" / "extra.mp4").write_bytes(mp4())  # unnamed, one directory down
    assert main(["videos", "check", "--corpus", str(corpus)]) == 1
    err = capsys.readouterr().err
    assert "clips.jsonl:1: the fields are [" in err and "a line carries [" in err
    assert "clips.jsonl:2: w is '320', not int or None" in err
    assert "clips.jsonl:3: path videos/clips/../clips/c.mp4 is not a file directly under videos/clips/" in err
    assert "videos/clips/nested/extra.mp4: no line of clips.jsonl names it" in err
    assert "videos/clips/c.mp4: no line of clips.jsonl names it" in err  # its line's path was not it
    assert err.rstrip().endswith("bellwether videos check: 5 problem(s)")


def test_an_empty_corpus_passes_and_videos_without_a_subcommand_prints_help_and_does_nothing(tmp_path, capsys):
    assert main(["videos", "check", "--corpus", str(tmp_path / "none")]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{tmp_path / 'none'}: 0 clips in 0 sets, 0 bytes"]
    assert main(["videos"]) == NOT_IMPLEMENTED
    assert "check" in capsys.readouterr().out
