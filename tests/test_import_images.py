import hashlib
import io
import json
import struct
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import exif, gif, jpeg, png, riff, vp8, vp8l, vp8x

from bellwether.importers import image_sets as images


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_format_of_names_the_four_formats_and_nothing_else():
    assert images.format_of(jpeg()) == "jpeg"
    assert images.format_of(png()) == "png"
    assert images.format_of(riff([(b"VP8 ", vp8(4, 4))])) == "webp"
    assert images.format_of(gif()) == "gif"
    assert images.format_of(b"BM" + bytes(40)) is None
    assert images.format_of(b"") is None


def test_a_baseline_jpeg_is_catalogued_from_its_frame_header():
    assert images.catalogue(jpeg(640, 480)) == images.Catalogue(
        format="jpeg",
        w=640,
        h=480,
        color="ycbcr",
        bits=8,
        transparency=False,
        orientation=None,
        progressive=False,
        subsampling="4:2:0",
        animated=False,
    )


@pytest.mark.parametrize(
    "factors, components, progressive, color, subsampling",
    [
        (((1, 1), (1, 1), (1, 1)), 3, True, "ycbcr", "4:4:4"),
        (((2, 1), (1, 1), (1, 1)), 3, False, "ycbcr", "4:2:2"),
        (((1, 2), (1, 1), (1, 1)), 3, False, "ycbcr", "4:4:0"),
        (((1, 1),), 1, False, "gray", "gray"),
        (((2, 1), (2, 1), (1, 1)), 3, False, "ycbcr", "other:2x1,2x1,1x1"),
    ],
)
def test_a_jpegs_subsampling_and_progressive_flag_come_from_its_frame(
    factors, components, progressive, color, subsampling
):
    found = images.catalogue(jpeg(components=components, factors=factors, progressive=progressive))
    assert (found.color, found.subsampling, found.progressive) == (color, subsampling, progressive)


@pytest.mark.parametrize(
    "components, transform, color",
    [(4, 2, "ycck"), (4, 1, "cmyk"), (4, None, "cmyk"), (3, 0, "rgb"), (3, 1, "ycbcr")],
)
def test_a_cmyk_ycck_or_rgb_jpeg_is_told_by_its_adobe_marker(components, transform, color):
    factors = ((1, 1),) * components
    assert images.catalogue(jpeg(components=components, factors=factors, adobe_transform=transform)).color == color


@pytest.mark.parametrize("order", ["<", ">"])
def test_the_exif_orientation_is_read_in_both_byte_orders(order):
    assert images.catalogue(jpeg(orientation=6, exif_order=order)).orientation == 6
    assert images.catalogue(jpeg(orientation=1, exif_order=order)).orientation == 1


def test_an_orientation_outside_one_to_eight_or_an_exif_block_without_one_gives_none():
    assert images.catalogue(jpeg(orientation=9)).orientation is None
    assert images.catalogue(jpeg(orientation=0)).orientation is None
    no_entry = b"\xff\xd8" + bytes([0xFF, 0xE1]) + struct.pack(">H", 2 + 6 + len(exif(None))) + b"Exif\x00\x00"
    assert images.catalogue(no_entry + exif(None) + jpeg()[2:]).orientation is None


def test_a_png_is_catalogued_from_its_chunks():
    assert images.catalogue(png(640, 480)) == images.Catalogue(
        format="png",
        w=640,
        h=480,
        color="rgb",
        bits=8,
        transparency=False,
        orientation=None,
        progressive=False,
        subsampling=None,
        animated=False,
    )
    gray16 = images.catalogue(png(color_type=0, depth=16))
    assert (gray16.color, gray16.bits, gray16.transparency) == ("gray", 16, False)
    palette = images.catalogue(png(color_type=3, trns=True))
    assert (palette.color, palette.transparency) == ("palette", True)
    rgba = images.catalogue(png(color_type=6, orientation=6, animated=True))
    assert (rgba.color, rgba.transparency, rgba.orientation, rgba.animated) == ("rgb-alpha", True, 6, True)
    assert images.catalogue(png(color_type=4)).color == "gray-alpha"


def test_a_webp_is_catalogued_from_its_chunks():
    lossy = images.catalogue(riff([(b"VP8 ", vp8(320, 200))]))
    assert (lossy.format, lossy.w, lossy.h, lossy.color, lossy.transparency, lossy.animated) == (
        "webp",
        320,
        200,
        "rgb",
        False,
        False,
    )
    lossless = images.catalogue(riff([(b"VP8L", vp8l(17, 9, alpha=True))]))
    assert (lossless.w, lossless.h, lossless.color, lossless.transparency) == (17, 9, "rgb-alpha", True)
    extended = images.catalogue(
        riff([(b"VP8X", vp8x(100, 50, alpha=True, animated=True, exif_block=True)), (b"EXIF", exif(3, ">"))])
    )
    assert (extended.w, extended.h, extended.color, extended.animated, extended.orientation) == (
        100,
        50,
        "rgb-alpha",
        True,
        3,
    )
    with_alpha_chunk = images.catalogue(riff([(b"VP8X", vp8x(8, 8)), (b"ALPH", b"\x00"), (b"VP8 ", vp8(8, 8))]))
    assert with_alpha_chunk.transparency is True


def test_a_gif_is_catalogued_from_its_blocks():
    still = images.catalogue(gif(12, 7))
    assert (still.format, still.w, still.h, still.color, still.transparency, still.animated) == (
        "gif",
        12,
        7,
        "palette",
        False,
        False,
    )
    moving = images.catalogue(gif(frames=3, transparent=True))
    assert (moving.animated, moving.transparency) == (True, True)


@pytest.mark.parametrize(
    "data, why",
    [
        (b"hello, not an image", "not a JPEG, PNG, WebP or GIF file"),
        (jpeg(frame=False), "the file's headers are not whole: no frame header before the scan"),
        (png(end=False)[:20], "the file's headers are not whole"),
        (riff([]), "the file's headers are not whole: no VP8, VP8L or VP8X chunk"),
    ],
)
def test_bytes_that_are_not_an_image_or_headers_cut_short_are_refused(data, why):
    with pytest.raises(images.NotAnImage, match=why):
        images.catalogue(data)


def image_table(rows: list[list[bytes | None | str]], columns: list[str]) -> pa.Table:
    """A parquet table as the datasets library stores images: a struct of bytes and path per cell."""
    kind = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    data = {}
    for index, column in enumerate(columns):
        cells = []
        for row in rows:
            value = row[index]
            if value is None:
                cells.append(None)
            elif isinstance(value, str):
                cells.append({"bytes": None, "path": value})
            else:
                cells.append({"bytes": value, "path": None})
        data[column] = pa.array(cells, type=kind)
    return pa.table(data)


def test_parquet_images_yields_rows_across_batches_with_their_column_and_skips_empty_cells(tmp_path):
    rows = [[png(w=1 + n, h=1), jpeg(w=1 + n, h=1) if n % 2 else None] for n in range(70)]
    rows[3] = [None, "only/a/path.png"]
    pq.write_table(image_table(rows, ["image_1", "image_2"]), tmp_path / "t.parquet")
    found = list(images.parquet_images(tmp_path / "t.parquet", ["image_1", "image_2", "image_9"]))
    assert len(found) == 69 + 34  # image_1 of every row but 3; image_2 of the odd rows but 3, whose cell is a path
    assert found[0] == (0, "image_1", png(w=1, h=1))
    assert found[1] == (1, "image_1", png(w=2, h=1)) and found[2] == (1, "image_2", jpeg(w=2, h=1))
    assert [row for row, _, _ in found if row == 3] == []
    assert found[-1] == (69, "image_2", jpeg(w=70, h=1))
    assert [row for row, column, _ in found if column == "image_1"] == [n for n in range(70) if n != 3]


def test_zip_images_reads_the_named_members_in_name_order(tmp_path):
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as archive:
        archive.writestr("val/b.jpg", jpeg(w=2, h=2))
        archive.writestr("val/a.jpg", jpeg(w=1, h=1))
        archive.writestr("val/notes.txt", "not an image")
        archive.writestr("val/dir/", "")
    assert list(images.zip_images(tmp_path / "a.zip")) == [("val/a.jpg", jpeg(w=1, h=1)), ("val/b.jpg", jpeg(w=2, h=2))]


ORIGIN = {"dataset": "ds", "source": "hf:datasets/org/ds@" + "a" * 40, "sha256": "f" * 64, "file": "data/t.parquet"}


def found(data: bytes, row: int) -> images.Found:
    return images.Found(data, {**ORIGIN, "row": row, "column": "image", "license": "Apache-2.0"}, f"ds test row {row}")


def test_write_writes_the_set_the_files_and_the_licence_copy_and_names_repeats(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    first, second = png(3, 2), jpeg(5, 4)
    written, summary = images.write(
        "ds",
        [found(first, 0), found(b"not an image", 1), found(second, 2), found(first, 3)],
        corpus,
        {"licenses/ds-LICENSE": b"the licence\n"},
    )
    lines = images.read_set(corpus / "image" / "ds.jsonl")
    assert [line["name"] for line in lines] == [f"ds-00001-{sha(first)[:8]}", f"ds-00002-{sha(second)[:8]}"]
    assert list(lines[0]) == list(images.FIELDS)
    assert lines[0]["path"] == f"images/ds/ds-00001-{sha(first)[:8]}.png"
    assert (lines[0]["bytes"], lines[0]["sha256"], lines[0]["format"], lines[0]["w"], lines[0]["h"]) == (
        len(first),
        sha(first),
        "png",
        3,
        2,
    )
    assert lines[0]["notes"] == "ds test row 0" and lines[0]["origin"]["row"] == 0
    assert lines[1]["path"].endswith(".jpg") and lines[1]["origin"]["row"] == 2
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "ds-LICENSE").read_bytes() == b"the licence\n"
    assert written == [
        corpus / lines[0]["path"],
        corpus / lines[1]["path"],
        corpus / "image" / "ds.jsonl",
        corpus / "licenses" / "ds-LICENSE",
    ]
    assert (summary.lines, summary.bytes, summary.repeats, summary.skipped) == (
        2,
        len(first) + len(second),
        [("ds test row 3", f"ds-00001-{sha(first)[:8]}")],
        [("ds test row 1", "not a JPEG, PNG, WebP or GIF file")],
    )
    images.report("ds", summary, corpus)
    assert capsys.readouterr().out.splitlines() == [
        f"no line for ds test row 3: it repeats ds-00001-{sha(first)[:8]}",
        "no line for ds test row 1: not a JPEG, PNG, WebP or GIF file",
        f"{corpus / 'image' / 'ds.jsonl'}: 2 images, {len(first) + len(second)} bytes, 1 left out as repeats, "
        "1 rows with no image",
    ]


def pointer(data: bytes) -> bytes:
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha(data)}\nsize {len(data)}\n".encode()


def test_a_re_import_leaves_an_unchanged_file_or_its_pointer_alone_and_removes_a_stale_one(tmp_path):
    corpus = tmp_path / "corpus"
    first, second = png(3, 2), jpeg(5, 4)
    images.write("ds", [found(first, 0), found(second, 1)], corpus)
    [one, two] = [corpus / line["path"] for line in images.read_set(corpus / "image" / "ds.jsonl")]
    one.write_bytes(pointer(first))  # as a clone of pointers holds it
    stale = corpus / "images" / "ds" / "ds-00009-deadbeef.png"
    stale.write_bytes(b"old")
    before = {path: path.stat().st_mtime_ns for path in (one, two)}
    images.write("ds", [found(first, 0), found(second, 1)], corpus)
    assert one.read_bytes() == pointer(first)
    assert {path: path.stat().st_mtime_ns for path in (one, two)} == before
    assert not stale.exists()


def test_check_passes_on_a_fresh_import_and_names_every_kind_of_difference(tmp_path):
    corpus = tmp_path / "corpus"
    first, second = png(3, 2), jpeg(5, 4)
    files = {"licenses/ds-LICENSE": b"the licence\n"}
    images.write("ds", [found(first, 0), found(second, 1)], corpus, files)
    problems, summary = images.check("ds", [found(first, 0), found(second, 1)], corpus, files)
    assert problems == [] and summary.lines == 2
    set_file = corpus / "image" / "ds.jsonl"
    [one, two] = [corpus / line["path"] for line in images.read_set(set_file)]
    one.write_bytes(pointer(first))  # the pointer passes for the content
    assert images.check("ds", [found(first, 0), found(second, 1)], corpus, files)[0] == []
    two.write_bytes(jpeg(5, 5))
    extra = corpus / "images" / "ds" / "ds-00009-deadbeef.png"
    extra.write_bytes(b"x")
    (corpus / "licenses" / "ds-LICENSE").write_bytes(b"another\n")
    problems, _ = images.check("ds", [found(first, 0), found(second, 1), found(png(9, 9), 2)], corpus, files)
    assert problems == [
        f"{two}: holds sha256 {sha(jpeg(5, 5))}, {len(jpeg(5, 5))} bytes; a fresh import gives {sha(second)}",
        f"{set_file}:3: differs from a fresh import",
        f"{corpus / 'images' / 'ds' / ('ds-00003-' + sha(png(9, 9))[:8] + '.png')}: missing",
        f"{extra}: no row of the import writes it",
        f"{corpus / 'licenses' / 'ds-LICENSE'}: differs from a fresh import",
    ]
    problems, _ = images.check("ds", [found(first, 0)], corpus, files)
    assert f"{set_file}: 1 line(s) past the 1 a fresh import writes" in problems
    problems, _ = images.check("other", [found(first, 0)], corpus)
    assert problems[0] == f"{corpus / 'image' / 'other.jsonl'}: missing"


def test_identity_reads_a_pointer_or_hashes_the_content_and_refuses_a_pointer_that_is_not_one(tmp_path):
    data = png(3, 2)
    (tmp_path / "content").write_bytes(data)
    (tmp_path / "pointer").write_bytes(pointer(data))
    (tmp_path / "broken").write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
    assert images.identity(tmp_path / "content") == (sha(data), len(data))
    assert images.identity(tmp_path / "pointer") == (sha(data), len(data))
    with pytest.raises(ValueError, match="begins like a Git LFS pointer but is not one"):
        images.identity(tmp_path / "broken")


def test_read_set_refuses_a_line_without_a_slug_name_or_a_path(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(json.dumps({"name": "Not A Slug", "path": "images/ds/x.png", "sha256": "0"}) + "\n")
    with pytest.raises(ValueError, match="`name` is a lowercase slug"):
        images.read_set(path)
    path.write_text(json.dumps({"name": "ds-00001-abcdef01"}) + "\n")
    with pytest.raises(ValueError, match="names its `path` and `sha256`"):
        images.read_set(path)


def test_a_line_keeps_non_ascii_notes_raw_as_every_corpus_line_does(tmp_path):
    corpus = tmp_path / "corpus"
    image = images.Found(png(2, 2), {**ORIGIN, "row": 0, "column": "image", "license": "Apache-2.0"}, "Übung, row 0")
    images.write("ds", [image], corpus)
    text = (corpus / "image" / "ds.jsonl").read_text(encoding="utf-8")
    assert '"notes": "Übung, row 0"' in text
    assert io.StringIO(text).readline().endswith("\n")
