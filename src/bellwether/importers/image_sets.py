"""Image sets: the image corpus as every image importer writes it, and the check that it is what a fresh import writes.

An image set is ``corpus/image/<dataset>.jsonl``, one JSON line per image in the order the importer read them, each
ending in "\\n", with the image's bytes beside the set as ``corpus/images/<dataset>/<name>.<ext>``, exactly as the
dataset ships them, kept in Git LFS. A line names the file (``name``, a slug that is the last part of a fixture id,
and ``path`` from the corpus root), its size and sha256, what the file's own headers say it is (``catalogue``), a note,
and ``origin``: the dataset, its source pinned by revision, the sha256 of the file the bytes were read from, the row
and column, and the licence as the source states it (``docs/image-sets.md``).

The catalogue comes from the headers, not from a decoder: the container format from the magic bytes, the width and
height as stored, the colour representation, the bit depth, whether the file carries transparency, its EXIF orientation
where it has one, progressive JPEG and its chroma subsampling, and animation. What the engine's loader makes of the
file, the decoded size after orientation among it, is the reference's to say. So the importers stay within the
standard library and pyarrow, which reads the parquet files as data.

``write`` and ``check`` take the images one at a time, as ``read`` of a dataset gives them, and hold no image's bytes
past its line: an import hashes and catalogues each image, writes its file, and lets it go. A row whose bytes equal an
earlier image's of the dataset gives no line, and is named with the image it repeats, as the text imports name their
repeats. ``check`` compares a fresh import with the committed set line for line, and each committed file with its line
through ``identity``: a file Git LFS has not fetched is a pointer, which names its content's sha256 and size, so a
clone of pointers is checked without a download; a fetched file is hashed.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from bellwether import jsonl, storage

KIND = "image"
IMAGES = "images"
FORMATS = {"jpeg": "jpg", "png": "png", "webp": "webp", "gif": "gif"}
# The columns of a line, in the order they are written.
FIELDS = (
    "name",
    "path",
    "bytes",
    "sha256",
    "format",
    "w",
    "h",
    "color",
    "bits",
    "transparency",
    "orientation",
    "progressive",
    "subsampling",
    "animated",
    "notes",
    "origin",
)
_NAME = re.compile(r"^[a-z0-9-]+$")
_POINTER = re.compile(rb"\Aversion https://git-lfs\.github\.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n\Z")
_EXIF_ORIENTATION = 0x0112


class NotAnImage(ValueError):
    """Bytes the catalogue does not read as an image: another format, or a file whose headers are not whole."""


@dataclass(frozen=True)
class Catalogue:
    """What a file's own headers say about an image."""

    format: str
    w: int
    h: int
    color: str
    bits: int
    transparency: bool
    orientation: int | None
    progressive: bool
    subsampling: str | None
    animated: bool


@dataclass(frozen=True)
class Found:
    """One image as a dataset reader gives it: its bytes, where they came from, and the note its line carries."""

    data: bytes
    origin: dict
    notes: str


def format_of(data: bytes) -> str | None:
    """The container format the magic bytes name; None for anything else."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def catalogue(data: bytes) -> Catalogue:
    """The catalogue of ``data``; ``NotAnImage`` for a format the catalogue does not read or headers that are not whole.

    Each reader walks the container's own structure, as the format's specification lays it out, and reads nothing it
    does not need: a JPEG's markers up to the start of its scan, a PNG's chunks, a WebP's RIFF chunks, a GIF's blocks.
    """
    try:
        if data[:3] == b"\xff\xd8\xff":
            return _jpeg(data)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return _png(data)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return _webp(data)
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return _gif(data)
    except (struct.error, IndexError, ValueError) as err:
        raise NotAnImage(f"the file's headers are not whole: {err}") from err
    raise NotAnImage("not a JPEG, PNG, WebP or GIF file")


def _exif_orientation(exif: bytes) -> int | None:
    """The orientation in a TIFF-structured EXIF block (the bytes after JPEG's ``Exif\\0\\0`` and WebP's ``EXIF`` chunk
    header, the whole of PNG's ``eXIf`` chunk): tag 0x0112 of the first IFD, 1 to 8; None when the block has none."""
    if len(exif) < 8:
        return None
    if exif[:2] == b"II":
        order = "<"
    elif exif[:2] == b"MM":
        order = ">"
    else:
        return None
    (magic, first) = struct.unpack(order + "HI", exif[2:8])
    if magic != 42 or first + 2 > len(exif):
        return None
    (count,) = struct.unpack(order + "H", exif[first : first + 2])
    for index in range(count):
        at = first + 2 + 12 * index
        if at + 12 > len(exif):
            return None
        tag, kind, number = struct.unpack(order + "HHI", exif[at : at + 8])
        if tag == _EXIF_ORIENTATION and kind in (3, 4) and number == 1:
            width = 2 if kind == 3 else 4
            (value,) = struct.unpack(order + ("H" if kind == 3 else "I"), exif[at + 8 : at + 8 + width])
            return value if 1 <= value <= 8 else None
    return None


def _jpeg(data: bytes) -> Catalogue:
    orientation = None
    adobe_transform = None
    frame = None
    at = 2
    while at + 4 <= len(data) and data[at] == 0xFF:
        marker = data[at + 1]
        if marker == 0xFF:
            at += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            at += 2
            continue
        if marker in (0xD9, 0xDA):
            break
        (length,) = struct.unpack(">H", data[at + 2 : at + 4])
        segment = data[at + 4 : at + 2 + length]
        if marker == 0xE1 and segment[:6] == b"Exif\x00\x00":
            orientation = _exif_orientation(segment[6:])
        elif marker == 0xEE and segment[:5] == b"Adobe" and len(segment) >= 12:
            adobe_transform = segment[11]
        elif marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF) and frame is None:
            bits, h, w, components = struct.unpack(">BHHB", segment[:6])
            factors = [(segment[7 + 3 * k] >> 4, segment[7 + 3 * k] & 15) for k in range(components)]
            frame = (marker, bits, w, h, components, factors)
        at += 2 + length
    if frame is None:
        raise ValueError("no frame header before the scan")
    marker, bits, w, h, components, factors = frame
    if components == 1:
        color, subsampling = "gray", "gray"
    else:
        color = (
            ("ycck" if adobe_transform == 2 else "cmyk")
            if components == 4
            else ("rgb" if adobe_transform == 0 else "ycbcr")
            if components == 3
            else f"{components} components"
        )
        subsampling = _subsampling(factors)
    return Catalogue(
        format="jpeg",
        w=w,
        h=h,
        color=color,
        bits=bits,
        transparency=False,
        orientation=orientation,
        progressive=marker in (0xC2, 0xC6, 0xCA, 0xCE),
        subsampling=subsampling,
        animated=False,
    )


def _subsampling(factors: list[tuple[int, int]]) -> str:
    """JPEG chroma subsampling from the components' sampling factors: the first component's against the others'."""
    (h0, v0), rest = factors[0], {factor for factor in factors[1:]}
    hmax, vmax = max(h for h, _ in factors), max(v for _, v in factors)
    if (h0, v0) == (hmax, vmax) and rest <= {(hmax, vmax)}:
        return "4:4:4"
    if (h0, v0) == (2, 1) and rest <= {(1, 1)}:
        return "4:2:2"
    if (h0, v0) == (2, 2) and rest <= {(1, 1)}:
        return "4:2:0"
    if (h0, v0) == (1, 2) and rest <= {(1, 1)}:
        return "4:4:0"
    return "other:" + ",".join(f"{h}x{v}" for h, v in factors)


_PNG_COLOR = {0: "gray", 2: "rgb", 3: "palette", 4: "gray-alpha", 6: "rgb-alpha"}


def _png(data: bytes) -> Catalogue:
    header = None
    trns = animated = False
    orientation = None
    at = 8
    while at + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[at : at + 8])
        chunk = data[at + 8 : at + 8 + length]
        if kind == b"IHDR":
            w, h, depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk[:13])
            header = (w, h, depth, color_type, interlace)
        elif kind == b"tRNS":
            trns = True
        elif kind == b"eXIf":
            orientation = _exif_orientation(chunk)
        elif kind == b"acTL":
            animated = True
        elif kind == b"IEND":
            break
        at += 12 + length
    if header is None:
        raise ValueError("no IHDR chunk")
    w, h, depth, color_type, _ = header
    if color_type not in _PNG_COLOR:
        raise ValueError(f"colour type {color_type}")
    return Catalogue(
        format="png",
        w=w,
        h=h,
        color=_PNG_COLOR[color_type],
        bits=depth,
        transparency=color_type in (4, 6) or trns,
        orientation=orientation,
        progressive=False,
        subsampling=None,
        animated=animated,
    )


def _webp(data: bytes) -> Catalogue:
    size = None
    alpha = animated = False
    orientation = None
    at = 12
    while at + 8 <= len(data):
        kind = data[at : at + 4]
        (length,) = struct.unpack("<I", data[at + 4 : at + 8])
        chunk = data[at + 8 : at + 8 + length]
        if kind == b"VP8X":
            flags = chunk[0]
            alpha = alpha or bool(flags & 0x10)
            animated = animated or bool(flags & 0x02)
            w = 1 + int.from_bytes(chunk[4:7], "little")
            h = 1 + int.from_bytes(chunk[7:10], "little")
            size = (w, h)
        elif kind == b"VP8 " and size is None:
            if chunk[3:6] != b"\x9d\x01\x2a":
                raise ValueError("no VP8 start code")
            w, h = struct.unpack("<HH", chunk[6:10])
            size = (w & 0x3FFF, h & 0x3FFF)
        elif kind == b"VP8L":
            if chunk[0] != 0x2F:
                raise ValueError("no VP8L signature")
            (bits,) = struct.unpack("<I", chunk[1:5])
            alpha = alpha or bool((bits >> 28) & 1)
            if size is None:
                size = (1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF))
        elif kind == b"ALPH":
            alpha = True
        elif kind == b"ANIM":
            animated = True
        elif kind == b"EXIF":
            orientation = _exif_orientation(chunk)
        at += 8 + length + (length & 1)
    if size is None:
        raise ValueError("no VP8, VP8L or VP8X chunk")
    return Catalogue(
        format="webp",
        w=size[0],
        h=size[1],
        color="rgb-alpha" if alpha else "rgb",
        bits=8,
        transparency=alpha,
        orientation=orientation,
        progressive=False,
        subsampling=None,
        animated=animated,
    )


def _gif_blocks(data: bytes, at: int) -> int:
    """Skip a GIF's data sub-blocks from ``at``; the offset after the terminator."""
    while True:
        size = data[at]
        at += 1
        if size == 0:
            return at
        at += size


def _gif(data: bytes) -> Catalogue:
    w, h, packed = struct.unpack("<HHB", data[6:11])
    at = 13
    if packed & 0x80:
        at += 3 * (2 << (packed & 7))
    frames = 0
    transparent = False
    while at < len(data):
        block = data[at]
        if block == 0x3B:
            break
        if block == 0x2C:
            frames += 1
            local = data[at + 9]
            at += 10
            if local & 0x80:
                at += 3 * (2 << (local & 7))
            at = _gif_blocks(data, at + 1)
        elif block == 0x21:
            label = data[at + 1]
            if label == 0xF9 and data[at + 2] >= 4:
                transparent = transparent or bool(data[at + 3] & 1)
            at = _gif_blocks(data, at + 2)
        else:
            raise ValueError(f"block {block:#x}")
    return Catalogue(
        format="gif",
        w=w,
        h=h,
        color="palette",
        bits=8,
        transparency=transparent,
        orientation=None,
        progressive=False,
        subsampling=None,
        animated=frames > 1,
    )


def parquet_images(path: Path, columns: Iterable[str]) -> Iterator[tuple[int, str, bytes]]:
    """``(row, column, bytes)`` for each image cell of a parquet file, in row order and then in the order of
    ``columns``; ``row`` counts from 0 over the whole file. A cell is the struct the datasets library stores an image
    as, ``{"bytes", "path"}``; a null cell, or one with no bytes, gives nothing. The file is read in batches, so no
    more than a batch of images is held at once; nothing in it runs."""
    import pyarrow.parquet as pq

    file = pq.ParquetFile(path)
    names = set(file.schema_arrow.names)
    wanted = [column for column in columns if column in names]
    row = 0
    for batch in file.iter_batches(batch_size=64, columns=wanted):
        cells = batch.to_pydict()
        for index in range(batch.num_rows):
            for column in wanted:
                value = cells[column][index]
                if isinstance(value, dict) and value.get("bytes"):
                    yield row + index, column, value["bytes"]
        row += batch.num_rows


def zip_images(path: Path, suffix: str = ".jpg") -> Iterator[tuple[str, bytes]]:
    """``(member, bytes)`` for each file of a zip archive whose name ends in ``suffix``, in name order, read as data."""
    with zipfile.ZipFile(path) as archive:
        members = sorted(m.filename for m in archive.infolist() if not m.is_dir() and m.filename.endswith(suffix))
        for member in members:
            yield member, archive.read(member)


def set_path(corpus_dir: Path, dataset: str) -> Path:
    return corpus_dir / KIND / f"{dataset}.jsonl"


def images_dir(corpus_dir: Path, dataset: str) -> Path:
    return corpus_dir / IMAGES / dataset


def _line(dataset: str, index: int, found: Found) -> tuple[dict, str] | None:
    """A found image's line and the reason it gets none: the name, the path, the size and sha256, the catalogue, the
    note and the origin, in ``FIELDS`` order. ``index`` is the image's position among the dataset's kept images."""
    try:
        what = catalogue(found.data)
    except NotAnImage as err:
        return None, str(err)
    digest = hashlib.sha256(found.data).hexdigest()
    name = f"{dataset}-{index:05d}-{digest[:8]}"
    line = {
        "name": name,
        "path": f"{IMAGES}/{dataset}/{name}.{FORMATS[what.format]}",
        "bytes": len(found.data),
        "sha256": digest,
        **asdict(what),
        "notes": found.notes,
        "origin": found.origin,
    }
    return line, ""


def _json(line: dict) -> str:
    return json.dumps(line, ensure_ascii=False)


@dataclass
class Summary:
    """What an import kept and left out, for ``report``: the lines written, the rows left out as repeats with the
    image they repeat, the rows that gave no line with their reason, and the bytes of the kept images."""

    lines: int = 0
    bytes: int = 0
    repeats: list[tuple[str, str]] | None = None
    skipped: list[tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        self.repeats = [] if self.repeats is None else self.repeats
        self.skipped = [] if self.skipped is None else self.skipped


def _lines(dataset: str, found: Iterable[Found], summary: Summary) -> Iterator[tuple[dict, bytes]]:
    """Each kept image's line with its bytes, in order; repeats and unreadable rows go into ``summary``."""
    seen: dict[str, str] = {}
    index = 0
    for image in found:
        where = image.notes
        digest = hashlib.sha256(image.data).hexdigest()
        if digest in seen:
            summary.repeats.append((where, seen[digest]))
            continue
        line, why = _line(dataset, index + 1, image)
        if line is None:
            summary.skipped.append((where, why))
            continue
        seen[digest] = line["name"]
        index += 1
        summary.lines += 1
        summary.bytes += line["bytes"]
        yield line, image.data


def identity(path: Path) -> tuple[str, int]:
    """The sha256 and size of the content at ``path``: read from its pointer when Git LFS has not fetched it, so a
    clone that holds pointers alone is checked without a download; a fetched file is hashed."""
    with path.open("rb") as handle:
        head = handle.read(len(storage.LFS_POINTER_PREFIX))
        if head == storage.LFS_POINTER_PREFIX:
            match = _POINTER.match(head + handle.read(512))
            if match is None:
                raise ValueError(f"{path} begins like a Git LFS pointer but is not one")
            return match.group(1).decode(), int(match.group(2))
        digest = hashlib.sha256(head)
        size = len(head)
        while piece := handle.read(storage.READ_SIZE):
            digest.update(piece)
            size += len(piece)
    return digest.hexdigest(), size


def _holds(path: Path, data: bytes) -> bool:
    """Whether the file at ``path`` already holds ``data``, as content or as its Git LFS pointer."""
    try:
        return identity(path) == (hashlib.sha256(data).hexdigest(), len(data))
    except (OSError, ValueError):
        return False


def write(
    dataset: str, found: Iterable[Found], corpus_dir: Path, files: dict[str, bytes] | None = None
) -> tuple[list[Path], Summary]:
    """Write the dataset's images and its set, then ``files`` (its licence copy), and remove what the import no longer
    writes under its images directory and its set file.

    Each image's file is written whole or not at all (``storage.write_whole``) as its line is built, and a file that
    already holds the image, as content or as the pointer Git LFS leaves, is left as it is, so a re-import touches
    nothing git would see as a change. The set file is written last, from the lines in order. Nothing is removed until
    everything is written, so an import cut short leaves the old files where it had not reached and the set file as it
    was. Returns the paths written and the ``Summary`` that ``report`` prints.
    """
    summary = Summary()
    directory = images_dir(corpus_dir, dataset)
    written: list[Path] = []
    lines: list[str] = []
    for line, data in _lines(dataset, found, summary):
        path = corpus_dir / line["path"]
        if not _holds(path, data):
            storage.write_whole(path, data)
        written.append(path)
        lines.append(_json(line) + "\n")
    target = set_path(corpus_dir, dataset)
    storage.write_whole(target, "".join(lines).encode("utf-8"))
    written.append(target)
    for relative, content in sorted((files or {}).items()):
        path = corpus_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written.append(path)
    kept = set(written)
    if directory.is_dir():
        for stale in sorted(p for p in directory.iterdir() if p.is_file() and p not in kept):
            stale.unlink()
    return written, summary


def read_set(path: Path) -> list[dict]:
    """The lines of a committed image set, each checked to be an object with a slug ``name`` and a ``path``."""
    lines: list[dict] = []
    for number, line in jsonl.loads(path.read_text(encoding="utf-8"), path):
        if not isinstance(line, dict) or not isinstance(line.get("name"), str) or not _NAME.match(line["name"]):
            raise ValueError(f"{path}:{number}: a line is an object whose `name` is a lowercase slug")
        if not isinstance(line.get("path"), str) or not isinstance(line.get("sha256"), str):
            raise ValueError(f"{path}:{number}: a line names its `path` and `sha256`")
        lines.append(line)
    return lines


def check(
    dataset: str, found: Iterable[Found], corpus_dir: Path, files: dict[str, bytes] | None = None
) -> tuple[list[str], Summary]:
    """One line per difference between a fresh import and the committed set, its files and ``files``; empty when
    there is none.

    The fresh lines are compared with the committed set's as they come, by their JSON, so a line that differs is named
    by its number; the committed file of each fresh image is compared through ``identity``, pointer or content, with
    the image's sha256 and size; a file under the images directory that no fresh line names is named, as is a set file
    or licence copy that is missing or differs.
    """
    summary = Summary()
    target = set_path(corpus_dir, dataset)
    problems: list[str] = []
    committed: list[str] = []
    if target.is_file():
        committed = [piece for piece in target.read_text(encoding="utf-8").split("\n")]
        committed = [f"{piece}\n" for piece in committed[:-1]] if committed and committed[-1] == "" else committed
    else:
        problems.append(f"{target}: missing")
    expected: set[Path] = set()
    number = 0
    for line, _data in _lines(dataset, found, summary):
        number += 1
        text = _json(line) + "\n"
        if target.is_file() and (number > len(committed) or committed[number - 1] != text):
            problems.append(f"{target}:{number}: differs from a fresh import")
        path = corpus_dir / line["path"]
        expected.add(path)
        if not path.is_file():
            problems.append(f"{path}: missing")
        else:
            try:
                digest, size = identity(path)
            except (OSError, ValueError) as err:
                problems.append(str(err))
                continue
            if (digest, size) != (line["sha256"], line["bytes"]):
                problems.append(f"{path}: holds sha256 {digest}, {size} bytes; a fresh import gives {line['sha256']}")
    if target.is_file() and len(committed) > number:
        problems.append(f"{target}: {len(committed) - number} line(s) past the {number} a fresh import writes")
    directory = images_dir(corpus_dir, dataset)
    if directory.is_dir():
        for path in sorted(p for p in directory.iterdir() if p.is_file() and p not in expected):
            problems.append(f"{path}: no row of the import writes it")
    for relative, content in sorted((files or {}).items()):
        path = corpus_dir / relative
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif path.read_bytes() != content:
            problems.append(f"{path}: differs from a fresh import")
    return problems, summary


def report(dataset: str, summary: Summary, corpus_dir: Path) -> None:
    """Print what an import left out and what it kept, in the words of the text imports."""
    for where, first in summary.repeats:
        print(f"no line for {where}: it repeats {first}")
    for where, why in summary.skipped:
        print(f"no line for {where}: {why}")
    print(
        f"{set_path(corpus_dir, dataset)}: {summary.lines} images, {summary.bytes} bytes, "
        f"{len(summary.repeats)} left out as repeats, {len(summary.skipped)} rows with no image"
    )
