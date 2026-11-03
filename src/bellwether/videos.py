"""bellwether videos check: the video sets and the files in the checkout agree, with no Git LFS fetch.

A video set is ``corpus/video/<dataset>.jsonl``, one JSON line per clip, with the clip's bytes beside it as
``corpus/videos/<dataset>/<name>.<ext>``, exactly as made or shipped, in Git LFS. A line names the file (``name``, a
slug, and ``path`` from the corpus root), its size and sha256, what the container's own headers say it holds
(``catalogue``: the container, the duration in seconds, the picture's width and height, the video codec and whether
there is an audio track), a note, and ``origin`` (the dataset, its source, and the licence), as an image line does.

The catalogue comes from the headers, not from a decoder: an MP4 or QuickTime file's boxes (``ftyp`` for the brand,
``mvhd`` for the duration, the video track's ``tkhd`` for the size, ``hdlr`` for the track kinds, ``stsd`` for the
codec's sample entry) and a WebM or Matroska file's EBML elements (``DocType``, ``Info``'s duration and timecode
scale, each ``TrackEntry``'s type, codec and video size). What the engine's decoder makes of the clip, the frames it
samples among it, is the reference's to say.

``check`` is what every pull request runs: every line of every set carries the fields above with their types and
names a file directly under its dataset's directory whose pointer or content has the line's sha256 and size, no file
under a videos directory is unnamed (nested ones included), names and paths are unique, each set's licence copy is
there, and a fetched file's catalogue is its line's. A container the catalogue does not read, or headers that are not
whole (a box or an element that runs past its parent, a DocType other than ``webm`` or ``matroska``), is refused.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from bellwether.importers import image_sets

KIND = "video"
VIDEOS = "videos"
LICENSES = "licenses"
# The columns of a line, in the order they are written.
FIELDS = (
    "name",
    "path",
    "bytes",
    "sha256",
    "container",
    "duration_s",
    "w",
    "h",
    "video_codec",
    "audio",
    "notes",
    "origin",
)
# What each field holds; None where the headers do not say (an EBML file without a Video element, say).
_TYPES = {
    "name": str,
    "path": str,
    "bytes": int,
    "sha256": str,
    "container": str,
    "duration_s": (int, float, type(None)),
    "w": (int, type(None)),
    "h": (int, type(None)),
    "video_codec": (str, type(None)),
    "audio": bool,
    "notes": str,
    "origin": dict,
}
_DOCTYPES = ("webm", "matroska")


class NotAVideo(ValueError):
    """Bytes the catalogue does not read as a video container, or headers that are not whole."""


@dataclass(frozen=True)
class Catalogue:
    """What a container's own headers say about a clip."""

    container: str
    duration_s: float | None
    w: int | None
    h: int | None
    video_codec: str | None
    audio: bool


def catalogue(data: bytes) -> Catalogue:
    """The catalogue of ``data``; ``NotAVideo`` for a container the catalogue does not read or headers not whole."""
    try:
        if len(data) >= 12 and data[4:8] == b"ftyp":
            return _isobmff(data)
        if data[:4] == b"\x1a\x45\xdf\xa3":
            return _ebml(data)
    except (struct.error, IndexError, ValueError, UnicodeDecodeError) as err:
        raise NotAVideo(f"the file's headers are not whole: {err}") from err
    raise NotAVideo("not an MP4, QuickTime, WebM or Matroska file")


# --- ISO base media (MP4, QuickTime) ---

_CONTAINERS = {b"qt  ": "mov"}
_PARENTS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def _boxes(data: bytes, start: int, end: int):
    """``(kind, body start, box end)`` for each box between ``start`` and ``end``."""
    at = start
    while at + 8 <= end:
        (size,) = struct.unpack(">I", data[at : at + 4])
        kind = data[at + 4 : at + 8]
        head = 8
        if size == 1:
            (size,) = struct.unpack(">Q", data[at + 8 : at + 16])
            head = 16
        elif size == 0:
            size = end - at
        if size < head:
            raise ValueError(f"box {kind!r} of size {size}")
        if at + size > end:
            raise ValueError(f"box {kind!r} of size {size} runs past its parent")
        yield kind, at + head, at + size
        at += size
    if at != end:
        raise ValueError(f"{end - at} byte(s) after the last box, too few for a box header")


def _isobmff(data: bytes) -> Catalogue:
    container = "mp4"
    duration = None
    size = None
    codec = None
    audio = False
    track: dict = {}

    def walk(start: int, end: int, depth: int = 0) -> None:
        nonlocal container, duration, size, codec, audio, track
        for kind, body, box_end in _boxes(data, start, end):
            if kind == b"ftyp":
                container = _CONTAINERS.get(data[body : body + 4], "mp4")
            elif kind == b"mvhd":
                version = data[body]
                if version == 1:
                    timescale, length = struct.unpack(">IQ", data[body + 20 : body + 32])
                else:
                    timescale, length = struct.unpack(">II", data[body + 12 : body + 20])
                duration = length / timescale if timescale else None
            elif kind == b"trak":
                track = {}
                walk(body, box_end, depth + 1)
                if track.get("handler") == b"vide":
                    size = size or track.get("size")
                    codec = codec or track.get("codec")
                elif track.get("handler") == b"soun":
                    audio = True
            elif kind == b"tkhd":
                version = data[body]
                at = (
                    body + 4 + (32 if version == 1 else 20) + 8 + 8 + 36
                )  # flags, times and id, reserved, layer group volume, matrix
                w, h = struct.unpack(">II", data[at : at + 8])
                track["size"] = (w >> 16, h >> 16)
            elif kind == b"hdlr":
                track.setdefault("handler", data[body + 8 : body + 12])
            elif kind == b"stsd":
                (entries,) = struct.unpack(">I", data[body + 4 : body + 8])
                if entries:
                    track["codec"] = data[body + 12 : body + 16].decode("latin1").strip()
            elif kind in _PARENTS:
                walk(body, box_end, depth + 1)

    walk(0, len(data))
    if duration is None and size is None and codec is None:
        raise ValueError("no moov box")
    return Catalogue(container, duration, size[0] if size else None, size[1] if size else None, codec, audio)


# --- EBML (WebM, Matroska) ---

_EBML = 0x1A45DFA3
_DOCTYPE = 0x4282
_SEGMENT = 0x18538067
_INFO = 0x1549A966
_TIMECODE_SCALE = 0x2AD7B1
_DURATION = 0x4489
_TRACKS = 0x1654AE6B
_TRACK_ENTRY = 0xAE
_TRACK_TYPE = 0x83
_CODEC_ID = 0x86
_VIDEO = 0xE0
_PIXEL_WIDTH = 0xB0
_PIXEL_HEIGHT = 0xBA
_MASTERS = {_EBML, _SEGMENT, _INFO, _TRACKS, _TRACK_ENTRY, _VIDEO}


def _vint(data: bytes, at: int, keep_marker: bool) -> tuple[int, int]:
    """An EBML variable-size integer at ``at``: its value and the offset after it. Element ids keep their length
    marker; sizes drop it, and an all-ones size reads as unknown (-1)."""
    first = data[at]
    length = 1
    mask = 0x80
    while length <= 8 and not first & mask:
        length += 1
        mask >>= 1
    if length > 8:
        raise ValueError("a variable-size integer longer than eight bytes")
    raw = data[at : at + length]
    value = int.from_bytes(raw, "big")
    if not keep_marker:
        value &= (1 << (7 * length)) - 1
        if value == (1 << (7 * length)) - 1:
            value = -1
    return value, at + length


def _elements(data: bytes, start: int, end: int):
    at = start
    while at < end:
        element_id, at = _vint(data, at, keep_marker=True)
        length, at = _vint(data, at, keep_marker=False)
        if length >= 0 and at + length > end:
            raise ValueError(f"element {element_id:#x} of size {length} runs past its parent")
        body_end = end if length < 0 else at + length
        yield element_id, at, body_end
        at = body_end


def _float(raw: bytes) -> float:
    return struct.unpack(">f", raw)[0] if len(raw) == 4 else struct.unpack(">d", raw)[0]


def _ebml(data: bytes) -> Catalogue:
    doctype = None
    scale = 1_000_000
    duration = None
    size = None
    codec = None
    audio = False

    def walk(start: int, end: int, parent: int | None = None, track: dict | None = None) -> None:
        nonlocal doctype, scale, duration, size, codec, audio
        for element_id, body, body_end in _elements(data, start, end):
            raw = data[body:body_end]
            if element_id == _DOCTYPE:
                doctype = raw.rstrip(b"\x00").decode("ascii")
            elif element_id == _TIMECODE_SCALE:
                scale = int.from_bytes(raw, "big")
            elif element_id == _DURATION:
                duration = _float(raw)
            elif element_id == _TRACK_ENTRY:
                entry: dict = {}
                walk(body, body_end, element_id, entry)
                if entry.get("type") == 1:
                    size = size or entry.get("size")
                    codec = codec or entry.get("codec")
                elif entry.get("type") == 2:
                    audio = True
            elif element_id == _TRACK_TYPE and track is not None:
                track["type"] = int.from_bytes(raw, "big")
            elif element_id == _CODEC_ID and track is not None:
                track["codec"] = raw.rstrip(b"\x00").decode("ascii")
            elif element_id == _VIDEO and track is not None:
                dims: dict = {}
                for inner_id, inner_body, inner_end in _elements(data, body, body_end):
                    if inner_id == _PIXEL_WIDTH:
                        dims["w"] = int.from_bytes(data[inner_body:inner_end], "big")
                    elif inner_id == _PIXEL_HEIGHT:
                        dims["h"] = int.from_bytes(data[inner_body:inner_end], "big")
                if "w" in dims and "h" in dims:
                    track["size"] = (dims["w"], dims["h"])
            elif element_id in _MASTERS:
                walk(body, body_end, element_id, track)

    walk(0, len(data))
    if doctype is None:
        raise ValueError("no EBML DocType")
    if doctype not in _DOCTYPES:
        raise ValueError(f"EBML DocType {doctype!r} is not webm or matroska")
    seconds = duration * scale / 1_000_000_000 if duration is not None else None
    return Catalogue(doctype, seconds, size[0] if size else None, size[1] if size else None, codec, audio)


# --- the check ---


def _type_name(kind) -> str:
    kinds = kind if isinstance(kind, tuple) else (kind,)
    return " or ".join("None" if k is type(None) else k.__name__ for k in kinds)


def problems(corpus: Path) -> list[str]:
    found: list[str] = []
    sets_dir = corpus / KIND
    sets = sorted(sets_dir.glob("*.jsonl")) if sets_dir.is_dir() else []
    names: dict[str, Path] = {}
    paths: dict[str, Path] = {}
    datasets: set[str] = set()
    for set_file in sets:
        dataset = set_file.name.removesuffix(".jsonl")
        datasets.add(dataset)
        try:
            lines = image_sets.read_set(set_file)
        except ValueError as err:
            found.append(str(err))
            continue
        expected: set[Path] = set()
        for number, line in enumerate(lines, 1):
            where = f"{set_file}:{number}"
            if list(line) != list(FIELDS):
                found.append(f"{where}: the fields are {list(line)}, a line carries {list(FIELDS)}")
            for field, kind in _TYPES.items():
                value = line.get(field)
                if field in line and (not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool)):
                    found.append(f"{where}: {field} is {value!r}, not {_type_name(kind)}")
            if line["name"] in names:
                found.append(f"{where}: name {line['name']} is already used in {names[line['name']]}")
            names.setdefault(line["name"], set_file)
            if line["path"] in paths:
                found.append(f"{where}: path {line['path']} is already used in {paths[line['path']]}")
                continue
            paths[line["path"]] = set_file
            parts = PurePosixPath(line["path"]).parts
            if len(parts) != 3 or parts[:2] != (VIDEOS, dataset) or parts[2] in ("", ".", ".."):
                found.append(f"{where}: path {line['path']} is not a file directly under {VIDEOS}/{dataset}/")
                continue
            path = corpus / line["path"]
            expected.add(path)
            if not path.is_file():
                found.append(f"{line['path']}: the set names it, the checkout lacks it")
                continue
            try:
                digest, size = image_sets.identity(path)
            except (OSError, ValueError) as err:
                found.append(str(err))
                continue
            if digest != line["sha256"]:
                found.append(f"{line['path']}: sha256 {digest}, the set says {line['sha256']}")
            if size != line.get("bytes"):
                found.append(f"{line['path']}: {size} bytes, the set says {line.get('bytes')}")
            if digest == line["sha256"] and not image_sets.storage.is_lfs_pointer(path):
                try:
                    what = asdict(catalogue(path.read_bytes()))
                except NotAVideo as err:
                    found.append(f"{line['path']}: {err}")
                    continue
                stated = {key: line.get(key) for key in what}
                if stated != what:
                    found.append(f"{where}: the headers say {what}, the line says {stated}")
        directory = corpus / VIDEOS / dataset
        if directory.is_dir():
            for path in sorted(p for p in directory.rglob("*") if p.is_file() and p not in expected):
                found.append(f"{path.relative_to(corpus)}: no line of {set_file.name} names it")
        if not (corpus / LICENSES / f"{dataset}-LICENSE").is_file():
            found.append(f"{LICENSES}/{dataset}-LICENSE: the licence copy of {set_file.name} is missing")
    videos_root = corpus / VIDEOS
    if videos_root.is_dir():
        for directory in sorted(p for p in videos_root.iterdir() if p.is_dir() and p.name not in datasets):
            found.append(f"{directory.relative_to(corpus)}: no set {KIND}/{directory.name}.jsonl names it")
    return found


def run(args: argparse.Namespace) -> int:
    found = problems(args.corpus)
    for problem in found:
        print(problem, file=sys.stderr)
    if found:
        print(f"bellwether videos check: {len(found)} problem(s)", file=sys.stderr)
        return 1
    sets_dir = args.corpus / KIND
    sets = sorted(sets_dir.glob("*.jsonl")) if sets_dir.is_dir() else []
    lines = [line for set_file in sets for line in image_sets.read_set(set_file)]
    print(f"{args.corpus}: {len(lines)} clips in {len(sets)} sets, {sum(int(line['bytes']) for line in lines)} bytes")
    return 0
