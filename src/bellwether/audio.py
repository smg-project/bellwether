"""bellwether audio check: the audio sets and the files in the checkout agree, with no Git LFS fetch.

An audio set is ``corpus/audio/<dataset>.jsonl``, one JSON line per clip, with the clip's bytes beside it as
``corpus/audios/<dataset>/<name>.<ext>``, exactly as shipped by the source or made by a recipe, in Git LFS. A line
names the file (``name``, a slug, and ``path`` from the corpus root), its size and sha256, what the container's own
headers say it holds (``catalogue``: the container, the codec, the sample rate, the channels, the bits per sample of a
lossless stream, the duration in seconds), a note, and ``origin`` (the dataset, its source, the file and row the bytes
came from, and the licence), as an image line does.

The catalogue comes from the headers, not from a decoder: a FLAC file's STREAMINFO block; a WAV file's ``fmt`` and
``data`` chunks; an Ogg file's first packet (``OpusHead``, the Vorbis identification header, or FLAC-in-Ogg's
STREAMINFO) and its last page's granule position for the duration; an MPEG audio file's first frame header after any
ID3v2 tag, with the Xing or Info frame's count when there is one and the bitrate's estimate when there is not; an MP4
or M4A file's ``mvhd`` duration and the audio track's sample entry. An Opus stream's sample rate is 48000, the rate a
decoder produces, whatever the header says the input was. What the engine's decoder makes of the clip, the samples it
resamples to, is the reference's to say.

``check`` is what every pull request runs: every line of every set carries the fields above with their types and
names a file directly under its dataset's directory whose pointer or content has the line's sha256 and size, no file
under an audios directory is unnamed (nested ones included), names and paths are unique, each set's licence copy is
there, and a fetched file's catalogue is its line's.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from bellwether import videos
from bellwether.importers import image_sets

KIND = "audio"
AUDIOS = "audios"
LICENSES = "licenses"
# The columns of a line, in the order they are written.
FIELDS = (
    "name",
    "path",
    "bytes",
    "sha256",
    "container",
    "codec",
    "sample_rate",
    "channels",
    "bits",
    "duration_s",
    "notes",
    "origin",
)
# What each field holds; None where the headers do not say (the bits of a lossy stream, say).
_TYPES = {
    "name": str,
    "path": str,
    "bytes": int,
    "sha256": str,
    "container": str,
    "codec": (str, type(None)),
    "sample_rate": (int, type(None)),
    "channels": (int, type(None)),
    "bits": (int, type(None)),
    "duration_s": (int, float, type(None)),
    "notes": str,
    "origin": dict,
}


class NotAudio(ValueError):
    """Bytes the catalogue does not read as an audio container, or headers that are not whole."""


@dataclass(frozen=True)
class Catalogue:
    """What a container's own headers say about a clip."""

    container: str
    codec: str | None
    sample_rate: int | None
    channels: int | None
    bits: int | None
    duration_s: float | None


def catalogue(data: bytes) -> Catalogue:
    """The catalogue of ``data``; ``NotAudio`` for a container the catalogue does not read or headers not whole."""
    try:
        if data[:4] == b"fLaC":
            return _flac(data)
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return _wav(data)
        if data[:4] == b"OggS":
            return _ogg(data)
        if len(data) >= 12 and data[4:8] == b"ftyp":
            return _isobmff(data)
        if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
            return _mpeg(data)
    except (struct.error, IndexError, ValueError, ZeroDivisionError) as err:
        raise NotAudio(f"the file's headers are not whole: {err}") from err
    raise NotAudio("not a FLAC, WAV, Ogg, MPEG audio, MP4 or M4A file")


def _seconds(samples: int | None, rate: int | None) -> float | None:
    if not samples or not rate:
        return None
    return round(samples / rate, 6)


# --- FLAC ---


def _streaminfo(block: bytes) -> tuple[int, int, int, int]:
    """``(sample rate, channels, bits, total samples)`` of a 34-byte STREAMINFO block."""
    if len(block) < 34:
        raise ValueError("STREAMINFO is shorter than 34 bytes")
    (packed,) = struct.unpack(">Q", block[10:18])
    rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    bits = ((packed >> 36) & 0x1F) + 1
    total = packed & ((1 << 36) - 1)
    return rate, channels, bits, total


def _flac(data: bytes) -> Catalogue:
    if data[4] & 0x7F != 0:
        raise ValueError("the first metadata block is not STREAMINFO")
    rate, channels, bits, total = _streaminfo(data[8:42])
    return Catalogue("flac", "flac", rate, channels, bits, _seconds(total, rate))


# --- WAV ---

_WAVE_FORMATS = {1: "pcm", 3: "float", 6: "alaw", 7: "mulaw"}


def _wav(data: bytes) -> Catalogue:
    at = 12
    fmt = None
    data_size = None
    while at + 8 <= len(data):
        kind = data[at : at + 4]
        (size,) = struct.unpack("<I", data[at + 4 : at + 8])
        body = at + 8
        if kind == b"fmt ":
            tag, channels, rate, byte_rate, _block, bits = struct.unpack("<HHIIHH", data[body : body + 16])
            if tag == 0xFFFE and size >= 26:
                (tag,) = struct.unpack("<H", data[body + 24 : body + 26])
            fmt = (tag, channels, rate, byte_rate, bits)
        elif kind == b"data":
            data_size = size
            break
        at = body + size + (size & 1)
    if fmt is None:
        raise ValueError("no fmt chunk")
    tag, channels, rate, byte_rate, bits = fmt
    family = _WAVE_FORMATS.get(tag)
    if family == "pcm":
        codec = "pcm_u8" if bits == 8 else f"pcm_s{bits}le"
    elif family == "float":
        codec = f"pcm_f{bits}le"
    elif family:
        codec = f"pcm_{family}"
    else:
        codec = f"wave_format_{tag:#06x}"
    duration = None if data_size in (None, 0, 0xFFFFFFFF) or not byte_rate else round(data_size / byte_rate, 6)
    return Catalogue("wav", codec, rate, channels, bits if family in ("pcm", "float") else None, duration)


# --- Ogg (Opus, Vorbis, FLAC) ---


def _ogg_page(data: bytes, at: int) -> tuple[int, int, int]:
    """``(granule position, body start, page end)`` of the page at ``at``."""
    if data[at : at + 4] != b"OggS":
        raise ValueError("not an Ogg page")
    (granule,) = struct.unpack("<q", data[at + 6 : at + 14])
    segments = data[at + 26]
    body = at + 27 + segments
    return granule, body, body + sum(data[at + 27 : at + 27 + segments])


def _ogg(data: bytes) -> Catalogue:
    _, body, _ = _ogg_page(data, 0)
    first = data[body:]
    if first[:8] == b"OpusHead":
        channels = first[9]
        (pre_skip,) = struct.unpack("<H", first[10:12])
        codec, rate, bits, skip = "opus", 48000, None, pre_skip
    elif first[:7] == b"\x01vorbis":
        channels = first[11]
        (rate,) = struct.unpack("<I", first[12:16])
        codec, bits, skip = "vorbis", None, 0
    elif first[:5] == b"\x7fFLAC":
        rate, channels, bits, total = _streaminfo(first[17:51])
        return Catalogue("ogg", "flac", rate, channels, bits, _seconds(total, rate))
    else:
        raise ValueError("the first Ogg packet is not an Opus, Vorbis or FLAC header")
    last = data.rfind(b"OggS")
    granule = _ogg_page(data, last)[0] if last > 0 else None
    duration = _seconds(granule - skip, rate) if granule is not None and granule > skip else None
    return Catalogue("ogg", codec, rate, channels, bits, duration)


# --- MPEG audio (MP3) ---

_MPEG_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}
_BITRATES = {
    (3, 1): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    (3, 2): (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    (3, 3): (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    (2, 1): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    (2, 2): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    (2, 3): (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
}
_LAYERS = {1: "mp3", 2: "mp2", 3: "mp1"}


def _mpeg(data: bytes) -> Catalogue:
    at = 0
    if data[:3] == b"ID3":
        size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
        at = 10 + size + (10 if data[5] & 0x10 else 0)
    while at + 4 <= len(data) and not (data[at] == 0xFF and data[at + 1] & 0xE0 == 0xE0):
        at += 1
    if at + 4 > len(data):
        raise ValueError("no MPEG audio frame")
    b1, b2, b3 = data[at + 1], data[at + 2], data[at + 3]
    version, layer = (b1 >> 3) & 3, (b1 >> 1) & 3
    if version == 1 or layer == 0:
        raise ValueError("reserved MPEG version or layer")
    rate = _MPEG_RATES[version][(b2 >> 2) & 3]
    kbps = _BITRATES[(3 if version == 3 else 2, layer)][b2 >> 4]
    channels = 1 if b3 >> 6 == 3 else 2
    per_frame = 384 if layer == 3 else 1152 if layer == 2 or version == 3 else 576
    side = (17 if channels == 1 else 32) if version == 3 else (9 if channels == 1 else 17)
    tag = at + 4 + side
    frames = None
    if data[tag : tag + 4] in (b"Xing", b"Info"):
        (flags,) = struct.unpack(">I", data[tag + 4 : tag + 8])
        if flags & 1:
            (frames,) = struct.unpack(">I", data[tag + 8 : tag + 12])
    if frames:
        duration = _seconds(frames * per_frame, rate)
    elif kbps:
        duration = round((len(data) - at) * 8 / (kbps * 1000), 6)
    else:
        duration = None
    return Catalogue("mp3" if layer == 1 else "mpeg", _LAYERS[layer], rate, channels, None, duration)


# --- ISO base media (MP4, M4A) ---


def _isobmff(data: bytes) -> Catalogue:
    video = videos.catalogue(data)
    entry = _sample_entry(data)
    codec = rate = channels = None
    if entry is not None:
        kind, body = entry
        codec = {"mp4a": "aac", "Opus": "opus", "fLaC": "flac", "alac": "alac", "ac-3": "ac3"}.get(kind, kind)
        # After the entry's size and type: 6 reserved bytes, the data reference index, then the sound sample entry
        # (version, revision, vendor; channels, sample size, compression id, packet size; the rate as 16.16).
        (version,) = struct.unpack(">H", data[body + 16 : body + 18])
        if version in (0, 1):
            (channels,) = struct.unpack(">H", data[body + 24 : body + 26])
            rate = struct.unpack(">I", data[body + 32 : body + 36])[0] >> 16
    container = "m4a" if video.container == "mp4" and video.video_codec is None else video.container
    return Catalogue(container, codec, rate, channels, None, video.duration_s)


def _sample_entry(data: bytes) -> tuple[str, int] | None:
    """``(kind, body start)`` of the first sample entry of the first sound track's ``stsd``, else None."""
    found = None

    def walk(start: int, end: int, sound: bool) -> None:
        nonlocal found
        for kind, body, box_end in videos._boxes(data, start, end):
            if found is not None:
                return
            if kind in videos._PARENTS:
                walk(body, box_end, sound)
            elif kind == b"hdlr":
                sound = data[body + 8 : body + 12] == b"soun"
                track_is_sound[0] = sound
            elif kind == b"stsd" and track_is_sound[0]:
                (size,) = struct.unpack(">I", data[body + 8 : body + 12])
                if size >= 36:
                    found = (data[body + 12 : body + 16].decode("latin1").strip(), body + 8)

    track_is_sound = [False]
    walk(0, len(data), False)
    return found


# --- the check ---


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
                    found.append(f"{where}: {field} is {value!r}, not {videos._type_name(kind)}")
            if line["name"] in names:
                found.append(f"{where}: name {line['name']} is already used in {names[line['name']]}")
            names.setdefault(line["name"], set_file)
            if line["path"] in paths:
                found.append(f"{where}: path {line['path']} is already used in {paths[line['path']]}")
                continue
            paths[line["path"]] = set_file
            parts = PurePosixPath(line["path"]).parts
            if len(parts) != 3 or parts[:2] != (AUDIOS, dataset) or parts[2] in ("", ".", ".."):
                found.append(f"{where}: path {line['path']} is not a file directly under {AUDIOS}/{dataset}/")
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
                except NotAudio as err:
                    found.append(f"{line['path']}: {err}")
                    continue
                stated = {key: line.get(key) for key in what}
                if stated != what:
                    found.append(f"{where}: the headers say {what}, the line says {stated}")
        directory = corpus / AUDIOS / dataset
        if directory.is_dir():
            for path in sorted(p for p in directory.rglob("*") if p.is_file() and p not in expected):
                found.append(f"{path.relative_to(corpus)}: no line of {set_file.name} names it")
        if not (corpus / LICENSES / f"{dataset}-LICENSE").is_file():
            found.append(f"{LICENSES}/{dataset}-LICENSE: the licence copy of {set_file.name} is missing")
    audios_root = corpus / AUDIOS
    if audios_root.is_dir():
        for directory in sorted(p for p in audios_root.iterdir() if p.is_dir() and p.name not in datasets):
            found.append(f"{directory.relative_to(corpus)}: no set {KIND}/{directory.name}.jsonl names it")
    return found


def run(args: argparse.Namespace) -> int:
    found = problems(args.corpus)
    for problem in found:
        print(problem, file=sys.stderr)
    if found:
        print(f"bellwether audio check: {len(found)} problem(s)", file=sys.stderr)
        return 1
    sets_dir = args.corpus / KIND
    sets = sorted(sets_dir.glob("*.jsonl")) if sets_dir.is_dir() else []
    lines = [line for set_file in sets for line in image_sets.read_set(set_file)]
    print(f"{args.corpus}: {len(lines)} clips in {len(sets)} sets, {sum(int(line['bytes']) for line in lines)} bytes")
    return 0
