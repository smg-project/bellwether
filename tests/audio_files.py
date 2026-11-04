"""Synthetic audio files for the tests: headers whole, streams empty or nonsense, enough for the catalogue."""

import struct


def flac(rate: int = 16000, channels: int = 1, bits: int = 16, samples: int = 56080) -> bytes:
    packed = (rate << 44) | ((channels - 1) << 41) | ((bits - 1) << 36) | samples
    info = struct.pack(">HH", 4096, 4096) + b"\x00\x00\x00" * 2 + struct.pack(">Q", packed) + b"\x00" * 16
    return b"fLaC" + bytes([0x80, 0, 0, 34]) + info + b"\xff\xf8" * 8


def wav(rate: int = 16000, channels: int = 1, bits: int = 16, tag: int = 1, data_bytes: int = 32000) -> bytes:
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", data_bytes)
    return b"RIFF" + struct.pack("<I", len(body) + data_bytes) + body + b"\x00" * data_bytes


def ogg_page(body: bytes, granule: int, flags: int = 0, sequence: int = 0) -> bytes:
    segments = [255] * (len(body) // 255) + [len(body) % 255]
    header = b"OggS" + bytes([0, flags]) + struct.pack("<qIII", granule, 1, sequence, 0) + bytes([len(segments)])
    return header + bytes(segments) + body


def opus(channels: int = 1, pre_skip: int = 312, samples: int = 96000) -> bytes:
    head = b"OpusHead" + bytes([1, channels]) + struct.pack("<HIhB", pre_skip, 16000, 0, 0)
    tags = b"OpusTags" + struct.pack("<I", 0) + struct.pack("<I", 0)
    return ogg_page(head, 0, 2) + ogg_page(tags, 0, 0, 1) + ogg_page(b"\xfc" * 40, pre_skip + samples, 4, 2)


def vorbis(channels: int = 2, rate: int = 44100, samples: int = 44100) -> bytes:
    ident = b"\x01vorbis" + struct.pack("<IBIiiiBB", 0, channels, rate, 0, 128000, 0, 0xB8, 1)
    return (
        ogg_page(ident, 0, 2) + ogg_page(b"\x03vorbis" + b"\x00" * 8, 0, 0, 1) + ogg_page(b"\x00" * 40, samples, 4, 2)
    )


def mp3(frames: int | None = 100, mono: bool = False, id3: bool = False) -> bytes:
    """MPEG-1 Layer III at 128 kbit/s and 44100 Hz, three frames, with a Xing count unless ``frames`` is None."""
    header = bytes([0xFF, 0xFB, 0x90, 0xC0 if mono else 0x00])
    side = b"\x00" * (17 if mono else 32)
    xing = b"" if frames is None else b"Xing" + struct.pack(">II", 1, frames)
    frame = header + side + xing
    frame += b"\x00" * (417 - len(frame))
    tag = b"ID3\x04\x00\x00" + bytes([0, 0, 0, 10]) + b"\x00" * 10 if id3 else b""
    return tag + frame * 3


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", 8 + len(body)) + kind + body


def m4a(duration_units: int = 44100 * 3, timescale: int = 44100, channels: int = 2, rate: int = 44100) -> bytes:
    ftyp = box(b"ftyp", b"M4A " + struct.pack(">I", 0) + b"M4A mp42isom")
    mvhd = box(b"mvhd", b"\x00" * 4 + struct.pack(">IIII", 0, 0, timescale, duration_units) + b"\x00" * 80)
    hdlr = box(b"hdlr", b"\x00" * 8 + b"soun" + b"\x00" * 12 + b"Sound\x00")
    entry = (
        b"\x00" * 6
        + struct.pack(">H", 1)
        + struct.pack(">HHI", 0, 0, 0)
        + struct.pack(">HHHH", channels, 16, 0, 0)
        + struct.pack(">I", rate << 16)
    )
    stsd = box(b"stsd", b"\x00" * 4 + struct.pack(">I", 1) + box(b"mp4a", entry))
    return ftyp + box(b"moov", mvhd + box(b"trak", box(b"mdia", hdlr + box(b"minf", box(b"stbl", stsd)))))
