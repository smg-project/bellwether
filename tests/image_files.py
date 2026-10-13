"""Image files built from their own structure, for the tests of the image importers: the catalogue reads headers, so a
file needs only the headers a format's specification lays out, and no decoder is involved on either side."""

import struct
import zlib


def exif(orientation: int | None, order: str = "<") -> bytes:
    """A TIFF-structured EXIF block with one IFD entry, the orientation; ``order`` is ``<`` (II) or ``>`` (MM)."""
    head = (b"II" if order == "<" else b"MM") + struct.pack(order + "HI", 42, 8)
    if orientation is None:
        return head + struct.pack(order + "H", 0) + struct.pack(order + "I", 0)
    entry = struct.pack(order + "HHI", 0x0112, 3, 1) + struct.pack(order + "H", orientation) + b"\x00\x00"
    return head + struct.pack(order + "H", 1) + entry + struct.pack(order + "I", 0)


def jpeg(
    w: int = 40,
    h: int = 30,
    components: int = 3,
    factors=((2, 2), (1, 1), (1, 1)),
    progressive: bool = False,
    adobe_transform: int | None = None,
    orientation: int | None = None,
    exif_order: str = "<",
    frame: bool = True,
) -> bytes:
    def segment(marker: int, data: bytes) -> bytes:
        return bytes([0xFF, marker]) + struct.pack(">H", len(data) + 2) + data

    out = b"\xff\xd8"
    if orientation is not None:
        out += segment(0xE1, b"Exif\x00\x00" + exif(orientation, exif_order))
    if adobe_transform is not None:
        out += segment(0xEE, b"Adobe" + struct.pack(">HHHB", 100, 0, 0, adobe_transform))
    out += segment(0xDB, bytes(65))  # a quantisation table, before the frame as encoders write it
    if frame:
        body = struct.pack(">BHHB", 8, h, w, components)
        for index in range(components):
            hv = factors[index] if index < len(factors) else (1, 1)
            body += bytes([index + 1, (hv[0] << 4) | hv[1], 0])
        out += segment(0xC2 if progressive else 0xC0, body)
    out += segment(0xDA, b"\x01\x01\x00\x00\x3f\x00") + b"\x00\x01" + b"\xff\xd9"
    return out


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png(
    w: int = 40,
    h: int = 30,
    color_type: int = 2,
    depth: int = 8,
    trns: bool = False,
    orientation: int | None = None,
    animated: bool = False,
    interlace: int = 0,
    end: bool = True,
) -> bytes:
    out = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, interlace))
    if animated:
        out += png_chunk(b"acTL", struct.pack(">II", 2, 0))
    if trns:
        out += png_chunk(b"tRNS", b"\x00")
    if orientation is not None:
        out += png_chunk(b"eXIf", exif(orientation, ">"))
    out += png_chunk(b"IDAT", zlib.compress(bytes(h * (1 + w * 3))))
    if end:
        out += png_chunk(b"IEND", b"")
    return out


def riff(chunks: list[tuple[bytes, bytes]]) -> bytes:
    body = b"WEBP"
    for kind, data in chunks:
        body += kind + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")
    return b"RIFF" + struct.pack("<I", len(body)) + body


def vp8(w: int, h: int) -> bytes:
    return b"\x30\x01\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", w, h) + bytes(10)


def vp8l(w: int, h: int, alpha: bool) -> bytes:
    bits = (w - 1) | ((h - 1) << 14) | (int(alpha) << 28)
    return b"\x2f" + struct.pack("<I", bits) + bytes(6)


def vp8x(w: int, h: int, alpha: bool = False, animated: bool = False, exif_block: bool = False) -> bytes:
    flags = (0x10 if alpha else 0) | (0x02 if animated else 0) | (0x08 if exif_block else 0)
    return bytes([flags, 0, 0, 0]) + (w - 1).to_bytes(3, "little") + (h - 1).to_bytes(3, "little")


def gif(w: int = 40, h: int = 30, frames: int = 1, transparent: bool = False) -> bytes:
    out = b"GIF89a" + struct.pack("<HHBBB", w, h, 0x80, 0, 0) + bytes(6)  # a global table of two colours
    for _ in range(frames):
        if transparent:
            out += b"\x21\xf9\x04" + bytes([0x01, 0, 0, 0]) + b"\x00"
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, w, h, 0) + b"\x02" + b"\x02\x44\x01" + b"\x00"
    return out + b"\x3b"
