"""bellwether derive: the derived image classes, made from COCO photographs in a pinned container, and their check.

The public datasets lack most of what a media pipeline gets wrong: WebP in its four forms, PNG at 16 bits, with a
palette, with transparency, interlaced or animated, JPEG in grey, progressive, at each chroma subsampling, in CMYK and
YCCK, every EXIF orientation on three containers, pictures that are tiny or huge, GIF. So ``bellwether derive`` makes
them from the COCO set (``corpus/image/coco.jsonl``): 51 variants, each a deterministic slice of the COCO lines in name
order (``PLAN``), encoded with Pillow, numpy and libjpeg-turbo at the versions the container pins
(``docker/derive/Dockerfile``), and written as the ``derived`` image set, one line per file in ``PLAN`` order with the
catalogue read from the file's own headers, under ``corpus/images/derived/`` in Git LFS, the licence copy COCO's.

The encoders' bytes depend on their versions, so the images are made by hand, in the container, once; ``--check``
needs no encoder: it builds the plan again from the committed COCO set and compares it with the committed derived set
(every line's name, variant, parameters and source in order, the source's sha256 the COCO line's), and compares each
committed file, pointer or content, with its line. ``--write`` (in the container) encodes every variant and writes
the set; a file that already holds the same bytes, as content or as the pointer Git LFS leaves, is left alone.

A derived line's ``origin`` names the COCO image it was made from (``source``, its path from the corpus root, and
``sha256``, its digest), the ``variant`` and the encoder ``params`` as the recipe states them, the ``tool`` (this
module, the Pillow, numpy and libjpeg-turbo versions), and COCO's ``license``.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from bellwether.importers import coco, image_sets

DATASET = "derived"
LICENSE_COPY = "licenses/derived-LICENSE"
Q = 90
TINY = [16, 24, 28, 31, 48, 56, 64, 80, 96, 100, 111]
HUGE = [4096, 4400, 4800, 5200]
ORIENTATIONS = range(2, 9)
# Adam7's passes: (x0, y0, dx, dy).
PASSES = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
# The pinned encoders (docker/derive/Dockerfile): what a derived line's tool field names.
PILLOW = "12.3.0"
NUMPY = "2.3.5"
TURBOJPEG = "libjpeg-turbo 2.1.5 (Debian trixie libturbojpeg0 1:2.1.5-4)"


@dataclass(frozen=True)
class Step:
    """One derived file as the plan names it: the variant, the COCO source line, the parameters and the extension."""

    variant: str
    source: dict
    params: str
    ext: str

    @property
    def name(self) -> str:
        return f"{self.variant}-{self.source['name']}"


def _slice(lines: list[dict], start: int, count: int) -> list[dict]:
    return lines[start : start + count]


def plan(sources: list[dict]) -> list[Step]:
    """Every derived file in the order it is made, from the COCO lines sorted by name: the recipe's slices, kept
    exactly as the first derivation took them so that the names and the files do not move."""
    photos = sorted(sources, key=lambda line: line["name"])
    steps: list[Step] = []

    def put(variant: str, source: dict, ext: str, params: str) -> None:
        steps.append(Step(variant, source, params, ext))

    for s in _slice(photos, 0, 300):
        put("webp-q85", s, "webp", "WEBP quality=85")
    for s in _slice(photos, 300, 150):
        put("webp-lossless", s, "webp", "WEBP lossless")
    for s in _slice(photos, 450, 300):
        put("png-rgb", s, "png", "PNG RGB8")
    for s in _slice(photos, 750, 100):
        put("png-gray8", s, "png", "PNG L8")
    for s in _slice(photos, 850, 100):
        put("jpeg-gray", s, "jpg", f"JPEG L q{Q}")
    for s in _slice(photos, 950, 150):
        put("jpeg-progressive", s, "jpg", f"JPEG progressive q{Q}")
    for s in _slice(photos, 1100, 100):
        put("jpeg-444", s, "jpg", f"JPEG 4:4:4 q{Q}")
    for s in _slice(photos, 1200, 100):
        put("jpeg-422", s, "jpg", f"JPEG 4:2:2 q{Q}")
    for s in _slice(photos, 1300, 50):
        put("jpeg-q100", s, "jpg", "JPEG q100 4:4:4")
    for s in _slice(photos, 1350, 100):
        put("png-palette", s, "png", "PNG palette 256")
    for s in _slice(photos, 1450, 60):
        put("png-palette-trns", s, "png", "PNG palette + tRNS index 0")
    for s in _slice(photos, 1510, 100):
        put("png-rgba", s, "png", "PNG RGBA (alpha gradient)")
    for s in _slice(photos, 1610, 60):
        put("png-la", s, "png", "PNG LA (alpha gradient)")
    for s in _slice(photos, 1670, 60):
        put("webp-rgba", s, "webp", "WEBP lossless RGBA (alpha gradient)")
    for s in _slice(photos, 1730, 60):
        put("png-gray16", s, "png", "PNG gray 16-bit (low byte varies)")
    for s in _slice(photos, 1790, 60):
        put("png-rgb16", s, "png", "PNG RGB 16-bit (low byte varies)")
    for s in _slice(photos, 1850, 60):
        put("png-interlaced", s, "png", "PNG RGB8 Adam7 interlaced")
    for s in _slice(photos, 1910, 200):
        put("cmyk-pillow", s, "jpg", f"Pillow CMYK JPEG q{Q}")
        # The first derivation stripped the marker from every second file it had written so far, counted after the
        # CMYK file itself: the files before this slice number 2,360 (even), so the stripped ones are every other
        # source of the slice, starting with the first.
        if len(steps) % 2 == 0:
            put("cmyk-noapp14", s, "jpg", "Pillow CMYK JPEG, Adobe APP14 removed")
    for s in _slice(photos, 2110, 100):
        put("ycck-420", s, "jpg", f"TurboJPEG YCCK 4:2:0 q{Q}")
    for s in _slice(photos, 2210, 50):
        put("ycck-444", s, "jpg", f"TurboJPEG YCCK 4:4:4 q{Q}")
    for s in _slice(photos, 2260, 200):
        for o in ORIENTATIONS:
            put(f"exif{o}-jpeg", s, "jpg", f"JPEG q{Q} EXIF orientation {o}")
    for s in _slice(photos, 2460, 30):
        put("exif1-jpeg", s, "jpg", f"JPEG q{Q} EXIF orientation 1")
    for s in _slice(photos, 2490, 50):
        for o in ORIENTATIONS:
            put(f"exif{o}-webp", s, "webp", f"WEBP q85 EXIF orientation {o}")
    for s in _slice(photos, 2540, 50):
        for o in ORIENTATIONS:
            put(f"exif{o}-png", s, "png", f"PNG eXIf orientation {o}")
    for s in _slice(photos, 2590, 30):
        put("exif6-png-rgba", s, "png", "PNG RGBA eXIf orientation 6")
    for k, s in enumerate(_slice(photos, 2620, 150)):
        side = TINY[k % len(TINY)]
        if k % 3 == 2:
            put("tiny-png", s, "png", f"PNG short side {side}")
        else:
            put("tiny-jpeg", s, "jpg", f"JPEG q{Q} short side {side}")
    for k, s in enumerate(_slice(photos, 2770, 24)):
        put("huge-jpeg", s, "jpg", f"JPEG q85 long side {HUGE[k % len(HUGE)]}")
    for s in _slice(photos, 2794, 10):
        put("gif-animated", s, "gif", "GIF 2 frames")
        put("apng-animated", s, "png", "APNG 2 frames")
        put("webp-animated", s, "webp", "WEBP animated 2 frames")
    for s in _slice(photos, 2804, 10):
        put("gif-static", s, "gif", "GIF single frame")
    return steps


def origin(step: Step) -> dict:
    """Where a derived line came from: the COCO image, the variant and its parameters, the tool, COCO's licence."""
    return {
        "dataset": DATASET,
        "source": step.source["path"],
        "sha256": step.source["sha256"],
        "variant": step.variant,
        "params": step.params,
        "tool": f"bellwether derive, Pillow {PILLOW}, numpy {NUMPY}, {TURBOJPEG}",
        "license": coco.LICENSE,
    }


# --- the encoders, used by --write inside the container only ---


def _rgb(path: Path):
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _alpha_for(im):
    import numpy as np
    from PIL import Image

    w, h = im.size
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    a = ((x + y) / 2).astype(np.uint8)
    a[: h // 8, : w // 8] = 0
    a[-h // 8 :, -w // 8 :] = 255
    return Image.fromarray(a, "L")


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _png(w: int, h: int, depth: int, color_type: int, rows: bytes, interlace: int = 0) -> bytes:
    ihdr = struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, interlace)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(rows, 6))
        + _png_chunk(b"IEND", b"")
    )


def _strip_app14(data: bytes) -> bytes | None:
    at = 2
    while at < len(data) - 4 and data[at] == 0xFF:
        marker = data[at + 1]
        length = (data[at + 2] << 8) | data[at + 3]
        if marker == 0xEE and data[at + 4 : at + 9] == b"Adobe":
            return data[:at] + data[at + 2 + length :]
        if marker == 0xDA:
            break
        at += 2 + length
    return None


def _exif(orientation: int) -> bytes:
    from PIL import Image

    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif.tobytes()


class _TurboJPEG:
    """libturbojpeg's tjCompress2 through ctypes, for YCCK, which Pillow cannot write."""

    def __init__(self) -> None:
        import ctypes

        self.ctypes = ctypes
        tj = ctypes.CDLL("libturbojpeg.so.0")
        tj.tjInitCompress.restype = ctypes.c_void_p
        tj.tjCompress2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        tj.tjCompress2.restype = ctypes.c_int
        tj.tjFree.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
        tj.tjDestroy.argtypes = [ctypes.c_void_p]
        self.tj = tj

    def ycck(self, im, subsampling: int, quality: int) -> bytes:
        import numpy as np

        ctypes, tj = self.ctypes, self.tj
        inverted = (255 - np.asarray(im.convert("CMYK"), dtype=np.uint8)).tobytes()  # Adobe-inverted ink values
        handle = tj.tjInitCompress()
        buffer = ctypes.POINTER(ctypes.c_ubyte)()
        size = ctypes.c_ulong(0)
        rc = tj.tjCompress2(
            handle,
            inverted,
            im.width,
            0,
            im.height,
            11,
            ctypes.byref(buffer),
            ctypes.byref(size),
            subsampling,
            quality,
            0,
        )
        if rc != 0:
            tj.tjDestroy(handle)
            raise RuntimeError("tjCompress2 failed")
        data = bytes(ctypes.string_at(buffer, size.value))
        tj.tjFree(buffer)
        tj.tjDestroy(handle)
        return data


def encode(step: Step, source_path: Path, turbo: _TurboJPEG | None, cmyk_cache: dict[str, bytes]) -> bytes:
    """The bytes of one derived file, encoded as the first derivation encoded them."""
    import numpy as np
    from PIL import Image, ImageOps

    variant, params = step.variant, step.params
    im = _rgb(source_path)

    def saved(image, fmt: str, **save) -> bytes:
        out = io.BytesIO()
        image.save(out, format=fmt, **save)
        return out.getvalue()

    if variant == "webp-q85":
        return saved(im, "WEBP", quality=85)
    if variant == "webp-lossless":
        return saved(im, "WEBP", lossless=True)
    if variant == "png-rgb":
        return saved(im, "PNG")
    if variant == "png-gray8":
        return saved(im.convert("L"), "PNG")
    if variant == "jpeg-gray":
        return saved(im.convert("L"), "JPEG", quality=Q)
    if variant == "jpeg-progressive":
        return saved(im, "JPEG", quality=Q, progressive=True)
    if variant == "jpeg-444":
        return saved(im, "JPEG", quality=Q, subsampling=0)
    if variant == "jpeg-422":
        return saved(im, "JPEG", quality=Q, subsampling=1)
    if variant == "jpeg-q100":
        return saved(im, "JPEG", quality=100, subsampling=0)
    if variant == "png-palette":
        return saved(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256), "PNG")
    if variant == "png-palette-trns":
        return saved(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=255), "PNG", transparency=0)
    if variant == "png-rgba":
        im.putalpha(_alpha_for(im))
        return saved(im, "PNG")
    if variant == "png-la":
        return saved(Image.merge("LA", [im.convert("L"), _alpha_for(im)]), "PNG")
    if variant == "webp-rgba":
        im.putalpha(_alpha_for(im))
        return saved(im, "WEBP", lossless=True)
    if variant == "png-gray16":
        a = np.asarray(im.convert("L"), dtype=np.uint16)
        h, w = a.shape
        yy, xx = np.indices((h, w))
        v = (a << 8) | ((xx * 37 + yy * 11) % 256).astype(np.uint16)
        return _png(w, h, 16, 0, b"".join(b"\x00" + v[r].astype(">u2").tobytes() for r in range(h)))
    if variant == "png-rgb16":
        a = np.asarray(im, dtype=np.uint16)
        h, w, _ = a.shape
        yy, xx = np.indices((h, w))
        v = (a << 8) | ((xx * 37 + yy * 11) % 256).astype(np.uint16)[:, :, None]
        return _png(w, h, 16, 2, b"".join(b"\x00" + v[r].astype(">u2").tobytes() for r in range(h)))
    if variant == "png-interlaced":
        a = np.asarray(im, dtype=np.uint8)
        h, w, _ = a.shape
        out = bytearray()
        for x0, y0, dx, dy in PASSES:
            sub = a[y0::dy, x0::dx]
            if sub.shape[0] == 0 or sub.shape[1] == 0:
                continue
            for r in range(sub.shape[0]):
                out += b"\x00" + sub[r].tobytes()
        return _png(w, h, 8, 2, bytes(out), interlace=1)
    if variant == "cmyk-pillow":
        data = saved(im.convert("CMYK"), "JPEG", quality=Q)
        cmyk_cache[step.source["name"]] = data
        return data
    if variant == "cmyk-noapp14":
        data = cmyk_cache.get(step.source["name"]) or saved(im.convert("CMYK"), "JPEG", quality=Q)
        stripped = _strip_app14(data)
        if stripped is None:
            raise ValueError(f"{step.name}: the CMYK JPEG carries no Adobe APP14 marker to remove")
        return stripped
    if variant in ("ycck-420", "ycck-444"):
        if turbo is None:
            raise RuntimeError("libturbojpeg.so.0 is needed for the YCCK variants; run in the derive container")
        return turbo.ycck(im, 2 if variant == "ycck-420" else 0, Q)
    if variant.startswith("exif") and variant.endswith("-jpeg"):
        return saved(im, "JPEG", quality=Q, exif=_exif(int(variant[4])))
    if variant.startswith("exif") and variant.endswith("-webp"):
        return saved(im, "WEBP", quality=85, exif=_exif(int(variant[4])))
    if variant == "exif6-png-rgba":
        im.putalpha(_alpha_for(im))
        return saved(im, "PNG", exif=_exif(6))
    if variant.startswith("exif") and variant.endswith("-png"):
        return saved(im, "PNG", exif=_exif(int(variant[4])))
    if variant in ("tiny-png", "tiny-jpeg"):
        side = int(params.rsplit(" ", 1)[1])
        w, h = im.size
        f = side / min(w, h)
        small = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.BICUBIC)
        return saved(small, "PNG") if variant == "tiny-png" else saved(small, "JPEG", quality=Q)
    if variant == "huge-jpeg":
        long = int(params.rsplit(" ", 1)[1])
        w, h = im.size
        f = long / max(w, h)
        return saved(im.resize((round(w * f), round(h * f)), Image.Resampling.BICUBIC), "JPEG", quality=85)
    if variant in ("gif-animated", "apng-animated", "webp-animated"):
        flipped = ImageOps.flip(im)
        fmt = {"gif-animated": "GIF", "apng-animated": "PNG", "webp-animated": "WEBP"}[variant]
        return saved(im, fmt, save_all=True, append_images=[flipped], duration=100, loop=0)
    if variant == "gif-static":
        return saved(im, "GIF")
    raise ValueError(f"no encoder for the variant {variant!r}")


# --- the set ---


def _found(steps: list[Step], corpus: Path) -> Iterator[image_sets.Found]:
    """Each planned file encoded, in plan order, as ``image_sets.write`` takes them."""
    turbo = None
    try:
        turbo = _TurboJPEG()
    except OSError:
        pass
    cache: dict[str, bytes] = {}
    for step in steps:
        data = encode(step, corpus / step.source["path"], turbo, cache)
        yield image_sets.Found(data, origin(step), step.name, name=step.name)


def read_sources(corpus: Path) -> list[dict]:
    path = image_sets.set_path(corpus, coco.DATASET)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the COCO set the derived classes are made from; import coco first")
    return image_sets.read_set(path)


def check(corpus: Path) -> list[str]:
    """One line per difference between the plan over the committed COCO set and the committed derived set and files;
    empty when there is none. No encoder runs: the plan names every file, its variant, parameters and source, and the
    files are compared through their pointers or content with the lines' sha256 and size."""
    problems: list[str] = []
    steps = plan(read_sources(corpus))
    target = image_sets.set_path(corpus, DATASET)
    if not target.is_file():
        return [f"{target}: missing"]
    lines = image_sets.read_set(target)
    if len(lines) != len(steps):
        problems.append(f"{target}: {len(lines)} lines, the plan over the COCO set names {len(steps)} files")
    expected: set[Path] = set()
    for number, (line, step) in enumerate(zip(lines, steps, strict=False), 1):
        wanted = origin(step)
        if line["name"] != step.name or line.get("origin") != wanted:
            problems.append(
                f"{target}:{number}: names {line['name']} from {line.get('origin')}; "
                f"the plan names {step.name} from {wanted}"
            )
            continue
        path = corpus / line["path"]
        expected.add(path)
        if not line["path"].startswith(f"{image_sets.IMAGES}/{DATASET}/{step.name}."):
            problems.append(f"{target}:{number}: path {line['path']} is not the plan's {step.name}.{step.ext}")
        if not path.is_file():
            problems.append(f"{path}: missing")
            continue
        try:
            digest, size = image_sets.identity(path)
        except (OSError, ValueError) as err:
            problems.append(str(err))
            continue
        if (digest, size) != (line["sha256"], line["bytes"]):
            problems.append(f"{path}: holds sha256 {digest}, {size} bytes; the set says {line['sha256']}")
    directory = image_sets.images_dir(corpus, DATASET)
    if directory.is_dir():
        for path in sorted(p for p in directory.iterdir() if p.is_file() and p not in expected):
            problems.append(f"{path}: no line of the set names it")
    licence = corpus / LICENSE_COPY
    source_licence = corpus / coco.LICENSE_COPY
    if not licence.is_file():
        problems.append(f"{licence}: missing")
    elif source_licence.is_file() and licence.read_bytes() != source_licence.read_bytes():
        problems.append(f"{licence}: differs from {source_licence}")
    return problems


def write(corpus: Path) -> image_sets.Summary:
    """Encode every planned file and write the set, in the container (``--write``)."""
    steps = plan(read_sources(corpus))
    copies = {LICENSE_COPY: (corpus / coco.LICENSE_COPY).read_bytes()}
    _, summary = image_sets.write(DATASET, _found(steps, corpus), corpus, copies)
    return summary


def run(args: argparse.Namespace) -> int:
    if args.write:
        summary = write(args.corpus)
        image_sets.report(DATASET, summary, args.corpus)
        return 0
    problems = check(args.corpus)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        steps = plan(read_sources(args.corpus))
        print(f"{args.corpus}: the derived set is the plan over the COCO set, {len(steps)} files")
    return 1 if problems else 0
