#!/usr/bin/env python3
"""The derived classes (Pillow 12.3.0, the engine stack's version; YCCK through libturbojpeg's tjCompress2 with TJPF_CMYK):
from the COCO val2017 photos, deterministic subsets by sorted name. Writes <root>/images/derived/<variant>/<id>.<ext> and
<root>/sources/derived.tsv (id, variant, source id, path, params); the root is MM_CORPUS_ROOT (default: corpus). Step 2
of the corpus build (multimodal/README.md).
usage: python3 -I derive.py"""

import ctypes
import io
import os
import struct
import zlib

import numpy as np
from PIL import Image, ImageOps

R = os.environ.get("MM_CORPUS_ROOT", "corpus")
OUT = f"{R}/images/derived"
rows = []
src = [l.rstrip("\n").split("\t") for l in open(f"{R}/sources/coco.tsv")][1:]
src = sorted(src, key=lambda r: r[0])
photos = [(r[0], r[3]) for r in src]  # (id, path)


def take(start, n):
    return photos[start : start + n]


Q = 90


def put(variant, sid, ext, data_or_img, params, **save):
    d = f"{OUT}/{variant}"
    os.makedirs(d, exist_ok=True)
    name = f"{variant}_{sid}.{ext}"
    p = f"{d}/{name}"
    if isinstance(data_or_img, (bytes, bytearray)):
        open(p, "wb").write(data_or_img)
    else:
        data_or_img.save(p, **save)
    rows.append((name.rsplit(".", 1)[0], variant, sid, p, params))


def rgb(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im


# --- JPEG container variants ---
for sid, p in take(0, 300):
    put("webp-q85", sid, "webp", rgb(p), "WEBP quality=85", format="WEBP", quality=85)
for sid, p in take(300, 150):
    put("webp-lossless", sid, "webp", rgb(p), "WEBP lossless", format="WEBP", lossless=True)
for sid, p in take(450, 300):
    put("png-rgb", sid, "png", rgb(p), "PNG RGB8", format="PNG")
for sid, p in take(750, 100):
    put("png-gray8", sid, "png", rgb(p).convert("L"), "PNG L8", format="PNG")
for sid, p in take(850, 100):
    put("jpeg-gray", sid, "jpg", rgb(p).convert("L"), f"JPEG L q{Q}", format="JPEG", quality=Q)
for sid, p in take(950, 150):
    put("jpeg-progressive", sid, "jpg", rgb(p), f"JPEG progressive q{Q}", format="JPEG", quality=Q, progressive=True)
for sid, p in take(1100, 100):
    put("jpeg-444", sid, "jpg", rgb(p), f"JPEG 4:4:4 q{Q}", format="JPEG", quality=Q, subsampling=0)
for sid, p in take(1200, 100):
    put("jpeg-422", sid, "jpg", rgb(p), f"JPEG 4:2:2 q{Q}", format="JPEG", quality=Q, subsampling=1)
for sid, p in take(1300, 50):
    put("jpeg-q100", sid, "jpg", rgb(p), "JPEG q100 4:4:4", format="JPEG", quality=100, subsampling=0)


# --- palette / alpha / 16-bit PNG ---
def alpha_for(im):
    w, h = im.size
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    a = ((x + y) / 2).astype(np.uint8)
    a[: h // 8, : w // 8] = 0
    a[-h // 8 :, -w // 8 :] = 255
    return Image.fromarray(a, "L")


for sid, p in take(1350, 100):
    im = rgb(p)
    put(
        "png-palette",
        sid,
        "png",
        im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256),
        "PNG palette 256",
        format="PNG",
    )
for sid, p in take(1450, 60):
    im = rgb(p).convert("P", palette=Image.Palette.ADAPTIVE, colors=255)
    put("png-palette-trns", sid, "png", im, "PNG palette + tRNS index 0", format="PNG", transparency=0)
for sid, p in take(1510, 100):
    im = rgb(p)
    im.putalpha(alpha_for(im))
    put("png-rgba", sid, "png", im, "PNG RGBA (alpha gradient)", format="PNG")
for sid, p in take(1610, 60):
    im = rgb(p)
    la = Image.merge("LA", [im.convert("L"), alpha_for(im)])
    put("png-la", sid, "png", la, "PNG LA (alpha gradient)", format="PNG")
for sid, p in take(1670, 60):
    im = rgb(p)
    im.putalpha(alpha_for(im))
    put("webp-rgba", sid, "webp", im, "WEBP lossless RGBA (alpha gradient)", format="WEBP", lossless=True)


def png_chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)


def write_png(path, w, h, bit_depth, color_type, rows_bytes, interlace=0):
    ihdr = struct.pack(">IIBBBBB", w, h, bit_depth, color_type, 0, 0, interlace)
    open(path, "wb").write(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(rows_bytes, 6))
        + png_chunk(b"IEND", b"")
    )


for sid, p in take(1730, 60):
    im = rgb(p).convert("L")
    a = np.asarray(im, dtype=np.uint16)
    h, w = a.shape
    yy, xx = np.indices((h, w))
    low = ((xx * 37 + yy * 11) % 256).astype(np.uint16)
    v = (a << 8) | low
    raw = b"".join(b"\x00" + v[r].astype(">u2").tobytes() for r in range(h))
    d = f"{OUT}/png-gray16"
    os.makedirs(d, exist_ok=True)
    name = f"png-gray16_{sid}.png"
    write_png(f"{d}/{name}", w, h, 16, 0, raw)
    rows.append((name[:-4], "png-gray16", sid, f"{d}/{name}", "PNG gray 16-bit (low byte varies)"))
for sid, p in take(1790, 60):
    a = np.asarray(rgb(p), dtype=np.uint16)
    h, w, _ = a.shape
    yy, xx = np.indices((h, w))
    low = ((xx * 37 + yy * 11) % 256).astype(np.uint16)[:, :, None]
    v = (a << 8) | low
    raw = b"".join(b"\x00" + v[r].astype(">u2").tobytes() for r in range(h))
    d = f"{OUT}/png-rgb16"
    os.makedirs(d, exist_ok=True)
    name = f"png-rgb16_{sid}.png"
    write_png(f"{d}/{name}", w, h, 16, 2, raw)
    rows.append((name[:-4], "png-rgb16", sid, f"{d}/{name}", "PNG RGB 16-bit (low byte varies)"))
# Adam7 interlaced RGB8
PASSES = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
for sid, p in take(1850, 60):
    a = np.asarray(rgb(p), dtype=np.uint8)
    h, w, _ = a.shape
    out = bytearray()
    for x0, y0, dx, dy in PASSES:
        sub = a[y0::dy, x0::dx]
        if sub.shape[0] == 0 or sub.shape[1] == 0:
            continue
        for r in range(sub.shape[0]):
            out += b"\x00" + sub[r].tobytes()
    d = f"{OUT}/png-interlaced"
    os.makedirs(d, exist_ok=True)
    name = f"png-interlaced_{sid}.png"
    write_png(f"{d}/{name}", w, h, 8, 2, bytes(out), interlace=1)
    rows.append((name[:-4], "png-interlaced", sid, f"{d}/{name}", "PNG RGB8 Adam7 interlaced"))


# --- CMYK / YCCK ---
def strip_app14(b):
    i = 2
    while i < len(b) - 4 and b[i] == 0xFF:
        m = b[i + 1]
        L = (b[i + 2] << 8) | b[i + 3]
        if m == 0xEE and b[i + 4 : i + 9] == b"Adobe":
            return b[:i] + b[i + 2 + L :]
        if m == 0xDA:
            break
        i += 2 + L
    return None


for sid, p in take(1910, 200):
    im = rgb(p).convert("CMYK")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=Q)
    b = buf.getvalue()
    put("cmyk-pillow", sid, "jpg", b, f"Pillow CMYK JPEG q{Q}")
    if len(rows) % 2 == 0:
        s = strip_app14(b)
        if s:
            put("cmyk-noapp14", sid, "jpg", s, "Pillow CMYK JPEG, Adobe APP14 removed")
tj = None
try:
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
except OSError as e:
    print("no libturbojpeg:", e)


def ycck(im, subsamp, quality):
    cmyk = np.asarray(im.convert("CMYK"), dtype=np.uint8)
    inv = (255 - cmyk).tobytes()  # Adobe-inverted ink values
    h_ = tj.tjInitCompress()
    buf = ctypes.POINTER(ctypes.c_ubyte)()
    size = ctypes.c_ulong(0)
    rc = tj.tjCompress2(h_, inv, im.width, 0, im.height, 11, ctypes.byref(buf), ctypes.byref(size), subsamp, quality, 0)
    if rc != 0:
        tj.tjDestroy(h_)
        raise RuntimeError("tjCompress2 failed")
    data = bytes(ctypes.string_at(buf, size.value))
    tj.tjFree(buf)
    tj.tjDestroy(h_)
    return data


if tj is not None:
    for sid, p in take(2110, 100):
        put("ycck-420", sid, "jpg", ycck(rgb(p), 2, Q), f"TurboJPEG YCCK 4:2:0 q{Q}")
    for sid, p in take(2210, 50):
        put("ycck-444", sid, "jpg", ycck(rgb(p), 0, Q), f"TurboJPEG YCCK 4:4:4 q{Q}")


# --- EXIF orientation 1-8 ---
def exif_bytes(o):
    e = Image.Exif()
    e[0x0112] = o
    return e.tobytes()


for sid, p in take(2260, 200):
    im = rgb(p)
    for o in range(2, 9):
        put(
            f"exif{o}-jpeg",
            sid,
            "jpg",
            im,
            f"JPEG q{Q} EXIF orientation {o}",
            format="JPEG",
            quality=Q,
            exif=exif_bytes(o),
        )
for sid, p in take(2460, 30):
    put("exif1-jpeg", sid, "jpg", rgb(p), f"JPEG q{Q} EXIF orientation 1", format="JPEG", quality=Q, exif=exif_bytes(1))
for sid, p in take(2490, 50):
    im = rgb(p)
    for o in range(2, 9):
        put(
            f"exif{o}-webp",
            sid,
            "webp",
            im,
            f"WEBP q85 EXIF orientation {o}",
            format="WEBP",
            quality=85,
            exif=exif_bytes(o),
        )
for sid, p in take(2540, 50):
    im = rgb(p)
    for o in range(2, 9):
        put(f"exif{o}-png", sid, "png", im, f"PNG eXIf orientation {o}", format="PNG", exif=exif_bytes(o))
for sid, p in take(2590, 30):
    im = rgb(p)
    im.putalpha(alpha_for(im))
    put("exif6-png-rgba", sid, "png", im, "PNG RGBA eXIf orientation 6", format="PNG", exif=exif_bytes(6))

# --- tiny and huge edge cases ---
TINY = [16, 24, 28, 31, 48, 56, 64, 80, 96, 100, 111]
for k, (sid, p) in enumerate(take(2620, 150)):
    im = rgb(p)
    s = TINY[k % len(TINY)]
    w, h = im.size
    f = s / min(w, h)
    small = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.BICUBIC)
    if k % 3 == 2:
        put("tiny-png", sid, "png", small, f"PNG short side {s}", format="PNG")
    else:
        put("tiny-jpeg", sid, "jpg", small, f"JPEG q{Q} short side {s}", format="JPEG", quality=Q)
HUGE = [4096, 4400, 4800, 5200]
for k, (sid, p) in enumerate(take(2770, 24)):
    im = rgb(p)
    L = HUGE[k % len(HUGE)]
    w, h = im.size
    f = L / max(w, h)
    big = im.resize((round(w * f), round(h * f)), Image.Resampling.BICUBIC)
    put("huge-jpeg", sid, "jpg", big, f"JPEG q85 long side {L}", format="JPEG", quality=85)

# --- animated containers (first frame = the photo) ---
for sid, p in take(2794, 10):
    im = rgb(p)
    im2 = ImageOps.flip(im)
    put(
        "gif-animated",
        sid,
        "gif",
        im,
        "GIF 2 frames",
        format="GIF",
        save_all=True,
        append_images=[im2],
        duration=100,
        loop=0,
    )
    put(
        "apng-animated",
        sid,
        "png",
        im,
        "APNG 2 frames",
        format="PNG",
        save_all=True,
        append_images=[im2],
        duration=100,
        loop=0,
    )
    put(
        "webp-animated",
        sid,
        "webp",
        im,
        "WEBP animated 2 frames",
        format="WEBP",
        save_all=True,
        append_images=[im2],
        duration=100,
        loop=0,
    )
for sid, p in take(2804, 10):
    put("gif-static", sid, "gif", rgb(p), "GIF single frame", format="GIF")

with open(f"{R}/sources/derived.tsv", "w") as f:
    f.write("id\tvariant\tsource_id\tpath\tparams\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print("DERIVE_DONE", len(rows))
