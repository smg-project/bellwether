#!/usr/bin/env python3
"""The catalogue: per image file the container format (magic bytes), width x height (as stored, before orientation),
the colour representation, EXIF orientation, progressive JPEG, chroma subsampling, alpha, 16-bit, animation and the
size class, one JSON line per file. Step 3 of the corpus build (multimodal/README.md).
"""

import json
import struct
import sys

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def jpeg_info(b):
    info = {
        "progressive": False,
        "components": None,
        "subsampling": None,
        "adobe_transform": None,
        "app1_exif": False,
        "sof": None,
    }
    i = 2
    n = len(b)
    while i + 4 <= n and b[i] == 0xFF:
        m = b[i + 1]
        if m == 0xD8 or 0xD0 <= m <= 0xD7 or m == 0x01:
            i += 2
            continue
        if m == 0xD9 or m == 0xDA:
            break
        L = struct.unpack(">H", b[i + 2 : i + 4])[0]
        seg = b[i + 4 : i + 2 + L]
        if m == 0xE1 and seg[:6] == b"Exif\x00\x00":
            info["app1_exif"] = True
        if m == 0xEE and seg[:5] == b"Adobe" and len(seg) >= 12:
            info["adobe_transform"] = seg[11]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF) and len(seg) >= 6:
            info["sof"] = m
            info["progressive"] = m in (0xC2, 0xC6, 0xCA, 0xCE)
            nc = seg[5]
            info["components"] = nc
            comps = [
                (seg[6 + 3 * k], seg[7 + 3 * k] >> 4, seg[7 + 3 * k] & 15) for k in range(nc) if 8 + 3 * k <= len(seg)
            ]
            if nc == 1:
                info["subsampling"] = "gray"
            elif comps:
                hmax = max(c[1] for c in comps)
                vmax = max(c[2] for c in comps)
                h0, v0 = comps[0][1], comps[0][2]
                others = {(c[1], c[2]) for c in comps[1:]}
                if nc in (3, 4) and h0 == hmax and v0 == vmax and all(o == (hmax, vmax) for o in others):
                    info["subsampling"] = "4:4:4"
                elif nc in (3, 4) and (h0, v0) == (2, 1) and others <= {(1, 1)}:
                    info["subsampling"] = "4:2:2"
                elif nc in (3, 4) and (h0, v0) == (2, 2) and others <= {(1, 1)}:
                    info["subsampling"] = "4:2:0"
                elif nc in (3, 4) and (h0, v0) == (1, 2) and others <= {(1, 1)}:
                    info["subsampling"] = "4:4:0"
                else:
                    info["subsampling"] = "other:" + ",".join(f"{c[1]}x{c[2]}" for c in comps)
        i += 2 + L
    return info


def png_info(b):
    info = {
        "bit_depth": None,
        "color_type": None,
        "interlaced": None,
        "trns": False,
        "exif_chunk": False,
        "animated": False,
        "frames": 1,
    }
    i = 8
    n = len(b)
    while i + 8 <= n:
        L, t = struct.unpack(">I4s", b[i : i + 8])
        d = b[i + 8 : i + 8 + L]
        if t == b"IHDR":
            info["bit_depth"], info["color_type"], info["interlaced"] = d[8], d[9], d[12]
        elif t == b"tRNS":
            info["trns"] = True
        elif t == b"eXIf":
            info["exif_chunk"] = True
        elif t == b"acTL":
            info["animated"] = True
            info["frames"] = struct.unpack(">I", d[:4])[0]
        elif t == b"IEND":
            break
        i += 12 + L
    return info


def webp_info(b):
    info = {"vp8": None, "alpha_flag": False, "animated": False, "exif_chunk": False}
    i = 12
    n = len(b)
    while i + 8 <= n:
        t = b[i : i + 4]
        L = struct.unpack("<I", b[i + 4 : i + 8])[0]
        if t in (b"VP8 ", b"VP8L"):
            info["vp8"] = t.decode().strip() if info["vp8"] is None else info["vp8"]
        elif t == b"VP8X":
            info["vp8"] = "VP8X"
            fl = b[i + 8]
            info["alpha_flag"] = bool(fl & 0x10)
            info["animated"] = bool(fl & 0x02)
        elif t == b"EXIF":
            info["exif_chunk"] = True
        elif t == b"ANIM":
            info["animated"] = True
        elif t == b"ALPH":
            info["alpha_flag"] = True
        i += 8 + L + (L & 1)
    return info


def fmt_of(b):
    if b[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if b[:2] == b"BM":
        return "bmp"
    if b[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return "unknown"


def size_class(w, h):
    m = max(w, h)
    return (
        "s320"
        if m <= 320
        else "s640"
        if m <= 640
        else "s1280"
        if m <= 1280
        else "s2000"
        if m <= 2000
        else "s4000"
        if m <= 4000
        else "s4000p"
    )


def one(path):
    b = open(path, "rb").read()
    rec = {"path": path, "name": path.rsplit("/", 1)[-1], "bytes": len(b), "format": fmt_of(b)}
    try:
        im = Image.open(path)
        rec.update({"w": im.width, "h": im.height, "mode": im.mode, "pil_format": im.format})
        try:
            rec["exif_orientation"] = int(im.getexif().get(0x0112, 0)) or None
        except Exception:
            rec["exif_orientation"] = "unreadable"
        rec["transparency_info"] = "transparency" in im.info
        rec["animated"] = bool(getattr(im, "is_animated", False))
        rec["frames"] = int(getattr(im, "n_frames", 1))
        if rec["format"] == "jpeg":
            j = jpeg_info(b)
            rec.update(j)
            rec["jpeg_color"] = (
                "gray"
                if j["components"] == 1
                else "ycck"
                if j["components"] == 4 and j["adobe_transform"] == 2
                else "cmyk"
                if j["components"] == 4
                else "rgb-adobe"
                if j["components"] == 3 and j["adobe_transform"] == 0
                else "ycbcr"
                if j["components"] == 3
                else f"c{j['components']}"
            )
        elif rec["format"] == "png":
            p = png_info(b)
            rec.update(p)
            ct = {0: "gray", 2: "rgb", 3: "palette", 4: "gray-alpha", 6: "rgb-alpha"}.get(
                p["color_type"], str(p["color_type"])
            )
            rec["png_color"] = (
                ct
                + (f"-{p['bit_depth']}bit" if p["bit_depth"] not in (8, None) else "")
                + ("+trns" if p["trns"] else "")
            )
        elif rec["format"] == "webp":
            rec.update(webp_info(b))
        s = min(im.width, im.height)
        rec["size_class"] = size_class(im.width, im.height)
        rec["tiny"] = s < 112
        rec["color_space"] = (
            "cmyk/ycck"
            if im.mode == "CMYK"
            else "grayscale"
            if im.mode in ("L", "1", "I;16", "I")
            else "gray-alpha"
            if im.mode == "LA"
            else "palette"
            if im.mode in ("P", "PA")
            else "rgb-alpha"
            if im.mode == "RGBA"
            else "rgb"
            if im.mode == "RGB"
            else im.mode
        )
        rec["bits16"] = im.mode in ("I;16", "I;16B", "I;16L", "I") or (
            rec["format"] == "png" and rec.get("bit_depth") == 16
        )
        rec["alpha"] = im.mode in ("RGBA", "LA", "PA") or rec["transparency_info"]
        im.close()
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return rec


paths = [l.strip() for l in open(sys.argv[1]) if l.strip()]
with open(sys.argv[2], "w") as f:
    for p in paths:
        f.write(json.dumps(one(p)) + "\n")
print("CATALOGUE_DONE", len(paths))
