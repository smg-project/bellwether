#!/usr/bin/env python3
# ruff: noqa: E501  (the recipe lines are kept as the commands they record)
"""Synthetic image corpus for the SMG-vs-vLLM multimodal comparison (deterministic, with a manifest).

Every file is derived from one seeded synthetic scene (gradients, blocks and a checker, so resize and
colour conversion have structure to work on). The manifest records name, path, bytes, sha256, the class
tags and the generator recipe. ffmpeg is used only where Pillow cannot write the form (16-bit RGB PNG,
AVIF, HEIC).

    python -I gen_images.py --out <corpus dir> --ffmpeg <ffmpeg>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

SEED = 20261105


def scene(width: int, height: int, seed: int = SEED) -> np.ndarray:
    """A deterministic RGB8 scene: two gradients, a checker, a few blocks, light noise."""
    rng = np.random.default_rng(seed + width * 7919 + height)
    ys, xs = np.mgrid[0:height, 0:width]
    r = (xs * 255 // max(width - 1, 1)).astype(np.uint8)
    g = (ys * 255 // max(height - 1, 1)).astype(np.uint8)
    checker = (((xs // max(width // 16, 1)) + (ys // max(height // 16, 1))) % 2) * 255
    b = checker.astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)
    for _ in range(6):
        x0, y0 = rng.integers(0, max(width, 1)), rng.integers(0, max(height, 1))
        w, h = rng.integers(1, max(width // 4, 2)), rng.integers(1, max(height // 4, 2))
        img[y0 : y0 + h, x0 : x0 + w] = rng.integers(0, 256, size=3, dtype=np.uint8)
    noise = rng.integers(0, 12, size=img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Corpus:
    def __init__(self, out: Path, ffmpeg: str):
        self.out = out
        self.ffmpeg = ffmpeg
        self.rows: list[dict] = []
        (out / "images").mkdir(parents=True, exist_ok=True)

    def add(self, name: str, path: Path, tags: dict, recipe: str) -> None:
        self.rows.append(
            {
                "name": name,
                "path": str(path.relative_to(self.out)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "modality": "image",
                "tags": tags,
                "recipe": recipe,
            }
        )
        print(f"  {name}: {path.stat().st_size} B", file=sys.stderr)

    def pil(self, name: str, image: Image.Image, fmt: str, ext: str, tags: dict, **save) -> Path:
        path = self.out / "images" / f"{name}.{ext}"
        image.save(path, fmt, **save)
        recipe = f"PIL.Image.save({fmt}, {json.dumps({k: (v if isinstance(v, (int, float, str, bool)) else type(v).__name__) for k, v in save.items()})})"
        self.add(name, path, tags, recipe)
        return path

    def ffmpeg_still(
        self, name: str, rgb: np.ndarray, ext: str, args: list[str], tags: dict, pix_in="rgb24"
    ) -> Path | None:
        path = self.out / "images" / f"{name}.{ext}"
        h, w = rgb.shape[:2]
        cmd = [
            self.ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_in,
            "-s",
            f"{w}x{h}",
            "-i",
            "-",
            *args,
            str(path),
        ]
        proc = subprocess.run(cmd, input=rgb.tobytes(), capture_output=True)
        if proc.returncode != 0 or not path.exists() or path.stat().st_size == 0:
            print(f"  {name}: ffmpeg failed: {proc.stderr.decode(errors='replace')[:200]}", file=sys.stderr)
            self.rows.append(
                {
                    "name": name,
                    "path": None,
                    "modality": "image",
                    "tags": tags,
                    "recipe": " ".join(cmd[1:]),
                    "error": proc.stderr.decode(errors="replace")[:300],
                }
            )
            return None
        self.add(name, path, tags, "ffmpeg " + " ".join(cmd[1:]).replace(str(path), "<out>"))
        return path


def exif_orientation(n: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = n
    return exif.tobytes()


def srgb_icc() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--skip-huge", action="store_true", help="skip the 8192x8192 and 7000x7000 classes")
    args = ap.parse_args()
    out = Path(args.out)
    c = Corpus(out, args.ffmpeg)

    base = Image.fromarray(scene(1024, 768))

    # --- sizes (JPEG baseline 4:2:0 q90) -------------------------------------------------
    sizes = [
        (1, 1),
        (2, 2),
        (7, 5),
        (16, 16),
        (64, 64),
        (256, 256),
        (640, 480),
        (1024, 768),
        (1920, 1080),
        (4096, 2304),
    ]
    if not args.skip_huge:
        sizes += [(8192, 8192)]
    for w, h in sizes:
        c.pil(
            f"size-{w}x{h}-jpeg",
            Image.fromarray(scene(w, h)),
            "JPEG",
            "jpg",
            {"class": "size", "w": w, "h": h, "format": "jpeg"},
            quality=90,
            subsampling="4:2:0",
        )
    # extreme aspect ratios
    for w, h in [(2000, 40), (40, 2000), (4000, 15), (10000, 10), (3, 900)]:
        c.pil(
            f"aspect-{w}x{h}-jpeg",
            Image.fromarray(scene(w, h)),
            "JPEG",
            "jpg",
            {"class": "aspect", "w": w, "h": h, "format": "jpeg", "ratio": round(max(w, h) / min(w, h), 1)},
            quality=90,
        )
    # png sizes for the perf classes
    for w, h in [(256, 256), (1024, 768), (1920, 1080), (3840, 2160)]:
        c.pil(
            f"size-{w}x{h}-png",
            Image.fromarray(scene(w, h)),
            "PNG",
            "png",
            {"class": "size", "w": w, "h": h, "format": "png"},
            compress_level=6,
        )

    # --- JPEG variants --------------------------------------------------------------------
    c.pil(
        "jpeg-baseline-420-q90",
        base,
        "JPEG",
        "jpg",
        {"class": "jpeg", "variant": "baseline-420"},
        quality=90,
        subsampling="4:2:0",
    )
    c.pil(
        "jpeg-baseline-444-q95",
        base,
        "JPEG",
        "jpg",
        {"class": "jpeg", "variant": "baseline-444"},
        quality=95,
        subsampling="4:4:4",
    )
    c.pil(
        "jpeg-baseline-422-q85",
        base,
        "JPEG",
        "jpg",
        {"class": "jpeg", "variant": "baseline-422"},
        quality=85,
        subsampling="4:2:2",
    )
    c.pil(
        "jpeg-progressive-q90",
        base,
        "JPEG",
        "jpg",
        {"class": "jpeg", "variant": "progressive"},
        quality=90,
        progressive=True,
    )
    c.pil(
        "jpeg-optimized-q75",
        base,
        "JPEG",
        "jpg",
        {"class": "jpeg", "variant": "optimized-q75"},
        quality=75,
        optimize=True,
    )
    c.pil("jpeg-grayscale", base.convert("L"), "JPEG", "jpg", {"class": "jpeg", "variant": "grayscale"}, quality=90)
    c.pil("jpeg-cmyk", base.convert("CMYK"), "JPEG", "jpg", {"class": "jpeg", "variant": "cmyk"}, quality=90)
    c.pil("jpeg-icc-srgb", base, "JPEG", "jpg", {"class": "jpeg", "variant": "icc"}, quality=90, icc_profile=srgb_icc())
    c.pil(
        "jpeg-q100-444", base, "JPEG", "jpg", {"class": "jpeg", "variant": "q100-444"}, quality=100, subsampling="4:4:4"
    )
    for n in range(1, 9):
        c.pil(
            f"jpeg-exif-orientation-{n}",
            Image.fromarray(scene(640, 480)),
            "JPEG",
            "jpg",
            {"class": "exif", "orientation": n, "format": "jpeg"},
            quality=90,
            exif=exif_orientation(n),
        )
    # a tall source with orientation 6 so the transposed size differs visibly
    c.pil(
        "jpeg-exif-orientation-6-tall",
        Image.fromarray(scene(480, 800)),
        "JPEG",
        "jpg",
        {"class": "exif", "orientation": 6, "format": "jpeg", "tall": True},
        quality=90,
        exif=exif_orientation(6),
    )

    # --- PNG variants ---------------------------------------------------------------------
    c.pil("png-rgb8", base, "PNG", "png", {"class": "png", "variant": "rgb8"}, compress_level=6)
    rgba = base.convert("RGBA")
    alpha = Image.fromarray((scene(1024, 768, SEED + 1)[:, :, 0]).astype(np.uint8), "L")
    rgba.putalpha(alpha)
    c.pil("png-rgba8", rgba, "PNG", "png", {"class": "png", "variant": "rgba8", "alpha": True}, compress_level=6)
    c.pil("png-gray8", base.convert("L"), "PNG", "png", {"class": "png", "variant": "gray8"})
    c.pil(
        "png-gray16",
        Image.fromarray(scene(640, 480)[:, :, 1].astype(np.uint16) * 257),
        "PNG",
        "png",
        {"class": "png", "variant": "gray16", "bits": 16},
    )
    c.pil(
        "png-palette",
        base.convert("P", palette=Image.ADAPTIVE, colors=64),
        "PNG",
        "png",
        {"class": "png", "variant": "palette"},
    )
    pal_t = base.convert("P", palette=Image.ADAPTIVE, colors=64)
    c.pil(
        "png-palette-trns",
        pal_t,
        "PNG",
        "png",
        {"class": "png", "variant": "palette-trns", "alpha": True},
        transparency=3,
    )
    c.pil("png-1bit", base.convert("1"), "PNG", "png", {"class": "png", "variant": "1bit"})
    c.pil("png-la8", base.convert("LA"), "PNG", "png", {"class": "png", "variant": "la8", "alpha": True})
    c.pil("png-icc-srgb", base, "PNG", "png", {"class": "png", "variant": "icc"}, icc_profile=srgb_icc())
    c.pil(
        "png-exif-orientation-6",
        Image.fromarray(scene(640, 480)),
        "PNG",
        "png",
        {"class": "exif", "orientation": 6, "format": "png"},
        exif=exif_orientation(6),
    )
    rgb16 = scene(640, 480).astype(np.uint16) * 257
    c.ffmpeg_still(
        "png-rgb16",
        rgb16.astype("<u2"),
        "png",
        ["-frames:v", "1", "-pix_fmt", "rgb48be"],
        {"class": "png", "variant": "rgb16", "bits": 16},
        pix_in="rgb48le",
    )
    rgba16 = np.dstack([rgb16, (scene(640, 480, SEED + 2)[:, :, 0].astype(np.uint16) * 257)])
    c.ffmpeg_still(
        "png-rgba16",
        rgba16.astype("<u2"),
        "png",
        ["-frames:v", "1", "-pix_fmt", "rgba64be"],
        {"class": "png", "variant": "rgba16", "bits": 16, "alpha": True},
        pix_in="rgba64le",
    )

    # --- WebP, GIF, BMP, TIFF, AVIF, HEIC --------------------------------------------------
    c.pil("webp-lossy-q80", base, "WEBP", "webp", {"class": "webp", "variant": "lossy"}, quality=80)
    c.pil("webp-lossless", base, "WEBP", "webp", {"class": "webp", "variant": "lossless"}, lossless=True)
    c.pil(
        "webp-lossy-alpha", rgba, "WEBP", "webp", {"class": "webp", "variant": "lossy-alpha", "alpha": True}, quality=80
    )
    c.pil(
        "webp-exif-orientation-6",
        Image.fromarray(scene(640, 480)),
        "WEBP",
        "webp",
        {"class": "exif", "orientation": 6, "format": "webp"},
        quality=90,
        exif=exif_orientation(6),
    )
    c.pil("gif-static", base.convert("P", palette=Image.ADAPTIVE), "GIF", "gif", {"class": "gif", "variant": "static"})
    frames = [Image.fromarray(scene(128, 96, SEED + i)).convert("P", palette=Image.ADAPTIVE) for i in range(10)]
    path = out / "images" / "gif-animated-10.gif"
    frames[0].save(path, "GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    c.add("gif-animated-10", path, {"class": "gif", "variant": "animated", "frames": 10}, "PIL GIF save_all 10 frames")
    c.pil("bmp-24", base, "BMP", "bmp", {"class": "bmp", "variant": "24bit"})
    c.pil("tiff-uncompressed", base, "TIFF", "tiff", {"class": "tiff", "variant": "uncompressed"})
    c.pil("tiff-lzw", base, "TIFF", "tiff", {"class": "tiff", "variant": "lzw"}, compression="tiff_lzw")
    c.pil(
        "tiff-exif-orientation-3",
        Image.fromarray(scene(640, 480)),
        "TIFF",
        "tiff",
        {"class": "exif", "orientation": 3, "format": "tiff"},
        exif=exif_orientation(3),
    )
    c.pil("ico-64", Image.fromarray(scene(64, 64)), "ICO", "ico", {"class": "ico"})
    small = scene(640, 480)
    c.ffmpeg_still(
        "avif-aom",
        small,
        "avif",
        ["-frames:v", "1", "-c:v", "libaom-av1", "-still-picture", "1", "-crf", "30", "-pix_fmt", "yuv420p"],
        {"class": "avif"},
    )
    c.ffmpeg_still(
        "heic-x265",
        small,
        "heic",
        ["-frames:v", "1", "-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params", "log-level=none", "-f", "heif"],
        {"class": "heic"},
    )
    c.ffmpeg_still("jxl", small, "jxl", ["-frames:v", "1", "-c:v", "libjxl"], {"class": "jxl"})

    # --- the big one for the concurrency test ------------------------------------------------
    if not args.skip_huge:
        rng = np.random.default_rng(SEED)
        noise = rng.integers(0, 256, size=(7000, 7000, 3), dtype=np.uint8)
        c.pil(
            "big-noise-7000-q100",
            Image.fromarray(noise),
            "JPEG",
            "jpg",
            {"class": "big", "w": 7000, "h": 7000},
            quality=100,
            subsampling="4:4:4",
        )
        del noise

    manifest = out / "manifest-images.json"
    manifest.write_text(json.dumps({"seed": SEED, "generator": "gen_images.py", "rows": c.rows}, indent=1))
    ok = sum(1 for r in c.rows if r.get("path"))
    print(f"IMAGES_DONE ok={ok} err={len(c.rows) - ok} manifest={manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
