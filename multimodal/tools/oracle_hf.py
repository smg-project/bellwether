#!/usr/bin/env python3
"""The cross-check: the engine's image path run outside the engine, with the engine stack's library versions (vllm 0.31.0,
transformers 5.17.0, torch 2.13.0, torchvision 0.28.0, Pillow 12.3.0): the engine image loader
(vllm/multimodal/media/image.py ImageMediaIO.load_bytes: PIL open -> ImageOps.exif_transpose (errors suppressed) -> load
-> RGB conversion that composites images with transparency (RGBA/LA/PA, or a 'transparency' info key) over WHITE, PIL
convert("RGB") otherwise), then the pinned reference processor (Qwen families: transformers' torchvision-backed Qwen2-VL
image processor from a family directory with its preprocessor config; MiniMax-M3: the checkpoint's image_processor.py at
the pinned revision, loaded from the file). Per image: sha256 of the loader's RGB bytes, of the un-normalized u8
patches, of the f32 pixel_values and of their bf16 (round-to-nearest-even) bits, the grid and shape; one JSON line per
image. --raw-out writes the npy arrays too. --selftest compares this file's copy of the loader with the engine's real
ImageMediaIO on the given files. Run it in a torch environment with those versions, never in bellwether's own; the
engine's own path (record_vllm.py) is the reference, this is the cross-check (multimodal/README.md).
usage: oracle_hf.py --family qwen2_vl|qwen3_vl|minimax_m3 --config <family dir> --list <file> --out <jsonl> [--raw-out dir] [--threads n]
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageOps


# ---- the engine loader, copied from vllm 0.31.0 (multimodal/image.py + media/image.py ImageMediaIO, defaults) ----
def normalize_image(image):
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    return image


def rgba_to_rgb(image, background_color=(255, 255, 255)):
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])
    return converted


def _has_transparency(image):
    if image.mode in ("RGBA", "LA", "PA"):
        return True
    return "transparency" in getattr(image, "info", {})


def convert_image_mode(image, to_mode, background_color=(255, 255, 255)):
    if image.mode == to_mode:
        return image
    if to_mode == "RGB" and _has_transparency(image):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return rgba_to_rgb(image, background_color)
    return image.convert(to_mode)


def engine_load(data: bytes):
    """ImageMediaIO(image_mode='RGB').load_bytes without the MediaWithBytes wrapper and the pixel-count guard."""
    image = Image.open(io.BytesIO(data))
    image = normalize_image(image)
    image.load()
    if image.mode == "RGB":
        return image, "RGB", False
    mode = image.mode
    if image.mode == "RGBA":
        return rgba_to_rgb(image, (255, 255, 255)), mode, True
    return convert_image_mode(image, "RGB", (255, 255, 255)), mode, _has_transparency(image)


# -----------------------------------------------------------------------------------------------------------------


def bf16_bits(a):
    b = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (b >> 16) & 1
    return ((b + 0x7FFF + lsb) >> 16).astype(np.uint16)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_processor(family, config):
    import torch
    import torchvision
    import transformers

    if family in ("qwen2_vl", "qwen3_vl"):
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(config)
        params = json.load(open(os.path.join(config, "preprocessor_config.json")))
    elif family == "minimax_m3":
        src = os.path.join(config, "image_processor.py")
        spec = importlib.util.spec_from_file_location("minimax_reference_image", src)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = json.load(open(os.path.join(config, "preprocessor_config.json")))
        comp = cfg.get("img_token_compression_config", {})
        proc = module.MiniMaxM3VLImageProcessor(
            patch_size=cfg.get("patch_size", 14),
            merge_size=comp.get("spatial_merge_size", 2),
            temporal_patch_size=comp.get("temporal_patch_size", 2),
            image_mean=cfg.get("image_mean"),
            image_std=cfg.get("image_std"),
        )
        params = {
            "patch_size": proc.patch_size,
            "merge_size": proc.merge_size,
            "temporal_patch_size": proc.temporal_patch_size,
            "max_pixels": proc.max_pixels,
            "image_mean": list(proc.image_mean),
            "image_std": list(proc.image_std),
        }
    else:
        raise SystemExit(f"unknown family {family}")
    meta = {
        "family": family,
        "processor": type(proc).__name__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": Image.__version__,
        "numpy": np.__version__,
        "params": params,
    }
    return proc, meta


def run_one(proc, data, name, raw_out):
    t0 = time.perf_counter()
    img, mode, transparent = engine_load(data)
    t1 = time.perf_counter()
    rgb = np.asarray(img, dtype=np.uint8)
    norm = proc(images=img, return_tensors="np")
    raw = proc(images=img, do_rescale=False, do_normalize=False, return_tensors="np")
    t2 = time.perf_counter()
    pv = norm["pixel_values"].astype(np.float32, copy=False)
    u8 = raw["pixel_values"]
    u8i = np.rint(u8).astype(np.uint8)
    integer = bool(np.array_equal(u8i.astype(np.float32), u8.astype(np.float32)))
    grid = norm["image_grid_thw"].tolist()
    bf = bf16_bits(pv)
    rec = {
        "name": name,
        "w": int(img.size[0]),
        "h": int(img.size[1]),
        "mode": mode,
        "transparent": bool(transparent),
        "grid": grid,
        "shape": list(pv.shape),
        "u8_integer": integer,
        "rgb_sha": sha(rgb),
        "u8_sha": sha(u8i),
        "bf16_sha": sha(bf),
        "f32_sha": sha(pv),
        "load_ms": round((t1 - t0) * 1e3, 3),
        "proc_ms": round((t2 - t1) * 1e3, 3),
    }
    if raw_out:
        np.save(f"{raw_out}/{name}.rgb.npy", rgb)
        np.save(f"{raw_out}/{name}.u8.npy", u8i)
        np.save(f"{raw_out}/{name}.norm.npy", pv)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--list", required=True, help="one image path per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument(
        "--selftest", action="store_true", help="compare engine_load with the engine's real ImageMediaIO on the list"
    )
    a = ap.parse_args()
    paths = [l.strip() for l in open(a.list) if l.strip()]
    if a.selftest:
        import sysconfig

        site = sysconfig.get_paths()["purelib"]
        src = os.path.join(site, "vllm", "multimodal", "image.py")
        spec = importlib.util.spec_from_file_location("vllm_mm_image", src)
        vmi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vmi)
        print("SELFTEST loader helpers from", src, flush=True)
        same = diff = 0
        for p in paths:
            data = open(p, "rb").read()
            mine = np.asarray(engine_load(data)[0], dtype=np.uint8)
            im = Image.open(io.BytesIO(data))
            im = vmi.normalize_image(im)
            im.load()
            if im.mode == "RGBA":
                theirs = vmi.rgba_to_rgb(im, (255, 255, 255))
            else:
                theirs = vmi.convert_image_mode(im, "RGB", (255, 255, 255))
            theirs = np.asarray(theirs, dtype=np.uint8)
            if mine.shape == theirs.shape and np.array_equal(mine, theirs):
                same += 1
            else:
                diff += 1
                print("SELFTEST DIFF", p, mine.shape, theirs.shape, flush=True)
        print(f"SELFTEST engine_load == ImageMediaIO.load_bytes on {same}/{same + diff} files", flush=True)
        return
    import torch

    torch.set_num_threads(a.threads)
    proc, meta = load_processor(a.family, a.config)
    if a.raw_out:
        os.makedirs(a.raw_out, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for p in paths:
            name = os.path.basename(p)
            try:
                rec = run_one(proc, open(p, "rb").read(), name, a.raw_out)
            except Exception as e:  # noqa: BLE001
                rec = {"name": name, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            rec["path"] = p
            f.write(json.dumps(rec) + "\n")
            f.flush()
    print("REF_DONE", len(paths), flush=True)


if __name__ == "__main__":
    main()
