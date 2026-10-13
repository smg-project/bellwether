#!/usr/bin/env python3
"""The recorder of the reference: runs INSIDE the engine's image (vllm/vllm-openai at the pinned release, CPU only, no
device): per family, vLLM's own multimodal processor for the served model config
(MULTIMODAL_REGISTRY.create_processor(ModelConfig(model dir)), no weights needed) and, per image, the engine's own loader
(ImageMediaIO.load_bytes) -> processor.apply(the family's image prompt from vLLM's dummy-inputs builder, {"image": [img]}, {})
-> pixel_values + image_grid_thw. Records sha256 of the loader's RGB bytes, of the f32 pixel_values and of their bf16
round-to-nearest-even bits, the grid and the shape (this path exposes no u8 patches), one JSON line per image; the first
line is the provenance (versions, processor classes, prompt). `bellwether multimodal import-references` turns the output
into the family's reference set (multimodal/README.md has the container command).
usage: record_vllm.py --model-dir D --list FILE --out JSONL [--rebase HOSTPREFIX=CONTAINERPREFIX] [--raw-out DIR]
       [--trust-remote-code] [--prompt TEXT] [--threads N]"""

import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rebase", default="")
ap.add_argument("--raw-out")
ap.add_argument("--trust-remote-code", action="store_true")
ap.add_argument("--prompt")
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
import vllm
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.media.image import ImageMediaIO

mc = ModelConfig(
    model=a.model_dir,
    tokenizer=a.model_dir,
    trust_remote_code=a.trust_remote_code,
    dtype="bfloat16",
    seed=0,
    limit_mm_per_prompt={"image": 1},
)
proc = MULTIMODAL_REGISTRY.create_processor(mc)
prompt = a.prompt or proc.dummy_inputs.get_dummy_text({"image": 1})
mio = ImageMediaIO()
hf = proc.info.get_hf_processor()
meta = {
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "pillow": Image.__version__,
    "numpy": np.__version__,
    "model_type": mc.hf_config.model_type,
    "processor": type(proc).__name__,
    "hf_processor": type(hf).__name__,
    "image_processor": type(getattr(hf, "image_processor", None)).__name__,
    "prompt": prompt,
    "model_dir": a.model_dir,
}
try:
    import transformers

    meta["transformers"] = transformers.__version__
except Exception:
    pass


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def bf16_bits(f):
    b = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (b >> 16) & 1
    return ((b + 0x7FFF + lsb) >> 16).astype(np.uint16)


def tensors_of(mm):
    """Every (field, tensor) in a MultiModalKwargsItems / dict / list structure."""
    found = {}

    def walk(o, key=None, depth=0):
        if depth > 6:
            return
        if isinstance(o, torch.Tensor):
            if key:
                found.setdefault(key, o)
            return
        if hasattr(o, "data") and isinstance(o.data, torch.Tensor):
            if key:
                found.setdefault(key, o.data)
                return
        if hasattr(o, "get_data"):
            try:
                walk(o.get_data(), key, depth + 1)
                return
            except Exception:
                pass
        if isinstance(o, dict) or hasattr(o, "items"):
            for k, v in o.items():
                walk(v, k if isinstance(k, str) else key, depth + 1)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, key, depth + 1)

    walk(mm)
    return found


hp, cp = (a.rebase.split("=", 1) + [""])[:2] if a.rebase else ("", "")
paths = [l.strip() for l in open(a.list) if l.strip()]
if a.raw_out:
    os.makedirs(a.raw_out, exist_ok=True)
with open(a.out, "w") as f:
    f.write(json.dumps({"meta": meta}) + "\n")
    for p0 in paths:
        p = cp + p0[len(hp) :] if hp and p0.startswith(hp) else p0
        name = os.path.basename(p0)
        rec = {"name": name, "path": p0}
        try:
            data = open(p, "rb").read()
            t0 = time.perf_counter()
            im0 = Image.open(__import__("io").BytesIO(data))
            rec["mode"] = im0.mode
            im0.close()
            loaded = mio.load_bytes(data)
            img = getattr(loaded, "media", loaded)
            t1 = time.perf_counter()
            rgb = np.asarray(img, dtype=np.uint8)
            out = proc.apply(prompt, {"image": [img]}, {})
            t2 = time.perf_counter()
            ten = tensors_of(out["mm_kwargs"])
            pv = ten.get("pixel_values")
            grid = ten.get("image_grid_thw")
            if pv is None:
                rec["error"] = f"no pixel_values in mm_kwargs; fields {sorted(ten)}"
            else:
                pv_np = pv.detach().cpu().numpy()
                rec.update(
                    {
                        "w": int(img.size[0]),
                        "h": int(img.size[1]),
                        "img_mode": img.mode,
                        "grid": grid.detach().cpu().numpy().reshape(-1).tolist() if grid is not None else None,
                        "shape": list(pv_np.shape),
                        "pixel_dtype": str(pv_np.dtype),
                        "rgb_sha": sha(rgb),
                        "fields": sorted(ten),
                        "prompt_tokens": len(out["prompt_token_ids"]),
                        "load_ms": round((t1 - t0) * 1e3, 3),
                        "apply_ms": round((t2 - t1) * 1e3, 3),
                    }
                )
                f32 = pv_np.astype(np.float32, copy=False)
                rec["f32_sha"] = sha(f32)
                rec["bf16_sha"] = sha(bf16_bits(f32))
                if a.raw_out:
                    np.save(f"{a.raw_out}/{name}.rgb.npy", rgb)
                    np.save(f"{a.raw_out}/{name}.norm.npy", f32)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        f.write(json.dumps(rec) + "\n")
        f.flush()
print("VLLM_RECORD_DONE", len(paths), flush=True)
