#!/usr/bin/env python3
"""The manifest (manifest.tsv) from the source TSVs + the catalogue, the class labels, the corpus-50 / corpus-500 tiers
and the MANIFEST summary; the root is MM_CORPUS_ROOT (default: corpus). Step 4 of the corpus build
(multimodal/README.md); multimodal/tools/import_corpus.py then lays the result out as the suite holds it.
"""

import collections
import hashlib
import json
import os
import sys

R = os.environ.get("MM_CORPUS_ROOT", "corpus")
cat = {}
for l in open(sys.argv[1]):
    d = json.loads(l)
    cat[d["path"]] = d
rows = []


def add(path, source, split, group, variant, sid, sha, nbytes):
    c = cat.get(path)
    if c is None or "error" in (c or {}):
        err = (c or {}).get("error", "not catalogued")
        rows.append(
            {
                "id": sid,
                "path": path,
                "source": source,
                "split": split,
                "group": group,
                "variant": variant,
                "class": f"{source}/undecodable",
                "format": (c or {}).get("format", "?"),
                "w": 0,
                "h": 0,
                "size_class": "-",
                "color": "-",
                "exif_orientation": "-",
                "progressive": "-",
                "subsampling": "-",
                "alpha": "-",
                "bits16": "-",
                "animated": "-",
                "tiny": "-",
                "bytes": nbytes,
                "sha256": sha,
                "note": err,
            }
        )
        return
    fmt = c["format"]
    color = c.get("color_space", "?")
    if fmt == "jpeg":
        color = c.get("jpeg_color", color)
    elif fmt == "png":
        color = c.get("png_color", color)
    o = c.get("exif_orientation")
    flags = []
    if isinstance(o, int) and 2 <= o <= 8:
        flags.append(f"exif{o}")
    if c.get("alpha"):
        flags.append("alpha")
    if c.get("animated"):
        flags.append("anim")
    if group == "derived":
        cls = f"derived/{variant}"
    elif group == "synthetic":
        cls = f"synthetic/{variant}"
    else:
        cls = f"{source}/{fmt}/{color}" + ("".join("+" + x for x in flags) if flags else "")
    rows.append(
        {
            "id": sid,
            "path": path,
            "source": source,
            "split": split,
            "group": group,
            "variant": variant,
            "class": cls,
            "format": fmt,
            "w": c["w"],
            "h": c["h"],
            "size_class": c["size_class"],
            "color": color,
            "exif_orientation": o if o is not None else "-",
            "progressive": c.get("progressive", "-"),
            "subsampling": c.get("subsampling", "-"),
            "alpha": c.get("alpha", False),
            "bits16": c.get("bits16", False),
            "animated": c.get("animated", False),
            "tiny": c.get("tiny", False),
            "bytes": nbytes,
            "sha256": sha,
            "note": "",
        }
    )


def tsv(p):
    hdr = None
    for l in open(p):
        c = l.rstrip("\n").split("\t")
        if hdr is None:
            hdr = c
            continue
        yield dict(zip(hdr, c))


SOURCES = [
    ("coco", "val2017"),
    ("textvqa", "validation"),
    ("docvqa", "validation"),
    ("ocrbench", "test"),
    ("chartqa", "test"),
    ("mmmu", "validation"),
    ("scienceqa", "test"),
]
for src, split in SOURCES:
    for r in tsv(f"{R}/sources/{src}.tsv"):
        add(r["path"], src, split, "source", "-", r["id"], r["sha256"], int(r["bytes"]))
for r in tsv(f"{R}/sources/derived.tsv"):
    b = open(r["path"], "rb").read()
    add(r["path"], "derived", "-", "derived", r["variant"], r["id"], hashlib.sha256(b).hexdigest(), len(b))
for d, label in (("synthetic-40", "parity-40"), ("synthetic-fixtures", None), ("synthetic-cmyk", "cmyk-extra")):
    for f in sorted(os.listdir(f"{R}/images/{d}")):
        p = f"{R}/images/{d}/{f}"
        b = open(p, "rb").read()
        v = label or ("fixtures-" + f.split("_")[0])
        add(p, "synthetic", "-", "synthetic", v, f"{d}_{f.rsplit('.', 1)[0]}", hashlib.sha256(b).hexdigest(), len(b))
cols = [
    "id",
    "path",
    "source",
    "split",
    "group",
    "variant",
    "class",
    "format",
    "w",
    "h",
    "size_class",
    "color",
    "exif_orientation",
    "progressive",
    "subsampling",
    "alpha",
    "bits16",
    "animated",
    "tiny",
    "bytes",
    "sha256",
    "note",
]
with open(f"{R}/manifest.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
# tiers
by_cls = collections.OrderedDict()
for r in rows:
    if r["class"].endswith("/undecodable"):
        continue
    by_cls.setdefault(r["class"], []).append(r)


def pick(lst, k):
    if k >= len(lst):
        return lst
    step = len(lst) / k
    return [lst[int(i * step)] for i in range(k)]


# corpus-500: every class at least once, the rest proportional to the class size (largest remainder), deterministic strided picks
total = sum(len(v) for v in by_cls.values())
n500 = 500
alloc = {c: 1 for c in by_cls}
rest = n500 - len(alloc)
if rest > 0:
    shares = {c: rest * len(v) / total for c, v in by_cls.items()}
    for c in by_cls:
        alloc[c] += int(shares[c])
    left = n500 - sum(alloc.values())
    for c, _ in sorted(shares.items(), key=lambda kv: -(kv[1] - int(kv[1])))[: max(0, left)]:
        alloc[c] += 1
tier500 = []
for c, v in by_cls.items():
    tier500 += pick(v, min(alloc[c], len(v)))
tier500 = tier500[:n500]
# corpus-50: round robin over a priority list of classes, one per class until 50
PRIORITY = [
    "coco/jpeg/ycbcr",
    "textvqa/jpeg/ycbcr",
    "docvqa/png/grayscale",
    "ocrbench/jpeg/ycbcr",
    "chartqa/png/rgb",
    "mmmu/png/rgb",
    "scienceqa/png/rgb",
    "derived/webp-q85",
    "derived/webp-lossless",
    "derived/png-rgba",
    "derived/png-palette",
    "derived/png-gray16",
    "derived/png-rgb16",
    "derived/cmyk-pillow",
    "derived/ycck-420",
    "derived/exif6-jpeg",
    "derived/exif3-jpeg",
    "derived/exif8-jpeg",
    "derived/exif6-webp",
    "derived/exif6-png",
    "derived/tiny-jpeg",
    "derived/huge-jpeg",
    "derived/jpeg-progressive",
    "derived/jpeg-gray",
    "derived/jpeg-444",
    "derived/png-la",
    "derived/png-palette-trns",
    "derived/png-interlaced",
    "derived/gif-animated",
    "derived/webp-animated",
    "derived/apng-animated",
    "derived/cmyk-noapp14",
    "derived/ycck-444",
    "derived/exif2-jpeg",
    "derived/exif4-jpeg",
    "derived/exif5-jpeg",
    "derived/exif7-jpeg",
    "derived/exif1-jpeg",
    "derived/jpeg-422",
    "derived/jpeg-q100",
    "derived/exif6-png-rgba",
    "derived/webp-rgba",
    "derived/png-gray8",
    "derived/png-rgb",
    "derived/tiny-png",
    "synthetic/parity-40",
    "synthetic/cmyk-extra",
    "synthetic/fixtures-exif6",
    "synthetic/fixtures-cmyk",
]
tier50 = []
seen = set()
order = [c for c in PRIORITY if c in by_cls] + [c for c in by_cls if c not in PRIORITY]
i = 0
while len(tier50) < 50 and i < 10:
    for c in order:
        v = by_cls[c]
        if i < len(v):
            r = pick(v, max(1, min(len(v), 8)))[i] if i < min(len(v), 8) else None
            if r and r["id"] not in seen:
                tier50.append(r)
                seen.add(r["id"])
        if len(tier50) >= 50:
            break
    i += 1
for name, t in (("corpus-50", tier50), ("corpus-500", tier500)):
    with open(f"{R}/{name}.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in t:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    with open(f"{R}/{name}.txt", "w") as f:
        for r in t:
            f.write(r["path"] + "\n")
with open(f"{R}/all.txt", "w") as f:
    for r in rows:
        f.write(r["path"] + "\n")
# MANIFEST.md
cnt = collections.Counter(r["class"] for r in rows)
size = collections.Counter(r["size_class"] for r in rows)
fmt = collections.Counter(r["format"] for r in rows)
colr = collections.Counter(r["color"] for r in rows)
ori = collections.Counter(str(r["exif_orientation"]) for r in rows)
flags = {
    "alpha": sum(1 for r in rows if r["alpha"] is True),
    "16-bit": sum(1 for r in rows if r["bits16"] is True),
    "animated": sum(1 for r in rows if r["animated"] is True),
    "progressive JPEG": sum(1 for r in rows if r["progressive"] is True),
    "tiny (short side < 112)": sum(1 for r in rows if r["tiny"] is True),
    "undecodable": sum(1 for r in rows if r["class"].endswith("/undecodable")),
}
sub = collections.Counter(str(r["subsampling"]) for r in rows if r["format"] == "jpeg")
by_src = collections.Counter(r["source"] for r in rows)
by_src_bytes = collections.Counter()
for r in rows:
    by_src_bytes[r["source"]] += r["bytes"]
md = (
    [
        "# multimodal parity corpus (MANIFEST)\n",
        "Built from public benchmark datasets (hub parquet files by their resolve URLs, the COCO val2017 zip from its origin) into their own directories under `corpus/raw/`; images written out exactly as stored (extension from the magic bytes), deduplicated by sha256 within a dataset; nothing from a dataset executed. Derived classes made with Pillow 12.3.0 (the engine stack's version, pinned in the lane venv) and, for YCCK, the host's libturbojpeg (tjCompress2 TJPF_CMYK). The seven clips and the synthetic's earlier sets (the 40-image set, the 30 fixture classes, the CMYK/YCCK extras) are included unchanged.\n",
        f"Totals: **{len(rows)} still images** ({sum(r['bytes'] for r in rows) / 1e9:.2f} GB) in {len(cnt)} classes + 7 clips; tiers corpus-50 ({len(tier50)}) and corpus-500 ({len(tier500)}).\n",
        "## Datasets\n",
        "| dataset | origin | split | revision | license (as stated by the source) | rows -> unique images | bytes |",
        "|---|---|---|---|---|---|---|",
        f"| COCO val2017 | images.cocodataset.org/zips/val2017.zip | val2017 | ETag d366be60d3dc737327160d62453e3973-98, Last-Modified 2018-07-11 | images under the Flickr terms of use, annotations CC BY 4.0 (cocodataset.org/#termsofuse) | 5000 files -> {by_src['coco']} | {by_src_bytes['coco'] / 1e6:.0f} MB |",
        f"| TextVQA | hub lmms-lab/textvqa (parquet) | validation | 9c0699cd | no license field on the hub card (TextVQA itself: CC BY 4.0) | 5000 rows -> {by_src['textvqa']} unique images | {by_src_bytes['textvqa'] / 1e6:.0f} MB |",
        f"| DocVQA | hub lmms-lab/DocVQA (parquet, DocVQA config) | validation | 539088ef | apache-2.0 on the hub card | 5349 rows -> {by_src['docvqa']} unique documents | {by_src_bytes['docvqa'] / 1e6:.0f} MB |",
        f"| OCRBench | hub echo840/OCRBench (parquet) | test | 92a54bd1 | no license field on the hub card | 1000 rows -> {by_src['ocrbench']} unique images | {by_src_bytes['ocrbench'] / 1e6:.0f} MB |",
        f"| ChartQA | hub HuggingFaceM4/ChartQA (parquet) | test | b605b6e0 | gpl-3.0 on the hub card | 2500 rows -> {by_src['chartqa']} unique charts | {by_src_bytes['chartqa'] / 1e6:.0f} MB |",
        f"| MMMU | hub MMMU/MMMU (parquet, 30 subjects, image_1..image_7) | validation | 876ce5cb | apache-2.0 on the hub card | 900 rows -> {by_src['mmmu']} unique images | {by_src_bytes['mmmu'] / 1e6:.0f} MB |",
        f"| ScienceQA | hub derek-thomas/ScienceQA (parquet) | test | f18b0a70 | cc-by-sa-4.0 on the hub card | 4241 rows -> {by_src['scienceqa']} unique images | {by_src_bytes['scienceqa'] / 1e6:.0f} MB |",
        f"| derived | this lane (tools/derive.py) from COCO photos | - | - | as COCO | {by_src['derived']} files | {by_src_bytes['derived'] / 1e6:.0f} MB |",
        f"| synthetic sets | the project's earlier parity sets: the 40-image set, the 30 fixture classes, the CMYK/YCCK extras | - | - | synthetic | {by_src['synthetic']} files | {by_src_bytes['synthetic'] / 1e6:.0f} MB |",
        "| clips | the project's 7 clips (mp4 h264 1 s / 30 s / 600 s / 720p 10 s / with audio 5 s, mov h264 5 s, webm vp9 5 s) | - | - | synthetic | 7 | - |",
        "",
        "## Classes (manifest column `class`)\n",
        "| class | images |",
        "|---|---|",
    ]
    + [f"| {c} | {n} |" for c, n in sorted(cnt.items())]
    + [
        "",
        "## Cross-cutting counts\n",
        "| dimension | counts |",
        "|---|---|",
        f"| format | {', '.join(f'{k} {v}' for k, v in sorted(fmt.items()))} |",
        f"| size class (longer side) | {', '.join(f'{k} {v}' for k, v in sorted(size.items()))} |",
        f"| colour representation | {', '.join(f'{k} {v}' for k, v in sorted(colr.items()))} |",
        f"| EXIF orientation | {', '.join(f'{k} {v}' for k, v in sorted(ori.items()))} |",
        f"| JPEG chroma subsampling | {', '.join(f'{k} {v}' for k, v in sorted(sub.items()))} |",
        f"| flags | {', '.join(f'{k} {v}' for k, v in flags.items())} |",
        "",
        "## Tiers\n",
        "- corpus-50 (`corpus-50.tsv` / `.txt`): one image from each of the most distinct classes (photos, documents, charts, text, diagrams, WebP, PNG variants, CMYK/YCCK, EXIF orientations on JPEG/WebP/PNG, tiny, huge, animated), the smoke set for every gate.",
        "- corpus-500 (`corpus-500.tsv` / `.txt`): every class at least once, the rest proportional to the class size, deterministic strided picks; the greedy-answer set for a real engine when decoded pixels change.",
        "- full (`manifest.tsv` / `all.txt`): everything, before a media-pipeline PR is flipped.",
        "",
        "## Oracle\n",
        "The reference for every image is the engine's own path: the engine image's media loader (PIL open, EXIF orientation applied with exif_transpose semantics, then the conversion to RGB that composites images with transparency over white) followed by the pinned reference processor (Qwen family: the torchvision-backed Qwen2-VL image processor of transformers 5.17 from the public preprocessor configs under `configs/`; MiniMax-M3: the hub's image_processor.py at the pinned revision), run with the engine stack's exact versions (see `parity/*/summary.md`).",
    ]
)
open(f"{R}/MANIFEST.md", "w").write("\n".join(md) + "\n")
print("MANIFEST_DONE", len(rows), "classes", len(cnt), "tier50", len(tier50), "tier500", len(tier500))
