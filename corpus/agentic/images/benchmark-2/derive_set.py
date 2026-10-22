#!/usr/bin/env python3
"""bench-r4-smokes-1: build the r4-500 image set directory from the fetched bellwether files (data) + 60 derived JPEG variants of 15 sources
(baseline 4:2:0 q85, progressive 4:4:4 q90, EXIF orientation 6 (rotated 90 deg, the stored pixels transposed back), CMYK q90), every file
sha256-summed; writes imageset/files/<class>/<name>, imageset/manifest.tsv (name, class, source, path, bytes, w, h, color, transparency, sha256, origin)
and the 20-image greedy sample list (imageset/greedy-20.tsv). usage: derive-set.py <bellwether checkout> <imageset dir>"""
import csv, hashlib, io, os, shutil, sys
from PIL import Image
bw, out = sys.argv[1], sys.argv[2]
Image.MAX_IMAGE_PIXELS = None
rows = list(csv.DictReader(open(f"{out}/manifest-bw.tsv"), delimiter="\t"))
srcs = list(csv.DictReader(open(f"{out}/jpeg-sources.tsv"), delimiter="\t"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
man = []
for r in rows:
    d = f"{out}/files/{r['class']}"; os.makedirs(d, exist_ok=True)
    dst = f"{d}/{r['name']}.png"; src = f"{bw}/{r['path']}"
    if not os.path.exists(dst): shutil.copyfile(src, dst)
    s = sha(dst); assert s == r["sha256"], (r["name"], s, r["sha256"])
    man.append(dict(name=r["name"], cls=r["class"], source=r["dataset"], path=f"files/{r['class']}/{r['name']}.png", bytes=os.path.getsize(dst), w=r["w"], h=r["h"],
                    color=r["color"], transparency=r["transparency"], sha256=s, origin=f"bellwether corpus/{r['path']}"))
# derived JPEG variants (class J): the pixels of the source after PIL's RGB conversion, re-encoded four ways
d = f"{out}/files/J"; os.makedirs(d, exist_ok=True)
for r in srcs:
    with Image.open(f"{bw}/{r['path']}") as im:
        rgb = im.convert("RGB")
    base = r["name"]
    variants = []
    p = f"{d}/{base}-j420.jpg"; rgb.save(p, format="JPEG", quality=85, subsampling=2, optimize=False); variants.append((p, "jpeg 4:2:0 q85 baseline", rgb.size))
    p = f"{d}/{base}-jprog444.jpg"; rgb.save(p, format="JPEG", quality=90, subsampling=0, progressive=True); variants.append((p, "jpeg 4:4:4 q90 progressive", rgb.size))
    # EXIF orientation 6: the stored pixels are the display image rotated 90 deg clockwise... store transpose(ROTATE_90) with Orientation=6 so that
    # exif_transpose gives back the original (orientation 6 = rotate 270 CW on display = rotate 90 CCW; stored = original rotated 90 CW)
    stored = rgb.transpose(Image.Transpose.ROTATE_270)
    exif = Image.Exif(); exif[0x0112] = 6
    p = f"{d}/{base}-jexif6.jpg"; stored.save(p, format="JPEG", quality=90, subsampling=2, exif=exif.tobytes()); variants.append((p, "jpeg 4:2:0 q90 EXIF orientation 6", stored.size))
    p = f"{d}/{base}-jcmyk.jpg"; rgb.convert("CMYK").save(p, format="JPEG", quality=90); variants.append((p, "jpeg CMYK (Adobe) q90", rgb.size))
    for p, desc, size in variants:
        man.append(dict(name=os.path.basename(p)[:-4], cls="J", source="derived", path=f"files/J/{os.path.basename(p)}", bytes=os.path.getsize(p), w=size[0], h=size[1],
                        color="cmyk" if "cmyk" in p else "rgb", transparency=0, sha256=sha(p), origin=f"{desc}; from bellwether corpus/{r['path']}"))
with open(f"{out}/manifest.tsv", "w") as f:
    w = csv.DictWriter(f, fieldnames=["name", "cls", "source", "path", "bytes", "w", "h", "color", "transparency", "sha256", "origin"], delimiter="\t"); w.writeheader(); w.writerows(man)
with open(f"{out}/list-all.txt", "w") as f:
    for m in man: f.write(f"{out}/{m['path']}\n")
# the greedy sample: 20 images across classes, each under 3 MB (request bodies stay small), stride-picked
sample = []
for cls, n in (("M", 6), ("S", 3), ("X", 3), ("D", 3), ("L", 3), ("J", 2)):
    pool = [m for m in man if m["cls"] == cls and m["bytes"] < 3_000_000]
    step = len(pool) / n
    sample += [pool[int(i * step)] for i in range(n)]
with open(f"{out}/greedy-20.tsv", "w") as f:
    w = csv.DictWriter(f, fieldnames=["name", "cls", "path", "bytes", "w", "h", "color", "sha256"], delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows(sample)
import collections
c = collections.Counter(m["cls"] for m in man); tot = sum(m["bytes"] for m in man)
print("r4 set:", len(man), "images", dict(sorted(c.items())), "%.1f MB" % (tot / 1e6), "max %.2f MB" % (max(m["bytes"] for m in man) / 1e6))
print("greedy-20:", [m["name"] for m in sample])
