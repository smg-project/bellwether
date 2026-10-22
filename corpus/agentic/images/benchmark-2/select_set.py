#!/usr/bin/env python3
"""bench-r4-smokes-1: choose the r4-500 image set from the bellwether image catalogues (corpus/image/{mmmu,docvqa}.jsonl, read as data),
per class by pixel count, deterministic (sorted by name, stride sampling, seed-free), every file under the 16 MiB inline cap.
Classes: S (<0.2 Mpx), M (0.2-0.5 Mpx: the benchmark's medium chat image), X (0.5-1 Mpx), D (1-2 Mpx: document pages),
L (2-8 Mpx: the benchmark's large chat image), H (>8 Mpx), J (derived JPEG variants of 15 rgb sources, written by derive-jpeg.py).
usage: select-set.py <bellwether checkout> <out dir>"""
import json, sys, os, collections
bw, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
CAP = 16 * 1024 * 1024
def cls(w, h):
    m = w * h / 1e6
    return "S" if m < 0.2 else "M" if m <= 0.5 else "X" if m < 1.0 else "D" if m < 2.0 else "L" if m <= 8.0 else "H"
rows = []
for ds in ("mmmu", "docvqa"):
    for l in open(f"{bw}/corpus/image/{ds}.jsonl"):
        r = json.loads(l); r["ds"] = ds; r["cls"] = cls(r["w"], r["h"]); rows.append(r)
rows.sort(key=lambda r: r["name"])
def pick(pool, n):
    pool = [r for r in pool if r["bytes"] <= CAP]
    if n >= len(pool): return pool
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]
by = collections.defaultdict(list)
for r in rows: by[(r["ds"], r["cls"])].append(r)
# S: every palette image first (rare decode class), then a stride over the rest
s_pool = by[("mmmu", "S")]; pal = [r for r in s_pool if r["color"] == "palette"]; rest = [r for r in s_pool if r["color"] != "palette"]
chosen = {"S": pal + pick(rest, 75 - len(pal)),
          "M": pick(by[("mmmu", "M")], 100) + pick(by[("docvqa", "M")], 10),
          "X": pick(by[("mmmu", "X")], 35) + pick(by[("docvqa", "X")], 20),
          "D": pick(by[("mmmu", "D")], 40) + pick(by[("docvqa", "D")], 40),
          "L": pick(by[("mmmu", "L")], 37) + pick(by[("docvqa", "L")], 63),
          "H": pick(by[("docvqa", "H")], 20)}
# the JPEG derivation sources: 15 rgb (no alpha) images, M/X from mmmu and L/D from docvqa (the rgb ones are few: take gray pages too, converted to RGB)
src = [r for r in by[("mmmu", "M")] + by[("mmmu", "X")] if r["color"] == "rgb"][:8] + pick(by[("docvqa", "L")][5::7], 7)
seen = set(); man = []
for c in ("S", "M", "X", "D", "L", "H"):
    for r in chosen[c]:
        if r["name"] in seen: continue
        seen.add(r["name"]); man.append((r["name"], c, r["ds"], f"corpus/{r['path']}", r["bytes"], r["w"], r["h"], r["color"], int(r["transparency"]), r["sha256"]))
with open(f"{out}/manifest-bw.tsv", "w") as f:
    f.write("name\tclass\tdataset\tpath\tbytes\tw\th\tcolor\ttransparency\tsha256\n")
    for m in man: f.write("\t".join(str(x) for x in m) + "\n")
with open(f"{out}/jpeg-sources.tsv", "w") as f:
    f.write("name\tdataset\tpath\tbytes\tw\th\tcolor\n")
    for r in src: f.write("\t".join(str(x) for x in (r["name"], r["ds"], f"corpus/{r['path']}", r["bytes"], r["w"], r["h"], r["color"])) + "\n")
need = sorted({m[3] for m in man} | {f"corpus/{r['path']}" for r in src})
open(f"{out}/lfs-include.txt", "w").write("\n".join(need) + "\n")
tot = sum(m[4] for m in man)
print("selected", len(man), "bellwether images,", len(src), "jpeg sources,", len(need), "files to pull, %.1f MB" % (tot / 1e6))
for c in ("S", "M", "X", "D", "L", "H"):
    bs = sorted(m[4] for m in man if m[1] == c); print(f"  {c}: n={len(bs)} bytes med={bs[len(bs)//2]/1e3:.0f}K max={bs[-1]/1e6:.2f}M mmmu={sum(1 for m in man if m[1]==c and m[2]=='mmmu')} docvqa={sum(1 for m in man if m[1]==c and m[2]=='docvqa')}")
