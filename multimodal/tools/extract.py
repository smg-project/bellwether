#!/usr/bin/env python3
"""Write the images of the downloaded public datasets as files (bytes exactly as stored; the extension from the
magic bytes, never from the dataset's metadata), deduplicated by sha256 within a dataset; one TSV per dataset under
<root>/sources/<dataset>.tsv (id, sha256, bytes, path, origin row). Nothing from a dataset is executed: parquet files are
read as data (pyarrow), the COCO zip is read as data (zipfile; only .jpg members are written out). The downloads sit in
their own directories under <root>/raw/<dataset>/ (the hub parquet files by their resolve URLs, the COCO val2017 zip from
its origin); the root is MM_CORPUS_ROOT (default: corpus). Step 1 of the corpus build (multimodal/README.md).
usage: python3 -I extract.py <dataset> ...   (datasets: coco, ocrbench, chartqa, scienceqa, mmmu, textvqa, docvqa)"""

import hashlib
import os
import sys
import zipfile

R = os.environ.get("MM_CORPUS_ROOT", "corpus")
RAW, IMG, SRC = f"{R}/raw", f"{R}/images", f"{R}/sources"
os.makedirs(SRC, exist_ok=True)


def ext_of(b):
    if b[:3] == b"\xff\xd8\xff":
        return "jpg"
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
    return "bin"


class Writer:
    def __init__(self, ds):
        self.ds, self.seen, self.n, self.dup, self.skipped = ds, {}, 0, 0, 0
        os.makedirs(f"{IMG}/{ds}", exist_ok=True)
        self.tsv = open(f"{SRC}/{ds}.tsv", "w")
        self.tsv.write("id\tsha256\tbytes\tpath\torigin\n")

    def add(self, b, origin):
        if not b:
            self.skipped += 1
            return
        h = hashlib.sha256(b).hexdigest()
        if h in self.seen:
            self.dup += 1
            return
        e = ext_of(b)
        if e == "bin":
            self.skipped += 1
            return
        self.n += 1
        idx = self.n
        name = f"{self.ds}_{idx:05d}_{h[:8]}.{e}"
        path = f"{IMG}/{self.ds}/{name}"
        with open(path, "wb") as f:
            f.write(b)
        self.seen[h] = name
        self.tsv.write(f"{name.rsplit('.', 1)[0]}\t{h}\t{len(b)}\t{path}\t{origin}\n")

    def close(self):
        self.tsv.close()
        print(f"{self.ds}: written {self.n}, duplicate bytes {self.dup}, skipped {self.skipped}", flush=True)


def parquet_images(ds, files, image_cols):
    import pyarrow.parquet as pq

    w = Writer(ds)
    for fp in files:
        pf = pq.ParquetFile(fp)
        cols = [c for c in pf.schema_arrow.names if c in image_cols]
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            d = batch.to_pydict()
            n = batch.num_rows
            for i in range(n):
                for c in cols:
                    v = d[c][i]
                    if v is None:
                        continue
                    b = v.get("bytes") if isinstance(v, dict) else None
                    if b is None and isinstance(v, dict) and v.get("path"):
                        continue
                    w.add(b, f"{os.path.basename(fp)}:{i}:{c}")
    w.close()


def coco():
    w = Writer("coco")
    with zipfile.ZipFile(f"{RAW}/coco-val2017/val2017.zip") as z:
        names = sorted(m.filename for m in z.infolist() if m.filename.lower().endswith(".jpg") and not m.is_dir())
        for m in names:
            w.add(z.read(m), m)
    w.close()


def files(ds, sub=""):
    d = f"{RAW}/{ds}{sub}"
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet"))


for ds in sys.argv[1:]:
    if ds == "coco":
        coco()
    elif ds == "ocrbench":
        parquet_images("ocrbench", files("ocrbench"), {"image"})
    elif ds == "chartqa":
        parquet_images("chartqa", files("chartqa"), {"image"})
    elif ds == "scienceqa":
        parquet_images("scienceqa", files("scienceqa"), {"image"})
    elif ds == "mmmu":
        parquet_images("mmmu", files("mmmu"), {f"image_{k}" for k in range(1, 8)})
    elif ds == "textvqa":
        parquet_images("textvqa", files("textvqa"), {"image"})
    elif ds == "docvqa":
        parquet_images("docvqa", files("docvqa"), {"image"})
    else:
        raise SystemExit(f"unknown dataset {ds}")
print("EXTRACT_DONE", flush=True)
