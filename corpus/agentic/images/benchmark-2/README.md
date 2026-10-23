# benchmark-2 image set

The 500 image files, in seven classes, that the benchmark's image rows attach: the chat image tenants (one image per
request, a medium or a large one) and the image slots of the agentic corpus (`corpus/agentic/v2`, 135 sessions with a
`screenshot` or `document` slot on the first user turn). 440 of the files are images of this repository's image sets
(`corpus/image/mmmu.jsonl`: 287, `corpus/image/docvqa.jsonl`: 153; the bytes under `corpus/images/<set>/`, in Git LFS)
and 60 are derived JPEG variants of 15 RGB images of those sets (8 of mmmu, 7 of docvqa). 339,733,890 bytes in all; the largest file is 9,153,267
bytes, under the 16 MiB inline cap of a data URL.

| class | files | what | pixels (stored w x h) | bytes | sources |
|---|---:|---|---|---:|---|
| S | 75 | small | < 0.2 Mpx | 4,093,965 | mmmu 75 |
| M | 110 | the medium chat image | 0.2-0.5 Mpx | 20,121,348 | mmmu 100, docvqa 10 |
| X | 55 | between the two chat points | 0.5-1 Mpx | 24,273,815 | mmmu 35, docvqa 20 |
| D | 80 | document pages | 1-2 Mpx | 55,809,233 | mmmu 40, docvqa 40 |
| L | 100 | the large chat image | 2-8 Mpx | 192,338,486 | mmmu 37, docvqa 63 |
| H | 20 | a robustness probe | > 8 Mpx (8.1-38.6) | 13,466,039 | docvqa 20 |
| J | 60 | derived JPEGs: 15 RGB sources x {4:2:0 q85 baseline, 4:4:4 q90 progressive, 4:2:0 q90 with EXIF orientation 6, CMYK (Adobe) q90} | 0.23-4.33 Mpx | 29,631,004 | 15 sources: mmmu 8, docvqa 7 |

Colour as the files store it: 171 rgb-alpha, 158 gray, 149 rgb, 15 cmyk, 7 palette; 174 files carry transparency.

## manifest.tsv

One row per file, tab-separated: `name`, `cls`, `source` (`mmmu`, `docvqa` or `derived`), `path` (the layout a driver
reads: `files/<class>/<name>`), `bytes`, `w`, `h`, `color`, `transparency` (0/1), `sha256`, `origin`. For a source
image, `origin` is `bellwether corpus/corpus/images/<set>/<file>`: the file `corpus/images/<set>/<file>` of this
repository (the leading `corpus/` is the name of the checkout directory of the tool that fetched it; the manifest is
kept byte for byte as the benchmark's traffic image carries it, sha256
`3abab8f5f7cf4825d7cfb051522fa06a51d3f3366415514b0f9a60aa76ffd330`). For a derived file, `origin` names the encoding
and the source image: `jpeg 4:2:0 q85 baseline; from bellwether corpus/corpus/images/<set>/<file>`.

The set is rebuilt from this repository and the two tools:

```
git lfs pull --include 'corpus/images/mmmu/**,corpus/images/docvqa/**' --exclude ''   # or the files lfs-include.txt names
python3 -I select_set.py <checkout> <out>    # reads corpus/image/{mmmu,docvqa}.jsonl, classes by w x h, sorts by name,
                                             # samples each class with a fixed stride; writes manifest-bw.tsv,
                                             # jpeg-sources.tsv, lfs-include.txt
python3 -I derive_set.py <checkout> <out>    # copies the files to files/<class>/, verifies each sha256 against the
                                             # catalogue, derives the J variants with Pillow, writes manifest.tsv
```

The selection is deterministic (sorted names, stride sampling, no seed). The derived files' bytes depend on the JPEG
encoder build Pillow links, so a re-derivation elsewhere can give other bytes: the `sha256` column identifies the files
the benchmark used. The 60 derived files (29.6 MB) are not in this repository; when they are wanted here they go in as
an image set of their own (`corpus/image/<set>.jsonl` with the files under `corpus/images/<set>/` and the licence copy),
as `images check` requires.

## Licences

mmmu and docvqa are Apache-2.0 (`corpus/licenses/mmmu-LICENSE`, `corpus/licenses/docvqa-LICENSE`); the derived JPEGs
are re-encodings of mmmu and docvqa images under the same licence.
