# The multimodal suite

Does SMG's media pipeline turn an image into the same pixels the serving engine's own image path gives, bit for bit:
the decoded RGB bytes, the u8 patches, the bf16 and the f32 tensors the engine receives? This suite answers that
without a GPU, as the text fixtures answer it for prompt tokens: a public-benchmark corpus of images, one reference per
processor family recorded inside the engine's own image on CPU, a cross-check from the vendor's processor outside the
engine, and `bellwether multimodal`, the commands that keep them consistent and compare SMG with them.

The images are inputs and the references are results, as `corpus/` and `fixtures/` are for text; they live together
here because a reference belongs to a processor family (every checkpoint whose image preprocessing inputs are equal)
and not to one checkpoint. Everything larger than a table is in Git LFS, and a clone holds pointers until a tier is
fetched.

## Vocabulary

| Term | Meaning |
|---|---|
| image | one file of the corpus, as its dataset ships it (bytes as stored), with its row in `manifest.tsv` |
| class | what the catalogue says an image is: dataset, container format and colour representation, or the derived variant (`coco/jpeg/ycbcr`, `derived/exif6-webp`) |
| tier | a list of image paths: `corpus-50` (the smoke set, every gate), `corpus-500` (every class at least once), `all` |
| family | the checkpoints whose image preprocessing inputs are equal, recorded once under a slug: `qwen2-vl`, `qwen3-vl`, `minimax-m3` |
| reference | what the engine's own image path gives for an image: its media loader, then its multimodal processor, run inside the engine's image on CPU (`references/<family>/vllm.jsonl.zst`) |
| cross-check | the vendor's processor (transformers at the engine's pinned version) behind the same loader semantics, run outside the engine (`references/<family>/hf.jsonl.zst`); a difference from the reference is a loader or processor detail of the engine's, recorded as a finding |
| dump | what SMG's media pipeline gives for the same images, one line per image, from the dump example |
| known difference | an image class where SMG is known to differ from the reference, listed in `known.toml` with its reason and where it is tracked; counted apart, never as parity |

## What a reference holds

One JSON line per image, keys sorted: `name` (the file name every recorder and dump keys by), `path` (from this
directory), `w` and `h` (the decoded size, after EXIF orientation), `mode` (the file's mode as opened), `transparent`
(whether it had transparency; null where the recorder did not say), `grid` (the image grid the processor reports, as
`t, h, w`), `shape` (the pixel tensor's shape), `rgb_sha256` (the loader's RGB bytes), `u8_sha256` (the un-normalized u8
patches; null where the engine's path exposes none), `bf16_sha256` (the f32 pixel values rounded to bf16, nearest even)
and `f32_sha256` (the f32 pixel values). An image the recorder could not process is `{"name", "path", "error"}`.
Each family's `sets.toml` lists every set with its form, cases, errors, plain size and sha256, the stored object's
sha256 and size (so `check` compares the Git LFS pointer without a fetch) and the recorder's provenance (library
versions, processor classes, prompt). The raw arrays of the corpus-50 tier (`raw/<name>.{rgb,u8,norm}.npy`) sit beside
the sets for the first differing element.

## Layout

```
multimodal/
  README.md               this file; the tables between the markers are written by `bellwether multimodal count`
  manifest.tsv            one row per still image: id, path, source, split, group, variant, class, format, w, h,
                          size_class, color, exif_orientation, progressive, subsampling, alpha, bits16, animated, tiny,
                          bytes, sha256, note
  sources/<dataset>.tsv   where each image came from: the download file, row and column (or the derivation and its parameters)
  tiers/<tier>.txt        the image paths of a tier, one a line
  images/<dataset>/...    the images, bytes as stored (Git LFS); the derived classes under images/derived/<variant>/
  clips/                  seven video clips (Git LFS) and clips.tsv
  families/<slug>/        family.toml (SMG processor, hub model and revision, the engine's classes, sha256 per input)
                          and the inputs: preprocessor_config.json, and for minimax-m3 its image_processor.py and config.json
  references/<slug>/      vllm.jsonl.zst (the reference), hf.jsonl.zst (the cross-check), sets.toml, raw/ (corpus-50)
  known.toml              the known differences
  tools/                  record_vllm.py, oracle_hf.py, smg_dump.rs, and the corpus build (extract, derive, catalogue, manifest, import_corpus)
```

## Commands

```bash
uv run bellwether multimodal check                                  # the tables agree with the files; no fetch (CI, every pull request)
uv run bellwether multimodal count --readme multimodal/README.md    # the tables below (--check compares, as CI does)
uv run bellwether multimodal fetch --tier corpus-50                 # the tier's images, raw arrays and reference sets from Git LFS
uv run bellwether multimodal verify --tier corpus-50 --smg-dump <bellwether_dump binary> --report runs/mm.json --junit runs/mm.xml
uv run bellwether multimodal cross-check                            # the reference against the cross-check, per family
uv run bellwether multimodal import-references --family qwen3-vl --source vllm --from rec.jsonl --raw-from raw
```

`verify` runs the dump binary over the tier for each family (`SMG_DUMP_LIST`, `SMG_DUMP_OUT`, `SMG_DUMP_PROCESSOR` from
`family.toml`, `SMG_DUMP_CONFIG` the family's preprocessor config) and compares every image with the family's
reference: the decoded RGB bytes, the patch shape, the u8 patches where the reference recorded them, the bf16 and the f32
wire bytes. Verdicts are `match`, `difference` (which artifacts, in the order rgb, shape, u8, bf16, f32), `known`,
`error` and `missing`. Exit 0 when every image matches or is known, 1 when one differs, 2 when the run gives no verdict.
`--dump FAMILY=PATH` reads a dump already made; `--source hf` compares with the cross-check instead.

## The SMG side

`tools/smg_dump.rs` is the protocol's reference implementation: an example of SMG's multimodal crate that decodes each
listed file the way the gateway decodes inline image bytes (orientation applied, no cap), runs the family's processor,
writes the bf16 and f32 wire bytes as the gateway writes them, and prints one JSON line per image with the size, the
grid, the shape and the four sha256 digests (`hash` mode), or the raw artifacts (`raw` mode). It uses the crate's public
API alone and carries its own SHA-256, so it builds against any smg checkout as
`cargo build --release -p llm-multimodal --example bellwether_dump` once copied to
`crates/multimodal/examples/bellwether_dump.rs`; the `multimodal` workflow does that until smg ships the example. An
smg test like the tokenizer's bellwether test (reading `BELLWETHER_MULTIMODAL`, skipping without it) is the follow-up on
smg's side.

## Families

| Family | SMG processor | Inputs taken from | Checkpoints |
|---|---|---|---|
| `qwen2-vl` | `qwen2_vl` | Qwen/Qwen2-VL-7B-Instruct (patch 14, CLIP mean and std, 3136 to 12845056 pixels) | Qwen2-VL and Qwen2.5-VL, every size: their preprocessing values are equal |
| `qwen3-vl` | `qwen3_vl` | Qwen/Qwen3-VL-235B-A22B-Instruct @ 710c1386 (patch 16, mean and std 0.5, 65536 to 16777216 pixels) | Qwen3-VL, every size, Instruct and Thinking |
| `minimax-m3` | `minimax_m3` | MiniMaxAI/MiniMax-M3 @ f0e1c1e0: preprocessor_config.json, image_processor.py (the checkpoint's own code), config.json | MiniMax-M3 |

## CI

- Every pull request (`test.yml`): `check` and `count --readme --check`, with no Git LFS fetch.
- `multimodal.yml`: a pull request that touches the suite or its commands verifies `corpus-50` against the dump example
  built from the smg commit in `.github/versions/smg.ref`; the schedule verifies `corpus-500` every night and `all`
  once a week; a dispatch names its tier. The tier's Git LFS objects are cached by their pointers.
- `multimodal-record.yml`: records a family's reference again inside the pinned engine image (a container job on the
  bellwether-record runners, CPU) and pushes a `record/multimodal/<run id>` branch, for when the engine pin moves or a
  family is added.

## How the suite was made (the regeneration recipe)

1. **Downloads**, each dataset into its own empty directory (nothing from a dataset is executed; the parquet files are
   read as data, the COCO zip as data): COCO val2017 from `images.cocodataset.org/zips/val2017.zip`; the parquet files
   of the hub datasets by their resolve URLs at the revisions in the table below (`lmms-lab/textvqa` validation,
   `lmms-lab/DocVQA` validation, `echo840/OCRBench` test, `HuggingFaceM4/ChartQA` test, `MMMU/MMMU` validation,
   `derek-thomas/ScienceQA` test).
2. **Extract** (`tools/extract.py <dataset>`): the images written out exactly as stored, the extension from the magic
   bytes, deduplicated by sha256 within a dataset, one `sources/<dataset>.tsv` each.
3. **Derive** (`tools/derive.py`, Pillow 12.3.0, the engine stack's version; YCCK through libturbojpeg's `tjCompress2`
   with `TJPF_CMYK`): the derived classes from COCO photos, deterministic subsets by sorted name: WebP (q85, lossless,
   RGBA, animated), PNG (RGB, gray8, gray16, rgb16, palette, palette with tRNS, RGBA, LA, interlaced, APNG), JPEG (gray,
   progressive, 4:4:4, 4:2:2, q100), CMYK (Pillow, APP14 stripped), YCCK (4:2:0, 4:4:4), EXIF orientation 2 to 8 on JPEG,
   WebP and PNG eXIf, tiny (short side 16 to 111) and huge (long side 4096 to 5200), GIF.
4. **Catalogue and manifest** (`tools/catalogue.py`, `tools/manifest.py`): per file the format, size, colour
   representation, EXIF orientation, progressive flag, chroma subsampling, alpha, bit depth, animation and size class;
   the class labels; the tiers (corpus-50: one image from each of the most distinct classes; corpus-500: every class at
   least once, the rest proportional to the class size, deterministic strided picks).
5. **Lay out** (`tools/import_corpus.py <corpus> multimodal`): the paths made relative, the project's own synthetic
   sets renamed `synthetic-*`.
6. **The reference** (`tools/record_vllm.py`, inside `vllm/vllm-openai:v0.31.0`, index digest
   `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`, CPU only, no device, no network; the
   family's model files from the Hub at the revision `family.toml` pins: configs and tokenizer, no weights):
   `python3 -I tools/record_vllm.py --model-dir <model dir> --list <tier paths> --out rec.jsonl [--raw-out raw]
   [--trust-remote-code]`, then `bellwether multimodal import-references --family <slug> --source vllm --from rec.jsonl
   --raw-from raw`. The engine's path: `ImageMediaIO.load_bytes` (PIL open, `exif_transpose` with errors suppressed,
   load, conversion to RGB that composites images with transparency over white), then
   `MULTIMODAL_REGISTRY.create_processor(ModelConfig(model dir)).apply(the family's image prompt, {"image": [img]}, {})`,
   whose `pixel_values` and `image_grid_thw` are hashed. The `multimodal-record` workflow runs these steps.
7. **The cross-check** (`tools/oracle_hf.py`, in an environment with the engine stack's versions: vllm 0.31.0,
   transformers 5.17.0, torch 2.13.0, torchvision 0.28.0, Pillow 12.3.0, numpy 2.3.5): the same loader semantics, copied
   from the engine and self-tested against it (`--selftest`), then transformers' processor from the family directory
   (MiniMax-M3: its `image_processor.py` imported by path); imported with `--source hf`.

The catalogue of this corpus, as built:

| dimension | counts |
|---|---|
| format | gif 20, jpeg 11737, png 6911, webp 888 |
| size class (longer side) | s1280 5932, s2000 272, s320 1178, s4000 1187, s4000p 62, s640 10925 |
| colour representation | cmyk 407, gray 1571, gray-16bit 60, gray-alpha 60, palette 125, palette+trns 63, rgb 3967, rgb-16bit 60, rgb-alpha 2060, ycbcr 11023, ycck 160 |
| EXIF orientation | - 17382, 1 39, 2 300, 3 300, 4 300, 5 300, 6 335, 7 300, 8 300 |
| JPEG chroma subsampling | 4:2:0 4207, 4:2:2 835, 4:4:0 50, 4:4:4 6396, gray 147, other 102 |
| flags | alpha 2183, 16-bit 120, animated 34, progressive JPEG 150, tiny (short side < 112) 653, undecodable 0 |

## Datasets and licenses

The images are redistributed here as the datasets publish them. Each row states the license as the source states it;
two hub cards state none, and COCO's images are under their photographers' Flickr terms. The table is the corpus
manifest's, kept as written when the corpus was built.

| dataset | origin | split | revision | license (as stated by the source) | rows -> unique images | bytes |
|---|---|---|---|---|---|---|
| COCO val2017 | images.cocodataset.org/zips/val2017.zip | val2017 | ETag d366be60d3dc737327160d62453e3973-98, Last-Modified 2018-07-11 | images under the Flickr terms of use, annotations CC BY 4.0 (cocodataset.org/#termsofuse) | 5000 files -> 5000 | 815 MB |
| TextVQA | hub lmms-lab/textvqa (parquet) | validation | 9c0699cd | no license field on the hub card (TextVQA itself: CC BY 4.0) | 5000 rows -> 3166 unique images | 916 MB |
| DocVQA | hub lmms-lab/DocVQA (parquet, DocVQA config) | validation | 539088ef | apache-2.0 on the hub card | 5349 rows -> 1285 unique documents | 833 MB |
| OCRBench | hub echo840/OCRBench (parquet) | test | 92a54bd1 | no license field on the hub card | 1000 rows -> 930 unique images | 75 MB |
| ChartQA | hub HuggingFaceM4/ChartQA (parquet) | test | b605b6e0 | gpl-3.0 on the hub card | 2500 rows -> 1509 unique charts | 65 MB |
| MMMU | hub MMMU/MMMU (parquet, 30 subjects, image_1..image_7) | validation | 876ce5cb | apache-2.0 on the hub card | 900 rows -> 959 unique images | 341 MB |
| ScienceQA | hub derek-thomas/ScienceQA (parquet) | test | f18b0a70 | cc-by-sa-4.0 on the hub card | 4241 rows -> 1799 unique images | 108 MB |
| derived | this suite (tools/derive.py) from COCO photos | - | - | as COCO | 4833 files | 1070 MB |
| synthetic sets | the project's earlier parity sets: the 40-image set, the 30 fixture classes, the CMYK/YCCK extras | - | - | synthetic | 75 files | 11 MB |
| clips | the project's 7 clips (mp4 h264 1 s / 30 s / 600 s / 720p 10 s / with audio 5 s, mov h264 5 s, webm vp9 5 s) | - | - | synthetic | 7 | - |

## Tiers

- `corpus-50`: one image from each of the most distinct classes (photos, documents, charts, text, diagrams, WebP, PNG
  variants, CMYK/YCCK, EXIF orientations on JPEG/WebP/PNG, tiny, huge, animated); the smoke set for every gate, with raw
  arrays beside the references.
- `corpus-500`: every class at least once, the rest proportional to the class size, deterministic strided picks; the
  set for greedy-answer parity on a real engine when decoded pixels change.
- `all`: everything, before a media-pipeline change is accepted.

## What the suite holds

<!-- bellwether multimodal count --readme writes everything from here to the end marker: edit none of it by hand -->

### Images by dataset

| Dataset | Images | Bytes | Classes | In corpus-50 | In corpus-500 |
|---|---:|---:|---:|---:|---:|
| chartqa | 1509 | 64.8 MB | 2 | 1 | 35 |
| coco | 5000 | 814.7 MB | 2 | 2 | 110 |
| derived | 4833 | 1070.1 MB | 51 | 38 | 156 |
| docvqa | 1285 | 832.7 MB | 2 | 0 | 30 |
| mmmu | 959 | 340.6 MB | 5 | 1 | 26 |
| ocrbench | 930 | 75.2 MB | 1 | 1 | 21 |
| scienceqa | 1799 | 108.5 MB | 1 | 1 | 40 |
| synthetic | 75 | 10.9 MB | 8 | 4 | 9 |
| textvqa | 3166 | 916.4 MB | 4 | 2 | 73 |
| all | 19556 | 4233.9 MB | 76 | 50 | 500 |

### Classes

| Class | Images | In corpus-50 | In corpus-500 |
|---|---:|---:|---:|
| chartqa/png/rgb | 159 | 1 | 5 |
| chartqa/png/rgb-alpha+alpha | 1350 | 0 | 30 |
| coco/jpeg/gray | 10 | 1 | 1 |
| coco/jpeg/ycbcr | 4990 | 1 | 109 |
| derived/apng-animated | 10 | 1 | 1 |
| derived/cmyk-noapp14 | 199 | 1 | 5 |
| derived/cmyk-pillow | 200 | 1 | 6 |
| derived/exif1-jpeg | 30 | 1 | 2 |
| derived/exif2-jpeg | 200 | 1 | 6 |
| derived/exif2-png | 50 | 0 | 2 |
| derived/exif2-webp | 50 | 0 | 2 |
| derived/exif3-jpeg | 200 | 1 | 6 |
| derived/exif3-png | 50 | 0 | 2 |
| derived/exif3-webp | 50 | 0 | 2 |
| derived/exif4-jpeg | 200 | 1 | 6 |
| derived/exif4-png | 50 | 0 | 2 |
| derived/exif4-webp | 50 | 0 | 2 |
| derived/exif5-jpeg | 200 | 1 | 6 |
| derived/exif5-png | 50 | 0 | 2 |
| derived/exif5-webp | 50 | 0 | 2 |
| derived/exif6-jpeg | 200 | 1 | 6 |
| derived/exif6-png | 50 | 1 | 2 |
| derived/exif6-png-rgba | 30 | 1 | 2 |
| derived/exif6-webp | 50 | 1 | 2 |
| derived/exif7-jpeg | 200 | 1 | 6 |
| derived/exif7-png | 50 | 0 | 2 |
| derived/exif7-webp | 50 | 0 | 2 |
| derived/exif8-jpeg | 200 | 1 | 5 |
| derived/exif8-png | 50 | 0 | 2 |
| derived/exif8-webp | 50 | 0 | 2 |
| derived/gif-animated | 10 | 1 | 1 |
| derived/gif-static | 10 | 0 | 1 |
| derived/huge-jpeg | 24 | 1 | 2 |
| derived/jpeg-422 | 100 | 1 | 3 |
| derived/jpeg-444 | 100 | 1 | 3 |
| derived/jpeg-gray | 100 | 1 | 3 |
| derived/jpeg-progressive | 150 | 1 | 4 |
| derived/jpeg-q100 | 50 | 1 | 2 |
| derived/png-gray16 | 60 | 1 | 2 |
| derived/png-gray8 | 100 | 1 | 3 |
| derived/png-interlaced | 60 | 1 | 2 |
| derived/png-la | 60 | 1 | 2 |
| derived/png-palette | 100 | 1 | 3 |
| derived/png-palette-trns | 60 | 1 | 2 |
| derived/png-rgb | 300 | 1 | 8 |
| derived/png-rgb16 | 60 | 1 | 2 |
| derived/png-rgba | 100 | 1 | 3 |
| derived/tiny-jpeg | 100 | 1 | 3 |
| derived/tiny-png | 50 | 1 | 2 |
| derived/webp-animated | 10 | 1 | 1 |
| derived/webp-lossless | 150 | 1 | 4 |
| derived/webp-q85 | 300 | 1 | 8 |
| derived/webp-rgba | 60 | 1 | 2 |
| derived/ycck-420 | 100 | 1 | 3 |
| derived/ycck-444 | 50 | 1 | 2 |
| docvqa/png/gray | 1281 | 0 | 29 |
| docvqa/png/rgb | 4 | 0 | 1 |
| mmmu/png/gray | 43 | 0 | 2 |
| mmmu/png/palette | 5 | 0 | 1 |
| mmmu/png/palette+trns+alpha | 3 | 0 | 1 |
| mmmu/png/rgb | 390 | 1 | 10 |
| mmmu/png/rgb-alpha+alpha | 518 | 0 | 12 |
| ocrbench/jpeg/ycbcr | 930 | 1 | 21 |
| scienceqa/png/rgb | 1799 | 1 | 40 |
| synthetic/cmyk-extra | 5 | 1 | 1 |
| synthetic/fixtures-cmyk | 5 | 1 | 1 |
| synthetic/fixtures-exif1 | 5 | 0 | 1 |
| synthetic/fixtures-exif6 | 5 | 1 | 1 |
| synthetic/fixtures-jpeg | 5 | 0 | 1 |
| synthetic/fixtures-png | 5 | 0 | 1 |
| synthetic/fixtures-webp | 5 | 0 | 1 |
| synthetic/parity-40 | 40 | 1 | 2 |
| textvqa/jpeg/gray | 37 | 0 | 2 |
| textvqa/jpeg/ycbcr | 3117 | 1 | 69 |
| textvqa/jpeg/ycbcr+anim | 4 | 0 | 1 |
| textvqa/jpeg/ycck | 8 | 1 | 1 |

### References

Each family's recorded sets, from its `sets.toml`; `cases` counts the images with a result, `errors` those the recorder could not process.

| Family | Source | Cases | Errors | Plain | Stored | Raw arrays | Recorder |
|---|---|---:|---:|---:|---:|---:|---|
| minimax-m3 | hf | 19556 | 0 | 9.8 MB | 2.8 MB |  | transformers 5.17.0, torch 2.13.0+cu130, pillow 12.3.0 |
| qwen2-vl | hf | 19556 | 0 | 9.8 MB | 2.8 MB |  | transformers 5.17.0, torch 2.13.0+cu130, pillow 12.3.0 |
| qwen3-vl | hf | 19556 | 0 | 9.8 MB | 2.8 MB |  | transformers 5.17.0, torch 2.13.0+cu130, pillow 12.3.0 |

<!-- end of what bellwether multimodal count --readme writes -->
