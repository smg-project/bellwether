# Image sets

Design for the image side of bellwether, written 2026-10-13. It records what was decided and why; the pull requests
that follow it are reviewed against it. It replaces #128, which put the whole image suite in one change: a catalogue
built outside the repository, references imported from a recorder's output, a verifier that needed an smg checkout
and a list of smg's known differences. This design keeps that data and lands it the way the text sets landed: an
importer per dataset that `--check` reproduces, `record` writing references with their provenance, the case schema
over every line, tables CI compares with the files, and the comparison with SMG left to SMG's own tests.

## What this is for

SMG decodes the images of a chat request and runs the checkpoint's vision processor itself, before the engine sees
them; the engine does the same in its own serving path. Does SMG turn an image into the same pixels, bit for bit: the
decoded RGB bytes, and the tensor the engine hands its model? The text side asks the same question of prompt tokens,
and bellwether answers both the same way, without a device: it holds the inputs (a corpus of images from public
benchmarks, each with its provenance and licence), the expected values (one reference per processor family, recorded
from the engine's own image path), and how both were produced.

The consumer in smg is the multimodal crate's own test: it reads the references, runs its pipeline over the same
images and compares, as the tokenizer crate's bellwether test reads the text fixtures. Nothing here builds or runs
smg, pins an smg commit, or lists what smg gets wrong; that list belongs with the consumer, where it is fixed.

## Vocabulary

| Term | Meaning |
|---|---|
| image | one file of the corpus, its bytes exactly as its dataset ships them, with its line in an image set |
| image set | one JSON Lines file per dataset under `corpus/image/`, one line per image, written by `bellwether import <dataset>` and never edited by hand |
| class | what a line's header fields say an image is (its dataset, container format, colour representation and flags), which the tiers are picked by |
| tier | a list of image paths over every set: `corpus-50`, one image of each of the most distinct classes, and `corpus-500`, every class at least once |
| family | the checkpoints whose image preprocessing inputs are equal (the processor configuration, and the processor's code where the checkpoint ships its own), recorded once under one slug: the image side's checkpoint group |
| reference | what the engine's own image path gives for an image: its media loader, then its multimodal processor, reached through the engine's render server inside the engine's official image; `reference.source` is `engine:vllm` |
| cross-check | transformers' processor run behind the same loader semantics, outside the engine: a tool whose agreement with the reference is evidence, from which nothing is recorded |

The text side has two sources of truth, the checkpoint's own files and vLLM at a pinned release, and records
Hugging Face's result as the reference. An image has no vendor-shipped reference of that kind: the processor is a
class of transformers (or the checkpoint's own code), and the engine's path is its loader followed by that class. The
maintainer's decision (2026-10-12) is that the reference is the engine's own image path, recorded inside the engine's
image, and that a processor run outside the engine is the cross-check. The first recording found the two equal on
every image of the corpus in each of the three families (19,556 of 19,556: the images as #128 held them, before
duplicate bytes were left out), which is the evidence that the engine's path adds nothing of its own to the
processor; the cross-check stays a tool so that a later engine release is checked the same way.

## Which datasets

Public benchmark datasets with images in them, each pinned by the revision of its files and read as data: the parquet
files with pyarrow, the zip with the standard library; nothing from a dataset runs. The counts are after duplicate
bytes within a dataset are left out.

| Dataset | Split | Source | Licence as the source states it | Images | Bytes |
|---|---|---|---|---:|---:|
| MMMU | validation | `hf:datasets/MMMU/MMMU@876ce5cb` (30 subject files, columns `image_1` to `image_7`) | apache-2.0 on the dataset card | 959 | 340.6 MB |
| DocVQA | validation | `hf:datasets/lmms-lab/DocVQA@539088ef` | apache-2.0 on the dataset card | 1,285 | 832.7 MB |
| COCO | val2017 | `images.cocodataset.org/zips/val2017.zip`, pinned by its ETag and sha256 | the images under their photographers' Flickr terms of use, the annotations CC BY 4.0 (cocodataset.org/#termsofuse) | 5,000 | 814.7 MB |
| TextVQA | validation | `hf:datasets/lmms-lab/textvqa@9c0699cd` | the dataset card states no licence (TextVQA itself: CC BY 4.0) | 3,166 | 916.4 MB |
| OCRBench | test | `hf:datasets/echo840/OCRBench@92a54bd1` | the dataset card states no licence | 930 | 75.2 MB |
| ChartQA | test | `hf:datasets/HuggingFaceM4/ChartQA@b605b6e0` | gpl-3.0 on the dataset card | 1,509 | 64.8 MB |
| ScienceQA | test | `hf:datasets/derek-thomas/ScienceQA@f18b0a70` | cc-by-sa-4.0 in the card's front matter; its licensing section states CC BY-NC-SA 4.0 (question 3) | 1,799 | 108.5 MB |
| derived | | made in this repository from COCO photos (below) | as COCO | 4,833 | 1,070 MB |

Together 19,481 images, 4.2 GB. The order is the order of delivery: the datasets whose licence is stated and
permissive first. The rules the text corpus follows apply: a dataset's licence is checked on every import against the
card's front matter (`importers/hf.py`), its licence file or terms are copied to `corpus/licenses/`, and a set built
from copyleft input is kept in files of its own with a `-copyleft` suffix, so it can be dropped or handled apart
(`chartqa-copyleft`, as the pylint rows of SWE-bench are kept). Four of the eight wait for the maintainers' word
before their importer lands: ChartQA under the GPL; TextVQA and OCRBench, whose cards state nothing; and ScienceQA,
whose card contradicts itself (question 3). COCO's images are under their photographers' terms, which the table
states; that too is in the question.

**The derived classes.** The datasets lack most of what a media pipeline gets wrong, so 4,833 images are made from
COCO photos, in 51 variants: WebP lossy (quality 85), lossless, with alpha and animated; PNG in RGB, 8- and 16-bit grey,
16-bit RGB, palette, palette with tRNS, RGBA, LA, interlaced and APNG; JPEG grey, progressive, 4:4:4, 4:2:2 and quality
100; CMYK with and without the APP14 marker; YCCK 4:2:0 and 4:4:4; EXIF orientations 2 to 8 on JPEG, WebP and PNG
(`eXIf`); tiny (shorter side 16 to 111 px) and huge (longer side 4,096 to 5,200 px); GIF. They are made by
`bellwether derive`, which runs in a container pinned by digest (`docker/derive/Dockerfile`: `python:3.12-slim`,
Pillow 12.3.0 installed by hash, libjpeg-turbo for YCCK), by hand, as the tau2 tools are built; CI does not derive
again. Each derived line's `origin` names the COCO image it was made from, the variant, the parameters and the
container, and `derive --check` compares the committed set with its sources and each file's sha256.

## The corpus

```
corpus/
  image/
    <dataset>.jsonl             one line per image, written by `bellwether import <dataset>`
    derived.jsonl               written by `bellwether derive`
    tiers/<tier>.txt            the image paths of a tier, one a line, written by `bellwether tiers`
  images/<dataset>/<name>.<ext> the bytes, exactly as stored, in Git LFS; a clone holds their pointers
  licenses/<dataset>-LICENSE    as for the text sets
```

A line is written in the order its fields are built, non-ASCII raw, as every corpus line is:

```json
{"name": "mmmu-00001-ed5d4d14", "path": "images/mmmu/mmmu-00001-ed5d4d14.png", "bytes": 67395,
 "sha256": "ed5d4d1416700c55309d59fa3a0de04a858ebfd636a40f9d0e342acf034c574e", "format": "png", "w": 733, "h": 237,
 "color": "rgb-alpha", "bits": 8, "transparency": true, "orientation": null, "progressive": false, "subsampling": null,
 "animated": false, "notes": "MMMU Accounting validation row 0, image_1",
 "origin": {"dataset": "mmmu", "source": "hf:datasets/MMMU/MMMU@876ce5cb130f7f7e290ce4d9984357737d4db5cf",
            "sha256": "f1bedfc8ad70d24f8c1daa50a309d9b3b88ea14caea60299b648c69049fa870f",
            "file": "Accounting/validation-00000-of-00001.parquet", "row": 0, "column": "image_1", "license": "Apache-2.0"}}
```

- **Names** are slugs, `<dataset>-<nnnnn>-<sha8>`: the dataset, the image's position among the kept images in the
  order the files, rows and columns are read, and the first eight hex digits of its sha256. A derived image is
  `<variant>-<source name>`. The name is the last part of a fixture id, so it carries no extension; `path` does.
- **The catalogue comes from the file's own headers**, not from a decoder: the container format from the magic bytes;
  width and height as stored; the colour representation (`ycbcr`, `gray`, `cmyk`, `ycck` or `rgb` for JPEG from its
  components and Adobe marker; `gray`, `rgb`, `palette`, `gray-alpha` or `rgb-alpha` for PNG from its colour type; `rgb`
  or `rgb-alpha` for WebP; `palette` for GIF); the bit depth; whether the file carries transparency (an alpha channel,
  a `tRNS` chunk, a GIF transparent index, WebP's alpha flag); the EXIF orientation where the file has one (JPEG's
  `APP1`, PNG's `eXIf`, WebP's `EXIF` chunk); progressive JPEG and its chroma subsampling; animation (`acTL`, `ANIM`,
  more than one GIF frame). What the engine's loader makes of the file, the decoded size after orientation among it,
  is the reference's to say. The importers stay within the standard library and pyarrow.
- **Repeats.** A row whose image bytes equal an earlier image's of the same dataset gives no line, and the import names
  it with the image it repeats, as the text imports name theirs. A row with no image, or bytes that are not an image
  format the catalogue reads, is named with its reason.
- **`--check`** builds the import again from the pinned files and compares it line for line with the committed set,
  and each committed file with its line: a file Git LFS has not fetched is a pointer, which names its content's sha256
  and size, so the comparison needs no download; a fetched file is hashed. A file under `corpus/images/<dataset>/`
  that no line names is a problem, as a stale set file is for the text imports. The import's downloads are large (the
  COCO zip 815 MB, the parquet files 3.3 GB in all), so this check runs weekly and when an importer changes, not on
  every pull request.
- **`bellwether images check`** is the check every pull request runs: every line of every set names a file in the
  checkout whose pointer or content has the line's sha256 and size, no file is unnamed, names are unique across the
  sets, every tier names set lines, and the licence copies exist. No download, a few seconds.
- **Tiers.** `bellwether tiers` writes `corpus-50` and `corpus-500` from the committed sets alone, deterministically:
  the classes are read from the lines, `corpus-500` takes every class at least once and the rest in proportion to the
  class's size by strided picks, `corpus-50` one image of each of the most distinct classes; `--check` compares. A
  consumer fetches a tier's bytes with `git lfs pull --include` over its paths and `--exclude ''`, as a fixture set is
  fetched.

## Families

```
families/
  <slug>/
    family.toml                 the family: its checkpoints, the model and revision its inputs were read from, each
                                input with its sha256, the engine's processor classes
    sets.toml                   one table per recorded set, as fixtures/<slug>/sets.toml
    image/<set>.jsonl.zst       one line per image of the corpus set, in Git LFS
```

A family is to images what a checkpoint group is to text: its members render an image identically because they read
the same inputs, so it is recorded once. The inputs are the processor configuration (`preprocessor_config.json`) and,
where the checkpoint ships its own processor code, that code (MiniMax-M3's `image_processor.py`, named by the
configuration's `auto_map`) and the parts of `config.json` it reads. `family.toml` pins them by sha256 as a manifest
pins its tokenizer files, and names the model and revision they were read from; whether the files are also committed
under the family, or read from the Hub at the revision as the oracle inputs are, is a question below.

```toml
model    = "Qwen/Qwen2-VL-7B-Instruct"
revision = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
checkpoints = ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]

[engines]
vllm = { processor = "Qwen2VLMultiModalProcessor", hf_processor = "Qwen2VLProcessor", image_processor = "Qwen2VLImageProcessor" }

[inputs]
"preprocessor_config.json" = "b5eaad0c2815f07631535dcc58f3c462b0d73693638ad21d19f3c50820eae1cc"
```

The three families to record first: `qwen2-vl` (Qwen2-VL and Qwen2.5-VL, every size: the same preprocessing values,
patch 14, CLIP mean and standard deviation, 3,136 to 12,845,056 pixels), `qwen3-vl` (every Qwen3-VL size, Instruct and
Thinking: patch 16, mean and standard deviation 0.5, 65,536 to 16,777,216 pixels) and `minimax-m3` (the checkpoint's
own processor code). As for text, an `[smg]` table may name the processor SMG's crate selects, for the consumer's
convenience; it is optional and changes nothing recorded.

## Recording

`bellwether record --kind image --oracle vllm --family <slug>` runs inside the engine's official image at the
repository's pin, `vllm/vllm-openai:v0.31.0` by digest (`sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`),
the same image and release the text side cites; it needs no device. The oracle is the engine's render server,
`vllm launch render <model>`, started with the family's model files at the pinned revision (configuration, processor and
tokenizer files, no weights), through the route the text side already uses, with no code of the engine's copied into
bellwether. Citations are file:line in vLLM v0.31.0 (db9527a4).

- **The request.** `POST /v1/chat/completions/render` takes a chat completion request; the recorder sends one user
  message with the image as an `image_url` content part, a `data:` URL of the file's bytes, and no text beyond what the
  template needs. The server renders it as `vllm serve` would: the chat parser hands the URL to the media connector
  (`vllm/entrypoints/chat_utils.py:1077-1078`, `vllm/multimodal/media/connector.py:494-513`), whose image loader is
  `ImageMediaIO.load_bytes` (`vllm/multimodal/media/image.py:79-92`): PIL opens the bytes, the EXIF orientation is applied
  with any failure suppressed (`normalize_image`, `vllm/multimodal/image.py:21-25`), the image is loaded and converted to
  RGB, an image with transparency (RGBA, LA, PA, or a `transparency` key from a `tRNS` chunk) composited over white
  (`convert_image_mode` and `rgba_to_rgb`, `vllm/multimodal/image.py:28-60`). The multimodal processor the engine
  builds for the checkpoint then runs over the loaded image, as it does in serving.
- **The answer** is the engine's token-in request (`GenerateRequest`,
  `vllm/entrypoints/scale_out/token_in_token_out/protocol.py:145-176`): the prompt token ids with the image's placeholder
  tokens in them, and `features` (`MultiModalFeatures`, `protocol.py:59-80`): the item's hash, its placeholder range,
  and `kwargs_data`, the processor's output for the item, `pixel_values` and `image_grid_thw`, serialized by the
  engine's own encoder (`mm_serde.py:17-26`: msgpack with the tensors inline, then base64). The render route builds it
  in `vllm/entrypoints/scale_out/render/serving.py:76-135`. The recorder decodes it with the engine's own decoder, which
  is why it runs inside the image rather than from bellwether's environment as the text client does.
- **What a line holds.** `id` is `<family>/image/<name>`, `kind` is `image`, `model` the family's model, `image` the
  corpus line's `path` and `sha256`, and `reference`: `source` `engine:vllm`; `w` and `h`, the decoded size after
  orientation; `grid`, the image grid the processor reports (`t, h, w`); `shape` and `dtype` of the pixel tensor;
  `rgb_sha256`, the loader's RGB bytes, read back from the engine as the processor received them; and the sha256 of
  the tensor the engine hands its model, in that tensor's own dtype, under `u8_sha256`, `bf16_sha256` or `f32_sha256`
  (one of the three, and `f32_sha256` with `bf16_sha256` beside it when the tensor is float32, the bits rounded to
  nearest even). The first in-image recording found uint8 patches for the `qwen2-vl` family and bfloat16 for the
  other two; the official image's answer is recorded as it comes and the line says which. `provenance` carries the
  image digest, the versions of vLLM, torch, transformers, Pillow and numpy inside it, the processor classes, the
  prompt, the model's revision and bellwether's version. Lines are sorted by id and canonical, as `write_fixture_file`
  writes every fixture, and each set's table in `sets.toml` gives its form, cases, rejected cases, plain size and
  plain-content sha256, so a re-record is reviewed through the table and a reviewer fetches nothing.
- **Rejections are reported.** An image the engine refuses (a 400 with its error object) is counted as rejected with
  its reason, as the text oracles count a case a template refuses; an answer that is not a 200 or a refusal measures
  nothing and stops the run, as `engines/vllm_render.py` distinguishes the two.
- **The cross-check** is a tool beside the recorder: transformers' processor at the version the image ships, behind
  the loader semantics copied from the lines cited above and self-tested against them; its agreement with the
  recorded set is reported with each recording and nothing is recorded from it.
- **The schema.** The case schema gains the kind `image`, an `image` object and the reference fields above; a rule
  requires an image line to carry its RGB hash and one tensor hash. The change is the sponsor's, as every change to
  the case schema is, and is its own pull request so that the rest of the series does not wait on it.

## Storage

- `corpus/images/**` and `families/**/*.jsonl.zst` are in Git LFS (`.gitattributes`), and `.lfsconfig` excludes both
  from a default fetch, as it excludes the fixture sets: a clone holds a 130-byte pointer per image, 2.5 MB for the
  corpus, and a consumer fetches a tier or a family by path. The set files under `corpus/image/` stay plain, as the
  text imports under the limit do: 19,481 lines, about 10 MB, so a change to an importer reads as a diff of lines.
- The images take 4.2 GB in LFS once; a re-import of the same pinned files writes the same bytes, which Git LFS stores
  once by content. The reference sets of a family take about 1.5 MB compressed over the whole corpus; a re-record adds
  as much again.
- Every image of the corpus is already in the repository's LFS store, uploaded for #128; the series references the
  same objects by their content id and uploads nothing twice.

## Consumers

`bellwether unpack --family <slug>` writes `<out>/<slug>/image/<set>.jsonl` with the family's `family.toml` and
`sets.toml` beside it, the plain tree a consumer points at, as `unpack` does for a model; the consumer fetches the
images it needs by tier. SMG's test takes the tree, runs its own pipeline over the images and compares, and keeps its
own list of what it is known to differ on. There is no `verify --kind image` here and none is planned: `verify` replays
requests through a running SMG, which the images do not need.

## CI

- **Every pull request** (`test.yml`): `bellwether images check`, `bellwether count --readme --check` with the images
  table, `git lfs fsck --pointers`, ruff and pytest, as today; the sets job checks the family sets a pull request
  changes, fetched in shards, as it checks the fixture sets.
- **Weekly, and when an importer changes** (`corpus-images.yml`, landing with the first importer): each image
  importer's `--check` with its downloads, one job per dataset, the pinned files cached per dataset under their own
  key, since all of them together would not fit the one cache the text imports share.
- **By hand** (`record-images.yml`, `workflow_dispatch`): a container job on the pinned official image on the record
  runners, which starts the render server for the family and runs `record --kind image`, then pushes the new set to
  `record/images/<slug>/<run id>` as `record.yml` does for text; a pull request is made from it afterwards.

## Delivery

Each step is its own pull request, one concern each; a step depends on an earlier one only where it says so.

1. This design.
2. The importers' shared machinery (the header catalogue, the readers, the names, the writer and `--check`,
   `bellwether images check`, the images table of `count`, the weekly `corpus-images.yml`), with the MMMU images.
3. to 8. One importer per dataset, in the table's order: DocVQA, COCO, TextVQA, OCRBench, ChartQA as
   `chartqa-copyleft`, ScienceQA; the last five after the maintainers' word on their licences.
9. `bellwether derive`, its container and the derived images; after COCO.
10. `bellwether tiers` and the two tiers; after the sets.
11. The case schema's `image` kind, for the sponsor.
12. `record --kind image --oracle vllm` through the render server, `unpack --family`, the sets check over
    `families/**`, the cross-check tool; tested against a loopback stand-in for the render server; after 11.
13. The references of the three families over the whole corpus, recorded by 12 inside the official image; data
    only; after 10 and 12.
14. The record workflow; after 12.
15. The README's image side: vocabulary, layout, recording, consumers.

## Questions for the maintainers

1. **Layout.** Inputs under `corpus/image/` and `corpus/images/`, results under `families/<slug>/image/`, mirroring
   `corpus/` and `fixtures/` with `family.toml` beside `sets.toml` as `manifest.toml` is; or the results under
   `fixtures/` as well, which would put a family's directory beside the checkpoints'.
2. **The case schema** gains the kind `image` (step 11), or the image line gets a schema file of its own.
3. **Licences.** ChartQA's card says gpl-3.0 (kept apart as `chartqa-copyleft`); the TextVQA and OCRBench cards state
   no licence; COCO's images are under their photographers' Flickr terms, its annotations CC BY 4.0; and ScienceQA's
   card contradicts itself: at `hf:datasets/derek-thomas/ScienceQA@f18b0a70359ebfb41f658fd564208d0355b013f4` its
   `README.md` (sha256 `3729fae304968ed8b9917b41034412aa142fb4760e2bfdab3dfe67a7fd78a17a`) opens its front matter
   with `license: cc-by-sa-4.0`, and its section "Licensing Information" reads "Attribution-NonCommercial-ShareAlike
   4.0 International (CC BY-NC-SA 4.0)", linking creativecommons.org/licenses/by-nc-sa/4.0/. The front-matter check
   alone would pass it, so it waits with the other four. Which of these five sets ship; the others land as the table
   orders them.
4. **The processor inputs** (`preprocessor_config.json`; MiniMax-M3's `image_processor.py`): pinned by sha256 in
   `family.toml` and read from the Hub at the revision, as the oracle inputs of a manifest are, or committed under the
   family as #128 did.
5. **What stays out** unless asked: the cross-check sets, the raw arrays of the smoke tier (1.6 GB for fifty images;
   the recorder writes them on request), the synthetic sets made by another project's scripts, and the clips, for
   which no reference exists yet.
