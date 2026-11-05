"""TextVQA's validation images as an image set: the photographs its questions read text from.

The data is the TextVQA dataset on the Hugging Face Hub at a pinned commit, the ``lmms-lab`` formatting of it: three
parquet files of the validation split, each row a question with its photograph in the ``image`` column, stored as the
datasets library stores images (the bytes and a path). Several questions are asked about one photograph, so the bytes
of a photograph recur across rows; every image cell gives a line in file and row order, and a cell whose bytes repeat
an earlier image's is left out and named, as the text imports name their repeats. The files are read as data
(``image_sets.parquet_images``); nothing in them runs.

The dataset was published as ``lmms-lab/textvqa`` and the Hub now redirects that id to ``lmms-lab-encoder/textvqa``,
which the import pins, so that the source reads as the Hub spells it; the files and the commit are the same.

The dataset card states no licence in its front matter. TextVQA's authors publish the dataset under the Creative
Commons Attribution 4.0 International licence: their own Hub card, ``facebook/textvqa``, states ``cc-by-4.0``, and
textvqa.org says the same. The import records CC-BY-4.0 and, as the formatting ships no licence file, copies the
licence text as the SPDX license list publishes it, from that list's repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "textvqa"
REPO = "lmms-lab-encoder/textvqa"
REVISION = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256; its front matter states no licence.
CARD_SHA256 = "3202c889cd037b73067968322c89627a985aedf224a20958a839759b2bba4ad4"
LICENSE = "CC-BY-4.0"
CARD_LICENSE = None  # the reviewed card's YAML front matter has no license key; LICENSE is the authors' statement
# The licence text as the SPDX license list publishes it (text/CC-BY-4.0.txt), from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CC-BY-4.0.txt"
LICENSE_SHA256 = "d557539df68e771cc1eedcc91d13f70fca930e508d11eedcafa4b15db49e3744"
LICENSE_COPY = "licenses/textvqa-LICENSE"
SPLIT = "validation"
COLUMNS = ("image",)
# The validation split's files, by the sha256 of their bytes, in the order the lines are built.
FILES = {
    "data/validation-00000-of-00003.parquet": "491649b5b8add2c5bde13a1e36707002fcd11898699255d65e2e568ac4fbddff",
    "data/validation-00001-of-00003.parquet": "909ad12a378217431548a7f4b66fda163a6def9d21e70844c96449e64206c2dd",
    "data/validation-00002-of-00003.parquet": "a770d03ec95fef58def04cda949d97669b7666caaf174af927856d75684e6fe6",
}


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the Creative Commons Attribution 4.0 International licence.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["Creative Commons Attribution 4.0 International"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the CC BY 4.0 licence text; review it")


def origin(file: str, row: int, column: str) -> dict:
    """Where a line came from, in the order the text imports write their origins: the dataset, its source, the file
    the bytes were read from with its sha256, the row and the column, and the licence."""
    return {
        "dataset": DATASET,
        "source": SOURCE,
        "sha256": FILES[file],
        "file": file,
        "row": row,
        "column": column,
        "license": LICENSE,
    }


def read(files: dict[str, Path]) -> Iterator[image_sets.Found]:
    """Every image cell of every row of the split's files, in the order of ``FILES`` and the rows."""
    for file in FILES:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            yield image_sets.Found(data, origin(file, row, column), f"TextVQA {SPLIT} row {row}, {column}")


def fetch(filename: str, sha256: str, cache: Path) -> Path:
    """The path of a dataset file at the pinned commit, checked against ``sha256``."""
    return hf.fetch(REPO, REVISION, filename, sha256, cache)


def run(args: argparse.Namespace) -> int:
    hf.check_card_license(REPO, REVISION, CARD_SHA256, CARD_LICENSE, args.cache)
    copy = args.corpus / LICENSE_COPY
    license_text = github.fetch(
        LICENSE_OWNER, LICENSE_REPO, LICENSE_COMMIT, LICENSE_PATH, LICENSE_SHA256, cache=args.cache, copy=copy
    ).read_bytes()
    check_license(license_text)
    files = {name: fetch(name, sha256, args.cache) for name, sha256 in FILES.items()}
    copies = {LICENSE_COPY: license_text}
    if args.check:
        problems, _ = image_sets.check(DATASET, read(files), args.corpus, copies)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the TextVQA set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
