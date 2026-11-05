"""OCRBench's test images as an image set: the pictures of text its questions are asked about.

The data is the OCRBench dataset on the Hugging Face Hub at a pinned commit: one parquet file of the test split, each
row a question with its picture in the ``image`` column, stored as the datasets library stores images (the bytes and a
path). Several questions are asked about one picture, so the bytes of a picture recur across rows; every image cell
gives a line in row order, and a cell whose bytes repeat an earlier image's is left out and named, as the text imports
name their repeats. The file is read as data (``image_sets.parquet_images``); nothing in it runs.

The dataset card states no licence in its front matter and names the benchmark's repository, which publishes
OCRBench under the MIT License (its ``LICENSE.txt``). The import records MIT and copies that file from the repository
at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "ocrbench"
REPO = "echo840/OCRBench"
REVISION = "92a54bd1384387c178d5a07140a2d85e0a3d12e1"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256; its front matter states no licence.
CARD_SHA256 = "af7ac20750b531a259436713e817cf72d2f3cf427e44f8ced4d5d8a54587cae1"
LICENSE = "MIT"
CARD_LICENSE = None  # the reviewed card's YAML front matter has no license key; LICENSE is the repository's
# The benchmark's repository, named on the card, and its LICENSE.txt at the one commit that file has.
LICENSE_OWNER, LICENSE_REPO = "Yuliang-Liu", "MultimodalOCR"
LICENSE_COMMIT = "102a804d770851b20c7fdbc8eec3e5424b99dcba"
LICENSE_PATH = "LICENSE.txt"
LICENSE_SHA256 = "3aac928dc1d29af861e9619a69a0032a9650a05750a8fb7af145d4a3f1a3655b"
LICENSE_COPY = "licenses/ocrbench-LICENSE"
SPLIT = "test"
COLUMNS = ("image",)
# The test split's file, by the sha256 of its bytes.
FILES = {
    "data/test-00000-of-00001.parquet": "d9cc9f488e5310143d8d89956c19d0bdfd71e7089025da5202643f73720df7f5",
}


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the MIT License.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["MIT License"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the MIT License; review it")


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
    """Every image cell of every row of the split's file, in row order."""
    for file in FILES:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            yield image_sets.Found(data, origin(file, row, column), f"OCRBench {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the OCRBench set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
