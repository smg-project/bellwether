"""ChartQA's test images as an image set: the charts its questions are asked about.

The data is the ChartQA dataset on the Hugging Face Hub at a pinned commit: one parquet file of the test split, each
row a question with its chart in the ``image`` column, stored as the datasets library stores images (the bytes and a
path). Several questions are asked about one chart, so the bytes of a chart recur across rows; every image cell gives
a line in row order, and a cell whose bytes repeat an earlier image's is left out and named, as the text imports name
their repeats. The file is read as data (``image_sets.parquet_images``); nothing in it runs.

The dataset card states ``gpl-3.0``, as the benchmark's repository does, whose ``LICENSE`` is the GNU General Public
License version 3. The import records GPL-3.0 and copies that file from the repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "chartqa"
REPO = "HuggingFaceM4/ChartQA"
REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "b472dec95769eeda282ccb070b2820b80035f2be5181547d10ed300772adc666"
LICENSE = "GPL-3.0"
CARD_LICENSE = "gpl-3.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The benchmark's repository and its LICENSE at the one commit that file has.
LICENSE_OWNER, LICENSE_REPO = "vis-nlp", "ChartQA"
LICENSE_COMMIT = "8f3b48f7fbe7c35d0f212102e8db72976798f75d"
LICENSE_PATH = "LICENSE"
LICENSE_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
LICENSE_COPY = "licenses/chartqa-LICENSE"
SPLIT = "test"
COLUMNS = ("image",)
# The test split's file, by the sha256 of its bytes.
FILES = {
    "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet": (
        "efcf31333368c485eb569ec7bd4cc3694bd0e8e1a06c503124b51d27c924491d"
    ),
}


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the GNU General Public License, version 3.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first two lines are checked as
    well, so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:2] != ["GNU GENERAL PUBLIC LICENSE", "Version 3, 29 June 2007"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the GNU General Public License v3; review it")


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
            yield image_sets.Found(data, origin(file, row, column), f"ChartQA {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the ChartQA set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
