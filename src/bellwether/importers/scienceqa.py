"""ScienceQA's test images as an image set: the diagrams and pictures of the questions that have one.

The data is the ScienceQA dataset on the Hugging Face Hub at a pinned commit: one parquet file of the test split, each
row a question with its picture in the ``image`` column where the question has one (about two in five rows do),
stored as the datasets library stores images (the bytes and a path). Pictures recur across rows; every image cell
gives a line in row order, a row without a picture gives nothing, and a cell whose bytes repeat an earlier image's is
left out and named, as the text imports name their repeats. The file is read as data (``image_sets.parquet_images``);
nothing in it runs.

The licence is stated twice and the two statements differ. The card's YAML front matter reads ``cc-by-sa-4.0``, which
the import checks on every run; the card's "Licensing Information" section and the authors' own ``LICENSE-DATA`` file
in the benchmark's repository read Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. The import
records what the licence file says, CC-BY-NC-SA-4.0, and copies that file from the repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "scienceqa"
REPO = "derek-thomas/ScienceQA"
REVISION = "f18b0a70359ebfb41f658fd564208d0355b013f4"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states a licence.
CARD_SHA256 = "3729fae304968ed8b9917b41034412aa142fb4760e2bfdab3dfe67a7fd78a17a"
LICENSE = "CC-BY-NC-SA-4.0"
CARD_LICENSE = "cc-by-sa-4.0"  # the license key of the reviewed card's front matter; LICENSE is the licence file's
# The benchmark's repository and its LICENSE-DATA at the one commit that file has.
LICENSE_OWNER, LICENSE_REPO = "lupantech", "ScienceQA"
LICENSE_COMMIT = "360b71cb6a36b9e3161dc9b65e9d86458b7d13a3"
LICENSE_PATH = "LICENSE-DATA"
LICENSE_SHA256 = "fae845447443179ecda7a2cdaf6f305e92fe8fc1b7e0c5a8f39a2e4d5de9bfce"
LICENSE_COPY = "licenses/scienceqa-LICENSE"
SPLIT = "test"
COLUMNS = ("image",)
# The test split's file, by the sha256 of its bytes.
FILES = {
    "data/test-00000-of-00001-f0e719df791966ff.parquet": (
        "235e92d1f30155266df76bc9f28fc6e4fcb6bec2c6a8c7d67f9086ea6b392a84"
    ),
}


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 licence.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["# Attribution-NonCommercial-ShareAlike 4.0 International"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the CC BY-NC-SA 4.0 licence text; review it")


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
            yield image_sets.Found(data, origin(file, row, column), f"ScienceQA {SPLIT} row {row}, {column}")


def fetch(filename: str, sha256: str, cache: Path) -> Path:
    """The path of a dataset file at the pinned commit, checked against ``sha256``."""
    return hf.fetch(REPO, REVISION, filename, sha256, cache)


def run(args: argparse.Namespace) -> int:
    hf.check_card_license(REPO, REVISION, CARD_SHA256, CARD_LICENSE, args.cache)
    license_text = github.fetch(
        LICENSE_OWNER, LICENSE_REPO, LICENSE_COMMIT, LICENSE_PATH, LICENSE_SHA256, cache=args.cache
    ).read_bytes()
    check_license(license_text)
    files = {name: fetch(name, sha256, args.cache) for name, sha256 in FILES.items()}
    copies = {LICENSE_COPY: license_text}
    if args.check:
        problems, _ = image_sets.check(DATASET, read(files), args.corpus, copies)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the ScienceQA set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
