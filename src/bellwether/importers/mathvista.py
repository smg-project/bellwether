"""MathVista's figures as an image set, one shard at a time: the testmini split's charts, diagrams, plots, tables and
scenes, gathered from 28 sources.

The data is the MathVista dataset on the Hugging Face Hub at a pinned commit: the one parquet file of its
``testmini`` split, 1,000 rows, each question with its figure in the ``decoded_image`` column, stored as the
datasets library stores images (the bytes and a path), beside the question's text and metadata (``image``
is the figure's file name in the release's archive, not the bytes).
Every row gives a line in file and row order; a figure whose bytes repeat an earlier image's is left out and named, as
the text imports name their repeats. The files are read as data (``image_sets.parquet_images``); nothing in them runs.
The set grows a shard at a time: ``FILES`` lists the files imported so far, in order, and a file added later appends its
figures after the earlier ones, so the names already written do not move. The dataset card states ``cc-by-sa-4.0`` for
the collection, which the import checks on every run; the card's License section says the images and questions keep
their original authors' copyright and names each source's terms in the dataset's ``source.json``. The dataset ships no
licence file, so the import copies the CC BY-SA 4.0 text as the SPDX license list publishes it, from that list's
repository at a pinned commit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "mathvista"
REPO = "AI4Math/MathVista"
REVISION = "2b6ad69445fbb5695c9b165475e8decdbeb97747"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "b2c1784e13e6a4bf1935930e20a7472a04b40e2d47b7509508266352d0e121ed"
LICENSE = "CC-BY-SA-4.0"
CARD_LICENSE = "cc-by-sa-4.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CC-BY-SA-4.0.txt"
LICENSE_SHA256 = "cde7883b9050a1104f4ac19a1572aafd6e5d7323b68351aaf51fbf4beba54966"
LICENSE_COPY = "licenses/mathvista-LICENSE"
SPLIT = "testmini"
COLUMNS = ("decoded_image",)
# The shards imported so far, by the sha256 of their bytes, in the order the lines are built; a new shard goes last.
FILES = {
    "data/testmini-00000-of-00001-725687bf7a18d64b.parquet": (
        "373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d"
    ),
}


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the CC BY-SA 4.0 licence text.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["Creative Commons Attribution-ShareAlike 4.0 International"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the CC BY-SA 4.0 licence text; review it")


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
    """Every row of the shards, in the order of ``FILES`` and the rows."""
    for file in FILES:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            yield image_sets.Found(data, origin(file, row, column), f"MathVista {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the MathVista set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
