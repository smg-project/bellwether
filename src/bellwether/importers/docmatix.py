"""Docmatix's document pages as an image set, one shard at a time: rendered pages of real documents.

The data is the Docmatix dataset on the Hugging Face Hub at a pinned commit, its ``images`` configuration: 1,106
parquet files of the train split, each row a document with its pages as a list in the ``images`` column, stored as the
datasets library stores images (the bytes and a path), and its questions in ``texts``. Every page of every row gives a
line in file, row and page order, named ``images[k]``; a page whose bytes repeat an earlier image's is left out and
named, as the text imports name their repeats. The files are read as data (``image_sets.parquet_images``); nothing in
them runs. The set grows a shard at a time: ``FILES`` lists the files imported so far, in order, and a file added later
appends its pages after the earlier ones, so the names already written do not move.

The dataset card states ``mit``, which the import checks on every run. The dataset ships no licence file, so the import
copies the MIT License text as the SPDX license list publishes it, from that list's repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "docmatix"
REPO = "HuggingFaceM4/Docmatix"
REVISION = "0725b65616e0e5f6024be10e38ddf8d8c48664fd"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "39c1a5f25dee9e26d1596624227b37e3ada582ff7386a3b6446d1a744cadd307"
LICENSE = "MIT"
CARD_LICENSE = "mit"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it (text/MIT.txt), from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/MIT.txt"
LICENSE_SHA256 = "b05785f9f18e6716bab63424b11454513b9943a222595b70411009202fc592b5"
LICENSE_COPY = "licenses/docmatix-LICENSE"
SPLIT = "train"
COLUMNS = ("images",)
# The shards imported so far, by the sha256 of their bytes, in the order the lines are built; a new shard goes last.
FILES = {
    "data/train-00000-of-01106-8ce5201aa40b0141.parquet": (
        "c39d9e27b80319759a5e52ffc35bd636e83a628c172c7707b8bc3cce44347974"
    ),
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
    the bytes were read from with its sha256, the row and the page (``images[k]``), and the licence."""
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
    """Every page of every row of the shards, in the order of ``FILES``, the rows and the pages."""
    for file in FILES:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            yield image_sets.Found(data, origin(file, row, column), f"Docmatix {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the Docmatix set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
