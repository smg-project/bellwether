"""VQA-RAD's scans as an image set, one shard at a time: the radiology images behind its clinical questions.

The data is the VQA-RAD dataset as its Hub mirror publishes it at a pinned commit: one parquet file per split (test and
train), each row a question with its radiology image in the ``image`` column, stored as the datasets library stores
images (the bytes and a path). Every row gives a line in file and row order; a scan whose bytes repeat an earlier
image's is left out and named, as the text imports name their repeats. The files are read as data
(``image_sets.parquet_images``); nothing in them runs. The set grows a shard at a time: ``FILES`` lists the files
imported so far, in order, and a file added later appends its scans after the earlier ones, so the names already written
do not move. The dataset card states ``cc0-1.0``, which the import checks on every run. The dataset ships no licence
file, so the import copies the CC0 1.0 text as the SPDX license list publishes it, from that list's repository at a
pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "vqarad"
REPO = "flaviagiammarino/vqa-rad"
REVISION = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "7474c04be14206155c02601be6d62379b177903afc49a715ba6a11d62c32c9bf"
LICENSE = "CC0-1.0"
CARD_LICENSE = "cc0-1.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CC0-1.0.txt"
LICENSE_SHA256 = "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"
LICENSE_COPY = "licenses/vqarad-LICENSE"
SPLIT = "test"
COLUMNS = ("image",)
# The shards imported so far, by the sha256 of their bytes, in the order the lines are built; a new shard goes last.
FILES = {
    "data/test-00000-of-00001-e5bc3d208bb4deeb.parquet": (
        "eb520bdab1116dd4f420120da19049d2315389fa126d031f65ec42e153264ea7"
    ),
    "data/train-00000-of-00001-eb8844602202be60.parquet": (
        "b07c3441467b99060e5ec412ddd05be06f86f01f23bfa3debfbbcab47874a06e"
    ),
}


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the CC0 1.0 text.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["Creative Commons Legal Code"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the CC0 1.0 text; review it")


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
            yield image_sets.Found(data, origin(file, row, column), f"VQA-RAD {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the VQA-RAD set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
