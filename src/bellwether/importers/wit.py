"""WIT's pictures as an image set, one shard at a time: the pictures of Wikipedia articles, from Wikimedia Commons.

The data is the ``wit_base`` dataset on the Hugging Face Hub at a pinned commit: 330 parquet files of the
train split, each row a picture in the ``image`` column, stored as the datasets library stores images (the
bytes and a path), with its Commons URL, a caption and the article's text features. A file holds some 20,000
rows, so the set is built from windows of rows (``SHARDS``: a file, a first row and a count).
Every row gives a line in file and row order; a picture whose bytes repeat an earlier image's is left out and named, as
the text imports name their repeats. The files are read as data (``image_sets.parquet_images``); nothing in them runs. A
window added later appends its pictures after the earlier ones, so the names already written do not move. The dataset
card states ``cc-by-sa-4.0``, as a one-item list, which the import checks as written; each picture carries its own
Commons licence, in the dataset's metadata and not restated here. The dataset ships no licence file, so the import
copies the CC BY-SA 4.0 text as the SPDX license list publishes it, from that list's repository at a pinned commit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "wit"
REPO = "wikimedia/wit_base"
REVISION = "ff6d4fb32fd566d3a1fa20e946cba3234179465e"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "b6b3157207f702d289bdb11ba726cdf8cf7c4f7bf13c87738cd7383086d07801"
LICENSE = "CC-BY-SA-4.0"
CARD_LICENSE = ["cc-by-sa-4.0"]  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CC-BY-SA-4.0.txt"
LICENSE_SHA256 = "cde7883b9050a1104f4ac19a1572aafd6e5d7323b68351aaf51fbf4beba54966"
LICENSE_COPY = "licenses/wit-LICENSE"
SPLIT = "train"
COLUMNS = ("image",)
# The files read so far, by the sha256 of their bytes.
FILES = {
    "data/train-00000-of-00330.parquet": ("8a9a449a0db937920b7c0dd13cd0bc9b1cadb19071567169dbac41a956273ec2"),
}
# The windows of rows that are in the set, in the order the lines are built: a file, its first row and the count. A
# window added later goes last, whether of a new file or of more rows of one already read.
SHARDS = (
    ("data/train-00000-of-00330.parquet", 0, 2000),
    ("data/train-00000-of-00330.parquet", 2000, 4000),
)


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
    """Every row of the windows, in the order of ``SHARDS`` and the rows."""
    for file, first, count in SHARDS:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            if row < first:
                continue
            if row >= first + count:
                break
            yield image_sets.Found(data, origin(file, row, column), f"WIT {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the WIT set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
