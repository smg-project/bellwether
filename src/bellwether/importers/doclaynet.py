"""DocLayNet's pages as an image set, one shard at a time: page renderings of financial reports, manuals, scientific
articles, laws, patents and tenders.

The data is the DocLayNet v1.1 release on the Hugging Face Hub at a pinned commit: two parquet files of the
test split, each row a page rendering in the ``image`` column, stored as the datasets library stores images
(the bytes and a path), beside its layout annotations. A file holds 2,500 pages of some 400 kB each, so the
set is built from windows of rows (``SHARDS``: a file, a first row and a count).
Every row gives a line in file and row order; a page whose bytes repeat an earlier image's is left out and named, as the
text imports name their repeats. The files are read as data (``image_sets.parquet_images``); nothing in them runs. A
window added later appends its pages after the earlier ones, so the names already written do not move. The dataset
card's front matter states ``other``, which the import checks as written; the card's Licensing Information section names
CDLA-Permissive-1.0, whose text the import copies as the SPDX license list publishes it (``text/CDLA-
Permissive-1.0.txt``), from that list's repository at a pinned commit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "doclaynet"
REPO = "ds4sd/DocLayNet-v1.1"
REVISION = "5e89392376049f4d589ea339ed64468310ed5c3f"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "78addf644bc11c5887322513f7a878e650ad8423da410a7d5b82ba1fe7b052bb"
LICENSE = "CDLA-Permissive-1.0"
CARD_LICENSE = "other"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CDLA-Permissive-1.0.txt"
LICENSE_SHA256 = "52aa39b7c72a3712b1e1ab0b2d6b9d790f443f985407672b9df8edab9eb64712"
LICENSE_COPY = "licenses/doclaynet-LICENSE"
SPLIT = "test"
COLUMNS = ("image",)
# The files read so far, by the sha256 of their bytes.
FILES = {
    "data/test-00000-of-00002-635b47e9044a436c.parquet": (
        "ad934b6790dd08e7fa28d986eb54ea2963450d0a58a912903478f8b6cb630f46"
    ),
}
# The windows of rows that are in the set, in the order the lines are built: a file, its first row and the count. A
# window added later goes last, whether of a new file or of more rows of one already read.
SHARDS = (
    ("data/test-00000-of-00002-635b47e9044a436c.parquet", 0, 1000),
    ("data/test-00000-of-00002-635b47e9044a436c.parquet", 1000, 1500),
)


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the CDLA-Permissive-1.0 text.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["Community Data License Agreement – Permissive – Version 1.0"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the CDLA-Permissive-1.0 text; review it")


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
            yield image_sets.Found(data, origin(file, row, column), f"DocLayNet {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the DocLayNet set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
