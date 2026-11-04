"""MNIST's digits as an image set, one shard at a time: 28 by 28 greyscale handwritten digits.

The data is the MNIST database as the ylecun organisation publishes it on the Hugging Face Hub at a pinned commit: one
parquet file per split, each row a 28 by 28 greyscale digit in the ``image`` column, stored as the datasets library
stores images (the bytes and a path), with its label. The test file has 10,000 rows, so the set is built from windows of
rows (``SHARDS``: a file, a first row and a count). Every row gives a line in file and row order; a digit whose bytes
repeat an earlier image's is left out and named, as the text imports name their repeats. The files are read as data
(``image_sets.parquet_images``); nothing in them runs. A window added later appends its digits after the earlier ones,
so the names already written do not move. The dataset card states ``mit`` (as a one-item list), which the import checks
on every run. The dataset ships no licence file, so the import copies the MIT License text as the SPDX license list
publishes it, from that list's repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "mnist"
REPO = "ylecun/mnist"
REVISION = "77f3279092a1c1579b2250db8eafed0ad422088c"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "8cd17b7ebe5c115eb401f2ce57f9b3c8b16a6e293dc312c35e4f37f7a81fc4a2"
LICENSE = "MIT"
CARD_LICENSE = ["mit"]  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/MIT.txt"
LICENSE_SHA256 = "b05785f9f18e6716bab63424b11454513b9943a222595b70411009202fc592b5"
LICENSE_COPY = "licenses/mnist-LICENSE"
SPLIT = "test"
COLUMNS = ("image",)
# The files read so far, by the sha256 of their bytes.
FILES = {
    "mnist/test-00000-of-00001.parquet": ("d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b"),
}
# The windows of rows that are in the set, in the order the lines are built: a file, its first row and the count. A
# window added later goes last, whether of a new file or of more rows of one already read.
SHARDS = (("mnist/test-00000-of-00001.parquet", 0, 2000),)


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
    """Every row of the windows, in the order of ``SHARDS`` and the rows."""
    for file, first, count in SHARDS:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            if row < first:
                continue
            if row >= first + count:
                break
            yield image_sets.Found(data, origin(file, row, column), f"MNIST {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the MNIST set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
