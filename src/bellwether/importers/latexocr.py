"""LaTeX-OCR's formulas as an image set, one shard at a time: rendered and handwritten mathematical formulas.

The data is the LaTeX_OCR data repository as its author publishes it on the Hugging Face Hub at a pinned commit:
configurations of rendered and handwritten formulas, each a parquet file per split with the formula's picture in the
``image`` column, stored as the datasets library stores images (the bytes and a path), and its LaTeX in ``text``. The
``full`` test file has 9,443 rows, so the set is built from windows of rows (``SHARDS``: a file, a first row and a
count); the ``human_handwrite`` test file's 70 rows are one window. Every row gives a line in file and row order; a
formula whose bytes repeat an earlier image's is left out and named, as the text imports name their repeats. The files
are read as data (``image_sets.parquet_images``); nothing in them runs. A window added later appends its formulas after
the earlier ones, so the names already written do not move. The dataset card states ``apache-2.0``, which the import
checks on every run. The dataset ships no licence file, so the import copies the Apache License 2.0 text as the SPDX
license list publishes it, from that list's repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "latexocr"
REPO = "linxy/LaTeX_OCR"
REVISION = "e4167ee052306f9fa006ea6703161956f8af9f6e"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "cf26f1f3a3a96b7263e61b4097a982eaa91e700c378984d7d462a67f0606384b"
LICENSE = "Apache-2.0"
CARD_LICENSE = "apache-2.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/Apache-2.0.txt"
LICENSE_SHA256 = "074e6e32c86a4c0ef8b3ed25b721ca23aca83df277cd88106ef7177c354615ff"
LICENSE_COPY = "licenses/latexocr-LICENSE"
SPLIT = "test"
COLUMNS = ("image",)
# The files read so far, by the sha256 of their bytes.
FILES = {
    "full/test-00000-of-00001.parquet": ("908015e4c891792d36ea8070e9e0d90b61b1f89f2b0ddf7a647b236b5b59496a"),
    "human_handwrite/test-00000-of-00001.parquet": ("56461d611a855d48d2e78b5716bdd26ebe150b12df2619701e4db450cf242497"),
}
# The windows of rows that are in the set, in the order the lines are built: a file, its first row and the count. A
# window added later goes last, whether of a new file or of more rows of one already read.
SHARDS = (
    ("full/test-00000-of-00001.parquet", 0, 2000),
    ("human_handwrite/test-00000-of-00001.parquet", 0, 70),
)


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the Apache License 2.0 text.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["Apache License"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the Apache License 2.0 text; review it")


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
            yield image_sets.Found(data, origin(file, row, column), f"LaTeX-OCR {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the LaTeX-OCR set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
