"""DocVQA's validation images as an image set: the document pages its questions are asked about.

The data is the DocVQA dataset on the Hugging Face Hub at a pinned commit, its ``DocVQA`` configuration (the
``InfographicVQA`` configuration beside it is not imported): six parquet files of the validation split, each row a
question with its page in the ``image`` column, stored as the datasets library stores images (the bytes and a path).
Several questions are asked about one page, so the bytes of a page recur across rows; every image cell gives a line in
file and row order, and a cell whose bytes repeat an earlier image's is left out and named, as the text imports name
their repeats. The files are read as data (``image_sets.parquet_images``); nothing in them runs.

The dataset was published as ``lmms-lab/DocVQA`` and the Hub now redirects that id to ``lmms-lab-encoder/DocVQA``,
which the import pins, so that the source reads as the Hub spells it; the files and the commit are the same.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "docvqa"
REPO = "lmms-lab-encoder/DocVQA"
REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "372cd2689649a29b4f4d1c9dbf7478d62b8331ab2af0720b5a1f6fa9d4f266c0"
LICENSE = "Apache-2.0"
CARD_LICENSE = "apache-2.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# Apache-2.0 4(a) asks that a copy of the License go with the work, and the dataset ships no LICENSE or NOTICE file, so
# the import copies the License as the Apache Software Foundation publishes it, from the foundation's website
# repository at the one commit that file has, as the Hermes, glaive-v2 and MMMU imports do.
LICENSE_OWNER, LICENSE_REPO = "apache", "www-site"
LICENSE_COMMIT = "01b1be9fbc5cd93b6794f5653a58b9b863807f84"
LICENSE_PATH = "content/licenses/LICENSE-2.0.txt"
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
LICENSE_COPY = "licenses/docvqa-LICENSE"
SPLIT = "validation"
COLUMNS = ("image",)
# The validation split's files, by the sha256 of their bytes, in the order the lines are built.
FILES = {
    "DocVQA/validation-00000-of-00006.parquet": "a31507f0b700ac64f3ead52057c5dc3ccfb0baadcc62bc0d19d17159f080a4d8",
    "DocVQA/validation-00001-of-00006.parquet": "ce63ba45f11bfb4e17362f9c073984b4c6c84a0e966ba3d33a86fa285a1fb3dc",
    "DocVQA/validation-00002-of-00006.parquet": "2d580a39996ad05226ad462f2d31bad0df02f0f47f46263d96ad8df584a5667b",
    "DocVQA/validation-00003-of-00006.parquet": "588a587567d1548a174b83b160132d97eb2b12a7d57374675d3c45b431403354",
    "DocVQA/validation-00004-of-00006.parquet": "b43b8aa3e28a1b0f8728243fdf776b798a684b5baafb27d6a734edbbe3983302",
    "DocVQA/validation-00005-of-00006.parquet": "5bba6d5f7b8e6ee0b64df2e4c2885f93cd759c732998e4d6f9b86ce4c690fcf2",
}


def check_license(text: bytes) -> None:
    """Refuse a License copy that is not the Apache License, Version 2.0.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first two lines are checked as
    well, so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:2] != ["Apache License", "Version 2.0, January 2004"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the Apache License, Version 2.0; review it")


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
    """Every image cell of every row of the split's files, in the order of ``FILES`` and the rows."""
    for file in FILES:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            yield image_sets.Found(data, origin(file, row, column), f"DocVQA {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the DocVQA set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
