"""InfographicVQA's infographics as an image set, one shard at a time: the web infographics behind the validation
questions.

The data is the DocVQA repository as the lmms-lab mirror publishes it on the Hugging Face Hub at a pinned commit, its
``InfographicVQA`` configuration: four parquet files of the validation split, each row a question with its infographic
in the ``image`` column, stored as the datasets library stores images (the bytes and a path). Every row gives a line
in file and row order; a infographic whose bytes repeat an earlier image's is left out and named, as the text imports
name their repeats. The files are read as data (``image_sets.parquet_images``); nothing in them runs. The set grows a
shard at a time: ``FILES`` lists the files imported so far, in order, and a file added later appends its infographics
after the earlier ones, so the names already written do not move. The dataset card states ``apache-2.0`` (the same
card the DocVQA import checks), which the import checks on every run. The dataset ships no licence file, so the import
copies the Apache License 2.0 text as the SPDX license list publishes it, from that list's repository at a pinned
commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "infovqa"
REPO = "lmms-lab-encoder/DocVQA"
REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "372cd2689649a29b4f4d1c9dbf7478d62b8331ab2af0720b5a1f6fa9d4f266c0"
LICENSE = "Apache-2.0"
CARD_LICENSE = "apache-2.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/Apache-2.0.txt"
LICENSE_SHA256 = "074e6e32c86a4c0ef8b3ed25b721ca23aca83df277cd88106ef7177c354615ff"
LICENSE_COPY = "licenses/infovqa-LICENSE"
SPLIT = "validation"
COLUMNS = ("image",)
# The shards imported so far, by the sha256 of their bytes, in the order the lines are built; a new shard goes last.
FILES = {
    "InfographicVQA/validation-00000-of-00004.parquet": (
        "a71b63eca8d3cedc5bda7f7d2a4c959b041dca8766592675f6bd0ac49500775a"
    ),
    "InfographicVQA/validation-00001-of-00004.parquet": (
        "12cbc18ad5a499deebca691fcdf9a852c3d3917e448b30dc604bfdfb4559e756"
    ),
}


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
    """Every row of the shards, in the order of ``FILES`` and the rows."""
    for file in FILES:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            yield image_sets.Found(data, origin(file, row, column), f"InfographicVQA {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the InfographicVQA set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
