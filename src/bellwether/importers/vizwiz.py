"""VizWiz-VQA's photographs as an image set, one shard at a time: photographs blind people took with their phones.

The data is the VizWiz-VQA dataset as the lmms-lab mirror publishes it on the Hugging Face Hub at a pinned commit:
five parquet files of the validation split, each row a question with its photograph in the ``image`` column, stored as
the datasets library stores images (the bytes and a path). Every row gives a line in file and row order; a photograph
whose bytes repeat an earlier image's is left out and named, as the text imports name their repeats. The files are
read as data (``image_sets.parquet_images``); nothing in them runs. The set grows a shard at a time: ``FILES`` lists
the files imported so far, in order, and a file added later appends its photographs after the earlier ones, so the
names already written do not move. The dataset card states no licence, which the import checks on every run; the
authors publish the data under CC BY 4.0 (vizwiz.org), which the import records. The formatting ships no licence file,
so the import copies the CC BY 4.0 text as the SPDX license list publishes it, from that list's repository at a pinned
commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "vizwiz"
REPO = "lmms-lab-encoder/VizWiz-VQA"
REVISION = "d428a2dae984f79cf1b9d99467dfa883e0c30686"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "b56a6c0e0e3ad508264e46496a0fb39eb440649b87c8060c8dcb2f612fea4e74"
LICENSE = "CC-BY-4.0"
CARD_LICENSE = None  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CC-BY-4.0.txt"
LICENSE_SHA256 = "d557539df68e771cc1eedcc91d13f70fca930e508d11eedcafa4b15db49e3744"
LICENSE_COPY = "licenses/vizwiz-LICENSE"
SPLIT = "val"
COLUMNS = ("image",)
# The shards imported so far, by the sha256 of their bytes, in the order the lines are built; a new shard goes last.
FILES = {
    "data/val-00000-of-00005-7775fd61bc6a3d98.parquet": (
        "62a5fddb577d9165c3b89d6fb565cddfb4e7d41294ca16373de6263baff7b36d"
    ),
}


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the CC BY 4.0 licence text.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["Creative Commons Attribution 4.0 International"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the CC BY 4.0 licence text; review it")


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
            yield image_sets.Found(data, origin(file, row, column), f"VizWiz-VQA {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the VizWiz-VQA set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
