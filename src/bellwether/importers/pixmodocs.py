"""PixMo-Docs's charts as an image set, one shard at a time: synthetic charts, rendered by code a language model wrote.

The data is the PixMo-Docs dataset on the Hugging Face Hub at a pinned commit, its ``charts`` configuration:
50 parquet files of the train split, each row a rendered chart in the ``image`` column, stored as the datasets
library stores images (the bytes and a path), with its id and its questions.
Every row gives a line in file and row order; a chart whose bytes repeat an earlier image's is left out and named, as
the text imports name their repeats. The files are read as data (``image_sets.parquet_images``); nothing in them runs.
The set grows a shard at a time: ``FILES`` lists the files imported so far, in order, and a file added later appends its
charts after the earlier ones, so the names already written do not move. The dataset card states ``odc-by``, which the
import checks on every run. The dataset ships no licence file, so the import copies the ODC Attribution License text as
the SPDX license list publishes it (``text/ODC-By-1.0.txt``), from that list's repository at a pinned commit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "pixmodocs"
REPO = "allenai/pixmo-docs"
REVISION = "d887597bf4af2bc61a4210071a8cef898287e6fb"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "d54a7bcafa53a707aee8bccbd9d6652afe03c99bc24f3a3d262fcab5ec95f42b"
LICENSE = "ODC-By-1.0"
CARD_LICENSE = "odc-by"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/ODC-By-1.0.txt"
LICENSE_SHA256 = "b1c4de4bf91c95eec945ea7cbe0575fd52e7ee8fa430acb01d323d8214b1c482"
LICENSE_COPY = "licenses/pixmodocs-LICENSE"
SPLIT = "train"
COLUMNS = ("image",)
# The shards imported so far, by the sha256 of their bytes, in the order the lines are built; a new shard goes last.
FILES = {
    "charts/train-00000-of-00050.parquet": ("bec02f60b35fc8b6d8bb2d275b8295751bcd5ee4204e22315f5d11b4de2f357f"),
}


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the ODC Attribution License.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["# ODC Attribution License (ODC-By)"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the ODC Attribution License; review it")


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
            yield image_sets.Found(data, origin(file, row, column), f"PixMo-Docs {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the PixMo-Docs set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
