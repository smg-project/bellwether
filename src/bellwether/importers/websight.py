"""WebSight's screenshots as an image set, one shard at a time: renderings of synthetic web pages.

The data is the WebSight dataset on the Hugging Face Hub at a pinned commit, its ``v0.2`` configuration: 738 parquet
files of the train split, each row a generated web page with its screenshot in the ``image`` column, stored as the
datasets library stores images (the bytes and a path), beside its HTML. Every row gives a line in file and row order;
a screenshot whose bytes repeat an earlier image's is left out and named, as the text imports name their repeats. The
files are read as data (``image_sets.parquet_images``); nothing in them runs. The set grows a shard at a time:
``FILES`` lists the files imported so far, in order, and a file added later appends its screenshots after the earlier
ones, so the names already written do not move.

The dataset card states ``cc-by-4.0``, which the import checks on every run. The dataset ships no licence file, so the
import copies the licence text as the SPDX license list publishes it, from that list's repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "websight"
REPO = "HuggingFaceM4/WebSight"
REVISION = "b11f8172f89c992b56ac702319e02c428cca4a4e"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "5985ce373605bee226e188c8f5e08cec0ffccd73d1d23baae4320886fc5ed18e"
LICENSE = "CC-BY-4.0"
CARD_LICENSE = "cc-by-4.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it (text/CC-BY-4.0.txt), from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CC-BY-4.0.txt"
LICENSE_SHA256 = "d557539df68e771cc1eedcc91d13f70fca930e508d11eedcafa4b15db49e3744"
LICENSE_COPY = "licenses/websight-LICENSE"
SPLIT = "train"
COLUMNS = ("image",)
# The shards imported so far, by the sha256 of their bytes, in the order the lines are built; a new shard goes last.
FILES = {
    "v0.2/train-00000-of-00738-80a58552f2fb3344.parquet": (
        "d26f5a5eb90223549052895be36f7ecf38cdcc3cf4596a680f3531e65346ee1f"
    ),
    "v0.2/train-00001-of-00738-b79d703dc79dce91.parquet": (
        "83720eb408682061180a3755444853f44b8299ee69d05289de708bf6a7f0de7f"
    ),
}


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the Creative Commons Attribution 4.0 International licence.

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
    """Every screenshot of every row of the shards, in the order of ``FILES`` and the rows."""
    for file in FILES:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            yield image_sets.Found(data, origin(file, row, column), f"WebSight {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the WebSight set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
