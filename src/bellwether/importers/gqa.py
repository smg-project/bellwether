"""GQA's photographs as an image set, one shard at a time: the Visual Genome photographs behind the balanced test-dev
and challenge questions.

The data is the GQA dataset as the lmms-lab mirror publishes it on the Hugging Face Hub at a pinned commit, its
``testdev_balanced_images`` and ``challenge_balanced_images`` configurations: one parquet file each, the photographs
in the ``image`` column, stored as the datasets library stores images (the bytes and a path), each with its ``id``.
Every row gives a line in file and row order; a photograph whose bytes repeat an earlier image's is left out and
named, as the text imports name their repeats. The files are read as data (``image_sets.parquet_images``); nothing in
them runs. The set grows a shard at a time: ``FILES`` lists the files imported so far, in order, and a file added
later appends its photographs after the earlier ones, so the names already written do not move. The dataset card
states ``mit``, which the import checks on every run; the photographs are Visual Genome's, whose own terms the card
does not restate. The dataset ships no licence file, so the import copies the MIT License text as the SPDX license
list publishes it, from that list's repository at a pinned commit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "gqa"
REPO = "lmms-lab-encoder/GQA"
REVISION = "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "10feb9ca3de7d975402f4bc165a7a76d698ee8bb97933d64baa7710aada18804"
LICENSE = "MIT"
CARD_LICENSE = "mit"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/MIT.txt"
LICENSE_SHA256 = "b05785f9f18e6716bab63424b11454513b9943a222595b70411009202fc592b5"
LICENSE_COPY = "licenses/gqa-LICENSE"
SPLIT = "testdev"
COLUMNS = ("image",)
# The shards imported so far, by the sha256 of their bytes, in the order the lines are built; a new shard goes last.
FILES = {
    "testdev_balanced_images/testdev-00000-of-00001.parquet": (
        "ba0a57a72c483d6e4a390ecbbee05d07a55365aeac3e93f4a923ac61c9e465eb"
    ),
    "challenge_balanced_images/challenge-00000-of-00001.parquet": (
        "ee9a88a519993a341314035875513a3e7999c0236b48d781e494286890332145"
    ),
}


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
    """Every row of the shards, in the order of ``FILES`` and the rows."""
    for file in FILES:
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            yield image_sets.Found(data, origin(file, row, column), f"GQA {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the GQA set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
