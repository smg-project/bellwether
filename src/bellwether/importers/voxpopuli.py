"""VoxPopuli's utterances as an audio set, one shard at a time: European Parliament speech in Croatian, the test split,
32-bit float WAV at 16 kHz.

The data is the VoxPopuli dataset on the Hugging Face Hub at a pinned commit, its ``hr`` configuration: the
one parquet file of the ``test`` split, 666 utterances in the ``audio`` column, stored as the datasets
library stores audio (the bytes and a path), each a WAV file of 32-bit floats at 16 kHz, mono, beside its
transcripts and speaker fields. The set is built from windows of rows (``SHARDS``: a file, a first row and a
count), so that a shard stays a reviewable size.
Every row gives a line in file and row order; an utterance whose bytes repeat an earlier clip's is left out and named,
as the text imports name their repeats. The files are read as data (``audio_sets.parquet_clips``); nothing in them runs.
A window added later appends its utterances after the earlier ones, so the names already written do not move.

The dataset card states ``cc0-1.0`` and ``other`` (the European Parliament's legal notice for the raw
recordings), which the import checks as written; its licensing section says the dataset is distributed under
CC0. The dataset ships no licence file, so the import copies the CC0 1.0 legal code as the SPDX license list
publishes it, from that list's repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import audio_sets, github, hf

DATASET = "voxpopuli"
REPO = "facebook/voxpopuli"
REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "eb1a3f01c5529449ce57a4bbbe5343361860a5e1335d8fec43657506d4059b45"
LICENSE = "CC0-1.0"
CARD_LICENSE = [
    "cc0-1.0",
    "other",
]  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CC0-1.0.txt"
LICENSE_SHA256 = "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"
LICENSE_COPY = "licenses/voxpopuli-LICENSE"
SPLIT = "test"
COLUMNS = ("audio",)
# The files read so far, by the sha256 of their bytes.
FILES = {
    "hr/test-00000-of-00001.parquet": ("341fa402043d7f8479280aa420d183731d7dbdc457c67b6a0d4e334132279ae4"),
}
# The windows of rows that are in the set, in the order the lines are built: a file, its first row and the count. A
# window added later goes last, whether of a new file or of more rows of one already read.
SHARDS = (("hr/test-00000-of-00001.parquet", 0, 300),)


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the CC0 1.0 legal code.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["Creative Commons Legal Code"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the CC0 1.0 legal code; review it")


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


def read(files: dict[str, Path]) -> Iterator[audio_sets.Found]:
    """Every row of the windows, in the order of ``SHARDS`` and the rows."""
    for file, first, count in SHARDS:
        for row, column, data in audio_sets.parquet_clips(files[file], COLUMNS):
            if row < first:
                continue
            if row >= first + count:
                break
            yield audio_sets.Found(data, origin(file, row, column), f"VoxPopuli {SPLIT} row {row}, {column}")


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
        problems, _ = audio_sets.check(DATASET, read(files), args.corpus, copies)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the VoxPopuli set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = audio_sets.write(DATASET, read(files), args.corpus, copies)
    audio_sets.report(DATASET, summary, args.corpus)
    return 0
