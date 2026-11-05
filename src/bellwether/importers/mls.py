"""Multilingual LibriSpeech's utterances as an audio set, one shard at a time: read Polish audiobook speech, the test
split, Opus in Ogg.

The data is the Multilingual LibriSpeech dataset on the Hugging Face Hub at a pinned commit, its ``polish``
configuration: the one parquet file of the ``test`` split, 520 utterances in the ``audio`` column, stored as
the datasets library stores audio (the bytes and a path), each an Opus stream in Ogg, mono, beside its
transcript, its timing in the source recording and its duration.
Every row gives a line in file and row order; an utterance whose bytes repeat an earlier clip's is left out and named,
as the text imports name their repeats. The files are read as data (``audio_sets.parquet_clips``); nothing in them runs.
The set grows a shard at a time: ``FILES`` lists the files imported so far, in order, and a file added later appends its
utterances after the earlier ones, so the names already written do not move.

The dataset card states ``cc-by-4.0``, as a one-item list, which the import checks as written; its licensing
section adds that the recordings are public domain audiobooks. The dataset ships no licence file, so the
import copies the CC BY 4.0 text as the SPDX license list publishes it, from that list's repository at a
pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import audio_sets, github, hf

DATASET = "mls"
REPO = "facebook/multilingual_librispeech"
REVISION = "2e83e61823b4c47dcbcb1980bb88601274127609"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "b79c762d17d4f8fd7f4c1a1c59f6154c554e4faebb0a97f1fccf49af94a265be"
LICENSE = "CC-BY-4.0"
CARD_LICENSE = ["cc-by-4.0"]  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CC-BY-4.0.txt"
LICENSE_SHA256 = "d557539df68e771cc1eedcc91d13f70fca930e508d11eedcafa4b15db49e3744"
LICENSE_COPY = "licenses/mls-LICENSE"
SPLIT = "test"
COLUMNS = ("audio",)
# The shards imported so far, by the sha256 of their bytes, in the order the lines are built; a new shard goes last.
FILES = {
    "polish/test-00000-of-00001.parquet": ("43656b9106d1e8d8c5cd0a3e2a50ef5a29bc659a3cbccf3541f0c1359255ac3b"),
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


def read(files: dict[str, Path]) -> Iterator[audio_sets.Found]:
    """Every row of the shards, in the order of ``FILES`` and the rows."""
    for file in FILES:
        for row, column, data in audio_sets.parquet_clips(files[file], COLUMNS):
            yield audio_sets.Found(
                data, origin(file, row, column), f"Multilingual LibriSpeech {SPLIT} row {row}, {column}"
            )


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
        problems, _ = audio_sets.check(DATASET, read(files), args.corpus, copies)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the Multilingual LibriSpeech set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = audio_sets.write(DATASET, read(files), args.corpus, copies)
    audio_sets.report(DATASET, summary, args.corpus)
    return 0
