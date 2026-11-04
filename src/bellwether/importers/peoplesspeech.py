"""People's Speech's utterances as an audio set, one shard at a time: English speech from public recordings, the clean
configuration's test split, FLAC at 16 kHz.

The data is the People's Speech dataset on the Hugging Face Hub at a pinned commit, its ``clean``
configuration: the first of nine parquet files of the ``test`` split, 3,878 utterances in the ``audio``
column, stored as the datasets library stores audio (the bytes and a path), each a FLAC file at 16 kHz, mono,
16 bits, beside its text and duration. The set is built from windows of rows (``SHARDS``: a file, a first
row and a count), so that a shard stays a reviewable size.
Every row gives a line in file and row order; an utterance whose bytes repeat an earlier clip's is left out and named,
as the text imports name their repeats. The files are read as data (``audio_sets.parquet_clips``); nothing in them runs.
A window added later appends its utterances after the earlier ones, so the names already written do not move.

The dataset card states six licences as a list (CC BY 2.0, 2.5, 3.0 and 4.0, CC BY-SA 3.0 and 4.0), which the
import checks as written; its licensing section says the dataset comes as CC-BY and CC-BY-SA subsets, and the
``clean`` configuration is the CC-BY one (the ``clean_sa`` configuration beside it is CC-BY-SA). The dataset
ships no licence file, so the import copies the CC BY 4.0 text as the SPDX license list publishes it, from
that list's repository at a pinned commit; a recording's own CC BY version is its source's.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import audio_sets, github, hf

DATASET = "peoplesspeech"
REPO = "MLCommons/peoples_speech"
REVISION = "f10597c5d3d3a63f8b6827701297c3afdf178272"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "6bcb310cfbf34f5a42bbe87de3e74792a238eebea5c4630fc979be41ef700cb0"
LICENSE = "CC-BY-4.0"
CARD_LICENSE = [
    "cc-by-2.0",
    "cc-by-2.5",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
]  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/CC-BY-4.0.txt"
LICENSE_SHA256 = "d557539df68e771cc1eedcc91d13f70fca930e508d11eedcafa4b15db49e3744"
LICENSE_COPY = "licenses/peoplesspeech-LICENSE"
SPLIT = "test"
COLUMNS = ("audio",)
# The files read so far, by the sha256 of their bytes.
FILES = {
    "clean/test-00000-of-00009.parquet": ("6cae812dd9f6a1e027ae7f672d7cb7c756253078d25b5d9eb4ae6171eb68fdf0"),
}
# The windows of rows that are in the set, in the order the lines are built: a file, its first row and the count. A
# window added later goes last, whether of a new file or of more rows of one already read.
SHARDS = (("clean/test-00000-of-00009.parquet", 0, 1000),)


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
    """Every row of the windows, in the order of ``SHARDS`` and the rows."""
    for file, first, count in SHARDS:
        for row, column, data in audio_sets.parquet_clips(files[file], COLUMNS):
            if row < first:
                continue
            if row >= first + count:
                break
            yield audio_sets.Found(data, origin(file, row, column), f"People's Speech {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the People's Speech set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = audio_sets.write(DATASET, read(files), args.corpus, copies)
    audio_sets.report(DATASET, summary, args.corpus)
    return 0
