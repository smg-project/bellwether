"""ScreenSpot-v2's screenshots as an image set: the desktop, mobile and web screens behind the grounding benchmark.

The data is the ScreenSpot-v2 release on the Hugging Face Hub at a pinned commit: one zip archive,
``screenspotv2_image.zip``, of 755 screenshots (PNG, a few JPEG) under one directory, beside three JSON files of
instructions and boxes that name the screenshots and are not read here. Every PNG and JPEG member gives a line in
member name order (``image_sets.zip_images``); the archive's one other member, a tar, is not an image and is not read.
A screenshot whose bytes repeat an earlier image's is left out and named, as the text imports name their repeats.
Nothing in the archive runs.

The dataset card is its front matter alone and states ``apache-2.0``, which the import checks on every run. The dataset
ships no licence file, so the import copies the Apache License 2.0 text as the SPDX license list publishes it, from
that list's repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "screenspot"
REPO = "OS-Copilot/ScreenSpot-v2"
REVISION = "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "98b45ea81164d1e1a1dd82255207053b15cd6c69d922a1c5cf3387ce604d4b74"
LICENSE = "Apache-2.0"
CARD_LICENSE = "apache-2.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/Apache-2.0.txt"
LICENSE_SHA256 = "074e6e32c86a4c0ef8b3ed25b721ca23aca83df277cd88106ef7177c354615ff"
LICENSE_COPY = "licenses/screenspot-LICENSE"
# The archive at the pinned commit, by the sha256 of its bytes.
ARCHIVE = "screenspotv2_image.zip"
ARCHIVE_SHA256 = "750555b8b06e3898634c610cd904600feb7ec7ca530e625a00c1416a73d829ee"
SUFFIXES = (".png", ".jpg")


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the Apache License 2.0 text.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["Apache License"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the Apache License 2.0 text; review it")


def origin(member: str) -> dict:
    """Where a line came from, in the order the text imports write their origins: the dataset, its source, the
    archive the bytes were read from with its sha256, the member, and the licence."""
    return {
        "dataset": DATASET,
        "source": SOURCE,
        "sha256": ARCHIVE_SHA256,
        "file": ARCHIVE,
        "member": member,
        "license": LICENSE,
    }


def read(archive: Path) -> Iterator[image_sets.Found]:
    """Every PNG and JPEG member of the archive, in member name order."""
    for member, data in image_sets.zip_images(archive, SUFFIXES):
        yield image_sets.Found(data, origin(member), f"ScreenSpot-v2 {member}")


def fetch(cache: Path) -> Path:
    """The path of the archive at the pinned commit, checked against ``ARCHIVE_SHA256``."""
    return hf.fetch(REPO, REVISION, ARCHIVE, ARCHIVE_SHA256, cache)


def run(args: argparse.Namespace) -> int:
    hf.check_card_license(REPO, REVISION, CARD_SHA256, CARD_LICENSE, args.cache)
    license_text = github.fetch(
        LICENSE_OWNER, LICENSE_REPO, LICENSE_COMMIT, LICENSE_PATH, LICENSE_SHA256, cache=args.cache
    ).read_bytes()
    check_license(license_text)
    archive = fetch(args.cache)
    copies = {LICENSE_COPY: license_text}
    if args.check:
        problems, _ = image_sets.check(DATASET, read(archive), args.corpus, copies)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the ScreenSpot-v2 set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(archive), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
