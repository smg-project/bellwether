"""COCO's val2017 photographs as an image set: the 5,000 pictures of the 2017 validation split.

The data is the ``val2017.zip`` archive the COCO Consortium publishes, pinned by its sha256 (the archive has no
revision; its ETag and date are noted below): one JPEG per photograph under ``val2017/``, read as data
(``image_sets.zip_images``) in member name order, which is the order of the pictures' ids; nothing in it runs. Every
member gives a line, and a member whose bytes repeat an earlier image's would be left out and named, as the text
imports name their repeats (none does).

The Consortium's terms of use state that the annotations and the website are licensed under Creative Commons
Attribution 4.0 and that the Consortium does not own the images' copyright: their use must abide by the Flickr Terms
of Use. The import records the images' terms as the Consortium states them and copies the terms-of-use page from the
Consortium's website repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, image_sets, web

DATASET = "coco"
URL = "http://images.cocodataset.org/zips/val2017.zip"
SOURCE = URL
# The archive's sha256, the pin; its ETag d366be60d3dc737327160d62453e3973-98 and Last-Modified 2018-07-11 as served.
ARCHIVE_SHA256 = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
ARCHIVE = "val2017.zip"
SPLIT = "val2017"
LICENSE = "images: Flickr Terms of Use; annotations: CC-BY-4.0"
# The terms-of-use page of cocodataset.org, from the website's repository at the last commit that changed it.
LICENSE_OWNER, LICENSE_REPO = "cocodataset", "cocodataset.github.io"
LICENSE_COMMIT = "aaa6a5a0cc24bf1350247169cc512edd7ddf28b9"
LICENSE_PATH = "dataset/termsofuse.htm"
LICENSE_SHA256 = "bd019f88ee44c29b2f19c5b99888cf5bc2e7c16f57b6e52af8ed3a60462e8bdd"
LICENSE_COPY = "licenses/coco-LICENSE"


def check_license(text: bytes) -> None:
    """Refuse a terms-of-use copy that does not state the two licences the import records.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. The two statements are looked for as
    well, so that moving the pin to a commit whose page says something else cannot pass on an updated hash alone.
    """
    page = text.decode("utf-8")
    if "Creative Commons Attribution 4.0 License" not in page or "Flickr Terms of Use" not in page:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the terms of use the import was reviewed for")


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
    """Every JPEG member of the archive, in member name order."""
    for member, data in image_sets.zip_images(archive, ".jpg"):
        yield image_sets.Found(data, origin(member), f"COCO {SPLIT} {member}")


def fetch(cache: Path) -> Path:
    """The path of the archive, checked against ``ARCHIVE_SHA256``."""
    return web.fetch(URL, ARCHIVE_SHA256, cache)


def run(args: argparse.Namespace) -> int:
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
            print(f"{args.corpus}: the COCO set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(archive), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
