"""MMMU's validation images as an image set: the figures of its questions, across its thirty subjects.

The data is the MMMU dataset on the Hugging Face Hub at a pinned commit: one parquet file per subject's validation
split, each row a question with up to seven images in the columns ``image_1`` to ``image_7``, stored as the datasets
library stores images (the bytes and a path). Every image cell of every row gives a line, in the order of the subjects'
files, the rows and the columns; bytes that repeat an earlier image's are left out and named, as the text imports name
their repeats. The files are read as data (``image_sets.parquet_images``); nothing in them runs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from . import github, hf, image_sets

DATASET = "mmmu"
REPO = "MMMU/MMMU"
REVISION = "876ce5cb130f7f7e290ce4d9984357737d4db5cf"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "0c9190b5582c5798f5744d37215858de0df51c74de49fc4e64f925c3efd725a8"
LICENSE = "Apache-2.0"
CARD_LICENSE = "apache-2.0"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# Apache-2.0 4(a) asks that a copy of the License go with the work, and the dataset ships no LICENSE or NOTICE file, so
# the import copies the License as the Apache Software Foundation publishes it, from the foundation's website
# repository at the one commit that file has, as the Hermes and glaive-v2 imports do.
LICENSE_OWNER, LICENSE_REPO = "apache", "www-site"
LICENSE_COMMIT = "01b1be9fbc5cd93b6794f5653a58b9b863807f84"
LICENSE_PATH = "content/licenses/LICENSE-2.0.txt"
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
LICENSE_COPY = "licenses/mmmu-LICENSE"
SPLIT = "validation"
COLUMNS = tuple(f"image_{n}" for n in range(1, 8))
# Each subject's validation file, by the sha256 of its bytes, in the order the lines are built.
FILES = {
    "Accounting/validation-00000-of-00001.parquet": "f1bedfc8ad70d24f8c1daa50a309d9b3b88ea14caea60299b648c69049fa870f",
    "Agriculture/validation-00000-of-00001.parquet": "f95765c549ebaeedd89c14bfe39d1deefcb9ec6b8b082f5945ffc547f2c4e20c",
    "Architecture_and_Engineering/validation-00000-of-00001.parquet": (
        "09792c72c949565ba9378366562818d9687eba5b4f6fc3e84482a8862cf4dd9c"
    ),
    "Art/validation-00000-of-00001.parquet": "d86be1080d7a33a76e9b98f0df0d7ebacbd37b1c29927bbf592489cc4b996c57",
    "Art_Theory/validation-00000-of-00001.parquet": "ba4e709ace78695df3b2ed8859e7cfb36e3cd9c60abd5a0fade9800653b695d6",
    "Basic_Medical_Science/validation-00000-of-00001.parquet": (
        "577102f6f742471a947a82579df0ffa8b7bfd1b10dc2eadbf90f9ea5ff0cddd4"
    ),
    "Biology/validation-00000-of-00001.parquet": "644e5b10eb90689a0e41a7c738526113551f59ab5e66c44a870e6bca52d39e08",
    "Chemistry/validation-00000-of-00001.parquet": "ff62af1b99c6f60d2b2228bf438152d9e44c06e42455d5eb5e229954aa95b00b",
    "Clinical_Medicine/validation-00000-of-00001.parquet": (
        "2c3d29a6c838e279974731ea92fa459e15107f9213970a9a4f9312c1f102bd58"
    ),
    "Computer_Science/validation-00000-of-00001.parquet": (
        "73c8ef8ba23a050b85c524363614801cdc73db36ad7a7348643006766da037f5"
    ),
    "Design/validation-00000-of-00001.parquet": "c69b48ecde9ad80bb8b530dec4da99ae051b17f90abefd2b98230313580c5b7f",
    "Diagnostics_and_Laboratory_Medicine/validation-00000-of-00001.parquet": (
        "cefb7363510a05284dd8a0b12cb7cc7f20843c65d6b7101eefe3929d4061a592"
    ),
    "Economics/validation-00000-of-00001.parquet": "1a6b0ef6edf3eb76f2fae37cbff28d6723eec4bd94d4b3d6e676aa4ea924d95b",
    "Electronics/validation-00000-of-00001.parquet": "558d6970c95c9961c63da8b1ebd7a885524059a83838fa67e6be8c6522a536ba",
    "Energy_and_Power/validation-00000-of-00001.parquet": (
        "46f8d320be746b990145c4e2bf6b88fc493f0e8e322759ee2a9e2fcc8ee404b5"
    ),
    "Finance/validation-00000-of-00001.parquet": "75c773f16c39b60ff7ff2ad8dd9fc82986418cc4bee9864c10ec4098adefc555",
    "Geography/validation-00000-of-00001.parquet": "536505f8dc6199713effcd7cf9b8ae2da2325e51ec17d0fcfabc75956965701c",
    "History/validation-00000-of-00001.parquet": "fe6ba5fa13133d43afbd63abfe75b49ac45c5de566ac896fbc370cb1492c8f82",
    "Literature/validation-00000-of-00001.parquet": "3ad02815b3af320c574b254c7a2134f31288a360c0168035ee051c8946cb8b1b",
    "Manage/validation-00000-of-00001.parquet": "838ecee5c3c9708b3fb70e414b83bfee35725814290908a853b24d06bbd832fe",
    "Marketing/validation-00000-of-00001.parquet": "0b4854d8a509e8e0f401255d5788fd83e489e4d5d5207826856dcffa1157feb2",
    "Materials/validation-00000-of-00001.parquet": "6881364f70c6911f0610005ad5aaf10f4394a7f578ead566e528b574da534e54",
    "Math/validation-00000-of-00001.parquet": "ebe73665a3b1db2bfd33ae67a00efcf86b6d817463a12d995e30716d69aae8e7",
    "Mechanical_Engineering/validation-00000-of-00001.parquet": (
        "8381e04c04dc8e0b3a21404e251a1fc2deb52d3f5b2485e5e8bc99e4907d21d7"
    ),
    "Music/validation-00000-of-00001.parquet": "132f8e3965276788f1b5388c39565d5475b8eb61af3142522be446e9a131af8f",
    "Pharmacy/validation-00000-of-00001.parquet": "93bc3de041d8f69b9e7cd29886ad68f26ac5b11bdf221cabb5f84b8b8d8ca6a9",
    "Physics/validation-00000-of-00001.parquet": "649a7c85ae9f1aa0762059eca0c082e3862f669f07a1283c36ada31def2c391d",
    "Psychology/validation-00000-of-00001.parquet": "16101fe448e07ce654d95b5ef5793988fb6e7f5fe42b94575c33c9ffbd4ad919",
    "Public_Health/validation-00000-of-00001.parquet": (
        "c8525f7af4a47ffd3c2959816147911203782c8e1116267f4e6e54026ef0f923"
    ),
    "Sociology/validation-00000-of-00001.parquet": "a38e31036b6cfb6df82c069f2715b8df0158969405e86a08c1732c334638ac32",
}


def check_license(text: bytes) -> None:
    """Refuse a License copy that is not the Apache License, Version 2.0.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first two lines are checked as
    well, so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:2] != ["Apache License", "Version 2.0, January 2004"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the Apache License, Version 2.0; review it")


def subject_of(file: str) -> str:
    """The subject a file belongs to, as the dataset names its directory, with spaces for the underscores."""
    return file.split("/")[0].replace("_", " ")


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
    """Every image cell of every row of every subject's file, in the order of ``FILES``, the rows and ``COLUMNS``."""
    for file in FILES:
        subject = subject_of(file)
        for row, column, data in image_sets.parquet_images(files[file], COLUMNS):
            yield image_sets.Found(data, origin(file, row, column), f"MMMU {subject} {SPLIT} row {row}, {column}")


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
            print(f"{args.corpus}: the MMMU set equals a fresh import of {SOURCE}")
        return 1 if problems else 0
    _, summary = image_sets.write(DATASET, read(files), args.corpus, copies)
    image_sets.report(DATASET, summary, args.corpus)
    return 0
