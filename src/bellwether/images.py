"""bellwether images check: the image sets and the files in the checkout agree, with no Git LFS fetch.

Every line of every set under ``corpus/image/`` names a file under ``corpus/images/<dataset>/`` whose content has the
line's sha256 and size; a file Git LFS has not fetched is a pointer, which names both, so a clone of pointers is
checked whole without a download, and a fetched file is hashed. No file under an images directory is unnamed, no
images directory is without its set, names and paths are unique across the sets, and each set's licence copy is
there. Problems are printed one a line; the exit status is 1 when there is any. It is the check every pull request
runs; an importer's ``--check``, which builds the import again from its pinned downloads, runs on its own schedule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bellwether.importers import image_sets

LICENSES = "licenses"


def problems(corpus: Path) -> list[str]:
    found: list[str] = []
    sets_dir = corpus / image_sets.KIND
    sets = sorted(sets_dir.glob("*.jsonl")) if sets_dir.is_dir() else []
    names: dict[str, Path] = {}
    paths: dict[str, Path] = {}
    datasets: set[str] = set()
    for set_file in sets:
        dataset = set_file.name.removesuffix(".jsonl")
        datasets.add(dataset)
        try:
            lines = image_sets.read_set(set_file)
        except ValueError as err:
            found.append(str(err))
            continue
        expected: set[Path] = set()
        for number, line in enumerate(lines, 1):
            where = f"{set_file}:{number}"
            if line["name"] in names:
                found.append(f"{where}: name {line['name']} is already used in {names[line['name']]}")
            names.setdefault(line["name"], set_file)
            if line["path"] in paths:
                found.append(f"{where}: path {line['path']} is already used in {paths[line['path']]}")
                continue  # the file was checked against its first line
            paths[line["path"]] = set_file
            if not line["path"].startswith(f"{image_sets.IMAGES}/{dataset}/"):
                found.append(f"{where}: path {line['path']} is not under {image_sets.IMAGES}/{dataset}/")
                continue
            path = corpus / line["path"]
            expected.add(path)
            if not path.is_file():
                found.append(f"{line['path']}: the set names it, the checkout lacks it")
                continue
            try:
                digest, size = image_sets.identity(path)
            except (OSError, ValueError) as err:
                found.append(str(err))
                continue
            if digest != line["sha256"]:
                found.append(f"{line['path']}: sha256 {digest}, the set says {line['sha256']}")
            if size != line.get("bytes"):
                found.append(f"{line['path']}: {size} bytes, the set says {line.get('bytes')}")
        directory = image_sets.images_dir(corpus, dataset)
        if directory.is_dir():
            for path in sorted(p for p in directory.iterdir() if p.is_file() and p not in expected):
                found.append(f"{path.relative_to(corpus)}: no line of {set_file.name} names it")
        if not (corpus / LICENSES / f"{dataset}-LICENSE").is_file():
            found.append(f"{LICENSES}/{dataset}-LICENSE: the licence copy of {set_file.name} is missing")
    images_root = corpus / image_sets.IMAGES
    if images_root.is_dir():
        for directory in sorted(p for p in images_root.iterdir() if p.is_dir() and p.name not in datasets):
            found.append(f"{directory.relative_to(corpus)}: no set {image_sets.KIND}/{directory.name}.jsonl names it")
    return found


def run(args: argparse.Namespace) -> int:
    found = problems(args.corpus)
    for problem in found:
        print(problem, file=sys.stderr)
    if found:
        print(f"bellwether images check: {len(found)} problem(s)", file=sys.stderr)
        return 1
    sets_dir = args.corpus / image_sets.KIND
    sets = sorted(sets_dir.glob("*.jsonl")) if sets_dir.is_dir() else []
    lines = [line for set_file in sets for line in image_sets.read_set(set_file)]
    print(f"{args.corpus}: {len(lines)} images in {len(sets)} sets, {sum(int(line['bytes']) for line in lines)} bytes")
    return 0
