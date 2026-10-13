"""``bellwether multimodal check``: the suite's files agree with its tables, with no Git LFS fetch.

Every image the manifest lists is in the checkout, as content or as a Git LFS pointer, with the manifest's sha256 and
size (a pointer names both, so a clone of pointers is checked whole); every tier names manifest paths and ``all`` names
them all; every source table names manifest ids and paths; every family's inputs are the files with the listed sha256;
every reference set in a ``sets.toml`` exists in the form the table says, with the stored object's sha256 and size, and
every raw array the table counts is there. Problems are printed one a line; the exit status is 1 when there is any.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from bellwether.multimodal import references, suite


def problems(root: Path) -> list[str]:
    found: list[str] = []
    try:
        images = suite.read_manifest(root)
        suite.by_name(images)
    except (OSError, ValueError) as err:
        return [str(err)]
    for image in images:
        path = root / image.path
        if not path.is_file():
            found.append(f"{image.path}: the manifest lists it, the checkout lacks it")
            continue
        try:
            sha256, size = suite.identity(path)
        except (OSError, ValueError) as err:
            found.append(str(err))
            continue
        if sha256 != image.sha256:
            found.append(f"{image.path}: sha256 {sha256}, the manifest says {image.sha256}")
        if size != image.bytes:
            found.append(f"{image.path}: {size} bytes, the manifest says {image.bytes}")
    paths = {image.path for image in images}
    for tier in suite.TIERS:
        try:
            listed = suite.read_tier(root, tier)
        except FileNotFoundError as err:
            found.append(str(err))
            continue
        for line in listed:
            if line not in paths:
                found.append(f"tiers/{tier}.txt: {line} is not in the manifest")
        if len(set(listed)) != len(listed):
            found.append(f"tiers/{tier}.txt: a path is listed twice")
        if tier == "all" and set(listed) != paths:
            found.append(f"tiers/all.txt names {len(set(listed))} paths, the manifest {len(paths)}")
    ids = {image.id for image in images}
    sources = root / "sources"
    if sources.is_dir():
        for table in sorted(sources.glob("*.tsv")):
            with table.open(newline="", encoding="utf-8") as handle:
                for number, row in enumerate(csv.DictReader(handle, delimiter="\t"), 2):
                    if row.get("id") not in ids:
                        found.append(f"sources/{table.name}:{number}: id {row.get('id')} is not in the manifest")
                    if row.get("path") not in paths:
                        found.append(f"sources/{table.name}:{number}: path {row.get('path')} is not in the manifest")
    try:
        families = suite.read_families(root)
    except (OSError, ValueError) as err:
        found.append(str(err))
        families = {}
    for family in families.values():
        for name, sha256 in family.inputs.items():
            path = family.path / name
            if not path.is_file():
                found.append(f"families/{family.slug}/{name}: listed as an input, not in the checkout")
            elif suite.identity(path)[0] != sha256:
                found.append(f"families/{family.slug}/{name}: sha256 differs from family.toml")
        if not family.config.is_file():
            found.append(f"families/{family.slug}: no preprocessor_config.json")
        found.extend(reference_problems(root, family.slug))
    stray = (
        {p.name for p in (root / "references").iterdir() if p.is_dir()} - set(families)
        if (root / "references").is_dir()
        else set()
    )
    for slug in sorted(stray):
        found.append(f"references/{slug}: no family of that name")
    return found


def reference_problems(root: Path, family: str) -> list[str]:
    found: list[str] = []
    tables = suite.read_sets(root / "references" / family / suite.SETS_FILE)
    for (source, slug), table in tables.items():
        if slug != family:
            found.append(f"references/{family}/sets.toml: a table for {slug}")
            continue
        if source not in suite.SOURCES:
            found.append(f"references/{family}/sets.toml: unknown source {source}")
        path = references.set_path(root, family, source)
        if not path.is_file():
            found.append(f"{path.relative_to(root)}: sets.toml lists it, the checkout lacks it")
            continue
        sha256, size = suite.identity(path)
        if sha256 != table.get("stored_sha256") or size != table.get("stored_bytes"):
            found.append(f"{path.relative_to(root)}: the stored object is not the one sets.toml names")
        raw_files = table.get("raw_files")
        if raw_files is not None:
            have = (
                sum(1 for p in references.raw_dir(root, family).glob("*.npy"))
                if references.raw_dir(root, family).is_dir()
                else 0
            )
            if have != raw_files:
                found.append(f"references/{family}/raw: {have} files, sets.toml counts {raw_files}")
    for source in suite.SOURCES:
        path = references.set_path(root, family, source)
        if path.is_file() and (source, family) not in tables:
            found.append(f"{path.relative_to(root)}: not in sets.toml")
    return found


def run(args: argparse.Namespace) -> int:
    found = problems(args.root)
    for problem in found:
        print(problem, file=sys.stderr)
    if found:
        print(f"bellwether multimodal check: {len(found)} problem(s)", file=sys.stderr)
        return 1
    images = suite.read_manifest(args.root)
    print(f"{args.root}: {len(images)} images, {len(suite.read_families(args.root))} families, tables agree")
    return 0
