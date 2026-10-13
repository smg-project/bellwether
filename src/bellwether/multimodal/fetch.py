"""``bellwether multimodal fetch``: the Git LFS objects a tier needs, and nothing else.

A clone holds the images, the clips, the raw arrays and the reference sets as pointers (``.lfsconfig``). A consumer
fetches one tier: its images, every family's reference sets, and the raw arrays of the tier's images where a family
has them. The pull runs ``git lfs pull --include <paths> --exclude ''`` from the checkout's root, in batches under the
command-line cap, as ``bellwether unpack`` fetches fixture sets.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bellwether.multimodal import references, suite
from bellwether.storage import is_lfs_pointer, lfs_include, lfs_pull_command, repository_root

# Linux caps one command-line argument at 131,072 bytes; a tier names thousands of paths.
INCLUDE_BYTES = 100_000


def paths_of(root: Path, tier: str, families: list[str] | None = None, raw: bool = True) -> list[Path]:
    """The files a tier needs, present or not."""
    images = suite.tier_images(root, tier)
    wanted = [root / image.path for image in images]
    for slug in families or sorted(suite.read_families(root)):
        for source in suite.SOURCES:
            path = references.set_path(root, slug, source)
            if path.is_file():
                wanted.append(path)
        if raw and references.raw_dir(root, slug).is_dir():
            for image in images:
                for kind in references.RAW_KINDS:
                    path = references.raw_dir(root, slug) / f"{image.name}.{kind}.npy"
                    if path.is_file():
                        wanted.append(path)
    return wanted


def lfs_fetch(paths: list[Path]) -> None:
    """``git lfs pull`` of exactly these files, in batches under the command-line cap, from the root of the checkout
    that holds them (as ``bellwether unpack`` fetches fixture sets; written here so that the command group imports
    nothing beyond the standard library and ``bellwether.storage``). Nothing is raised: without git-lfs, a network or
    a checkout the files stay pointers, and the caller names them."""
    root = repository_root(paths[0])
    if root is None:
        return
    batch: list[Path] = []
    size = 0
    for path in paths:
        item = len(lfs_include([path], root).encode()) + 1
        if batch and size + item > INCLUDE_BYTES:
            subprocess.run(["git", "lfs", "pull", "--include", lfs_include(batch, root), "--exclude", ""], cwd=root)
            batch, size = [], 0
        batch.append(path)
        size += item
    if batch:
        subprocess.run(["git", "lfs", "pull", "--include", lfs_include(batch, root), "--exclude", ""], cwd=root)


def run(args: argparse.Namespace) -> int:
    try:
        wanted = paths_of(args.root, args.tier, args.families, raw=not args.no_raw)
    except (OSError, ValueError) as err:
        print(f"bellwether multimodal fetch: {err}", file=sys.stderr)
        return 2
    pointers = [path for path in wanted if is_lfs_pointer(path)]
    if not pointers:
        print(f"{args.tier}: every one of its {len(wanted)} files is already content")
        return 0
    if args.dry_run:
        print(lfs_pull_command(pointers, repository_root(pointers[0])))
        return 0
    lfs_fetch(pointers)
    missing = [path for path in pointers if is_lfs_pointer(path)]
    if missing:
        listing = "".join(f"\n  {path}" for path in missing[:20])
        print(
            f"bellwether multimodal fetch: Git LFS has not fetched {len(missing)} of the tier's files:{listing}\n"
            f"Fetch them with: {lfs_pull_command(missing, repository_root(missing[0]))}",
            file=sys.stderr,
        )
        return 1
    print(f"{args.tier}: {len(pointers)} files fetched, {len(wanted) - len(pointers)} were content already")
    return 0
