"""bellwether unpack: one plain tree of every fixture set, in the layout of ``fixtures/``, for consumers.

Benchmark sets are stored compressed in Git LFS, and a clone holds only their pointers until they are fetched
(``.lfsconfig``), so ``unpack`` first fetches the selected models' sets that are still pointers. A consumer
(Symphony's fixture test, smg's consumer test) points at the unpacked root instead, where every set is plain JSON
Lines beside the hand-written ones, and each model's ``manifest.toml`` and ``sets.toml`` come along.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from bellwether.manifest import KINDS, Manifest, find_manifest, load_manifest
from bellwether.record import sets as set_tables
from bellwether.record.fixtures import (
    COMPRESSED_SUFFIX,
    is_lfs_pointer,
    lfs_include,
    lfs_pull_command,
    plain_text,
    repository_root,
)

# Linux caps one command-line argument at 131,072 bytes; a full re-record changes thousands of sets.
INCLUDE_BYTES = 100_000


def selected(fixtures: Path, model: str | None) -> list[Manifest]:
    """The manifest of every model under ``fixtures``, or of ``model`` only, which must have one."""
    if model is not None:
        return [find_manifest(fixtures, model)]
    return [load_manifest(path) for path in sorted(fixtures.glob("*/manifest.toml"))]


def set_files(manifest: Manifest) -> list[tuple[str, str, Path]]:
    """``(kind, set, file)`` for each of the model's fixture sets, in either form."""
    found = []
    for kind in KINDS:
        kind_dir = manifest.path.parent / kind
        if kind_dir.is_dir():
            for path in sorted(kind_dir.iterdir()):
                name = path.name.removesuffix(COMPRESSED_SUFFIX).removesuffix(".jsonl")
                if name != path.name:
                    found.append((kind, name, path))
    return found


def fetch(paths: list[Path]) -> None:
    """``git lfs pull`` of exactly these sets, run from the root of the checkout that holds them. Nothing is raised:
    without git-lfs, a network or a checkout, the sets stay pointers, and the caller names them."""
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


def unpack(fixtures: Path, out: Path, model: str | None = None) -> list[Path]:
    """Write each selected model's tree under ``out``, replacing what an earlier unpack left there, so a set the
    fixtures no longer have does not stay behind for a consumer that reads every ``*.jsonl``."""
    manifests = selected(fixtures, model)
    refuse_out_over_fixtures(out, manifests)
    written: list[Path] = []
    for manifest in manifests:
        target_dir = out / manifest.slug
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        for name in ("manifest.toml", set_tables.FILE):
            if (manifest.path.parent / name).is_file():
                shutil.copyfile(manifest.path.parent / name, target_dir / name)
        for kind, name, path in set_files(manifest):
            target = target_dir / kind / f"{name}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(plain_text(path).encode("utf-8"))
            written.append(target)
    return written


def refuse_out_over_fixtures(out: Path, manifests: list[Manifest]) -> None:
    """Unpacking replaces each model's tree under ``out``; with ``out`` the fixtures directory, that would delete the
    fixtures it reads."""
    for manifest in manifests:
        if (out / manifest.slug).resolve() == manifest.path.parent.resolve():
            raise ValueError(f"--out {out} is the fixtures directory itself; unpack writes a separate tree")


def run(args: argparse.Namespace) -> int:
    try:
        manifests = selected(args.fixtures, args.model)
        refuse_out_over_fixtures(args.out, manifests)
    except (FileNotFoundError, ValueError) as err:
        print(f"bellwether unpack: {err}", file=sys.stderr)
        return 1
    pointers = [path for manifest in manifests for _, _, path in set_files(manifest) if is_lfs_pointer(path)]
    if pointers:
        fetch(pointers)
        missing = [path for path in pointers if is_lfs_pointer(path)]
        if missing:
            listing = "".join(f"\n  {path}" for path in missing)
            print(
                f"bellwether unpack: Git LFS has not fetched these sets, so nothing was written:{listing}\n"
                f"Fetch them with: {lfs_pull_command(missing, repository_root(missing[0]))}",
                file=sys.stderr,
            )
            return 1
    written = unpack(args.fixtures, args.out, args.model)
    print(f"{args.out}: {len(written)} plain fixture sets", file=sys.stdout)
    return 0
