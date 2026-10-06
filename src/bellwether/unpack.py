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

from bellwether.manifest import KINDS, Manifest, find_manifest, load_manifest, load_manifests
from bellwether.record import sets as set_tables
from bellwether.storage import (
    is_lfs_pointer,
    lfs_include,
    lfs_pull_command,
    plain_text,
    repository_root,
    stem,
)

# Linux caps one command-line argument at 131,072 bytes; a full re-record changes thousands of sets.
INCLUDE_BYTES = 100_000


def selected(fixtures: Path, model: str | None) -> list[Manifest]:
    """The manifest of every model under ``fixtures``, or of ``model`` only, which must have one."""
    if model is not None:
        return [find_manifest(fixtures, model)]
    return load_manifests(fixtures)


def set_files(manifest: Manifest) -> list[tuple[str, str, Path]]:
    """``(kind, set, file)`` for each of the model's fixture sets, in either form."""
    found = []
    for kind in KINDS:
        kind_dir = manifest.path.parent / kind
        if kind_dir.is_dir():
            for path in sorted(kind_dir.iterdir()):
                name = stem(path)
                if name is not None:
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
    fixtures no longer have does not stay behind for a consumer that reads every ``*.jsonl``. Unpacking every model
    also removes the tree of a model the fixtures no longer have. Only what an earlier unpack wrote is removed, a
    directory whose ``manifest.toml`` loads as a bellwether manifest: a model's directory there that holds anything
    else stops the run before anything is deleted, and any other directory is left as it is."""
    manifests = selected(fixtures, model)
    refuse_out_over_fixtures(out, manifests)
    refuse_what_unpack_did_not_write(out, manifests)
    if model is None and out.is_dir():
        slugs = {manifest.slug for manifest in manifests}
        for stale in sorted(p for p in out.iterdir() if p.name not in slugs and loads_as_manifest(p / "manifest.toml")):
            shutil.rmtree(stale)  # a model the fixtures no longer have
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


def unpacked(directory: Path) -> bool:
    """A directory an earlier unpack wrote: it holds a model's manifest, or nothing at all."""
    return loads_as_manifest(directory / "manifest.toml") or not any(directory.iterdir())


def loads_as_manifest(path: Path) -> bool:
    """Whether ``path`` loads as a bellwether manifest: another project's file of that name is not one."""
    try:
        load_manifest(path)
    except Exception:  # whatever stops it loading, unpack did not write it, so nothing removes it
        return False
    return True


def refuse_what_unpack_did_not_write(out: Path, manifests: list[Manifest]) -> None:
    """Unpacking replaces each model's tree under ``out``; a directory there that unpack did not write is someone
    else's, so nothing is deleted and the run stops."""
    for manifest in manifests:
        target = out / manifest.slug
        if target.exists() and not (target.is_dir() and unpacked(target)):
            raise ValueError(f"{target} holds files unpack did not write; move them away or choose another --out")


def run(args: argparse.Namespace) -> int:
    try:
        manifests = selected(args.fixtures, args.model)
        refuse_out_over_fixtures(args.out, manifests)
        refuse_what_unpack_did_not_write(args.out, manifests)
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
