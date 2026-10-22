"""bellwether unpack: one plain tree of every fixture set, in the layout of ``fixtures/``, for consumers.

Benchmark sets are stored compressed in Git LFS, and a clone holds only their pointers until they are fetched
(``.lfsconfig``), so ``unpack`` first fetches the selected models' sets that are still pointers. A consumer
(Symphony's fixture test, smg's consumer test) points at the unpacked root instead, where every set is plain JSON
Lines beside the hand-written ones, and each model's ``manifest.toml`` and ``sets.toml`` come along. So do the shape
patterns' recordings (``bellwether patterns``), from ``patterns/`` beside the fixture root into ``patterns/`` of the
tree, so that a consumer reads one root.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from bellwether.manifest import KINDS, Manifest, find_manifest, load_manifest, load_manifests
from bellwether.patterns import FILES as PATTERN_FILES
from bellwether.record import sets as set_tables
from bellwether.storage import (
    is_compressed,
    is_lfs_pointer,
    lfs_include,
    lfs_pull_command,
    plain_text,
    repository_root,
    stem,
)

# Linux caps one command-line argument at 131,072 bytes; a full re-record changes thousands of sets.
INCLUDE_BYTES = 100_000
# The shape patterns' recordings, beside the fixture root and in the unpacked tree.
PATTERNS = "patterns"


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


def unpack(
    fixtures: Path, out: Path, model: str | None = None, plain: bool = False, sample_sets: int | None = None
) -> list[Path]:
    """Write each selected model's tree under ``out``, replacing what an earlier unpack left there, so a set the
    fixtures no longer have does not stay behind for a consumer that reads every ``*.jsonl``. Unpacking every model
    also removes the tree of a model the fixtures no longer have. Only what an earlier unpack wrote is removed, a
    directory whose ``manifest.toml`` loads as a bellwether manifest: a model's directory there that holds anything
    else stops the run before anything is deleted, and any other directory is left as it is.

    With ``plain``, the sets stored plain alone (the hand-written ones) go into the tree, and with ``sample_sets``
    every Nth compressed set of the selection besides them (``chosen``); each model's ``sets.toml`` then holds the
    tables of the sets written, so a consumer that checks a set against its table reads a tree that agrees with
    itself, and a set the tree does not hold is not listed as if it did."""
    manifests = selected(fixtures, model)
    refuse_out_over_fixtures(out, manifests)
    refuse_what_unpack_did_not_write(out, manifests)
    if model is None and out.is_dir():
        slugs = {manifest.slug for manifest in manifests}
        for stale in sorted(p for p in out.iterdir() if p.name not in slugs and loads_as_manifest(p / "manifest.toml")):
            shutil.rmtree(stale)  # a model the fixtures no longer have
    written: list[Path] = []
    for manifest, sets in chosen(manifests, plain, sample_sets):
        target_dir = out / manifest.slug
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        shutil.copyfile(manifest.path, target_dir / "manifest.toml")
        tables = manifest.path.parent / set_tables.FILE
        if tables.is_file():
            if plain:
                names = {(kind, name) for kind, name, _ in sets}
                kept = {key: table for key, table in set_tables.read(tables).items() if key in names}
                set_tables.write(target_dir / set_tables.FILE, kept)
            else:
                shutil.copyfile(tables, target_dir / set_tables.FILE)
        for kind, name, path in sets:
            target = target_dir / kind / f"{name}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(plain_text(path).encode("utf-8"))
            written.append(target)
    written.extend(copy_patterns(fixtures, out))
    return written


def chosen(
    manifests: list[Manifest], plain: bool, sample_sets: int | None
) -> list[tuple[Manifest, list[tuple[str, str, Path]]]]:
    """Each model with the sets the run writes for it: every set, or with ``plain`` the sets stored plain (the
    hand-written ones, which Git holds as they are) and, with ``sample_sets`` N, every Nth compressed set of the
    whole selection counted in the order of the models and their sets, so the sample is the same on every run of the
    same tree and spreads over the models rather than falling on the first ones. A compressed set is one whole file in
    Git LFS: a sample of its cases would still fetch the file, so the unit of the sample is the set, written whole."""
    if not plain:
        return [(manifest, set_files(manifest)) for manifest in manifests]
    if sample_sets is not None and sample_sets < 1:
        raise ValueError(f"--sample-sets takes a positive count, not {sample_sets}")
    picked: list[tuple[Manifest, list[tuple[str, str, Path]]]] = []
    compressed_seen = 0
    for manifest in manifests:
        sets = []
        for kind, name, path in set_files(manifest):
            if not is_compressed(path):
                sets.append((kind, name, path))
            elif sample_sets is not None:
                if compressed_seen % sample_sets == 0:
                    sets.append((kind, name, path))
                compressed_seen += 1
        picked.append((manifest, sets))
    return picked


def copy_patterns(fixtures: Path, out: Path) -> list[Path]:
    """The shape patterns' recordings, copied by name from ``patterns/`` beside the fixture root into ``patterns/``
    under ``out``, each replacing the one an earlier unpack wrote; nothing when the root has none (a fixture tree of a
    test, or a commit before the recordings)."""
    source = fixtures.resolve().parent / PATTERNS
    if not source.is_dir():
        return []
    target_dir = out / PATTERNS
    target_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in PATTERN_FILES:
        if (source / name).is_file():
            shutil.copyfile(source / name, target_dir / name)
            written.append(target_dir / name)
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
    plain, sample_sets = bool(args.plain), args.sample_sets
    try:
        if sample_sets is not None and not plain:
            raise ValueError("--sample-sets samples the compressed sets beside the plain ones: give --plain too")
        manifests = selected(args.fixtures, args.model)
        refuse_out_over_fixtures(args.out, manifests)
        refuse_what_unpack_did_not_write(args.out, manifests)
        sets = chosen(manifests, plain, sample_sets)
    except (FileNotFoundError, ValueError) as err:
        print(f"bellwether unpack: {err}", file=sys.stderr)
        return 1
    pointers = [path for _, picked in sets for _, _, path in picked if is_lfs_pointer(path)]
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
    written = unpack(args.fixtures, args.out, args.model, plain=plain, sample_sets=sample_sets)
    print(f"{args.out}: {len(written)} plain fixture sets", file=sys.stdout)
    return 0
